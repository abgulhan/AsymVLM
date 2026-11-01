"""Isolated sampled-attention launcher for the frozen batch-one benchmark.

Normal use accepts all ``mlsys_sprint.benchmark`` arguments, plus:
  --text-policy sampled_attention --recent-window 128 --refresh-interval 32

Label-free GPU validation (not a speed or quality measurement):
  python -m learned_score.mlsys_sprint.importance_benchmark --smoke \
    --adapter qwen2_5_vl --model /local/snapshot --image /local/image.jpg \
    --output smoke.json

The policy is an approximation: one last-layer decode attention row is sampled
every 32 generated inputs, accumulated, and combined with the latest 128 tokens.
It does not request model attention matrices or claim to reproduce the original
paper's eviction policy. Original benchmark/runtime source files are not edited.
"""

import argparse
import copy
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

from learned_score.mlsys_scorer.checkpoints import atomic_json, canonical_digest
from learned_score.mlsys_scorer.protocol import digest_file


_WRAPPER_LOCK = threading.Lock()
_POLICY_METADATA = "text_policy.metadata.json"


def policy_configuration(text_policy="sampled_attention", recent_window=128, refresh_interval=32):
    if text_policy not in {"sampled_attention", "recency"}:
        raise ValueError("text policy must be sampled_attention or recency")
    if (type(recent_window) is not int or recent_window < 0 or
            type(refresh_interval) is not int or refresh_interval < 1):
        raise ValueError("recent window must be nonnegative and refresh interval positive")
    policy_file = "importance.py" if text_policy == "sampled_attention" else "policies.py"
    return {
        "format": "mlsys_text_policy_launcher_v1",
        "text_policy": text_policy,
        "recent_window": recent_window if text_policy == "sampled_attention" else None,
        "refresh_interval": refresh_interval if text_policy == "sampled_attention" else None,
        "description": (
            "sum sampled final-layer decode attention; preserve latest generated window"
            if text_policy == "sampled_attention" else "keep latest generated tokens only"
        ),
        "approximation_not_original_paper_policy": text_policy == "sampled_attention",
        "batch_size": 1,
        "serving_measurement": False,
        "source_sha256": {
            "launcher": digest_file(Path(__file__)),
            "policy": digest_file(Path(__file__).parents[1] / "mlsys_runtime" / policy_file),
        },
    }


def guard_output_directory(output, configuration):
    """Refuse mixing policies, source versions, or historical recency results.

    The short exclusive lock only serializes metadata initialization across
    different ranks. Request/task locking remains the autonomous worker's job.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".text_policy.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = output / _POLICY_METADATA
        expected = {"configuration": configuration,
                    "configuration_sha256": canonical_digest(configuration)}
        if path.exists():
            if json.loads(path.read_text()) != expected:
                raise ValueError("output text policy/configuration/source differs; use a new directory")
        else:
            if any(item.name != ".text_policy.lock" for item in output.iterdir()):
                raise ValueError("refusing an existing unmarked/recency directory; use a new output directory")
            atomic_json(path, expected)


@dataclass(frozen=True)
class PolicyCell:
    """Duck-typed benchmark cell; original all-token control keeps its name."""

    cell: object
    text_policy: str
    configuration_sha256: str

    @property
    def selector(self):
        return self.cell.selector

    @property
    def visual_retention(self):
        return self.cell.visual_retention

    @property
    def generated_budget(self):
        return self.cell.generated_budget

    @property
    def key(self):
        if self.cell.method == "all_tokens":
            return self.cell.key
        return f"{self.cell.key}_{self.text_policy}_{self.configuration_sha256[:12]}"

    @property
    def method(self):
        return "all_tokens" if self.cell.method == "all_tokens" else self.key


def include_policy_memory(metrics, policy_metrics):
    """Add persistent scores once to owned cache bytes, not to allocator peaks.

    Temporary attention rows/selection tensors are covered by the existing CUDA
    allocator peak, not misrepresented as persistent cache capacity.
    """
    score_bytes = policy_metrics["persistent_score_bytes"]
    if type(score_bytes) is not int or score_bytes < 0:
        raise ValueError("invalid persistent score byte count")
    cache = metrics["cache"]
    if "storage_owned_bytes" not in cache:
        cache["storage_owned_bytes"] = cache["total_owned_bytes"]
        cache["policy_score_bytes"] = score_bytes
        cache["total_owned_bytes"] += score_bytes
    elif (cache.get("policy_score_bytes") != score_bytes or
          cache["total_owned_bytes"] != cache["storage_owned_bytes"] + score_bytes):
        raise ValueError("inconsistent or already modified policy memory accounting")
    metrics["importance_policy"] = dict(policy_metrics)
    metrics["policy_memory_accounting"] = "persistent_scores_in_owned_cache_bytes; temporary_tensors_in_CUDA_peak_only"
    return metrics


def policy_run_request(adapter, row, select_visual, *, configuration, policy_enabled,
                       run=None, policy_factory=None, **kwargs):
    """Install/remove hooks *inside* the benchmark's external request timer."""
    if run is None:
        from learned_score.mlsys_runtime import run_request
        run = run_request
    if "policy" in kwargs:
        raise ValueError("the launcher owns the text policy argument")
    sampled = policy_enabled and configuration["text_policy"] == "sampled_attention"
    if not sampled:
        result = run(adapter, row, select_visual, **kwargs)
    else:
        if policy_factory is None:
            from learned_score.mlsys_runtime.importance import SampledAttentionPolicy
            policy_factory = SampledAttentionPolicy
        with policy_factory(adapter.model, recent_window=configuration["recent_window"],
                            refresh_interval=configuration["refresh_interval"]) as policy:
            result = run(adapter, row, select_visual, policy=policy, **kwargs)
            policy_metrics = policy.metrics()
        include_policy_memory(result.metrics, policy_metrics)
    result.metrics["text_policy_requested"] = configuration["text_policy"]
    result.metrics["text_policy_effective"] = configuration["text_policy"] if policy_enabled else "disabled"
    result.metrics["policy_hooks_in_end_to_end_timing"] = True
    return result


@contextmanager
def scoped_benchmark(configuration, module=None, run=None, policy_factory=None):
    """Scoped module wrappers; unsuitable for concurrent threads in one process."""
    if module is None:
        from . import benchmark as module
    if not _WRAPPER_LOCK.acquire(blocking=False):
        raise RuntimeError("benchmark wrappers may not be nested/concurrent in one process")
    original_grid, original_measured, original_atomic = module.make_grid, module.measured_request, module.atomic_json
    configuration = copy.deepcopy(configuration)
    configuration_hash = canonical_digest(configuration)

    def grid(*args, **kwargs):
        return [PolicyCell(cell, configuration["text_policy"], configuration_hash)
                for cell in original_grid(*args, **kwargs)]

    def measured(adapter, row, select_visual, cell, **kwargs):
        if "run" in kwargs:
            raise ValueError("the launcher owns the measured request runner")

        def wrapped_run(adapter, row, selector, **generation_kwargs):
            return policy_run_request(
                adapter, row, selector, configuration=configuration,
                policy_enabled=cell.generated_budget is not None, run=run,
                policy_factory=policy_factory, **generation_kwargs)

        prediction, ids, timing, metrics = original_measured(
            adapter, row, select_visual, cell, run=wrapped_run, **kwargs)
        if "importance_policy" in metrics:
            metrics["physical_cache_measurement"] = "allocated_KV_capacity_plus_scratch_source_metadata_and_policy_scores"
        return prediction, ids, timing, metrics

    def write(path, payload):
        if isinstance(payload, dict) and payload.get("format") == "mlsys_request_result_v1":
            payload = dict(payload)
            payload["text_policy_configuration"] = configuration
            payload["text_policy_configuration_sha256"] = configuration_hash
            payload["text_policy_effective"] = configuration["text_policy"] if payload.get("generated_budget") is not None else "disabled"
        return original_atomic(path, payload)

    try:
        module.make_grid, module.measured_request, module.atomic_json = grid, measured, write
        yield module
    finally:
        module.make_grid, module.measured_request, module.atomic_json = original_grid, original_measured, original_atomic
        _WRAPPER_LOCK.release()


def native_off_parity(model, pruned, native_inputs, steps=32):
    """Compare NEW bounded generation directly with native multimodal generate."""
    import torch
    from learned_score.mlsys_runtime import generate_bounded

    if steps < 1:
        raise ValueError("native parity steps must be positive")
    # The benchmark implements unpenalized greedy argmax, not a sampling or
    # repetition-penalty configuration shipped alongside some checkpoints.
    generation = copy.deepcopy(model.generation_config)
    generation.do_sample = False
    generation.num_beams = generation.num_beam_groups = generation.num_return_sequences = 1
    generation.repetition_penalty = generation.encoder_repetition_penalty = 1.0
    generation.no_repeat_ngram_size = generation.encoder_no_repeat_ngram_size = 0
    generation.min_length = 0
    generation.min_new_tokens = None
    generation.forced_bos_token_id = generation.forced_eos_token_id = None
    generation.bad_words_ids = generation.force_words_ids = None
    generation.suppress_tokens = generation.begin_suppress_tokens = None
    generation.sequence_bias = None
    generation.cache_implementation = None
    generation.return_dict_in_generate = False
    generation.output_scores = generation.output_logits = False
    with torch.no_grad():
        native = model.generate(**native_inputs, generation_config=generation,
                                max_new_tokens=steps, use_cache=True, output_attentions=False)
        expected = native[:, native_inputs["input_ids"].shape[-1]:].detach().cpu()
        result = generate_bounded(model, pruned, max_new_tokens=steps,
                                  generated_budget=steps, stop_on_eos=True)
        actual = result.ids.detach().cpu()
    report = {
        "native_ids_equal": torch.equal(expected, actual),
        "native_generated_tokens": expected.shape[-1],
        "bounded_generated_tokens": actual.shape[-1],
        "native_ids": expected[0].tolist(), "bounded_ids": actual[0].tolist(),
        "newcacheoff_compaction_events": result.metrics["cache"]["compaction_events"],
        "newcacheoff": result.metrics["cache"]["compaction_events"] == 0,
        "natural_eos": True, "max_new_tokens": steps,
        "decoding": "unpenalized_greedy; no forced/suppressed tokens; native EOS",
        "comparison": "native_multimodal_model_generate_vs_new_bounded_runtime",
        "dtype": str(pruned.inputs_embeds.dtype),
    }
    del native, result
    return report


def _tiny_smoke_inputs(family, device, dtype):
    """Random real HF models; no checkpoint download or quality claim."""
    import torch
    if family == "qwen2_5_vl":
        from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
        config = Qwen2_5_VLConfig(
            vocab_size=64, hidden_size=32, intermediate_size=48,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=2048, sliding_window=None,
            rope_scaling={"type": "mrope", "mrope_section": [1, 1, 2]},
            eos_token_id=2, pad_token_id=0, image_token_id=63,
            video_token_id=62, vision_start_token_id=61, vision_end_token_id=60,
            vision_config={"depth": 1, "hidden_size": 16, "intermediate_size": 32,
                           "num_heads": 2, "out_hidden_size": 32, "fullatt_block_indexes": [0]},
        )
        config._attn_implementation = "sdpa"
        model = Qwen2_5_VLForConditionalGeneration(config)
    else:
        from transformers import LlavaNextConfig, LlavaNextForConditionalGeneration
        config = LlavaNextConfig(
            text_config={"model_type": "mistral", "vocab_size": 64, "hidden_size": 32,
                         "intermediate_size": 48, "num_hidden_layers": 2,
                         "num_attention_heads": 4, "num_key_value_heads": 2,
                         "max_position_embeddings": 2048, "sliding_window": 128,
                         "eos_token_id": 2, "pad_token_id": 0},
            vision_config={"model_type": "clip_vision_model", "hidden_size": 16,
                           "intermediate_size": 32, "num_hidden_layers": 1,
                           "num_attention_heads": 2, "image_size": 8,
                           "patch_size": 4, "projection_dim": 16}, image_token_index=63,
        )
        config._attn_implementation = "sdpa"
        model = LlavaNextForConditionalGeneration(config)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    ids = torch.arange(4, 52, device=device).reshape(1, -1)
    source = torch.arange(ids.shape[-1], device=device)
    positions = source.reshape(1, -1)
    if family == "qwen2_5_vl":
        positions = positions.unsqueeze(0).expand(3, -1, -1)
    native = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    pruned = SimpleNamespace(inputs_embeds=model.get_input_embeddings()(ids),
                             attention_mask=native["attention_mask"], position_ids=positions,
                             retained_source_positions=source)
    return model, pruned, native


def _load_smoke_inputs(args):
    import torch
    if args.tiny_model:
        return _tiny_smoke_inputs(args.adapter, args.device, torch.bfloat16)
    from learned_score.multimodel.adapters import LlavaNextAdapter, Qwen2_5_VLAdapter
    classes = {"qwen2_5_vl": Qwen2_5_VLAdapter, "llava_next": LlavaNextAdapter}

    class Dataset:
        @staticmethod
        def build_prompt(row):
            return [{"type": "image", "value": str(args.image)},
                    {"type": "text", "value": args.prompt}]

    adapter = classes[args.adapter](dataset=Dataset(), device=args.device).load_model_and_processor(SimpleNamespace(
        model=str(args.model), model_revision=args.model_revision, processor_revision=args.model_revision,
        max_visual_patches=args.max_visual_patches, attn_implementation="sdpa",
    ))
    adapter.model.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    prepared = adapter.prepare_sample({})
    selected = torch.arange(prepared.visual_positions.numel(), device=adapter.device)
    pruned = adapter.physically_prune(prepared, selected, "original")
    return adapter.model, pruned, prepared.native_inputs


def _hook_counts(attention):
    return (len(attention._forward_pre_hooks), len(attention._forward_hooks),
            len(attention.q_proj._forward_hooks))


def smoke_main(argv, configuration):
    parser = argparse.ArgumentParser(description="Label-free BF16 correctness/dispatch checks; no quality or speed claim")
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next"), required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    parser.add_argument("--tiny-model", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=512)
    parser.add_argument("--generated-budget", type=int, default=256)
    parser.add_argument("--slack", type=int, default=32)
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--native-steps", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if configuration["text_policy"] != "sampled_attention":
        parser.error("--smoke requires --text-policy sampled_attention")
    if args.tiny_model:
        if args.model or args.image:
            parser.error("--tiny-model cannot be combined with --model/--image")
    elif args.model is None or not args.model.is_dir() or args.image is None or not args.image.is_file():
        parser.error("checkpoint smoke requires an existing local --model directory and --image file")
    recent, refresh = configuration["recent_window"], configuration["refresh_interval"]
    if args.generated_budget < max(1, recent) or args.slack < 1 or args.native_steps < 1 or args.max_visual_patches < 1:
        parser.error("invalid budget/window/slack/native-step/visual-token configuration")
    profile_wait = args.generated_budget + args.slack + 1
    profile_active = refresh + 2
    if refresh > args.generated_budget + args.slack or args.steps < profile_wait + 1 + profile_active:
        parser.error("smoke must sample before eviction and run beyond compaction plus one sampled profiling interval")
    import torch
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("--smoke requires a CUDA GPU; CPU unit tests do not establish fused GPU dispatch")
    from learned_score.mlsys_runtime import generate_bounded
    from learned_score.mlsys_runtime.importance import SampledAttentionPolicy
    from learned_score.mlsys_runtime.profiling import classify_attention_events
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions

    model, pruned, native_inputs = _load_smoke_inputs(args)
    versions = frozen_versions(model)
    parity = native_off_parity(model, pruned, native_inputs, args.native_steps)
    del native_inputs
    assert_frozen(model, versions)
    policy = SampledAttentionPolicy(model, recent_window=recent, refresh_interval=refresh)
    hooks_before = _hook_counts(policy.attention)
    # Profile a bounded post-compaction window, not hundreds of full model
    # forwards: retaining a full 512-token trace can itself consume many GB.
    with policy:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=profile_wait, warmup=1, active=profile_active, repeat=1),
        ) as profiler:
            profiler_hook = policy.attention.register_forward_hook(lambda *_: profiler.step())
            try:
                result = generate_bounded(model, pruned, max_new_tokens=args.steps,
                                          generated_budget=args.generated_budget, slack=args.slack,
                                          stop_on_eos=False, policy=policy)
                statistics = policy.metrics()
            finally:
                profiler_hook.remove()
    assert_frozen(model, versions)
    fused = classify_attention_events(
        [event.key for event in profiler.key_averages()],
        sum("CUDA" in str(event.device_type) for event in profiler.events()),
    )
    include_policy_memory(result.metrics, statistics)
    hooks_removed = _hook_counts(policy.attention) == hooks_before
    capacity = result.metrics["cache"]["capacity_tokens_per_layer"]
    heads = policy.attention.q_proj.out_features // policy.attention.head_dim
    checks = {
        "native_ids_equal": parity["native_ids_equal"],
        "newcacheoff": parity["newcacheoff"],
        "compaction_occurred": result.metrics["cache"]["compaction_events"] > 0,
        "selection_matches_compactions": statistics["selection_events"] == result.metrics["cache"]["compaction_events"],
        "sampled_queries_occurred": statistics["sampled_queries"] > 0,
        "query_capture_matches_sampling": statistics["captured_query_projections"] == statistics["sampled_queries"],
        "no_auxiliary_prefill_attention": statistics["prefill_attention_rows_computed"] == 0,
        "single_query_row_bound": 0 < statistics["largest_auxiliary_attention_row_elements"] <= heads * capacity,
        "no_model_attention_weights": not result.metrics["attention_weights_requested"] and not statistics["model_attention_weights_requested"],
        "hooks_removed": hooks_removed,
        "backbone_parameter_versions_unchanged": True,
        "fused_sdpa_verified": fused["fused_sdpa_verified"],
    }
    report = {
        "format": "mlsys_sampled_attention_gpu_smoke_v1", "passed": all(checks.values()),
        "checks": checks, "native_no_eviction_parity": parity,
        "forced_decode": {"generated_tokens": int(result.ids.shape[-1]), "stop_on_eos": False,
                          "cache": result.metrics["cache"], "importance_policy": statistics,
                          "native_sliding_window": result.metrics["native_sliding_window"]},
        "fused_attention": fused, "profile_scope": "post_first_compaction_forward_window",
        "profile_wait_forwards": profile_wait, "profile_active_forwards": profile_active,
        "configuration": configuration, "configuration_sha256": canonical_digest(configuration),
        "adapter": args.adapter, "model": str(args.model) if args.model else "random_tiny_real_HF_model",
        "image": str(args.image) if args.image else None, "prompt": args.prompt if args.image else None,
        "dtype": "bfloat16", "device": torch.cuda.get_device_name(torch.device(args.device)),
        "batch_size": 1, "performance_measurement": False, "quality_measurement": False,
        "test_labels_read": False,
    }
    del result
    atomic_json(args.output, report)
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--text-policy", choices=("sampled_attention", "recency"), default="sampled_attention")
    parser.add_argument("--recent-window", type=int, default=128)
    parser.add_argument("--refresh-interval", type=int, default=32)
    parser.add_argument("--smoke", action="store_true")
    options, remainder = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    try:
        configuration = policy_configuration(options.text_policy, options.recent_window, options.refresh_interval)
    except ValueError as error:
        parser.error(str(error))
    if options.smoke:
        return smoke_main(remainder, configuration)
    from . import benchmark
    args = benchmark.parse_args(remainder)
    budgets = [budget for budget in args.text_budgets if budget is not None]
    if options.text_policy == "sampled_attention" and budgets:
        if min(budgets) < options.recent_window:
            parser.error("every text budget must cover --recent-window")
        if options.refresh_interval > min(budgets) + args.slack:
            parser.error("--refresh-interval must allow a sample before the first eviction")
    guard_output_directory(args.output, configuration)
    with scoped_benchmark(configuration):
        return benchmark.main(remainder)


if __name__ == "__main__":
    main()
