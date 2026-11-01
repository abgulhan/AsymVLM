"""Versioned Phi EOS repair; frozen v3 weights, scorer and runtime are reused.

The pinned tokenizer ends answers with 32000, whereas the observed native
generation config selected 2 (32007 is the separate chat-end token). Select the
tokenizer EOS explicitly, retain the actual prior value in provenance, and
reject generation that continues after the selected EOS.
No snapshot, frozen implementation, or historical result is modified.
"""

import argparse
import copy
from contextlib import contextmanager
from pathlib import Path
import sys

import torch

from learned_score.mlsys_scorer.protocol import digest_file
from . import phi_adaptive_v3 as original
from . import phi_adaptive_v3_compat as compat


EXPECTED_TOKENIZER_EOS = 32000
STOPPING_POLICY = "phi_tokenizer_eos_unpenalized_greedy_v4"


def repair_stopping(adapter):
    """Correct in-memory generation metadata only; preserve the prior values."""
    tokenizer_eos = adapter.processor.tokenizer.eos_token_id
    if type(tokenizer_eos) is not int or tokenizer_eos != EXPECTED_TOKENIZER_EOS:
        raise ValueError("pinned Phi tokenizer must declare EOS 32000")
    native = adapter.model.native
    previous = getattr(native, "generation_config", None)
    if previous is None:
        raise ValueError("Phi native generation configuration is missing")
    stopping = {
        "policy": STOPPING_POLICY,
        "tokenizer_eos_token_id": tokenizer_eos,
        "snapshot_generation_eos_token_id": copy.deepcopy(previous.eos_token_id),
        "snapshot_model_eos_token_id": copy.deepcopy(getattr(native.config, "eos_token_id", None)),
        "effective_eos_token_id": tokenizer_eos,
        "max_new_tokens_is_only_a_safety_cap": True,
    }
    corrected = copy.deepcopy(previous)
    corrected.eos_token_id = tokenizer_eos
    native.generation_config = corrected
    adapter.model.generation_config = corrected
    adapter.stopping_identity = stopping
    validate_stopping(adapter)
    return stopping


def validate_stopping(adapter):
    if (adapter.processor.tokenizer.eos_token_id != EXPECTED_TOKENIZER_EOS
            or adapter.model.generation_config.eos_token_id != EXPECTED_TOKENIZER_EOS
            or adapter.model.native.generation_config.eos_token_id != EXPECTED_TOKENIZER_EOS
            or getattr(adapter, "stopping_identity", {}).get("policy") != STOPPING_POLICY):
        raise ValueError("Phi v4 stopping contract was not installed or was mutated")


def generation_trace(ids, max_new_tokens, eos_token_id=EXPECTED_TOKENIZER_EOS):
    """Keep emitted IDs (including EOS), distinguishing EOS from length limits."""
    if torch.is_tensor(ids):
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("generation tracing requires one unpadded sequence")
        values = ids.detach().cpu()[0].tolist()
    else:
        values = list(ids)
    if (not values or len(values) > max_new_tokens
            or any(type(value) is not int or value < 0 for value in values)):
        raise ValueError("invalid generated token sequence")
    if eos_token_id in values[:-1]:
        raise RuntimeError("Phi generation continued after its tokenizer EOS")
    ended = values[-1] == eos_token_id
    if not ended and len(values) != max_new_tokens:
        raise RuntimeError("generation stopped before EOS or its declared safety cap")
    return {"generated_token_ids": values,
            "stop_reason": "eos_token" if ended else "max_new_tokens",
            "ended_with_eos": ended, "effective_eos_token_id": eos_token_id,
            "first_eos_position": len(values) - 1 if ended else None}


def annotate_generation(adapter, ids, runtime, max_new_tokens):
    """Public API for v4 runners; metrics remain plain JSON-serializable data."""
    validate_stopping(adapter)
    if runtime.get("stop_on_eos") is not True:
        raise ValueError("Phi v4 measurements require natural EOS stopping")
    return {**runtime, **generation_trace(ids, max_new_tokens),
            "stopping": copy.deepcopy(adapter.stopping_identity)}


class PhiStoppedAdapterV4(compat.PhiEagerAdapterV3):
    version = "mlsys_adaptive_v4_explicit_eager_tokenizer_eos"

    def load_model_and_processor(self, config):
        super().load_model_and_processor(config)
        repair_stopping(self)
        return self


@contextmanager
def installed():
    """Process-local factory substitution, also usable by the generic v4 runner."""
    with compat.installed():
        previous_adapter, previous_factory = original.PhiAdaptiveAdapterV3, original.load_phi_backbone
        original.PhiAdaptiveAdapterV3 = PhiStoppedAdapterV4

        def factory(args, manifest):
            adapter, dataset, identity = previous_factory(args, manifest)
            validate_stopping(adapter)
            return adapter, dataset, {**identity, "stopping": copy.deepcopy(adapter.stopping_identity),
                "source_sha256": {**identity["source_sha256"], Path(__file__).name: digest_file(__file__)}}

        original.load_phi_backbone = factory
        try:
            yield
        finally:
            original.load_phi_backbone, original.PhiAdaptiveAdapterV3 = previous_factory, previous_adapter


def load_phi_backbone(args, manifest):
    with installed():
        return original.load_phi_backbone(args, manifest)


def source_hashes():
    return {"mlsys_sprint/phi_adaptive_v3_compat.py": digest_file(compat.__file__),
            "mlsys_sprint/phi_adaptive_v4.py": digest_file(__file__)}


@contextmanager
def instrument_experiment(module):
    previous_hashes, previous_measured = module.source_hashes, module.measured

    def hashes(adapter):
        result = previous_hashes(adapter)
        return {**result, **source_hashes()} if adapter == "phi3_vision" else result

    def measured(adapter, row, selector, cell, max_new_tokens):
        if adapter.name == "phi3_vision":
            validate_stopping(adapter)
        prediction, ids, timing, runtime = previous_measured(adapter, row, selector, cell, max_new_tokens)
        if adapter.name == "phi3_vision":
            runtime = annotate_generation(adapter, ids, runtime, max_new_tokens)
        return prediction, ids, timing, runtime

    module.source_hashes, module.measured = hashes, measured
    try:
        yield
    finally:
        module.source_hashes, module.measured = previous_hashes, previous_measured


def native_cache():
    """Supply the pinned remote model's old cache API on current Transformers."""
    from transformers.cache_utils import DynamicCache

    class PhiNativeCache(DynamicCache):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)

        def get_max_length(self):
            return None

        @property
        def seen_tokens(self):
            return self.get_seq_length()

    return PhiNativeCache()


@torch.no_grad()
def native_pruned_ids(adapter, pruned, max_new_tokens):
    """Unbounded native-model greedy reference, preserving original positions."""
    validate_stopping(adapter)
    cache = native_cache()
    output = adapter.model.native(inputs_embeds=pruned.inputs_embeds,
        attention_mask=pruned.attention_mask, position_ids=pruned.position_ids,
        past_key_values=cache, use_cache=True, return_dict=True)
    token = output.logits[:, -1].argmax(-1, keepdim=True)
    generated = [token]
    position = pruned.position_ids[:, -1:] + 1
    while len(generated) < max_new_tokens and int(token.item()) != EXPECTED_TOKENIZER_EOS:
        output = adapter.model.native(input_ids=token,
            attention_mask=torch.ones((1, cache.get_seq_length() + 1), device=adapter.device, dtype=torch.long),
            position_ids=position, past_key_values=cache, use_cache=True, return_dict=True)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated.append(token)
        position = position + 1
    return torch.cat(generated, 1).detach().cpu()


@torch.no_grad()
def native_full_generate_ids(adapter, prepared, max_new_tokens):
    """Actual native multimodal generate, not just the manual reference loop."""
    validate_stopping(adapter)
    config = copy.deepcopy(adapter.model.native.generation_config)
    config.do_sample = False
    config.num_beams = config.num_beam_groups = config.num_return_sequences = 1
    config.repetition_penalty = config.encoder_repetition_penalty = 1.0
    config.no_repeat_ngram_size = config.encoder_no_repeat_ngram_size = 0
    config.min_length, config.min_new_tokens = 0, None
    config.forced_bos_token_id = config.forced_eos_token_id = None
    config.bad_words_ids = config.force_words_ids = None
    config.suppress_tokens = config.begin_suppress_tokens = config.sequence_bias = None
    config.cache_implementation = None
    config.return_dict_in_generate = config.output_scores = config.output_logits = False
    ids = adapter.model.native.generate(**prepared.native_inputs, generation_config=config,
        max_new_tokens=max_new_tokens, use_cache=True, output_attentions=False,
        past_key_values=native_cache())
    return ids[:, prepared.native_inputs["input_ids"].shape[-1]:].detach().cpu()


def parity(args):
    """Stopping-aware full/native-generate and learned-pruning parity on fit IDs."""
    from learned_score.mlsys_runtime.generation import generate_bounded
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from .adaptive_benchmark_v1 import select_exact_count
    from .state import atomic_json, digest
    manifest = original._manifest(args.manifest)
    _, entries = original.selected_entries(manifest, args)
    adapter, dataset, identity = load_phi_backbone(args, manifest)
    scorer, _ = load_scorer(args.checkpoint, args.device)
    scorer.eval().requires_grad_(False)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    provenance = {"format": "mlsys_phi_stopping_parity_v4", "backbone": identity,
        "manifest_sha256": digest_file(args.manifest), "checkpoint_sha256": digest_file(args.checkpoint),
        "max_new_tokens": args.max_new_tokens, "retentions_percent": args.retentions,
        "split": "fit", "world_size": args.world_size, "source_sha256": source_hashes()}
    signature = digest(provenance)
    args.output.mkdir(parents=True, exist_ok=True)
    passed = 0
    for entry in entries[args.rank::args.world_size]:
        path = args.output / (entry["row_hash"] + ".json")
        if path.exists():
            import json
            previous = json.loads(path.read_text())
            if not args.resume or previous.get("identity_sha256") != signature or previous.get("status") != "passed":
                raise ValueError("existing stopping parity artifact differs")
            passed += 1
            continue
        with torch.no_grad():
            prepared = adapter.prepare_sample(original.read_entry(dataset, entry), include_answer=False)
            vision = prepared.inputs_embeds[0, prepared.visual_positions]
            scores = scorer(vision, prepared.inputs_embeds[0, prepared.semantic_text_positions]).float()
            outcomes = []
            for percent in args.retentions:
                count = max(1, len(vision) * percent // 100)
                selected = select_exact_count(scores, vision, count)
                pruned = adapter.physically_prune(prepared, selected)
                expected = native_pruned_ids(adapter, pruned, args.max_new_tokens)
                result = generate_bounded(adapter.model, pruned, args.max_new_tokens, 512,
                    slack=1, stop_on_eos=True, require_fused_attention=False)
                actual = result.ids.detach().cpu()
                outcome = {"retention_percent": percent, "selected_visual_tokens": count,
                    "native_unbounded": generation_trace(expected, args.max_new_tokens),
                    "bounded": generation_trace(actual, args.max_new_tokens),
                    "native_ids_equal": torch.equal(expected, actual),
                    "cache": result.cache.memory_stats()}
                if percent == 100:
                    official = native_full_generate_ids(adapter, prepared, args.max_new_tokens)
                    outcome["native_generate"] = generation_trace(official, args.max_new_tokens)
                    outcome["native_generate_ids_equal"] = torch.equal(official, actual)
                outcomes.append(outcome)
                if (not outcome["native_ids_equal"] or outcome.get("native_generate_ids_equal") is False
                        or outcome["cache"]["compaction_events"] or outcome["cache"]["evicted_tokens"]):
                    atomic_json(path, {**provenance, "identity_sha256": signature, "status": "failed",
                        "request_id": entry["row_hash"], "outcomes": outcomes})
                    raise RuntimeError("Phi stopping-aware native/bounded parity failed")
                del result, pruned
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        atomic_json(path, {**provenance, "identity_sha256": signature, "status": "passed",
            "request_id": entry["row_hash"], "input_visual_tokens": len(vision), "outcomes": outcomes})
        passed += 1
        print(f"Phi stopping parity passed: {entry['row_hash']}", flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json", {
        "identity_sha256": signature, "status": "passed", "requests": passed, "stopping": identity["stopping"]})


def parse_parity_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("parity",), required=True)
    for name in ("manifest", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path)
    parser.add_argument("--split", choices=("fit",), default="fit")
    parser.add_argument("--model", default=original.DEFAULT_MODEL)
    parser.add_argument("--model-revision", default="12b77fb40b63a2c73c68243d3f767aab688a1b2a")
    parser.add_argument("--backbone-dtype", choices=("bfloat16",), default="bfloat16")
    parser.add_argument("--num-crops", type=int, choices=(16,), default=16)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--retentions", type=int, nargs="+", default=[10, 40, 70, 100])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 0 <= args.rank < args.world_size or args.max_new_tokens < 1 or args.limit < 1
            or 100 not in args.retentions or len(set(args.retentions)) != len(args.retentions)
            or any(not 1 <= value <= 100 for value in args.retentions)):
        parser.error("valid shard, positive limits, and unique retention percentages including 100 required")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--experiment" in argv:
        if argv.count("--experiment") != 1:
            raise ValueError("--experiment must occur exactly once")
        argv.remove("--experiment")
        from . import adaptive_experiment_v3 as experiment
        with installed(), instrument_experiment(experiment):
            return experiment.main(argv)
    return parity(parse_parity_args(argv))


if __name__ == "__main__":
    main()
