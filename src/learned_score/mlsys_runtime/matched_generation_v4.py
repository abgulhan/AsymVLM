"""Explicit v4 generation with a request-local fixed-priority cache.

Reuse the frozen native-position Qwen/Mistral attention adapter, but construct
MatchedKVCacheV4 explicitly: no global cache-class or factory substitution.
Only the attention forwards are scoped to the request and restored on errors.
"""

import threading
import time

import torch

from .generation import GenerationResult, _synchronize, attention_backend, decoder_forward_function
from .masking import native_sliding_window
from .matched_generation_v3 import matched_attention_v3
from .matched_cache_v4 import MatchedKVCacheV4, MAX_GENERATED_POSITIONS, RUNTIME_REVISION


_LOCK = threading.Lock()


@torch.no_grad()
def generate_matched_v4(model, pruned, max_new_tokens, *, full_prefix_tokens, budget_profile,
                        policy, staging=None, seed=42, optimized=True, record_events=False,
                        trace=False, teacher_ids=None, on_first_token=None, stop_on_eos=True,
                        visual_source_positions=None, recent_window=None,
                        recent_mode="eight", phase=0, storage="compact"):
    """Return GenerationResult(ids, metrics, cache) without changing model state.

    teacher_ids[1,T] are the reference emitted sequence. At output index t>0,
    the input is teacher_ids[t-1], while result.ids contains this policy's own
    argmax at every index. Teacher forcing disables EOS stopping. trace=True
    stores scalar diagnostics/snapshots; trace=callback streams the same events
    plus detached vocabulary logits in each 'logits' event (not retained here).
    """
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("v4 generation supports one request per process")
    try:
        return _generate(model, pruned, max_new_tokens, full_prefix_tokens=full_prefix_tokens,
                         budget_profile=budget_profile, policy=policy, staging=staging, seed=seed,
                         optimized=optimized, record_events=record_events, trace=trace,
                         teacher_ids=teacher_ids, on_first_token=on_first_token,
                         stop_on_eos=stop_on_eos, visual_source_positions=visual_source_positions,
                         recent_window=recent_window, recent_mode=recent_mode, phase=phase, storage=storage)
    finally:
        _LOCK.release()


def _generate(model, pruned, max_new_tokens, *, full_prefix_tokens, budget_profile,
              policy, staging, seed, optimized, record_events, trace, teacher_ids,
              on_first_token, stop_on_eos, visual_source_positions, recent_window, recent_mode, phase, storage):
    entry_started = time.perf_counter()
    if not 1 <= max_new_tokens <= MAX_GENERATED_POSITIONS or getattr(model, "training", False):
        raise ValueError("generation length must be 1..4096 and the model must be in eval mode")
    if pruned.inputs_embeds.ndim != 3 or pruned.inputs_embeds.shape[0] != 1:
        raise ValueError("one prepared request is required")
    prefix = pruned.inputs_embeds.shape[1]
    if prefix < 1 or pruned.attention_mask.shape != (1, prefix) or not bool(pruned.attention_mask.bool().all()):
        raise ValueError("prepared prefix must be nonempty and unpadded")
    if pruned.position_ids.shape[-1] != prefix:
        raise ValueError("native positions must cover the prepared prefix")
    device = pruned.inputs_embeds.device
    source = getattr(pruned, "retained_source_positions", None)
    if source is None:
        source = torch.arange(prefix, device=device)
    if int(source[-1]) >= full_prefix_tokens:
        raise ValueError("full_prefix_tokens must cover every original source identity")
    if teacher_ids is not None:
        teacher_ids = torch.as_tensor(teacher_ids, device=device, dtype=torch.long)
        if teacher_ids.ndim == 1:
            teacher_ids = teacher_ids[None]
        if teacher_ids.shape != (1, max_new_tokens):
            raise ValueError("teacher_ids must contain exactly max_new_tokens reference IDs")
        stop_on_eos = False
    cache = MatchedKVCacheV4(prefix, full_prefix_tokens, budget_profile, policy, staging=staging,
                            prefix_source_positions=source, seed=seed, optimized=optimized,
                            trace=bool(trace), visual_source_positions=visual_source_positions,
                            recent_window=recent_window, recent_mode=recent_mode, phase=phase, storage=storage)
    forward = decoder_forward_function(model)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    versions = tuple((name, parameter._version) for name, parameter in model.named_parameters())
    diagnostics, events, device_events = [], [], []
    compaction_cpu = 0.0

    def emit(event):
        if callable(trace):
            trace(event)
        else:
            diagnostics.append(event)

    def log_output(logits, index):
        if not trace:
            return
        scores = logits[0, -1].detach().float()
        top = scores.topk(2).values
        logprobs = scores.log_softmax(-1)
        target = int(teacher_ids[0, index]) if teacher_ids is not None else None
        event = {"event": "logits", "emitted_index": index, "predicted_id": int(scores.argmax()),
                 "target_id": target, "target_log_probability": float(logprobs[target]) if target is not None else None,
                 "top_two_margin": float(top[0] - top[1]),
                 "eos_probability": float(logprobs[list(eos)].exp().sum()) if eos else 0.,
                 "cache_tokens": cache.get_seq_length(), "compaction_count": cache.compaction_count}
        if callable(trace):
            event["logits"] = scores
        emit(event)

    def snapshot(emitted, boundary):
        if trace:
            emit({"event": "cache_snapshot", "emitted_tokens": emitted,
                  "boundary": boundary, **cache.trace_snapshot()})

    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = None
    with matched_attention_v3(model, cache) as family:
        _synchronize(device)
        started = time.perf_counter()
        cache.current_query_sources = source
        cache.current_query_position_ids = pruned.position_ids
        output = forward(inputs_embeds=pruned.inputs_embeds, attention_mask=pruned.attention_mask,
                         position_ids=pruned.position_ids, cache_position=torch.arange(prefix, device=device),
                         past_key_values=cache, use_cache=True, output_attentions=False)
        if output.past_key_values is not cache:
            raise RuntimeError("model replaced the supplied request cache")
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated = [token]
        _synchronize(device)
        ttft = time.perf_counter() - started
        if on_first_token is not None:
            on_first_token()
        log_output(output.logits, 0)
        snapshot(1, "after_prefill")
        next_position = pruned.position_ids[..., -1:].clone() + 1
        for step in range(max_new_tokens - 1):
            if stop_on_eos and eos and int(token.item()) in eos:
                break
            if cache.get_seq_length() + 1 > cache.capacity:
                if trace and cache.compaction_count == 0:
                    snapshot(len(generated), "before_first_eviction")
                if record_events:
                    tick = time.perf_counter()
                    if device.type == "cuda":
                        before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        before.record()
                event = cache.compact_if_needed()
                if record_events:
                    if device.type == "cuda":
                        after.record()
                        device_events.append((before, after))
                    compaction_cpu += time.perf_counter() - tick
                    events.append({"generated_tokens_emitted": len(generated), **event})
                if trace and cache.compaction_count == 1:
                    snapshot(len(generated), "after_first_eviction")
            length = cache.get_seq_length()
            cache.current_query_sources = torch.tensor([full_prefix_tokens + step], device=device)
            cache.current_query_position_ids = next_position
            input_token = teacher_ids[:, step:step+1] if teacher_ids is not None else token
            output = forward(input_ids=input_token,
                             attention_mask=torch.ones(1, length + 1, dtype=pruned.attention_mask.dtype, device=device),
                             position_ids=next_position, cache_position=torch.tensor([length], device=device),
                             past_key_values=cache, use_cache=True, output_attentions=False)
            if output.past_key_values is not cache:
                raise RuntimeError("model replaced the supplied request cache")
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            log_output(output.logits, len(generated) - 1)
            if trace and len(generated) in {512, 1024, 1536, 2048, 3072}:
                snapshot(len(generated), "periodic")
            next_position = next_position + 1
        _synchronize(device)
        elapsed = time.perf_counter() - started
    if versions != tuple((name, parameter._version) for name, parameter in model.named_parameters()):
        raise RuntimeError("model parameter versions changed during cache-policy generation")
    ids = torch.cat(generated, dim=1)
    snapshot(int(ids.shape[1]), "final")
    decode_seconds = max(0., elapsed - ttft)
    metrics = {
        "matched_runtime_revision": RUNTIME_REVISION,
        "timing_scope": "prepared_embeddings_to_final_token",
        "attention_backend_requested": attention_backend(model),
        "attention_output_backend": "shared_torch_sdpa",
        "inactive_window_uses_native_causal_fast_path": True,
        "attention_statistics_scope": None, "attention_statistics_accumulation": None,
        "priority_rule_uses_attention": False, "selection_shared_across_layers_and_heads": True,
        "recent_mode": recent_mode, "priority_phase": phase, "storage": storage,
        "method": policy, "policy": policy, "policy_configuration": cache.configuration,
        "batch_size": 1, "position_policy": "native_original_positions",
        "native_sliding_window": native_sliding_window(model),
        "generated_tokens": int(ids.shape[1]), "stop_on_eos": bool(stop_on_eos),
        "teacher_forced": teacher_ids is not None, "diagnostic_tracing": bool(trace),
        "optimized": bool(optimized), "compaction_instrumentation": bool(record_events),
        "prefill_ttft_seconds": ttft, "generation_setup_seconds": started - entry_started,
        "generation_seconds": elapsed, "decode_seconds": decode_seconds,
        "decode_tokens_per_second": max(0, ids.shape[1] - 1) / max(decode_seconds, 1e-12),
        "compaction_cpu_seconds": compaction_cpu if record_events else None,
        "compaction_device_seconds": (sum(a.elapsed_time(b) for a, b in device_events) / 1000
                                      if record_events and device.type == "cuda" else None),
        "max_active_cache_tokens": cache.peak_active_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "allocated_bytes_before_generation": allocated_before,
        "cache": cache.memory_stats(), "priority_reference": "fixed online original-index priorities; independent v4 implementation",
        "model_parameter_versions_unchanged": True,
    }
    if teacher_ids is not None:
        metrics["teacher_forced_input_ids"] = teacher_ids[0].tolist()
    if trace and not callable(trace):
        metrics["diagnostics"] = diagnostics
    if record_events:
        metrics["compactions"] = events
    return GenerationResult(ids=ids, metrics=metrics, cache=cache)

