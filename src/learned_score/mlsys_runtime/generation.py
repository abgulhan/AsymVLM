"""Single-request generation for existing Qwen/LLaVA prepared-input adapters."""

import inspect
import time
from dataclasses import dataclass
from typing import Any

import torch

from .cache import ProtectedPrefixCache
from .masking import logical_attention_mask, native_sliding_window
from .policies import RecencyPolicy


@dataclass
class GenerationResult:
    ids: torch.Tensor
    metrics: dict[str, Any]
    cache: ProtectedPrefixCache


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def attention_backend(model):
    config = model.config
    text_config = getattr(config, "text_config", config)
    return getattr(text_config, "_attn_implementation", None) or getattr(config, "_attn_implementation", None)


def decoder_forward_function(model):
    """Resolve last-token logits support once, not once per generated token."""
    parameters = inspect.signature(model.forward).parameters
    if "logits_to_keep" in parameters:
        return lambda **kwargs: model(logits_to_keep=1, return_dict=True, **kwargs)
    if "num_logits_to_keep" in parameters:
        return lambda **kwargs: model(num_logits_to_keep=1, return_dict=True, **kwargs)
    # Qwen2/2.5-VL wrappers in the pinned Transformers 4.51 environment expose
    # their text decoder as .model and their output head as .lm_head.
    if getattr(model.config, "model_type", "") in {"qwen2_vl", "qwen2_5_vl"}:
        def forward(**kwargs):
            from types import SimpleNamespace
            output = model.model(return_dict=True, **kwargs)
            return SimpleNamespace(
                logits=model.lm_head(output.last_hidden_state[:, -1:]),
                past_key_values=output.past_key_values,
            )
        return forward
    return lambda **kwargs: model(return_dict=True, **kwargs)


@torch.no_grad()
def generate_bounded(
    model,
    pruned,
    max_new_tokens,
    generated_budget,
    slack=32,
    stop_on_eos=True,
    policy=None,
    require_fused_attention=True,
    record_events=False,
    on_first_token=None,
):
    """Generate from *already prepared* embeddings using bounded KV storage.

    Reported prefill TTFT excludes image processing/selection. ``run_request``
    wraps the adapter and selector when end-to-end request timing is required.
    One request, greedy decoding and an unpadded protected prefix are supported.
    This routine never asks the model to materialize attention weights.
    """
    entry_started = time.perf_counter()
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if pruned.inputs_embeds.ndim != 3 or pruned.inputs_embeds.shape[0] != 1:
        raise ValueError("generate_bounded supports exactly one request")
    prefix_length = pruned.inputs_embeds.shape[1]
    if prefix_length < 1:
        raise ValueError("generation requires a nonempty protected prefix")
    if pruned.attention_mask.shape != (1, prefix_length) or not bool(pruned.attention_mask.bool().all()):
        raise ValueError("the prepared prefix must be unpadded with an all-valid attention mask")
    if pruned.position_ids.shape[-1] != prefix_length:
        raise ValueError("position IDs must match the protected prefix")
    if getattr(model, "training", False):
        raise ValueError("put the model in eval mode before measuring generation")
    backend = attention_backend(model)
    if require_fused_attention and backend not in {"sdpa", "flash_attention_2", "flash_attention_3"}:
        raise ValueError(f"optimized generation requires SDPA/FlashAttention, found {backend!r}")
    device = pruned.inputs_embeds.device
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = None
    source_positions = getattr(pruned, "retained_source_positions", None)
    window = native_sliding_window(model)
    last_source = int(source_positions[-1]) if source_positions is not None else prefix_length - 1
    first_source = int(source_positions[0]) if source_positions is not None else 0
    window_relevant = window is not None and last_source + max_new_tokens - 1 - first_source >= window
    if window_relevant and backend != "sdpa":
        raise NotImplementedError(
            "native sliding-window visibility after compaction currently requires "
            "SDPA with an explicit logical-position mask"
        )
    if source_positions is None:
        source_positions = torch.arange(prefix_length, device=device)
    prefill_mask = (
        logical_attention_mask(source_positions, source_positions, pruned.inputs_embeds.dtype, window)
        if window is not None and last_source - first_source >= window else pruned.attention_mask
    )
    cache = ProtectedPrefixCache(
        prefix_length, generated_budget, slack,
        prefix_source_positions=source_positions,
        # The last emitted token is not forwarded into the decoder cache.
        max_generated_tokens=max_new_tokens - 1,
    )
    policy = RecencyPolicy() if policy is None else policy
    forward = decoder_forward_function(model)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    events = []
    compaction_cuda_events = []
    compaction_cpu_seconds = 0.0
    _synchronize(device)
    started = time.perf_counter()
    output = forward(
        inputs_embeds=pruned.inputs_embeds,
        attention_mask=prefill_mask,
        position_ids=pruned.position_ids,
        cache_position=torch.arange(prefix_length, device=device),
        past_key_values=cache,
        use_cache=True,
        output_attentions=False,
    )
    if output.past_key_values is not cache:
        raise RuntimeError("model replaced the supplied bounded cache")
    token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [token]
    _synchronize(device)
    ttft = time.perf_counter() - started
    if on_first_token is not None:
        on_first_token()
    logical_position = pruned.position_ids[..., -1:].clone() + 1
    next_source = last_source + 1
    max_active_tokens = cache.get_seq_length()
    for _ in range(max_new_tokens - 1):
        if stop_on_eos and eos and int(token.item()) in eos:
            break
        if cache.get_seq_length() + 1 > cache.capacity:
            cpu_start = time.perf_counter()
            if device.type == "cuda":
                cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                cuda_start.record()
            event = cache.compact_if_needed(policy=policy)
            if device.type == "cuda":
                cuda_end.record()
                compaction_cuda_events.append((cuda_start, cuda_end))
            compaction_cpu_seconds += time.perf_counter() - cpu_start
            if record_events and event is not None:
                events.append({
                    "generated_tokens_emitted": len(generated),
                    "removed_tokens": event.removed_tokens,
                    "active_cache_tokens": cache.get_seq_length(),
                })
        length = cache.get_seq_length()
        step_sources = torch.tensor([next_source], device=device)
        step_mask = (
            logical_attention_mask(
                step_sources, torch.cat((cache.source_positions, step_sources)),
                pruned.inputs_embeds.dtype, window,
            )
            if window is not None and next_source - first_source >= window else
            torch.ones((1, length + 1), dtype=pruned.attention_mask.dtype, device=device)
        )
        output = forward(
            input_ids=token,
            attention_mask=step_mask,
            position_ids=logical_position,
            cache_position=torch.tensor([length], device=device),
            past_key_values=cache,
            use_cache=True,
            output_attentions=False,
        )
        if output.past_key_values is not cache:
            raise RuntimeError("model replaced the supplied bounded cache")
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(token)
        logical_position = logical_position + 1
        next_source += 1
        max_active_tokens = max(max_active_tokens, cache.get_seq_length())
    _synchronize(device)
    elapsed = time.perf_counter() - started
    ids = torch.cat(generated, dim=1)
    decode_seconds = max(0.0, elapsed - ttft)
    metrics = {
        "timing_scope": "prepared_embeddings_to_final_token",
        "attention_backend_requested": backend,
        "attention_weights_requested": False,
        "native_sliding_window": window,
        "sliding_window_mask_uses_original_positions": window_relevant,
        "batch_size": 1,
        "policy": getattr(policy, "name", type(policy).__name__),
        "generated_tokens": int(ids.shape[1]),
        "stop_on_eos": bool(stop_on_eos),
        "prefill_ttft_seconds": ttft,
        "generation_setup_seconds": started - entry_started,
        "generation_seconds": elapsed,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": max(0, ids.shape[1] - 1) / max(decode_seconds, 1e-12),
        "compaction_cpu_seconds": compaction_cpu_seconds,
        "compaction_device_seconds": (
            sum(start.elapsed_time(end) for start, end in compaction_cuda_events) / 1000
            if device.type == "cuda" else None
        ),
        "max_active_cache_tokens": max_active_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "allocated_bytes_before_generation": allocated_before,
        "cache": cache.memory_stats(),
    }
    if record_events:
        metrics["compactions"] = events
    return GenerationResult(ids=ids, metrics=metrics, cache=cache)


def run_request(adapter, row, select_visual, **generation_kwargs):
    """Time adapter preparation, selection, compaction and generation together.

    ``select_visual(prepared)`` returns visual-local retained indices. Selection
    strategy is independent of the cache policy. Data loading before ``row``
    was supplied is outside this boundary, but adapter image loading is inside.
    """
    device = torch.device(adapter.device)
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    request_start = time.perf_counter()
    prepared = adapter.prepare_sample(row)
    _synchronize(device)
    prepared_at = time.perf_counter()
    selected = select_visual(prepared)
    pruned = adapter.physically_prune(prepared, selected, "original")
    _synchronize(device)
    selected_at = time.perf_counter()
    preprocessing_peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    result = generate_bounded(adapter.model, pruned, **generation_kwargs)
    result.metrics.update({
        "request_timing_scope": "adapter_image_loading_through_final_generated_token",
        "prepare_seconds": prepared_at - request_start,
        "selection_and_visual_compaction_seconds": selected_at - prepared_at,
        "full_request_seconds": time.perf_counter() - request_start,
        "full_request_ttft_seconds": (
            selected_at - request_start + result.metrics["generation_setup_seconds"]
            + result.metrics["prefill_ttft_seconds"]
        ),
        "full_request_peak_allocated_bytes": (
            max(preprocessing_peak, result.metrics["peak_allocated_bytes"])
            if device.type == "cuda" else None
        ),
        "original_visual_tokens": int(prepared.visual_positions.numel()),
        "retained_visual_tokens": int(selected.numel()),
    })
    return result
