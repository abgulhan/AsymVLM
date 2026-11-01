"""Matched-total-KV generation with shared SDPA execution and explicit ports.

This is a single-request research adapter, not an upstream reproduction claim.
H2O: cumulative full-prefill/every-decode attention at every layer; GQA query
heads are summed into their KV group. Native model rotary positions are kept.
StreamingLLM: sink + recent selection, raw-key storage and cache-index rotary
position rolling after eviction. Mistral is supported. Qwen's image mRoPE has
no direct 1-D upstream equivalent, so its exploratory flattening is disabled
unless explicitly requested and must never be pooled with primary results.

Pinned reference revisions:
H2O ac75c2a8a9e76832b2a4139b9363373b56336bfb
streaming-llm 2e5042606d69933d88fbf909bd77907456b9b4dd
"""

from contextlib import contextmanager
import time
from types import MethodType

import torch
import torch.nn.functional as F

from .generation import GenerationResult, _synchronize, attention_backend, decoder_forward_function
from .masking import native_sliding_window
from .matched_cache import MatchedKVCache, attention_to_kv_mass


def _family_and_layers(model):
    family = getattr(model.config, "model_type", None)
    if family == "qwen2_5_vl":
        return family, model.model.layers, model.model.rotary_emb
    if family == "llava_next" and model.config.text_config.model_type == "mistral":
        return "mistral", model.language_model.model.layers, model.language_model.model.rotary_emb
    raise NotImplementedError("matched baseline supports only Qwen2.5-VL and LLaVA-NeXT/Mistral")


def _repeat_kv(states, groups):
    if groups == 1:
        return states
    batch, heads, length, dim = states.shape
    return states[:, :, None].expand(batch, heads, groups, length, dim).reshape(
        batch, heads * groups, length, dim)


def source_attention_mask(query_sources, key_sources, groups, dtype, window=None):
    """Native causal/local mask for independently selected KV heads."""
    if query_sources.ndim != 1 or key_sources.ndim != 3 or groups < 1:
        raise ValueError("expected queries[Q], keys[B,Hkv,K], and positive GQA groups")
    allowed = key_sources[..., None, :] <= query_sources[None, None, :, None]
    if window is not None:
        allowed &= key_sources[..., None, :] > query_sources[None, None, :, None] - window
    mask = torch.zeros(allowed.shape, dtype=dtype, device=key_sources.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask.repeat_interleave(groups, dim=1)


def _rotate(states, cos, sin, family, mrope_section=None):
    if family == "qwen2_5_vl":
        sections = mrope_section * 2
        cos = torch.cat([part[i % 3] for i, part in enumerate(cos.split(sections, dim=-1))], dim=-1)
        sin = torch.cat([part[i % 3] for i, part in enumerate(sin.split(sections, dim=-1))], dim=-1)
    half = states.shape[-1] // 2
    rotated = torch.cat((-states[..., half:], states[..., :half]), dim=-1)
    return states * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)


@torch.no_grad()
def accumulate_query_key_mass(query, repeated_keys, query_sources, key_sources, groups, window=None,
                              query_chunk_size=64, scratch_observer=None, shared_key_positions=False):
    """All query/head attention, chunked only to bound prefill temporary memory.

    This recomputes statistics beside the SAME fused SDPA output path used by
    every method. Its extra matmul/softmax, allocations and time are included in
    the request; no sampled layers/heads/queries or cached oracle scores.
    """
    if query_chunk_size < 1:
        raise ValueError("query chunk size must be positive")
    kv_heads = key_sources.shape[1]
    mass = torch.zeros((query.shape[0], kv_heads, repeated_keys.shape[-2]),
                       dtype=torch.float32, device=query.device)
    scale = query.shape[-1] ** -0.5
    for begin in range(0, query.shape[-2], query_chunk_size):
        end = min(query.shape[-2], begin + query_chunk_size)
        # Matmul uses model dtype, then softmax/accumulation use float32. No
        # temperature, normalization over history, or decay is introduced.
        logits = torch.matmul(query[..., begin:end, :], repeated_keys.transpose(-1, -2)) * scale
        # Prefill keys are identical across heads; one broadcast mask avoids a
        # quadratic prefill allocation multiplied by all attention heads.
        mask_sources = key_sources[:, :1] if shared_key_positions else key_sources
        mask_groups = 1 if shared_key_positions else groups
        mask = source_attention_mask(query_sources[begin:end], mask_sources, mask_groups,
                                     logits.dtype, window)
        probabilities = torch.softmax((logits + mask).float(), dim=-1)
        mass.add_(attention_to_kv_mass(probabilities, kv_heads))
        if scratch_observer is not None:
            scratch_observer(sum(x.numel() * x.element_size() for x in (logits, mask, probabilities, mass)))
    return mass


@contextmanager
def matched_attention(model, cache, allow_exploratory=False):
    """Temporarily adapt attention forwards; no model parameters/config changes."""
    family, layers, rotary = _family_and_layers(model)
    if cache.method == "streaming_llm_rolling" and family == "qwen2_5_vl" and not allow_exploratory:
        raise NotImplementedError(
            "StreamingLLM has no validated Qwen image-mRoPE position-rolling port; "
            "allow_exploratory=True is required for the non-primary 1-D adaptation")
    if attention_backend(model) != "sdpa":
        raise ValueError("all matched cache methods require the same SDPA model backend")
    window = native_sliding_window(model)
    originals = []

    def forward(module, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, **kwargs):
        if past_key_value is not cache:
            raise ValueError("matched forward requires its own request-local cache")
        if module.training or hidden_states.shape[0] != 1:
            raise ValueError("matched attention supports eval mode and one unpadded request")
        if position_embeddings is None or cache.current_query_sources is None:
            raise ValueError("model rotary embeddings and original source indices are required")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        keys = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        values = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        groups = module.num_key_value_groups
        section = module.rope_scaling["mrope_section"] if family == "qwen2_5_vl" else None
        if cache.method == "streaming_llm_rolling":
            # The cache contains raw K ONLY. Re-rotation is therefore exact and
            # does not accumulate error from repeatedly undoing/reapplying RoPE.
            keys, values = cache.update(keys, values, module.layer_idx)
            positions = cache.rope_positions
            cos, sin = rotary(values, positions)
            query = _rotate(query, cos[..., -length:, :], sin[..., -length:, :], family, section)
            keys = _rotate(keys, cos, sin, family, section)
            # Raw K payload is already counted as cache. Rotated K and its
            # phase arrays are extra temporary memory, never a hidden free
            # second cache. CUDA request peaks remain the measured authority.
            cache.rotary_scratch_peak_bytes = max(
                cache.rotary_scratch_peak_bytes,
                sum(x.numel() * x.element_size() for x in (cos, sin, query, keys)),
            )
        else:
            cos, sin = position_embeddings
            query = _rotate(query, cos, sin, family, section)
            keys = _rotate(keys, cos, sin, family, section)
            keys, values = cache.update(keys, values, module.layer_idx)
        key_sources = cache.layer_source_positions(module.layer_idx)
        query_sources = cache.current_query_sources
        if cache.method == "streaming_llm_rolling" and cache.compaction_count:
            # Streaming positions and the native window width must agree. Using
            # original sources here would hide old sink tokens after rolling.
            key_sources = torch.arange(keys.shape[-2], device=keys.device)[None, None]
            key_sources = key_sources.expand(1, keys.shape[1], -1)
            query_sources = torch.arange(keys.shape[-2] - length, keys.shape[-2], device=keys.device)
        repeated_keys = _repeat_kv(keys, groups)
        repeated_values = _repeat_kv(values, groups)
        # With no local window, all decode keys are past/current and prefill is
        # contiguous in chronological order even if original image slots skip.
        # Keep native SDPA's causal fast path for prefill in EVERY method.
        shared_positions = cache.compaction_count == 0 or cache.method != "h2o_gqa_native_positions"
        mask = source_attention_mask(query_sources, key_sources[:, :1] if shared_positions else key_sources,
                                     1 if shared_positions else groups, query.dtype, window) if window else None
        output = F.scaled_dot_product_attention(
            query.contiguous(), repeated_keys.contiguous(), repeated_values.contiguous(),
            attn_mask=mask, dropout_p=0.0, is_causal=mask is None and length > 1,
            scale=module.head_dim ** -0.5,
        )
        if cache.method == "h2o_gqa_native_positions":
            def observe_scratch(size):
                cache.attention_scratch_peak_bytes = max(cache.attention_scratch_peak_bytes, size)
            mass = accumulate_query_key_mass(query, repeated_keys, query_sources, key_sources, groups,
                                             window, scratch_observer=observe_scratch,
                                             shared_key_positions=cache.compaction_count == 0)
            cache.add_attention_mass(module.layer_idx, mass)
        output = module.o_proj(output.transpose(1, 2).contiguous().reshape(batch, length, -1))
        return (output, None, cache) if family == "qwen2_5_vl" else (output, None)

    try:
        for layer in layers:
            module = layer.self_attn
            originals.append((module, module.__dict__.get("forward")))
            module.forward = MethodType(forward, module)
        yield family
    finally:
        for module, original in originals:
            if original is None:
                del module.forward
            else:
                module.forward = original


@torch.no_grad()
def generate_matched(model, pruned, max_new_tokens, generated_budget, method, slack=1,
                     total_retained_tokens=None, stop_on_eos=True, record_events=False,
                     on_first_token=None, allow_exploratory=False):
    """Generate with identical per-request total KV payload capacity.

    ``total_retained_tokens`` can fix C to original full-prefix+B across visual
    retention settings. With visual pruning, AsymVLM can then spend freed image
    slots on generated text. Without it C=actually supplied prefix+B. Score and
    scratch memory are additional, measured/reported, not hidden in KV budget.
    """
    entry_started = time.perf_counter()
    if max_new_tokens < 1 or getattr(model, "training", False):
        raise ValueError("positive generation length and eval model required")
    if pruned.inputs_embeds.ndim != 3 or pruned.inputs_embeds.shape[0] != 1:
        raise ValueError("one prepared request is required")
    prefix = pruned.inputs_embeds.shape[1]
    if prefix < 1 or pruned.attention_mask.shape != (1, prefix) or not bool(pruned.attention_mask.bool().all()):
        raise ValueError("prepared prefix must be nonempty and unpadded")
    if pruned.position_ids.shape[-1] != prefix:
        raise ValueError("native model position IDs must cover the prepared prefix")
    device = pruned.inputs_embeds.device
    source = getattr(pruned, "retained_source_positions", None)
    if source is None:
        source = torch.arange(prefix, device=device)
    last_source = int(source[-1])
    cache = MatchedKVCache(prefix, generated_budget, method, slack, prefix_source_positions=source,
                           max_generated_tokens=max_new_tokens - 1,
                           total_retained_tokens=total_retained_tokens)
    forward = decoder_forward_function(model)
    eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(model.config, "eos_token_id", None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    _synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = None
    events, device_events = [], []
    compaction_cpu = 0.0
    with matched_attention(model, cache, allow_exploratory) as family:
        _synchronize(device)
        started = time.perf_counter()
        cache.current_query_sources = source
        cache.current_query_position_ids = pruned.position_ids
        output = forward(inputs_embeds=pruned.inputs_embeds, attention_mask=pruned.attention_mask,
                         position_ids=pruned.position_ids, cache_position=torch.arange(prefix, device=device),
                         past_key_values=cache, use_cache=True, output_attentions=False)
        if output.past_key_values is not cache:
            raise RuntimeError("model replaced the supplied matched cache")
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated = [token]
        _synchronize(device)
        ttft = time.perf_counter() - started
        if on_first_token is not None:
            on_first_token()
        next_position = pruned.position_ids[..., -1:].clone() + 1
        for step in range(max_new_tokens - 1):
            if stop_on_eos and eos and int(token.item()) in eos:
                break
            if cache.get_seq_length() + 1 > cache.capacity:
                tick = time.perf_counter()
                if device.type == "cuda":
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                event = cache.compact_if_needed()
                if device.type == "cuda":
                    after.record()
                    device_events.append((before, after))
                compaction_cpu += time.perf_counter() - tick
                if record_events:
                    events.append({"generated_tokens_emitted": len(generated), **event})
            length = cache.get_seq_length()
            cache.current_query_sources = torch.tensor([last_source + step + 1], device=device)
            cache.current_query_position_ids = next_position
            output = forward(input_ids=token,
                             attention_mask=torch.ones(1, length + 1, dtype=pruned.attention_mask.dtype, device=device),
                             position_ids=next_position, cache_position=torch.tensor([length], device=device),
                             past_key_values=cache, use_cache=True, output_attentions=False)
            if output.past_key_values is not cache:
                raise RuntimeError("model replaced the supplied matched cache")
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            next_position = next_position + 1
        _synchronize(device)
        elapsed = time.perf_counter() - started
    ids = torch.cat(generated, dim=1)
    decode_seconds = max(0.0, elapsed - ttft)
    metrics = {
        "timing_scope": "prepared_embeddings_to_final_token",
        "attention_backend_requested": attention_backend(model),
        "attention_output_backend": "shared_torch_sdpa",
        "attention_statistics_scope": "all_queries_all_heads_all_layers_prefill_and_every_decode" if method.startswith("h2o") else None,
        "attention_statistics_accumulation": "float32_sum_no_decay" if method.startswith("h2o") else None,
        "h2o_gqa_adaptation": "sum_query_heads_within_each_KV_group" if method.startswith("h2o") else None,
        "method": method, "policy": method, "batch_size": 1,
        "position_policy": "cache_index_rolling_after_first_eviction" if method == "streaming_llm_rolling" else "native_original_positions",
        "exploratory_mrope_flattening": method == "streaming_llm_rolling" and family == "qwen2_5_vl",
        "native_sliding_window": native_sliding_window(model),
        "generated_tokens": int(ids.shape[1]), "stop_on_eos": bool(stop_on_eos),
        "prefill_ttft_seconds": ttft, "generation_setup_seconds": started - entry_started,
        "generation_seconds": elapsed, "decode_seconds": decode_seconds,
        "decode_tokens_per_second": max(0, ids.shape[1] - 1) / max(decode_seconds, 1e-12),
        "compaction_cpu_seconds": compaction_cpu,
        "compaction_device_seconds": sum(a.elapsed_time(b) for a, b in device_events) / 1000 if device.type == "cuda" else None,
        "max_active_cache_tokens": cache.peak_active_tokens,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "allocated_bytes_before_generation": allocated_before,
        "cache": cache.memory_stats(),
        "baseline_reference_commits": {
            "H2O": "ac75c2a8a9e76832b2a4139b9363373b56336bfb",
            "streaming_llm": "2e5042606d69933d88fbf909bd77907456b9b4dd",
        },
    }
    if record_events:
        metrics["compactions"] = events
    return GenerationResult(ids=ids, metrics=metrics, cache=cache)
