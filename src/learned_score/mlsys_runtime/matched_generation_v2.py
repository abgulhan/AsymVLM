"""Versioned Qwen repair: preserve native SDPA dispatch for inactive windows.

V1 always supplied an explicit local-attention mask whenever config contained a
window, even when every source position fitted inside it. Qwen's native SDPA
path omits that mask. Equivalent masks can select different BF16 kernels and
change close greedy ties, invalidating exact native-output parity. This module
omits the mask precisely when source-span bounds prove it is unnecessary.

Frozen v1 code/results stay untouched. Per-head original-position masks still
apply when a native local window is actually active. No weights, model config,
budget, retention policy, or H2O score definition are changed.
"""

from contextlib import contextmanager
import threading
from types import MethodType

import torch.nn.functional as F

from . import matched_generation as v1


_ORIGINAL_GENERATE = v1.generate_matched
_LOCK = threading.Lock()


def window_can_affect_attention(window, first_original_source, last_original_source, generated_inputs):
    """CPU-side conservative bound; no per-layer GPU synchronization."""
    return window is not None and last_original_source + generated_inputs - first_original_source >= window


@contextmanager
def matched_attention_v2(model, cache, allow_exploratory=False):
    family, layers, _ = v1._family_and_layers(model)
    if family != "qwen2_5_vl" or cache.method not in {"asymvlm", "h2o_gqa_native_positions"}:
        raise NotImplementedError("v2 is a Qwen-only native-position repair, not a StreamingLLM mRoPE port")
    if v1.attention_backend(model) != "sdpa":
        raise ValueError("matched methods require the same native SDPA backend")
    window = v1.native_sliding_window(model)
    sources = cache._prefix_source
    first = int(sources[0]) if sources is not None else 0
    last = int(sources[-1]) if sources is not None else cache.prefix_length - 1
    originals = []

    def forward(module, hidden_states, position_embeddings=None, attention_mask=None,
                past_key_value=None, **kwargs):
        if past_key_value is not cache or module.training or hidden_states.shape[0] != 1:
            raise ValueError("matched attention requires eval mode, one request, and its request-local cache")
        if position_embeddings is None or cache.current_query_sources is None:
            raise ValueError("native rotary embeddings and original source indices are required")
        batch, length, _ = hidden_states.shape
        shape = (batch, length, -1, module.head_dim)
        query = module.q_proj(hidden_states).view(shape).transpose(1, 2)
        keys = module.k_proj(hidden_states).view(shape).transpose(1, 2)
        values = module.v_proj(hidden_states).view(shape).transpose(1, 2)
        cos, sin = position_embeddings
        section = module.rope_scaling["mrope_section"]
        query = v1._rotate(query, cos, sin, family, section)
        keys = v1._rotate(keys, cos, sin, family, section)
        keys, values = cache.update(keys, values, module.layer_idx)
        groups = module.num_key_value_groups
        repeated_keys, repeated_values = v1._repeat_kv(keys, groups), v1._repeat_kv(values, groups)
        key_sources = cache.layer_source_positions(module.layer_idx)
        query_sources = cache.current_query_sources
        active_window = window_can_affect_attention(window, first, last, cache._seen_tokens - cache.prefix_length)
        # HF elides the native mask only for physical K < window (strict).
        # At K == window the causal/local mask is semantically redundant for
        # the oldest visible key, but exact BF16 dispatch still must match it.
        native_requires_mask = window is not None and keys.shape[-2] >= window
        shared_positions = cache.compaction_count == 0 or cache.method != "h2o_gqa_native_positions"
        mask = (v1.source_attention_mask(query_sources, key_sources[:, :1] if shared_positions else key_sources,
                                         1 if shared_positions else groups, query.dtype, window)
                if active_window or native_requires_mask else None)
        # Match the pinned native Qwen SDPA path, including its contiguous
        # workaround only when an explicit additive mask is needed.
        if query.device.type == "cuda" and mask is not None:
            query, repeated_keys, repeated_values = (query.contiguous(), repeated_keys.contiguous(),
                                                     repeated_values.contiguous())
        output = F.scaled_dot_product_attention(query, repeated_keys, repeated_values,
                                               attn_mask=mask, dropout_p=0.0,
                                               is_causal=mask is None and length > 1)
        if cache.method == "h2o_gqa_native_positions":
            def observe_scratch(size):
                cache.attention_scratch_peak_bytes = max(cache.attention_scratch_peak_bytes, size)
            mass = v1.accumulate_query_key_mass(query, repeated_keys, query_sources, key_sources, groups,
                                                window, scratch_observer=observe_scratch,
                                                shared_key_positions=cache.compaction_count == 0)
            cache.add_attention_mass(module.layer_idx, mass)
        output = module.o_proj(output.transpose(1, 2).contiguous().reshape(batch, length, -1))
        return output, None, cache

    try:
        for layer in layers:
            originals.append((layer.self_attn, layer.self_attn.__dict__.get("forward")))
            layer.self_attn.forward = MethodType(forward, layer.self_attn)
        yield family
    finally:
        for module, original in originals:
            if original is None:
                del module.forward
            else:
                module.forward = original


def generate_matched(*args, **kwargs):
    """Use the frozen runner/budget/cache with the corrected attention adapter.

    The scoped factory substitution follows the existing DynamicCache runner's
    pattern. It is single-request/process-local and rejects nested requests.
    The original callable is captured before benchmark-driver substitution to
    avoid recursion when the versioned driver routes frozen imports here.
    """
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError("matched-v2 runner cannot be used concurrently in one process")
    original = v1.matched_attention
    try:
        v1.matched_attention = matched_attention_v2
        result = _ORIGINAL_GENERATE(*args, **kwargs)
        result.metrics["matched_runtime_revision"] = "qwen_native_sdpa_mask_dispatch_v2"
        result.metrics["inactive_window_uses_native_causal_fast_path"] = True
        return result
    finally:
        v1.matched_attention = original
        _LOCK.release()
