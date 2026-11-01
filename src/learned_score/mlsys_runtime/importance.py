"""Sampled final-layer attention plus a protected recent generated window.

This is an approximation: it accumulates one final-layer attention row every
32 generated inputs, rather than every layer/step. The model continues using
SDPA with output_attentions=False. There is no prefill attention computation in
this policy and no quadratic Q-by-K matrix: auxiliary queries always have Q=1.

Usage (one request per context):
    with SampledAttentionPolicy(model) as policy:
        result = generate_bounded(model, pruned, ..., policy=policy)
        result.metrics["importance_policy"] = policy.metrics()

The implementation is pinned to the inspected Transformers 4.51.3 Qwen2.5-VL
and LLaVA-NeXT/Mistral attention APIs. It deliberately rejects unknown models.
"""

import math
import weakref

import torch
import transformers

from .cache import ProtectedPrefixCache
from .generation import attention_backend


_ACTIVE_MODELS = weakref.WeakKeyDictionary()


def select_scored_recent(scores, keep_count, recent_window):
    """Return sorted suffix-relative indices, with stable older-token ties."""
    if scores.ndim != 1 or not scores.dtype.is_floating_point:
        raise ValueError("scores must be a one-dimensional floating tensor")
    if recent_window < 0 or not recent_window <= keep_count <= scores.numel():
        raise ValueError("retained budget must cover the recent window and fit the suffix")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("nonfinite sampled attention scores")
    recent_start = scores.numel() - recent_window
    old_count = keep_count - recent_window
    old = torch.argsort(scores[:recent_start], descending=True, stable=True)[:old_count]
    recent = torch.arange(recent_start, scores.numel(), device=scores.device)
    return torch.cat((old, recent)).sort().values


class SampledAttentionPolicy:
    """Install scoped hooks, maintain GPU scores, and choose cache entries.

    ``on_refresh`` is an optional diagnostic callback receiving the sampled
    attention row and rotated query. It must not be enabled in timing runs.
    All persistent score values remain on the model device. Shape/count checks
    use Python metadata; finite/selection checks synchronize only at eviction.
    """

    name = "sampled_final_layer_attention_plus_recent"

    def __init__(self, model, recent_window=128, refresh_interval=32, on_refresh=None):
        if recent_window < 0 or refresh_interval < 1:
            raise ValueError("recent_window must be nonnegative and refresh_interval positive")
        if transformers.__version__ != "4.51.3":
            raise NotImplementedError("sampled attention hooks are validated only for Transformers 4.51.3")
        if getattr(model, "training", False):
            raise ValueError("sampled attention policy requires model.eval()")
        if attention_backend(model) != "sdpa":
            raise NotImplementedError("sampled attention policy currently requires the SDPA backend")
        family = getattr(model.config, "model_type", None)
        if family == "qwen2_5_vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
                Qwen2_5_VLSdpaAttention, apply_multimodal_rotary_pos_emb,
            )
            attention = model.model.layers[-1].self_attn
            if not isinstance(attention, Qwen2_5_VLSdpaAttention):
                raise NotImplementedError("unexpected Qwen2.5-VL attention implementation")
            self._rotate = apply_multimodal_rotary_pos_emb
        elif family == "llava_next" and getattr(model.config.text_config, "model_type", None) == "mistral":
            from transformers.models.mistral.modeling_mistral import (
                MistralAttention, apply_rotary_pos_emb,
            )
            attention = model.language_model.model.layers[-1].self_attn
            if not isinstance(attention, MistralAttention):
                raise NotImplementedError("unexpected LLaVA-NeXT/Mistral attention implementation")
            self._rotate = apply_rotary_pos_emb
        else:
            raise NotImplementedError(f"unsupported sampled-attention model: {family!r}")
        self.model = model
        self.family = family
        self.attention = attention
        self.recent_window = int(recent_window)
        self.refresh_interval = int(refresh_interval)
        self.on_refresh = on_refresh
        self._handles = []
        self._last_metrics = None
        self._clear_request()

    def _clear_request(self):
        self._cache = None
        self._scores = None
        self._score_length = 0
        self._pending = None
        self._raw_query = None
        self._forwards = 0
        self._decode_steps = 0
        self._refreshes = 0
        self._selection_events = 0
        self._largest_row_elements = 0
        self._query_projection_captures = 0

    def __enter__(self):
        if self._handles or self.model in _ACTIVE_MODELS:
            raise RuntimeError("only one sampled policy context may be active on a model")
        if attention_backend(self.model) != "sdpa" or getattr(self.model, "training", False):
            raise ValueError("model backend/training state changed before policy context entry")
        self._clear_request()
        self._last_metrics = None
        _ACTIVE_MODELS[self.model] = self
        try:
            self._handles.append(self.attention.register_forward_pre_hook(self._before_attention, with_kwargs=True))
            self._handles.append(self.attention.q_proj.register_forward_hook(self._capture_query))
            self._handles.append(self.attention.register_forward_hook(self._after_attention, with_kwargs=True))
        except BaseException:
            for handle in self._handles:
                handle.remove()
            self._handles.clear()
            _ACTIVE_MODELS.pop(self.model, None)
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._last_metrics = self._metrics()
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        _ACTIVE_MODELS.pop(self.model, None)
        self._cache = self._scores = self._pending = self._raw_query = None
        return False

    @torch.no_grad()
    def _before_attention(self, module, args, kwargs):
        if kwargs.get("output_attentions", False):
            raise ValueError("sampled policy requires output_attentions=False")
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        cache = kwargs.get("past_key_value")
        if hidden is None or hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError("sampled attention requires one unbatched request")
        if not isinstance(cache, ProtectedPrefixCache):
            raise TypeError("sampled policy must run with ProtectedPrefixCache")
        if self._cache is None:
            if cache.get_seq_length(module.layer_idx) != 0 or hidden.shape[1] != cache.prefix_length:
                raise ValueError("install the policy before the request's complete prefill")
            if cache.generated_budget < self.recent_window:
                raise ValueError("generated budget must cover the protected recent window")
            self._cache = cache
            self._scores = torch.zeros(cache.capacity - cache.prefix_length, dtype=torch.float32, device=hidden.device)
        elif cache is not self._cache:
            raise ValueError("each policy context supports one request; open a new context for another")
        self._pending = None
        self._raw_query = None
        if self._forwards:
            if hidden.shape[1] != 1:
                raise ValueError("sampled attention supports single-token decode, not chunked prefill")
            self._decode_steps += 1
            if self._decode_steps % self.refresh_interval == 0:
                position_embeddings = kwargs.get("position_embeddings")
                if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
                    raise ValueError("native attention must supply its RoPE cos/sin tensors")
                mask = kwargs.get("attention_mask")
                if mask is not None and (mask.ndim != 4 or mask.shape[-2] != 1):
                    raise ValueError("expected a single-query native additive attention mask")
                self._pending = {"position_embeddings": position_embeddings, "mask": mask}
        self._forwards += 1

    @torch.no_grad()
    def _capture_query(self, module, args, output):
        if self._pending is not None:
            if output.ndim != 3 or output.shape[:2] != (1, 1):
                raise ValueError("only one decode query may be sampled")
            self._raw_query = output.detach()
            self._query_projection_captures += 1

    @torch.no_grad()
    def _after_attention(self, module, args, kwargs, output):
        cache = self._cache
        length = cache.get_seq_length(module.layer_idx) - cache.prefix_length
        if length < self._score_length or length > self._scores.numel():
            raise RuntimeError("score/cache lengths changed without policy-directed compaction")
        self._score_length = length
        if self._pending is None:
            return
        if self._raw_query is None:
            raise RuntimeError("native query projection hook did not run")
        query = self._raw_query.view(1, 1, -1, module.head_dim).transpose(1, 2)
        cos, sin = self._pending["position_embeddings"]
        # Reusing the inspected native helper also preserves Qwen's interleaved
        # temporal/height/width mRoPE channel sections. The dummy key has Q=1;
        # actual cached keys have already been rotated by the native layer.
        if self.family == "qwen2_5_vl":
            query, _ = self._rotate(query, query, cos, sin, module.rope_scaling["mrope_section"])
        else:
            query, _ = self._rotate(query, query, cos, sin)
        keys = cache.key_cache[module.layer_idx]
        # Group query heads by their shared KV head. This is equivalent to
        # native repeat_kv, without allocating a repeated copy of the prefix K.
        grouped_query = query.reshape(1, keys.shape[1], module.num_key_value_groups, module.head_dim)
        logits = torch.matmul(grouped_query, keys.transpose(-2, -1)).reshape(1, query.shape[1], 1, keys.shape[-2])
        if self.family == "qwen2_5_vl":
            logits = logits / math.sqrt(module.head_dim)
        else:
            logits = logits * module.scaling
        mask = self._pending["mask"]
        if mask is not None:
            logits = logits + mask[:, :, :, :keys.shape[-2]]
        # This is the inspected Qwen eager-row precision rule. It is a policy
        # score calculation only; model inference itself continues through SDPA.
        if self.family == "qwen2_5_vl" and query.dtype == torch.float16:
            logits = torch.where(torch.isinf(logits), torch.zeros_like(logits), logits)
        probabilities = torch.softmax(logits, dim=-1, dtype=torch.float32).to(query.dtype)
        score_update = probabilities[..., cache.prefix_length:].float().mean(dim=(0, 1, 2))
        self._scores[:length].add_(score_update)
        self._refreshes += 1
        self._largest_row_elements = max(self._largest_row_elements, probabilities.numel())
        if self.on_refresh is not None:
            self.on_refresh({
                "decode_step": self._decode_steps,
                "rotated_query": query,
                "rotated_keys": keys,
                "attention_row": probabilities,
                "native_mask": mask,
                "source_positions": cache.source_positions,
                "generated_scores": self._scores[:length],
            })
        self._pending = self._raw_query = None

    @torch.no_grad()
    def select(self, source_positions, keep_count):
        if not self._handles or self._cache is None:
            raise RuntimeError("use the policy inside its context after a prefill")
        if source_positions.ndim != 1 or source_positions.numel() != self._score_length:
            raise ValueError("supplied generated positions do not match the score cache")
        current = self._cache.source_positions[self._cache.prefix_length:]
        if source_positions.device != current.device or source_positions.data_ptr() != current.data_ptr():
            raise ValueError("selection requires the current cache's generated-position view")
        if self._refreshes == 0 and keep_count > self.recent_window:
            raise RuntimeError("no sampled attention observations before importance selection")
        indices = select_scored_recent(self._scores[:self._score_length], keep_count, self.recent_window)
        retained_scores = self._scores.index_select(0, indices)
        self._scores[:keep_count].copy_(retained_scores)
        self._scores[keep_count:].zero_()
        self._score_length = keep_count
        self._selection_events += 1
        return indices

    def _metrics(self):
        return {
            "policy": self.name,
            "approximation": "sum_of_sampled_final_layer_decode_attention_rows",
            "recent_generated_tokens": self.recent_window,
            "refresh_interval_generated_tokens": self.refresh_interval,
            "decode_steps": self._decode_steps,
            "sampled_queries": self._refreshes,
            "captured_query_projections": self._query_projection_captures,
            "selection_events": self._selection_events,
            "persistent_score_bytes": 0 if self._scores is None else self._scores.numel() * self._scores.element_size(),
            "largest_auxiliary_attention_row_elements": self._largest_row_elements,
            "prefill_attention_rows_computed": 0,
            "model_attention_weights_requested": False,
            "diagnostic_callback_enabled": self.on_refresh is not None,
            "validated_transformers_version": "4.51.3",
        }

    def metrics(self):
        return dict(self._last_metrics) if self._last_metrics is not None and not self._handles else self._metrics()
