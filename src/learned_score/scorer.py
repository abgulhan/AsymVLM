"""
Learned importance scorer for vision token pruning.

Replaces raw cosine similarity with a learned per-dimension weighting
that is optimized to preserve the model's output quality.

    cross-sim:       importance = (v_norm @ t_norm.T).max(dim=-1)
    learned (ours):  importance = ((v_norm * w) @ t_norm.T).max(dim=-1)

w: [D] is a learnable vector that re-weights embedding dimensions.
Initialized to ones (equivalent to cross-sim at start).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache


class LearnedImportanceScorer(nn.Module):
    """Learned per-dimension weighting for vision-text importance scoring.

    Args:
        hidden_dim: embedding dimension (3072 for Phi-3.5)
    """

    def __init__(self, hidden_dim=3072):
        super().__init__()
        # Initialize to ones = equivalent to cross-sim at start
        self.w = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, vision_embeds, text_embeds, aggregation="max"):
        """Compute per-vision-token importance.

        Args:
            vision_embeds: [1, N_vis, D] or [N_vis, D]
            text_embeds: [1, N_txt, D] or [N_txt, D]
            aggregation: "max", "sum", or "mean"

        Returns:
            importance: [N_vis] scores (higher = more important)
        """
        if vision_embeds.dim() == 3:
            vision_embeds = vision_embeds.squeeze(0)
        if text_embeds.dim() == 3:
            text_embeds = text_embeds.squeeze(0)

        v_norm = F.normalize(vision_embeds.float(), dim=-1)  # [N_vis, D]
        t_norm = F.normalize(text_embeds.float(), dim=-1)    # [N_txt, D]

        # Apply learned per-dimension weighting
        v_weighted = v_norm * self.w  # [N_vis, D]

        # Compute similarity
        sim = v_weighted @ t_norm.T  # [N_vis, N_txt]

        if aggregation == "max":
            importance = sim.max(dim=-1).values  # [N_vis]
        elif aggregation == "sum":
            importance = sim.sum(dim=-1)
        else:
            importance = sim.mean(dim=-1)

        return importance


class ScoreMaskCache(DynamicCache):
    """Cache that applies a pre-computed score mask to attention.

    The mask is computed externally (from the learned scorer) and set
    on the cache before the forward pass. The mask is the same for
    all layers and all heads (uniform pruning).

    Usage:
        cache = ScoreMaskCache()
        cache.set_score_mask(mask)  # [1, 1, 1, seq_len]
        model(inputs_embeds=..., past_key_values=cache, use_cache=True)
    """

    def __init__(self):
        super().__init__()
        self._score_mask = None
        self._current_layer = 0

    def reset(self):
        if hasattr(self, 'key_cache'):
            self.key_cache.clear()
            self.value_cache.clear()
        if hasattr(self, 'layers'):
            self.layers.clear()
        self._score_mask = None
        self._current_layer = 0

    def set_score_mask(self, mask):
        """Set pre-computed score mask. mask: [1, 1, 1, seq_len]"""
        self._score_mask = mask

    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        self._current_layer = layer_idx
        return super().update(key_states, value_states, layer_idx, cache_kwargs)

    def align_attention_mask(self, attention_mask, kv_seq_len):
        """Add score mask to causal attention mask."""
        if self._score_mask is not None:
            # attention_mask: [1, 1, q_len, kv_len]
            # score_mask: [1, 1, 1, seq_len] — broadcasts over q_len
            # Result: [1, 1, q_len, kv_len] — same shape, passes validation
            return attention_mask + self._score_mask
        return attention_mask


def compute_score_mask(importance, boundary, seq_len, keep_ratio, C=5.0, tau=1.0,
                       device=None, dtype=None):
    """Compute score mask from importance scores.

    Args:
        importance: [N_vis] importance scores (from scorer)
        boundary: vision/text boundary
        seq_len: full sequence length
        keep_ratio: fraction of vision tokens to keep
        C: masking strength
        tau: sigmoid temperature

    Returns:
        score_mask: [1, 1, 1, seq_len]
    """
    n_vis = min(boundary, len(importance))
    n_keep = max(1, int(n_vis * keep_ratio))

    # Threshold: importance of the k-th most important token
    threshold = importance[:n_vis].topk(n_keep)[0][-1].detach()

    # Score mask: pruned tokens get -C, kept tokens get 0
    vision_mask = -C * torch.sigmoid(-(importance[:n_vis] - threshold) / tau)

    # Build full sequence mask
    score_mask = torch.zeros(1, 1, 1, seq_len, device=device, dtype=dtype)
    score_mask[0, 0, 0, :n_vis] = vision_mask

    return score_mask
