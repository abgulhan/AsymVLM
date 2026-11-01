"""Paper-faithful differentiable attention masks for LLaVA's Llama decoder."""

import torch


def compute_key_penalty(
    importance,
    image_positions,
    sequence_length,
    keep_ratio,
    mask_c=5.0,
    mask_tau=1.0,
):
    """Map vision-token importance to an additive penalty on key positions."""
    if importance.ndim != 1 or image_positions.ndim != 1:
        raise ValueError("importance and image_positions must be one-dimensional")
    if importance.numel() != image_positions.numel():
        raise ValueError("importance and image_positions must have equal length")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    if mask_tau <= 0.0:
        raise ValueError("mask_tau must be positive")
    if image_positions.numel() == 0:
        raise ValueError("at least one image position is required")
    if image_positions.min() < 0 or image_positions.max() >= sequence_length:
        raise ValueError("image_positions contains an out-of-range index")

    keep_count = max(1, int(importance.numel() * keep_ratio))
    threshold = importance.topk(keep_count).values[-1].detach()
    vision_penalty = -mask_c * torch.sigmoid(
        -(importance - threshold) / mask_tau
    )
    return importance.new_zeros(sequence_length).scatter(
        0, image_positions, vision_penalty
    )


def build_additive_causal_mask(attention_mask, key_penalty, dtype):
    """Build the 4D inverted mask accepted by transformers LlamaModel.

    The key penalty is broadcast over query positions, layers, and attention
    heads. It is added only where a causal attention edge is otherwise valid.
    """
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    batch_size, sequence_length = attention_mask.shape
    if key_penalty.ndim == 1:
        key_penalty = key_penalty.unsqueeze(0)
    if key_penalty.shape not in {
        (1, sequence_length),
        (batch_size, sequence_length),
    }:
        raise ValueError(
            "key_penalty must have shape [sequence] or [batch, sequence]"
        )
    if key_penalty.shape[0] == 1 and batch_size != 1:
        key_penalty = key_penalty.expand(batch_size, -1)

    device = attention_mask.device
    min_value = torch.finfo(dtype).min
    causal = torch.full(
        (sequence_length, sequence_length),
        min_value,
        device=device,
        dtype=dtype,
    ).triu(diagonal=1)
    causal = causal.unsqueeze(0).unsqueeze(0).expand(
        batch_size, 1, -1, -1
    )
    valid_key = attention_mask[:, None, None, :].bool()
    causal = causal.masked_fill(~valid_key, min_value)
    valid_edge = causal.eq(0) & valid_key
    penalty = key_penalty[:, None, None, :].to(dtype=dtype)
    return torch.where(valid_edge, causal + penalty, causal)


def paper_regularization(weight, coefficient=0.001):
    """The explicit scorer regularizer used by the original Phi trainer."""
    return coefficient * (weight - 1.0).pow(2).mean()
