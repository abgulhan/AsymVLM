"""Shared paper-faithful visual-key masking semantics."""

import torch

from .config import MaskingConfig


def retention_count(token_count: int, keep_ratio: float) -> int:
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("keep_ratio must be in (0, 1]")
    return min(token_count, max(1, int(token_count * keep_ratio)))


def visual_key_penalty(
    scores: torch.Tensor,
    visual_positions: torch.Tensor,
    sequence_length: int,
    config: MaskingConfig,
    hard_forward: bool = False,
) -> torch.Tensor:
    """Return one additive attention-logit penalty per sequence key."""

    config.validate()
    if scores.ndim != 1 or visual_positions.ndim != 1:
        raise ValueError("scores and visual_positions must be one-dimensional")
    if scores.numel() != visual_positions.numel() or scores.numel() == 0:
        raise ValueError("scores and visual positions must be non-empty/equal")
    if visual_positions.min() < 0 or visual_positions.max() >= sequence_length:
        raise ValueError("visual position is outside the sequence")
    if torch.unique(visual_positions).numel() != visual_positions.numel():
        raise ValueError("visual positions must be unique")

    penalty = scores.new_zeros(sequence_length)
    # A full-retention path must be exactly identical to the native model.
    # The historical Phi function did not special-case this endpoint.
    if config.keep_ratio == 1.0 or config.C == 0.0:
        return penalty
    keep_count = retention_count(scores.numel(), config.keep_ratio)
    selected = scores.topk(keep_count).indices
    threshold = scores[selected].min().detach()
    drop_probability = torch.sigmoid((threshold - scores) / config.tau)
    if hard_forward:
        hard_drop = torch.ones_like(scores).scatter(0, selected, 0.0)
        drop_probability = (
            hard_drop.detach()
            - drop_probability.detach()
            + drop_probability
        )
    return penalty.scatter(
        0, visual_positions, -config.C * drop_probability
    )


def phi_prefix_score_mask(
    scores: torch.Tensor,
    boundary: int,
    sequence_length: int,
    config: MaskingConfig,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Convert the shared penalty to recovered Phi's 4D cache-mask shape."""

    positions = torch.arange(
        min(boundary, scores.numel()), device=scores.device
    )
    penalty = visual_key_penalty(
        scores[: len(positions)], positions, sequence_length, config
    )
    return penalty.to(dtype=dtype)[None, None, None, :]
