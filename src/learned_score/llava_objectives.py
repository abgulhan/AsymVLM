"""Objective utilities for answer-aware LLaVA visual-token scoring.

This module is deliberately model-independent so its masking and loss
semantics can be tested on CPU without loading LLaVA.
"""

from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class GateResult:
    """Soft probabilities and the gate used by the forward pass."""

    soft: torch.Tensor
    forward: torch.Tensor
    threshold: torch.Tensor


def topk_gate(
    scores: torch.Tensor,
    keep_count: int,
    temperature: float,
    mode: str = "soft",
) -> GateResult:
    """Build a fixed-budget soft or straight-through top-k gate.

    ``ste`` has exactly ``keep_count`` ones in the forward pass while its
    backward pass follows the sigmoid relaxation. It masks embeddings; it
    does not physically shorten the sequence during training.
    """

    if scores.ndim != 1 or scores.numel() == 0:
        raise ValueError("scores must be a non-empty one-dimensional tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if mode not in {"soft", "ste"}:
        raise ValueError(f"unsupported gate mode: {mode}")

    keep_count = min(max(int(keep_count), 1), scores.numel())
    chosen = scores.topk(keep_count).indices
    threshold = scores[chosen].min().detach()
    soft = torch.sigmoid((scores - threshold) / temperature)
    if mode == "soft":
        forward = soft
    else:
        hard = torch.zeros_like(scores).scatter(0, chosen, 1.0)
        forward = hard.detach() - soft.detach() + soft
    return GateResult(soft=soft, forward=forward, threshold=threshold)


def answer_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    answer_positions: torch.Tensor,
    input_ids: torch.Tensor,
    kl_temperature: float = 1.0,
    kl_weight: float = 1.0,
    hidden_weight: float = 1.0,
    ce_weight: float = 0.0,
) -> dict:
    """Distill next-token logits and states at teacher-forced answer tokens.

    Hidden states are compared at answer-token positions. Causal logits at
    position ``p - 1`` predict the gold answer token at position ``p``.
    """

    if answer_positions.ndim != 1 or answer_positions.numel() == 0:
        raise ValueError("answer_positions must be a non-empty 1D tensor")
    if answer_positions.min().item() < 1:
        raise ValueError("answer tokens must have a preceding predictor token")
    if kl_temperature <= 0:
        raise ValueError("kl_temperature must be positive")

    predictor_positions = answer_positions - 1
    student_answer_logits = student_logits[:, predictor_positions, :].float()
    teacher_answer_logits = teacher_logits[:, predictor_positions, :].float()
    temperature = float(kl_temperature)
    student_distribution = student_answer_logits.reshape(
        -1, student_answer_logits.shape[-1]
    )
    teacher_distribution = teacher_answer_logits.reshape(
        -1, teacher_answer_logits.shape[-1]
    )
    kl = F.kl_div(
        F.log_softmax(student_distribution / temperature, dim=-1),
        F.softmax(teacher_distribution / temperature, dim=-1),
        reduction="batchmean",
    ) * temperature**2
    hidden = F.mse_loss(
        student_hidden[:, answer_positions, :].float(),
        teacher_hidden[:, answer_positions, :].float(),
    )
    targets = input_ids[:, answer_positions]
    ce = F.cross_entropy(
        student_answer_logits.reshape(-1, student_answer_logits.shape[-1]),
        targets.reshape(-1),
    )
    total = kl_weight * kl + hidden_weight * hidden + ce_weight * ce
    return {"total": total, "kl": kl, "hidden": hidden, "ce": ce}


def weighted_redundancy_loss(
    vision_embeds: torch.Tensor, selection_weights: torch.Tensor
) -> torch.Tensor:
    """Penalize cosine redundancy among tokens receiving selection mass."""

    if vision_embeds.ndim != 2:
        raise ValueError("vision_embeds must have shape [tokens, hidden]")
    if selection_weights.shape != vision_embeds.shape[:1]:
        raise ValueError("selection_weights must have shape [tokens]")
    if vision_embeds.shape[0] < 2:
        return selection_weights.sum() * 0.0

    normalized = F.normalize(vision_embeds.float(), dim=-1)
    similarity = (normalized @ normalized.T).clamp_min(0.0)
    similarity.fill_diagonal_(0.0)
    weights = selection_weights.float().clamp_min(0.0)
    pair_weights = weights[:, None] * weights[None, :]
    normalizer = pair_weights.sum().clamp_min(1e-8)
    return (pair_weights * similarity).sum() / normalizer


def hybrid_relevance_diversity(
    scores: torch.Tensor,
    vision_embeds: torch.Tensor,
    keep_count: int,
    diversity_strength: float,
) -> torch.Tensor:
    """Greedy relevance/coverage selector for learned-score evaluation.

    Scores and minimum cosine distance to the selected set are normalized to
    comparable ranges. ``diversity_strength=0`` recovers relevance top-k.
    """

    if scores.ndim != 1 or vision_embeds.ndim != 2:
        raise ValueError("expected scores [tokens] and embeddings [tokens, dim]")
    if scores.shape[0] != vision_embeds.shape[0]:
        raise ValueError("score and embedding token counts differ")
    if diversity_strength < 0:
        raise ValueError("diversity_strength must be non-negative")
    keep_count = min(max(int(keep_count), 1), scores.numel())
    if diversity_strength == 0:
        return scores.topk(keep_count).indices.sort().values

    relevance = _minmax(scores.float())
    normalized = F.normalize(vision_embeds.float(), dim=-1)
    distance = (1.0 - normalized @ normalized.T).clamp(0.0, 2.0)
    selected: List[int] = [int(relevance.argmax().item())]
    unavailable = torch.zeros_like(relevance, dtype=torch.bool)
    unavailable[selected[0]] = True
    while len(selected) < keep_count:
        coverage = distance[selected].min(dim=0).values
        objective = relevance + diversity_strength * _minmax(coverage)
        objective = objective.masked_fill(unavailable, -float("inf"))
        chosen = int(objective.argmax().item())
        selected.append(chosen)
        unavailable[chosen] = True
    return torch.tensor(
        sorted(selected), device=scores.device, dtype=torch.long
    )


def expanded_token_positions_for_char_span(
    tokenizer,
    text: str,
    char_start: int,
    char_end: int,
    expanded_input_ids: Sequence[int],
    image_token_id: int,
) -> torch.Tensor:
    """Map a character span through LLaVA's expanded image placeholders."""

    try:
        encoded = tokenizer(
            text,
            add_special_tokens=True,
            return_offsets_mapping=True,
        )
        offsets = _as_list(encoded["offset_mapping"])
        plain_positions = [
            index
            for index, (start, end) in enumerate(offsets)
            if end > char_start and start < char_end and end > start
        ]
    except (NotImplementedError, TypeError, ValueError, KeyError):
        encoded = tokenizer(text, add_special_tokens=True)
        plain_positions = _prefix_aligned_token_span(
            tokenizer, text, char_start, char_end, encoded["input_ids"]
        )
    plain_ids = _as_list(encoded["input_ids"])
    expanded_ids = _as_list(expanded_input_ids)
    mapping = _align_expanded_image_tokens(
        plain_ids, expanded_ids, image_token_id
    )
    positions: List[int] = []
    for plain_index in plain_positions:
        positions.extend(mapping[plain_index])
    if not positions:
        raise ValueError(
            f"character span [{char_start}, {char_end}) mapped to no tokens"
        )
    return torch.tensor(positions, dtype=torch.long)


def _prefix_aligned_token_span(
    tokenizer,
    text: str,
    char_start: int,
    char_end: int,
    full_input_ids,
) -> List[int]:
    """Recover a token span for slow tokenizers without offset mappings."""

    full_ids = _as_list(full_input_ids)
    before_ids = _as_list(
        tokenizer(text[:char_start], add_special_tokens=True)["input_ids"]
    )
    through_ids = _as_list(
        tokenizer(text[:char_end], add_special_tokens=True)["input_ids"]
    )
    start = _common_prefix_length(full_ids, before_ids)
    end = _common_prefix_length(full_ids, through_ids)
    if end <= start:
        raise ValueError(
            "slow-tokenizer prefix alignment produced an empty token span"
        )
    return list(range(start, end))


def _align_expanded_image_tokens(
    plain_ids: Sequence[int],
    expanded_ids: Sequence[int],
    image_token_id: int,
) -> List[List[int]]:
    mapping: List[List[int]] = []
    expanded_index = 0
    for plain_index, token_id in enumerate(plain_ids):
        if token_id == image_token_id:
            start = expanded_index
            while (
                expanded_index < len(expanded_ids)
                and expanded_ids[expanded_index] == image_token_id
            ):
                expanded_index += 1
            if expanded_index == start:
                raise ValueError("processor output has no expanded image tokens")
            mapping.append(list(range(start, expanded_index)))
            continue
        if (
            expanded_index >= len(expanded_ids)
            or expanded_ids[expanded_index] != token_id
        ):
            actual = (
                None
                if expanded_index >= len(expanded_ids)
                else expanded_ids[expanded_index]
            )
            raise ValueError(
                "tokenizer/processor alignment failed at plain token "
                f"{plain_index}: expected {token_id}, found {actual}"
            )
        mapping.append([expanded_index])
        expanded_index += 1
    if expanded_index != len(expanded_ids):
        raise ValueError(
            "processor output contains trailing tokens absent from tokenizer"
        )
    return mapping


def _as_list(value: Iterable):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list) and len(value) == 1:
        return value[0]
    return list(value)


def _common_prefix_length(left: Sequence[int], right: Sequence[int]) -> int:
    length = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        length += 1
    return length


def _minmax(values: torch.Tensor) -> torch.Tensor:
    return (values - values.min()) / (
        values.max() - values.min()
    ).clamp_min(1e-8)
