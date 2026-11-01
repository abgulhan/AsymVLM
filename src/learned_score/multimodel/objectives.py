"""Shared scorer computation and named training objectives."""

import math
from typing import Dict

import torch
import torch.nn.functional as F


def z_normalize(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    return (values - values.mean()) / values.std(
        unbiased=False
    ).clamp_min(1e-8)


def saliency_from_answer_ce_gradient(
    visual_key_bias_gradient: torch.Tensor,
) -> torch.Tensor:
    """Convert answer-CE key-bias gradients to the specified saliency sign."""

    if visual_key_bias_gradient.ndim != 1:
        raise ValueError("visual key-bias gradient must be one-dimensional")
    return z_normalize(
        torch.relu(-visual_key_bias_gradient.detach().float())
    )


def listwise_saliency_loss(
    scorer_scores: torch.Tensor, normalized_saliency: torch.Tensor
) -> torch.Tensor:
    """Distill a token saliency distribution into scorer rankings."""

    if scorer_scores.shape != normalized_saliency.shape:
        raise ValueError("scorer and saliency vectors must have equal shape")
    target = torch.softmax(normalized_saliency.detach().float(), dim=0)
    return -(
        target * torch.log_softmax(z_normalize(scorer_scores), dim=0)
    ).sum()


def normalized_scorer_entropy(scores: torch.Tensor) -> torch.Tensor:
    """Differentiable entropy in [0, 1] used by V4 and the controller."""

    if scores.ndim != 1 or scores.numel() < 2:
        raise ValueError("entropy requires at least two scorer values")
    probabilities = torch.softmax(z_normalize(scores), dim=0)
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum()
    return entropy / math.log(scores.numel())


def ranknet_confidence_loss(
    entropies: torch.Tensor, pruning_distortions: torch.Tensor
) -> torch.Tensor:
    """Make higher scorer entropy correspond to higher physical distortion."""

    if (
        entropies.ndim != 1
        or pruning_distortions.ndim != 1
        or entropies.shape != pruning_distortions.shape
    ):
        raise ValueError("RankNet inputs must be equal one-dimensional vectors")
    left, right = torch.triu_indices(
        entropies.numel(), entropies.numel(), offset=1,
        device=entropies.device,
    )
    target_delta = (
        pruning_distortions[left] - pruning_distortions[right]
    ).detach()
    informative = target_delta != 0
    if not informative.any():
        return entropies.sum() * 0.0
    logits = entropies[left[informative]] - entropies[right[informative]]
    targets = (target_delta[informative] > 0).to(logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, targets)


def score_embeddings(
    vision: torch.Tensor,
    text: torch.Tensor,
    weight: torch.Tensor,
    aggregation: str = "max",
) -> torch.Tensor:
    if vision.ndim == 3:
        vision = vision.squeeze(0)
    if text.ndim == 3:
        text = text.squeeze(0)
    if vision.ndim != 2 or text.ndim != 2:
        raise ValueError("vision and text must have shape [tokens, hidden]")
    if vision.shape[1] != text.shape[1] or vision.shape[1] != weight.numel():
        raise ValueError("hidden dimensions do not match")
    similarity = (
        F.normalize(vision.float(), dim=-1) * weight.float()
    ) @ F.normalize(text.float(), dim=-1).T
    if aggregation == "max":
        return similarity.max(dim=-1).values
    if aggregation == "mean":
        return similarity.mean(dim=-1)
    if aggregation == "normalized_logsumexp":
        return torch.logsumexp(similarity, dim=-1) - math.log(
            similarity.shape[-1]
        )
    raise ValueError(f"unsupported aggregation: {aggregation}")


def paper_prompt_mse(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    prompt_positions: torch.Tensor,
    weight: torch.Tensor,
    regularization_coefficient: float = 0.001,
) -> Dict[str, torch.Tensor]:
    if prompt_positions.ndim != 1 or prompt_positions.numel() == 0:
        raise ValueError("prompt_positions must be a non-empty vector")
    mse = F.mse_loss(
        student_hidden[:, prompt_positions].float(),
        teacher_hidden[:, prompt_positions].float(),
    )
    regularization = regularization_coefficient * (
        weight.float() - 1.0
    ).square().mean()
    return {
        "total": mse + regularization,
        "prompt_mse": mse,
        "weight_regularization": regularization,
    }
