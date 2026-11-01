"""Label-free geometry features and physical answer supervision for V3."""

import math

import torch
import torch.nn.functional as F


def _grid_shape(prepared, count):
    grid = prepared.metadata.get("image_grid_thw")
    if grid is not None:
        values = torch.as_tensor(grid).reshape(-1, 3)[0].tolist()
        merge = int(round(math.sqrt(max(1, values[1] * values[2] / count))))
        height, width = int(values[1] // merge), int(values[2] // merge)
        if height * width == count:
            return height, width
    side = int(round(math.sqrt(count)))
    return (side, side) if side * side == count else (1, count)


def spatial_redundancy_features(vision, scores, prepared):
    """Cheap answer-free statistics available before generation."""

    values = F.normalize(vision.float(), dim=-1)
    height, width = _grid_shape(prepared, values.shape[0])
    grid = values.reshape(height, width, -1)
    score_grid = scores.float().reshape(height, width)
    similarities = []
    score_pairs = []
    if width > 1:
        similarities.append((grid[:, :-1] * grid[:, 1:]).sum(-1).reshape(-1))
        score_pairs.append((score_grid[:, :-1].reshape(-1), score_grid[:, 1:].reshape(-1)))
    if height > 1:
        similarities.append((grid[:-1] * grid[1:]).sum(-1).reshape(-1))
        score_pairs.append((score_grid[:-1].reshape(-1), score_grid[1:].reshape(-1)))
    adjacent = torch.cat(similarities) if similarities else values.new_zeros(1)
    left = torch.cat([pair[0] for pair in score_pairs])
    right = torch.cat([pair[1] for pair in score_pairs])
    left = left - left.mean()
    right = right - right.mean()
    correlation = (left * right).mean() / (
        left.square().mean().sqrt() * right.square().mean().sqrt()
    ).clamp_min(1e-8)
    centered = values - values.mean(0, keepdim=True)
    # Participation ratio of the feature covariance, computed without SVD.
    gram = centered @ centered.T
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    effective_rank = eigenvalues.sum().square() / eigenvalues.square().sum().clamp_min(1e-8)
    result = {
        "spatial_adjacent_cosine_mean": float(adjacent.mean()),
        "spatial_adjacent_cosine_std": float(adjacent.std(unbiased=False)),
        "spatial_score_autocorrelation": float(correlation),
        "visual_effective_rank_fraction": float(effective_rank / values.shape[0]),
        "visual_grid_aspect_log": float(math.log(max(height, width) / max(1, min(height, width)))),
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise RuntimeError("non-finite V3 spatial/redundancy feature")
    return result


def physical_answer_metrics(adapter, prepared, selected, teacher):
    """CE, teacher KL, and hidden cosine distortion after real deletion."""

    pruned = adapter.physically_prune_with_answer(
        prepared, selected, position_policy="original"
    )
    student = adapter.forward_pruned_with_answer(pruned)
    answer_source = prepared.answer_positions
    answer_pruned = pruned.metadata["answer_positions"]
    source_predictors = answer_source - 1
    pruned_predictors = answer_pruned - 1
    student_logits = student["logits"][:, pruned_predictors].float()
    teacher_logits = teacher["logits"][:, source_predictors].float()
    targets = prepared.native_inputs["input_ids"][:, answer_source]
    ce = F.cross_entropy(
        student_logits.reshape(-1, student_logits.shape[-1]), targets.reshape(-1)
    )
    kl = F.kl_div(
        F.log_softmax(student_logits.reshape(-1, student_logits.shape[-1]), dim=-1),
        F.softmax(teacher_logits.reshape(-1, teacher_logits.shape[-1]), dim=-1),
        reduction="batchmean",
    )
    student_hidden = student["hidden"][:, answer_pruned].float()
    teacher_hidden = teacher["hidden"][:, answer_source].float()
    cosine_distortion = 1.0 - F.cosine_similarity(
        student_hidden, teacher_hidden, dim=-1
    ).mean()
    return {
        "answer_ce": float(ce),
        "answer_kl": float(kl),
        "hidden_cosine_distortion": float(cosine_distortion),
        "teacher_forced_answer_tokens": int(answer_source.numel()),
    }
