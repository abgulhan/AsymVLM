"""FP32 query-conditioned patch scorers and exact, stable selectors."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def require_finite(value, name):
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"non-finite {name}")
    return value


def _inputs(vision, question):
    if vision.ndim != 2 or question.ndim != 2:
        raise ValueError("embeddings must have shape [tokens, hidden]")
    if not len(vision) or not len(question) or vision.shape[1] != question.shape[1]:
        raise ValueError("nonempty visual/question embeddings must share hidden size")
    return (require_finite(vision.float(), "visual embeddings"),
            require_finite(question.float(), "question embeddings"))


class OriginalScorer(nn.Module):
    """Submitted diagonal similarity with question-only, FP32 computation."""

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.w = nn.Parameter(torch.ones(self.hidden_size, dtype=torch.float32))

    def forward(self, vision, question):
        vision, question = _inputs(vision, question)
        scores = ((F.normalize(vision, dim=-1) * self.w.float())
                  @ F.normalize(question, dim=-1).T).max(dim=-1).values
        return require_finite(scores, "original scores")


class Rank32Scorer(nn.Module):
    """Learn separate visual/question projections; never consume an answer."""

    rank = 32

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.visual = nn.Linear(self.hidden_size, self.rank, bias=False)
        self.question = nn.Linear(self.hidden_size, self.rank, bias=False)
        # Same initialization gives a stable random-projection similarity;
        # subsequent training learns cross-modal alignment independently.
        nn.init.normal_(self.visual.weight, std=1.0 / math.sqrt(self.hidden_size))
        with torch.no_grad():
            self.question.weight.copy_(self.visual.weight)

    def forward(self, vision, question):
        vision, question = _inputs(vision, question)
        visual = F.normalize(F.linear(F.normalize(vision, dim=-1),
                                      self.visual.weight.float()), dim=-1)
        query = F.normalize(F.linear(F.normalize(question, dim=-1),
                                     self.question.weight.float()), dim=-1)
        return require_finite((visual @ query.T).max(dim=-1).values,
                              "rank32 scores")


def build_scorer(variant, hidden_size):
    if variant == "original":
        return OriginalScorer(hidden_size)
    if variant in {"rank32", "rank32_coverage"}:
        return Rank32Scorer(hidden_size)
    raise ValueError(f"unknown scorer variant: {variant}")


def saliency_loss(scores, saliency):
    """Distill the frozen teacher's patch-importance distribution in FP32."""
    scores = require_finite(scores.float(), "student scores")
    saliency = require_finite(saliency.detach().float(), "teacher saliency")
    if scores.ndim != 1 or scores.shape != saliency.shape or not scores.numel():
        raise ValueError("score and saliency vectors must be nonempty and equal")
    normalize = lambda x: (x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)
    target = normalize(saliency).softmax(dim=0)
    return require_finite(-(target * normalize(scores).log_softmax(dim=0)).sum(),
                          "saliency loss")


@torch.no_grad()
def select_tokens(scores, vision, retention, variant="rank32"):
    """Retain floor(N*r), min one; ties follow original patch order.

    Coverage variant reserves ceil(20% of the retained budget) for greedy
    farthest-first cosine coverage after selecting the learned top 80%.
    Protected newline/image-boundary tokens are handled by the model adapter.
    """
    if variant not in {"original", "rank32", "rank32_coverage"}:
        raise ValueError(f"unknown scorer variant: {variant}")
    if not 0 < retention <= 1 or not math.isfinite(retention):
        raise ValueError("retention must be finite and in (0, 1]")
    if scores.ndim != 1 or vision.ndim != 2 or len(scores) != len(vision) or not len(scores):
        raise ValueError("scores and visual embedding counts must agree")
    require_finite(scores, "selection scores")
    require_finite(vision, "selection embeddings")
    count = min(len(scores), max(1, int(len(scores) * retention)))
    if count == len(scores):
        return torch.arange(count, device=scores.device)
    order = torch.argsort(scores.float(), descending=True, stable=True)
    if variant != "rank32_coverage":
        return order[:count].sort().values
    learned_count = max(1, count * 4 // 5)
    selected = torch.empty(count, dtype=torch.long, device=scores.device)
    selected[:learned_count] = order[:learned_count]
    normalized = F.normalize(vision.float(), dim=-1)
    # Stream chosen vectors instead of materializing an N-by-N distance matrix.
    min_distance = torch.full((len(scores),), float("inf"), device=scores.device)
    for start in range(0, learned_count, 64):
        batch = normalized[selected[start:min(start + 64, learned_count)]]
        distance = (1 - normalized @ batch.T).clamp(0, 2).min(dim=1).values
        torch.minimum(min_distance, distance, out=min_distance)
    min_distance[selected[:learned_count]] = -float("inf")
    for offset in range(learned_count, count):
        # argmax returns the first index on ties.
        index = min_distance.argmax().reshape(1)
        selected[offset:offset + 1] = index
        distance = (1 - (normalized @ normalized.index_select(0, index).T).squeeze(1)).clamp(0, 2)
        torch.minimum(min_distance, distance, out=min_distance)
        min_distance.index_fill_(0, index, -float("inf"))
    return selected.sort().values


def optimizer_step(loss, scorer, optimizer, max_grad_norm=1.0):
    """Fail closed before applying a non-finite update."""
    require_finite(loss, "training loss")
    if loss.dtype != torch.float32:
        raise TypeError("scorer loss must be FP32")
    loss.backward()
    parameters = list(scorer.parameters())
    for parameter in parameters:
        if parameter.dtype != torch.float32:
            raise TypeError("scorer parameters must be FP32")
        if parameter.grad is not None:
            require_finite(parameter.grad, "scorer gradient")
    torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm, error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        if isinstance(scorer, OriginalScorer):
            scorer.w.clamp_(0, 3)
        for parameter in parameters:
            require_finite(parameter, "updated scorer weights")
