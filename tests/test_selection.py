"""Visual-token selection: learned top-k, DivPrune and CDPruner baselines."""

import math

import pytest
import torch

models = pytest.importorskip("learned_score.mlsys_scorer.models")


def reference_topk(scores, ratio):
    """Paper rule: keep k = ceil(N * r) highest-scoring patches, original order."""
    k = min(len(scores), max(1, math.ceil(len(scores) * ratio)))
    return torch.argsort(scores, descending=True, stable=True)[:k].sort().values


def _check_indices(idx, n, k):
    assert idx.dtype == torch.long
    assert len(idx) == k
    assert len(idx.unique()) == k
    assert int(idx.min()) >= 0 and int(idx.max()) < n


@pytest.mark.parametrize("n", [1, 7, 64, 577])
@pytest.mark.parametrize("ratio", [0.1, 0.25, 0.4, 0.5, 1.0])
def test_reference_topk_properties(n, ratio):
    torch.manual_seed(n)
    scores = torch.randn(n)
    idx = reference_topk(scores, ratio)
    k = min(n, max(1, math.ceil(n * ratio)))
    _check_indices(idx, n, k)
    assert torch.equal(idx, idx.sort().values)
    assert int(scores.argmax()) in idx.tolist()
    # Every kept score >= every dropped score.
    mask = torch.zeros(n, dtype=torch.bool)
    mask[idx] = True
    if (~mask).any():
        assert scores[mask].min() >= scores[~mask].max()


@pytest.mark.parametrize("variant", ["rank32", "original", "rank32_coverage"])
@pytest.mark.parametrize("n,ratio", [(10, 0.4), (577, 0.1), (2880, 0.4), (33, 0.9)])
def test_runtime_select_tokens(variant, n, ratio):
    torch.manual_seed(5)
    scores, vision = torch.randn(n), torch.randn(n, 16)
    idx = models.select_tokens(scores, vision, ratio, variant)
    k = min(n, max(1, int(n * ratio)))  # runtime documents floor(N*r), min one
    _check_indices(idx, n, k)
    assert torch.equal(idx, idx.sort().values)
    assert int(scores.argmax()) in idx.tolist()
    assert torch.equal(idx, models.select_tokens(scores, vision, ratio, variant))
    if variant != "rank32_coverage":
        top = torch.argsort(scores, descending=True, stable=True)[:k].sort().values
        assert torch.equal(idx, top)


def test_runtime_count_differs_from_ceil_only_by_rounding():
    """Floor vs ceil retained count never differs by more than one patch."""
    for n in range(1, 300, 7):
        for r in (0.1, 0.25, 0.4, 0.5):
            floor_k = len(models.select_tokens(torch.randn(n), torch.randn(n, 4), r))
            assert 0 <= math.ceil(n * r) - floor_k <= 1 or floor_k == 1


def test_ties_follow_original_order():
    idx = models.select_tokens(torch.zeros(10), torch.randn(10, 4), 0.5)
    assert idx.tolist() == [0, 1, 2, 3, 4]


# --------------------------------------------------------------------------- #
# Diversity baselines
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("n,k", [(2, 1), (2, 2), (50, 5), (196, 78), (64, 64)])
def test_stable_divprune(n, k):
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_divprune_v15c")
    torch.manual_seed(n + k)
    vision = torch.randn(n, 24)
    idx = mod.stable_divprune_order(vision, k)
    _check_indices(idx, n, k)
    assert torch.equal(idx, mod.stable_divprune_order(vision, k))


def test_stable_divprune_unique_with_duplicate_features():
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_divprune_v15c")
    vision = torch.ones(12, 8)  # all distances zero: worst case for repeats
    idx = mod.stable_divprune_order(vision, 12)
    _check_indices(idx, 12, 12)


def test_stable_divprune_prefers_outlier():
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_divprune_v15c")
    vision = torch.tensor([[1.0, 0.0]] * 9 + [[0.0, 1.0]])
    assert 9 in mod.stable_divprune_order(vision, 2).tolist()


@pytest.mark.parametrize("bad_k", [0, 11, 2.0])
def test_stable_divprune_rejects_bad_count(bad_k):
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_divprune_v15c")
    with pytest.raises(ValueError):
        mod.stable_divprune_order(torch.randn(10, 4), bad_k)


@pytest.mark.parametrize("n,k", [(1, 1), (30, 3), (196, 78), (40, 40)])
def test_stable_cdpruner(n, k):
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_cdpruner")
    torch.manual_seed(n * 3 + k)
    vision, relevance = torch.randn(n, 16), torch.randn(n)
    idx = mod.stable_cdpruner_order(vision, relevance, k)
    _check_indices(idx, n, k)
    assert torch.equal(idx, mod.stable_cdpruner_order(vision, relevance, k))


def test_stable_cdpruner_degenerate_kernel_still_unique():
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_cdpruner")
    vision = torch.ones(16, 4)  # rank-one kernel: all but one gain is zero
    idx = mod.stable_cdpruner_order(vision, torch.zeros(16), 8)
    _check_indices(idx, 16, 8)
    # zero-gain ties are resolved by original patch order
    assert idx[1:].tolist() == sorted(idx[1:].tolist())


def test_stable_cdpruner_first_pick_is_most_relevant():
    mod = pytest.importorskip("learned_score.mlsys_sprint.stable_cdpruner")
    torch.manual_seed(9)
    vision = torch.randn(20, 8)
    relevance = torch.zeros(20)
    relevance[13] = 5.0
    assert int(mod.stable_cdpruner_order(vision, relevance, 3)[0]) == 13


# --------------------------------------------------------------------------- #
# Public asymvlm API (optional; skipped until the package provides it)
# --------------------------------------------------------------------------- #

def test_asymvlm_keep_count_is_ceil():
    pruning = pytest.importorskip("asymvlm.pruning")
    for n in (1, 7, 64, 576, 2880):
        for r in (0.1, 0.2, 0.25, 0.4, 0.7, 1.0):
            assert pruning.keep_count(n, r) == min(n, max(1, math.ceil(n * r)))
    with pytest.raises(ValueError):
        pruning.keep_count(10, 0.0)


@pytest.mark.parametrize("n,ratio", [(10, 0.4), (577, 0.1), (100, 0.25), (12, 1.0)])
def test_asymvlm_select_tokens(n, ratio):
    pruning = pytest.importorskip("asymvlm.pruning")
    from types import SimpleNamespace
    torch.manual_seed(1)
    hidden, text = 16, 5
    embeds = torch.randn(1, 3 + n + text, hidden)
    prepared = SimpleNamespace(
        inputs_embeds=embeds, visual_positions=torch.arange(3, 3 + n),
        semantic_text_positions=torch.arange(3 + n, 3 + n + text),
        answer_positions=torch.empty(0, dtype=torch.long))
    scorer = models.Rank32Scorer(hidden).eval()
    idx = pruning.select_tokens(prepared, ratio, scorer)
    k = min(n, max(1, math.ceil(n * ratio)))
    _check_indices(idx, n, k)
    scores = scorer(embeds[0, 3:3 + n], embeds[0, 3 + n:])
    assert torch.equal(idx, reference_topk(scores, ratio))
