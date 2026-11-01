"""Adaptive self-consistency gate (pure numpy reference + optional asymvlm.gate).

Gate: answer at a low keep ratio (10%) and at a second ratio; accept the 10%
answer iff the two normalized answers agree AND the minimum generated-token
probability is >= t; otherwise fall back to the higher keep ratio. ``t`` is
calibrated to minimize mean kept fraction subject to mean positive harm
(accuracy lost w.r.t. the fallback answer) <= eps, with 2-fold
cross-calibration (each fold is gated by a threshold tuned on the other fold).
"""

import re
import string

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Reference implementation
# --------------------------------------------------------------------------- #

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = str.maketrans({c: " " for c in string.punctuation})


def normalize_answer(text):
    text = str(text).lower().translate(_PUNCT)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def accept_low(answer_low, answer_check, min_prob, threshold):
    return normalize_answer(answer_low) == normalize_answer(answer_check) and min_prob >= threshold


def gate_outcomes(agree, min_prob, threshold, score_low, score_high, keep_low=0.1, keep_high=0.4):
    accept = np.asarray(agree, bool) & (np.asarray(min_prob) >= threshold)
    kept = np.where(accept, keep_low, keep_high)
    harm = np.where(accept, np.maximum(0.0, np.asarray(score_high) - np.asarray(score_low)), 0.0)
    return accept, kept.mean(), harm.mean()


def calibrate_threshold(agree, min_prob, score_low, score_high, eps, keep_low=0.1, keep_high=0.4):
    """Return t minimizing mean kept s.t. mean positive harm <= eps (ties -> larger t)."""
    candidates = np.unique(np.concatenate([np.asarray(min_prob, float), [np.inf]]))
    best = None
    for t in candidates[::-1]:  # high -> low so ties keep the most conservative t
        _, kept, harm = gate_outcomes(agree, min_prob, t, score_low, score_high, keep_low, keep_high)
        if harm <= eps + 1e-12 and (best is None or kept < best[1] - 1e-12):
            best = (t, kept)
    return best[0]


def cross_calibrate(agree, min_prob, score_low, score_high, eps, folds, calibrate=calibrate_threshold):
    folds = np.asarray(folds)
    thresholds = np.empty(len(folds))
    for fold in np.unique(folds):
        train = folds != fold
        t = calibrate(np.asarray(agree)[train], np.asarray(min_prob)[train],
                      np.asarray(score_low)[train], np.asarray(score_high)[train], eps)
        thresholds[folds == fold] = t
    return thresholds


def normalized_entropy(scores):
    """Entropy of softmax(scores) divided by log N, in [0, 1]."""
    scores = np.asarray(scores, float)
    if scores.size <= 1:
        return 0.0
    p = np.exp(scores - scores.max())
    p /= p.sum()
    h = -(p * np.log(np.clip(p, 1e-300, None))).sum()
    return float(np.clip(h / np.log(scores.size), 0.0, 1.0))


# --------------------------------------------------------------------------- #
# Tests of the reference
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("a,b", [("The Cat.", "cat"), ("  New   York ", "new york"),
                                 ("an apple!", "apple"), ("Yes", "yes,")])
def test_normalize_answer_equal(a, b):
    assert normalize_answer(a) == normalize_answer(b)


@pytest.mark.parametrize("a,b", [("cat", "cats"), ("12", "13"), ("red car", "blue car")])
def test_normalize_answer_distinct(a, b):
    assert normalize_answer(a) != normalize_answer(b)


def test_accept_requires_agreement_and_confidence():
    assert accept_low("Stop", "stop.", 0.9, 0.5)
    assert not accept_low("stop", "go", 0.99, 0.5)          # disagree
    assert not accept_low("stop", "stop", 0.49, 0.5)        # low confidence
    assert accept_low("stop", "stop", 0.5, 0.5)             # boundary is inclusive


def _synthetic():
    # 8 questions. min_prob sorted for readability; harm (high - low) only on
    # low-confidence items. Item 5 disagrees and can never be accepted.
    min_prob = np.array([0.95, 0.9, 0.85, 0.8, 0.6, 0.55, 0.4, 0.3])
    agree = np.array([1, 1, 1, 1, 1, 0, 1, 1], bool)
    score_high = np.array([1, 1, 1, 1, 1, 1, 1, 1], float)
    score_low = np.array([1, 1, 1, 1, 1, 0, 0, 1], float)  # item 6 harmed if accepted
    return agree, min_prob, score_low, score_high


def test_calibration_known_answer():
    agree, p, lo, hi = _synthetic()
    # eps=0: must exclude item 6 (p=0.4) -> best t = 0.6 (accept items 0-4).
    t = calibrate_threshold(agree, p, lo, hi, eps=0.0)
    assert t == pytest.approx(0.6)
    accept, kept, harm = gate_outcomes(agree, p, t, lo, hi)
    assert accept.tolist() == [1, 1, 1, 1, 1, 0, 0, 0]
    assert harm == 0.0
    assert kept == pytest.approx((5 * 0.1 + 3 * 0.4) / 8)
    # eps=1/8 tolerates item 6 -> t drops to 0.3 and accepts everything that agrees.
    t = calibrate_threshold(agree, p, lo, hi, eps=1 / 8)
    assert t == pytest.approx(0.3)
    accept, kept, harm = gate_outcomes(agree, p, t, lo, hi)
    assert accept.sum() == 7 and harm == pytest.approx(1 / 8)


def test_calibration_is_feasible_and_optimal_by_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(25):
        n = 30
        agree, p = rng.random(n) < 0.7, rng.random(n)
        hi = (rng.random(n) < 0.8).astype(float)
        lo = hi * (rng.random(n) < 0.5 + 0.5 * p)
        eps = rng.choice([0.0, 0.02, 0.05])
        t = calibrate_threshold(agree, p, lo, hi, eps)
        _, kept, harm = gate_outcomes(agree, p, t, lo, hi)
        assert harm <= eps + 1e-12
        for cand in np.concatenate([p, [np.inf]]):
            _, k2, h2 = gate_outcomes(agree, p, cand, lo, hi)
            if h2 <= eps + 1e-12:
                assert kept <= k2 + 1e-12


def test_infinite_threshold_means_never_accept():
    agree, p, lo, hi = _synthetic()
    accept, kept, harm = gate_outcomes(agree, p, np.inf, lo, hi)
    assert not accept.any() and kept == pytest.approx(0.4) and harm == 0


def test_cross_calibration_never_tunes_on_own_fold():
    agree, p, lo, hi = _synthetic()
    agree, p, lo, hi = (np.tile(x, 2) for x in (agree, p, lo, hi))
    folds = np.repeat([0, 1], 8)
    ids = np.arange(16)
    seen = []

    def spy(a, mp, sl, sh, eps):
        # identify the rows passed in by matching on min_prob+position bookkeeping
        seen.append(len(a))
        return calibrate_threshold(a, mp, sl, sh, eps)

    # Make each fold's rows identifiable through a unique offset on min_prob.
    p_id = p + ids * 1e-6
    calls = []

    def recording(a, mp, sl, sh, eps):
        calls.append(set(np.round((mp - np.round(mp, 2)) * 1e6).astype(int)))
        return spy(a, mp, sl, sh, eps)

    thresholds = cross_calibrate(agree, p_id, lo, hi, 0.0, folds, calibrate=recording)
    assert len(calls) == 2 and seen == [8, 8]
    fold_rows = [set(ids[folds == f]) for f in (0, 1)]
    # Call i tunes fold i: it must have seen exactly the other fold's rows.
    assert calls[0] == fold_rows[1] and calls[1] == fold_rows[0]
    assert calls[0].isdisjoint(fold_rows[0]) and calls[1].isdisjoint(fold_rows[1])
    assert thresholds.shape == (16,)
    assert len(set(thresholds[folds == 0])) == 1 and len(set(thresholds[folds == 1])) == 1


def test_cross_calibration_threshold_depends_only_on_other_fold():
    agree, p, lo, hi = _synthetic()
    folds = np.array([0, 1] * 4)
    base = cross_calibrate(agree, p, lo, hi, 0.0, folds)
    # Perturb fold-0 outcomes: fold-0 thresholds must be unchanged.
    lo2 = lo.copy()
    lo2[folds == 0] = 0.0
    changed = cross_calibrate(agree, p, lo2, hi, 0.0, folds)
    np.testing.assert_array_equal(base[folds == 0], changed[folds == 0])


def test_normalized_entropy_bounds():
    assert normalized_entropy(np.zeros(576)) == pytest.approx(1.0)
    peaked = np.zeros(576)
    peaked[3] = 100.0
    assert normalized_entropy(peaked) < 1e-3
    rng = np.random.default_rng(1)
    for _ in range(20):
        v = normalized_entropy(rng.normal(size=rng.integers(2, 500)) * rng.uniform(0.1, 10))
        assert 0.0 <= v <= 1.0
    assert normalized_entropy([1.0]) == 0.0


# --------------------------------------------------------------------------- #
# asymvlm.gate (optional): checked against the reference when present
# --------------------------------------------------------------------------- #

def _gate():
    return pytest.importorskip("asymvlm.gate")


@pytest.mark.parametrize("a,b", [("Stop.", "stop"), ("  New   York ", "new york"), ("Yes", "yes,"),
                                 ("3:45", "3 45")])
def test_asymvlm_normalize_answer(a, b):
    gate = _gate()
    assert gate.normalize_answer(a) == gate.normalize_answer(b)
    assert gate.normalize_answer("cat") != gate.normalize_answer("dog")
    assert gate.normalize_answer(None) == ""


def test_asymvlm_min_answer_confidence_excludes_eos_step():
    gate = _gate()
    assert gate.min_answer_confidence([0.9, 0.7, 0.01]) == 0.7
    assert gate.min_answer_confidence([0.4]) == 0.4


def test_asymvlm_gate_decision():
    gate = _gate()
    assert gate.gate_decision("Stop", 0.9, "stop.", 0.8, 0.5)
    assert not gate.gate_decision("stop", 0.99, "go", 0.99, 0.5)      # disagree
    assert not gate.gate_decision("stop", 0.9, "stop", 0.49, 0.5)     # either probe below t
    assert not gate.gate_decision("stop", 0.49, "stop", 0.9, 0.5)
    assert gate.gate_decision("stop", 0.5, "stop", 0.5, 0.5)          # inclusive


def test_asymvlm_calibration_known_answer():
    gate = _gate()
    agree, p, lo, hi = _synthetic()
    quality = np.stack([lo, lo, hi], 1)  # levels (10, 20, 40); cap=40
    t, kept = gate.calibrate_threshold(quality, agree, p, eps=0.0)
    assert 0.4 < t <= 0.6
    assert kept == pytest.approx((5 * 10 + 3 * 40) / 8)
    accept, _, harm = gate_outcomes(agree, p, t, lo, hi)
    assert harm == 0.0 and accept.sum() == 5
    t, kept = gate.calibrate_threshold(quality, agree, p, eps=100 / 8)  # eps in pp
    assert t <= 0.3
    assert kept == pytest.approx((7 * 10 + 40) / 8)


def test_asymvlm_calibration_matches_reference_on_random():
    gate = _gate()
    rng = np.random.default_rng(3)
    grid = np.linspace(0, 1, 401)
    for _ in range(10):
        n = 40
        agree, p = rng.random(n) < 0.7, np.round(rng.random(n), 3)
        hi = (rng.random(n) < 0.8).astype(float)
        lo = hi * (rng.random(n) < 0.6)
        quality = np.stack([lo, lo, hi], 1)
        t, kept = gate.calibrate_threshold(quality, agree, p, eps=2.0, grid=grid)
        _, kept_ref, harm = gate_outcomes(agree, p, t, lo, hi, 10, 40)
        assert kept == pytest.approx(kept_ref) and 100 * harm <= 2.0 + 1e-9
        for cand in grid:
            _, k2, h2 = gate_outcomes(agree, p, cand, lo, hi, 10, 40)
            if 100 * h2 <= 2.0:
                assert kept <= k2 + 1e-9


def test_asymvlm_calibration_infeasible():
    gate = _gate()
    quality = np.array([[0.0, 0.0, 1.0]])
    # t = 1.0 still accepts conf==1.0 rows; with all conf at 1.0 nothing is feasible at eps < 100.
    assert gate.calibrate_threshold(quality, [True], [1.0], eps=0.0) == (None, None)


def test_asymvlm_scorer_entropy():
    gate = _gate()
    assert gate.scorer_entropy(np.zeros(100)) == pytest.approx(1.0)
    peaked = np.zeros(576)
    peaked[0] = 1.0
    assert gate.scorer_entropy(peaked) < 1e-3
    rng = np.random.default_rng(0)
    for _ in range(20):
        v = gate.scorer_entropy(rng.normal(size=int(rng.integers(2, 600))))
        assert 0.0 <= v <= 1.0
    s = rng.normal(size=50)
    assert gate.scorer_entropy(s) == pytest.approx(gate.scorer_entropy(3 * s + 7))  # affine invariant
    assert gate.scorer_entropy([1.0]) == 0.0


def test_asymvlm_entropy_rule():
    gate = _gate()
    cal = np.linspace(0.2, 0.9, 101)
    assert gate.entropy_percentile(0.0, cal) == 0.0 and gate.entropy_percentile(1.0, cal) == 1.0
    peaked = np.zeros(576)
    peaked[0] = 1.0
    assert gate.entropy_rule(peaked, cal, 0.5) == 10
    assert gate.entropy_rule(np.zeros(576), cal, 0.5) == 40
