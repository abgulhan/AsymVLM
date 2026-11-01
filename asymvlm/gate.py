"""Adaptive per-request budgets.

1. Self-consistency gate (two cheap probes, escalate on disagreement/low confidence):
     run AsymVLM at 10% and 20% kept visual tokens; if the normalized answers agree
     AND, for both probes, the minimum top-1 probability over answer tokens
     (excluding the final EOS step) is >= t, output the 10% answer; otherwise run
     at the cap K (default 40%). Thresholds below were calibrated at cap 40 for
     <= 1 pp mean harm vs. uniform-40.
2. ``calibrate_threshold``: pure-numpy threshold search (min mean kept fraction
   subject to mean positive harm vs. uniform-cap <= eps).
3. Training-free single-pass rule on scorer entropy.

Reference settings for the gate passes: greedy, max_new_tokens 128,
generated_budget 128, slack 1.
"""
import math
import re
from contextlib import contextmanager

import numpy as np

from .config import GATE_CAP, GATE_PROBES, GATE_THRESHOLDS

GATE_PASS_KWARGS = dict(max_new_tokens=128, generated_budget=128, slack=1, stop_on_eos=True)


def normalize_answer(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def min_answer_confidence(step_top1_probs):
    """Min top-1 probability over answer tokens, excluding the final (EOS) step."""
    p = list(step_top1_probs)
    ans = p[:-1] if len(p) > 1 else p
    return min(ans or [1.0])


def gate_decision(answer_low, conf_low, answer_mid, conf_mid, threshold):
    """True -> accept the low-budget (10%) answer; False -> escalate to the cap."""
    return normalize_answer(answer_low) == normalize_answer(answer_mid) and min(conf_low, conf_mid) >= threshold


@contextmanager
def record_top1_probs(limit=64):
    """Patch the decoder forward used by ``generate_bounded`` to record per-step top-1 probs."""
    import torch
    from learned_score.mlsys_runtime import generation
    original = generation.decoder_forward_function
    probs = []

    def patched(model):
        forward = original(model)

        def wrapped(**kwargs):
            out = forward(**kwargs)
            if len(probs) < limit:
                probs.append(float(torch.softmax(out.logits[0, -1].float(), -1).max()))
            return out
        return wrapped
    generation.decoder_forward_function = patched
    try:
        yield probs
    finally:
        generation.decoder_forward_function = original


def gate_pass(runtime, prepared, percent, **kwargs):
    """One AsymVLM pass at ``percent`` kept; returns (answer, min answer confidence)."""
    from .pruning import prune_and_generate
    with record_top1_probs() as probs:
        text, result = prune_and_generate(runtime, prepared, percent / 100.0, **{**GATE_PASS_KWARGS, **kwargs})
        p = list(probs[:result.ids.shape[1]])
    return text, min_answer_confidence(p)


def self_consistency_gate(runtime, prepared, threshold=None, cap=GATE_CAP, probes=GATE_PROBES, **kwargs):
    """Return ``(answer, kept_percent)`` for one prepared request."""
    if threshold is None:
        threshold = GATE_THRESHOLDS[runtime.spec.key]
    low, mid = probes
    a_low, c_low = gate_pass(runtime, prepared, low, **kwargs)
    a_mid, c_mid = gate_pass(runtime, prepared, mid, **kwargs)
    if gate_decision(a_low, c_low, a_mid, c_mid, threshold):
        return a_low, low
    a_cap, _ = gate_pass(runtime, prepared, cap, **kwargs)
    return a_cap, cap


def calibrate_threshold(quality, agree, conf, levels=(10, 20, 40), cap=40, eps=1.0, grid=None):
    """Minimum-mean-kept threshold subject to mean positive harm vs. uniform-cap <= eps (pp).

    quality: [R, L] per-request score (0..1) at each level in ``levels``.
    agree:   [R] bool, normalized 10% and 20% answers agree.
    conf:    [R] min over both probes of the min answer-token top-1 prob.
    Returns ``(threshold, mean_kept_percent)`` or ``(None, None)`` if infeasible.
    """
    quality = np.asarray(quality, float); agree = np.asarray(agree, bool); conf = np.asarray(conf, float)
    lo, hi = levels.index(GATE_PROBES[0]), levels.index(cap)
    grid = np.linspace(0, 1, 401) if grid is None else grid
    best = None
    for t in grid:
        final = np.where(agree & (conf >= t), lo, hi)
        q = quality[np.arange(len(final)), final]
        kept = float(np.mean([levels[i] for i in final]))
        harm = 100 * np.maximum(0, quality[:, hi] - q).mean()
        if harm <= eps and (best is None or kept < best[1]):
            best = (float(t), kept)
    return best if best is not None else (None, None)


# ---- Training-free single-pass rule (scorer entropy) ------------------------------

def scorer_entropy(scores):
    """Normalized entropy of softmax(scores / std(scores)), in [0, 1] (divided by log N)."""
    s = np.asarray(scores, float).ravel()
    n = s.size
    if n < 2:
        return 0.0
    std = s.std()
    z = (s - s.mean()) / (std if std > 0 else 1.0)
    p = np.exp(z - z.max()); p /= p.sum()
    return float(-(p * np.log(np.clip(p, 1e-30, None))).sum() / math.log(n))


def entropy_percentile(entropy, calibration_entropies):
    """Empirical percentile (0..1) of ``entropy`` within the calibration distribution."""
    cal = np.sort(np.asarray(calibration_entropies, float))
    return float(np.searchsorted(cal, entropy, side="right") / max(1, cal.size))


def entropy_rule(scores, calibration_entropies, t, low=GATE_PROBES[0], cap=GATE_CAP):
    """Keep ``low`` percent if calibration-percentile(entropy) <= t, else ``cap``."""
    return low if entropy_percentile(scorer_entropy(scores), calibration_entropies) <= t else cap
