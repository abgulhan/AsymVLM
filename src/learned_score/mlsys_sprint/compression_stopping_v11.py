"""Answer-free, selector-independent stopping statistics for v11.

Selectors decide *which* patches to keep; these curves measure remaining score
mass and geometric coverage, not a probability that an answer is correct.  A
threshold must be calibrated using separate physical quality measurements.

The public API deliberately accepts only tensors, counts and policy parameters,
never answers.  Original patch order is restored only by ``selected_positions``.
Coverage uses at most N-by-64 similarities, not a stored N-by-N matrix.  The
optional full-count endpoint is exact and needs no full ordering/computation.
"""

import hashlib
import json
import math


FORMAT = "mlsys_selector_stopping_curve_v11"
SELECTORS = ("asymvlm", "divprune", "cdpruner")
STATISTICS = ("missing_mass", "uncovered_mass", "removed_fraction")
MAX_BLOCK_SIZE = 64


def _digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _block_size(value):
    if type(value) is not int or not 1 <= value <= MAX_BLOCK_SIZE:
        raise ValueError("block_size must be an integer from 1 to 64")
    return value


def _visual(value):
    import torch
    if not isinstance(value, torch.Tensor) or value.ndim != 2 or not all(value.shape):
        raise ValueError("nonempty N-by-D visual tensor required")
    value = value.detach().float()
    if not bool(torch.isfinite(value).all()):
        raise ValueError("visual features must be finite")
    norms = value.norm(dim=-1)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("visual norms must be finite and nonzero")
    return value / norms[:, None]


def _scores(value, n=None, device=None):
    import torch
    if not isinstance(value, torch.Tensor) or value.ndim != 1 or not value.numel():
        raise ValueError("nonempty one-dimensional scores required")
    if n is not None and len(value) != n:
        raise ValueError("scores and patches differ in length")
    value = value.detach().to(device=device, dtype=torch.float32)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("scores must be finite")
    return value


def _count(value, n):
    if type(value) is not int or not 1 <= value <= n:
        raise ValueError("retained count must be an integer from 1 to N")
    return value


def learned_score_weights(scores):
    """Softmax of population-zscored learned scores; not accuracy confidence.

    A flat score vector gives uniform weights.  FP64 standardization prevents
    overflow for otherwise finite FP32 scores; the final probability vector is
    also FP64 to make conservation independent of the similarity kernel dtype.
    """
    import torch
    values = _scores(scores).double()
    standardized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    return torch.softmax(standardized, dim=0)


def selector_order(selector, *, visual=None, scores=None, relevance=None,
                   keep_count=None, block_size=MAX_BLOCK_SIZE):
    """Return a unique, deterministic *selection-order* prefix.

    ``asymvlm`` sorts learned scores descending, breaking ties by original index.
    ``divprune`` preserves the max-min rule but explicitly excludes chosen
    indices and handles N=1.  Its initial nearest-other distance is computed in
    blocks; neither initialization nor greedy updates retain an N-by-N matrix.
    ``cdpruner`` calls the existing versioned stable conditional-DPP implementation;
    its internal DPP kernel still uses N-by-N memory.  Passing CD relevance does
    not certify that a model's relevance/features match the official pipeline.
    """
    import torch
    _block_size(block_size)
    if selector not in SELECTORS:
        raise ValueError("unsupported selector")
    with torch.no_grad():
        if selector == "asymvlm":
            values = _scores(scores)
            k = _count(len(values) if keep_count is None else keep_count, len(values))
            return torch.argsort(values, descending=True, stable=True)[:k]
        normalized = _visual(visual)
        n = len(normalized)
        k = _count(n if keep_count is None else keep_count, n)
        if selector == "cdpruner":
            from .stable_cdpruner import stable_cdpruner_order
            return stable_cdpruner_order(visual, _scores(relevance, n, visual.device), k)
        if n == 1:
            return torch.zeros(1, dtype=torch.long, device=normalized.device)
        # Match the historical second-smallest-distance initialization, including
        # the self distance; later selections explicitly mask already chosen rows.
        nearest_two = torch.full((2, n), float("inf"), device=normalized.device)
        for start in range(0, n, block_size):
            distances = 1.0 - normalized[start:start + block_size] @ normalized.T
            nearest_two = torch.cat((nearest_two, distances), dim=0).topk(
                2, dim=0, largest=False).values
        priority = nearest_two[1]
        available = torch.ones(n, dtype=torch.bool, device=normalized.device)
        order = torch.empty(k, dtype=torch.long, device=normalized.device)
        nearest_selected = torch.full((n,), float("inf"), device=normalized.device)
        for index in range(k):
            chosen = priority.masked_fill(~available, -float("inf")).argmax()
            order[index] = chosen
            available[chosen] = False
            if index + 1 < k:
                distance = 1.0 - (normalized @ normalized[chosen])
                nearest_selected = torch.minimum(nearest_selected, distance)
                priority = nearest_selected
        return order


def selected_positions(order, count, *, input_visual_tokens=None):
    """Validate a prefix and restore original image-token order for inference."""
    import torch
    n = len(order) if input_visual_tokens is None else input_visual_tokens
    _count(count, n)
    order = _order(order, n)
    if count == n:
        return torch.arange(n, device=order.device)
    if len(order) < count:
        raise ValueError("selection-order prefix is shorter than requested count")
    return order[:count].sort().values


def _order(order, n, device=None):
    import torch
    if isinstance(order, (list, tuple)) and not order:
        order = torch.empty(0, dtype=torch.long, device=device)
    order = torch.as_tensor(order, device=device)
    if order.ndim != 1 or order.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64,
                                             torch.uint8):
        raise ValueError("selection order must contain one-dimensional integer indices")
    order = order.detach().long()
    if (len(order) > n or bool(((order < 0) | (order >= n)).any())
            or len(order.unique()) != len(order)):
        raise ValueError("selection order must contain unique in-range patch indices")
    return order


def _weights(n, device, *, scores, weights, weight_mode, selector):
    import torch
    if selector == "divprune" and weight_mode != "uniform":
        raise ValueError("native DivPrune stopping uses uniform weights, not distance gains")
    if weight_mode == "uniform":
        if weights is not None:
            raise ValueError("explicit weights require weight_mode='explicit'")
        return torch.full((n,), 1. / n, dtype=torch.float64, device=device)
    if weight_mode == "learned_scores":
        if weights is not None:
            raise ValueError("learned-score weighting cannot also receive explicit weights")
        return learned_score_weights(_scores(scores, n, device))
    if weight_mode != "explicit" or weights is None:
        raise ValueError("weight_mode must be uniform, learned_scores, or explicit")
    result = torch.as_tensor(weights, device=device, dtype=torch.float64).detach()
    if (result.shape != (n,) or not bool(torch.isfinite(result).all())
            or bool((result < 0).any()) or not float(result.sum()) > 0
            or not math.isfinite(float(result.sum()))):
        raise ValueError("explicit weights must be finite, nonnegative, and have positive mass")
    return result / result.sum()


def stopping_curve(*, visual, order, counts, scores=None, weights=None,
                   weight_mode="uniform", block_size=MAX_BLOCK_SIZE,
                   selector="external", geometry="unspecified"):
    """Compute answer-free missing mass M(k) and uncovered mass U(k).

    M(k) = sum of weights of unselected patches.
    U(k) = sum_i p_i * (1 - max_{j in selected} max(0, cosine(i,j))).

    Each chosen patch covers itself exactly.  Therefore 0 <= U <= M <= 1,
    both curves are nonincreasing and their full-count endpoints are exactly 0.
    ``counts`` must be strictly increasing positive integer counts.  ``order``
    need only cover the largest *non-full* requested count: N itself is a full
    retention sentinel with no additional similarity work or ordering needed.
    A named geometry and fingerprints prevent accidental replay with different
    weights/order; caller metadata must additionally bind the actual embeddings.
    """
    import torch
    _block_size(block_size)
    with torch.no_grad():
        visual = _visual(visual)
        n = len(visual)
        order = _order(order, n, visual.device)
        counts = list(counts)
        if not counts or any(_count(c, n) != c for c in counts) or counts != sorted(set(counts)):
            raise ValueError("counts must be nonempty and strictly increasing")
        nonfull = [c for c in counts if c < n]
        if nonfull and len(order) < max(nonfull):
            raise ValueError("selection-order prefix does not cover requested non-full counts")
        if not isinstance(selector, str) or not selector or not isinstance(geometry, str) or not geometry:
            raise ValueError("selector and geometry identifiers must be nonempty strings")
        probability = _weights(n, visual.device, scores=scores, weights=weights,
                               weight_mode=weight_mode, selector=selector)
        coverage = torch.zeros(n, dtype=torch.float32, device=visual.device)
        selected = torch.zeros(n, dtype=torch.bool, device=visual.device)
        missing, uncovered, processed = [], [], 0
        for count in counts:
            if count == n:
                missing.append(0.)
                uncovered.append(0.)
                continue
            for start in range(processed, count, block_size):
                chosen = order[start:min(start + block_size, count)]
                similarity = (visual @ visual[chosen].T).clamp_(0., 1.)
                coverage = torch.maximum(coverage, similarity.max(dim=1).values)
                selected[chosen] = True
                coverage[selected] = 1.
            processed = count
            m = float(probability[~selected].sum())
            u = float((probability * (1. - coverage).double()).sum())
            missing.append(min(1., max(0., m)))
            uncovered.append(min(missing[-1], max(0., u)))
        return {"format": FORMAT, "input_visual_tokens": n, "counts": counts,
                "missing_mass": missing, "uncovered_mass": uncovered,
                "removed_fraction": [1. - c / n for c in counts],
                "selector": selector, "geometry": geometry, "weight_mode": weight_mode,
                "weights_sha256": _digest(probability.cpu().tolist()),
                "order_prefix_sha256": _digest(order.cpu().tolist()),
                "order_prefix_length": len(order), "block_size": block_size,
                "invalid": False, "fallback": False, "fallback_reason": None,
                "answers_consumed": False,
                "risk_interpretation": "answer_free_proxy_not_calibrated_accuracy_probability"}


def full_fallback_curve(input_visual_tokens, reason, *, selector="external",
                        geometry="unspecified", weight_mode="uniform"):
    """Explicit full-retention record for invalid geometry or selector parity."""
    if type(input_visual_tokens) is not int or input_visual_tokens < 1:
        raise ValueError("fallback requires a trustworthy positive input patch count")
    if not isinstance(reason, str) or not reason:
        raise ValueError("fallback requires a nonempty reason")
    n = input_visual_tokens
    return {"format": FORMAT, "input_visual_tokens": n, "counts": [n],
            "missing_mass": [0.], "uncovered_mass": [0.], "removed_fraction": [0.],
            "selector": selector, "geometry": geometry, "weight_mode": weight_mode,
            "invalid": True, "fallback": True, "fallback_reason": reason,
            "answers_consumed": False,
            "risk_interpretation": "invalid_input_full_retention"}


def build_stopping_curve(*, fallback_on_invalid=True, **kwargs):
    """Compute a curve or explicitly record invalid-input full retention.

    Runtime/hardware exceptions are not swallowed.  A malformed visual shape
    with no trustworthy positive input count cannot be converted to a fallback.
    """
    try:
        return stopping_curve(**kwargs)
    except (ValueError, FloatingPointError) as error:
        if not fallback_on_invalid:
            raise
        visual = kwargs.get("visual")
        if getattr(visual, "ndim", None) != 2 or not len(visual):
            raise
        return full_fallback_curve(len(visual), str(error),
            selector=kwargs.get("selector", "external"),
            geometry=kwargs.get("geometry", "unspecified"),
            weight_mode=kwargs.get("weight_mode", "uniform"))


def _validate_curve(curve):
    if curve.get("format") != FORMAT:
        raise ValueError("unknown stopping curve schema")
    n = curve.get("input_visual_tokens")
    if type(n) is not int or n < 1:
        raise ValueError("curve must declare a positive input visual count")
    counts = curve.get("counts", [])
    if not isinstance(counts, list) or not counts or counts != sorted(set(counts)):
        raise ValueError("curve counts must be a strictly increasing list")
    for count in counts:
        _count(count, n)
    for name in STATISTICS:
        values = curve.get(name)
        if not isinstance(values, list) or len(values) != len(counts):
            raise ValueError("curve statistic length differs from counts")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not 0 <= v <= 1 for v in values):
            raise ValueError("curve statistics must be finite values between zero and one")
        if any(right > left + 1e-12 for left, right in zip(values, values[1:])):
            raise ValueError("curve statistics must be nonincreasing")
        if counts[-1] == n and values[-1] != 0:
            raise ValueError("full-count risk must equal zero")
    if any(u > m + 1e-12 for u, m in zip(curve["uncovered_mass"], curve["missing_mass"])):
        raise ValueError("uncovered mass cannot exceed missing mass")
    if any(abs(r - (1. - count / n)) > 1e-12
           for r, count in zip(curve["removed_fraction"], counts)):
        raise ValueError("removed fraction differs from exact count")
    return n


def select_threshold(curve, threshold, *, statistic="uncovered_mass", minimum_count=1):
    """Choose the smallest supported count whose proxy is at most threshold.

    No interpolation or rounding to an unmeasured count is performed.  Invalid
    curves or unavailable support produce an explicit full-retention fallback.
    A valid full endpoint selected by the rule is not called an invalid fallback.
    """
    n = _validate_curve(curve)
    _count(minimum_count, n)
    if statistic not in STATISTICS:
        raise ValueError("unsupported stopping statistic")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be finite and nonnegative")
    if curve.get("invalid") or curve.get("fallback"):
        return {"count": n, "fallback": True,
                "reason": curve.get("fallback_reason") or "invalid_curve", "statistic": statistic}
    for count, value in zip(curve["counts"], curve[statistic]):
        if count >= minimum_count and value <= threshold:
            return {"count": count, "fallback": False, "reason": "threshold_satisfied",
                    "statistic": statistic, "risk": value}
    return {"count": n, "fallback": True, "reason": "no_supported_count_satisfies_threshold",
            "statistic": statistic}


def nonnegative_risk(curve, coefficients):
    """Tiny no-intercept monotone linear risk, with separately fitted weights.

    Only M, U and removed fraction may be inputs.  Coefficients must be finite
    and nonnegative; they need not sum to one.  Uniform-weight M is identical to
    removed fraction, so including both is rejected to avoid a duplicate cue.
    This helper does no fitting and makes no calibrated-probability claim.
    """
    _validate_curve(curve)
    if not isinstance(coefficients, dict) or not coefficients or set(coefficients) - set(STATISTICS):
        raise ValueError("coefficients must name one or more supported risk inputs")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
           for v in coefficients.values()):
        raise ValueError("risk coefficients must be finite and nonnegative")
    if (curve.get("weight_mode") == "uniform" and coefficients.get("missing_mass", 0) > 0
            and coefficients.get("removed_fraction", 0) > 0):
        raise ValueError("uniform missing mass duplicates removed fraction; use only one")
    result = [sum(coefficients[name] * curve[name][index] for name in coefficients)
              for index in range(len(curve["counts"]))]
    if any(not math.isfinite(value) for value in result):
        raise ValueError("linear risk overflow")
    return result
