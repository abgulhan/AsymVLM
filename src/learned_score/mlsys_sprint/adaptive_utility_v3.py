"""One-statistic allocation from measured, possibly nonmonotone benefits.

This version does not import or change any frozen v1/v2 experiment. Fit accepts
physical development qualities; inference accepts a strictly answer-free schema.
The caller derives the visual-token total from its verified allocated-cache
contract. All requests in one allocation must have the same cache bytes/slot.
Actual inference at the returned integer counts is required for quality claims.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import heapq
import json
import math
from typing import Mapping

import numpy as np


POLICY_FORMAT = "mlsys_adaptive_utility_policy_v3"
ASSIGNMENT_FORMAT = "mlsys_adaptive_utility_assignments_v3"
STATISTICS = ("normalized_entropy", "local_shell", "nearest_neighbor_diversity")
PHYSICAL_RETENTIONS = (0.10, 0.25, 0.40, 0.55, 0.70)
FORMULAS = {
    "normalized_entropy": "H(softmax((s-mean(s))/max(std_population(s),1e-8)))/log(N)",
    "local_shell": "top_mass(center+0.10)-top_mass(center-0.10); population-zscore softmax; floor(N*r)",
    "nearest_neighbor_diversity": "mean_i(1-max_{j!=i} cosine(projected_vision_i,projected_vision_j))",
}
_FIT_FIELDS = {"row_hash", "group_id", "dataset", "statistic_value", "quality"}
_INFERENCE_FIELDS = {"row_hash", "group_id", "statistic_value", "input_visual_tokens", "input_text_tokens"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sealed(value, field):
    result = json.loads(_canonical(value))
    result[field] = _digest(result)
    return result


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _name(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _integer(value, name, *, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def calculate_statistic(scores, statistic, *, center=None, vision=None, chunk_size=256):
    """Compute one answer-free statistic without mutating tensors or gradients.

    Diversity is evaluated in row chunks so it never materializes an N-by-N
    distance matrix. ``scores`` may be None for diversity; ``vision`` is required.
    Shell centers are numeric at extraction time; policy center metadata may be
    None when a target's uniform count determines the center for each cohort.
    """
    import torch
    import torch.nn.functional as F

    if statistic not in STATISTICS:
        raise ValueError("unknown allocation statistic")
    if statistic == "nearest_neighbor_diversity":
        _integer(chunk_size, "chunk_size", minimum=1)
        if vision is None:
            raise ValueError("diversity requires projected vision embeddings")
        values = torch.as_tensor(vision).detach()
        if values.ndim != 2 or not values.shape[0] or not values.shape[1] or values.is_complex() or values.dtype == torch.bool:
            raise ValueError("vision must be a nonempty real matrix")
        values = values.float()
        if not bool(torch.isfinite(values).all()):
            raise ValueError("vision contains nonfinite values")
        if len(values) == 1:
            return 0.0
        normalized = F.normalize(values, dim=-1)
        pieces = []
        for start in range(0, len(values), chunk_size):
            stop = min(len(values), start + chunk_size)
            similarity = normalized[start:stop] @ normalized.T
            local = torch.arange(stop - start, device=values.device)
            similarity[local, local + start] = -float("inf")
            pieces.append((1.0 - similarity.max(dim=1).values).clamp(0.0, 2.0).double().sum())
        return _finite(float(torch.stack(pieces).sum() / len(values)), "diversity")

    values = torch.as_tensor(scores).detach()
    if values.ndim != 1 or not len(values) or values.is_complex() or values.dtype == torch.bool:
        raise ValueError("scores must be a nonempty real vector")
    values = values.float()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores contain nonfinite values")
    standardized = (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)
    if not bool(torch.isfinite(standardized).all()):
        raise ValueError("standardized scores contain nonfinite values")
    probabilities = torch.softmax(standardized, dim=0)
    if statistic == "normalized_entropy":
        if len(values) == 1:
            return 0.0
        value = -(probabilities * probabilities.clamp_min(1e-12).log()).sum() / math.log(len(values))
        return min(1.0, max(0.0, _finite(float(value), "entropy")))
    center = _finite(center, "shell center")
    if not 0 <= center <= 1:
        raise ValueError("shell center must lie in [0,1]")
    low = int(math.floor(len(values) * max(0.0, center - 0.10) + 1e-10))
    high = int(math.floor(len(values) * min(1.0, center + 0.10) + 1e-10))
    ordered = probabilities.sort(descending=True, stable=True).values
    return _finite(float(ordered[low:high].sum()), "local shell")


def grouped_folds(rows):
    """Deterministic document/image folds, shared across statistics and models."""
    groups = sorted({_name(row["group_id"], "group_id") for row in rows},
                    key=lambda value: (_digest(["adaptive-utility-v3-fold", value]), value))
    if len(groups) < 5:
        raise ValueError("five-fold calibration requires five independent groups")
    return {group: index % 5 for index, group in enumerate(groups)}


def _fit_rows(rows):
    if not isinstance(rows, (tuple, list)) or not rows:
        raise ValueError("fit rows must be nonempty")
    checked, seen, group_datasets = [], set(), {}
    optional = {"input_visual_tokens", "input_text_tokens"}
    for row in rows:
        if not isinstance(row, Mapping) or not _FIT_FIELDS <= set(row) or set(row) - _FIT_FIELDS - optional:
            raise ValueError("fit rows must contain only the declared development fields")
        item = {name: _name(row[name], name) for name in ("row_hash", "group_id", "dataset")}
        if item["row_hash"] in seen:
            raise ValueError("duplicate fit row identity")
        seen.add(item["row_hash"])
        if group_datasets.setdefault(item["group_id"], item["dataset"]) != item["dataset"]:
            raise ValueError("a calibration group spans multiple dataset names")
        item["statistic_value"] = _finite(row["statistic_value"], "statistic_value")
        quality = row["quality"]
        if not isinstance(quality, Mapping):
            raise ValueError("physical quality must be a retention-keyed mapping")
        normalized = {}
        for key, value in quality.items():
            try:
                ratio = float(key)
            except (TypeError, ValueError) as error:
                raise ValueError("invalid physical retention key") from error
            if ratio not in PHYSICAL_RETENTIONS or ratio in normalized:
                raise ValueError("exactly the five v3 physical knots are required")
            score = _finite(value, "physical quality")
            if not 0 <= score <= 1:
                raise ValueError("physical quality must lie in [0,1]")
            normalized[ratio] = score
        if set(normalized) != set(PHYSICAL_RETENTIONS):
            raise ValueError("missing physical quality knots")
        item["quality"] = [normalized[ratio] for ratio in PHYSICAL_RETENTIONS]
        checked.append(item)
    return sorted(checked, key=lambda row: row["row_hash"])


def _groups_in(rows, indices):
    return len({rows[index]["group_id"] for index in indices})


def _fit_member(rows, min_groups, shrinkage):
    if len({row["group_id"] for row in rows}) < min_groups:
        raise ValueError("each training fold needs at least min_groups independent groups")
    x = np.asarray([row["statistic_value"] for row in rows], dtype=np.float64)
    y = np.asarray([row["quality"] for row in rows], dtype=np.float64)
    dataset_counts = Counter(row["dataset"] for row in rows)
    weights = np.asarray([len(rows) / (len(dataset_counts) * dataset_counts[row["dataset"]]) for row in rows])
    prior = np.average(y, axis=0, weights=weights)
    order = np.argsort(x, kind="stable")
    # Equal-scalar observations never straddle bins. Small neighboring bins are
    # merged, rather than weakening the minimum independent-group requirement.
    bins = [part.tolist() for part in np.array_split(order, min(5, len(order))) if len(part)]
    index = 0
    while index < len(bins) - 1:
        if x[bins[index][-1]] == x[bins[index + 1][0]]:
            bins[index:index + 2] = [bins[index] + bins[index + 1]]
        else:
            index += 1
    while len(bins) > 1:
        small = next((index for index, part in enumerate(bins) if _groups_in(rows, part) < min_groups), None)
        if small is None:
            break
        neighbor = 1 if small == 0 else small - 1
        if 0 < small < len(bins) - 1 and len(bins[small + 1]) < len(bins[small - 1]):
            neighbor = small + 1
        low, high = sorted((small, neighbor))
        bins[low:high + 1] = [sum(bins[low:high + 1], [])]
    centers, curves, counts, group_counts = [], [], [], []
    for part in bins:
        weight = weights[part]
        centers.append(float(np.average(x[part], weights=weight)))
        curves.append(np.clip(((y[part] * weight[:, None]).sum(axis=0) + shrinkage * prior) / (weight.sum() + shrinkage), 0., 1.))
        counts.append(len(part))
        group_counts.append(_groups_in(rows, part))
    return {"centers": centers, "curves": np.asarray(curves).tolist(), "counts": counts,
            "group_counts": group_counts, "prior_curve": prior.tolist(),
            "training_groups": len({row["group_id"] for row in rows}),
            "dataset_counts": dict(sorted(dataset_counts.items()))}


def fit_policy(fit_rows, *, statistic, center=None, identity=None, min_groups=30, shrinkage=30.0):
    """Fit five group-exclusion members; this is the only quality-bearing API.

    Curves are never isotonic-projected. ``center=None`` means the scalar was
    extracted around its cohort's answer-free uniform retention. Identity should
    bind model/scorer/manifest, target reduction and that center convention.
    Numeric calibration is model-specific, not an inference dataset input.
    """
    if statistic not in STATISTICS:
        raise ValueError("unknown allocation statistic")
    if center is not None:
        center = _finite(center, "center")
        if not 0 <= center <= 1:
            raise ValueError("center must lie in [0,1]")
    _integer(min_groups, "min_groups", minimum=1)
    shrinkage = _finite(shrinkage, "shrinkage")
    if shrinkage < 0:
        raise ValueError("shrinkage must be nonnegative")
    if identity is None:
        identity = {}
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a JSON object")
    identity = json.loads(_canonical(dict(identity)))
    rows = _fit_rows(fit_rows)
    folds = grouped_folds(rows)
    members = []
    for fold in range(5):
        subset = [row for row in rows if folds[row["group_id"]] != fold]
        member = _fit_member(subset, min_groups, shrinkage)
        member["excluded_fold"] = fold
        members.append(member)
    values = [row["statistic_value"] for row in rows]
    payload = {"format": POLICY_FORMAT, "statistic": statistic, "statistic_formula": FORMULAS[statistic],
               "center": center, "center_rule": "numeric_center" if center is not None else "target_uniform_band",
               "identity": identity, "physical_retentions": list(PHYSICAL_RETENTIONS),
               "risk_penalty": 0.5, "allocation_radius": 0.10, "retention_step": 0.01,
               "min_groups_per_bin": min_groups, "pseudo_observations_per_bin": shrinkage,
               "curve_constraint": "none; observed nonmonotone quality is preserved",
               "fit_weighting": "equal total weight per dataset; dataset is not a prediction input",
               "support": [min(values), max(values)], "members": members,
               "calibration_rows": [row["row_hash"] for row in rows],
               "calibration_groups": sorted(folds), "group_folds": folds,
               "calibration_data_sha256": _digest(rows), "confirmation_labels_read": False}
    return _sealed(payload, "policy_sha256")


def load_policy(payload):
    """Validate a JSON string/object and return a detached JSON-compatible copy."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("policy must be a JSON object")
    result = json.loads(_canonical(dict(payload)))
    claimed = result.pop("policy_sha256", None)
    if claimed != _digest(result):
        raise ValueError("policy digest mismatch")
    if (result.get("format") != POLICY_FORMAT or result.get("statistic") not in STATISTICS
            or result.get("statistic_formula") != FORMULAS[result["statistic"]]
            or result.get("physical_retentions") != list(PHYSICAL_RETENTIONS)
            or result.get("risk_penalty") != 0.5 or result.get("allocation_radius") != 0.10
            or result.get("retention_step") != 0.01):
        raise ValueError("unsupported policy contract")
    support = result.get("support")
    if not isinstance(support, list) or len(support) != 2 or not all(math.isfinite(v) for v in support) or support[0] > support[1]:
        raise ValueError("invalid policy support")
    members = result.get("members")
    if not isinstance(members, list) or len(members) != 5:
        raise ValueError("five ensemble members are required")
    for fold, member in enumerate(members):
        centers = np.asarray(member["centers"], dtype=float)
        curves = np.asarray(member["curves"], dtype=float)
        if (centers.ndim != 1 or not len(centers) or len(centers) > 5 or not np.isfinite(centers).all()
                or np.any(np.diff(centers) <= 0) or curves.shape != (len(centers), 5)
                or not np.isfinite(curves).all() or np.any(curves < 0) or np.any(curves > 1)
                or member["excluded_fold"] != fold
                or any(value < result["min_groups_per_bin"] for value in member["group_counts"])):
            raise ValueError("invalid member calibration")
    result["policy_sha256"] = claimed
    return result


def serialize_policy(policy):
    return _canonical(load_policy(policy))


def _curves_at(policy, values):
    return np.asarray([np.column_stack([np.interp(values, member["centers"], np.asarray(member["curves"])[:, column])
                                       for column in range(5)]) for member in policy["members"]])


def predict_utilities(policy, statistic_values, retentions):
    """Return mean quality, fold SD and support flags; never reads labels.

    Retentions must lie within the physical support. This convenience function
    is for diagnostic plots; allocation uses paired gains relative to uniform
    before calculating the uncertainty penalty.
    """
    policy = load_policy(policy)
    values = np.asarray(statistic_values, dtype=float)
    ratios = np.asarray(retentions, dtype=float)
    if values.ndim != 1 or ratios.ndim != 1 or not len(values) or not len(ratios) or not np.isfinite(values).all() or not np.isfinite(ratios).all():
        raise ValueError("finite nonempty vectors are required")
    if ratios.min() < PHYSICAL_RETENTIONS[0] or ratios.max() > PHYSICAL_RETENTIONS[-1]:
        raise ValueError("no extrapolation outside physical retention support")
    curves = _curves_at(policy, values)
    predictions = np.asarray([[np.interp(ratios, PHYSICAL_RETENTIONS, row) for row in member] for member in curves])
    outside = (values < policy["support"][0]) | (values > policy["support"][1])
    return predictions.mean(axis=0), predictions.std(axis=0), outside


def _inference_rows(rows, policy):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("assignment rows must be nonempty")
    checked, seen = [], set()
    fit_rows, fit_groups = set(policy["calibration_rows"]), set(policy["calibration_groups"])
    for row in rows:
        if not isinstance(row, Mapping) or not _INFERENCE_FIELDS <= set(row) or set(row) - _INFERENCE_FIELDS - {"dataset"}:
            raise ValueError("assignment rows must contain only declared answer-free fields")
        item = {name: _name(row[name], name) for name in ("row_hash", "group_id")}
        if item["row_hash"] in seen:
            raise ValueError("duplicate assignment identity")
        if item["row_hash"] in fit_rows or item["group_id"] in fit_groups:
            raise ValueError("assignment identity overlaps calibration")
        seen.add(item["row_hash"])
        item["statistic_value"] = _finite(row["statistic_value"], "statistic_value")
        item["input_visual_tokens"] = _integer(row["input_visual_tokens"], "input_visual_tokens", minimum=1)
        item["input_text_tokens"] = _integer(row["input_text_tokens"], "input_text_tokens")
        if "dataset" in row:
            item["dataset"] = _name(row["dataset"], "dataset")
        checked.append(item)
    return sorted(checked, key=lambda row: row["row_hash"])


def _uniform(rows, goal):
    total = sum(row["input_visual_tokens"] for row in rows)
    counts = [row["input_visual_tokens"] * goal // total for row in rows]
    order = sorted(range(len(rows)), key=lambda i: (-(rows[i]["input_visual_tokens"] * goal % total), rows[i]["row_hash"]))
    for index in order[:goal - sum(counts)]:
        counts[index] += 1
    return counts


def _physical_count(n, percent):
    return max(1, n * percent // 100)


def allocate(policy, answer_free_rows, *, target_visual_tokens, uniform_counts=None):
    """Allocate exact integer image counts; dataset names never affect choices.

    ``uniform_counts`` is an optional row-hash -> integer mapping from the
    caller's cache accounting. Its total must equal ``target_visual_tokens``.
    Out-of-support inputs are explicitly locked to their uniform count.
    All-token fallback is deliberately absent: uniform is always a feasible
    reference within supported compression, without spending hidden memory.
    """
    policy = load_policy(policy)
    rows = _inference_rows(answer_free_rows, policy)
    goal = _integer(target_visual_tokens, "target_visual_tokens", minimum=1)
    if goal > sum(row["input_visual_tokens"] for row in rows):
        raise ValueError("target exceeds all image tokens")
    if uniform_counts is None:
        uniform = _uniform(rows, goal)
    else:
        if not isinstance(uniform_counts, Mapping) or set(uniform_counts) != {row["row_hash"] for row in rows}:
            raise ValueError("uniform counts must exactly match assignment identities")
        uniform = [_integer(uniform_counts[row["row_hash"]], "uniform count", minimum=1) for row in rows]
    if sum(uniform) != goal:
        raise ValueError("uniform counts do not match the exact resource target")
    sizes = [row["input_visual_tokens"] for row in rows]
    for n, count in zip(sizes, uniform):
        if not _physical_count(n, 10) <= count <= _physical_count(n, 70):
            raise ValueError("uniform allocation lies outside measured [10%,70%] support")
    values = np.asarray([row["statistic_value"] for row in rows])
    outside = (values < policy["support"][0]) | (values > policy["support"][1])
    curves = _curves_at(policy, values)
    base = np.asarray([[np.interp(u / n, PHYSICAL_RETENTIONS, curve) for curve, u, n in zip(member, uniform, sizes)] for member in curves])

    def utility(index, count):
        gains = np.asarray([np.interp(count / sizes[index], PHYSICAL_RETENTIONS, member[index]) for member in curves]) - base[:, index]
        return float(gains.mean() - policy["risk_penalty"] * gains.std())

    choices, bounds = [], []
    for index, (n, u) in enumerate(zip(sizes, uniform)):
        low, high = max(.10, u / n - .10), min(.70, u / n + .10)
        percentages = range(max(10, int(math.ceil(100 * low - 1e-10))), min(70, int(math.floor(100 * high + 1e-10))) + 1)
        counts = {u} if outside[index] else {_physical_count(n, p) for p in percentages} | {u}
        ordered = sorted(counts, key=lambda count: (abs(count - u), count))
        choices.append(ordered)
        bounds.append((min(ordered), max(ordered)))
    if not sum(low for low, _ in bounds) <= goal <= sum(high for _, high in bounds):
        raise ValueError("allocation bounds cannot meet the exact resource target")
    utilities = [np.asarray([utility(i, count) for count in counts]) for i, counts in enumerate(choices)]
    scale = max(1.0, float(np.median(uniform)))

    def choose(price):
        result = []
        for i, counts in enumerate(choices):
            objective = utilities[i] - price * (np.asarray(counts) - uniform[i]) / scale
            # Uniform/near-uniform is first in each count list; ties cannot
            # manufacture variation from an uninformative fitted surface.
            best = np.flatnonzero(objective >= objective.max() - 1e-12)[0]
            result.append(counts[int(best)])
        return result

    def reconcile(initial):
        counts = list(initial)
        direction = 1 if sum(counts) < goal else -1
        residual = abs(goal - sum(counts))
        heap = []

        def push(index):
            proposed = counts[index] + direction
            if bounds[index][0] <= proposed <= bounds[index][1]:
                loss = utility(index, counts[index]) - utility(index, proposed)
                heapq.heappush(heap, (loss, rows[index]["row_hash"], index, proposed))

        for i in range(len(counts)):
            push(i)
        for _ in range(residual):
            if not heap:
                raise ValueError("integer reconciliation exhausted feasible capacity")
            _, _, index, proposed = heapq.heappop(heap)
            counts[index] = proposed
            push(index)
        return counts, residual

    price, candidates = 0.0, [(list(uniform), 0, 0.0)]
    reason = "predicted_no_gain_uniform"
    if policy["support"][0] != policy["support"][1] and any(low != high for low, high in bounds):
        low, high = -1.0, 1.0
        for _ in range(64):
            if sum(choose(low)) >= goal:
                break
            low *= 2
        else:
            raise ValueError("failed to bracket low resource price")
        for _ in range(64):
            if sum(choose(high)) <= goal:
                break
            high *= 2
        else:
            raise ValueError("failed to bracket high resource price")
        for _ in range(80):
            middle = (low + high) / 2
            if sum(choose(middle)) > goal:
                low = middle
            else:
                high = middle
        for candidate_price in (low, high):
            counts, residual = reconcile(choose(candidate_price))
            candidates.append((counts, residual, candidate_price))
    elif policy["support"][0] == policy["support"][1]:
        reason = "no_statistic_variation_uniform"
    elif bool(outside.all()):
        reason = "all_out_of_support_uniform"
    best_counts, residual, best_score = list(uniform), 0, 0.0
    for counts, repaired, candidate_price in candidates[1:]:
        score = sum(utility(i, count) for i, count in enumerate(counts))
        tie_key = (sum(a != b for a, b in zip(counts, uniform)), tuple(counts))
        best_key = (sum(a != b for a, b in zip(best_counts, uniform)), tuple(best_counts))
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and tie_key < best_key):
            best_counts, residual, best_score, price = counts, repaired, score, candidate_price
    if best_counts != uniform:
        reason = "adaptive_predicted_benefit"
    records = {}
    for i, row in enumerate(rows):
        count, u = best_counts[i], uniform[i]
        records[row["row_hash"]] = {**row, "uniform_selected_count": u, "adaptive_selected_count": count,
                                   "uniform_retention": u / sizes[i], "adaptive_retention": count / sizes[i],
                                   "out_of_support": bool(outside[i]), "fallback": bool(outside[i]),
                                   "fallback_kind": "uniform_out_of_support" if outside[i] else None,
                                   "changed_from_uniform": count != u,
                                   "integer_reconciliation_allowed": True,
                                   "predicted_conservative_gain": utility(i, count)}
    payload = {"format": ASSIGNMENT_FORMAT, "policy_sha256": policy["policy_sha256"],
               "statistic": policy["statistic"], "target_visual_tokens": goal,
               "uniform_visual_tokens": sum(uniform), "adaptive_visual_tokens": sum(best_counts),
               "token_error": 0, "changed_rows": sum(a != b for a, b in zip(best_counts, uniform)),
               "uniform_fallback_rows": int(outside.sum()), "full_fallback_rows": 0,
               "integer_reconciliation_tokens": residual, "resource_price": price,
               "predicted_conservative_total_gain": best_score, "decision": reason,
               "allocation_scope": "one homogeneous cache geometry; caller verifies allocated bytes",
               "quality_scope": "predicted screening utility only; exact-count physical inference required",
               "inputs_sha256": _digest(rows), "rows": records}
    return _sealed(payload, "assignments_sha256")


def load_assignments(payload, *, policy=None):
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping):
        raise ValueError("assignments must be a JSON object")
    result = json.loads(_canonical(dict(payload)))
    claimed = result.pop("assignments_sha256", None)
    if claimed != _digest(result) or result.get("format") != ASSIGNMENT_FORMAT:
        raise ValueError("assignment digest or format mismatch")
    rows = result.get("rows", {})
    if not rows or any(key != row.get("row_hash") for key, row in rows.items()):
        raise ValueError("invalid assignment row identities")
    if (sum(row["adaptive_selected_count"] for row in rows.values()) != result["target_visual_tokens"]
            or sum(row["uniform_selected_count"] for row in rows.values()) != result["target_visual_tokens"]
            or result["adaptive_visual_tokens"] != result["target_visual_tokens"]
            or result["uniform_visual_tokens"] != result["target_visual_tokens"]):
        raise ValueError("assignments do not conserve the exact target")
    if policy is not None and result["policy_sha256"] != load_policy(policy)["policy_sha256"]:
        raise ValueError("assignments belong to a different policy")
    result["assignments_sha256"] = claimed
    return result


def serialize_assignments(assignments, *, policy=None):
    return _canonical(load_assignments(assignments, policy=policy))
