"""Local physical-response allocation and exact-budget negative controls.

The v3 experiment is immutable. This module fits paired changes relative to
each request's actual uniform operating point, not absolute quality at distant
global knots. Only fit_policy and fit_benefit_scalar accept quality labels.
All assigned counts still require physical inference before making claims.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import heapq
import json
import math
from typing import Mapping

import numpy as np

from .adaptive_utility_v3 import grouped_folds


STATISTICS = ("normalized_entropy", "local_shell", "nearest_neighbor_diversity")
OFFSETS = (-10, -5, 0, 5, 10)
TARGETS = (40, 50, 60, 70)
FORMAT = "mlsys_local_physical_allocation_v4"
_INFERENCE = {"row_hash", "group_id", "dataset", "input_visual_tokens", "input_text_tokens", "statistic_value"}
_FIT = _INFERENCE | {"uniform_selected_count", "quality_by_count"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _seal(value, name="sha256"):
    value = json.loads(_canonical(value))
    return {**value, name: _digest(value)}


def _verify(value, name="sha256"):
    if not isinstance(value, Mapping):
        raise ValueError("artifact must be an object")
    result = json.loads(_canonical(dict(value)))
    checksum = result.pop(name, None)
    if checksum != _digest(result):
        raise ValueError("artifact digest mismatch")
    result[name] = checksum
    return result


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _name(value, name):
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _geometry_rows(rows, *, inference=False):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be nonempty")
    checked, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("each row must be an object")
        required = _INFERENCE if inference else _INFERENCE - {"statistic_value"}
        if not required <= set(row) or (inference and set(row) != required):
            raise ValueError("inference requires only the declared answer-free fields")
        item = {key: _name(row[key], key) for key in ("row_hash", "group_id", "dataset")}
        if item["row_hash"] in seen:
            raise ValueError("duplicate request identity")
        seen.add(item["row_hash"])
        item["input_visual_tokens"] = _integer(row["input_visual_tokens"], "input_visual_tokens", 1)
        item["input_text_tokens"] = _integer(row["input_text_tokens"], "input_text_tokens")
        if inference:
            item["statistic_value"] = _finite(row["statistic_value"], "statistic_value")
        checked.append(item)
    return sorted(checked, key=lambda row: row["row_hash"])


def uniform_budget(answer_free_rows, reduction_percent, *, reserved_generated_slots=127):
    """Derive a common-retention allocation from the full allocated-KV count.

    The caller must use one homogeneous bytes-per-cache-slot geometry. All
    methods retain the same protected text and generated-token reservation.
    Integer flooring of the aggregate byte-slot target is disclosed, never
    replaced with a looser feasible target.
    """
    rows = _geometry_rows(answer_free_rows)
    reduction = _integer(reduction_percent, "reduction_percent")
    if reduction > 100:
        raise ValueError("reduction_percent must not exceed100")
    reserve = _integer(reserved_generated_slots, "reserved_generated_slots")
    visual = sum(row["input_visual_tokens"] for row in rows)
    protected = sum(row["input_text_tokens"] + reserve for row in rows)
    full = visual + protected
    target = full * (100 - reduction) // 100
    goal = target - protected
    base = {"format": FORMAT + "_uniform", "reduction_percent": reduction,
            "full_allocated_slots": full, "protected_allocated_slots": protected,
            "target_allocated_slots": target, "target_visual_tokens": goal,
            "reserved_generated_slots": reserve, "minimum_visual_tokens_per_request": 1}
    if goal < len(rows) or goal > visual:
        return _seal({**base, "feasible": False, "reason": "protected_tokens_and_one_image_token_exceed_target",
                      "counts": {}, "maximum_reduction_fraction": 1 - (protected + len(rows)) / full})
    # Lower-bound water filling. Simply flooring N*ratio then forcing K>=1
    # can overspend the target for mixed token geometries.
    fixed, active = {}, list(range(len(rows)))
    remaining = goal
    while active:
        weight = sum(rows[index]["input_visual_tokens"] for index in active)
        small = [index for index in active if rows[index]["input_visual_tokens"] * remaining < weight]
        if not small:
            break
        for index in small:
            fixed[index] = 1
        remaining -= len(small)
        small_set = set(small)
        active = [index for index in active if index not in small_set]
    counts = dict(fixed)
    if active:
        weight = sum(rows[index]["input_visual_tokens"] for index in active)
        for index in active:
            counts[index] = rows[index]["input_visual_tokens"] * remaining // weight
        order = sorted(active, key=lambda i: (-(rows[i]["input_visual_tokens"] * remaining % weight), rows[i]["row_hash"]))
        for index in order[:remaining - sum(counts[i] for i in active)]:
            counts[index] += 1
    result = {row["row_hash"]: counts[index] for index, row in enumerate(rows)}
    if sum(result.values()) != goal or any(not 1 <= result[row["row_hash"]] <= row["input_visual_tokens"] for row in rows):
        raise AssertionError("uniform allocation does not conserve feasible resource count")
    return _seal({**base, "feasible": True, "reason": None, "counts": result,
                  "actual_reduction_fraction": 1 - target / full, "token_error": 0})


def local_probe_counts(input_visual_tokens, uniform_selected_count):
    """Five local physical counts; aliases at boundaries share one inference."""
    n = _integer(input_visual_tokens, "input_visual_tokens", 1)
    uniform = _integer(uniform_selected_count, "uniform_selected_count", 1)
    if uniform > n:
        raise ValueError("uniform count exceeds input image tokens")
    return {offset: min(n, max(1, uniform + n * offset // 100)) for offset in OFFSETS}


def statistic_from_features(feature, statistic, uniform_count):
    """One frozen scorer statistic, with shell centered on actual uniform K/N."""
    if statistic not in STATISTICS:
        raise ValueError("unknown statistic")
    n = _integer(feature["input_visual_tokens"], "input_visual_tokens", 1)
    uniform = _integer(uniform_count, "uniform_count", 1)
    if uniform > n:
        raise ValueError("uniform count exceeds image count")
    if statistic != "local_shell":
        result = _finite(feature[statistic], statistic)
        if not 0 <= result <= (2 if statistic == "nearest_neighbor_diversity" else 1):
            raise ValueError("statistic outside its mathematical range")
        return result
    masses = np.asarray(feature["top_mass_by_percent"], dtype=float)
    if (masses.shape != (101,) or not np.isfinite(masses).all() or np.any(np.diff(masses) < -1e-7)
            or abs(masses[0]) > 1e-8 or abs(masses[-1] - 1) > 2e-6):
        raise ValueError("invalid cached top-mass feature")
    # Features were measured at integer percentages, so bind the convention
    # explicitly: nearest integer percent, half up, then +/-10 percentagepoints.
    center = (100 * uniform + n // 2) // n
    return float(masses[min(100, center + 10)] - masses[max(0, center - 10)])


def _fit_rows(rows):
    _geometry_rows(rows)
    checked = []
    group_dataset = {}
    for row in rows:
        if set(row) != _FIT:
            raise ValueError("fit rows require exactly the declared physical development fields")
        item = _geometry_rows([row], inference=False)[0]
        if group_dataset.setdefault(item["group_id"], item["dataset"]) != item["dataset"]:
            raise ValueError("one image group spans multiple benchmarks")
        item["statistic_value"] = _finite(row["statistic_value"], "statistic_value")
        uniform = _integer(row["uniform_selected_count"], "uniform_selected_count", 1)
        probes = local_probe_counts(item["input_visual_tokens"], uniform)
        quality = row["quality_by_count"]
        if not isinstance(quality, Mapping):
            raise ValueError("quality_by_count must map exact physical integer counts to scores")
        converted = {}
        for key, value in quality.items():
            if isinstance(key, bool) or str(int(key)) != str(key) or int(key) in converted:
                raise ValueError("quality count keys must be unique canonical integers")
            score = _finite(value, "physical quality")
            if not 0 <= score <= 1:
                raise ValueError("physical qualities must lie in[0,1]")
            converted[int(key)] = score
        if set(converted) != set(probes.values()):
            raise ValueError("all and only the planned local physical counts are required")
        measured = sorted(converted)
        actual_offsets = [(count - uniform) / item["input_visual_tokens"] for count in measured]
        gains = [converted[count] - converted[uniform] for count in measured]
        # Integer-token rounding is handled at the actually measured offsets.
        # Endpoint extension only describes clipped requests; inference cannot
        # leave physical count bounds1..N or the requested +/-10% neighborhood.
        item.update(uniform_selected_count=uniform, quality_by_count={str(k): converted[k] for k in measured},
                    gains=np.interp(np.asarray(OFFSETS) / 100, actual_offsets, gains).tolist())
        item["gains"][2] = 0.0
        checked.append(item)
    return sorted(checked, key=lambda row: row["row_hash"])


def _fit_member(rows, min_groups, shrinkage):
    groups = {row["group_id"] for row in rows}
    if len(groups) < min_groups:
        raise ValueError("too few independent groups in training fold")
    x = np.asarray([row["statistic_value"] for row in rows])
    y = np.asarray([row["gains"] for row in rows])
    # Both multiple questions from one image and benchmark sizes are balanced.
    group_counts = Counter(row["group_id"] for row in rows)
    domain_groups = Counter({dataset: len({r["group_id"] for r in rows if r["dataset"] == dataset}) for dataset in {r["dataset"] for r in rows}})
    weights = np.asarray([len(rows) / (len(domain_groups) * domain_groups[r["dataset"]] * group_counts[r["group_id"]]) for r in rows])
    prior = np.average(y, axis=0, weights=weights)
    order = np.argsort(x, kind="stable")
    bins = [part.tolist() for part in np.array_split(order, min(5, len(order))) if len(part)]
    index = 0
    while index < len(bins) - 1:
        if x[bins[index][-1]] == x[bins[index + 1][0]]:
            bins[index:index + 2] = [bins[index] + bins[index + 1]]
        else:
            index += 1
    while len(bins) > 1:
        small = next((i for i, part in enumerate(bins) if len({rows[j]["group_id"] for j in part}) < min_groups), None)
        if small is None:
            break
        neighbor = 1 if small == 0 else small - 1
        lo, hi = sorted((small, neighbor))
        bins[lo:hi + 1] = [sum(bins[lo:hi + 1], [])]
    records = []
    for indices in bins:
        w = weights[indices]
        curve = ((y[indices] * w[:, None]).sum(axis=0) + shrinkage * prior) / (w.sum() + shrinkage)
        curve[2] = 0
        records.append({"center": float(np.average(x[indices], weights=w)), "gains": curve.tolist(),
                        "groups": len({rows[j]["group_id"] for j in indices}), "rows": len(indices)})
    return {"bins": records, "training_groups": len(groups), "dataset_group_counts": dict(domain_groups), "prior_gain": prior.tolist()}


def fit_policy(rows, *, statistic, identity=None, min_groups=20, shrinkage=30.0):
    """Five group-exclusion fits of *paired local benefit*, not correctness."""
    if statistic not in (*STATISTICS, "learned_benefit_scalar"):
        raise ValueError("unknown statistic")
    minimum = _integer(min_groups, "min_groups", 1)
    shrinkage = _finite(shrinkage, "shrinkage")
    if shrinkage < 0:
        raise ValueError("negative shrinkage")
    checked = _fit_rows(rows)
    folds = grouped_folds(checked)
    members = []
    for fold in range(5):
        member = _fit_member([r for r in checked if folds[r["group_id"]] != fold], minimum, shrinkage)
        member["excluded_fold"] = fold
        members.append(member)
    values = [row["statistic_value"] for row in checked]
    counts = [row["input_visual_tokens"] for row in checked]
    ratios = [row["uniform_selected_count"] / row["input_visual_tokens"] for row in checked]
    return _seal({"format": FORMAT + "_policy", "statistic": statistic, "identity": identity or {},
                  "offset_percents": list(OFFSETS), "members": members, "risk_penalty": .5,
                  "min_groups_per_bin": minimum, "shrinkage": shrinkage,
                  "support": {"statistic": [min(values), max(values)], "input_visual_tokens": [min(counts), max(counts)],
                              "uniform_retention": [min(ratios), max(ratios)]},
                  "calibration_rows": [r["row_hash"] for r in checked], "calibration_groups": sorted(folds),
                  "calibration_datasets": sorted({r["dataset"] for r in checked}), "group_folds": folds,
                  "calibration_data_sha256": _digest(checked), "curve_constraint": "none; measured nonmonotone gains preserved",
                  "prediction_inputs": ["one scalar"], "support_checks": ["scalar", "input image count", "uniform image retention"],
                  "fit_weighting": "equal benchmark weight; equal image-group weight within benchmark",
                  "quality_scope": "fit-only physically measured local responses"}, "policy_sha256")


def load_policy(policy):
    result = _verify(policy, "policy_sha256")
    if (result.get("format") != FORMAT + "_policy" or result.get("offset_percents") != list(OFFSETS)
            or result.get("statistic") not in (*STATISTICS, "learned_benefit_scalar") or result.get("risk_penalty") != .5
            or len(result.get("members", [])) != 5):
        raise ValueError("unsupported local-response policy")
    for fold, member in enumerate(result["members"]):
        centers = np.asarray([part["center"] for part in member["bins"]])
        curves = np.asarray([part["gains"] for part in member["bins"]])
        if (member["excluded_fold"] != fold or not 1 <= len(centers) <= 5 or not np.isfinite(centers).all()
                or np.any(np.diff(centers) <= 0) or curves.shape != (len(centers), 5)
                or not np.isfinite(curves).all() or np.any(np.abs(curves) > 1 + 1e-12)
                or np.any(curves[:, 2] != 0) or any(b["groups"] < result["min_groups_per_bin"] for b in member["bins"])):
            raise ValueError("invalid local-response calibration member")
    for bounds in result["support"].values():
        if len(bounds) != 2 or not all(math.isfinite(value) for value in bounds) or bounds[0] > bounds[1]:
            raise ValueError("invalid calibration support")
    return result


def _curves(policy, value):
    return np.asarray([[np.interp(value, [b["center"] for b in member["bins"]], [b["gains"][j] for b in member["bins"]])
                        for j in range(5)] for member in policy["members"]])


def allocate(policy, answer_free_rows, *, uniform_counts):
    """Globally price tokens then deterministically reconcile to exact total."""
    policy = load_policy(policy)
    rows = _geometry_rows(answer_free_rows, inference=True)
    if not isinstance(uniform_counts, Mapping) or set(uniform_counts) != {r["row_hash"] for r in rows}:
        raise ValueError("uniform count identities differ")
    fitted_rows, fitted_groups = set(policy["calibration_rows"]), set(policy["calibration_groups"])
    uniform, sizes, reasons, curves, bounds, choices = [], [], [], [], [], []
    support = policy["support"]
    for row in rows:
        if row["row_hash"] in fitted_rows or row["group_id"] in fitted_groups:
            raise ValueError("assignment identities overlap fitting")
        n = row["input_visual_tokens"]
        u = _integer(uniform_counts[row["row_hash"]], "uniform count", 1)
        if u > n:
            raise ValueError("uniform count exceeds image count")
        why = []
        checks = (("statistic", row["statistic_value"], 1e-12), ("input_visual_tokens", n, 0),
                  ("uniform_retention", u / n, 1 / n))
        for name, value, tolerance in checks:
            if not support[name][0] - tolerance <= value <= support[name][1] + tolerance:
                why.append(name)
        probes = local_probe_counts(n, u)
        lo, hi = (u, u) if why else (min(probes.values()), max(probes.values()))
        counts = {u} if why else {min(hi, max(lo, u + n * offset // 100)) for offset in range(-10, 11)} | {u, lo, hi}
        sizes.append(n)
        uniform.append(u)
        reasons.append(why)
        curves.append(_curves(policy, row["statistic_value"]))
        bounds.append((lo, hi))
        choices.append(sorted(counts, key=lambda count: (abs(count - u), count)))
    goal = sum(uniform)

    def utility(index, count):
        offset = (count - uniform[index]) / sizes[index]
        gains = np.asarray([np.interp(offset, np.asarray(OFFSETS) / 100, curve) for curve in curves[index]])
        return float(gains.mean() - .5 * gains.std())

    utilities = [np.asarray([utility(i, count) for count in counts]) for i, counts in enumerate(choices)]
    scale = max(1., float(np.median(uniform)))

    def choose(price):
        result = []
        for i, counts in enumerate(choices):
            values = utilities[i] - price * (np.asarray(counts) - uniform[i]) / scale
            best = int(np.flatnonzero(values >= values.max() - 1e-12)[0])
            result.append(counts[best])
        return result

    def reconcile(counts):
        counts = list(counts)
        direction = 1 if sum(counts) < goal else -1
        residual = abs(goal - sum(counts))
        heap = []

        def push(i):
            proposed = counts[i] + direction
            if bounds[i][0] <= proposed <= bounds[i][1]:
                heapq.heappush(heap, (utility(i, counts[i]) - utility(i, proposed), rows[i]["row_hash"], i, proposed))

        for i in range(len(rows)):
            push(i)
        for _ in range(residual):
            if not heap:
                raise AssertionError("exact token reconciliation exhausted capacity")
            _, _, i, proposed = heapq.heappop(heap)
            counts[i] = proposed
            push(i)
        return counts, residual

    candidates = [(uniform, 0, 0.0)]
    if any(lo != hi for lo, hi in bounds) and support["statistic"][0] != support["statistic"][1]:
        low, high = -1., 1.
        for _ in range(64):
            if sum(choose(low)) >= goal:
                break
            low *= 2
        else:
            raise ValueError("resource-price lower bracket failed")
        for _ in range(64):
            if sum(choose(high)) <= goal:
                break
            high *= 2
        else:
            raise ValueError("resource-price upper bracket failed")
        for _ in range(80):
            middle = (low + high) / 2
            if sum(choose(middle)) > goal:
                low = middle
            else:
                high = middle
        for price in (low, high):
            counts, residual = reconcile(choose(price))
            candidates.append((counts, residual, price))
    best, repaired, price, best_score = list(uniform), 0, 0., 0.
    for candidate, residual, candidate_price in candidates[1:]:
        score = sum(utility(i, count) for i, count in enumerate(candidate))
        key = (sum(a != b for a, b in zip(candidate, uniform)), tuple(candidate))
        best_key = (sum(a != b for a, b in zip(best, uniform)), tuple(best))
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and key < best_key):
            best, repaired, price, best_score = candidate, residual, candidate_price, score
    records = {}
    for i, row in enumerate(rows):
        records[row["row_hash"]] = {**row, "uniform_selected_count": uniform[i], "adaptive_selected_count": best[i],
                                   "changed_from_uniform": best[i] != uniform[i], "uniform_retention": uniform[i] / sizes[i],
                                   "adaptive_retention": best[i] / sizes[i], "out_of_support": bool(reasons[i]),
                                   "fallback": bool(reasons[i]), "support_failures": reasons[i],
                                   "fallback_kind": "uniform_out_of_support" if reasons[i] else None,
                                   "predicted_conservative_gain": utility(i, best[i])}
    return _seal({"format": FORMAT + "_assignments", "policy_sha256": policy["policy_sha256"],
                  "statistic": policy["statistic"], "target_visual_tokens": goal, "uniform_visual_tokens": goal,
                  "adaptive_visual_tokens": sum(best), "token_error": sum(best) - goal, "rows": records,
                  "changed_rows": sum(a != b for a, b in zip(best, uniform)), "uniform_fallback_rows": sum(bool(r) for r in reasons),
                  "full_fallback_rows": 0, "integer_reconciliation_tokens": repaired, "resource_price": price,
                  "predicted_conservative_total_gain": best_score,
                  "decision": "adaptive_predicted_benefit" if best != uniform else "uniform_no_supported_predicted_gain",
                  "quality_scope": "predicted utility only; exact-count physical inference required"}, "assignments_sha256")


def shuffle_assignments(assignments, *, seed=42):
    """Permute already assigned counts within equal-size, same-benchmark groups.

    This tests *which input* receives a budget without changing the budget
    multiset, benchmark aggregate, or image-token feasibility. Singleton and
    equal-budget strata are explicitly immovable, not fictitious controls.
    """
    source = _verify(assignments, "assignments_sha256")
    if source.get("format") != FORMAT + "_assignments":
        raise ValueError("wrong assignment format")
    seed = _integer(seed, "seed")
    strata = defaultdict(list)
    for key, row in sorted(source["rows"].items()):
        if key != row["row_hash"]:
            raise ValueError("assignment identity mismatch")
        n = _integer(row["input_visual_tokens"], "input_visual_tokens", 1)
        count = _integer(row["adaptive_selected_count"], "adaptive_selected_count", 1)
        if count > n:
            raise ValueError("invalid original assigned count")
        strata[(row["dataset"], n)].append(key)
    if sum(r["adaptive_selected_count"] for r in source["rows"].values()) != source["target_visual_tokens"]:
        raise ValueError("source allocation fails exact budget")
    rows, movable, changed, descriptions = {}, 0, 0, []
    for (dataset, n), keys in sorted(strata.items()):
        counts = [source["rows"][key]["adaptive_selected_count"] for key in keys]
        permuted = list(counts)
        can_move = len(set(counts)) > 1
        if can_move:
            movable += len(keys)
            # SHA ordering makes the control independent of numpy RNG versions
            # and independent of outcomes, statistic values, and traversal order.
            order = sorted(range(len(keys)), key=lambda i: (_digest(["v4-budget-shuffle", seed, dataset, n, keys[i]]), keys[i]))
            permuted = [counts[i] for i in order]
        for key, count, original in zip(keys, permuted, counts):
            row = source["rows"][key]
            rows[key] = {**row, "source_adaptive_selected_count": original, "adaptive_selected_count": count,
                         "adaptive_retention": count / n, "changed_from_uniform": count != row["uniform_selected_count"],
                         "changed_from_source_assignment": count != original, "shuffle_movable_stratum": can_move,
                         "allocation": f"shuffled_seed{seed}", "predicted_conservative_gain": None,
                         "fallback": False, "fallback_kind": None,
                         "support_note": "shuffled negative control does not apply the policy support fallback"}
            changed += count != original
        descriptions.append({"dataset": dataset, "input_visual_tokens": n, "rows": len(keys),
                             "distinct_budgets": len(set(counts)), "movable": can_move,
                             "original_token_sum": sum(counts), "shuffled_token_sum": sum(permuted)})
    total = len(rows)
    return _seal({"format": FORMAT + "_shuffle", "seed": seed, "source_assignments_sha256": source["assignments_sha256"],
                  "target_visual_tokens": source["target_visual_tokens"], "adaptive_visual_tokens": sum(r["adaptive_selected_count"] for r in rows.values()),
                  "token_error": 0, "rows": rows, "strata": descriptions, "movable_rows": movable,
                  "movable_fraction": movable / total, "changed_rows": changed, "changed_fraction": changed / total,
                  "informative_control": changed > 0,
                  "reason": None if changed else "no_assignment_changed; control cannot distinguish intelligent assignment",
                  "quality_scope": "requires physical exact-count inference; no fitted quality is consumed"}, "assignments_sha256")


def leave_one_benchmark_out(rows, *, statistic, identity=None, **kwargs):
    """Fit all-benchmark and each complete-benchmark exclusion policy."""
    datasets = sorted({row["dataset"] for row in rows})
    if len(datasets) < 2:
        raise ValueError("benchmark transfer requires at least two datasets")
    result = {"all_fit": fit_policy(rows, statistic=statistic, identity={**(identity or {}), "held_out_benchmark": None}, **kwargs)}
    for dataset in datasets:
        result[f"leave_out:{dataset}"] = fit_policy([r for r in rows if r["dataset"] != dataset], statistic=statistic,
                                                   identity={**(identity or {}), "held_out_benchmark": dataset}, **kwargs)
    return result


def select_physical_candidate(records):
    """Choose one rule across models using completed physical screening only.

    A record is one candidate/model/benchmark/target panel with paired deltas
    against uniform and the mean of physically executed shuffled controls.
    Scores are fractions, not percentage points. Equal panel coverage is
    mandatory; missing or non-informative shuffle panels cannot manufacturewins.
    """
    fields = {"candidate", "model", "dataset", "target", "stage", "physical", "delta_uniform", "delta_shuffled",
              "delta_uniform_ci_low", "delta_shuffled_ci_low", "shuffle_informative", "shuffle_movable_fraction"}
    grouped = defaultdict(dict)
    for row in records:
        if set(row) != fields or row["stage"] != "screening" or row["physical"] is not True:
            raise ValueError("selection accepts exact schema and physical screening only")
        candidate = _name(row["candidate"], "candidate")
        key = (_name(row["model"], "model"), _name(row["dataset"], "dataset"), _integer(row["target"], "target"))
        if key in grouped[candidate]:
            raise ValueError("duplicate candidate screening panel")
        if type(row["shuffle_informative"]) is not bool:
            raise ValueError("shuffle_informative must be boolean")
        checked = {**row, **{name: _finite(row[name], name) for name in
                            ("delta_uniform", "delta_shuffled", "delta_uniform_ci_low", "delta_shuffled_ci_low", "shuffle_movable_fraction")}}
        if not 0 <= checked["shuffle_movable_fraction"] <= 1:
            raise ValueError("invalid movable fraction")
        if any(checked[name + "_ci_low"] > checked[name] for name in ("delta_uniform", "delta_shuffled")):
            raise ValueError("paired CI lower endpoint exceeds point estimate")
        grouped[candidate][key] = checked
    if not grouped:
        raise ValueError("no physical screening panels")
    coverage = next(iter(grouped.values())).keys()
    if any(panel.keys() != coverage for panel in grouped.values()):
        raise ValueError("candidate physical panel coverage differs")
    summaries = []
    for candidate, panel in grouped.items():
        by_model = defaultdict(list)
        for key, row in panel.items():
            by_model[key[0]].append(row)
        model_means = {model: float(np.mean([r["delta_uniform"] for r in values])) for model, values in by_model.items()}
        def informative(row):
            return row["shuffle_informative"] and row["shuffle_movable_fraction"] >= .2

        def passes(row):
            return (informative(row) and row["delta_uniform"] >= 0 and row["delta_shuffled"] > 0
                    and row["delta_uniform_ci_low"] > -.01 and row["delta_shuffled_ci_low"] > -.01)

        wins = sum(row["delta_uniform"] > 0 and row["delta_shuffled"] > 0 and informative(row) for row in panel.values())
        passed = sum(passes(row) for row in panel.values())
        models_passing = sum(any(passes(r) for r in values) for values in by_model.values())
        # A model with many feasible panels cannot dominate the mean; first
        # prioritize models whose mean adaptive-minus-uniform result is positive.
        summaries.append({"candidate": candidate, "positive_models": sum(value > 0 for value in model_means.values()),
                          "assignment_wins": wins, "passing_panels": passed, "models_with_passing_panel": models_passing,
                          "worst_model_delta_uniform": min(model_means.values()),
                          "worst_panel_delta_uniform": min(row["delta_uniform"] for row in panel.values()),
                          "model_balanced_mean_delta_uniform": float(np.mean(list(model_means.values()))),
                          "model_means": model_means, "panels": len(panel),
                          "informative_shuffle_panels": sum(informative(r) for r in panel.values()),
                          "promotion_eligible": (models_passing == len(by_model) and min(model_means.values()) >= 0
                                                 and min(row["delta_uniform"] for row in panel.values()) >= -.01)})
    summaries.sort(key=lambda row: (-row["models_with_passing_panel"], -row["passing_panels"], -row["positive_models"], -row["assignment_wins"], -row["worst_model_delta_uniform"],
                                    -row["worst_panel_delta_uniform"], -row["model_balanced_mean_delta_uniform"], row["candidate"]))
    entropy = next((row for row in summaries if row["candidate"] == "normalized_entropy"), None)
    learned_needed = entropy is not None and not entropy["promotion_eligible"]
    return _seal({"format": FORMAT + "_physical_selection", "selected_candidate": summaries[0]["candidate"],
                  "ranking": summaries, "selection_stage": "physical screening only",
                  "promoted_candidate": summaries[0]["candidate"] if summaries[0]["promotion_eligible"] else None,
                  "promotion_rule": "every model has a noninferior paired-CI panel beating informative shuffled assignment; all model means nonnegative; no domain-target point loss exceeds1percentagepoint",
                  "learned_benefit_branch_needed": learned_needed,
                  "learned_benefit_branch_reason": ("entropy has not passed across-model physical robustness; learned scalar is exploratory, especially when shuffled controls are uninformative" if learned_needed else None)})


def fit_benefit_scalar(rows, *, identity=None, ridge=10.0):
    """Conditional baseline: three frozen statistics -> one learned benefit.

    This is NOT a single-statistic rule: it uses three input features and one
    scalar output, explicitly recorded in the artifact. Its physical objective
    is endpoint quality change per1% more image tokens, not absolute accuracy.
    Call only after physical screening activates the conditional branch.
    """
    ridge = _finite(ridge, "ridge")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    clean, x, y = [], [], []
    for row in rows:
        if set(row) != _FIT | {"benefit_features"}:
            raise ValueError("benefit fit requires physical fields and exactly three declared features")
        features = row["benefit_features"]
        if not isinstance(features, Mapping) or set(features) != set(STATISTICS):
            raise ValueError("benefit_features must name all three frozen statistics")
        physical = _fit_rows([{k: v for k, v in row.items() if k != "benefit_features"}])[0]
        n, u = physical["input_visual_tokens"], physical["uniform_selected_count"]
        probes = local_probe_counts(n, u)
        low, high = min(probes.values()), max(probes.values())
        slope = 0.0 if high == low else (physical["quality_by_count"][str(high)] - physical["quality_by_count"][str(low)]) / ((high - low) / n * 100)
        clean.append(physical)
        x.append([_finite(features[key], key) for key in STATISTICS])
        y.append(slope)
    if not clean or len({r["row_hash"] for r in clean}) != len(clean):
        raise ValueError("benefit training identities are empty or duplicated")
    x, y = np.asarray(x), np.asarray(y)
    group_counts = Counter(r["group_id"] for r in clean)
    domain_groups = {dataset: len({r["group_id"] for r in clean if r["dataset"] == dataset}) for dataset in {r["dataset"] for r in clean}}
    weights = np.asarray([1 / (len(domain_groups) * domain_groups[r["dataset"]] * group_counts[r["group_id"]]) for r in clean])
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(1e-8)
    z = (x - mean) / scale
    intercept = float(np.average(y, weights=weights))
    weights = weights * len(clean)
    coefficients = np.linalg.solve(z.T @ (weights[:, None] * z) + ridge * np.eye(3), z.T @ (weights * (y - intercept)))
    return _seal({"format": FORMAT + "_benefit_scalar", "identity": identity or {}, "feature_names": list(STATISTICS),
                  "mean": mean.tolist(), "scale": scale.tolist(), "coefficients": coefficients.tolist(), "intercept": intercept,
                  "ridge": ridge, "target": "physical endpoint quality change per1percentagepoint more image tokens",
                  "runtime_inputs": "three frozen statistics; one scalar output, not a one-statistic rule",
                  "calibration_rows": [r["row_hash"] for r in clean], "calibration_groups": sorted({r["group_id"] for r in clean}),
                  "calibration_data_sha256": _digest({"rows": clean, "features": x.tolist()})})


def predict_benefit_scalar(model, features):
    model = _verify(model)
    if model.get("format") != FORMAT + "_benefit_scalar" or model.get("feature_names") != list(STATISTICS):
        raise ValueError("unsupported learned benefit scalar")
    if not isinstance(features, Mapping) or set(features) != set(STATISTICS):
        raise ValueError("prediction requires exactly three answer-free statistic features")
    x = np.asarray([_finite(features[key], key) for key in STATISTICS])
    result = (x - np.asarray(model["mean"])) / np.asarray(model["scale"]) @ np.asarray(model["coefficients"]) + model["intercept"]
    return _finite(result, "predicted benefit")


def crossfit_benefit_scalar(rows, *, identity=None, ridge=10.0):
    """Build training-only scalar predictions without each image's labels.

    Use these out-of-fold values when fitting the downstream local-response
    policy. Use the all-fit scalar model only for disjoint screening/evaluation
    requests. Benchmark-held-out policies must call this on their training
    benchmarks only, never reuse an all-benchmark scalar model.
    """
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("benefit cross-fitting requires nonempty training rows")
    folds = grouped_folds(rows)
    predictions, members = {}, []
    for fold in range(5):
        training = [row for row in rows if folds[row["group_id"]] != fold]
        model = fit_benefit_scalar(training, identity={**(identity or {}), "excluded_fold": fold}, ridge=ridge)
        members.append(model)
        for row in rows:
            if folds[row["group_id"]] == fold:
                predictions[row["row_hash"]] = predict_benefit_scalar(model, row["benefit_features"])
    all_fit = fit_benefit_scalar(rows, identity={**(identity or {}), "excluded_fold": None}, ridge=ridge)
    return _seal({"format": FORMAT + "_benefit_crossfit", "model": all_fit, "members": members,
                  "out_of_fold_values": predictions, "group_folds": folds,
                  "training_only": True, "policy_training_instruction": "replace statistic_value with out_of_fold_values before fit_policy"})


def fit_crossfitted_benefit_policy(rows, *, identity=None, ridge=10.0, min_groups=20, shrinkage=30.0):
    """Fully nested learned-scalar calibration, including ensemble uncertainty.

    Reusing a single global set of OOF scalar features inside each utility
    ensemble member would indirectly let that member's excluded image labels
    enter its other training features. Each outer member therefore rebuilds
    scalar cross-fitting using only that member's training image groups.
    """
    crossfit = crossfit_benefit_scalar(rows, identity=identity, ridge=ridge)

    def scalar_rows(source, fitted):
        return [{**{key: value for key, value in row.items() if key != "benefit_features"},
                 "statistic_value": fitted["out_of_fold_values"][row["row_hash"]]} for row in source]

    policy = fit_policy(scalar_rows(rows, crossfit), statistic="learned_benefit_scalar", identity=identity,
                        min_groups=min_groups, shrinkage=shrinkage)
    members = []
    for fold in range(5):
        source = [row for row in rows if crossfit["group_folds"][row["group_id"]] != fold]
        nested = crossfit_benefit_scalar(source, identity={**(identity or {}), "utility_excluded_fold": fold}, ridge=ridge)
        member = _fit_member(_fit_rows(scalar_rows(source, nested)), min_groups, shrinkage)
        member.update(excluded_fold=fold, scalar_training_groups=sorted({row["group_id"] for row in source}),
                      scalar_crossfit_sha256=nested["sha256"])
        members.append(member)
    policy = {key: value for key, value in policy.items() if key != "policy_sha256"}
    policy.update(members=members, learned_scalar_crossfit="nested within each excluded-group utility member",
                  all_fit_scalar_crossfit_sha256=crossfit["sha256"])
    return {"policy": _seal(policy, "policy_sha256"), "crossfit": crossfit}
