"""Fit-only, full-relative patch budgeting on an exact physical count menu.

The four statistics are separate candidate families. ``combined_ridge`` maps
three inputs to ONE scalar; the other families use only their named statistic.
Three scalar bins and eight count choices require at most 8**3 exact tuples.
No count/quality interpolation or monotonic-response assumption is made.

All losses are fractions in [−1, 1]; allowance arguments are percentage points.
Fit selection and grouped OOF diagnostics are DEVELOPMENT evidence, never a
calibration safety certificate. The caller must independently certify complete
paired cohorts against full-token inference before selecting/deploying a policy.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np

from . import adaptive_utility_v3 as grouping


FORMAT = "mlsys_compression_policy_v8"
KEEP_PERCENTAGES = tuple(range(30, 101, 10))
FIT_ALLOWANCES_PP = (0.0, 0.25, 0.5)
FEATURE_NAMES = ("normalized_entropy", "local_shell", "nearest_neighbor_diversity")
STATISTICS = FEATURE_NAMES + ("combined_ridge",)
FAMILIES = STATISTICS + ("static",)
RIDGE = 10.0
_GEOMETRY = {"row_hash", "group_id", "dataset", "input_visual_tokens", "input_text_tokens"}
_INFERENCE = _GEOMETRY | {"benefit_features"}
_FIT = _INFERENCE | {"quality_by_count"}
_OPTIONAL = {"quality_metric", "category"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _seal(value):
    value = json.loads(_canonical(value))
    return {**value, "sha256": _digest(value)}


def _checked(value):
    if not isinstance(value, Mapping):
        raise ValueError("sealed artifact must be a mapping")
    value = json.loads(_canonical(dict(value)))
    sha = value.pop("sha256", None)
    if sha != _digest(value):
        raise ValueError("artifact hash mismatch")
    return {**value, "sha256": sha}


def sources():
    """Source pins for the new math and the unchanged deterministic fold rule."""
    return {str(Path(module).resolve()): hashlib.sha256(Path(module).read_bytes()).hexdigest()
            for module in (__file__, grouping.__file__)}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite numeric data")
    return float(value)


def _name(value, name):
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"invalid {name}")
    return value


def _identity(value):
    if value is not None and not isinstance(value, Mapping):
        raise ValueError("identity must be a mapping")
    value = json.loads(_canonical(dict(value or {})))
    if "split" in value and value["split"] != "fit":
        raise ValueError("policy fitting is restricted to split=fit")
    return value


def physical_count_menu(input_visual_tokens):
    """Percentage -> exact CEILING count; aliases require only one measurement."""
    n = _integer(input_visual_tokens, "input_visual_tokens", 1)
    return {percent: (n * percent + 99) // 100 for percent in KEEP_PERCENTAGES}


def _feature_value(value, name):
    value = _finite(value, name)
    if not 0 <= value <= (2 if name == "nearest_neighbor_diversity" else 1):
        raise ValueError(f"{name} outside its mathematical range")
    return value


def features_from_cached(feature):
    """Extract the fixed answer-free inputs; shell is mass[60] minus mass[40].

    This deliberately ignores any older operating-point-dependent local_shell.
    The feature cache itself must first pass the caller's provenance audit.
    """
    if not isinstance(feature, Mapping):
        raise ValueError("cached feature must be a mapping")
    masses = np.asarray(feature.get("top_mass_by_percent"), dtype=float)
    if (masses.shape != (101,) or not np.isfinite(masses).all()
            or np.any(np.diff(masses) < -1e-7) or abs(masses[0]) > 1e-8
            or abs(masses[-1] - 1) > 2e-6):
        raise ValueError("invalid top_mass_by_percent")
    return {"normalized_entropy": _feature_value(feature.get("normalized_entropy"), "normalized_entropy"),
            "local_shell": _feature_value(float(masses[60] - masses[40]), "local_shell"),
            "nearest_neighbor_diversity": _feature_value(feature.get("nearest_neighbor_diversity"),
                                                          "nearest_neighbor_diversity")}


def _geometry_rows(rows, *, fit=False, require_mme_pairs=True):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a nonempty list or tuple")
    result, seen = [], set()
    for row in rows:
        if not isinstance(row, Mapping) or not _GEOMETRY <= set(row):
            raise ValueError("row lacks required identity/geometry")
        if (fit and (not _FIT <= set(row) or set(row) - _FIT - _OPTIONAL)) or (not fit and set(row) - _INFERENCE - _OPTIONAL):
            raise ValueError("only the declared fit or answer-free inference fields are allowed")
        item = {key: _name(row[key], key) for key in ("row_hash", "group_id", "dataset")}
        if item["row_hash"] in seen:
            raise ValueError("duplicate row_hash")
        seen.add(item["row_hash"])
        item["input_visual_tokens"] = _integer(row["input_visual_tokens"], "input_visual_tokens", 1)
        item["input_text_tokens"] = _integer(row["input_text_tokens"], "input_text_tokens")
        item["quality_metric"] = row.get("quality_metric")
        item["category"] = row.get("category")
        if item["quality_metric"] not in (None, "anls", "vqa", "relaxed", "mme"):
            raise ValueError("unknown quality_metric")
        if item["category"] is not None:
            _name(item["category"], "category")
        features = row.get("benefit_features")
        if isinstance(features, Mapping) and set(features) - set(FEATURE_NAMES):
            raise ValueError("undeclared benefit feature (labels are forbidden)")
        item["benefit_features"] = dict(features) if isinstance(features, Mapping) else features
        if fit:
            if not isinstance(features, Mapping) or set(features) != set(FEATURE_NAMES):
                raise ValueError("fit requires exactly three benefit_features")
            item["benefit_features"] = {key: _feature_value(features[key], key) for key in FEATURE_NAMES}
            quality = row["quality_by_count"]
            expected = {str(k) for k in physical_count_menu(item["input_visual_tokens"]).values()}
            if not isinstance(quality, Mapping) or set(quality) != expected:
                raise ValueError("all and only exact menu count qualities, including full N, are required")
            item["quality_by_count"] = {key: _finite(value, "quality") for key, value in quality.items()}
            if any(not 0 <= value <= 1 for value in item["quality_by_count"].values()):
                raise ValueError("quality must be a fraction in [0,1]")
        result.append(item)
    if fit and len({row["dataset"] for row in result}) != 1:
        raise ValueError("fit one model/dataset policy at a time; do not pool datasets")
    if fit and len({row["quality_metric"] for row in result}) != 1:
        raise ValueError("one policy cannot mix quality metrics")
    if fit and result[0]["quality_metric"] == "mme":
        groups = {}
        for row in result:
            if not row["category"] or any(value not in (0., 1.) for value in row["quality_by_count"].values()):
                raise ValueError("MME requires categories and binary question qualities")
            groups.setdefault(row["group_id"], []).append(row)
        if require_mme_pairs and any(len(pair) != 2 or len({row["category"] for row in pair}) != 1 for pair in groups.values()):
            raise ValueError("MME requires complete two-question groups in one category")
    return sorted(result, key=lambda row: row["row_hash"])


def _metric_score(rows, qualities):
    """Native normalized metric; MME is category-balanced question/pair accuracy."""
    if rows[0].get("quality_metric") != "mme":
        return float(np.mean(qualities))
    categories = {}
    for row, quality in zip(rows, qualities):
        categories.setdefault(row["category"], {}).setdefault(row["group_id"], []).append(float(quality))
    scores = []
    for groups in categories.values():
        if any(len(pair) != 2 for pair in groups.values()):
            raise ValueError("incomplete MME pair during metric calculation")
        questions = [value for pair in groups.values() for value in pair]
        scores.append(.5 * float(np.mean(questions)) + .5 * float(np.mean([all(value == 1. for value in pair)
                                                                        for pair in groups.values()])))
    return float(np.mean(scores))


def positive_harm_target(row):
    """Mean positive full-relative loss over seven RETENTION levels.

    Rounded count aliases share their physical score; each retention level has
    equal weight in this severity target. This scalar is not the signed risk
    constraint used to select count tuples.
    """
    row = _geometry_rows([row], fit=True, require_mme_pairs=False)[0]
    full = row["quality_by_count"][str(row["input_visual_tokens"])]
    return float(np.mean([max(0., full - row["quality_by_count"][str(count)])
                          for percent, count in physical_count_menu(row["input_visual_tokens"]).items()
                          if percent != 100]))


def _group_weights(rows):
    counts = Counter(row["group_id"] for row in rows)
    return np.asarray([1. / (len(counts) * counts[row["group_id"]]) for row in rows])


def _fit_scalar(rows, identity):
    x = np.asarray([[row["benefit_features"][name] for name in FEATURE_NAMES] for row in rows])
    y = np.asarray([positive_harm_target(row) for row in rows])
    weights = _group_weights(rows)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(1e-8)
    z, intercept = (x - mean) / scale, float(np.average(y, weights=weights))
    w = weights * len(rows)
    coefficients = np.linalg.solve(z.T @ (w[:, None] * z) + RIDGE * np.eye(3),
                                   z.T @ (w * (y - intercept)))
    return _seal({"format": FORMAT + "_scalar", "identity": identity,
                  "feature_names": list(FEATURE_NAMES), "ridge": RIDGE,
                  "mean": mean.tolist(), "scale": scale.tolist(),
                  "coefficients": coefficients.tolist(), "intercept": intercept,
                  "target": "mean_positive_full_relative_question_score_loss_over_seven_retention_levels",
                  "target_note": "question-score severity surrogate; tuple selection uses the native dataset metric",
                  "fit_row_hashes": [row["row_hash"] for row in rows],
                  "fit_group_ids": sorted({row["group_id"] for row in rows}),
                  "fit_data_sha256": _digest(rows)})


def _predict_scalar(model, features):
    x = np.asarray([features[name] for name in FEATURE_NAMES])
    return float(model["intercept"] + ((x - model["mean"]) / model["scale"]) @ model["coefficients"])


def crossfit_combined_scalar(rows, *, identity=None):
    """Every OOF value excludes its entire image/document group from training."""
    rows, identity = _geometry_rows(rows, fit=True), _identity(identity)
    folds = grouping.grouped_folds(rows)
    members, values = [], {}
    for fold in range(5):
        training = [row for row in rows if folds[row["group_id"]] != fold]
        model = _fit_scalar(training, {**identity, "excluded_fold": fold})
        members.append(model)
        for row in rows:
            if folds[row["group_id"]] == fold:
                values[row["row_hash"]] = _predict_scalar(model, row["benefit_features"])
    return _seal({"format": FORMAT + "_scalar_crossfit", "model": _fit_scalar(rows, identity),
                  "members": members, "group_folds": folds, "out_of_fold_values": values})


def _bins(rows, values, min_groups, *, static=False):
    if len({row["group_id"] for row in rows}) < min_groups:
        return [], []
    order = sorted(range(len(rows)), key=lambda i: (values[i], rows[i]["row_hash"]))
    # Move quantile cuts to the END of a tied block. Merging whole provisional
    # quantile buckets can incorrectly collapse two distinct, well-supported
    # scalar values into one bin when a middle bucket contains both values.
    boundaries = []
    if not static:
        provisional = np.array_split(np.asarray(order), min(3, len(order)))
        for position in np.cumsum([len(part) for part in provisional])[:-1]:
            position = int(position)
            while position < len(order) and values[order[position - 1]] == values[order[position]]:
                position += 1
            if position < len(order) and position not in boundaries:
                boundaries.append(position)
    ends = [0, *boundaries, len(order)]
    bins = [order[left:right] for left, right in zip(ends, ends[1:])]
    while len(bins) > 1:
        small = next((i for i, indices in enumerate(bins)
                      if len({rows[j]["group_id"] for j in indices}) < min_groups), None)
        if small is None:
            break
        neighbor = 1 if small == 0 else small - 1
        lo, hi = sorted((small, neighbor))
        bins[lo:hi + 1] = [bins[lo] + bins[hi]]
    cutpoints = [(values[left[-1]] + values[right[0]]) / 2 for left, right in zip(bins, bins[1:])]
    records = []
    for indices in bins:
        loss_sums, tokens = [], []
        for percent in KEEP_PERCENTAGES:
            counts = [physical_count_menu(rows[i]["input_visual_tokens"])[percent] for i in indices]
            losses = [rows[i]["quality_by_count"][str(rows[i]["input_visual_tokens"])]
                      - rows[i]["quality_by_count"][str(k)] for i, k in zip(indices, counts)]
            loss_sums.append(float(sum(losses)))
            tokens.append(sum(counts))
        records.append({"rows": len(indices), "groups": len({rows[i]["group_id"] for i in indices}),
                        "scalar_min": float(min(values[i] for i in indices)),
                        "scalar_max": float(max(values[i] for i in indices)),
                        "signed_loss_sums": loss_sums, "visual_token_sums": tokens})
    return records, cutpoints


def _allowances(values):
    if not isinstance(values, (tuple, list)) or not values:
        raise ValueError("declare nonempty fit allowances")
    values = tuple(_finite(value, "fit_allowance_pp") for value in values)
    if len(set(values)) != len(values) or any(value not in FIT_ALLOWANCES_PP for value in values):
        raise ValueError("only distinct predeclared allowances 0, .25, .5 pp are supported")
    return values


def _fit_family(rows, statistic, identity, allowances, min_groups):
    if statistic not in FAMILIES:
        raise ValueError("unknown scalar family")
    supported = len({row["group_id"] for row in rows}) >= min_groups
    crossfit = None
    if statistic == "combined_ridge" and supported:
        crossfit = crossfit_combined_scalar(rows, identity=identity)
        values = [crossfit["out_of_fold_values"][row["row_hash"]] for row in rows]
    elif statistic not in ("static", "combined_ridge"):
        values = [row["benefit_features"][statistic] for row in rows]
    else:
        values = [0.] * len(rows)
    bins, cutpoints = _bins(rows, values, min_groups, static=statistic == "static")
    full_tokens = sum(row["input_visual_tokens"] for row in rows)
    common = {"format": FORMAT, "identity": identity, "statistic": statistic,
              "dataset": rows[0]["dataset"], "quality_metric": rows[0]["quality_metric"],
              "keep_percentages": list(KEEP_PERCENTAGES),
              "min_groups_per_bin": min_groups, "maximum_bins": 3, "bins": bins, "cutpoints": cutpoints,
              "scalar": crossfit["model"] if crossfit else None,
              "scalar_crossfit_sha256": crossfit["sha256"] if crossfit else None,
              "scalar_training": "grouped OOF values train risk bins; all-fit scalar predicts new inputs"
                                  if crossfit else "no learned scalar",
              "feature_definition": {"local_shell": "top_mass_by_percent[60]-top_mass_by_percent[40]"},
              "support": {"scalar": [float(min(values)), float(max(values))],
                          "categories": sorted({row["category"] for row in rows if row["category"] is not None}),
                          **{name: [min(row[name] for row in rows), max(row[name] for row in rows)]
                             for name in ("input_visual_tokens", "input_text_tokens")}},
              "supported": bool(bins), "unsupported_reason": None if bins else "fewer_than_minimum_independent_groups",
              "fit_row_hashes": [row["row_hash"] for row in rows],
              "fit_group_ids": sorted({row["group_id"] for row in rows}), "fit_data_sha256": _digest(rows),
              "source_sha256": sources(), "curve_constraint": "none; signed nonmonotone physical losses preserved",
              "fit_weighting": "native dataset metric (MME category-balanced question/pair accuracy); exact integer visual count sum",
              "bin_response_estimand": "question-score full-relative loss; MME tuple constraint is evaluated jointly across pairs",
              "evidence_scope": "fit-only development; independent full-relative safety calibration required"}
    tuples = []
    if bins:
        bin_ids = np.searchsorted(cutpoints, values, side="right")
        menus = [physical_count_menu(row["input_visual_tokens"]) for row in rows]
        full_score = _metric_score(rows, [row["quality_by_count"][str(row["input_visual_tokens"])] for row in rows])
        for choices in itertools.product(range(len(KEEP_PERCENTAGES)), repeat=len(bins)):
            tokens = sum(part["visual_token_sums"][j] for part, j in zip(bins, choices))
            if rows[0]["quality_metric"] == "mme":
                qualities = [row["quality_by_count"][str(menu[KEEP_PERCENTAGES[choices[index]]])]
                             for row, menu, index in zip(rows, menus, bin_ids)]
                loss = full_score - _metric_score(rows, qualities)
            else:
                loss = sum(part["signed_loss_sums"][j] for part, j in zip(bins, choices)) / len(rows)
            tuples.append((tokens, loss, tuple(KEEP_PERCENTAGES[j] for j in choices)))
    policies = []
    for allowance in allowances:
        feasible = [item for item in tuples if item[1] <= allowance / 100 + 1e-12]
        selected = min(feasible) if feasible else (full_tokens, 0., tuple(100 for _ in bins))
        policies.append(_seal({**common, "fit_allowance_pp": allowance, "bin_keep_percent": list(selected[2]),
                               "fit_summary": {"requests": len(rows), "independent_groups": len(common["fit_group_ids"]),
                                               "enumerated_tuples": len(tuples), "feasible_tuples": len(feasible),
                                               "selected_visual_tokens": selected[0], "full_visual_tokens": full_tokens,
                                               "signed_loss_fraction": selected[1], "signed_loss_pp": 100 * selected[1]}}))
    return policies, crossfit


def load_policy(policy):
    policy = _checked(policy)
    if (policy.get("format") != FORMAT or policy.get("statistic") not in FAMILIES
            or policy.get("keep_percentages") != list(KEEP_PERCENTAGES)
            or policy.get("maximum_bins") != 3 or policy.get("fit_allowance_pp") not in FIT_ALLOWANCES_PP):
        raise ValueError("unsupported v8 policy")
    bins, choices, cuts = policy.get("bins", []), policy.get("bin_keep_percent", []), policy.get("cutpoints", [])
    if (len(bins) > 3 or len(choices) != len(bins) or len(cuts) != max(0, len(bins) - 1)
            or bool(bins) != policy.get("supported") or any(k not in KEEP_PERCENTAGES for k in choices)
            or any(not math.isfinite(v) for v in cuts) or any(a >= b for a, b in zip(cuts, cuts[1:]))
            or any(part["groups"] < policy["min_groups_per_bin"] for part in bins)):
        raise ValueError("invalid scalar bins or count choices")
    for name in ("scalar", "input_visual_tokens", "input_text_tokens"):
        pair = policy["support"][name]
        if len(pair) != 2 or not all(math.isfinite(value) for value in pair) or pair[0] > pair[1]:
            raise ValueError("invalid policy support")
    if policy["statistic"] == "combined_ridge" and policy["supported"]:
        scalar = _checked(policy["scalar"])
        if (scalar.get("format") != FORMAT + "_scalar" or scalar.get("feature_names") != list(FEATURE_NAMES)
                or scalar.get("ridge") != RIDGE
                or any(len(scalar[name]) != 3 for name in ("mean", "scale", "coefficients"))
                or any(value <= 0 or not math.isfinite(value) for value in scalar["scale"])
                or any(not math.isfinite(value) for name in ("mean", "coefficients") for value in scalar[name])
                or not math.isfinite(scalar["intercept"])):
            raise ValueError("invalid combined scalar")
    return policy


def predict_counts(policy, answer_free_rows):
    """Answer-free assignment, with explicit full fallback instead of extrapolation.

    Unknown/label fields and fit-group overlap are errors. Missing/invalid
    required features, unsupported geometry/scalars and foreign datasets use N.
    Invalid N itself cannot be safely repaired and is an input error.
    """
    policy, rows = load_policy(policy), _geometry_rows(answer_free_rows)
    fitted_rows, fitted_groups = set(policy["fit_row_hashes"]), set(policy["fit_group_ids"])
    result = {}
    for row in rows:
        if row["row_hash"] in fitted_rows or row["group_id"] in fitted_groups:
            raise ValueError("prediction identities overlap fit groups")
        reasons, scalar, index = [], None, None
        if not policy["supported"]:
            reasons.append(policy["unsupported_reason"])
        if row["dataset"] != policy["dataset"]:
            reasons.append("dataset_out_of_support")
        if row["quality_metric"] != policy["quality_metric"]:
            reasons.append("quality_metric_missing_or_changed")
        if policy["quality_metric"] == "mme" and row["category"] not in policy["support"]["categories"]:
            reasons.append("category_out_of_support")
        for name in ("input_visual_tokens", "input_text_tokens"):
            lower, upper = policy["support"][name]
            if not lower <= row[name] <= upper:
                reasons.append(name + "_out_of_support")
        required = FEATURE_NAMES if policy["statistic"] == "combined_ridge" else (() if policy["statistic"] == "static" else (policy["statistic"],))
        features = row["benefit_features"]
        if required and (not isinstance(features, Mapping) or any(name not in features for name in required)):
            reasons.append("missing_features")
        elif required:
            try:
                values = {name: _feature_value(features[name], name) for name in required}
                scalar = (_predict_scalar(policy["scalar"], values) if policy["statistic"] == "combined_ridge"
                          and policy["scalar"] else values.get(policy["statistic"]))
                if scalar is not None and not math.isfinite(scalar):
                    reasons.append("invalid_features")
                    scalar = None
            except (TypeError, ValueError, OverflowError):
                reasons.append("invalid_features")
        else:
            scalar = 0.
        if scalar is not None:
            lower, upper = policy["support"]["scalar"]
            if not lower - 1e-12 <= scalar <= upper + 1e-12:
                reasons.append("scalar_out_of_support")
        if not reasons:
            index = int(np.searchsorted(policy["cutpoints"], scalar, side="right"))
        percent = 100 if reasons else policy["bin_keep_percent"][index]
        count = physical_count_menu(row["input_visual_tokens"])[percent]
        result[row["row_hash"]] = {**{key: row[key] for key in _GEOMETRY | _OPTIONAL}, "selected_count": count,
                                   "keep_percent": percent, "statistic_value": scalar, "bin_index": index,
                                   "fallback": bool(reasons), "fallback_reasons": reasons,
                                   "selector": "full" if count == row["input_visual_tokens"] else "scorer"}
    return _seal({"format": FORMAT + "_assignments", "policy_sha256": policy["sha256"], "rows": result,
                  "counts": {key: row["selected_count"] for key, row in result.items()},
                  "total_visual_tokens": sum(row["selected_count"] for row in result.values()),
                  "full_visual_tokens": sum(row["input_visual_tokens"] for row in rows),
                  "full_fallback_rows": sum(row["fallback"] for row in result.values()),
                  "quality_claim": "none; exact-count physical inference and independent calibration required"})


def _diagnostic_summary(records):
    if not records:
        return {"requests": 0, "signed_loss_pp": None}
    loss = _metric_score(records, [row["full_quality"] for row in records]) - _metric_score(records, [row["selected_quality"] for row in records])
    return {"requests": len(records), "independent_groups": len({row["group_id"] for row in records}),
            "signed_loss_pp": 100 * loss,
            "mean_positive_harm_pp": 100 * float(np.mean([max(0., row["signed_loss"]) for row in records])),
            "selected_visual_tokens": sum(row["selected_count"] for row in records),
            "full_visual_tokens": sum(row["full_count"] for row in records),
            "fallback_rows": sum(row["fallback"] for row in records)}


def _crossfit_pipeline(rows, statistic, identity, allowances, min_groups):
    if len({row["group_id"] for row in rows}) < 5:
        return _seal({"format": FORMAT + "_diagnostics", "available": False,
                      "reason": "fewer_than_five_independent_groups", "safety_certificate": False})
    folds = grouping.grouped_folds(rows)
    records = {str(value): [] for value in allowances}
    fold_records = []
    for fold in range(5):
        training = [row for row in rows if folds[row["group_id"]] != fold]
        held_out = [row for row in rows if folds[row["group_id"]] == fold]
        policies, nested = _fit_family(training, statistic, {**identity, "outer_excluded_fold": fold}, allowances, min_groups)
        fold_record = {"fold": fold, "fit_group_ids": sorted({row["group_id"] for row in training}),
                       "held_out_group_ids": sorted({row["group_id"] for row in held_out}),
                       "policy_sha256": [policy["sha256"] for policy in policies],
                       "nested_scalar_crossfit_sha256": nested["sha256"] if nested else None, "by_allowance_pp": {}}
        for allowance, policy in zip(allowances, policies):
            assigned = predict_counts(policy, [{key: value for key, value in row.items() if key != "quality_by_count"}
                                               for row in held_out])
            current = []
            for row in held_out:
                prediction = assigned["rows"][row["row_hash"]]
                count, full = prediction["selected_count"], row["input_visual_tokens"]
                current.append({"row_hash": row["row_hash"], "group_id": row["group_id"], "fold": fold,
                                "category": row["category"], "quality_metric": row["quality_metric"],
                                "selected_count": count, "full_count": full,
                                "full_quality": row["quality_by_count"][str(full)],
                                "selected_quality": row["quality_by_count"][str(count)],
                                "signed_loss": row["quality_by_count"][str(full)] - row["quality_by_count"][str(count)],
                                "fallback": prediction["fallback"], "fallback_reasons": prediction["fallback_reasons"]})
            records[str(allowance)].extend(current)
            fold_record["by_allowance_pp"][str(allowance)] = _diagnostic_summary(current)
        fold_records.append(fold_record)
    return _seal({"format": FORMAT + "_diagnostics", "available": True, "statistic": statistic,
                  "group_folds": folds, "folds": fold_records,
                  "by_allowance_pp": {key: {**_diagnostic_summary(value), "records": value} for key, value in records.items()},
                  "scope": "entire scalar/bin/tuple-selection pipeline refit without each held-out group",
                  "safety_certificate": False})


def fit_policy_family(rows, *, statistic, identity=None, allowances_pp=FIT_ALLOWANCES_PP,
                      min_groups=20, diagnostics=True):
    """Fit up to three frozen tuple policies; no calibration labels are accepted."""
    rows, identity = _geometry_rows(rows, fit=True), _identity(identity)
    min_groups = _integer(min_groups, "min_groups", 5)
    allowances = _allowances(allowances_pp)
    policies, scalar = _fit_family(rows, statistic, identity, allowances, min_groups)
    diagnostic = (_crossfit_pipeline(rows, statistic, identity, allowances, min_groups) if diagnostics
                  else _seal({"format": FORMAT + "_diagnostics", "available": False,
                              "reason": "not_requested", "safety_certificate": False}))
    return _seal({"format": FORMAT + "_family", "identity": identity, "statistic": statistic,
                  "policies": policies, "scalar_crossfit": scalar, "diagnostics": diagnostic,
                  "source_sha256": sources()})


def fit_candidates(rows, *, identity=None, min_groups=20, diagnostics=True, include_static=True):
    """Four scalar families plus an optional static family, each at 0/.25/.5 pp."""
    families = {name: fit_policy_family(rows, statistic=name, identity=identity, min_groups=min_groups,
                                        diagnostics=diagnostics)
                for name in (FAMILIES if include_static else STATISTICS)}
    return _seal({"format": FORMAT + "_candidates", "identity": _identity(identity), "families": families,
                  "policies": [policy for family in families.values() for policy in family["policies"]],
                  "source_sha256": sources(), "safety_certificate": False})


def uniform_at_visual_total(answer_free_rows, total_visual_tokens):
    """Deterministic lower-bound waterfilling at an EXACT visual-token total.

    This is not uniform_budget's allocated-KV reduction percentage interface.
    Matched control counts can lie outside the menu and need physical inference.
    """
    rows = _geometry_rows(answer_free_rows)
    goal = _integer(total_visual_tokens, "total_visual_tokens")
    visual = sum(row["input_visual_tokens"] for row in rows)
    if not len(rows) <= goal <= visual:
        return _seal({"format": FORMAT + "_uniform", "feasible": False,
                      "reason": "total_outside_one_to_N_per_request", "counts": {},
                      "target_visual_tokens": goal, "full_visual_tokens": visual, "token_error": None})
    counts, active, remaining = {}, list(range(len(rows))), goal
    while active:
        weight = sum(rows[index]["input_visual_tokens"] for index in active)
        small = [index for index in active if rows[index]["input_visual_tokens"] * remaining < weight]
        if not small:
            break
        counts.update({index: 1 for index in small})
        remaining -= len(small)
        active = [index for index in active if index not in set(small)]
    if active:
        weight = sum(rows[index]["input_visual_tokens"] for index in active)
        counts.update({index: rows[index]["input_visual_tokens"] * remaining // weight for index in active})
        order = sorted(active, key=lambda i: (-(rows[i]["input_visual_tokens"] * remaining % weight), rows[i]["row_hash"]))
        for index in order[:remaining - sum(counts[index] for index in active)]:
            counts[index] += 1
    result = {row["row_hash"]: counts[index] for index, row in enumerate(rows)}
    if sum(result.values()) != goal or any(not 1 <= result[row["row_hash"]] <= row["input_visual_tokens"] for row in rows):
        raise AssertionError("waterfilling violated exact visual count")
    return _seal({"format": FORMAT + "_uniform", "feasible": True, "counts": result,
                  "target_visual_tokens": goal, "total_visual_tokens": sum(result.values()),
                  "full_visual_tokens": visual, "token_error": 0,
                  "quality_claim": "none; matched-count controls require physical inference"})
