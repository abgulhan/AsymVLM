"""Entropy-only compression policies with nested development selection.

Three quantiles x two directions are six fitting procedures. Each procedure
selects its two retention levels *inside* each training fold. The repeated OOF
scores choose a procedure, not an accuracy certificate. All eight fixed rules
survive, together with every gentler lift of the selected adaptive rule.

Calibration is independent, simultaneous across the frozen compressed rules,
and explicitly approximate when using a paired cluster percentile bootstrap.
No module function reads labels, writes files, or changes the patch scorer.
"""

from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path

import numpy as np

from . import compression_policy_v8 as old
from . import compression_report_v8 as report


FORMAT = "mlsys_compression_policy_v9"
KEEP_PERCENTAGES = old.KEEP_PERCENTAGES
QUANTILES = (.25, .5, .75)
REPEAT_SEEDS = (20261019, 20261020, 20261021)
FIT_ALLOWANCE_PP = .5
REPEAT_ALLOWANCE_PP = 1.
MIN_GROUPS = 20
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261019
physical_count_menu = old.physical_count_menu


def sources():
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, old.__file__, report.__file__)}


def _rows(rows, *, fit=False):
    # Accept the v8 row interface but ignore the two unused statistics entirely.
    # Unknown feature names remain forbidden, to catch accidental label inputs.
    normalized = []
    for row in rows:
        item = copy.deepcopy(row)
        values = item.get("benefit_features")
        if isinstance(values, dict):
            if set(values) - set(old.FEATURE_NAMES):
                raise ValueError("undeclared benefit feature (labels are forbidden)")
            item["benefit_features"] = {"normalized_entropy": values.get("normalized_entropy"),
                                        "local_shell": 0., "nearest_neighbor_diversity": 0.}
        normalized.append(item)
    return old._geometry_rows(normalized, fit=fit)


def _entropy(row):
    return old._feature_value(row["benefit_features"]["normalized_entropy"], "normalized_entropy")


def grouped_folds(rows, seed):
    """Balanced deterministic group folds, stratifying intact MME image pairs."""
    strata = {}
    for row in rows:
        category = row.get("category") if row.get("quality_metric") == "mme" else "all"
        strata.setdefault(category, set()).add(row["group_id"])
    if sum(map(len, strata.values())) != len({r["group_id"] for r in rows}):
        raise ValueError("one group cannot belong to multiple categories")
    result, offset = {}, 0
    for category, groups in sorted(strata.items()):
        order = sorted(groups, key=lambda g: hashlib.sha256(f"{seed}|{category}|{g}".encode()).hexdigest())
        for i, group in enumerate(order):
            result[group] = (offset + i) % 5
        offset += len(order)
    return result


def _threshold(rows, quantile, min_groups):
    values = [_entropy(row) for row in rows]
    threshold = float(np.quantile(values, quantile))
    branch_groups = [{r["group_id"] for r, x in zip(rows, values) if (x >= threshold) == side}
                     for side in (False, True)]
    if min(map(len, branch_groups)) < min_groups:
        return None
    return threshold


def _summary(rows, counts):
    full = [r["quality_by_count"][str(r["input_visual_tokens"])] for r in rows]
    selected = [r["quality_by_count"][str(k)] for r, k in zip(rows, counts)]
    loss = old._metric_score(rows, full) - old._metric_score(rows, selected)
    return {"requests": len(rows), "independent_groups": len({r["group_id"] for r in rows}),
            "mean_loss_pp": 100 * loss,
            "positive_harm_pp": 100 * float(np.mean(np.maximum(0., np.asarray(full) - selected))),
            "retained_visual_tokens": sum(counts),
            "full_visual_tokens": sum(r["input_visual_tokens"] for r in rows)}


def _spec_id(spec):
    if spec["kind"] == "fixed":
        return f"fixed-{spec['below_keep_percent']:03d}"
    return (f"entropy-q{round(100 * spec['threshold_quantile']):02d}"
            f"-below{spec['below_keep_percent']:03d}-above{spec['above_keep_percent']:03d}")


def _fit_procedure(rows, quantile, direction, min_groups):
    threshold = _threshold(rows, quantile, min_groups)
    if threshold is None:
        return None
    candidates = []
    for below in KEEP_PERCENTAGES:
        for above in KEEP_PERCENTAGES:
            if (above - below) * direction <= 0:
                continue
            counts = [physical_count_menu(r["input_visual_tokens"])[above if _entropy(r) >= threshold else below]
                      for r in rows]
            summary = _summary(rows, counts)
            if summary["mean_loss_pp"] <= FIT_ALLOWANCE_PP + 1e-10:
                spec = {"kind": "adaptive", "threshold_quantile": quantile, "threshold": threshold,
                        "below_keep_percent": below, "above_keep_percent": above,
                        "direction": direction, "fit_summary": summary}
                candidates.append(spec)
    return min(candidates, key=lambda s: (s["fit_summary"]["retained_visual_tokens"],
                                         s["fit_summary"]["positive_harm_pp"],
                                         abs(s["above_keep_percent"] - s["below_keep_percent"]),
                                         _spec_id(s))) if candidates else None


def _policy(spec, rows, identity, min_groups):
    return old._seal({"format": FORMAT, **spec, "policy_id": _spec_id(spec),
                      "identity": identity, "dataset": rows[0]["dataset"],
                      "quality_metric": rows[0]["quality_metric"],
                      "statistic": "normalized_entropy" if spec["kind"] == "adaptive" else "none",
                      "min_groups_per_branch": min_groups,
                      "support": {"entropy": [min(map(_entropy, rows)), max(map(_entropy, rows))],
                                  "categories": sorted({r["category"] for r in rows if r["category"]}),
                                  **{key: [min(r[key] for r in rows), max(r[key] for r in rows)]
                                     for key in ("input_visual_tokens", "input_text_tokens")}},
                      "fit_group_ids": sorted({r["group_id"] for r in rows}),
                      "fit_row_hashes": [r["row_hash"] for r in rows],
                      "fit_data_sha256": old._digest(rows), "source_sha256": sources(),
                      "evidence": "development only; independent calibration required"})


def load_policy(policy):
    policy = old._checked(policy)
    if policy.get("format") != FORMAT or policy.get("kind") not in {"fixed", "adaptive"}:
        raise ValueError("not a v9 policy")
    if any(policy.get(k) not in KEEP_PERCENTAGES for k in ("below_keep_percent", "above_keep_percent")):
        raise ValueError("invalid retention menu")
    if policy["kind"] == "fixed":
        if policy["below_keep_percent"] != policy["above_keep_percent"]:
            raise ValueError("fixed policy must have one retention")
    elif (policy.get("threshold_quantile") not in QUANTILES
          or not isinstance(policy.get("threshold"), (int, float))
          or not math.isfinite(policy["threshold"]) or not 0 <= policy["threshold"] <= 1
          or policy["below_keep_percent"] == policy["above_keep_percent"]):
        raise ValueError("invalid adaptive threshold or retention levels")
    if policy.get("policy_id") != _spec_id(policy):
        raise ValueError("policy identity mismatch")
    return policy


def predict_counts(policy, answer_free_rows, *, allow_development_overlap=False):
    """Answer-free exact counts; fixed rules do not use entropy/range fallback."""
    if type(allow_development_overlap) is not bool:
        raise ValueError("allow_development_overlap must be an explicit boolean")
    policy, rows = load_policy(policy), _rows(answer_free_rows)
    fitted_groups, fitted_rows = set(policy["fit_group_ids"]), set(policy["fit_row_hashes"])
    result = {}
    for row in rows:
        if not allow_development_overlap and (row["group_id"] in fitted_groups or row["row_hash"] in fitted_rows):
            raise ValueError("prediction identities overlap development groups")
        reasons, entropy = [], None
        if row["dataset"] != policy["dataset"] or row["quality_metric"] != policy["quality_metric"]:
            reasons.append("dataset_or_metric_out_of_support")
        if policy["kind"] == "adaptive":
            if row["quality_metric"] == "mme" and row["category"] not in policy["support"]["categories"]:
                reasons.append("category_out_of_support")
            for key in ("input_visual_tokens", "input_text_tokens"):
                if not policy["support"][key][0] <= row[key] <= policy["support"][key][1]:
                    reasons.append(key + "_out_of_support")
            try:
                entropy = _entropy(row)
                if not policy["support"]["entropy"][0] <= entropy <= policy["support"]["entropy"][1]:
                    reasons.append("entropy_out_of_support")
            except (KeyError, TypeError, ValueError):
                reasons.append("missing_or_invalid_entropy")
        percent = (100 if reasons else policy["above_keep_percent"]
                   if entropy is not None and entropy >= policy.get("threshold", 1.) else policy["below_keep_percent"])
        count = physical_count_menu(row["input_visual_tokens"])[percent]
        result[row["row_hash"]] = {**{k: row[k] for k in old._GEOMETRY | old._OPTIONAL},
                                   "selected_count": count, "keep_percent": percent,
                                   "statistic_value": entropy, "fallback": bool(reasons),
                                   "fallback_reasons": reasons, "selector": "full" if count == row["input_visual_tokens"] else "scorer"}
    return old._seal({"format": FORMAT + "_assignments", "policy_sha256": policy["sha256"],
                      "rows": result, "counts": {k: v["selected_count"] for k, v in result.items()},
                      "total_visual_tokens": sum(v["selected_count"] for v in result.values()),
                      "full_visual_tokens": sum(r["input_visual_tokens"] for r in rows),
                      "full_fallback_rows": sum(v["fallback"] for v in result.values()),
                      "development_overlap_permitted": allow_development_overlap,
                      "evidence": "development only" if allow_development_overlap else "answer-free assignment; no quality claim"})


def _validate_procedure(rows, quantile, direction, identity, min_groups):
    repeats, folds = [], []
    supported = True
    for seed in REPEAT_SEEDS:
        partition = grouped_folds(rows, seed)
        assigned = {}
        for fold in range(5):
            train = [r for r in rows if partition[r["group_id"]] != fold]
            valid = [r for r in rows if partition[r["group_id"]] == fold]
            spec = _fit_procedure(train, quantile, direction, min_groups) if train and valid else None
            entry = {"seed": seed, "fold": fold, "fit_group_ids": sorted({r["group_id"] for r in train}),
                     "validation_group_ids": sorted({r["group_id"] for r in valid}),
                     "selected_spec": spec, "supported": spec is not None}
            if spec is None:
                supported = False
                counts = {r["row_hash"]: r["input_visual_tokens"] for r in valid}
            else:
                policy = _policy(spec, train, {**identity, "repeat_seed": seed, "excluded_fold": fold}, min_groups)
                counts = predict_counts(policy, [{k: v for k, v in r.items() if k != "quality_by_count"}
                                                 for r in valid])["counts"]
            assigned.update(counts)
            entry["validation_assignments"] = counts
            folds.append(entry)
        if len(assigned) != len(rows):
            raise AssertionError("cross validation omitted or repeated requests")
        repeats.append({"seed": seed, **_summary(rows, [assigned[r["row_hash"]] for r in rows])})
    mean_loss = float(np.mean([r["mean_loss_pp"] for r in repeats]))
    return {"procedure_id": f"q{round(100*quantile):02d}-direction{direction:+d}",
            "threshold_quantile": quantile, "direction": direction, "supported": supported,
            "eligible": bool(supported and mean_loss <= FIT_ALLOWANCE_PP + 1e-10
                             and all(r["mean_loss_pp"] <= REPEAT_ALLOWANCE_PP + 1e-10 for r in repeats)),
            "mean_validation_loss_pp": mean_loss,
            "mean_validation_positive_harm_pp": float(np.mean([r["positive_harm_pp"] for r in repeats])),
            "mean_validation_tokens": float(np.mean([r["retained_visual_tokens"] for r in repeats])),
            "mean_retention_span_pp": float(np.mean([abs(f["selected_spec"]["above_keep_percent"] - f["selected_spec"]["below_keep_percent"])
                                                     for f in folds if f["selected_spec"]])) if supported else None,
            "repeats": repeats, "folds": folds,
            "independent_groups": len({r["group_id"] for r in rows}),
            "evidence": "repeated nested development selection; repeats are not independent observations"}


def fit_shortlist(rows, *, identity=None, min_groups=MIN_GROUPS):
    """Freeze all fixed rules plus a CV-selected adaptive base and gentler lifts."""
    rows, identity = _rows(rows, fit=True), old._identity(identity)
    if min_groups != MIN_GROUPS:
        raise ValueError("v9 predeclares 20 independent groups per branch")
    specs = [{"kind": "fixed", "threshold_quantile": None, "threshold": None,
              "below_keep_percent": p, "above_keep_percent": p, "direction": 0}
             for p in KEEP_PERCENTAGES]
    procedures = [_validate_procedure(rows, q, direction, identity, min_groups)
                  for q in QUANTILES for direction in (1, -1)]
    eligible = sorted((d for d in procedures if d["eligible"]),
                      key=lambda d: (d["mean_validation_tokens"], d["mean_validation_positive_harm_pp"],
                                     d["mean_retention_span_pp"], d["procedure_id"]))
    selected = None
    for diagnostic in eligible:
        base = _fit_procedure(rows, diagnostic["threshold_quantile"], diagnostic["direction"], min_groups)
        if base is None:
            continue
        selected = {"procedure_id": diagnostic["procedure_id"], "policy_id": _spec_id(base),
                    "validation": {k: v for k, v in diagnostic.items() if k not in {"folds", "repeats"}}}
        for lift in range(0, 80, 10):
            below, above = (min(100, base[key] + lift) for key in ("below_keep_percent", "above_keep_percent"))
            if below == above:
                break  # Structural full already occurs in the fixed list.
            specs.append({**base, "below_keep_percent": below, "above_keep_percent": above,
                          "base_policy_id": _spec_id(base), "gentler_lift_pp": lift,
                          "fit_summary": _summary(rows, [physical_count_menu(r["input_visual_tokens"])[above if _entropy(r) >= base["threshold"] else below]
                                                         for r in rows])})
        break
    policies = [_policy(spec, rows, identity, min_groups) for spec in specs]
    if len({_spec_id(p) for p in policies}) != len(policies) or len(policies) > 15:
        raise AssertionError("shortlist deduplication or bound violated")
    return old._seal({"format": FORMAT + "_shortlist", "identity": identity,
                      "policies": policies, "selected_adaptive": selected,
                      "diagnostics": {"procedures": procedures, "repeat_seeds": list(REPEAT_SEEDS),
                                      "folds_per_repeat": 5, "maximum_fit_adaptive_templates": 168,
                                      "candidate_selection_inside_training_folds": True,
                                      "quantile_thresholds_inside_training_folds": True},
                      "fit_group_ids": sorted({r["group_id"] for r in rows}),
                      "source_sha256": sources(), "evidence": "development only"})


def _adjusted_bound(candidate, full, *, candidate_count, seed=BOOTSTRAP_SEED):
    metric = next(iter(candidate.values()))["quality_metric"]
    groups, strata = report._clusters(candidate, full, metric)
    alpha = .05 / candidate_count
    losses = {key: full[key]["quality_score"] - candidate[key]["quality_score"] for key in candidate}
    base = {"candidate_count": candidate_count, "alpha": alpha, "confidence": 1 - alpha,
            "interval_sides": "upper_one_sided", "multiplicity": "Bonferroni across frozen compressed candidates",
            "independent_groups": len(groups), "bootstrap_repetitions": BOOTSTRAP_REPETITIONS,
            "bootstrap_seed": seed, "upper_loss_pp": None, "uncertainty_is_approximate": True}
    if all(value == 0 for value in losses.values()):
        if metric == "mme":
            bounds = {cat: -math.expm1(math.log(alpha / len(strata)) / len(values))
                      for cat, values in strata.items()}
            return {**base, "status": "computed", "upper_loss_pp": 100 * float(np.mean(list(bounds.values()))),
                    "bound_method": "zero_harm_category_Bonferroni", "category_upper_bounds": bounds,
                    "uncertainty_is_approximate": False}
        if len({len(group) for group in groups.values()}) != 1:
            return {**base, "status": "zero_difference_variable_group_sizes_inconclusive"}
        return {**base, "status": "computed", "upper_loss_pp": -100 * math.expm1(math.log(alpha) / len(groups)),
                "bound_method": "zero_harm_exact_binomial", "uncertainty_is_approximate": False}
    rng = np.random.default_rng(seed)
    estimates = np.zeros(BOOTSTRAP_REPETITIONS)
    if metric == "mme":
        if any(len(values) < 2 for values in strata.values()):
            return {**base, "status": "insufficient_image_pairs_per_category"}
        for _, values in sorted(strata.items()):
            values = np.asarray(values)
            for start in range(0, len(estimates), 128):
                stop = min(start + 128, len(estimates))
                draws = rng.integers(len(values), size=(stop-start, len(values)))
                estimates[start:stop] += values[draws].mean(axis=1) / len(strata)
    else:
        if len(groups) < 2:
            return {**base, "status": "insufficient_independent_groups"}
        sums = np.asarray([sum(losses[key] for key in requests) for _, requests in sorted(groups.items())])
        sizes = np.asarray([len(requests) for _, requests in sorted(groups.items())])
        for start in range(0, len(estimates), 128):
            stop = min(start + 128, len(estimates))
            draws = rng.integers(len(groups), size=(stop-start, len(groups)))
            estimates[start:stop] = sums[draws].sum(axis=1) / sizes[draws].sum(axis=1)
    if np.ptp(estimates) <= 1e-12:
        return {**base, "status": "degenerate_cluster_losses_inconclusive"}
    return {**base, "status": "computed", "upper_loss_pp": float(100*np.quantile(estimates, 1-alpha)),
            "bound_method": "paired_cluster_percentile_bootstrap_approximate"}


def compare_to_full(candidate_rows, full_rows, *, expected_request_ids=None, candidate_count=1,
                    evidence="development", confirmation_audit=None, structural_full=False):
    """v8 comparison interface with explicitly candidate-adjusted uncertainty."""
    if type(candidate_count) is not int or candidate_count < 1:
        raise ValueError("candidate_count must be a positive integer")
    result = report.compare_to_full(candidate_rows, full_rows, expected_request_ids=expected_request_ids,
                                    evidence=evidence, confirmation_audit=confirmation_audit,
                                    structural_full=structural_full)
    if not result["cohort_complete"]:
        return {**result, "format": FORMAT + "_comparison"}
    candidate, full = report._index(candidate_rows), report._index(full_rows)
    bound = ({"status": "computed", "upper_loss_pp": 0., "candidate_count": candidate_count,
              "bound_method": "structural_identity", "uncertainty_is_approximate": False}
             if structural_full else _adjusted_bound(candidate, full, candidate_count=candidate_count))
    numeric = (bound["status"] == "computed" and result["mean_loss_pp"] <= 1.
               and bound["upper_loss_pp"] < 1.)
    accepted = numeric and result["confirmation_audit_passed"]
    positive_harm = 100 * float(np.mean([max(0., full[k]["quality_score"] - candidate[k]["quality_score"])
                                       for k in candidate]))
    return {**result, **bound, "format": FORMAT + "_comparison",
            "status": bound["status"] if evidence == "development" or result["confirmation_audit_passed"] else "unverified_confirmation",
            "uncertainty_status": bound["status"], "passes_numeric_rule": bool(numeric),
            "accepted": bool(accepted), "confirmed": bool(accepted), "positive_harm_pp": positive_harm}


def calibrate_shortlist(shortlist, candidate_rows_by_policy_id, full_rows, *, expected_request_ids=None,
                        calibration_audit=None):
    """Choose minimum tokens among independently calibrated frozen candidates.

    Calibration selects a rule; it is NOT the separate final confirmation set.
    Missing comparisons retain their place in the multiplicity denominator.
    """
    shortlist = old._checked(shortlist)
    if shortlist.get("format") != FORMAT + "_shortlist":
        raise ValueError("not a frozen v9 shortlist")
    policies = {p["policy_id"]: load_policy(p) for p in shortlist["policies"]}
    if set(candidate_rows_by_policy_id) - policies.keys():
        raise ValueError("unfrozen calibration candidate")
    if any(row["group_id"] in set(shortlist["fit_group_ids"]) for row in full_rows):
        raise ValueError("calibration identities overlap development")
    count, comparisons = max(1, len(policies)-1), {}
    for key, policy in policies.items():
        is_full = key == "fixed-100"
        rows = full_rows if is_full else candidate_rows_by_policy_id.get(key, [])
        comparison = compare_to_full(rows, full_rows, expected_request_ids=expected_request_ids,
                                     candidate_count=count, evidence="confirmation",
                                     confirmation_audit=calibration_audit, structural_full=is_full)
        comparisons[key] = {**comparison, "evidence": "calibration", "confirmed": False,
                            "evidence_label": "independent calibration; separate final test required"}
    eligible = [(key, c) for key, c in comparisons.items() if key != "fixed-100" and c["accepted"]]
    selected = (min(eligible, key=lambda item: (item[1]["memory"]["aggregate_retained_visual_tokens"],
                                               item[1]["positive_harm_pp"], policies[item[0]]["kind"] != "fixed", item[0]))[0]
                if eligible else "fixed-100")
    return old._seal({"format": FORMAT + "_calibration", "shortlist_sha256": shortlist["sha256"],
                      "candidate_count": count, "comparisons": comparisons, "selected_policy_id": selected,
                      "selected_policy": policies[selected], "compressed_calibrated": selected != "fixed-100",
                      "calibration_group_ids": sorted({r["group_id"] for r in full_rows}),
                      "selection_rule": "minimum exact tokens; positive harm; fixed simplicity; lexical ID",
                      "evidence": "independent calibration, not final confirmation"})
