"""Pooled, single-scalar compression policies; frozen v9 remains unchanged.

One model has ONE threshold and ONE retention pair for all four benchmarks.
Every fixed/two-level template is evaluated directly in grouped out-of-fold
development predictions. Development selection is not independent confirmation.
Only the short, frozen shortlist reaches independent calibration.

This module consumes already audited rows. It performs no file/data reads
except hashing source files, and never launches inference or changes a scorer.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
import copy
import hashlib
import math
from pathlib import Path
import re

import numpy as np

from . import compression_policy_v8 as old
from . import compression_policy_v9 as previous
from . import compression_report_v8 as report


FORMAT = "mlsys_compression_policy_v10"
DATASETS = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
CONFIRMATION_DATASETS = ("TextVQA", "ChartQA_TEST")
STATISTICS = ("normalized_entropy", "query_weighted_coverage")
KEEP_PERCENTAGES = tuple(range(30, 101, 10))
EXPANDED_KEEP_PERCENTAGES = tuple(range(10, 101, 10))
QUANTILES = (.25, .5, .75)
REPEAT_SEEDS = (20261020, 20261021, 20261022)
MIN_GROUPS = 20
MEAN_LOSS_LIMIT_PP = .5
REPEAT_LOSS_LIMIT_PP = 1.
TOKEN_ALLOWANCE = .02
CALIBRATION_COMPARISONS = 12
FINAL_COMPARISONS = 4
EXPECTED_FRESH_GROUPS = 1000
_FEATURES = set(old.FEATURE_NAMES) | {"query_weighted_coverage"}
_GEOMETRY = {"row_hash", "group_id", "input_visual_tokens", "input_text_tokens"}
_OPTIONAL = {"dataset", "quality_metric", "category", "model_tag",
             "checkpoint_sha256", "native_supported"}
_METRICS = {"DocVQA": "anls", "TextVQA": "vqa", "ChartQA_TEST": "relaxed", "MME": "mme"}
_ALIASES = {
    "DocVQA": "DocVQA", "docvqa": "DocVQA", "pixparse/docvqa-single-page-questions": "DocVQA",
    "TextVQA": "TextVQA", "textvqa": "TextVQA", "lmms-lab/textvqa": "TextVQA",
    "ChartQA": "ChartQA_TEST", "chartqa": "ChartQA_TEST", "ChartQA_TEST": "ChartQA_TEST",
    "chartqa_test": "ChartQA_TEST", "lmms-lab/ChartQA": "ChartQA_TEST",
    "MME": "MME", "mme": "MME", "lmms-lab/MME": "MME",
}


def sources():
    return {str(Path(p).resolve()): hashlib.sha256(Path(p).read_bytes()).hexdigest()
            for p in (__file__, old.__file__, previous.__file__, report.__file__)}


def canonical_dataset(value):
    if not isinstance(value, str) or value not in _ALIASES:
        raise ValueError("unknown v10 development/confirmation dataset")
    return _ALIASES[value]


def _menu(keep_percentages):
    menu = tuple(keep_percentages)
    if (menu not in (KEEP_PERCENTAGES, EXPANDED_KEEP_PERCENTAGES)
            or any(type(p) is not int for p in menu)):
        raise ValueError("retention menu must be 30..100 or 10..100 by 10")
    return menu


def physical_count_menu(input_visual_tokens, keep_percentages=KEEP_PERCENTAGES):
    n = old._integer(input_visual_tokens, "input_visual_tokens", 1)
    return {p: (n*p + 99)//100 for p in _menu(keep_percentages)}


def _scalar(value):
    value = old._finite(value, "statistic")
    if not 0 <= value <= 1:
        raise ValueError("statistic outside its mathematical [0,1] range")
    return value


def _identity(identity):
    value = old._identity(identity)
    old._name(value.get("model"), "model")
    if not isinstance(value.get("checkpoint_sha256"), str) or not re.fullmatch(
            "[0-9a-f]{64}", value["checkpoint_sha256"]):
        raise ValueError("identity requires the frozen checkpoint_sha256")
    value.setdefault("split", "fit")
    return value


def _rows(rows, *, fit=False, statistic="normalized_entropy",
          keep_percentages=KEEP_PERCENTAGES):
    if statistic not in (*STATISTICS, "none"):
        raise ValueError("unknown single statistic")
    menu = _menu(keep_percentages)
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("rows must be a nonempty list or tuple")
    result, seen = [], set()
    for raw in rows:
        allowed = _GEOMETRY | _OPTIONAL | {"benefit_features"} | ({"quality_by_count"} if fit else set())
        if not isinstance(raw, Mapping) or not _GEOMETRY <= set(raw) or set(raw) - allowed:
            raise ValueError("only declared fit/answer-free fields are allowed")
        row = {key: old._name(raw[key], key) for key in ("row_hash", "group_id")}
        if row["row_hash"] in seen:
            raise ValueError("duplicate row identity")
        seen.add(row["row_hash"])
        row["input_visual_tokens"] = old._integer(raw["input_visual_tokens"], "input_visual_tokens", 1)
        row["input_text_tokens"] = old._integer(raw["input_text_tokens"], "input_text_tokens")
        for key in _OPTIONAL:
            if key in raw:
                row[key] = copy.deepcopy(raw[key])
        if "native_supported" in row and type(row["native_supported"]) is not bool:
            raise ValueError("native_supported must be boolean")
        features = raw.get("benefit_features")
        if isinstance(features, Mapping) and set(features) - _FEATURES:
            raise ValueError("undeclared benefit feature; labels are forbidden")
        value = features.get(statistic) if isinstance(features, Mapping) else None
        try:
            value = _scalar(value) if statistic != "none" else None
        except (TypeError, ValueError):
            if fit:
                raise ValueError("fit requires a valid selected scalar") from None
            value = None
        row["benefit_features"] = {statistic: value} if statistic != "none" else {}
        if fit:
            row["dataset"] = canonical_dataset(raw.get("dataset"))
            if raw.get("quality_metric") != _METRICS[row["dataset"]]:
                raise ValueError("dataset/native metric mismatch")
            quality = raw.get("quality_by_count")
            expected = {str(k) for k in physical_count_menu(row["input_visual_tokens"], menu).values()}
            if not isinstance(quality, Mapping) or set(quality) != expected:
                raise ValueError("all and only exact physical menu count qualities are required")
            row["quality_by_count"] = {key: old._finite(value, "quality") for key, value in quality.items()}
            if any(not 0 <= v <= 1 for v in row["quality_by_count"].values()):
                raise ValueError("quality must be a fraction in [0,1]")
            if raw.get("native_supported", True) is not True:
                raise ValueError("development geometry must be natively supported")
        result.append(row)
    result.sort(key=lambda r: r["row_hash"])
    if fit:
        if {r["dataset"] for r in result} != set(DATASETS):
            raise ValueError("pooled fitting requires all four datasets")
        mme = [r for r in result if r["dataset"] == "MME"]
        groups = defaultdict(list)
        for row in mme:
            if not row.get("category") or any(v not in (0., 1.) for v in row["quality_by_count"].values()):
                raise ValueError("MME requires categories and binary physical labels")
            groups[row["group_id"]].append(row)
        if any(len(pair) != 2 or len({r["category"] for r in pair}) != 1 for pair in groups.values()):
            raise ValueError("MME requires complete two-question image groups")
    return result


def grouped_folds(rows, seed):
    """Keep global image/document groups intact, even across dataset aliases."""
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group, members in groups.items():
        stratum = tuple(sorted({(canonical_dataset(r["dataset"]),
                                 r.get("category") if canonical_dataset(r["dataset"]) == "MME" else "")
                                for r in members}))
        strata[stratum].append(group)
    result, loads = {}, [0]*5
    for stratum, members in sorted(strata.items()):
        ordered = sorted(members, key=lambda g: hashlib.sha256(
            f"{seed}|{stratum}|{g}".encode()).hexdigest())
        offset = min(range(5), key=lambda f: (loads[f], f))
        for index, group in enumerate(ordered):
            fold = (offset + index) % 5
            result[group] = fold
            loads[fold] += 1
    return result


def weighted_threshold(rows, statistic, quantile, *, min_groups=MIN_GROUPS):
    """Inverse-CDF quantile: each benchmark has total weight one quarter."""
    if quantile not in QUANTILES or min_groups != MIN_GROUPS:
        raise ValueError("v10 fixes three quantiles and 20 pooled branch groups")
    sizes = {d: sum(r["dataset"] == d for r in rows) for d in DATASETS}
    if not all(sizes.values()):
        return None
    values = sorted((_scalar(r["benefit_features"][statistic]), r["row_hash"],
                     1. / (len(DATASETS)*sizes[r["dataset"]])) for r in rows)
    cumulative, threshold = 0., values[-1][0]
    for value, _, weight in values:
        cumulative += weight
        if cumulative + 1e-12 >= quantile:
            threshold = value
            break
    branches = [{r["group_id"] for r in rows
                 if (_scalar(r["benefit_features"][statistic]) >= threshold) == side}
                for side in (False, True)]
    return threshold if min(map(len, branches)) >= min_groups else None


def _native_losses(rows, selected):
    full = [r["quality_by_count"][str(r["input_visual_tokens"])] for r in rows]
    actual = [r["quality_by_count"][str(k)] for r, k in zip(rows, selected)]
    if rows[0]["quality_metric"] != "mme":
        losses = np.asarray(full) - actual
        weights = np.full(len(rows), 1./len(rows))
    else:
        pairs = defaultdict(list)
        for row, baseline, value in zip(rows, full, actual):
            pairs[(row["category"], row["group_id"])].append((baseline, value))
        categories = defaultdict(int)
        for category, _ in pairs:
            categories[category] += 1
        losses, weights = [], []
        for (category, _), pair in sorted(pairs.items()):
            if len(pair) != 2:
                raise ValueError("native MME comparison requires intact image pairs")
            a, b = zip(*pair)
            losses.append(.5*(np.mean(a)-np.mean(b)) + .5*(int(all(a))-int(all(b))))
            weights.append(1./(len(categories)*categories[category]))
        losses, weights = np.asarray(losses), np.asarray(weights)
    return {"mean_loss_pp": float(100*np.dot(weights, losses)),
            "positive_harm_pp": float(100*np.dot(weights, np.maximum(losses, 0))),
            "pruning_gain_pp": float(100*np.dot(weights, np.maximum(-losses, 0)))}


def _summary(rows, selected):
    by_domain = {}
    for dataset in DATASETS:
        indices = [i for i, r in enumerate(rows) if r["dataset"] == dataset]
        domain = [rows[i] for i in indices]
        counts = [int(selected[i]) for i in indices]
        by_domain[dataset] = {
            **_native_losses(domain, counts), "requests": len(domain),
            "independent_groups": len({r["group_id"] for r in domain}),
            "retained_visual_tokens": sum(counts),
            "full_visual_tokens": sum(r["input_visual_tokens"] for r in domain),
            "retained_visual_tokens_per_request": float(np.mean(counts)),
            "full_visual_tokens_per_request": float(np.mean([r["input_visual_tokens"] for r in domain])),
        }
    cost = float(np.mean([r["retained_visual_tokens_per_request"] for r in by_domain.values()]))
    full = float(np.mean([r["full_visual_tokens_per_request"] for r in by_domain.values()]))
    harm = float(np.mean([r["positive_harm_pp"] for r in by_domain.values()]))
    gain = float(np.mean([r["pruning_gain_pp"] for r in by_domain.values()]))
    return {"domains": by_domain, "cost": cost, "full_cost": full, "retained_ratio": cost/full,
            "positive_harm_pp": harm, "pruning_gain_pp": gain, "harm_score_pp": harm-.5*gain,
            "retained_visual_tokens": sum(map(int, selected)),
            "full_visual_tokens": sum(r["input_visual_tokens"] for r in rows)}


def _spec_id(spec):
    if spec["kind"] == "fixed":
        return f"fixed-{spec['below_keep_percent']:03d}"
    prefix = "entropy" if spec["statistic"] == "normalized_entropy" else "coverage"
    return (f"{prefix}-q{round(100*spec['threshold_quantile']):02d}"
            f"-below{spec['below_keep_percent']:03d}-above{spec['above_keep_percent']:03d}")


def candidate_specs(statistic="normalized_entropy", keep_percentages=KEEP_PERCENTAGES):
    if statistic not in STATISTICS:
        raise ValueError("unknown statistic")
    menu = _menu(keep_percentages)
    result = [{"kind": "fixed", "statistic": "none", "threshold_quantile": None,
               "threshold": None, "below_keep_percent": p, "above_keep_percent": p, "direction": 0}
              for p in menu]
    result += [{"kind": "adaptive", "statistic": statistic, "threshold_quantile": q,
                "threshold": None, "below_keep_percent": below, "above_keep_percent": above,
                "direction": 1 if above > below else -1}
               for q in QUANTILES for below in menu for above in menu if below != above]
    return result


def _counts(rows, spec, thresholds=None):
    values = []
    for i, row in enumerate(rows):
        threshold = thresholds[i] if thresholds is not None else spec["threshold"]
        percent = spec["below_keep_percent"]
        if spec["kind"] == "adaptive":
            percent = (100 if threshold is None else spec["above_keep_percent"]
                       if row["benefit_features"][spec["statistic"]] >= threshold else percent)
        values.append((row["input_visual_tokens"]*percent + 99)//100)
    return values


def _simplicity(item):
    spec = item.get("spec", item)
    return (spec["kind"] != "fixed", abs(spec["above_keep_percent"]-spec["below_keep_percent"]))


def robust_select(candidates):
    """Two-point token allowance is relative to the same full-token workload."""
    values = list(candidates)
    if not values:
        return None
    full = values[0]["full_cost"]
    if full <= 0 or any(not math.isclose(v["full_cost"], full, rel_tol=0, abs_tol=1e-8) for v in values):
        raise ValueError("robust selection requires one common full-token workload")
    minimum = min(v["cost"] for v in values)
    nearby = [v for v in values if v["cost"] <= minimum + TOKEN_ALLOWANCE*full + 1e-10]
    return min(nearby, key=lambda v: (v["harm_score_pp"], v["cost"], v["positive_harm_pp"],
                                      _simplicity(v), v["policy_id"]))


def _policy(candidate, rows, identity, menu):
    return old._seal({
        "format": FORMAT, **candidate["spec"], "policy_id": candidate["policy_id"],
        "identity": identity, "model": identity["model"], "checkpoint_sha256": identity["checkpoint_sha256"],
        "datasets": list(DATASETS), "dataset_is_prediction_input": False, "keep_percentages": list(menu),
        "minimum_branch_groups": MIN_GROUPS,
        "fit_group_ids": sorted({r["group_id"] for r in rows}),
        "fit_row_hashes": [r["row_hash"] for r in rows], "fit_data_sha256": old._digest(rows),
        "validation": {k: v for k, v in candidate.items() if k not in {"repeats", "spec"}},
        "source_sha256": sources(), "evidence": "development selection; independent calibration required",
    })


def fit_shortlist(rows, *, identity=None, statistic="normalized_entropy",
                  keep_percentages=KEEP_PERCENTAGES, min_groups=MIN_GROUPS):
    """Directly validate all 176/280 settings before choosing at most three."""
    if min_groups != MIN_GROUPS:
        raise ValueError("v10 requires 20 independent groups per pooled branch")
    identity, menu = _identity(identity), _menu(keep_percentages)
    rows = _rows(rows, fit=True, statistic=statistic, keep_percentages=menu)
    for row in rows:
        if "model_tag" in row and row["model_tag"] != identity["model"]:
            raise ValueError("development model identity mismatch")
        if "checkpoint_sha256" in row and row["checkpoint_sha256"] != identity["checkpoint_sha256"]:
            raise ValueError("development checkpoint identity mismatch")
    thresholds, folds = {}, []
    for seed in REPEAT_SEEDS:
        partition = grouped_folds(rows, seed)
        for quantile in QUANTILES:
            per_fold = {}
            for fold in range(5):
                train = [r for r in rows if partition[r["group_id"]] != fold]
                valid = [r for r in rows if partition[r["group_id"]] == fold]
                value = weighted_threshold(train, statistic, quantile) if train and valid else None
                per_fold[fold] = value
                folds.append({"seed": seed, "fold": fold, "quantile": quantile, "threshold": value,
                              "fit_group_ids": sorted({r["group_id"] for r in train}),
                              "validation_group_ids": sorted({r["group_id"] for r in valid})})
            thresholds[(seed, quantile)] = [per_fold[partition[r["group_id"]]] for r in rows]
    final_thresholds = {q: weighted_threshold(rows, statistic, q) for q in QUANTILES}
    diagnostics = []
    for spec in candidate_specs(statistic, menu):
        spec = {**spec, "threshold": final_thresholds.get(spec["threshold_quantile"])}
        supported = spec["kind"] == "fixed" or (spec["threshold"] is not None and all(
            t is not None for seed in REPEAT_SEEDS for t in thresholds[(seed, spec["threshold_quantile"])]))
        repeats = []
        for seed in REPEAT_SEEDS:
            values = thresholds[(seed, spec["threshold_quantile"])] if spec["kind"] == "adaptive" else None
            repeats.append({"seed": seed, **_summary(rows, _counts(rows, spec, values))})
        domains = {}
        for dataset in DATASETS:
            summaries = [r["domains"][dataset] for r in repeats]
            domains[dataset] = {key: float(np.mean([r[key] for r in summaries])) for key in (
                "mean_loss_pp", "positive_harm_pp", "pruning_gain_pp",
                "retained_visual_tokens_per_request", "full_visual_tokens_per_request",
                "retained_visual_tokens", "full_visual_tokens")}
            domains[dataset].update(requests=summaries[0]["requests"],
                                    independent_groups=summaries[0]["independent_groups"],
                                    worst_repeat_loss_pp=max(r["mean_loss_pp"] for r in summaries))
        eligible = supported and all(d["mean_loss_pp"] <= MEAN_LOSS_LIMIT_PP+1e-10
                                     and d["worst_repeat_loss_pp"] <= REPEAT_LOSS_LIMIT_PP+1e-10
                                     for d in domains.values())
        item = {"policy_id": _spec_id(spec), "spec": spec, "supported": bool(supported),
                "eligible": bool(eligible), "per_domain": domains, "repeats": repeats,
                "fit_summary": _summary(rows, _counts(rows, spec)),
                **{key: float(np.mean([r[key] for r in repeats])) for key in (
                    "cost", "full_cost", "retained_ratio", "positive_harm_pp",
                    "pruning_gain_pp", "harm_score_pp", "retained_visual_tokens", "full_visual_tokens")}}
        diagnostics.append(item)
    by_id = {d["policy_id"]: d for d in diagnostics}
    fixed = robust_select(d for d in diagnostics if d["eligible"] and d["spec"]["kind"] == "fixed")
    adaptive = robust_select(d for d in diagnostics if d["eligible"] and d["spec"]["kind"] == "adaptive")
    selected = [fixed]
    gentle = None
    if adaptive is not None:
        selected.append(adaptive)
        for lift in range(10, 100, 10):
            spec = {**adaptive["spec"],
                    "below_keep_percent": min(100, adaptive["spec"]["below_keep_percent"]+lift),
                    "above_keep_percent": min(100, adaptive["spec"]["above_keep_percent"]+lift)}
            if spec["below_keep_percent"] == spec["above_keep_percent"]:
                break
            value = by_id[_spec_id(spec)]
            if value["eligible"]:
                gentle = value
                selected.append(gentle)
                break
    selected.append(by_id["fixed-100"])
    selected = list({d["policy_id"]: d for d in selected if d is not None}.values())
    policies = [_policy(d, rows, identity, menu) for d in selected]
    if len([p for p in policies if p["policy_id"] != "fixed-100"]) > 3:
        raise AssertionError("too many frozen compressed candidates")
    return old._seal({
        "format": FORMAT+"_shortlist", "identity": identity, "statistic": statistic,
        "keep_percentages": list(menu), "datasets": list(DATASETS), "policies": policies,
        "selected_fixed": fixed["policy_id"], "selected_adaptive": adaptive["policy_id"] if adaptive else None,
        "selected_gentler": gentle["policy_id"] if gentle else None,
        "fit_group_ids": sorted({r["group_id"] for r in rows}), "fit_row_hashes": [r["row_hash"] for r in rows],
        "diagnostics": {"candidates": diagnostics, "folds": folds, "repeat_seeds": list(REPEAT_SEEDS),
                        "folds_per_repeat": 5, "templates": len(diagnostics),
                        "all_conservative_templates_validated_directly": True,
                        "selection_is_development_only": True,
                        "independent_groups": len({r["group_id"] for r in rows})},
        "token_allowance_fraction_of_full": TOKEN_ALLOWANCE,
        "source_sha256": sources(), "evidence": "development selection, not independent confirmation",
    })


def load_policy(policy):
    policy = old._checked(policy)
    if policy.get("format") != FORMAT or policy.get("kind") not in {"fixed", "adaptive"}:
        raise ValueError("not a v10 policy")
    identity = _identity(policy["identity"])
    if policy.get("model") != identity["model"] or policy.get("checkpoint_sha256") != identity["checkpoint_sha256"]:
        raise ValueError("policy identity mismatch")
    menu = _menu(policy["keep_percentages"])
    if (set(policy.get("datasets", [])) != set(DATASETS) or policy.get("dataset_is_prediction_input") is not False
            or any(type(policy.get(k)) is not int or policy[k] not in menu
                   for k in ("below_keep_percent", "above_keep_percent"))):
        raise ValueError("invalid pooled retention policy")
    if policy["kind"] == "fixed":
        if (policy["below_keep_percent"] != policy["above_keep_percent"] or policy["statistic"] != "none"
                or policy.get("threshold") is not None or policy.get("threshold_quantile") is not None
                or type(policy.get("direction")) is not int or policy["direction"] != 0):
            raise ValueError("invalid fixed policy")
    elif (policy.get("statistic") not in STATISTICS or policy.get("threshold_quantile") not in QUANTILES
          or policy["below_keep_percent"] == policy["above_keep_percent"]):
        raise ValueError("invalid adaptive policy")
    else:
        _scalar(policy.get("threshold"))
        direction = 1 if policy["above_keep_percent"] > policy["below_keep_percent"] else -1
        if type(policy.get("direction")) is not int or policy["direction"] != direction:
            raise ValueError("adaptive direction disagrees with retained counts")
    if policy.get("minimum_branch_groups") != MIN_GROUPS:
        raise ValueError("invalid frozen branch support")
    if policy.get("policy_id") != _spec_id(policy):
        raise ValueError("policy identifier mismatch")
    return policy


def predict_counts(policy, answer_free_rows, *, allow_development_overlap=False):
    """No benchmark-name branch, empirical range gate, labels, or scorer change."""
    if type(allow_development_overlap) is not bool:
        raise ValueError("development overlap permission must be explicit boolean")
    policy = load_policy(policy)
    rows = _rows(answer_free_rows, statistic=policy["statistic"], keep_percentages=policy["keep_percentages"])
    fitted_groups, fitted_ids = set(policy["fit_group_ids"]), set(policy["fit_row_hashes"])
    result = {}
    for row in rows:
        if not allow_development_overlap and (row["group_id"] in fitted_groups or row["row_hash"] in fitted_ids):
            raise ValueError("prediction identities overlap development")
        if row.get("model_tag", policy["model"]) != policy["model"]:
            raise ValueError("model identity mismatch")
        if row.get("checkpoint_sha256", policy["checkpoint_sha256"]) != policy["checkpoint_sha256"]:
            raise ValueError("checkpoint identity mismatch")
        reasons, value = [], None
        if row.get("native_supported", True) is not True:
            reasons.append("native_geometry_unsupported")
        if policy["kind"] == "adaptive":
            value = row["benefit_features"].get(policy["statistic"])
            if value is None:
                reasons.append("missing_or_invalid_statistic")
        percent = (100 if reasons else policy["above_keep_percent"]
                   if value is not None and value >= policy["threshold"] else policy["below_keep_percent"])
        count = (row["input_visual_tokens"]*percent+99)//100
        result[row["row_hash"]] = {**row, "selected_count": count, "keep_percent": percent,
                                  "statistic_value": value, "fallback": bool(reasons),
                                  "fallback_reasons": reasons,
                                  "selector": "full" if count == row["input_visual_tokens"] else "scorer"}
    return old._seal({
        "format": FORMAT+"_assignments", "policy_sha256": policy["sha256"], "rows": result,
        "counts": {key: r["selected_count"] for key, r in result.items()},
        "total_visual_tokens": sum(r["selected_count"] for r in result.values()),
        "full_visual_tokens": sum(r["input_visual_tokens"] for r in rows),
        "full_fallback_rows": sum(r["fallback"] for r in result.values()),
        "dataset_is_prediction_input": False, "development_overlap_permitted": allow_development_overlap,
        "evidence": "development only" if allow_development_overlap else "answer-free assignment, not a quality claim",
    })


def select_shared_statistic(shortlists_by_statistic):
    """Select ONE formula across models, never reward mere adaptive eligibility."""
    if not isinstance(shortlists_by_statistic, Mapping) or not shortlists_by_statistic:
        raise ValueError("shared selection requires frozen model shortlists")
    model_sets = [set(values) for values in shortlists_by_statistic.values()]
    if not model_sets[0] or any(values != model_sets[0] for values in model_sets):
        raise ValueError("every statistic must cover the identical model set")
    choices, signatures = [], {}
    for statistic, values in sorted(shortlists_by_statistic.items()):
        if statistic not in STATISTICS:
            raise ValueError("unknown shared statistic")
        per_model = {}
        for model, raw in sorted(values.items()):
            shortlist = old._checked(raw)
            if shortlist.get("format") != FORMAT+"_shortlist" or shortlist.get("statistic") != statistic:
                raise ValueError("wrong shared-selection shortlist")
            if shortlist["identity"]["model"] != model:
                raise ValueError("shared-selection model identity mismatch")
            signature = (shortlist["identity"]["checkpoint_sha256"], tuple(shortlist["fit_group_ids"]),
                         tuple(shortlist["fit_row_hashes"]), tuple(shortlist["keep_percentages"]))
            if model in signatures and signatures[model] != signature:
                raise ValueError("statistic comparison changes model/cohort/retention menu")
            signatures[model] = signature
            policies = {p["policy_id"]: load_policy(p) for p in shortlist["policies"]}
            fixed = policies[shortlist["selected_fixed"]]["validation"]
            adaptive = policies[shortlist["selected_adaptive"]]["validation"] if shortlist["selected_adaptive"] else None
            saving = (fixed["cost"]-adaptive["cost"])/fixed["full_cost"] if adaptive else 0.
            per_model[model] = {"adaptive_improves_compression": bool(adaptive and saving > 1e-10),
                                "saving_fraction_of_full": float(saving),
                                "selected_fixed": shortlist["selected_fixed"],
                                "selected_adaptive": shortlist["selected_adaptive"],
                                "shortlist_sha256": shortlist["sha256"]}
        savings = [v["saving_fraction_of_full"] for v in per_model.values()]
        choices.append({"statistic": statistic, "models": per_model,
                        "models_with_adaptive_improvement": sum(v["adaptive_improves_compression"] for v in per_model.values()),
                        "worst_model_saving": min(savings), "mean_model_saving": float(np.mean(savings))})
    selected = min(choices, key=lambda c: (-c["models_with_adaptive_improvement"], -c["worst_model_saving"],
                                         -c["mean_model_saving"], c["statistic"] != "normalized_entropy",
                                         c["statistic"]))
    return old._seal({"format": FORMAT+"_shared_statistic", "selected_statistic": selected["statistic"],
                      "diagnostics": choices, "source_sha256": sources(), "evidence": "development selection only"})


def compare_to_full(candidate_rows, full_rows, *, expected_request_ids=None, comparison_count=CALIBRATION_COMPARISONS,
                    evidence="development", confirmation_audit=None, structural_full=False):
    """Candidate-adjusted paired bound; nonzero bootstrap remains approximate."""
    if type(comparison_count) is not int or comparison_count not in (CALIBRATION_COMPARISONS, FINAL_COMPARISONS):
        raise ValueError("v10 fixes 12 calibration or 4 final comparisons")
    result = previous.compare_to_full(candidate_rows, full_rows, expected_request_ids=expected_request_ids,
                                     candidate_count=comparison_count, evidence=evidence,
                                     confirmation_audit=confirmation_audit, structural_full=structural_full)
    if not result["cohort_complete"]:
        return {**result, "format": FORMAT+"_comparison", "comparison_count": comparison_count}
    candidate, full = report._index(candidate_rows), report._index(full_rows)
    paired = []
    for key in sorted(full):
        reference, actual = full[key], candidate[key]
        paired.append({"row_hash": key, "group_id": reference["group_id"],
                       "dataset": reference["dataset"], "quality_metric": reference["quality_metric"],
                       "category": reference.get("category"), "input_visual_tokens": reference["input_visual_tokens"],
                       "quality_by_count": {"full": reference["quality_score"], "selected": actual["quality_score"]}})
    # Native harm needs two separate values even when selected/full count aliases.
    if paired[0]["quality_metric"] == "mme":
        groups = defaultdict(list)
        for row in paired:
            groups[(row["category"], row["group_id"])].append(row)
        categories = defaultdict(list)
        for (category, _), group in groups.items():
            a = [r["quality_by_count"]["full"] for r in group]
            b = [r["quality_by_count"]["selected"] for r in group]
            categories[category].append(.5*(np.mean(a)-np.mean(b))+.5*(int(all(a))-int(all(b))))
        harm = 100*float(np.mean([np.mean(np.maximum(v, 0)) for v in categories.values()]))
        gain = 100*float(np.mean([np.mean(np.maximum(-np.asarray(v), 0)) for v in categories.values()]))
    else:
        losses = np.asarray([full[k]["quality_score"]-candidate[k]["quality_score"] for k in sorted(full)])
        harm, gain = 100*float(np.maximum(losses, 0).mean()), 100*float(np.maximum(-losses, 0).mean())
    return {**result, "format": FORMAT+"_comparison", "comparison_count": comparison_count,
            "multiplicity": f"Bonferroni over {comparison_count} predeclared model/dataset/candidate comparisons",
            "positive_harm_pp": harm, "pruning_gain_pp": gain, "harm_score_pp": harm-.5*gain}


def _confirmation_rows(rows):
    result = []
    for raw in rows:
        row = copy.deepcopy(raw)
        row["dataset"] = canonical_dataset(row["dataset"])
        if row["dataset"] not in CONFIRMATION_DATASETS or row["quality_metric"] != _METRICS[row["dataset"]]:
            raise ValueError("fresh v10 confirmation is restricted to TextVQA and ChartQA")
        result.append(row)
    report._index(result)
    return result


def calibrate_shortlist(shortlist, candidate_rows_by_policy_id, full_rows, *, expected_request_ids,
                        calibration_audit=None, expected_groups_per_dataset=EXPECTED_FRESH_GROUPS):
    """Joint TextVQA/ChartQA calibration, never final confirmation or label IO."""
    shortlist = old._checked(shortlist)
    if shortlist.get("format") != FORMAT+"_shortlist":
        raise ValueError("not a v10 shortlist")
    policies = {p["policy_id"]: load_policy(p) for p in shortlist["policies"]}
    if set(candidate_rows_by_policy_id)-policies.keys():
        raise ValueError("unfrozen calibration candidate")
    if len(policies)-1 > 3 or "fixed-100" not in policies:
        raise ValueError("invalid calibration shortlist")
    if type(expected_groups_per_dataset) is not int or expected_groups_per_dataset < 1:
        raise ValueError("expected group count must be a positive integer")
    if not isinstance(expected_request_ids, Mapping):
        raise ValueError("expected cohorts must be supplied per dataset")
    expected = {canonical_dataset(k): list(v) for k, v in expected_request_ids.items()}
    if set(expected) != set(CONFIRMATION_DATASETS) or len(expected_request_ids) != 2:
        raise ValueError("both predeclared TextVQA/ChartQA cohorts are required")
    ids = [key for values in expected.values() for key in values]
    if len(ids) != len(set(ids)) or any(not isinstance(k, str) or not k for k in ids):
        raise ValueError("fresh request identities must be globally unique")
    full_rows = _confirmation_rows(full_rows)
    if {r["request_id"] for r in full_rows} != set(ids):
        raise ValueError("full calibration cohort is incomplete")
    if any(report._model(r) != shortlist["identity"]["model"] for r in full_rows):
        raise ValueError("calibration model differs from frozen policy")
    fit_groups, fit_ids = set(shortlist["fit_group_ids"]), set(shortlist["fit_row_hashes"])
    if any(r["group_id"] in fit_groups or r["request_id"] in fit_ids for r in full_rows):
        raise ValueError("calibration overlaps development")
    full_by_domain = {d: [r for r in full_rows if r["dataset"] == d] for d in CONFIRMATION_DATASETS}
    for dataset, rows in full_by_domain.items():
        if ({r["request_id"] for r in rows} != set(expected[dataset])
                or len(rows) != expected_groups_per_dataset
                or len({r["group_id"] for r in rows}) != expected_groups_per_dataset):
            raise ValueError("calibration cohort/group count differs from predeclared cohort")
    if len({r["group_id"] for r in full_rows}) != len(full_rows):
        raise ValueError("calibration requires globally independent image groups")
    audits = {}
    for dataset, audit in (calibration_audit or {}).items():
        audits[canonical_dataset(dataset)] = audit
    comparisons, eligible = {}, []
    for key, policy in policies.items():
        structural = key == "fixed-100"
        rows = full_rows if structural else _confirmation_rows(candidate_rows_by_policy_id.get(key, []))
        by_domain = {}
        for dataset in CONFIRMATION_DATASETS:
            domain = [r for r in rows if r["dataset"] == dataset]
            value = compare_to_full(domain, full_by_domain[dataset], expected_request_ids=expected[dataset],
                                    comparison_count=CALIBRATION_COMPARISONS, evidence="confirmation",
                                    confirmation_audit=audits.get(dataset), structural_full=structural)
            by_domain[dataset] = {**value, "evidence": "calibration", "confirmed": False}
        accepted = all(v["accepted"] for v in by_domain.values())
        complete = all(v["cohort_complete"] for v in by_domain.values())
        entry = {"policy_id": key, "accepted": accepted, "cohort_complete": complete,
                 "per_domain": by_domain, "spec": {k: policy[k] for k in
                    ("kind", "below_keep_percent", "above_keep_percent")},
                 "compression_accepted": accepted and not structural,
                 "evidence": "independent calibration, not final confirmation"}
        if complete:
            entry.update(cost=float(np.mean([np.mean([r["retained_visual_tokens"] for r in rows
                                                       if r["dataset"] == d]) for d in CONFIRMATION_DATASETS])),
                         full_cost=float(np.mean([np.mean([r["input_visual_tokens"] for r in full_by_domain[d]])
                                                  for d in CONFIRMATION_DATASETS])),
                         positive_harm_pp=float(np.mean([v["positive_harm_pp"] for v in by_domain.values()])),
                         pruning_gain_pp=float(np.mean([v["pruning_gain_pp"] for v in by_domain.values()])))
            entry["harm_score_pp"] = entry["positive_harm_pp"]-.5*entry["pruning_gain_pp"]
        comparisons[key] = entry
        if accepted:
            eligible.append(entry)
    best = robust_select(eligible)
    selected = best["policy_id"] if best else "fixed-100"
    return old._seal({
        "format": FORMAT+"_calibration", "shortlist_sha256": shortlist["sha256"],
        "comparison_count": CALIBRATION_COMPARISONS, "comparisons": comparisons,
        "selected_policy_id": selected, "selected_policy": policies[selected],
        "compressed_calibrated": selected != "fixed-100",
        "calibration_group_ids": sorted({r["group_id"] for r in full_rows}),
        "calibration_request_ids": sorted(ids), "calibrated_datasets": list(CONFIRMATION_DATASETS),
        "development_only_datasets": ["DocVQA", "MME"],
        "token_allowance_fraction_of_full": TOKEN_ALLOWANCE,
        "source_sha256": sources(), "evidence": "independent calibration; separate final test required",
    })
