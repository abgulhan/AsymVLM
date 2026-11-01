"""Physical-only development diagnosis of selector quality and allocation headroom.

This module never reads confirmation labels, trains a selector, or launches GPU
work. Oracle assignments USE ANSWERS and are not deployable policies. A bounded
solution is called optimal only when its valid upper bound meets its incumbent.
Existing baseline outputs are selector adaptations, not official-pipeline runs.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from itertools import product
import math
from pathlib import Path

from . import compression_matrix_v8 as matrix
from . import compression_study_v8 as physical
from .followup_common import write_immutable
from .state import digest, file_digest

FORMAT = "mlsys_compression_diagnostics_v11"
DATASETS = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
DEFAULT_STUDY = Path("reproduction_results/mlsys_compression_20261020_v10/study.json")
DEFAULT_OUTPUT = Path("reproduction_results/mlsys_compression_20261020_v11/diagnostics")
SELECTOR_MENU = tuple(range(30, 100, 10))
LEARNED_MENU = tuple(range(10, 101, 10))
BASELINE_SCOPE = "projected_feature_selector_adaptations_not_official_pipelines"


def load_development(model, study_path=DEFAULT_STUDY):
    """Load only pinned v9/v10 fit/screening, including completed low-count runs."""
    from . import compression_study_v10 as previous
    from . import compression_execution_v10 as execution
    from .compression_policy_v10 import canonical_dataset
    if model not in ("qwen25", "llava_next"):
        raise ValueError("unsupported development model")
    study_path = Path(study_path).resolve()
    study = matrix.checked(study_path)
    previous.verify(study)
    cache = previous.cached_development(study, model)
    if cache.get("old_evaluation_opened") is not False:
        raise ValueError("confirmation/evaluation labels are forbidden")
    low = execution.collect(study, model)
    if low is None:
        raise ValueError("physical 10/20 percent development matrix is incomplete")
    rows = [{**r, "dataset": canonical_dataset(r["dataset"])}
            for r in previous.expanded_rows(cache, low)]
    index = {key: {**r, "dataset": canonical_dataset(r["dataset"])} for key, r in
             physical.physical_index(list(cache["physical_rows"]) + low).items()}
    ids = {row["row_hash"] for row in cache["features"]["screening"]}
    cache_path = Path(study["v9_study"]).parent / model / "development/cache.json"
    inputs = {str(study_path): file_digest(study_path), str(cache_path): file_digest(cache_path)}
    for split in ("fit", "screening"):
        stage = split + "-p010-p020"
        plan = Path(study["output_root"]) / "plans" / (stage + "-" + model) / "plan.json"
        inputs[str(plan)] = file_digest(plan)
    provenance = {
        "study_sha256": study["sha256"], "manifest_sha256": study["manifest_sha256"],
        "checkpoint_sha256": file_digest(study["models"][model]["checkpoint"]),
        "input_hashes": inputs, "embedded_physical_rows_sha256": digest(cache["physical_rows"]),
        "low_physical_rows_sha256": digest(low), "fit_rows_sha256": digest(rows),
        "evaluation_opened": False, "confirmation_opened": False,
    }
    return rows, ids, index, provenance


def selector_rows(rows, index, selector, menu, ids=None):
    """Require every exact physical outcome; missing options never interpolate."""
    if selector not in ("scorer", "divprune", "cdpruner"):
        raise ValueError("unknown selector")
    if not menu or any(type(p) is not int or not 1 <= p <= 100 for p in menu):
        raise ValueError("invalid physical retention menu")
    allowed = set(ids) if ids is not None else {r["row_hash"] for r in rows}
    if not allowed <= {r["row_hash"] for r in rows}:
        raise ValueError("requested cohort is not in development")
    result = []
    for row in sorted(rows, key=lambda r: r["row_hash"]):
        if row["row_hash"] not in allowed:
            continue
        n, quality = row["input_visual_tokens"], {}
        for p in sorted(set(menu) | {100}):
            count = (n * p + 99) // 100
            source = index.get((row["row_hash"], "full" if count == n else selector, count))
            if source is None:
                raise ValueError(f"missing physical {selector} outcome: {row['row_hash']} count={count}")
            if any(source.get(k) != row[k] for k in
                   ("group_id", "dataset", "input_visual_tokens", "input_text_tokens", "quality_metric")):
                raise ValueError("physical outcome geometry or metric mismatch")
            score = source.get("quality_score")
            if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid physical quality")
            quality[str(count)] = float(score)
        result.append({**copy.deepcopy(row), "quality_by_count": quality})
    return result


def _checked_rows(rows):
    if not rows or len({r["row_hash"] for r in rows}) != len(rows):
        raise ValueError("nonempty unique physical rows required")
    if len({r["dataset"] for r in rows}) != 1:
        raise ValueError("native oracle operates within one benchmark")
    for row in rows:
        n = row["input_visual_tokens"]
        if type(n) is not int or n < 1 or str(n) not in row["quality_by_count"]:
            raise ValueError("positive input count and physical full reference required")
        for k, v in row["quality_by_count"].items():
            if str(int(k)) != k or not 1 <= int(k) <= n or not math.isfinite(v) or not 0 <= v <= 1:
                raise ValueError("invalid physical count/quality")
        if row["dataset"] == "MME" and any(q not in (0, 1) for q in row["quality_by_count"].values()):
            raise ValueError("MME native metric requires binary question correctness")


def _pair_quality(scores):
    return .25 * sum(scores) + .5 * float(all(scores))


def make_actions(rows, objective="max_quality"):
    """Build one choice group per question, or joint question-pair for MME."""
    _checked_rows(rows)
    if objective not in ("max_quality", "min_positive_harm"):
        raise ValueError("unknown oracle objective")
    mme = rows[0]["dataset"] == "MME"
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"] if mme else row["row_hash"]].append(row)
    categories = Counter()
    if mme:
        for pair in grouped.values():
            if len(pair) != 2 or not pair[0].get("category") or len({r.get("category") for r in pair}) != 1:
                raise ValueError("MME needs intact two-question, single-category image pairs")
            categories[pair[0]["category"]] += 1
    result = []
    for key, group in sorted(grouped.items()):
        group = sorted(group, key=lambda r: r["row_hash"])
        weight = 1. / (len(categories) * categories[group[0]["category"]]) if mme else 1. / len(rows)
        full_scores = [r["quality_by_count"][str(r["input_visual_tokens"])] for r in group]
        full = _pair_quality(full_scores) if mme else full_scores[0]
        actions = []
        for counts in product(*(sorted(map(int, r["quality_by_count"])) for r in group)):
            scores = [r["quality_by_count"][str(k)] for r, k in zip(group, counts)]
            quality = _pair_quality(scores) if mme else scores[0]
            harm = max(0., full - quality)
            actions.append({"cost": sum(counts), "utility": weight * (quality if objective == "max_quality" else -harm),
                "quality": weight * quality, "positive_harm": weight * harm,
                "counts": dict(zip((r["row_hash"] for r in group), counts))})
        result.append({"group_id": key, "actions": actions})
    return result


def _prepare_groups(groups):
    if not groups or len({g["group_id"] for g in groups}) != len(groups):
        raise ValueError("unique nonempty action groups required")
    result = []
    for group in sorted(groups, key=lambda g: g["group_id"]):
        if not group["actions"]:
            raise ValueError("every group needs physical actions")
        by_cost = {}
        for action in group["actions"]:
            c, u = action["cost"], action["utility"]
            if type(c) is not int or c < 1 or not math.isfinite(u):
                raise ValueError("integer physical cost and finite utility required")
            old = by_cost.get(c)
            if old is None or u > old["utility"]:
                by_cost[c] = action
        frontier, best = [], -math.inf
        for c, action in sorted(by_cost.items()):
            if action["utility"] > best:
                frontier.append(action)
                best = action["utility"]
        result.append({"group_id": group["group_id"], "actions": frontier})
    return result


def _result(groups, choices, budget, upper, solver):
    chosen = [g["actions"][i] for g, i in zip(groups, choices)]
    cost, utility = sum(a["cost"] for a in chosen), sum(a["utility"] for a in chosen)
    if cost > budget or upper < utility - 1e-10:
        raise AssertionError("invalid oracle incumbent or bound")
    counts = {}
    for action in chosen:
        if set(counts) & set(action.get("counts", {})):
            raise ValueError("oracle choices reuse a request")
        counts.update(action.get("counts", {}))
    gap = max(0., upper - utility)
    return {"solver": solver, "budget_tokens": budget, "used_tokens": cost, "unused_tokens": budget - cost,
        "feasible_utility": utility, "upper_bound": max(upper, utility), "optimality_gap": gap,
        "optimality_certified_within_1e_9": gap <= 1e-9,
        "native_quality": sum(a.get("quality", 0.) for a in chosen),
        "positive_harm": sum(a.get("positive_harm", 0.) for a in chosen),
        "counts": counts, "answer_informed_not_deployable": True}


def solve_oracle(groups, budget, *, exact_group_limit=32, max_states=20000, iterations=72):
    """Sparse exact DP where affordable; valid Lagrange bound otherwise.

    No assumptions about monotonic quality, equal image sizes, or equal counts
    within an MME pair. Slack is allowed, reported, and never rounded away.
    """
    if type(budget) is not int or budget < 1 or max_states < 1 or iterations < 1:
        raise ValueError("invalid oracle limits")
    groups = _prepare_groups(groups)
    if sum(g["actions"][0]["cost"] for g in groups) > budget:
        raise ValueError("budget below minimum physical menu cost")
    if len(groups) <= exact_group_limit:
        frontier = {0: (0., ())}
        completed = True
        for group in groups:
            candidate = {}
            for cost, (value, choices) in frontier.items():
                for j, action in enumerate(group["actions"]):
                    c, v = cost + action["cost"], value + action["utility"]
                    if c <= budget and (c not in candidate or v > candidate[c][0]):
                        candidate[c] = v, choices + (j,)
            frontier, best = {}, -math.inf
            for c, (v, choices) in sorted(candidate.items()):
                if v > best:
                    frontier[c] = v, choices
                    best = v
            if len(frontier) > max_states:
                completed = False
                break
        if completed:
            _, (value, choices) = max(frontier.items(), key=lambda item: (item[1][0], -item[0]))
            return _result(groups, choices, budget, value, "exact_sparse_pareto_dp")

    candidates = []
    upper = math.inf

    def evaluate(price):
        nonlocal upper
        choices = [max(range(len(g["actions"])), key=lambda j:
                       (g["actions"][j]["utility"] - price * g["actions"][j]["cost"],
                        -g["actions"][j]["cost"], -j)) for g in groups]
        chosen = [g["actions"][i] for g, i in zip(groups, choices)]
        cost = sum(a["cost"] for a in chosen)
        utility = sum(a["utility"] for a in chosen)
        upper = min(upper, price * budget + sum(a["utility"] - price * a["cost"] for a in chosen))
        if cost <= budget:
            candidates.append(choices)
        return cost

    if evaluate(0.) > budget:
        low, high = 0., 1.
        while evaluate(high) > budget:
            high *= 2.
            if not math.isfinite(high):
                raise ArithmeticError("oracle price search overflow")
        for _ in range(iterations):
            middle = (low + high) / 2
            if evaluate(middle) > budget:
                low = middle
            else:
                high = middle
        evaluate(high)
    candidates.append([0] * len(groups))
    best = None
    # Price plateaus often repeat assignments; deterministic deduplication also
    # avoids repeated greedy work. Improvements consume only physical actions.
    for start in sorted(set(map(tuple, candidates))):
        choices = list(start)
        cost = sum(g["actions"][j]["cost"] for g, j in zip(groups, choices))
        while True:
            upgrades = []
            for i, (group, current) in enumerate(zip(groups, choices)):
                a = group["actions"][current]
                for j in range(current + 1, len(group["actions"])):
                    b = group["actions"][j]
                    dc, du = b["cost"] - a["cost"], b["utility"] - a["utility"]
                    if dc <= budget - cost and du > 0:
                        upgrades.append((du / dc, du, -dc, -i, -j, i, j, dc))
            if not upgrades:
                break
            *_, i, j, dc = max(upgrades)
            choices[i], cost = j, cost + dc
        value = sum(g["actions"][j]["utility"] for g, j in zip(groups, choices))
        key = (value, -cost, tuple(-j for j in choices))
        if best is None or key > best[0]:
            best = key, choices
    return _result(groups, best[1], budget, upper + 1e-12, "lagrangian_upper_bound_feasible_greedy")


def summarize_assignment(rows, counts):
    """Native metrics with equal category weights for MME, not question mean."""
    if set(counts) != {r["row_hash"] for r in rows}:
        raise ValueError("assignment must cover the exact cohort")
    actions = make_actions([{**r, "quality_by_count": {
        str(k): r["quality_by_count"][str(k)] for k in {counts[r["row_hash"]], r["input_visual_tokens"]}}}
        for r in rows])
    selected = []
    for group in actions:
        selected.append(next(a for a in group["actions"] if all(counts[key] == value for key, value in a["counts"].items())))
    return {"quality": sum(a["quality"] for a in selected),
            "positive_harm": sum(a["positive_harm"] for a in selected),
            "retained_tokens": sum(counts.values()), "input_tokens": sum(r["input_visual_tokens"] for r in rows)}


def selector_comparison(rows_by_selector, menu=SELECTOR_MENU):
    """Compare exact same identities/counts; equally average domains and knots."""
    cohorts = [{r["row_hash"] for r in rows} for rows in rows_by_selector.values()]
    if not cohorts or any(ids != cohorts[0] for ids in cohorts):
        raise ValueError("selectors must share exactly the same comparison cohort")
    result = []
    for selector, rows in sorted(rows_by_selector.items()):
        if {r["dataset"] for r in rows} != set(DATASETS):
            raise ValueError("selector comparison requires all four benchmarks")
        panels = []
        for dataset in DATASETS:
            group = [r for r in rows if r["dataset"] == dataset]
            for percent in menu:
                counts = {r["row_hash"]: (r["input_visual_tokens"] * percent + 99) // 100 for r in group}
                summary = summarize_assignment(group, counts)
                panels.append({"dataset": dataset, "keep_percent": percent, "requests": len(group),
                    "independent_groups": len({r["group_id"] for r in group}), **summary})
        domain = {d: sum(p["quality"] for p in panels if p["dataset"] == d) / len(menu) for d in DATASETS}
        result.append({"selector": selector, "mean_quality_pp": 100 * sum(domain.values()) / len(DATASETS),
            "worst_domain_mean_quality_pp": 100 * min(domain.values()), "per_domain_mean_quality_pp": {
                d: 100 * q for d, q in domain.items()}, "panels": panels,
            "baseline_scope": BASELINE_SCOPE, "cohort_sha256": digest(sorted(cohorts[0]))})
    return sorted(result, key=lambda r: (-r["mean_quality_pp"], -r["worst_domain_mean_quality_pp"], r["selector"]))


def oracle_panels(rows, selector, *, current_counts=None):
    result = []
    for dataset in DATASETS:
        group = [r for r in rows if r["dataset"] == dataset]
        total = sum(r["input_visual_tokens"] for r in group)
        budgets = {str(p): sum((r["input_visual_tokens"] * p + 99) // 100 for r in group) for p in (50, 70, 90)}
        if current_counts is not None:
            budgets["current_v10"] = sum(current_counts[r["row_hash"]] for r in group)
        for objective in ("max_quality", "min_positive_harm"):
            actions = make_actions(group, objective)
            for target, budget in budgets.items():
                answer = solve_oracle(actions, budget)
                counts = answer.pop("counts")
                reference_counts = ({r["row_hash"]: current_counts[r["row_hash"]] for r in group}
                                    if target == "current_v10" else {
                                        r["row_hash"]: (r["input_visual_tokens"] * int(target) + 99) // 100
                                        for r in group})
                reference = None
                if all(str(reference_counts[r["row_hash"]]) in r["quality_by_count"] for r in group):
                    reference = summarize_assignment(group, reference_counts)
                full = summarize_assignment(group, {r["row_hash"]: r["input_visual_tokens"] for r in group})
                result.append({"dataset": dataset, "selector": selector, "objective": objective,
                    "target": target, "input_tokens": total, "requests": len(group),
                    "independent_groups": len({r["group_id"] for r in group}),
                    "full_quality": full["quality"],
                    "reference_kind": "v10_assignment_applied_to_this_selector" if target == "current_v10" else "uniform_physical_retention",
                    "reference": reference,
                    "feasible_quality_minus_reference_pp": 100 * (answer["native_quality"] - reference["quality"]) if reference else None,
                    "counts_sha256": digest(counts), "counts": counts, **answer})
    return result


def build_report(model, study_path=DEFAULT_STUDY):
    from . import compression_policy_v10 as policy
    from . import compression_study_v10 as previous
    rows, screening_ids, index, provenance = load_development(model, study_path)
    selectors = ("scorer", "divprune") + (("cdpruner",) if model == "llava_next" else ())
    common = {s: selector_rows(rows, index, s, (*SELECTOR_MENU, 100), screening_ids) for s in selectors}
    comparison = selector_comparison(common)
    candidate_path = Path(study_path).resolve().parent / "coordinator/candidates.json"
    candidate = matrix.checked(candidate_path)
    if candidate["study_sha256"] != provenance["study_sha256"]:
        raise ValueError("v10 current policy belongs to a different study")
    selected = previous.selected_policy(candidate["models"][model])
    current = policy.predict_counts(selected, [{k: v for k, v in r.items() if k != "quality_by_count"} for r in rows],
                                    allow_development_overlap=True)["counts"]
    provenance["input_hashes"][str(candidate_path)] = file_digest(candidate_path)
    oracle = oracle_panels(rows, "scorer", current_counts=current)
    common_oracles = {s: oracle_panels(data, s, current_counts=current) for s, data in common.items()}
    return matrix.seal({"format": FORMAT, "model": model,
        "evidence": "already inspected development only; never independent confirmation",
        "oracle_warning": "answer-informed achievable incumbents and upper bounds, not deployable policies",
        "baseline_scope": BASELINE_SCOPE, "selector_comparison": comparison,
        "strongest_cached_alternative": next(r["selector"] for r in comparison if r["selector"] != "scorer"),
        "selection_tiebreak": "mean across four benchmarks and seven knots; worst benchmark mean; lexical selector name",
        "runtime_tiebreak_unavailable": "cached timings are not isolated comparable timings; no speed superiority inferred",
        "learned_full_development_oracles": oracle, "common_screening_oracles": common_oracles,
        "menu": {"learned_all_development": list(LEARNED_MENU), "common_screening": [*SELECTOR_MENU, 100]},
        "provenance": provenance,
        "source_hashes": {str(Path(p).resolve()): file_digest(p) for p in (__file__, physical.__file__, matrix.__file__,
            policy.__file__, previous.__file__)}})


def markdown(reports):
    lines = ["# v11 cached diagnosis", "", "Development only. Oracle allocations use answers and are not deployable policies.",
        "Baseline results are projected-feature selector adaptations, not official end-to-end implementations.", "",
        "## Selector comparison", "", "Same 236 questions per model; equal average over four benchmarks and physical retention levels 30–90%.", "",
        "| Model | Patch selector | Mean quality | DocVQA | TextVQA | ChartQA | MME |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for report in reports:
        for row in report["selector_comparison"]:
            vals = [row["per_domain_mean_quality_pp"][d] for d in DATASETS]
            lines.append(f"| {report['model']} | {row['selector']} | {row['mean_quality_pp']:.2f} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines += ["", "## Answer-informed allocation headroom", "",
        "708 development questions per model; 10–100% physically measured learned-selector options. Each budget is the exact sum of per-input ceiling counts, not a rounded mean ratio. Slack is unused budget, not a measurement error.", "",
        "| Model | Dataset | Token cap | All tokens | Uniform / current rule | Best feasible oracle | Quality upper bound | Used / allowed tokens |",
        "|---|---|---:|---:|---:|---:|---:|---:|"]
    for report in reports:
        for row in report["learned_full_development_oracles"]:
            if row["objective"] != "max_quality":
                continue
            reference = f"{100*row['reference']['quality']:.2f}" if row['reference'] else "unmeasured"
            lines.append(f"| {report['model']} | {row['dataset']} | {row['target']} | {100*row['full_quality']:.2f} | "
                f"{reference} | {100*row['native_quality']:.2f} | "
                f"{100*row['upper_bound']:.2f} | {row['used_tokens']} / {row['budget_tokens']} |")
    lines += ["", "Quality and harm use native benchmark metrics; MME keeps image pairs together and averages categories equally. "
        "A feasible answer-informed result above uniform indicates potential headroom, not that an answer-free statistic can recover it. "
        "Detailed JSON also contains minimum-positive-harm oracles and common-cohort selector oracles, assignments, input hashes, and bound gaps.", ""]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=DEFAULT_STUDY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", nargs="+", choices=("qwen25", "llava_next"), default=["qwen25", "llava_next"])
    args = parser.parse_args(argv)
    output = args.output.resolve()
    if output.name != "diagnostics" or not output.parent.name.endswith("_v11"):
        raise ValueError("diagnostics may only write into a new v11/diagnostics namespace")
    reports = []
    for model in args.models:
        report = build_report(model, args.study)
        write_immutable(output / (model + ".json"), report)
        reports.append(report)
        print(model, "cached alternative:", report["strongest_cached_alternative"], flush=True)
    from .adaptive_report_v1 import _atomic_text
    _atomic_text(output / "TABLES.md", markdown(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
