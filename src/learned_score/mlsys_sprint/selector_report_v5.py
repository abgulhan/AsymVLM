"""Paired selector and allocation development tables from already-audited rows.

The caller owns shard/source/manifest audits. This module never reads labels,
fits a policy, or scans an incomplete filesystem into a completed experiment.
Selection additionally requires explicit feasible screening panels and exact
cohorts/counts matched against both frozen selector controls.
"""

from collections import Counter, defaultdict
import math
import statistics

from .adaptive_report_v3 import paired_ci, quality_summary
from .adaptive_report_v4 import _full_cache, paired_against_shuffles
from .adaptive_data_v3 import dataset_family


FORMAT = "mlsys_selector_report_v5"
INITIAL_MODELS = ("qwen25", "llava_next")
CONTROLS = ("current_rank32", "saliency_control")
CANDIDATES = (*CONTROLS, "query_saliency", "answer_soft", "query_answer_soft",
              "answer_hard", "rank32_coverage")
LABELS = {
    "current_rank32": "Current learned patch scorer",
    "saliency_control": "Retrained patch-importance control",
    "query_saliency": "Contextual query + patch-importance training",
    "answer_soft": "Answer-likelihood training, soft selection during training",
    "query_answer_soft": "Contextual query + answer-likelihood soft training",
    "answer_hard": "Answer-likelihood training, hard selection during training",
    "rank32_coverage": "Current learned scorer + 20% diversity coverage",
    "divprune": "DivPrune selector adaptation",
    "cdpruner": "CDPruner selector adaptation",
    "full": "All image tokens",
}


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"nonfinite or missing {name}")
    return float(value)


def _dataset(name):
    aliases = {"chartqa": "ChartQA", "lmms-lab/chartqa": "ChartQA", "lmms-lab/mme": "MME"}
    family = aliases.get(name.lower(), dataset_family(name))
    return {"ChartQA_TEST": "ChartQA"}.get(family, family)


def _mode(row):
    return row.get("mode", row.get("schedule_cell", {}).get("mode", "uniform"))


def _seed(row):
    return row.get("shuffle_seed", row.get("schedule_cell", {}).get("shuffle_seed"))


def _model(row):
    return row.get("model_tag", row.get("model"))


def _key(row):
    target = row.get("target")
    if row["selector"] == "full":
        target = 0
    mode = _mode(row)
    if mode == "shuffle":
        source = row.get("allocation_source_mode", row.get("schedule_cell", {}).get("allocation_source_mode"))
        if source not in {"continuous_1pp", "continuous_5pp"} or _seed(row) not in {42, 43, 44}:
            raise ValueError("shuffle rows require their source grid and a predeclared seed")
        mode = f"shuffle:{source}"
    return (_model(row), _dataset(row["dataset"]), target,
            row["candidate"], row["selector"], mode, _seed(row))


def _index(rows):
    groups = defaultdict(list)
    identities = set()
    for row in rows:
        if row.get("status", "ok") != "ok":
            raise ValueError("only successfully audited physical rows may be reported")
        key = _key(row)
        if not key[0] or not key[1] or not key[3] or key[4] not in {"scorer", "full", "divprune", "cdpruner"}:
            raise ValueError("row lacks model, dataset, candidate or a known selector")
        if type(key[2]) is not int or key[2] not in {0, 40, 50, 60, 70}:
            raise ValueError("unknown target KV reduction")
        identity = key + (row["request_id"],)
        if identity in identities:
            raise ValueError("duplicate candidate/mode/request observation")
        identities.add(identity)
        if not 0 <= _finite(row["quality_score"], "quality score") <= 1:
            raise ValueError("quality must be a fraction in [0,1]")
        for field in ("input_visual_tokens", "retained_visual_tokens", "input_text_tokens"):
            if type(row[field]) is not int or row[field] < (0 if field == "input_text_tokens" else 1):
                raise ValueError(f"invalid {field}")
        if row["retained_visual_tokens"] > row["input_visual_tokens"]:
            raise ValueError("retained count exceeds image size")
        groups[key].append(row)
    for group in groups.values():
        group.sort(key=lambda row: row["request_id"])
    return groups


def _metric(rows):
    values = {row["quality_metric"] for row in rows}
    if len(values) != 1:
        raise ValueError("one panel cannot mix quality metrics")
    return next(iter(values))


def _compare(candidate, baseline, *, same_counts, repetitions):
    a = {row["request_id"]: row for row in candidate}
    b = {row["request_id"]: row for row in baseline}
    if not a or set(a) != set(b):
        return {"status": "missing_or_incomplete_cohort", "delta_pp": None,
                "ci_lower_pp": None, "ci_upper_pp": None}
    for request in a:
        if (a[request]["group_id"] != b[request]["group_id"]
                or a[request].get("category") != b[request].get("category")
                or any(a[request][name] != b[request][name]
                       for name in ("input_visual_tokens", "input_text_tokens"))):
            raise ValueError("paired identity/category/input geometry changed")
    if _metric(candidate) != _metric(baseline):
        raise ValueError("paired quality metrics changed")
    if same_counts and any(a[key]["retained_visual_tokens"] != b[key]["retained_visual_tokens"] for key in a):
        return {"status": "per_input_budget_mismatch", "delta_pp": None,
                "ci_lower_pp": None, "ci_upper_pp": None}
    if any(sum(row["metrics"]["cache"][field] for row in candidate)
           != sum(row["metrics"]["cache"][field] for row in baseline)
           for field in ("kv_capacity_bytes", "total_owned_bytes")):
        return {"status": "aggregate_budget_mismatch", "delta_pp": None,
                "ci_lower_pp": None, "ci_upper_pp": None}
    return {**paired_ci(candidate, baseline, _metric(candidate), repetitions=repetitions),
            "cohort_complete": True, "exact_matched_budget": True}


def _memory(rows):
    kv = [row["metrics"]["cache"]["kv_capacity_bytes"] for row in rows]
    owned = [row["metrics"]["cache"]["total_owned_bytes"] for row in rows]
    for value in kv + owned:
        if type(value) is not int or value <= 0:
            raise ValueError("cache bytes must be positive integers")
    full = [_full_cache(row) for row in rows]
    overhead = [row["metrics"].get("selection_and_visual_compaction_seconds") for row in rows]
    present = all(value is not None for value in overhead)
    if present and any(_finite(value, "selection overhead") < 0 for value in overhead):
        raise ValueError("selection overhead cannot be negative")
    return {
        "mean_kv_bytes": statistics.mean(kv), "aggregate_kv_bytes": sum(kv),
        "aggregate_owned_cache_bytes": sum(owned),
        "mean_per_input_kv_reduction_percent": statistics.mean(100 * (1 - x / y) for x, y in zip(kv, full)),
        "aggregate_kv_reduction_percent": 100 * (1 - sum(kv) / sum(full)),
        "inputs_with_kv_reduction_percent": 100 * sum(x < y for x, y in zip(kv, full)) / len(rows),
        "mean_visual_retention_percent": statistics.mean(100 * row["retained_visual_tokens"] / row["input_visual_tokens"] for row in rows),
        "distinct_retained_counts": len({row["retained_visual_tokens"] for row in rows}),
        "mean_selection_and_compaction_seconds": statistics.mean(overhead) if present else None,
        "mean_selector_seconds": _selector_seconds(rows),
        "overhead_timing_isolated": False,
    }


def _selector_seconds(rows):
    values = [row.get("selector_seconds", row.get("metrics", {}).get("selector_seconds")) for row in rows]
    if any(value is None for value in values):
        return None
    if any(_finite(value, "selector time") < 0 for value in values):
        raise ValueError("selector time cannot be negative")
    return statistics.mean(values)


def _expected(panels):
    if panels is None:
        return None
    result = []
    for panel in panels:
        if isinstance(panel, dict):
            value = (panel.get("model_tag", panel.get("model")), _dataset(panel["dataset"]), panel["target"])
        else:
            model, dataset, target = panel
            value = (model, _dataset(dataset), target)
        if not value[0] or not value[1] or type(value[2]) is not int or value[2] not in {40, 50, 60, 70}:
            raise ValueError("invalid declared feasible panel")
        result.append(value)
    if not result or len(result) != len(set(result)):
        raise ValueError("expected panels must be nonempty and unique")
    return set(result)


def _decision(panels, *, complete, expected, expected_candidates, stage):
    base = {"stage": stage, "selected_candidate": None, "qualified_candidates": [],
            "candidate_results": [], "confirmation": False,
            "selection_rule": {"both_initial_models_positive_vs_both_controls": True,
                               "minimum_model_balanced_gain_vs_current_rank32_pp": .5,
                               "minimum_panel_gain_pp": -1.,
                               "tie_break": ["worst_panel_gain", "selection_overhead", "candidate_name"]}}
    if stage != "screening":
        return {**base, "status": "decision_frozen_elsewhere_not_selected_from_this_stage"}
    if not complete or expected is None:
        return {**base, "status": "incomplete_or_no_declared_panel_contract"}
    if (set(panel[0] for panel in expected) != set(INITIAL_MODELS)
            or any(panel[2] not in {60, 70} for panel in expected)):
        raise ValueError("selector screening selection requires both initial models at 60/70% targets only")
    uniform = {(panel["model"], panel["dataset"], panel["target"], panel["candidate"]): panel
               for panel in panels if panel["selector"] == "scorer" and panel["mode"] == "uniform"}
    expected_keys = {(*panel, candidate) for panel in expected for candidate in expected_candidates}
    missing = sorted(expected_keys - uniform.keys())
    if missing:
        return {**base, "status": "missing_predeclared_panels", "missing_panels": missing}
    unexpected = sorted(set(uniform) - expected_keys)
    if unexpected:
        raise ValueError("screening contains an undeclared candidate or feasible panel")
    invalid = []
    for key, panel in uniform.items():
        if abs(panel["memory"]["aggregate_kv_reduction_percent"] - panel["target"]) > .2:
            invalid.append((*key, "target_memory_mismatch"))
        for control in CONTROLS:
            pair = panel["differences"][control]
            if not pair.get("cohort_complete") or not pair.get("exact_matched_budget"):
                invalid.append((*key, control, pair["status"]))
    if invalid:
        return {**base, "status": "incomplete_or_budget_mismatched_controls", "invalid_panels": invalid}
    rankings = []
    for candidate in expected_candidates:
        if candidate in CONTROLS:
            continue
        entries = [uniform[(*panel, candidate)] for panel in sorted(expected)]
        per_control = {}
        all_deltas = []
        for control in CONTROLS:
            models = {model: statistics.mean(panel["differences"][control]["delta_pp"] for panel in entries if panel["model"] == model)
                      for model in INITIAL_MODELS}
            deltas = [panel["differences"][control]["delta_pp"] for panel in entries]
            all_deltas.extend(deltas)
            per_control[control] = {"model_mean_delta_pp": models,
                                    "model_balanced_mean_delta_pp": statistics.mean(models.values()),
                                    "worst_panel_delta_pp": min(deltas)}
        qualified = all(all(value > 0 for value in result["model_mean_delta_pp"].values())
                        and result["worst_panel_delta_pp"] >= -1.
                        for result in per_control.values())
        qualified = qualified and per_control["current_rank32"]["model_balanced_mean_delta_pp"] >= .5
        overheads = [panel["memory"]["mean_selector_seconds"] for panel in entries]
        overhead = (statistics.mean(statistics.mean(panel["memory"]["mean_selector_seconds"]
                                                   for panel in entries if panel["model"] == model)
                                    for model in INITIAL_MODELS) if all(value is not None for value in overheads) else None)
        result = {"candidate": candidate, "qualified": qualified, "comparisons": per_control,
                  "worst_panel_delta_pp": min(all_deltas),
                  "model_balanced_selection_seconds": overhead}
        base["candidate_results"].append(result)
        if qualified:
            rankings.append(result)
    rankings.sort(key=lambda row: (-row["worst_panel_delta_pp"],
                                  row["model_balanced_selection_seconds"] if row["model_balanced_selection_seconds"] is not None else math.inf,
                                  row["candidate"]))
    return {**base, "status": "selected_for_development" if rankings else "no_candidate_qualified",
            "selected_candidate": rankings[0]["candidate"] if rankings else None,
            "qualified_candidates": [row["candidate"] for row in rankings]}


def summarize(rows, *, complete=False, expected_panels=None, expected_candidates=CANDIDATES,
              stage="screening", repetitions=10000):
    rows = list(rows)
    if not set(CONTROLS) <= set(expected_candidates) or len(expected_candidates) != len(set(expected_candidates)):
        raise ValueError("declare both unique selector controls")
    if stage not in {"screening", "evaluation", "allocation_screening", "allocation_evaluation",
                     "allocation", "replication", "query_diagnostic"}:
        raise ValueError("unknown reporting stage")
    groups = _index(rows)
    panels, allocation = [], []
    for key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        model, dataset, target, candidate, selector, mode, seed = key
        item = {"model": model, "dataset": dataset, "target": target, "candidate": candidate,
                "selector": selector, "mode": _mode(values[0]), "shuffle_seed": seed,
                "allocation_source_mode": values[0].get("allocation_source_mode", values[0].get("schedule_cell", {}).get("allocation_source_mode")),
                "label": LABELS.get(candidate, candidate) if selector == "scorer" else LABELS[selector],
                "quality": quality_summary(values, _metric(values)), "memory": _memory(values),
                "differences": {}, "evidence": "physical_development_not_fresh_confirmation"}
        if selector != "full":
            for control in CONTROLS:
                baseline = groups.get((model, dataset, target, control, "scorer", "uniform", None), [])
                item["differences"][control] = _compare(values, baseline,
                    same_counts=mode == "uniform", repetitions=repetitions)
        panels.append(item)
        if selector != "scorer" or mode == "full":
            continue
        own_uniform = groups.get((model, dataset, target, candidate, "scorer", "uniform", None), [])
        difference = _compare(values, own_uniform, same_counts=False, repetitions=repetitions)
        uniform_map = {row["request_id"]: row for row in own_uniform}
        changed = None
        if difference.get("cohort_complete"):
            changed = 100 * sum(row["retained_visual_tokens"] != uniform_map[row["request_id"]]["retained_visual_tokens"] for row in values) / len(values)
        entry = {**item, "difference_from_same_selector_uniform": difference,
                 "inputs_with_changed_budget_percent": changed,
                 "difference_from_shuffled_budgets": None,
                 "shuffle_controls_status": "not_applicable"}
        if mode.startswith("continuous_"):
            shuffles = [group for other_key, group in groups.items()
                        if other_key[:5] == key[:5] and other_key[5] == f"shuffle:{mode}"]
            entry["shuffle_controls_status"] = "missing_or_incomplete_three_seed_control"
            if len(shuffles) == 3 and all(_compare(values, group, same_counts=False, repetitions=100).get("cohort_complete") for group in shuffles):
                distribution = Counter((row["input_visual_tokens"], row["retained_visual_tokens"]) for row in values)
                if any(Counter((row["input_visual_tokens"], row["retained_visual_tokens"]) for row in group) != distribution for group in shuffles):
                    raise ValueError("shuffle changed the within-image-size budget distribution")
                source_counts = {row["request_id"]: row["retained_visual_tokens"] for row in values}
                changed = {str(_seed(group[0])): 100 * sum(row["retained_visual_tokens"] != source_counts[row["request_id"]] for row in group) / len(group)
                           for group in shuffles}
                entry["shuffle_changed_inputs_percent_by_seed"] = changed
                entry["shuffle_controls_status"] = ("complete_but_unchanged_uninformative" if not any(changed.values())
                                                    else "complete_exact_distribution")
                entry["difference_from_shuffled_budgets"] = paired_against_shuffles(values, shuffles, _metric(values), repetitions=repetitions)
        allocation.append(entry)
    decision = _decision(panels, complete=complete, expected=_expected(expected_panels),
                         expected_candidates=expected_candidates, stage=stage)
    return {"format": FORMAT, "complete": bool(complete), "stage": stage,
            "panels": panels, "allocation_panels": allocation, "selector_decision": decision,
            "limitations": ["Previously inspected development data, not fresh confirmation.",
                            "DivPrune/CDPruner remain selector adaptations unless separately validated.",
                            "Selection overhead timings are descriptive, not isolated system measurements.",
                            "Soft training weights do not imply fractional inference tokens: physical integer counts are evaluated."]}


def _number(value, signed=False):
    return "—" if value is None else format(value, "+.2f" if signed else ".2f")


def _interval(value):
    if value.get("delta_pp") is None:
        return value.get("status", "unavailable").replace("_", " ")
    return f"{_number(value['delta_pp'], True)} [{_number(value.get('ci_lower_pp'))}, {_number(value.get('ci_upper_pp'))}]"


def render_markdown(data):
    lines = ["# Selector and budget development comparisons", "",
             "All deltas are percentage points. Matched comparisons require the same questions and measured KV-cache memory.", "",
             "## Table 1: Patch selection at fixed per-input budgets", "",
             "| Model | Benchmark | KV reduction target | Method | Questions | Quality (%) | Difference vs current learned scorer [95% CI] | Difference vs retrained control [95% CI] | Mean KV reduction (%) | Inputs saving KV (%) |",
             "|---|---|---:|---|---:|---:|---|---|---:|---:|"]
    for panel in data["panels"]:
        if panel["mode"] not in {"uniform", "full"}:
            continue
        differences = panel["differences"]
        lines.append(f"| {panel['model']} | {panel['dataset']} | {panel['target']}% | {panel['label']} | {panel['quality']['questions']} | {_number(panel['quality']['score_percent'])} | "
                     + " | ".join(_interval(differences[control]) if control in differences else "not applicable" for control in CONTROLS)
                     + f" | {_number(panel['memory']['mean_per_input_kv_reduction_percent'])} | {_number(panel['memory']['inputs_with_kv_reduction_percent'])} |")
    lines += ["", "## Table 2: Budget allocation with the patch selector held fixed", "",
              "| Model | Benchmark | Target | Patch selector | Budget rule | Quality (%) | Difference vs uniform [95% CI] | Difference vs shuffled budgets [95% CI] | Mean KV reduction (%) | Inputs saving KV (%) | Budgets changed (%) |",
              "|---|---|---:|---|---|---:|---|---|---:|---:|---:|"]
    for panel in data["allocation_panels"]:
        if panel["mode"] == "uniform" and not any(other["candidate"] == panel["candidate"] and other["model"] == panel["model"] and other["mode"] != "uniform" for other in data["allocation_panels"]):
            continue
        rule = panel["mode"] + (f" ({panel['allocation_source_mode']}) seed {panel['shuffle_seed']}" if panel["shuffle_seed"] is not None else "")
        shuffled = panel["difference_from_shuffled_budgets"]
        shuffle_text = _interval(shuffled) if shuffled else "unavailable"
        if panel["shuffle_controls_status"] == "complete_but_unchanged_uninformative":
            shuffle_text += "; unchanged budgets, uninformative"
        lines.append(f"| {panel['model']} | {panel['dataset']} | {panel['target']}% | {panel['label']} | {rule} | {_number(panel['quality']['score_percent'])} | {_interval(panel['difference_from_same_selector_uniform'])} | {shuffle_text} | {_number(panel['memory']['mean_per_input_kv_reduction_percent'])} | {_number(panel['memory']['inputs_with_kv_reduction_percent'])} | {_number(panel['inputs_with_changed_budget_percent'])} |")
    lines += ["", "Selection status: " + data["selector_decision"]["status"].replace("_", " ") + ".", ""]
    lines += ["- " + item for item in data["limitations"]]
    return "\n".join(lines) + "\n"


def report(rows, *, complete=False, expected_panels=None, **kwargs):
    result = summarize(rows, complete=complete, expected_panels=expected_panels, **kwargs)
    return {**result, "tables_markdown": render_markdown(result)}
