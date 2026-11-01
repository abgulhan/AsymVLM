"""Two recovery tables from complete, audited, exact-budget physical cohorts."""

from collections import Counter, defaultdict
import copy
from pathlib import Path
import statistics
import time

from . import recovery_study_v6 as study_api
from . import selector_study_v5 as previous
from .adaptive_report_v3 import quality_summary
from .adaptive_report_v4 import audit_cache, paired_against_shuffles
from .selector_report_v5 import _compare, _dataset, _memory
from .state import atomic_json, digest, file_digest


LABELS = {
    "full": "Baseline (all image tokens)", "uniform": "AsymVLM static",
    "entropy": "AsymVLM adaptive: entropy", "endpoint": "AsymVLM adaptive: endpoint-trained score",
    "removal_harm": "AsymVLM adaptive: removal-harm score",
    "divprune": "DivPrune selector adaptation", "cdpruner": "CDPruner selector adaptation",
    **{f"shuffled_seed{s}": f"Shuffled removal-harm budgets, seed {s}" for s in study_api.SEEDS},
}


def normalize(source, cell, model, provenance):
    """Change logical experiment labels only; never change measured quantities."""
    if source["selector"] != cell["selector"] or source["selected_count"] != cell["selected_count"]:
        raise ValueError("cached physical observation does not match exact selector/count")
    row = copy.deepcopy(source)
    row.update(model_tag=model, candidate="current_rank32", method=cell["key"],
               mode=cell["mode"], policy=cell["policy"], target=cell["target"],
               schedule_cell=cell, recovery_provenance=provenance)
    return row


def load_rows(study, model, stage="allocation"):
    root = Path(study["output_root"]) / model
    master_path = root / f"schedules/{stage}.json"
    if not master_path.exists():
        return [], {"status": "no_frozen_schedule", "complete": False}
    master = study_api.checked(master_path)
    reuse_path, execution_path = (root / f"schedules/{stage}-{kind}.json" for kind in ("reuse", "execution"))
    if not reuse_path.exists() or not execution_path.exists():
        return [], {"status": "awaiting_execution_publication", "complete": False}
    reuse, execution = study_api.checked(reuse_path), study_api.checked(execution_path)
    if any(item["master_schedule_sha256"] != master["sha256"] for item in (reuse, execution)):
        raise ValueError("recovery report master/reuse/execution disagreement")
    # Re-audit historical sources rather than trusting a previously saved path.
    inventory = study_api.cache_inventory(study, model)
    cached = {(item["path"], item["sha256"]): item["row"] for item in inventory.values()}
    directory = root / stage / "current_rank32"
    measured = previous.scan_physical(study, model, "current_rank32", directory, execution_path) if execution["rows"] else []
    current = {(row["request_id"], row["method"]): row for row in measured or []}
    rows, missing, cached_count = [], [], 0
    for request, planned in master["rows"].items():
        for cell in planned["cells"]:
            key = f"{request}.{cell['key']}"
            entry = reuse["cells"].get(key)
            if entry and entry["status"] == "exact_cached_physical":
                pointer = entry["path"], entry["sha256"]
                if pointer not in cached or file_digest(entry["path"]) != entry["sha256"]:
                    raise ValueError("reused physical evidence is no longer in the audited inventory")
                source, provenance = cached[pointer], {"kind": "exact_cached_physical", **entry}
                cached_count += 1
            else:
                source = current.get((request, cell["key"]))
                provenance = {"kind": "new_physical", "directory": str(directory)}
            if source is None:
                missing.append({"request_id": request, "method": cell["key"],
                                "reason": entry["status"] if entry else "awaiting_physical_shard"})
                continue
            row = normalize(source, cell, model, provenance)
            if row["request_id"] != request:
                raise ValueError("physical cache request identity mismatch")
            audit_cache(row, master["cache_contract"], cell, planned)
            rows.append(row)
    required_missing = [item for item in missing if not item["reason"].startswith("not_run_no_exact_cached")]
    return rows, {"status": "complete" if not required_missing else "awaiting_physical_inference",
                  "complete": not required_missing, "cached_cells": cached_count,
                  "new_cells": len(rows) - cached_count, "missing": missing,
                  "master_schedule_sha256": master["sha256"]}


def method(row):
    return row["selector"] if row["selector"] != "scorer" else row["mode"]


def behavior(candidate, uniform):
    if any(row["schedule_cell"].get("fallback_kind") == "full" for row in candidate):
        raise ValueError("v6 does not authorize full-token fallback")
    baseline = {row["request_id"]: row for row in uniform}
    changed, up, down = [], [], []
    for row in candidate:
        reference = baseline[row["request_id"]]
        difference = row["retained_visual_tokens"] - reference["retained_visual_tokens"]
        if difference:
            changed.append(row)
            (up if difference > 0 else down).append(row)
    def effects(rows):
        values = [row["quality_score"] - baseline[row["request_id"]]["quality_score"] for row in rows]
        return {"inputs": len(rows), "improved": sum(v > 0 for v in values), "worsened": sum(v < 0 for v in values),
                "mean_delta_pp": 100 * statistics.mean(values) if values else None,
                "quality_sum_delta": sum(values)}
    return {"changed_inputs": len(changed), "changed_percent": 100 * len(changed) / len(candidate),
            "changed": effects(changed), "recipients": effects(up), "donors": effects(down),
            "uniform_fallback_inputs": sum(row["schedule_cell"].get("fallback", False) for row in candidate),
            "full_fallback_inputs": sum(row["schedule_cell"].get("fallback_kind") == "full" for row in candidate)}


def summarize(rows, *, repetitions=10000):
    groups = defaultdict(list)
    for row in rows:
        key = row["model_tag"], _dataset(row["dataset"]), row["target"], method(row)
        groups[key].append(row)
    panels = []
    for (model, dataset, target, name), cohort in sorted(groups.items()):
        if len(cohort) != 60 or len({r["request_id"] for r in cohort}) != 60:
            # Never turn a partially completed cohort into a matched-budget result.
            continue
        metric = cohort[0]["quality_metric"]
        if any(row["quality_metric"] != metric for row in cohort):
            raise ValueError("recovery panel mixes quality metrics")
        panel = {"model": model, "dataset": dataset, "target": target, "method": name,
                 "questions": 60, "metric": metric,
                 "quality_percent": quality_summary(cohort, metric)["score_percent"],
                 "memory": _memory(cohort),
                 "mean_live_kv_bytes": statistics.mean(r["metrics"]["cache"]["kv_active_bytes"] for r in cohort),
                 "mean_peak_gpu_bytes": statistics.mean(r["metrics"]["full_request_peak_allocated_bytes"] for r in cohort),
                 "systems_claim": "descriptive only; cached/mixed-run timings are not controlled speed evidence"}
        uniform = groups.get((model, dataset, target, "uniform"), [])
        if target:
            pair = _compare(cohort, uniform, same_counts=False, repetitions=repetitions)
            panel["vs_uniform"] = pair
            if not pair.get("exact_matched_budget"):
                panel["status"] = "unmatched_or_incomplete_control"
            elif abs(panel["memory"]["aggregate_kv_reduction_percent"] - target) > .2:
                panel["status"] = "aggregate_cache_target_mismatch"
            else:
                panel["status"] = "complete_exact_budget"
                panel["behavior"] = behavior(cohort, uniform)
        else:
            panel["status"] = "all_token_reference_not_budget_matched"
        if name == "removal_harm":
            shuffles = [groups.get((model, dataset, target, f"shuffled_seed{seed}"), []) for seed in study_api.SEEDS]
            if all(_compare(cohort, shuffled, same_counts=False, repetitions=100).get("exact_matched_budget")
                   for shuffled in shuffles):
                informative = all(all(r["schedule_cell"].get("shuffle_informative")
                                      and r["schedule_cell"].get("shuffle_movable_fraction", 0) >= .2 for r in group)
                                  for group in shuffles)
                histogram = lambda group: Counter((r["input_visual_tokens"], r["retained_visual_tokens"]) for r in group)
                if any(histogram(group) != histogram(cohort) for group in shuffles):
                    raise ValueError("shuffled budget distribution differs within image geometry")
                panel["vs_shuffled"] = {**paired_against_shuffles(cohort, shuffles, metric, repetitions=repetitions),
                                        "informative": informative}
            else:
                panel["vs_shuffled"] = {"status": "missing_or_unmatched_shuffles", "informative": False}
        panels.append(panel)
    return panels


def decision(panels):
    expected = {(model, dataset, target) for model in study_api.MODELS
                for dataset in ("DocVQA", "TextVQA") for target in study_api.TARGETS}
    values = [p for p in panels if p["method"] == "removal_harm"
              and (p["model"], p["dataset"], p["target"]) in expected]
    if (len(values) != len(expected) or {(p["model"], p["dataset"], p["target"]) for p in values} != expected
            or any(p["status"] != "complete_exact_budget" or not p.get("vs_shuffled", {}).get("informative") for p in values)):
        return {"passed": False, "status": "incomplete_or_uninformative_controls", "confirmation": False}
    means = {model: {"vs_uniform_pp": statistics.mean(p["vs_uniform"]["delta_pp"] for p in values if p["model"] == model),
                     "vs_shuffled_pp": statistics.mean(p["vs_shuffled"]["delta_pp"] for p in values if p["model"] == model)}
             for model in study_api.MODELS}
    balanced = statistics.mean(item["vs_uniform_pp"] for item in means.values())
    worst = min(p["vs_uniform"]["delta_pp"] for p in values)
    passed = all(item["vs_uniform_pp"] > 0 and item["vs_shuffled_pp"] > 0 for item in means.values()) and balanced >= .5 and worst >= -1.
    return {"passed": passed, "status": "promising_development_only" if passed else "failed_development_screen",
            "model_means": means, "balanced_vs_uniform_pp": balanced, "worst_panel_vs_uniform_pp": worst,
            "confirmation": False, "confidence_intervals_reported_not_superiority_gate": True}


def interval(value):
    if not value or value.get("delta_pp") is None:
        return "not available"
    low, high = value.get("ci_lower_pp"), value.get("ci_upper_pp")
    return f"{value['delta_pp']:+.2f} [{low:+.2f}, {high:+.2f}]" if low is not None and high is not None else f"{value['delta_pp']:+.2f} [unavailable]"


def tables(panels, admission, execution):
    lines = ["# Six-hour recovery: development evidence", "",
             "Previously inspected development questions; no fresh confirmation. All differences are percentage points.", "",
             "KV reduction means allocated cache capacity relative to all image tokens, with the same protected prompt and 127 generated-token slots. It is not peak-GPU-memory reduction.", "",
             "## Table 1: Quality and matched cache capacity", "",
             "| Model | Dataset | KV target | Method | N | Quality (%) | Difference vs AsymVLM static [95% CI] | Aggregate KV (GiB) | Mean per-input KV reduction (%) | Inputs saving KV (%) |",
             "|---|---|---:|---|---:|---:|---|---:|---:|---:|"]
    for p in panels:
        m = p["memory"]
        lines.append(f"| {p['model']} | {p['dataset']} | {p['target']}% | {LABELS[p['method']]} | {p['questions']} | {p['quality_percent']:.2f} | {interval(p.get('vs_uniform'))} | {m['aggregate_kv_bytes']/2**30:.4f} | {m['mean_per_input_kv_reduction_percent']:.2f} | {m['inputs_with_kv_reduction_percent']:.2f} |")
    if not panels:
        lines.append("| — | — | — | New physical comparisons not run | — | — | — | — | — | — |")
    lines += ["", "## Table 2: Allocation behavior", "",
              "| Model | Dataset | KV target | Budget rule | Budgets changed (%) | Inputs helped / harmed | Recipient quality change (pp) | Donor quality change (pp) | Uniform fallbacks | Difference vs shuffled [95% CI] |",
              "|---|---|---:|---|---:|---|---:|---:|---:|---|"]
    for p in panels:
        if p["method"] not in {"entropy", "endpoint", "removal_harm"} or "behavior" not in p:
            continue
        b = p["behavior"]
        fmt = lambda value: "not applicable" if value is None else f"{value:+.2f}"
        lines.append(f"| {p['model']} | {p['dataset']} | {p['target']}% | {LABELS[p['method']]} | {b['changed_percent']:.2f} | {b['changed']['improved']} / {b['changed']['worsened']} | {fmt(b['recipients']['mean_delta_pp'])} | {fmt(b['donors']['mean_delta_pp'])} | {b['uniform_fallback_inputs']} | {interval(p.get('vs_shuffled'))} |")
    lines += ["", "Recipient/donor changes are averages within those subsets, not overall gains. Full-token fallback is not used. Competitor methods are native-layout selector adaptations, not official end-to-end reproductions.", "",
              f"CPU admission: `{admission.get('decision', {}).get('decision', 'not fitted')}`.",
              "Physical execution: " + "; ".join(f"{model}: {value['status']}" for model, value in execution.items()) + ".",
              "", "Cached measurements are reused only at identical question, scorer, selector, retained count and frozen inference configuration. Timing and GPU peaks are descriptive, not a new controlled systems comparison."]
    return "\n".join(lines) + "\n"


def write_report(study, *, repetitions=10000):
    from .adaptive_report_v1 import _atomic_text
    study_api.verify(study)
    root = Path(study["output_root"])
    admission_path = root / "policies/admission.json"
    admission = study_api.checked(admission_path) if admission_path.exists() else {}
    rows, states, stress_states = [], {}, {}
    for model in study_api.MODELS:
        measured, state = load_rows(study, model)
        rows.extend(measured)
        states[model] = state
        if (root / model / "schedules/stress.json").exists():
            stress_rows, stress_state = load_rows(study, model, stage="stress")
            rows.extend(stress_rows)
            stress_states[model] = stress_state
    panels = summarize(rows, repetitions=repetitions)
    training = training_status(study, repetitions=repetitions)
    blocked_path = root / "coordinator/access-blocked.json"
    access = study_api.read(blocked_path) if blocked_path.exists() else None
    result = {"format": "mlsys_recovery_report_v6", "study_sha256": study["sha256"],
              "admission": admission, "execution": states, "stress_execution": stress_states,
              "training": training, "access_blocker": access, "panels": panels, "decision": decision(panels),
              "evidence_status": study["evidence_status"], "deadline_utc_epoch": study["deadline_utc_epoch"]}
    atomic_json(root / "reports/main.json", result)
    _atomic_text(root / "reports/TABLES.md", tables(panels, admission, states))
    allocation_complete = bool(admission) and (not study_api.admission_passed(admission)
                                              or all(state["complete"] for state in states.values()))
    atomic_json(root / "coordinator/status.json", {"time": time.time(), "deadline_utc_epoch": study["deadline_utc_epoch"],
                "cpu_admitted": study_api.admission_passed(admission) if admission else False,
                "physical": states, "decision": result["decision"], "training": training,
                "access_blocker": access, "allocation_complete": allocation_complete,
                "complete": allocation_complete and training["complete"]
                    and all(state["complete"] for state in stress_states.values())})
    return {"report": str(root / "reports/main.json"), "tables": str(root / "reports/TABLES.md"),
            "decision": result["decision"], "execution": states}


def training_status(study, *, repetitions=10000):
    """Expected diagnostic stops are terminal, not accuracy successes."""
    root = Path(study["output_root"])
    runs, complete = {}, True
    for gate in ("detached", "corrected_kth"):
        directory = root / "qwen25/training" / f"{gate}-seed42"
        outcome = directory / "outcome.json"
        if not outcome.exists():
            complete = False
            runs[gate] = {"status": "not_completed", "success": False,
                          "window_expired": time.time() >= study["training_cutoff_utc_epoch"]}
        else:
            value = study_api.read(outcome)
            if (value.get("format") != "mlsys_recovery_training_task_v6_outcome"
                    or value.get("identity_sha256") != digest(value.get("identity"))
                    or value.get("identity", {}).get("study_sha256") != study["sha256"]
                    or value["identity"].get("gate") != gate
                    or value["status"] not in {"diagnostic_stopped", "deadline_skipped", "physical_sanity_complete"}
                    or value.get("success") != (value["status"] == "physical_sanity_complete")):
                raise ValueError("training outcome identity/status is invalid")
            for path, sha in value.get("artifact_hashes", {}).items():
                if file_digest(path) != sha:
                    raise ValueError("training outcome evidence changed")
            runs[gate] = value
    comparisons, original_cache, physical = [], None, {}
    for gate, outcome in runs.items():
        if outcome["status"] != "physical_sanity_complete":
            continue
        if original_cache is None:
            original_cache = study_api.cache_inventory(study, "qwen25")
        path = root / "qwen25" / "schedules" / f"training-sanity-{gate}.json"
        schedule = study_api.checked(path)
        physical[gate] = defaultdict(list)
        baseline = defaultdict(list)
        for request, planned in schedule["rows"].items():
            for cell in planned["cells"]:
                row_path = root / "qwen25/training_sanity" / gate / "rows" / f"{request}.{cell['key']}.json"
                if str(row_path) not in outcome["artifact_hashes"]:
                    raise ValueError("training sanity row is not bound to the audited outcome")
                row = study_api.read(row_path)
                key = _dataset(row["dataset"]), cell["target"]
                original = original_cache.get((request, "scorer", cell["selected_count"]))
                if original is None:
                    raise ValueError("training sanity lacks an exact-count original baseline")
                physical[gate][key].append(row)
                baseline[key].append(original["row"])
        for (dataset, target), rows in physical[gate].items():
            comparisons.append({"candidate": gate, "dataset": dataset, "parent_target": target,
                "questions": len(rows), "subset_not_rebalanced": True, "aggregate_target_claim": False,
                "quality_percent": quality_summary(rows, rows[0]["quality_metric"])["score_percent"],
                "vs_original": _compare(rows, baseline[(dataset, target)], same_counts=True, repetitions=repetitions)})
    if {"detached", "corrected_kth"} <= physical.keys():
        for entry in comparisons:
            if entry["candidate"] == "corrected_kth":
                key = entry["dataset"], entry["parent_target"]
                entry["vs_detached"] = _compare(physical["corrected_kth"][key], physical["detached"][key],
                                                  same_counts=True, repetitions=repetitions)
    return {"complete": complete, "runs": runs, "physical_comparisons": comparisons,
            "llava_next": "not_run_bounded_Qwen_priority",
            "interpretation": "gradient/short-training diagnostic; never used to select the allocation ranker"}
