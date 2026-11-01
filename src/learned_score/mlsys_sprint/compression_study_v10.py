"""Pooled, one-statistic compression study; all v9 artifacts remain read-only.

Development diagnostics are never confirmation. Every selected rule is shared
across datasets within a model, and fresh inference needs a separate admission
and pre-label policy gate. The coordinator does not submit cluster jobs itself.
"""

import argparse
from collections import Counter, defaultdict
import copy
import fcntl
import json
from pathlib import Path
import time

from . import compression_matrix_v8 as matrix
from . import compression_policy_v8 as geometry
from . import compression_report_v8 as reporting
from . import compression_study_v8 as projection
from .adaptive_data_v3 import dataset_family
from .adaptive_report_v1 import _atomic_text
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path("reproduction_results/mlsys_compression_20261020_v10")
V9_ROOT = Path("reproduction_results/mlsys_compression_20261018_v9")
FORMAT = "mlsys_compression_study_v10"
MODELS = ("qwen25", "llava_next")
DATASETS = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
BASE_MENU = tuple(range(30, 101, 10))
EXPANDED_MENU = tuple(range(10, 101, 10))
STATISTICS = ("normalized_entropy", "query_weighted_coverage")
read, checked, seal = matrix.read, matrix.checked, matrix.seal


def prepare(cwd, *, hours=24):
    """Bind immutable previous development data without reopening evaluation."""
    cwd = Path(cwd).resolve()
    path = cwd / ROOT / "study.json"
    if path.exists():
        result = checked(path)
        verify(result)
        return result
    if not 0 < hours <= 24:
        raise ValueError("v10 execution window must be positive and at most 24 hours")
    old_path = cwd / V9_ROOT / "study.json"
    old = checked(old_path)
    if old.get("format") != "mlsys_compression_study_v9":
        raise ValueError("expected the frozen v9 study")
    pins = {str(old_path): file_digest(old_path), **old["input_hashes"]}
    for model in MODELS:
        cache_path = old_path.parent / model / "development/cache.json"
        cache = checked(cache_path)
        if (cache.get("study_sha256") != old["sha256"]
                or cache.get("old_evaluation_opened") is not False):
            raise ValueError("only audited v9 development caches can be reused")
        pins[str(cache_path)] = file_digest(cache_path)
        pins[old["models"][model]["checkpoint"]] = file_digest(old["models"][model]["checkpoint"])
    now = time.time()
    study = seal({
        "format": FORMAT, "cwd": str(cwd), "output_root": str(cwd / ROOT),
        "v9_study": str(old_path), "v8_study": old["v8_study"],
        "old_study": old["old_study"], "models": {m: old["models"][m] for m in MODELS},
        "manifest": old["manifest"], "manifest_sha256": old["manifest_sha256"],
        "input_hashes": pins, "created_utc_epoch": now,
        "deadline_utc_epoch": now + hours * 3600,
        "data_admission_deadline_utc_epoch": now + min(4, hours) * 3600,
        "candidate_freeze_deadline_utc_epoch": now + min(12, hours) * 3600,
        "world_size": 16, "keep_percent": list(EXPANDED_MENU),
        "initial_keep_percent": list(BASE_MENU), "datasets": list(DATASETS),
        "statistics": list(STATISTICS), "primary_statistic": STATISTICS[0],
        "pooled_rule_per_model": True, "dataset_is_policy_input": False,
        "dataset_weighting": "equal benchmarks; native metric within benchmark",
        "robust_retention_allowance_pp": 2., "pruning_gain_credit": .5,
        "maximum_measured_loss_pp": 1., "strict_upper_loss_limit_pp": 1.,
        "coverage_trigger_saving_pp": 2.,
        "max_new_tokens": 128, "generated_budget": 512,
        "text_eviction": False, "scorer_training": False,
        "calibration_groups": 1000, "final_groups": 1000,
        "calibration_comparison_slots": 12, "final_comparison_slots": 4,
        "fresh_datasets": ["TextVQA", "ChartQA_TEST"],
        "baseline_scope": old["baseline_scope"],
        "objective": "minimum retained image tokens with negligible full-relative quality loss",
        "old_evaluation_opened": False,
        "evidence": "development until separately admitted calibration and final evaluation",
    })
    verify(study)
    write_immutable(path, study)
    return study


def verify(study):
    if (study.get("sha256") != digest({k: v for k, v in study.items() if k != "sha256"})
            or study.get("format") != FORMAT
            or Path(study["output_root"]) != Path(study["cwd"]) / ROOT
            or study.get("dataset_is_policy_input") is not False):
        raise ValueError("invalid pooled v10 study")
    for path, expected in study["input_hashes"].items():
        if file_digest(path) != expected:
            raise ValueError("frozen study input changed: " + path)


def cached_development(study, model):
    if model not in MODELS:
        raise ValueError("unsupported model")
    path = Path(study["v9_study"]).parent / model / "development/cache.json"
    cached = checked(path)
    if file_digest(path) != study["input_hashes"][str(path)]:
        raise ValueError("previous development cache changed")
    if cached["old_evaluation_opened"] is not False:
        raise ValueError("historical evaluation is not development")
    return cached


def expanded_rows(cached, physical_rows):
    """Attach only complete, exactly measured low-count outcomes."""
    rows = copy.deepcopy(cached["fit_rows"])
    index = projection.physical_index(physical_rows)
    for row in rows:
        n, request = row["input_visual_tokens"], row["row_hash"]
        for percent in (10, 20):
            count = (n * percent + 99) // 100
            source = index.get((request, "full" if count == n else "scorer", count))
            if source is None:
                raise ValueError("incomplete low-count cohort")
            if any(source.get(key) != row[key] for key in
                   ("group_id", "dataset", "input_visual_tokens", "input_text_tokens")):
                raise ValueError("low-count identity/geometry mismatch")
            previous = row["quality_by_count"].get(str(count))
            if previous is not None and previous != source["quality_score"]:
                raise ValueError("rounded count alias has inconsistent physical quality")
            row["quality_by_count"][str(count)] = source["quality_score"]
    return rows


def with_coverage(rows, features):
    result = copy.deepcopy(rows)
    if set(features) != {r["row_hash"] for r in rows}:
        raise ValueError("coverage must include the complete declared development cohort")
    for row in result:
        feature = features[row["row_hash"]]
        if (feature.get("answers_consumed_by_feature_extractor") is not False
                or feature.get("request_id", row["row_hash"]) != row["row_hash"]
                or feature.get("input_visual_tokens") != row["input_visual_tokens"]):
            raise ValueError("coverage feature lacks answer-free geometry binding")
        row["benefit_features"]["query_weighted_coverage"] = feature["query_weighted_coverage"]
    return result


def fitted(study, model, rows, statistic, menu):
    from . import compression_policy_v10 as policy
    identity = {
        "split": "fit", "model": model, "study_sha256": study["sha256"],
        "checkpoint_sha256": file_digest(study["models"][model]["checkpoint"]),
        "adapter": study["models"][model]["adapter"],
    }
    key = digest({"rows": rows, "identity": identity, "statistic": statistic,
                  "menu": list(menu), "policy_source": file_digest(policy.__file__)})
    path = Path(study["output_root"]) / model / "development" / (key + ".json")
    if path.exists():
        return checked(path)
    shortlist = policy.fit_shortlist(rows, identity=identity, statistic=statistic,
                                    keep_percentages=menu)
    result = seal({"format": FORMAT + "_development_fit", "study_sha256": study["sha256"],
        "model": model, "statistic": statistic, "keep_percentages": list(menu),
        "data_sha256": digest(rows), "policy_source_sha256": file_digest(policy.__file__),
        "shortlist": shortlist, "evidence": "development, not independent confirmation"})
    write_immutable(path, result)
    return result


def coverage_needed(fits, minimum_saving_pp=2.):
    """More adaptive settings alone do not establish useful compression."""
    for model in MODELS:
        shortlist = fits[model]["shortlist"]
        indexed = {r["policy_id"]: r for r in shortlist["diagnostics"]["candidates"]}
        fixed = indexed.get(shortlist["selected_fixed"])
        adaptive = indexed.get(shortlist["selected_adaptive"])
        if fixed is None or adaptive is None:
            return True
        saving = 100 * (fixed["cost"] - adaptive["cost"]) / adaptive["full_cost"]
        if saving < minimum_saving_pp:
            return True
    return False


def freeze_candidates(study, fits, *, optional_status):
    from . import compression_policy_v10 as policy
    path = Path(study["output_root"]) / "coordinator/candidates.json"
    if path.exists():
        result = checked(path)
        verify_candidates(study, result)
        return result
    shared = policy.select_shared_statistic({
        statistic: {model: result["shortlist"] for model, result in per_model.items()}
        for statistic, per_model in fits.items()
    })
    statistic = shared.get("selected_statistic", shared.get("statistic"))
    if statistic not in fits:
        raise ValueError("shared-statistic selector did not return an available statistic")
    result = seal({
        "format": FORMAT + "_candidates", "study_sha256": study["sha256"],
        "statistic": statistic, "selection": shared, "evidence_stage": "development_shortlist",
        "models": {model: fits[statistic][model] for model in MODELS},
        "optional_status": optional_status, "frozen_utc_epoch": time.time(),
        "final_labels_opened": False,
        "source_hashes": {str(Path(p).resolve()): file_digest(p)
                          for p in (__file__, policy.__file__)},
    })
    write_immutable(path, result)
    return result


def verify_candidates(study, frozen):
    if (frozen.get("sha256") != digest({k: v for k, v in frozen.items() if k != "sha256"})
            or frozen.get("study_sha256") != study["sha256"]
            or set(frozen.get("models", {})) != set(MODELS)
            or frozen.get("final_labels_opened") is not False):
        raise ValueError("frozen candidate identity changed")
    for name, expected in frozen["source_hashes"].items():
        if file_digest(name) != expected:
            raise ValueError("frozen candidate source changed: " + name)


def plan_names():
    from . import compression_execution_v10 as execution
    if hasattr(execution, "plan_names"):
        return execution.plan_names()
    raise ValueError("v10 execution must enumerate its allowed future plan names")


def configure(study):
    from . import compression_operations_v10 as operations
    root = Path(study["output_root"])
    for role in ("clusterb", "clustera"):
        paths = [root / "plans" / name / "plan.json" for name in plan_names()
                 if role == "clusterb" or "llava_next" in name]
        path = root / "coordinator" / (role + "-config.json")
        if path.exists():
            operations.read_config(path)
            continue
        config = operations.build_config(
            study["cwd"], paths, role=role, deadline_utc_epoch=study["deadline_utc_epoch"],
            created_utc_epoch=study["created_utc_epoch"], study_root=root)
        write_immutable(path, config)


def worker_summary(study):
    from .worker import read_plan, _state
    records = []
    for path in sorted((Path(study["output_root"]) / "plans").glob("*/plan.json")):
        plan = read_plan(path)
        for task in plan["tasks"]:
            state = _state(path.parent / "state" / (task["id"] + ".json"))
            if state and state.get("plan_sha256") != plan["sha256"]:
                raise ValueError("worker state belongs to a different plan")
            records.append({"task_id": task["id"], "plan": str(path),
                "status": state.get("status", "queued"), "failures": state.get("failures", 0),
                "log": state.get("log"), "reason": state.get("reason")})
    return {"counts": dict(Counter(r["status"] for r in records)), "tasks": records,
            "terminal_failures": [r for r in records if r["status"] in {"failed", "blocked"}]}


def data_admission(study):
    root = Path(study["output_root"])
    frozen = root / "data/admission-decision.json"
    if frozen.exists():
        return checked(frozen)
    path = root / "data/status.json"
    status = checked(path) if path.exists() else seal({"domains": {}, "status": "audit_pending"})
    domains = status.get("domains", {})
    eligible = all(domains.get(d, {}).get("eligible") is True for d in study["fresh_datasets"])
    closed = time.time() >= study["data_admission_deadline_utc_epoch"]
    ready = eligible and not closed
    result = seal({"format": FORMAT + "_admission", "study_sha256": study["sha256"],
        "eligible": ready, "status": "admitted" if ready else
        "confirmation_unavailable" if closed else "confirmation_blocked" if status.get("status") == "blocked" else "audit_pending",
        "domains": domains, "audit_sha256": status["sha256"],
        "admission_window_closed": closed, "development_unaffected": True})
    if ready or closed:
        write_immutable(frozen, result)
    return result


def optional_admitted(study, name, complete, *, now=None):
    """Only complete branches audited before the cutoff can affect selection."""
    now = time.time() if now is None else now
    path = Path(study["output_root"]) / "coordinator" / (name + "-audited.json")
    if path.exists():
        receipt = checked(path)
        if receipt["study_sha256"] != study["sha256"]:
            raise ValueError("optional completion receipt belongs to another study")
        return bool(complete and receipt["audited_utc_epoch"] <= study["candidate_freeze_deadline_utc_epoch"])
    if not complete or now > study["candidate_freeze_deadline_utc_epoch"]:
        return False
    write_immutable(path, seal({"study_sha256": study["sha256"], "branch": name,
                               "audited_utc_epoch": now, "complete_cohort_verified": True}))
    return True


def _number(value, digits=2):
    return "not run" if value is None else f"{value:.{digits}f}"


def write_tables(study, status):
    """Keep descriptive fitted-policy views separate from OOF/confirmation."""
    root = Path(study["output_root"])
    lines = ["# Pooled adaptive budgeting v10", "",
        "Development results below reuse fitting data. They are not independent accuracy confirmation.",
        "One scalar, threshold and retention pair per model; dataset name is not a policy input.", "",
        "## Development validation", "",
        "| Model | Statistic | Fixed rule | Adaptive rule | Eligible settings | Status |",
        "|---|---|---|---|---:|---|"]
    fits = status.get("fits", {})
    for statistic, models in fits.items():
        for model, result in models.items():
            shortlist = result["shortlist"]
            candidates = shortlist["diagnostics"]["candidates"]
            lines.append(f"| {model} | {statistic} | {shortlist['selected_fixed'] or 'none'} | "
                f"{shortlist['selected_adaptive'] or 'none'} | "
                f"{sum(bool(c['eligible']) for c in candidates)} | development only |")
    lines += ["", "## Quality and compression", "",
        "| Model | Dataset | Method | Questions | Full-token quality | Method quality | Loss (pp) | Image retention | Matched static | DivPrune |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    panels = status.get("panels", [])
    for p in panels:
        c = p["comparison"]
        lines.append(f"| {p['model']} | {p['dataset']} | {p['policy_id']} | {p['questions']} | "
            f"{_number(c.get('full_quality_percent'))} | {_number(c.get('candidate_quality_percent'))} | "
            f"{_number(c.get('mean_loss_pp'))} | {_number(p['retention_percent'])}% | not run | not run |")
    lines += ["", "## Allocation and measured cache capacity", "",
        "| Model | Dataset | Method | Retention distribution | Intentional full | Emergency full | Mean KV reduction | Samples with KV reduction | GPU peak reduction |",
        "|---|---|---|---|---:|---:|---:|---:|---|"]
    for p in panels:
        m = p["comparison"]["memory"]
        allocation = ", ".join(f"{k}%: {n}" for k, n in sorted(p["retention_counts"].items(), key=lambda x: int(x[0])))
        lines.append(f"| {p['model']} | {p['dataset']} | {p['policy_id']} | {allocation} | "
            f"{p['intentional_full']} | {p['emergency_full']} | "
            f"{_number(m.get('mean_per_input_allocated_kv_reduction_percent'))}% | "
            f"{_number(p.get('kv_reduced_fraction_percent'))}% | not controlled yet |")
    if any(status.get("matched", {}).values()):
        lines += ["", "## Exact-budget physical comparison", "",
            "The following is the old screening subset, already used in fitting. All compressed methods have exactly matched aggregate image-token counts.", "",
            "| Model | Dataset | Questions | All tokens | Selected rule | AsymVLM static | DivPrune | CDPruner adapted | Selected − static (pp) | Paired interval |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for result in status["matched"].values():
            for p in (result or {}).get("panels", []):
                methods = p["methods"]
                contrast = p["contrasts"]["matched_static"]
                interval = (f"[{_number(contrast.get('ci_lower_pp'))}, {_number(contrast.get('ci_upper_pp'))}]"
                            if contrast.get("ci_lower_pp") is not None else "insufficient groups")
                scores = [_number(methods.get(name, {}).get("candidate_quality_percent"))
                          for name in ("full", "selected", "matched_static", "matched_divprune", "matched_cdpruner")]
                lines.append(f"| {p['model']} | {p['dataset']} | {methods['full']['questions']} | " +
                             " | ".join(scores) + f" | {_number(contrast['delta_pp'])} | {interval} |")
    if any(status.get("systems", {}).values()):
        lines += ["", "## Controlled systems measurements", "",
                  "Sequential batch-one measurements include the live scorer/statistic/budget computation. Detailed per-method timing, memory and cache results are in reports/*-systems.json."]
    lines += ["", "All quality scores use the dataset's native metric, normalized to 0–100.",
        "Mean KV reduction uses per-input allocated-cache measurements, not model-weight memory or GPU peak.",
        "Fixed comparison settings are not claimed to have the adaptive token budget until exact matched inference completes.",
        "Repeated validation folds do not increase the independent sample count.", "",
        "## Independent evidence", "",
        f"- Admission: {status.get('admission', {}).get('status', 'pending')}.",
        "- DocVQA and MME remain development evidence; no independent final cohorts are currently admitted.",
        f"- Worker states: {json.dumps(status.get('workers', {}).get('counts', {}), sort_keys=True)}.",
        f"- Overall state: {status['status']}.",
        "- Paired bounds and full numerical diagnostics are recorded in status.json and the immutable fit artifacts.",
        ""]
    _atomic_text(root / "reports/TABLES.md", "\n".join(lines))


def development_panels(study, fits, caches, extra_physical):
    from . import compression_policy_v10 as policy
    panels = []
    for model, result in fits.items():
        cache = caches[model]
        physical = list(cache["physical_rows"]) + list(extra_physical.get(model) or [])
        index = projection.physical_index(physical)
        rows = cache["fit_rows"]
        for p in result["shortlist"]["policies"]:
            if p["policy_id"] not in {result["shortlist"]["selected_fixed"], result["shortlist"]["selected_adaptive"]}:
                continue
            # Labels never enter predict_counts.
            answer_free = [{k: v for k, v in r.items() if k != "quality_by_count"} for r in rows]
            assignment = policy.predict_counts(p, answer_free, allow_development_overlap=True)
            for dataset in DATASETS:
                group = [r for r in rows if dataset_family(r["dataset"]) == dataset]
                counts = {r["row_hash"]: assignment["counts"][r["row_hash"]] for r in group}
                full_counts = {r["row_hash"]: r["input_visual_tokens"] for r in group}
                full = projection.projected_rows(group, index, full_counts, "full")
                measured = projection.projected_rows(group, index, counts, p["policy_id"],
                                                      assignments=assignment["rows"])
                comparison = reporting.compare_to_full(
                    measured, full, expected_request_ids=list(counts), evidence="development")
                allocations = [assignment["rows"][r["row_hash"]] for r in group]
                pairs = list(zip(measured, full))
                reduced = [a.get("metrics", {}).get("cache", {}).get("kv_capacity_bytes") <
                           b.get("metrics", {}).get("cache", {}).get("kv_capacity_bytes")
                           for a, b in pairs
                           if a.get("metrics", {}).get("cache", {}).get("kv_capacity_bytes") is not None
                           and b.get("metrics", {}).get("cache", {}).get("kv_capacity_bytes") is not None]
                panels.append({"model": model, "dataset": dataset, "policy_id": p["policy_id"],
                    "questions": len(group), "independent_groups": len({r["group_id"] for r in group}),
                    "comparison": comparison, "evidence": "development fit-reuse only",
                    "retention_percent": 100 * sum(counts.values()) / sum(full_counts.values()),
                    "retention_counts": dict(Counter(str(a["keep_percent"]) for a in allocations)),
                    "intentional_full": sum(a["keep_percent"] == 100 and not a["fallback"] for a in allocations),
                    "emergency_full": sum(a["fallback"] for a in allocations),
                    "kv_reduced_fraction_percent": 100 * sum(reduced) / len(reduced) if len(reduced) == len(group) else None})
    return panels


def selected_policy(bundle):
    """Show the best adaptive rule explicitly, even if a fixed rule is better."""
    shortlist = bundle["shortlist"]
    selected = shortlist["selected_adaptive"] or shortlist["selected_fixed"]
    return next(p for p in shortlist["policies"] if p["policy_id"] == selected)


def publish_matched(study, model, frozen, cache):
    from . import compression_execution_v10 as execution
    from . import compression_policy_v10 as policy
    root = Path(study["output_root"])
    stage = "matched-screening"
    path = root / model / "schedules" / (stage + "-execution.json")
    if path.exists():
        value = checked(path)
        if value["candidates_sha256"] != frozen["sha256"]:
            raise ValueError("matched controls belong to another frozen shortlist")
        execution.publish_schedule(study, model, stage, "screening", value)
        return path
    p = selected_policy(frozen["models"][model])
    _, _, metadata, raw = execution.development_features(study, model, "screening")
    features = [{k: v for k, v in r.items() if k != "quality_by_count"}
                for r in cache["fit_rows"] if r["row_hash"] in raw]
    assigned = policy.predict_counts(p, features, allow_development_overlap=True)
    rows, totals = {}, {}
    for dataset in DATASETS:
        group = [r for r in features if dataset_family(r["dataset"]) == dataset]
        target = sum(assigned["counts"][r["row_hash"]] for r in group)
        # This comparator uses no statistic. Reconcile proportional counts
        # deterministically, exactly to the adaptive total within each dataset.
        uniform = geometry.uniform_at_visual_total([
            {k: v for k, v in r.items() if k != "benefit_features"} for r in group], target)
        if not uniform["feasible"] or uniform["token_error"] != 0:
            raise ValueError("exact matched uniform assignment failed")
        totals[dataset] = target
        for r in group:
            request, n = r["row_hash"], r["input_visual_tokens"]
            adaptive, fixed = assigned["counts"][request], uniform["counts"][request]
            cells = [{"key": "full", "selector": "full", "selected_count": n, "mode": "full"},
                {"key": "selected", "selector": "full" if adaptive == n else "scorer",
                 "selected_count": adaptive, "mode": "adaptive" if p["kind"] == "adaptive" else "uniform",
                 "policy_sha256": p["sha256"], "full_fallback": assigned["rows"][request]["fallback"]},
                {"key": "matched_static", "selector": "full" if fixed == n else "scorer",
                 "selected_count": fixed, "mode": "exact_aggregate_matched_uniform"},
                {"key": "matched_divprune", "selector": "full" if fixed == n else "divprune",
                 "selected_count": fixed, "mode": "exact_aggregate_matched_uniform"}]
            if model == "llava_next":
                cells.append({"key": "matched_cdpruner", "selector": "full" if fixed == n else "cdpruner",
                              "selected_count": fixed, "mode": "exact_aggregate_matched_uniform"})
            rows[request] = {**matrix.planning._geometry(raw[request]), "cells": cells}
    schedule = seal({"format": FORMAT + "_matched_schedule", "study_sha256": study["sha256"],
        "model": model, "split": "screening", "candidate": "current_rank32",
        "manifest_sha256": study["manifest_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
        "cache_contract": metadata["cache_contract"], "rows": rows,
        "candidates_sha256": frozen["sha256"], "policy_sha256": p["sha256"],
        "selected_policy_kind": p["kind"], "statistic": frozen["statistic"],
        "dataset_token_totals": totals, "answers_used_to_choose_counts": False,
        "evidence": "physical development comparison, not independent confirmation"})
    execution.publish_schedule(study, model, stage, "screening", schedule)
    return path


def matched_report(study, model):
    from . import compression_execution_v10 as execution
    rows = execution.collect_stage(study, model, "matched-screening")
    if rows is None:
        return None
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        groups[dataset_family(row["dataset"])][row["method"]].append(row)
    panels = []
    for dataset, methods in groups.items():
        full = methods["full"]
        ids = [r["request_id"] for r in full]
        quality = {name: reporting.compare_to_full(value, full, expected_request_ids=ids)
                   for name, value in methods.items()}
        contrasts = {name: projection.compare_matched(methods["selected"], methods[name], expected_request_ids=ids)
                     for name in methods if name.startswith("matched_")}
        panels.append({"model": model, "dataset": dataset, "methods": quality,
                       "contrasts": contrasts, "evidence": "development only"})
    result = seal({"format": FORMAT + "_matched_report", "study_sha256": study["sha256"],
                   "model": model, "panels": panels, "evidence": "development only"})
    write_immutable(Path(study["output_root"]) / "reports" / (model + "-matched.json"), result)
    return result


def advance(study):
    from . import compression_execution_v10 as execution
    verify(study)
    root = Path(study["output_root"])
    admission = data_admission(study)
    caches = {m: cached_development(study, m) for m in MODELS}
    deadline = time.time() >= study["deadline_utc_epoch"]
    frozen_path = root / "coordinator/candidates.json"
    frozen = checked(frozen_path) if frozen_path.exists() else None
    if frozen:
        verify_candidates(study, frozen)
    selection_open = time.time() < study["candidate_freeze_deadline_utc_epoch"]
    if not deadline and frozen is None and selection_open:
        for model in MODELS:
            execution.prepare(study, model)
    low = {m: execution.collect(study, m) for m in MODELS}
    expanded = (frozen["optional_status"]["expanded_menu_complete"] if frozen else
                optional_admitted(study, "low-retention", all(low[m] is not None for m in MODELS)))
    if expanded and not all(low[m] is not None for m in MODELS):
        raise ValueError("frozen complete low-retention evidence disappeared")
    menu = EXPANDED_MENU if expanded else BASE_MENU
    rows = {m: expanded_rows(caches[m], low[m]) if expanded else copy.deepcopy(caches[m]["fit_rows"])
            for m in MODELS}
    fits = ({frozen["statistic"]: frozen["models"]} if frozen else
            {STATISTICS[0]: {m: fitted(study, m, rows[m], STATISTICS[0], menu) for m in MODELS}})
    needs_coverage = (frozen["statistic"] == STATISTICS[1] if frozen else
                     coverage_needed(fits[STATISTICS[0]], study["coverage_trigger_saving_pp"]))
    coverage_complete = False
    if needs_coverage:
        if not deadline and frozen is None and selection_open:
            for model in MODELS:
                execution.prepare_features(study, model)
        coverage = {m: execution.collect_features(study, m) for m in MODELS}
        coverage_complete = (all(coverage[m] is not None for m in MODELS) if frozen else
            optional_admitted(study, "coverage", all(coverage[m] is not None for m in MODELS)))
        if coverage_complete and frozen is None:
            coverage_rows = {m: with_coverage(rows[m], coverage[m]) for m in MODELS}
            fits[STATISTICS[1]] = {m: fitted(study, m, coverage_rows[m], STATISTICS[1], menu) for m in MODELS}
    optional = frozen["optional_status"] if frozen else {
        "expanded_menu_complete": expanded, "coverage_requested": needs_coverage,
        "coverage_complete": coverage_complete}
    ready = expanded and (not needs_coverage or coverage_complete)
    if frozen is None and (ready or time.time() >= study["candidate_freeze_deadline_utc_epoch"]):
        frozen = freeze_candidates(study, fits, optional_status=optional)
    selected_fits = frozen["models"] if frozen else fits[STATISTICS[0]]
    if frozen and frozen["statistic"] == STATISTICS[1]:
        for model in MODELS:
            caches[model] = {**caches[model], "fit_rows": with_coverage(caches[model]["fit_rows"], coverage[model])}
    panels = development_panels(study, selected_fits, caches, low if expanded else {})
    matched, systems = {}, {}
    if frozen:
        from . import compression_systems_v10 as timing
        for model in MODELS:
            if not deadline:
                schedule_path = publish_matched(study, model, frozen, caches[model])
                timing.publish(study, model, schedule_path, selected_policy(frozen["models"][model]),
                               Path(study["old_study"]).parent / model / "screening_features")
            matched[model] = matched_report(study, model)
            measured = timing.collect(study, model)
            if measured is not None:
                systems[model] = timing.summarize(measured)
                atomic_json(root / "reports" / (model + "-systems.json"), seal(systems[model]))
            else:
                systems[model] = None
    workers = worker_summary(study)
    state = "development_running"
    fatal = [r for r in workers["terminal_failures"] if
             "matched-screening" in r["task_id"] or "systems-" in r["task_id"] or
             frozen is not None and ((expanded and "p010-p020" in r["task_id"]) or
                                     (frozen["statistic"] == STATISTICS[1] and "coverage" in r["task_id"]))]
    workers["fatal_failures"] = fatal
    development_complete = bool(frozen and all(matched.get(m) is not None and systems.get(m) is not None for m in MODELS))
    if fatal:
        state = "worker_failure"
    elif frozen:
        if development_complete:
            state = ("development_complete_confirmation_unavailable" if admission["status"] == "confirmation_unavailable"
                     else "fresh_execution_required" if admission["eligible"] else "awaiting_fresh_admission")
        else:
            state = "matched_and_systems_running"
    if deadline and not development_complete:
        state = "deadline_reached"
    status = seal({"format": FORMAT + "_status", "study_sha256": study["sha256"],
        "status": state, "complete": False, "development_complete": development_complete,
        "scientific_confirmation_complete": False, "updated_utc_epoch": time.time(),
        "fits": fits, "candidates": frozen, "optional": optional, "admission": admission,
        "panels": panels, "matched": matched, "systems": systems,
        "workers": workers, "fresh_evaluation_started": False})
    atomic_json(root / "coordinator/status.json", status)
    write_tables(study, status)
    return status


def watch(study, interval=60):
    path = Path(study["output_root"]) / "coordinator/driver.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                result = advance(study)
            except Exception as error:
                atomic_json(path.parent / "error.json", seal({
                    "status": "driver_failed", "study_sha256": study["sha256"],
                    "updated_utc_epoch": time.time(), "error": type(error).__name__ + ": " + str(error)}))
                raise
            print(json.dumps({"status": result["status"], "workers": result["workers"]["counts"],
                              "updated_utc_epoch": result["updated_utc_epoch"]}), flush=True)
            if result["status"] in {"deadline_reached", "worker_failure",
                                    "development_complete_confirmation_unavailable"} or time.time() >= study["deadline_utc_epoch"]:
                return result
            time.sleep(min(interval, max(0, study["deadline_utc_epoch"] - time.time())))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "configure", "advance", "watch", "status"), required=True)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--hours", type=float, default=24.)
    parser.add_argument("--interval", type=float, default=60.)
    args = parser.parse_args(argv)
    if not 1 <= args.interval <= 60:
        parser.error("interval must be between 1 and 60 seconds")
    if args.phase == "prepare":
        result = prepare(args.cwd, hours=args.hours)
    else:
        study = checked(args.cwd / ROOT / "study.json")
        if args.phase == "status":
            result = read(Path(study["output_root"]) / "coordinator/status.json")
        elif args.phase == "configure":
            configure(study)
            result = {"configured": True, "root": study["output_root"]}
        else:
            result = watch(study, args.interval) if args.phase == "watch" else advance(study)
    print(json.dumps(result if args.phase in {"prepare", "status"} else
                     {"status": result.get("status"), "root": str(args.cwd / ROOT)}, indent=2))


if __name__ == "__main__":
    main()
