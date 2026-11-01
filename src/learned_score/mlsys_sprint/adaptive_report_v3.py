"""Audited paired quality/memory tables for the physical adaptive v3 study.

Partial matched intersections are explicitly provisional. Exact-budget claims
require a complete dataset cohort, not just a convenient completed subset.
This reporter never fits policies, opens reference answers, or launches GPUs.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .adaptive_data_v3 import DEFAULT_OUTPUT, dataset_family, load_manifest
from .adaptive_report_v1 import _atomic_text
from .state import atomic_json, digest, file_digest


FULL = "full_full_000"
ROLES = ("adaptive", "uniform", "divprune", "cdpruner")
LABELS = {"full": "All image tokens", "adaptive": "AsymVLM adaptive",
          "uniform": "AsymVLM uniform", "divprune": "DivPrune adaptation",
          "cdpruner": "CDPruner adaptation"}
TARGETS = (40, 50, 60, 70)
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261016
REQUIRED_STAGES = ("calibration", "screening_features", "evaluation_features", "screening", "evaluation")
COMPLETE_SCAN_STATUSES = {"complete", "complete_feasible_subset"}


def _read(path):
    return json.loads(Path(path).read_text())


def cell(role, target):
    if role == "full":
        return FULL
    selector = "scorer" if role in {"adaptive", "uniform"} else role
    allocation = "adaptive" if role == "adaptive" else "uniform"
    return f"{selector}_{allocation}_{target:03d}"


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"nonfinite/missing {name}")
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid {name}")
    return value


def _sealed(path):
    payload = _read(path)
    if payload.get("sha256") != digest({k: v for k, v in payload.items() if k != "sha256"}):
        raise ValueError(f"changed content-addressed artifact: {path}")
    return payload


def quality_summary(rows, metric):
    rows = list(rows)
    if not rows:
        raise ValueError("cannot score an empty matched cohort")
    for row in rows:
        value = _finite(row["quality_score"], "quality")
        if not 0 <= value <= 1:
            raise ValueError("quality must be normalized to [0,1]")
    if metric != "mme":
        return {"score_percent": 100 * float(np.mean([row["quality_score"] for row in rows])),
                "metric": metric, "questions": len(rows), "groups": len({row["group_id"] for row in rows})}
    categories = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if not row.get("category"):
            raise ValueError("MME category is required")
        categories[row["category"]][row["group_id"]].append(row["quality_score"])
    category_scores = {}
    for category, groups in sorted(categories.items()):
        if any(len(values) != 2 for values in groups.values()):
            raise ValueError("MME official metrics require complete two-question image pairs")
        question_accuracy = float(np.mean([value for values in groups.values() for value in values]))
        pair_accuracy = float(np.mean([all(value == 1 for value in values) for values in groups.values()]))
        category_scores[category] = {"question_accuracy_percent": question_accuracy * 100,
                                     "paired_accuracy_percent": pair_accuracy * 100,
                                     "official_score": (question_accuracy + pair_accuracy) * 100,
                                     "images": len(groups)}
    total = sum(item["official_score"] for item in category_scores.values())
    return {"score_percent": total / (2 * len(categories)), "metric": "mme_category_macro_percent",
            "official_sum_points": total, "official_maximum_points": 200 * len(categories),
            "category_results": category_scores, "questions": len(rows),
            "groups": sum(len(groups) for groups in categories.values())}


def paired_ci(candidate, baseline, metric, *, repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Question-weighted clustered CI; MME resamples image pairs per category."""
    if repetitions < 100:
        raise ValueError("bootstrap requires at least 100 repetitions")
    a = {row["request_id"]: row for row in candidate}
    b = {row["request_id"]: row for row in baseline}
    if not a or set(a) != set(b) or len(a) != len(candidate) or len(b) != len(baseline):
        raise ValueError("paired bootstrap requires identical nonduplicate request sets")
    for request in a:
        if (a[request]["group_id"] != b[request]["group_id"]
                or a[request].get("category") != b[request].get("category")):
            raise ValueError("paired group/category mismatch")
    point = quality_summary(candidate, metric)["score_percent"] - quality_summary(baseline, metric)["score_percent"]
    groups = defaultdict(list)
    for request in sorted(a):
        groups[a[request]["group_id"]].append(request)
    base = {"delta_pp": point, "questions": len(a), "groups": len(groups),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed, "confidence": 0.95}
    if len(groups) < 2:
        return {**base, "ci_lower_pp": None, "ci_upper_pp": None, "status": "insufficient_independent_groups"}
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repetitions)
    if metric == "mme":
        by_category = defaultdict(list)
        for group, requests in sorted(groups.items()):
            if len(requests) != 2:
                raise ValueError("MME bootstrap requires intact image pairs")
            category = a[requests[0]]["category"]
            av = [a[key]["quality_score"] for key in requests]
            bv = [b[key]["quality_score"] for key in requests]
            # Each pair contributes half question accuracy plus half pair accuracy.
            delta = .5 * (np.mean(av) - np.mean(bv)) + .5 * (float(all(v == 1 for v in av)) - float(all(v == 1 for v in bv)))
            by_category[category].append(delta)
        if any(len(values) < 2 for values in by_category.values()):
            return {**base, "ci_lower_pp": None, "ci_upper_pp": None,
                    "status": "insufficient_image_pairs_per_category"}
        for category, values in sorted(by_category.items()):
            values = np.asarray(values)
            for start in range(0, repetitions, 256):
                stop = min(start + 256, repetitions)
                draws = rng.integers(0, len(values), size=(stop-start, len(values)))
                estimates[start:stop] += values[draws].mean(axis=1) / len(by_category)
        resampling = "image_pairs_stratified_by_category"
    else:
        sums = np.asarray([sum(a[key]["quality_score"] - b[key]["quality_score"] for key in keys)
                           for _, keys in sorted(groups.items())])
        counts = np.asarray([len(keys) for _, keys in sorted(groups.items())])
        for start in range(0, repetitions, 256):
            stop = min(start + 256, repetitions)
            draws = rng.integers(0, len(sums), size=(stop-start, len(sums)))
            estimates[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        resampling = "document_or_image_groups"
    low, high = np.quantile(estimates * 100, [.025, .975])
    return {**base, "ci_lower_pp": float(low), "ci_upper_pp": float(high), "status": "computed",
            "resampling_unit": resampling}


def _audit_cache(row, contract, assigned):
    n = _integer(row.get("input_visual_tokens"), "visual count", 1)
    p = _integer(row.get("input_text_tokens"), "text count")
    kept = _integer(row.get("retained_visual_tokens"), "retained count", 1)
    if kept > n:
        raise ValueError("retained visual count exceeds original")
    if row["allocation"] == "full":
        if kept != n:
            raise ValueError("all-token baseline was pruned")
    elif assigned is None or kept != assigned[f"{row['allocation']}_selected_count"] or row.get("assignment") != assigned:
        raise ValueError("physical selection differs from frozen exact assignment")
    if assigned is not None:
        for key, value in (("input_visual_tokens", n), ("input_text_tokens", p)):
            if key in assigned and assigned[key] != value:
                raise ValueError("physical token geometry differs from frozen assignment")
    capacity = kept + p + contract["reserved_generated_slots"]
    kv = capacity * contract["kv_bytes_per_slot"]
    owned = capacity * contract["total_owned_bytes_per_slot"]
    runtime = row.get("metrics", {})
    cache = runtime.get("cache", {})
    if (cache.get("kv_capacity_bytes") != kv or cache.get("total_owned_bytes") != owned
            or cache.get("source_metadata_bytes") != capacity * contract["source_metadata_bytes_per_slot"]
            or cache.get("scratch_bytes") != 0 or cache.get("compaction_events") != 0
            or row.get("expected_cache_bytes") != owned or runtime.get("peak_cache_bytes") != owned):
        raise ValueError("measured allocated-cache bytes disagree with count/geometry")
    _finite(runtime.get("full_request_peak_allocated_bytes"), "full request GPU peak")


def scan_model(root, name, spec, manifest, *, world_size=16, targets=TARGETS, phase="evaluation"):
    root = Path(root)
    if phase not in {"evaluation", "screening"}:
        raise ValueError("only physical evaluation/screening panels can be reported")
    physical = root / name / phase
    active_targets = tuple(target for target in targets if phase != "screening" or target in {60, 70})
    active_roles = ROLES if phase == "evaluation" else ("adaptive", "uniform")
    entries = sorted(manifest.splits[phase], key=lambda row: (row["dataset"], row["row_hash"]))
    expected = {row["row_hash"]: row for row in entries}
    assignment_path = root / name / f"{phase}_assignments.json"
    assignments = _sealed(assignment_path) if assignment_path.exists() else None
    if assignments is not None:
        if (assignments.get("manifest_sha256") != manifest.file_sha256 or assignments.get("split") != phase
                or set(assignments.get("targets", {})) - {str(target) for target in targets}):
            raise ValueError("assignment manifest/split/targets mismatch")
    target_map = assignments.get("targets", {}) if assignments else {}
    grid = {FULL: ("full", 0, "full", "full")} if phase == "evaluation" else {}
    for target in active_targets:
        for role in active_roles:
            grid[cell(role, target)] = (role, target, "scorer" if role in {"adaptive", "uniform"} else role,
                                       "adaptive" if role == "adaptive" else "uniform")
    expected_keys = {(request, FULL) for request in expected} if phase == "evaluation" else set()
    infeasible = {}
    for target in active_targets:
        data = target_map.get(str(target))
        if data is None:
            continue
        infeasible[str(target)] = data.get("infeasible_datasets", {})
        if set(data.get("rows", {})) - set(expected):
            raise ValueError("assignment includes requests outside the manifest")
        for request, entry in expected.items():
            present = request in data.get("rows", {})
            unavailable = entry["dataset"] in data.get("infeasible_datasets", {})
            if present == unavailable:
                raise ValueError("assignment must cover every request or explicitly mark its whole dataset infeasible")
        for request in data.get("rows", {}):
            expected_keys.update((request, cell(role, target)) for role in active_roles)
    metadata_by_hash, contexts, ranks, complete_ranks = {}, set(), [], []
    checkpoint_hash = file_digest(spec["checkpoint"]) if Path(spec["checkpoint"]).is_file() else None
    for rank in range(world_size):
        path, marker = physical / f"rank{rank}.metadata.json", physical / f"rank{rank}.complete.json"
        if not path.exists():
            if marker.exists():
                raise ValueError("completion marker without metadata")
            continue
        meta = _read(path)
        if (assignments is None or meta.get("format") != "mlsys_adaptive_experiment_v3"
                or meta.get("manifest_sha256") != manifest.file_sha256 or meta.get("phase") != phase
                or meta.get("split") != phase or meta.get("world_size") != world_size or meta.get("limit") is not None
                or meta.get("checkpoint_sha256") != checkpoint_hash
                or meta.get("assignments_sha256") != file_digest(assignment_path)
                or assignments.get("checkpoint_sha256") != checkpoint_hash
                or meta.get("cache_contract") != assignments.get("cache_contract")
                or meta.get("backbone", {}).get("model") != spec["model_id"]
                or not meta.get("source_sha256") or meta.get("max_new_tokens") != 128):
            raise ValueError("runtime metadata differs from frozen study")
        allowed_cells = ({FULL} if phase == "evaluation" else set()) | {
            cell(role, target) for target in active_targets if str(target) in target_map for role in active_roles}
        if set(meta.get("cells", [])) != allowed_cells:
            raise ValueError("runtime cells differ from assignment targets")
        identity = digest(meta)
        metadata_by_hash[identity] = meta
        contexts.add(digest({key: value for key, value in meta.items() if key != "hardware"}))
        ranks.append(rank)
        if marker.exists():
            completion = _read(marker)
            ids = {row["row_hash"] for row in entries[rank::world_size]}
            result_count = sum(request in ids for request, _ in expected_keys)
            if (completion.get("identity_sha256") != identity or completion.get("requests") != len(ids)
                    or completion.get("results") != result_count):
                raise ValueError("completion marker counts differ from planned shard")
            complete_ranks.append(rank)
    if len(contexts) > 1:
        raise ValueError("runtime source/configuration changed between shards")
    rows, failures, seen = [], [], set()
    for path in sorted((physical / "rows").glob("*.json")):
        row = _read(path)
        request, method = row.get("request_id"), row.get("method")
        if request not in expected or method not in grid or (request, method) in seen:
            raise ValueError("unexpected/duplicate physical row")
        if path.name != f"{request}.{method}.json" or (request, method) not in expected_keys:
            raise ValueError("physical row lacks its declared assignment")
        seen.add((request, method))
        meta = metadata_by_hash.get(row.get("identity_sha256"))
        entry = expected[request]
        role, target, selector, allocation = grid[method]
        if (meta is None or row.get("format") != "mlsys_adaptive_experiment_v3" or row.get("phase") != phase
                or row.get("model") != spec["model_id"] or row.get("selector") != selector
                or row.get("allocation") != allocation or row.get("percent") != target
                or any(row.get(key) != entry.get(key) for key in ("dataset", "group_id", "image_sha256", "category"))
                or row.get("quality_metric") != entry["metric"] or row.get("generated_budget") != 512
                or row.get("max_new_tokens") != 128 or row.get("hardware") != meta.get("hardware")):
            raise ValueError("physical row identity/context mismatch")
        if row.get("status") == "failed":
            failures.append({"request_id": request, "method": method, "error": row.get("error")})
            continue
        if row.get("status") != "ok":
            raise ValueError("unknown physical row status")
        value = _finite(row.get("quality_score"), "quality score")
        if not 0 <= value <= 1:
            raise ValueError("quality out of range")
        assigned = None if not target else target_map[str(target)]["rows"][request]
        _audit_cache(row, meta["cache_contract"], assigned)
        rows.append(row)
    missing = len(expected_keys) - len(rows)
    all_targets = set(target_map) == {str(target) for target in targets}
    complete = all_targets and not missing and len(complete_ranks) == world_size and not failures
    status = ("complete_feasible_subset" if any(infeasible.values()) else "complete") if complete else "partial"
    phase_progress = {}
    for progress_phase in ("calibration", "screening_features", "evaluation_features", "screening"):
        directory = root / name / progress_phase
        phase_progress[progress_phase] = {"complete_markers": len(list(directory.glob("rank*.complete.json"))),
                                 "features": len(list((directory / "features").glob("*.json"))),
                                 "physical_rows": len(list((directory / "rows").glob("*.json"))),
                                 "feature_mismatch_diagnostics": len(list(directory.glob("rank*.feature_mismatch.json")))}
    summary = {"model": name, "model_id": spec["model_id"], "phase": phase, "status": status,
               "valid_physical_rows": len(rows), "expected_physical_rows": len(expected_keys) if assignments else None,
               "requested_full_matrix_rows": len(expected) * ((phase == "evaluation") + len(active_roles) * len(active_targets)), "missing_rows": missing,
               "complete_ranks": complete_ranks, "present_ranks": ranks, "world_size": world_size,
               "failed_rows": failures, "infeasible_datasets": infeasible, "phase_progress": phase_progress,
               "targets_assigned": sorted(map(int, target_map)), "manifest_sha256": manifest.file_sha256,
               "checkpoint_sha256": checkpoint_hash,
               "assignments_sha256": file_digest(assignment_path) if assignments else None}
    return summary, rows


def memory_summary(rows, baseline, uniform):
    base = {row["request_id"]: row for row in baseline}
    uniform = {row["request_id"]: row for row in uniform}
    if not rows or {row["request_id"] for row in rows} != set(base):
        raise ValueError("memory comparison requires the same nonempty requests")
    kv = [row["metrics"]["cache"]["kv_capacity_bytes"] for row in rows]
    owned = [row["metrics"]["cache"]["total_owned_bytes"] for row in rows]
    full_kv = [base[row["request_id"]]["metrics"]["cache"]["kv_capacity_bytes"] for row in rows]
    full_owned = [base[row["request_id"]]["metrics"]["cache"]["total_owned_bytes"] for row in rows]
    gpu = [row["metrics"]["full_request_peak_allocated_bytes"] for row in rows]
    full_gpu = [base[row["request_id"]]["metrics"]["full_request_peak_allocated_bytes"] for row in rows]
    return {"questions": len(rows), "mean_kv_bytes": float(np.mean(kv)),
            "mean_full_kv_bytes": float(np.mean(full_kv)), "mean_total_owned_cache_bytes": float(np.mean(owned)),
            "mean_full_total_owned_cache_bytes": float(np.mean(full_owned)),
            "kv_reduction_percent_ratio_of_sums": 100 * (1 - sum(kv) / sum(full_kv)),
            "owned_reduction_percent_ratio_of_sums": 100 * (1 - sum(owned) / sum(full_owned)),
            "mean_per_input_kv_reduction_percent": float(np.mean([100 * (1 - a/b) for a,b in zip(kv, full_kv)])),
            "inputs_with_kv_reduction_percent": 100 * sum(a < b for a,b in zip(kv, full_kv)) / len(rows),
            "mean_full_request_peak_gpu_bytes": float(np.mean(gpu)),
            "gpu_peak_reduction_percent_ratio_of_sums": 100 * (1 - sum(gpu)/sum(full_gpu)) if sum(full_gpu) else None,
            "inputs_with_gpu_peak_reduction_percent": 100 * sum(a < b for a,b in zip(gpu, full_gpu)) / len(rows),
            "inputs_with_less_kv_than_uniform_percent": 100 * sum(
                row["metrics"]["cache"]["kv_capacity_bytes"] < uniform[row["request_id"]]["metrics"]["cache"]["kv_capacity_bytes"]
                for row in rows) / len(rows),
            "mean_visual_retention_percent": float(np.mean([100 * row["retained_visual_tokens"] / row["input_visual_tokens"] for row in rows])),
            "distinct_retained_counts": len({row["retained_visual_tokens"] for row in rows}),
            "allocation_overhead_included_in_end_to_end_timing": False,
            "timing_isolated": False}


def comparison_tables(rows, expected_entries, *, targets=TARGETS, repetitions=BOOTSTRAP_REPETITIONS):
    by_dataset = defaultdict(list)
    for entry in expected_entries:
        by_dataset[entry["dataset"]].append(entry)
    lookup = {(row["request_id"], row["method"]): row for row in rows}
    quality, memory = [], []
    for dataset, entries in sorted(by_dataset.items()):
        expected = {entry["row_hash"]: entry for entry in entries}
        metric = entries[0]["metric"]
        for target in targets:
            methods = {"full": FULL, **{role: cell(role, target) for role in ROLES}}
            matched = {request for request in expected if all((request, method) in lookup for method in methods.values())}
            if metric == "mme":
                groups = defaultdict(list)
                for request, entry in expected.items():
                    groups[entry["group_id"]].append(request)
                matched = {request for ids in groups.values() if len(ids) == 2 and set(ids) <= matched for request in ids}
            if not matched:
                continue
            matched_rows = {role: [lookup[request, method] for request in sorted(matched)] for role, method in methods.items()}
            complete = matched == set(expected)
            owned_sums = {role: sum(row["metrics"]["cache"]["total_owned_bytes"] for row in matched_rows[role]) for role in ROLES}
            equal = len(set(owned_sums.values())) == 1
            full_owned = sum(row["metrics"]["cache"]["total_owned_bytes"] for row in matched_rows["full"])
            actual_reduction = 100 * (1 - owned_sums["adaptive"] / full_owned)
            budget_verified = complete and equal and abs(actual_reduction-target) <= .2
            point = {role: quality_summary(selected, metric) for role, selected in matched_rows.items()}
            comparison = {"dataset": dataset_family(dataset), "source_dataset": dataset, "target_kv_reduction_percent": target,
                          "matched_questions": len(matched), "expected_questions": len(expected),
                          "cohort_complete": complete, "status": "complete_cohort" if complete else "provisional_partial_intersection",
                          "evidence_status": sorted({expected[key]["evaluation_stage"] for key in matched}),
                          "exact_matched_budget_verified": budget_verified,
                          "equal_aggregate_owned_cache_bytes_on_observed_rows": equal,
                          "actual_owned_cache_reduction_percent": actual_reduction,
                          "quality": point,
                          "adaptive_minus_uniform": paired_ci(matched_rows["adaptive"], matched_rows["uniform"], metric, repetitions=repetitions),
                          "adaptive_minus_divprune": paired_ci(matched_rows["adaptive"], matched_rows["divprune"], metric, repetitions=repetitions),
                          "adaptive_minus_full": paired_ci(matched_rows["adaptive"], matched_rows["full"], metric, repetitions=repetitions)}
            quality.append(comparison)
            for role in ROLES:
                memory.append({"dataset": comparison["dataset"], "target_kv_reduction_percent": target,
                               "role": role, "method": LABELS[role], "status": comparison["status"],
                               "evidence_status": comparison["evidence_status"],
                               "exact_matched_budget_verified": budget_verified,
                               **memory_summary(matched_rows[role], matched_rows["full"], matched_rows["uniform"])})
    return quality, memory


def screening_comparisons(rows, entries, *, repetitions=BOOTSTRAP_REPETITIONS):
    by_dataset = defaultdict(list)
    for entry in entries:
        by_dataset[entry["dataset"]].append(entry)
    lookup = {(row["request_id"], row["method"]): row for row in rows}
    result = []
    for dataset, expected in sorted(by_dataset.items()):
        for target in (60, 70):
            ids = [entry["row_hash"] for entry in expected
                   if all((entry["row_hash"], cell(role, target)) in lookup for role in ("adaptive", "uniform"))]
            if not ids:
                continue
            a = [lookup[request, cell("adaptive", target)] for request in ids]
            b = [lookup[request, cell("uniform", target)] for request in ids]
            complete = len(ids) == len(expected)
            equal = sum(row["metrics"]["cache"]["total_owned_bytes"] for row in a) == sum(row["metrics"]["cache"]["total_owned_bytes"] for row in b)
            full_owned = sum((row["input_visual_tokens"] + row["input_text_tokens"] + 127)
                             * row["metrics"]["cache"]["total_owned_bytes"]
                             / (row["retained_visual_tokens"] + row["input_text_tokens"] + 127) for row in a)
            actual = 100 * (1-sum(row["metrics"]["cache"]["total_owned_bytes"] for row in a)/full_owned)
            result.append({"dataset": dataset_family(dataset), "target_percent": target,
                           "questions": len(ids), "expected_questions": len(expected), "cohort_complete": complete,
                           "exact_matched_budget_verified": complete and equal and abs(actual-target) <= .2,
                           "actual_owned_cache_reduction_percent": actual,
                           "adaptive_score_percent": quality_summary(a, expected[0]["metric"])["score_percent"],
                           "uniform_score_percent": quality_summary(b, expected[0]["metric"])["score_percent"],
                           "adaptive_minus_uniform": paired_ci(a, b, expected[0]["metric"], repetitions=repetitions),
                           "evidence": "physical_selected_rule_development_screening_not_fresh_confirmation"})
    return result


def allocation_ablation(root, manifest):
    path = Path(root) / "fit_report.json"
    if not path.exists():
        return {"status": "awaiting_calibration_fit", "candidates": {}}
    report = _sealed(path)
    if (report.get("manifest_sha256") != manifest.file_sha256
            or report.get("screening_labels_read") is not False or report.get("evaluation_labels_read") is not False):
        raise ValueError("fit report used forbidden selection labels or a different manifest")
    candidates = {name: {"mean_delta_pp": 100 * _finite(item["mean_delta"], "OOF mean"),
                         "worst_delta_pp": 100 * _finite(item["worst_delta"], "OOF worst"),
                         "domains": item["domains"]}
                  for name, item in report["aggregate_candidates"].items()}
    return {"status": "fitted", "selected_statistic": report["selected_statistic"],
            "evidence": "out_of_fold_interpolated_estimates_not_physical_integer_count_results",
            "definition": "Each calibration image/document is scored only by a policy fitted without that group; quality between measured retention knots is interpolated.",
            "fit_report_sha256": file_digest(path), "candidates": candidates}


def execution_states(root):
    """Surface worker failures even if warmup failed before a result row existed."""
    counts, active, failures = Counter(), [], []
    for path in sorted((Path(root) / "plans").glob("*/state/*.json")):
        state = _read(path)
        status = state.get("status", "unknown")
        counts[status] += 1
        item = {"task": path.stem, "plan": path.parents[1].name, "status": status,
                "state_path": str(path), "reason": state.get("reason"),
                "worker_id": state.get("worker_id"), "failures": state.get("failures", 0)}
        if status in {"running", "retry"}:
            active.append(item)
        if status in {"failed", "blocked"}:
            failures.append(item)
    return {"status_counts": dict(counts), "active_or_retry_records": active,
            "failed_or_blocked_records": failures,
            "note": "Includes preserved superseded queues; a failed historical record does not imply its recovery queue is blocked."}


def stage_completion(root, model, world_size):
    """Check every declared shard marker, including the pre-evaluation stages."""
    result = {}
    for stage in REQUIRED_STAGES:
        directory = Path(root) / model / stage
        ready = len(list(directory.glob("rank*.complete.json"))) == world_size
        for rank in range(world_size):
            marker = directory / f"rank{rank}.complete.json"
            metadata = directory / f"rank{rank}.metadata.json"
            if not marker.is_file() or not metadata.is_file():
                ready = False
                continue
            identity, complete = _read(metadata), _read(marker)
            if complete.get("identity_sha256") != digest(identity):
                raise ValueError(f"stage completion identity mismatch: {model}/{stage}/rank{rank}")
            if identity.get("world_size") != world_size:
                raise ValueError(f"stage completion world-size mismatch: {model}/{stage}/rank{rank}")
        result[stage] = ready
    return result


def completion_status(models, expected_models):
    """Execution can finish with explicit infeasibility; the full matrix cannot."""
    execution_complete = bool(expected_models) and set(models) == set(expected_models) and all(
        value.get("status") in COMPLETE_SCAN_STATUSES
        and value.get("screening_status") in COMPLETE_SCAN_STATUSES
        and all(value.get("stage_completion", {}).get(stage) is True for stage in REQUIRED_STAGES)
        for value in models.values())
    full_matrix_complete = execution_complete and all(
        value["status"] == "complete" and value["screening_status"] == "complete"
        for value in models.values())
    return {"execution_complete": execution_complete, "full_matrix_complete": full_matrix_complete,
            "status": "complete" if full_matrix_complete else "complete_feasible_subset" if execution_complete else "partial"}


def primary_macro_ci(rows_by_model, entries, expected_models, *, target=60,
                     repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED):
    """Shared group draws across models; equal weighting of models/datasets."""
    waiting = {"status": "awaiting_complete_primary_adaptive_uniform_cohorts", "target_percent": target}
    if set(rows_by_model) != set(expected_models):
        return waiting
    by_dataset = defaultdict(list)
    for entry in entries:
        by_dataset[entry["dataset"]].append(entry)
    lookup = {name: {(row["request_id"], row["method"]): row for row in rows}
              for name, rows in rows_by_model.items()}
    rng = np.random.default_rng(seed)
    bootstrap = np.zeros(repetitions)
    domain_points = []
    for dataset, cohort in sorted(by_dataset.items()):
        ids = [entry["row_hash"] for entry in cohort]
        metric = cohort[0]["metric"]
        groups = defaultdict(list)
        for entry in cohort:
            groups[entry["group_id"]].append(entry)
        arrays = {}
        for name in sorted(expected_models):
            if any((request, cell(role, target)) not in lookup[name] for request in ids for role in ("adaptive", "uniform")):
                return waiting
            a = [lookup[name][request, cell("adaptive", target)] for request in ids]
            b = [lookup[name][request, cell("uniform", target)] for request in ids]
            if sum(row["metrics"]["cache"]["total_owned_bytes"] for row in a) != sum(row["metrics"]["cache"]["total_owned_bytes"] for row in b):
                return {"status": "invalid_primary_aggregate_budget_mismatch", "target_percent": target}
            full_owned = sum((row["input_visual_tokens"] + row["input_text_tokens"] + 127)
                             * row["metrics"]["cache"]["total_owned_bytes"]
                             / (row["retained_visual_tokens"] + row["input_text_tokens"] + 127) for row in a)
            actual = 100 * (1-sum(row["metrics"]["cache"]["total_owned_bytes"] for row in a)/full_owned)
            if abs(actual-target) > .2:
                return {"status": "invalid_primary_requested_reduction_mismatch", "target_percent": target}
            domain_points.append(quality_summary(a, metric)["score_percent"] - quality_summary(b, metric)["score_percent"])
            arrays[name] = ({row["request_id"]: row for row in a}, {row["request_id"]: row for row in b})
        contribution = np.zeros(repetitions)
        if metric == "mme":
            categories = defaultdict(list)
            for group, members in sorted(groups.items()):
                if len(members) != 2:
                    raise ValueError("primary MME comparison lacks intact image pairs")
                values = []
                for a, b in arrays.values():
                    av = [a[entry["row_hash"]]["quality_score"] for entry in members]
                    bv = [b[entry["row_hash"]]["quality_score"] for entry in members]
                    values.append(.5 * (np.mean(av)-np.mean(bv)) + .5 * (float(all(v == 1 for v in av))-float(all(v == 1 for v in bv))))
                categories[members[0]["category"]].append(float(np.mean(values)))
            for category, values in sorted(categories.items()):
                if len(values) < 2:
                    return {"status": "insufficient_primary_mme_image_pairs", "target_percent": target}
                values = np.asarray(values)
                for start in range(0, repetitions, 256):
                    stop = min(start+256, repetitions)
                    draw = rng.integers(0, len(values), size=(stop-start, len(values)))
                    contribution[start:stop] += values[draw].mean(axis=1) / len(categories)
        else:
            sums, counts = [], []
            for group, members in sorted(groups.items()):
                sums.append(float(np.mean([sum(a[entry["row_hash"]]["quality_score"]-b[entry["row_hash"]]["quality_score"]
                                              for entry in members) for a,b in arrays.values()])))
                counts.append(len(members))
            if len(sums) < 2:
                return {"status": "insufficient_primary_independent_groups", "target_percent": target}
            sums, counts = np.asarray(sums), np.asarray(counts)
            for start in range(0, repetitions, 256):
                stop = min(start+256, repetitions)
                draw = rng.integers(0, len(sums), size=(stop-start, len(sums)))
                contribution[start:stop] = sums[draw].sum(axis=1)/counts[draw].sum(axis=1)
        bootstrap += contribution / len(by_dataset)
    low, high = np.quantile(bootstrap*100, [.025, .975])
    return {"status": "complete_development_primary_comparison", "target_percent": target,
            "delta_pp": float(np.mean(domain_points)), "ci_lower_pp": float(low), "ci_upper_pp": float(high),
            "models": list(sorted(expected_models)), "datasets": len(by_dataset), "model_dataset_cells": len(domain_points),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "weighting": "equal model and dataset weights; normalized MME category macro percent",
            "resampling": "same document/image draw across models, independent datasets; MME image pairs stratified by category",
            "fresh_confirmation_claim": False}


def _fmt(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}"


def _ci(value):
    return (f"{value['delta_pp']:+.2f} [{_fmt(value['ci_lower_pp'])}, {_fmt(value['ci_upper_pp'])}]"
            if value["ci_lower_pp"] is not None else f"{value['delta_pp']:+.2f} [insufficient groups]")


def render_markdown(report):
    lines = ["# Adaptive per-input vision budgeting: physical results", "",
             f"Updated {report['updated_utc']}. Overall status: **{report['status']}**.", "",
             "These are development/transfer results, not certified fresh confirmation. DivPrune and CDPruner are projected-feature selector adaptations, not their official end-to-end pipelines. All methods use the same protected image/prompt cache and a fixed 512-token generated-text allowance; short answers do not activate text eviction.", "",
             "## Table 1: Accuracy at the requested allocated KV-cache reduction", "",
             "Every row uses the same completed question intersection for all five methods. Partial intersections are provisional and cannot establish the requested aggregate memory match. Differences are percentage points with paired 95% group-bootstrap intervals (10,000 draws).", "",
             "| Model | Dataset | Target reduction | Questions | Status | All image tokens | AsymVLM adaptive | AsymVLM uniform | DivPrune adaptation | CDPruner adaptation | Adaptive − uniform | Adaptive − DivPrune |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|---|"]
    for row in report["quality_table"]:
        q = row["quality"]
        status = "complete; matched" if row["exact_matched_budget_verified"] else "partial/unmatched"
        lines.append(f"| {row['model']} | {row['dataset']} | {row['target_kv_reduction_percent']}% | {row['matched_questions']}/{row['expected_questions']} | {status} | "
                     + " | ".join(_fmt(q[role]["score_percent"]) for role in ("full", *ROLES))
                     + f" | {_ci(row['adaptive_minus_uniform'])} | {_ci(row['adaptive_minus_divprune'])} |")
    if not report["quality_table"]:
        lines.append("| — | No five-method matched physical rows yet | — | 0 | incomplete | — | — | — | — | — | — | — |")
    lines += ["", "DocVQA: ANLS; TextVQA: official VQA accuracy; ChartQA: relaxed correctness, all in percent. MME uses the category mean of (question accuracy + both-questions-correct accuracy)/2, in percent. Official MME summed scores and category details are preserved in the JSON report; MME intervals resample intact image pairs within each category.", "",
              "## Table 2: Memory reduction and how often it occurs", "",
              "Adaptive rows are shown below; the JSON includes every selector. A verified match means adaptive, uniform, DivPrune and CDPruner have identical aggregate allocated-cache bytes on the complete dataset. Owned cache includes KV tensors plus position metadata. GPU peak is a separate quantity, not a cache-saving claim.", "",
              "| Model | Dataset | Target | Questions | All-token KV MiB | Adaptive KV MiB | Adaptive owned-cache MiB | Actual KV reduction | Inputs with lower KV | Inputs below uniform KV | GPU peak reduction | Matched budget |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in report["memory_table"]:
        if row["role"] != "adaptive":
            continue
        lines.append(f"| {row['model']} | {row['dataset']} | {row['target_kv_reduction_percent']}% | {row['questions']} | "
                     f"{row['mean_full_kv_bytes']/2**20:.2f} | {row['mean_kv_bytes']/2**20:.2f} | {row['mean_total_owned_cache_bytes']/2**20:.2f} | "
                     f"{row['kv_reduction_percent_ratio_of_sums']:.2f}% | {row['inputs_with_kv_reduction_percent']:.1f}% | "
                     f"{row['inputs_with_less_kv_than_uniform_percent']:.1f}% | {_fmt(row['gpu_peak_reduction_percent_ratio_of_sums'])}% | "
                     f"{'verified' if row['exact_matched_budget_verified'] else 'not verified'} |")
    if not report["memory_table"]:
        lines.append("| — | No matched physical rows yet | — | 0 | — | — | — | — | — | — | — | not verified |")
    lines += ["", "Memory reductions use the ratio of aggregate bytes, not the mean of per-input percentages. Per-input averages are separately retained in JSON. Timing is not isolated and excludes offline allocation overhead; these runs do not establish a serving-throughput or end-to-end latency improvement.", "", "## Completion and audit", ""]
    if report.get("execution_complete"):
        if report.get("full_matrix_complete"):
            lines.append("All five execution stages and both physical comparison panels are complete for every model; the full requested matrix is available.")
        else:
            lines.append("All feasible scheduled work is finished for every model, including physical screening. Some requested model/dataset/compression cells are explicitly infeasible, so the full requested matrix is not complete; no further runs are pending for those infeasible cells.")
        lines.append("")
    for name, model in report["models"].items():
        if model["status"] == "invalid":
            lines.append(f"- {name}: invalid evidence; {model['error']}")
        else:
            lines.append(f"- {name}: {model['status']}; {model['valid_physical_rows']}/{model['expected_physical_rows'] or 'unassigned'} physical rows; {len(model['complete_ranks'])}/{model['world_size']} completed shards; {len(model['failed_rows'])} failed rows.")
    if report.get("execution", {}).get("status_counts"):
        lines.append("- Queue state records: " + ", ".join(f"{count} {status}" for status, count in sorted(report["execution"]["status_counts"].items()))
                     + ". Preserved failed queues and their replacement queues are both counted; this is not a count of currently blocked experiments.")
    ablation = report.get("allocation_ablation", {})
    lines += ["", "## Allocation-rule development check", ""]
    if ablation.get("status") == "fitted":
        names = {"normalized_entropy": "Entropy (spread of learned patch scores)",
                 "local_shell": "Importance mass around the uniform cutoff",
                 "nearest_neighbor_diversity": "Nearest-neighbor visual-feature diversity"}
        lines.append("The three-statistic comparison below uses held-out calibration groups and interpolated quality between measured token counts. These are development estimates, not physical results at the assigned integer counts. Only the selected rule receives the physical screening comparison below.")
        lines.append("")
        for name, value in sorted(ablation["candidates"].items()):
            selected = " (selected)" if name == ablation["selected_statistic"] else ""
            lines.append(f"- {names.get(name, name)}{selected}: estimated mean adaptive-minus-uniform {value['mean_delta_pp']:+.2f} pp; worst domain {value['worst_delta_pp']:+.2f} pp; {value['domains']} model/dataset/target domains.")
    else:
        lines.append(f"Calibration comparison: {ablation.get('status', 'awaiting_calibration_fit')}.")
    for value in report.get("screening_physical", []):
        state = "complete matched cohort" if value["exact_matched_budget_verified"] else "provisional partial cohort"
        lines.append(f"- Physical selected-rule screening: {value['model']}, {value['dataset']}, {value['target_percent']}% reduction; "
                     f"{value['questions']}/{value['expected_questions']} questions, {state}; adaptive {_fmt(value['adaptive_score_percent'])}, "
                     f"uniform {_fmt(value['uniform_score_percent'])}; difference {_ci(value['adaptive_minus_uniform'])} pp.")
    primary = report.get("primary_macro_comparison", {})
    if primary.get("status") == "complete_development_primary_comparison":
        lines += ["", f"Primary {primary['target_percent']}% reduction: equal-model/equal-dataset mean adaptive-minus-uniform "
                  f"{primary['delta_pp']:+.2f} pp [{primary['ci_lower_pp']:.2f}, {primary['ci_upper_pp']:.2f}]. "
                  "The paired bootstrap draws the same images/documents across models. This remains development evidence."]
    return "\n".join(lines) + "\n"


def build_report(root, *, repetitions=BOOTSTRAP_REPETITIONS):
    root = Path(root).resolve()
    study = _sealed(root / "study.json")
    manifest = load_manifest(study["manifest"])
    models, quality, memory, screening, physical_by_model = {}, [], [], [], {}
    targets = tuple(study.get("kv_reduction_percent", TARGETS))
    for name, spec in sorted(study["models"].items()):
        try:
            summary, rows = scan_model(root, name, spec, manifest, world_size=study.get("world_size", 16), targets=targets)
            q, m = comparison_tables(rows, manifest.splits["evaluation"], targets=targets, repetitions=repetitions)
            models[name] = summary
            physical_by_model[name] = rows
            quality.extend({"model": name, "model_id": spec["model_id"], **row} for row in q)
            memory.extend({"model": name, "model_id": spec["model_id"], **row} for row in m)
        except (ValueError, KeyError, TypeError, OSError) as error:
            models[name] = {"status": "invalid", "error": f"{type(error).__name__}: {error}"}
        try:
            screening_status, screening_rows = scan_model(root, name, spec, manifest,
                world_size=study.get("world_size", 16), targets=targets, phase="screening")
            models[name]["screening_status"] = screening_status["status"]
            screening.extend({"model": name, **row} for row in screening_comparisons(
                screening_rows, manifest.splits["screening"], repetitions=repetitions))
        except (ValueError, KeyError, TypeError, OSError) as error:
            models[name]["screening_status"] = "invalid"
            models[name]["screening_error"] = f"{type(error).__name__}: {error}"
        try:
            models[name]["stage_completion"] = stage_completion(root, name, study.get("world_size", 16))
        except (ValueError, KeyError, TypeError, OSError) as error:
            models[name]["stage_completion"] = {stage: False for stage in REQUIRED_STAGES}
            models[name]["stage_completion_error"] = f"{type(error).__name__}: {error}"
    try:
        ablation = allocation_ablation(root, manifest)
    except (ValueError, KeyError, TypeError, OSError) as error:
        ablation = {"status": "invalid", "error": f"{type(error).__name__}: {error}", "candidates": {}}
    primary = primary_macro_ci(physical_by_model, manifest.splits["evaluation"], study["models"],
                               target=study.get("primary_target", 60), repetitions=repetitions)
    completion = completion_status(models, study["models"])
    return {"format": "mlsys_adaptive_report_v3", "updated_utc": datetime.now(timezone.utc).isoformat(),
            **completion, "study_sha256": file_digest(root / "study.json"),
            "manifest_sha256": manifest.file_sha256, "manifest_evaluation_status": manifest.metadata["evaluation_status"],
            "models": models, "quality_table": quality, "memory_table": memory,
            "execution": execution_states(root),
            "allocation_ablation": ablation, "screening_physical": screening,
            "primary_macro_comparison": primary,
            "bootstrap_repetitions": repetitions, "bootstrap_seed": BOOTSTRAP_SEED,
            "baseline_scope": "projected_feature_adaptations_not_official_pipelines",
            "fresh_confirmation_claim": False, "text_eviction_claim": False,
            "timing_isolated": False, "primary_target_kv_reduction_percent": study.get("primary_target", 60),
            "deadline_utc_epoch": study.get("deadline_utc_epoch")}


def write_report(root, *, repetitions=BOOTSTRAP_REPETITIONS):
    report = build_report(root, repetitions=repetitions)
    directory = Path(root) / "reports"
    atomic_json(directory / "results.json", report)
    _atomic_text(directory / "tables.md", render_markdown(report))
    print(json.dumps({"updated_utc": report["updated_utc"], "status": report["status"],
                      "execution_complete": report["execution_complete"], "full_matrix_complete": report["full_matrix_complete"],
                      "models": {name: row["status"] for name, row in report["models"].items()},
                      "quality_rows": len(report["quality_table"])}), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--once", action="store_true")
    modes.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=30.)
    args = parser.parse_args(argv)
    if not 1 <= args.interval <= 60:
        parser.error("interval must lie between 1 and 60 seconds")
    while True:
        report = write_report(args.root)
        if (not args.watch or report.get("execution_complete") is True
                or (report.get("deadline_utc_epoch") is not None and time.time() >= report["deadline_utc_epoch"])):
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
