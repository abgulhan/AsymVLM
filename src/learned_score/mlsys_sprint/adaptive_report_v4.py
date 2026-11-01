"""Audited physical allocation, shuffle-control, transfer, and memory evidence.

V4 fixes Phi stopping and keeps the prior outputs immutable.  This module never
mixes v3 predictions into the new comparisons, never selects a rule using
evaluation labels, and never presents a partial cohort as an exact budget match.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

import numpy as np

from .adaptive_data_v4 import DEFAULT_OUTPUT, load_manifest
from .adaptive_data_v3 import dataset_family
from .adaptive_report_v1 import _atomic_text
from .adaptive_report_v3 import quality_summary, paired_ci, execution_states
from .state import atomic_json, digest, file_digest


FORMAT = "mlsys_adaptive_experiment_v4"
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261017
PHI_STOPPING_POLICY = "phi_tokenizer_eos_unpenalized_greedy_v4"


def _read(path):
    return json.loads(Path(path).read_text())


def _sealed(path):
    payload = _read(path)
    if payload.get("sha256") != digest({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError(f"artifact content hash mismatch: {path}")
    return payload


def _number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"invalid finite number: {field}")
    return float(value)


def _integer(value, field, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"invalid integer: {field}")
    return value


def audit_stopping(row, metadata, *, is_phi=False):
    runtime = row.get("metrics", {})
    ids = row.get("generated_token_ids")
    if (not isinstance(ids, list) or not ids or any(type(token) is not int or token < 0 for token in ids)
            or len(ids) > 128 or row.get("generated_tokens") != len(ids)
            or runtime.get("generated_token_ids") != ids or runtime.get("stop_on_eos") is not True):
        raise ValueError("missing or inconsistent natural-stopping token trace")
    if is_phi:
        stopping = runtime.get("stopping", {})
        if (stopping.get("policy") != PHI_STOPPING_POLICY
                or stopping.get("effective_eos_token_id") != 32000
                or metadata.get("backbone", {}).get("stopping") != stopping
                or runtime.get("effective_eos_token_id") != 32000):
            raise ValueError("Phi evidence lacks the corrected tokenizer-EOS stopping contract")
        eos = {32000}
    else:
        values = runtime.get("effective_eos_token_ids")
        if not isinstance(values, list) or any(type(token) is not int for token in values):
            raise ValueError("missing effective stopping-token identities")
        eos = set(values)
    if any(token in eos for token in ids[:-1]):
        raise ValueError("generation continued after an effective stopping token")
    ended = ids[-1] in eos
    expected = "eos_token" if ended else "max_new_tokens"
    if (row.get("stop_reason") != expected or runtime.get("stop_reason") != expected
            or runtime.get("ended_with_eos") is not ended
            or runtime.get("first_eos_position") != (len(ids) - 1 if ended else None)
            or (not ended and len(ids) != 128)):
        raise ValueError("generation stop reason does not match token trace")


def audit_cache(row, contract, cell, row_schedule):
    n = _integer(row.get("input_visual_tokens"), "input_visual_tokens", 1)
    p = _integer(row.get("input_text_tokens"), "input_text_tokens")
    kept = _integer(row.get("retained_visual_tokens"), "retained_visual_tokens", 1)
    if (kept > n or kept != cell["selected_count"] or row.get("selected_count") != kept
            or (cell["selector"] == "full" and kept != n)):
        raise ValueError("physical retained count differs from the scheduled exact count")
    for name, actual in (("input_visual_tokens", n), ("input_text_tokens", p)):
        if name in row_schedule and row_schedule[name] != actual:
            raise ValueError("physical geometry differs from the schedule")
    capacity = kept + p + contract["reserved_generated_slots"]
    kv = capacity * contract["kv_bytes_per_slot"]
    owned = capacity * contract["total_owned_bytes_per_slot"]
    runtime, cache = row.get("metrics", {}), row.get("metrics", {}).get("cache", {})
    if (cache.get("kv_capacity_bytes") != kv or cache.get("total_owned_bytes") != owned
            or cache.get("source_metadata_bytes") != capacity * contract["source_metadata_bytes_per_slot"]
            or cache.get("scratch_bytes") != 0 or cache.get("compaction_events") != 0
            or runtime.get("peak_cache_bytes") != owned or row.get("expected_cache_bytes") != owned):
        raise ValueError("measured allocated-cache bytes disagree with exact token geometry")
    _number(runtime.get("full_request_peak_allocated_bytes"), "GPU peak bytes")


def scan_stage(root, model, spec, manifest, stage, schedule_path=None, *, world_size=16):
    """Audit a generic physical stage against its immutable exact-count schedule."""
    root, directory = Path(root), Path(root) / model / stage
    schedule_path = Path(schedule_path) if schedule_path else root / model / "schedules" / f"{stage}.json"
    if not schedule_path.is_file():
        skipped = directory / "skipped.json"
        if stage == "benefit_screening" and skipped.is_file():
            return {"model": model, "stage": stage, "status": "skipped_by_predeclared_condition",
                    "valid_physical_rows": 0, "skip_record": _read(skipped)}, []
        return {"model": model, "stage": stage, "status": "awaiting_schedule", "valid_physical_rows": 0}, []
    schedule = _sealed(schedule_path)
    split = schedule.get("split")
    if schedule.get("cache_contract", {}).get("reserved_generated_slots") != 127:
        raise ValueError("report requires the declared 127 reserved generated-cache slots")
    checkpoint_sha = file_digest(spec["checkpoint"])
    if (split not in manifest.splits or schedule.get("manifest_sha256") != manifest.file_sha256
            or schedule.get("checkpoint_sha256") != checkpoint_sha):
        raise ValueError("schedule does not match the frozen manifest/checkpoint")
    manifest_entries = {entry["row_hash"]: entry for entry in manifest.splits[split]}
    planned = schedule.get("rows", {})
    if not planned or set(planned) - set(manifest_entries):
        raise ValueError("physical schedule has no rows or out-of-split identities")
    expected = {}
    for request, item in planned.items():
        if not item.get("cells"):
            raise ValueError("scheduled request has no cells")
        for cell in item["cells"]:
            key = (request, cell["key"])
            if key in expected:
                raise ValueError("duplicate scheduled physical cell")
            expected[key] = cell
    entries = sorted((manifest_entries[key] for key in planned), key=lambda entry: (entry["dataset"], entry["row_hash"]))
    identities, contexts, present_ranks, complete_ranks = {}, set(), [], []
    schedule_sha = file_digest(schedule_path)
    for rank in range(world_size):
        meta_path, marker_path = directory / f"rank{rank}.metadata.json", directory / f"rank{rank}.complete.json"
        if not meta_path.is_file():
            if marker_path.exists():
                raise ValueError("completion marker exists without metadata")
            continue
        meta = _read(meta_path)
        if (meta.get("format") != FORMAT or meta.get("phase") != "physical" or meta.get("split") != split
                or meta.get("world_size") != world_size or meta.get("limit") is not None
                or meta.get("manifest_sha256") != manifest.file_sha256
                or meta.get("checkpoint_sha256") != checkpoint_sha
                or meta.get("schedule_sha256") != schedule_sha
                or meta.get("schedule_content_sha256") != schedule["sha256"]
                or meta.get("cache_contract") != schedule["cache_contract"]
                or meta.get("backbone", {}).get("model") != spec["model_id"]
                or not meta.get("source_sha256") or meta.get("max_new_tokens") != 128
                or meta.get("generated_budget") != 512 or meta.get("text_eviction_claim") is not False):
            raise ValueError("physical metadata differs from frozen experiment context")
        identity = digest(meta)
        identities[identity] = meta
        contexts.add(digest({key: value for key, value in meta.items() if key != "hardware"}))
        present_ranks.append(rank)
        if marker_path.exists():
            marker = _read(marker_path)
            rank_ids = {entry["row_hash"] for entry in entries[rank::world_size]}
            count = sum(request in rank_ids for request, _ in expected)
            if (marker.get("identity_sha256") != identity or marker.get("requests") != len(rank_ids)
                    or marker.get("results") != count):
                raise ValueError("completion marker does not match scheduled shard counts")
            complete_ranks.append(rank)
    if len(contexts) > 1:
        raise ValueError("source/configuration changed between physical shards")
    rows, failed, seen = [], [], set()
    for path in sorted((directory / "rows").glob("*.json")):
        row = _read(path)
        key = (row.get("request_id"), row.get("method"))
        if key not in expected or key in seen or path.name != f"{key[0]}.{key[1]}.json":
            raise ValueError("unexpected, duplicate, or misnamed physical result")
        seen.add(key)
        meta, cell, entry = identities.get(row.get("identity_sha256")), expected[key], manifest_entries[key[0]]
        if (meta is None or row.get("format") != FORMAT or row.get("phase") != "physical"
                or row.get("split") != split or row.get("model") != spec["model_id"]
                or row.get("schedule_cell") != cell or row.get("schedule_content_sha256") != schedule["sha256"]
                or any(row.get(field) != cell.get(field) for field in ("selector", "target", "policy", "mode"))
                or any(row.get(field) != entry.get(field) for field in ("dataset", "group_id", "image_sha256", "category"))
                or row.get("quality_metric") != entry["metric"] or row.get("max_new_tokens") != 128
                or row.get("generated_budget") != 512 or row.get("hardware") != meta.get("hardware")):
            raise ValueError("physical row identity/context mismatch")
        if row.get("status") == "failed":
            failed.append({"request_id": key[0], "method": key[1], "error": row.get("error")})
            continue
        value = _number(row.get("quality_score"), "physical quality")
        if row.get("status") != "ok" or not 0 <= value <= 1:
            raise ValueError("invalid physical quality/status")
        audit_cache(row, meta["cache_contract"], cell, planned[key[0]])
        audit_stopping(row, meta, is_phi=spec.get("adapter") == "phi3_vision")
        rows.append(row)
    row_map = {(row["request_id"], row["method"]): row for row in rows}
    for row in rows:
        measured = row.get("physical_measurement_performed")
        if measured is True:
            if row.get("physical_reuse_of_cell") is not None or row.get("timing_reused_not_an_independent_observation") is not False:
                raise ValueError("direct physical measurement carries inconsistent reuse metadata")
        elif measured is False:
            source = row_map.get((row["request_id"], row.get("physical_reuse_of_cell")))
            if (source is None or source.get("physical_measurement_performed") is not True
                    or row.get("physical_source_row_sha256") != digest(source)
                    or row.get("timing_reused_not_an_independent_observation") is not True
                    or row.get("physical_source_host") != source.get("host")
                    or row.get("physical_source_job_id") != source.get("job_id")
                    or any(row.get(field) != source.get(field) for field in (
                        "identity_sha256", "selector", "selected_count", "prediction", "quality_score",
                        "generated_tokens", "generated_token_ids", "stop_reason", "expected_cache_bytes", "metrics",
                        "input_visual_tokens", "input_text_tokens", "retained_visual_tokens", "end_to_end_seconds",
                        "ttft_seconds", "queue_seconds", "compression_seconds", "stage_seconds", "unattributed_seconds",
                        "timing_contract", "timing_mode", "includes_preprocessing", "file_loading_included"))):
                raise ValueError("reused result lacks an identical directly measured source")
        else:
            raise ValueError("physical row must distinguish direct inference from exact-count reuse")
    complete = len(rows) == len(expected) and not failed and len(complete_ranks) == world_size
    restricted = bool(schedule.get("infeasible_panels") or schedule.get("unavailable_panels"))
    return {"model": model, "stage": stage, "split": split,
            "status": ("complete_feasible_subset" if restricted else "complete") if complete else "partial",
            "valid_physical_rows": len(rows), "expected_physical_rows": len(expected),
            "independent_physical_inferences": sum(row["physical_measurement_performed"] for row in rows),
            "exact_count_aliases": sum(not row["physical_measurement_performed"] for row in rows),
            "missing_rows": len(expected) - len(rows), "failed_rows": failed,
            "present_ranks": present_ranks, "complete_ranks": complete_ranks, "world_size": world_size,
            "manifest_sha256": manifest.file_sha256, "checkpoint_sha256": checkpoint_sha,
            "schedule_sha256": schedule_sha, "schedule_content_sha256": schedule["sha256"],
            "infeasible_panels": schedule.get("infeasible_panels", []),
            "unavailable_panels": schedule.get("unavailable_panels", []),
            "corrected_phi_stopping_required": spec.get("adapter") == "phi3_vision",
            "evidence": "physical_development_not_fresh_confirmation"}, rows


def _mode(row):
    mode = row.get("mode") or row.get("schedule_cell", {}).get("mode")
    if mode in {"adaptive", "all_fit"}:
        return "raw"
    if isinstance(mode, str) and (mode.startswith("shuffled") or mode.startswith("shuffle")):
        return "shuffled"
    if isinstance(mode, str) and (mode.startswith("leave_out:") or mode.startswith("lobo") or mode == "leave_one_benchmark_out"):
        return "lobo"
    return mode


def _method_index(rows):
    index = {}
    for row in rows:
        key = row["request_id"], row["method"]
        if key in index:
            raise ValueError("duplicate physical question/method")
        index[key] = row
    return index


def _complete_pair_ids(ids, entries, metric):
    ids = set(ids)
    if metric != "mme":
        return ids
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["group_id"]].append(entry["row_hash"])
    return {key for group in grouped.values() if len(group) == 2 and set(group) <= ids for key in group}


def paired_against_shuffles(candidate, shuffles, metric, *, repetitions=BOOTSTRAP_REPETITIONS,
                            seed=BOOTSTRAP_SEED):
    """CI against mean physical shuffled quality; MME pair scoring precedes averaging."""
    if not shuffles or repetitions < 100:
        raise ValueError("physical shuffle controls and at least 100 bootstrap draws are required")
    a = {row["request_id"]: row for row in candidate}
    baselines = [{row["request_id"]: row for row in rows} for rows in shuffles]
    if not a or len(a) != len(candidate) or any(set(b) != set(a) or len(b) != len(rows) for b, rows in zip(baselines, shuffles)):
        raise ValueError("shuffle bootstrap requires identical nonduplicate request sets")
    for request, row in a.items():
        if any((b[request]["group_id"], b[request].get("category")) != (row["group_id"], row.get("category")) for b in baselines):
            raise ValueError("shuffle group/category mismatch")
    candidate_quality = quality_summary(candidate, metric)["score_percent"]
    seed_scores = [quality_summary(rows, metric)["score_percent"] for rows in shuffles]
    result = {"delta_pp": candidate_quality - float(np.mean(seed_scores)),
              "shuffle_score_percent_by_seed": seed_scores, "shuffle_seeds": len(shuffles),
              "questions": len(a), "groups": len({row["group_id"] for row in candidate}),
              "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
              "confidence": .95, "seed_averaging": "average physical benchmark metrics before paired group bootstrap"}
    groups = defaultdict(list)
    for request, row in sorted(a.items()):
        groups[row["group_id"]].append(request)
    rng, estimates = np.random.default_rng(seed), np.zeros(repetitions)
    if metric == "mme":
        categories = defaultdict(list)
        for requests in groups.values():
            if len(requests) != 2:
                raise ValueError("MME shuffled comparison requires complete image pairs")
            values = [a[key]["quality_score"] for key in requests]
            score = .5 * (float(np.mean(values)) + float(all(v == 1 for v in values)))
            reference = []
            for baseline in baselines:
                values = [baseline[key]["quality_score"] for key in requests]
                reference.append(.5 * (float(np.mean(values)) + float(all(v == 1 for v in values))))
            categories[a[requests[0]]["category"]].append(score - float(np.mean(reference)))
        if any(len(values) < 2 for values in categories.values()):
            return {**result, "ci_lower_pp": None, "ci_upper_pp": None,
                    "status": "insufficient_image_pairs_per_category"}
        for values in categories.values():
            values = np.asarray(values)
            for start in range(0, repetitions, 256):
                stop = min(start + 256, repetitions)
                draws = rng.integers(0, len(values), size=(stop - start, len(values)))
                estimates[start:stop] += values[draws].mean(axis=1) / len(categories)
        unit = "image_pairs_stratified_by_category"
    else:
        if len(groups) < 2:
            return {**result, "ci_lower_pp": None, "ci_upper_pp": None,
                    "status": "insufficient_independent_groups"}
        sums = np.asarray([sum(a[key]["quality_score"] - np.mean([b[key]["quality_score"] for b in baselines])
                                  for key in requests) for requests in groups.values()])
        counts = np.asarray([len(requests) for requests in groups.values()])
        for start in range(0, repetitions, 256):
            stop = min(start + 256, repetitions)
            draws = rng.integers(0, len(sums), size=(stop - start, len(sums)))
            estimates[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        unit = "document_or_image_groups"
    low, high = np.quantile(estimates * 100, [.025, .975])
    return {**result, "ci_lower_pp": float(low), "ci_upper_pp": float(high),
            "status": "computed", "resampling_unit": unit}


def _full_cache(row, *, owned=False):
    runtime = row["metrics"]["cache"]
    retained = row["retained_visual_tokens"] + row["input_text_tokens"] + 127
    measured = runtime["total_owned_bytes" if owned else "kv_capacity_bytes"]
    if measured % retained:
        raise ValueError("cache geometry is not an integer number of bytes per slot")
    return (row["input_visual_tokens"] + row["input_text_tokens"] + 127) * (measured // retained)


def _equal_budget(a, b):
    if {row["request_id"] for row in a} != {row["request_id"] for row in b}:
        return False
    return all(sum(row["metrics"]["cache"][field] for row in a) == sum(row["metrics"]["cache"][field] for row in b)
               for field in ("kv_capacity_bytes", "total_owned_bytes"))


def memory_summary(rows, uniform, full=None):
    if not rows or not _equal_budget(rows, uniform):
        raise ValueError("memory comparison requires exact paired aggregate budgets")
    uniform_map = {row["request_id"]: row for row in uniform}
    full_map = {row["request_id"]: row for row in full or []}
    kv = [row["metrics"]["cache"]["kv_capacity_bytes"] for row in rows]
    owned = [row["metrics"]["cache"]["total_owned_bytes"] for row in rows]
    full_kv = [_full_cache(row) for row in rows]
    full_owned = [_full_cache(row, owned=True) for row in rows]
    gpu = [row["metrics"]["full_request_peak_allocated_bytes"] for row in rows]
    have_full = set(full_map) == {row["request_id"] for row in rows}
    full_gpu = [full_map[row["request_id"]]["metrics"]["full_request_peak_allocated_bytes"] for row in rows] if have_full else []
    return {"questions": len(rows), "mean_kv_bytes": float(np.mean(kv)),
            "mean_full_kv_bytes": float(np.mean(full_kv)), "mean_total_owned_cache_bytes": float(np.mean(owned)),
            "mean_full_total_owned_cache_bytes": float(np.mean(full_owned)),
            "kv_reduction_percent_ratio_of_sums": 100 * (1 - sum(kv) / sum(full_kv)),
            "mean_per_input_kv_reduction_percent": float(np.mean([100 * (1 - x / y) for x, y in zip(kv, full_kv)])),
            "inputs_with_kv_reduction_percent": 100 * sum(x < y for x, y in zip(kv, full_kv)) / len(rows),
            "inputs_with_less_kv_than_uniform_percent": 100 * sum(row["retained_visual_tokens"] < uniform_map[row["request_id"]]["retained_visual_tokens"] for row in rows) / len(rows),
            "inputs_with_changed_budget_percent": 100 * sum(row["retained_visual_tokens"] != uniform_map[row["request_id"]]["retained_visual_tokens"] for row in rows) / len(rows),
            "mean_visual_retention_percent": float(np.mean([100 * row["retained_visual_tokens"] / row["input_visual_tokens"] for row in rows])),
            "distinct_retained_counts": len({row["retained_visual_tokens"] for row in rows}),
            "mean_full_request_peak_gpu_bytes": float(np.mean(gpu)),
            "gpu_peak_reduction_percent_ratio_of_sums": 100 * (1 - sum(gpu) / sum(full_gpu)) if full_gpu and sum(full_gpu) else None,
            "full_gpu_reference_measured": have_full, "timing_isolated": False,
            "allocation_overhead_included_in_end_to_end_timing": False}


def _panels(rows, entries):
    _method_index(rows)
    by_dataset = defaultdict(list)
    for entry in entries:
        by_dataset[entry["dataset"]].append(entry)
    for dataset, cohort in sorted(by_dataset.items()):
        subset = [row for row in rows if row["dataset"] == dataset]
        choices = {(row.get("target"), row.get("policy"), _mode(row), row["method"])
                   for row in subset if row.get("selector") == "scorer" and _mode(row) in {"raw", "fallback", "lobo"}}
        for target, policy, mode, method in sorted(choices, key=str):
            candidate = [row for row in subset if row["method"] == method]
            uniform = [row for row in subset if row.get("target") == target and row["selector"] == "scorer" and _mode(row) == "uniform"]
            if not uniform:
                continue
            uniform_methods = {row["method"] for row in uniform}
            if len(uniform_methods) != 1:
                raise ValueError("ambiguous uniform baseline within target/benchmark")
            yield dataset, cohort, target, policy, mode, method, candidate, uniform, subset


def comparison_tables(rows, entries, *, repetitions=BOOTSTRAP_REPETITIONS):
    quality, memory, diagnostics = [], [], []
    for dataset, cohort, target, policy, mode, method, candidate, uniform, subset in _panels(rows, entries):
        metric = cohort[0]["metric"]
        expected = {entry["row_hash"] for entry in cohort}
        matched = _complete_pair_ids({row["request_id"] for row in candidate} & {row["request_id"] for row in uniform}, cohort, metric)
        if not matched:
            continue
        a = sorted((row for row in candidate if row["request_id"] in matched), key=lambda row: row["request_id"])
        b = sorted((row for row in uniform if row["request_id"] in matched), key=lambda row: row["request_id"])
        complete, equal = matched == expected, _equal_budget(a, b)
        actual = 100 * (1 - sum(row["metrics"]["cache"]["kv_capacity_bytes"] for row in a) / sum(_full_cache(row) for row in a))
        verified = complete and equal and abs(actual - target) <= .2
        bases = {"full": [row for row in subset if row["selector"] == "full"],
                 "divprune": [row for row in subset if row["selector"] == "divprune" and row.get("target") == target],
                 "cdpruner": [row for row in subset if row["selector"] == "cdpruner" and row.get("target") == target]}
        scores = {"adaptive": quality_summary(a, metric), "uniform": quality_summary(b, metric)}
        differences = {"adaptive_minus_uniform": paired_ci(a, b, metric, repetitions=repetitions)}
        base_status = {}
        for role, values in bases.items():
            chosen = [row for row in values if row["request_id"] in matched]
            ids = [row["request_id"] for row in chosen]
            if set(ids) == matched and len(ids) == len(matched) and (role == "full" or _equal_budget(a, chosen)):
                scores[role] = quality_summary(chosen, metric)
                differences[f"adaptive_minus_{role}"] = paired_ci(a, chosen, metric, repetitions=repetitions)
                base_status[role] = "paired_complete"
            else:
                base_status[role] = "missing_or_incomplete_or_budget_mismatched"
        item = {"dataset": dataset_family(dataset), "source_dataset": dataset, "target_kv_reduction_percent": target,
                "policy": policy, "mode": mode, "method": method, "matched_questions": len(matched),
                "expected_questions": len(expected), "cohort_complete": complete,
                "status": "complete_cohort" if complete else "provisional_partial_intersection",
                "exact_matched_budget_verified": verified, "actual_kv_reduction_percent": actual,
                "equal_aggregate_budget_on_observed_rows": equal, "baseline_status": base_status,
                "quality": scores, "evidence": "physical_development_not_fresh_confirmation", **differences}
        quality.append(item)
        if equal:
            memory.append({key: item[key] for key in ("dataset", "target_kv_reduction_percent", "policy", "mode", "method", "status", "exact_matched_budget_verified")}
                          | memory_summary(a, b, bases["full"]))
        better = sum(x["quality_score"] > y["quality_score"] for x, y in zip(a, b))
        worse = sum(x["quality_score"] < y["quality_score"] for x, y in zip(a, b))
        cells = [row["schedule_cell"] for row in a]
        diagnostics.append({"dataset": item["dataset"], "target": target, "policy": policy, "mode": mode,
                            "questions": len(a), "better_answers": better, "worse_answers": worse,
                            "unchanged_answers": len(a) - better - worse,
                            "changed_budgets": sum(x["retained_visual_tokens"] != y["retained_visual_tokens"] for x, y in zip(a, b)),
                            "reported_fallbacks": sum(bool(cell.get("fallback") or cell.get("fallback_kind")) for cell in cells),
                            "fallback_reasons": dict(Counter(str(cell.get("fallback_reason", cell.get("fallback_kind"))) for cell in cells if cell.get("fallback") or cell.get("fallback_kind"))),
                            "out_of_support_inputs": sum(bool(cell.get("out_of_support")) for cell in cells),
                            "support_failure_counts": dict(Counter(reason for cell in cells for reason in cell.get("support_failures", []))),
                            "max_token_cap_answers": sum(row["stop_reason"] == "max_new_tokens" for row in a),
                            "mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in a])),
                            "transfer_scope": sorted({cell.get("transfer_scope", "benchmark excluded from fitting; policy-selection scope must be disclosed") for cell in cells}) if mode == "lobo" else None,
                            "unseen_benchmark_confirmation": False,
                            "raw_policy_not_replaced_by_best_baseline": mode != "fallback"})
    return quality, memory, diagnostics


def physical_selection_records(rows_by_model, manifest, schedules=None, *, repetitions=BOOTSTRAP_REPETITIONS):
    """Produce allocation.select_physical_candidate records from screening only.

    A panel is emitted only when all expected questions, seeds, and exact
    aggregate budgets are present. Partial seeds cannot become easy controls.
    """
    records, details, waiting = [], [], []
    entries = manifest.splits["screening"]
    for model, rows in sorted(rows_by_model.items()):
        if any(row.get("split") != "screening" or row.get("phase") != "physical" for row in rows):
            raise ValueError("policy selection accepts physical screening only; evaluation labels forbidden")
        schedule = (schedules or {}).get(model)
        for dataset, cohort, target, policy, mode, method, candidate, uniform, subset in _panels(rows, entries):
            if mode != "raw":
                continue
            ids, metric = {entry["row_hash"] for entry in cohort}, cohort[0]["metric"]
            control_rows = [row for row in subset if row.get("target") == target and row.get("policy") == policy and _mode(row) == "shuffled"]
            controls = defaultdict(list)
            for row in control_rows:
                controls[row["method"]].append(row)
            expected_control_keys = set(controls)
            if schedule is not None:
                expected_control_keys = {cell["key"] for request, item in schedule["rows"].items() if request in ids
                                         for cell in item["cells"] if cell.get("target") == target and cell.get("policy") == policy
                                         and _mode(cell) == "shuffled"}
            complete = ({row["request_id"] for row in candidate} == ids
                        and {row["request_id"] for row in uniform} == ids and expected_control_keys
                        and set(controls) == expected_control_keys
                        and all({row["request_id"] for row in values} == ids for values in controls.values()))
            panel = {"candidate": policy, "model": model, "dataset": dataset_family(dataset), "target": target}
            actual_reduction = 100 * (1 - sum(row["metrics"]["cache"]["kv_capacity_bytes"] for row in candidate)
                                     / sum(_full_cache(row) for row in candidate))
            if (not complete or not _equal_budget(candidate, uniform)
                    or any(not _equal_budget(candidate, values) for values in controls.values())
                    or abs(actual_reduction - target) > .2):
                waiting.append({**panel, "reason": "incomplete physical cohort/control seeds or aggregate budget mismatch"})
                continue
            source_map = {row["request_id"]: row for row in candidate}
            source_histograms = defaultdict(Counter)
            for row in candidate:
                source_histograms[row["input_visual_tokens"]][row["retained_visual_tokens"]] += 1
            for values in controls.values():
                shuffled_histograms = defaultdict(Counter)
                for row in values:
                    original = source_map[row["request_id"]]
                    if any(row[key] != original[key] for key in ("input_visual_tokens", "input_text_tokens", "dataset")):
                        raise ValueError("shuffled control changes request geometry")
                    shuffled_histograms[row["input_visual_tokens"]][row["retained_visual_tokens"]] += 1
                if dict(shuffled_histograms) != dict(source_histograms):
                    raise ValueError("shuffled control does not preserve each image-size budget multiset")
            contrast = paired_ci(candidate, uniform, metric, repetitions=repetitions)
            shuffled = paired_against_shuffles(candidate, list(controls.values()), metric, repetitions=repetitions)
            if contrast["ci_lower_pp"] is None or shuffled["ci_lower_pp"] is None:
                waiting.append({**panel, "reason": "insufficient independent groups for physical selection"})
                continue
            by_size = defaultdict(set)
            size_counts = Counter()
            source_counts = {row["request_id"]: row["retained_visual_tokens"] for row in candidate}
            for row in candidate:
                by_size[row["input_visual_tokens"]].add(row["retained_visual_tokens"])
                size_counts[row["input_visual_tokens"]] += 1
            movable = sum(size_counts[n] for n, counts in by_size.items() if len(counts) > 1) / len(candidate)
            changed = [sum(row["retained_visual_tokens"] != source_counts[row["request_id"]] for row in values) / len(values)
                       for values in controls.values()]
            records.append({**panel, "stage": "screening", "physical": True,
                            "delta_uniform": contrast["delta_pp"] / 100,
                            "delta_shuffled": shuffled["delta_pp"] / 100,
                            "delta_uniform_ci_low": contrast["ci_lower_pp"] / 100,
                            "delta_shuffled_ci_low": shuffled["ci_lower_pp"] / 100,
                            "shuffle_informative": any(value > 0 for value in changed),
                            "shuffle_movable_fraction": movable})
            details.append({**panel, "adaptive_minus_uniform": contrast, "adaptive_minus_mean_shuffled": shuffled,
                            "shuffle_changed_fractions": changed, "shuffle_movable_fraction": movable,
                            "exact_matched_budget_verified": True})
    observed = {(row["model"], row["dataset"], row["target"], row["candidate"]) for row in details + waiting}
    by_id = {entry["row_hash"]: entry for entry in entries}
    for model, schedule in (schedules or {}).items():
        for request, planned in schedule["rows"].items():
            for cell in planned["cells"]:
                if _mode(cell) != "raw":
                    continue
                key = model, dataset_family(by_id[request]["dataset"]), cell["target"], cell["policy"]
                if key not in observed:
                    waiting.append({"candidate": key[3], "model": key[0], "dataset": key[1], "target": key[2],
                                    "reason": "scheduled candidate/uniform panel has no paired physical rows yet"})
                    observed.add(key)
    return {"records": records, "details": details, "waiting": waiting,
            "evidence": "independent_physical_development_selection_not_confirmation",
            "evaluation_labels_read": False}


def phi_replay_diagnostics(study, *, repetitions=BOOTSTRAP_REPETITIONS):
    """Separate before/after stopping-repair diagnostics; never policy evidence."""
    root = Path(study["output_root"])
    old_manifest_path = Path(study["old_manifest"])
    from .adaptive_data_v3 import load_manifest as load_v3_manifest
    manifest = load_v3_manifest(old_manifest_path)
    old_root = old_manifest_path.parent.parent / "phi35"
    checkpoint = file_digest(study["models"]["phi35"]["checkpoint"])
    report = {"status": "awaiting_corrected_replays", "used_for_policy_selection": False,
              "old_predictions_are_invalid_for_primary_claims": True, "comparisons": [], "stages": {}}
    for stage in ("calibration", "screening", "evaluation"):
        directory = root / "phi_corrected" / stage
        rows_paths = sorted((directory / "rows").glob("*.json"))
        if not rows_paths:
            report["stages"][stage] = {"status": "awaiting_rows", "matched_results": 0}
            continue
        split = "fit" if stage == "calibration" else stage
        entries = {entry["row_hash"]: entry for entry in manifest.splits[split]}
        old_meta, new_meta = {}, {}
        try:
            for physical, destination, corrected in ((old_root / stage, old_meta, False), (directory, new_meta, True)):
                for path in sorted(physical.glob("rank*.metadata.json")):
                    metadata = _read(path)
                    if (metadata.get("format") != "mlsys_adaptive_experiment_v3"
                            or metadata.get("manifest_sha256") != manifest.file_sha256
                            or metadata.get("checkpoint_sha256") != checkpoint
                            or metadata.get("phase") != stage or metadata.get("split") != split
                            or metadata.get("max_new_tokens") != 128 or metadata.get("limit") is not None):
                        raise ValueError("Phi replay metadata changed original identities/model/count contract")
                    if corrected and metadata.get("backbone", {}).get("stopping", {}).get("policy") != PHI_STOPPING_POLICY:
                        raise ValueError("Phi replay lacks corrected stopping metadata")
                    destination[digest(metadata)] = metadata
            groups = defaultdict(list)
            matched_count = 0
            for path in rows_paths:
                new = _read(path)
                if new.get("status") != "ok":
                    continue
                old_path = old_root / stage / "rows" / path.name
                if not old_path.is_file():
                    raise ValueError("Phi corrected replay has no original exact method/request match")
                old = _read(old_path)
                if old.get("status") != "ok":
                    raise ValueError("original replay comparison did not succeed")
                meta, prior_meta = new_meta.get(new.get("identity_sha256")), old_meta.get(old.get("identity_sha256"))
                if meta is None or prior_meta is None or new.get("request_id") not in entries:
                    raise ValueError("unrecognized Phi replay identity")
                if any(row.get("format") != "mlsys_adaptive_experiment_v3" or row.get("phase") != stage for row in (old, new)):
                    raise ValueError("Phi replay must retain the original versioned evaluation format")
                for field in ("request_id", "method", "dataset", "group_id", "image_sha256", "model", "quality_metric",
                              "selector", "allocation", "percent", "input_visual_tokens", "input_text_tokens",
                              "retained_visual_tokens", "expected_cache_bytes", "max_new_tokens", "generated_budget"):
                    if new.get(field) != old.get(field):
                        raise ValueError(f"Phi replay is not an exact original input/method/count match: {field}")
                if any(meta.get(field) != prior_meta.get(field) for field in ("cache_contract", "assignments_sha256")):
                    raise ValueError("Phi replay changed cache or allocation assignment contract")
                runtime = new["metrics"]
                trace = {**new, "generated_token_ids": runtime.get("generated_token_ids"),
                         "stop_reason": runtime.get("stop_reason")}
                audit_stopping(trace, meta, is_phi=True)
                for row in (old, new):
                    score = _number(row.get("quality_score"), "Phi replay quality")
                    if not 0 <= score <= 1:
                        raise ValueError("Phi replay quality outside normalized range")
                groups[(new["dataset"], new["method"])].append((old, new))
                matched_count += 1
            for (dataset, method), pairs in sorted(groups.items()):
                metric = pairs[0][1]["quality_metric"]
                cohort = [entry for entry in entries.values() if entry["dataset"] == dataset]
                ids = _complete_pair_ids({new["request_id"] for _, new in pairs}, cohort, metric)
                pairs = [(old, new) for old, new in pairs if new["request_id"] in ids]
                if not pairs:
                    continue
                before, after = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
                report["comparisons"].append({"stage": stage, "dataset": dataset_family(dataset), "method": method,
                    "percent": after[0].get("percent"), "allocation": after[0].get("allocation"),
                    "questions": len(pairs), "expected_questions": len(cohort), "complete_cohort": len(pairs) == len(cohort),
                    "before_score_percent": quality_summary(before, metric)["score_percent"],
                    "corrected_score_percent": quality_summary(after, metric)["score_percent"],
                    "corrected_minus_before": paired_ci(after, before, metric, repetitions=repetitions),
                    "before_mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in before])),
                    "corrected_mean_generated_tokens": float(np.mean([row["generated_tokens"] for row in after])),
                    "before_reaching_128_tokens_percent": 100 * sum(row["generated_tokens"] == 128 for row in before) / len(before),
                    "corrected_max_token_cap_percent": 100 * sum(row["metrics"]["stop_reason"] == "max_new_tokens" for row in after) / len(after),
                    "corrected_eos_stop_percent": 100 * sum(row["metrics"]["stop_reason"] == "eos_token" for row in after) / len(after),
                    "same_input_selector_exact_count_checkpoint": True,
                    "evidence": "diagnostic_exact_count_stopping_replay_not_allocation_policy_evidence"})
            complete_markers = list(directory.glob("rank*.complete.json"))
            expected_rows = len(list((old_root / stage / "rows").glob("*.json")))
            complete = matched_count == expected_rows and len(complete_markers) == study.get("world_size", 16)
            for path in complete_markers:
                marker = _read(path)
                if marker.get("identity_sha256") not in new_meta:
                    raise ValueError("Phi replay completion marker lacks validated metadata")
            report["stages"][stage] = {"status": "complete" if complete else "partial", "matched_results": matched_count,
                                      "expected_results": expected_rows, "complete_markers": len(complete_markers)}
        except (ValueError, KeyError, TypeError, OSError) as error:
            report["stages"][stage] = {"status": "invalid", "error": str(error)}
    statuses = [value["status"] for value in report["stages"].values()]
    report["status"] = "invalid" if "invalid" in statuses else "complete" if all(status == "complete" for status in statuses) else "partial"
    return report


def _fmt(value):
    return "—" if value is None else f"{value:.2f}"


def _ci(value):
    if value is None:
        return "—"
    return f"{value['delta_pp']:+.2f} [{_fmt(value.get('ci_lower_pp'))}, {_fmt(value.get('ci_upper_pp'))}]"


def render_markdown(report):
    lines = ["# Repaired adaptive allocation: physical development results", "",
             f"Updated {report['updated_utc']}. Status: {report['status']}.", "",
             "All comparisons below use new physical outputs. Phi must use its corrected end-of-answer token; old Phi quality and latency results are excluded. These are previously inspected development panels, not fresh confirmation. DivPrune/CDPruner are selector adaptations, not official pipelines.", "",
             "Protocol change: v4 permits a minimum of one image token, measured directly near the operating point; v3 imposed a 10% image-retention floor. This makes Qwen MME at 50% allocated-KV reduction feasible, while 60% and 70% remain impossible under protected-text and reserved-generation slots. The expanded feasibility is not an EOS-fix or allocation-quality gain.", "",
             "## Table 1: Accuracy and allocation benefit", "",
             "Raw learned allocation, explicit fallback, and benchmark-excluded fitting are shown separately. The three predeclared statistic transfer ablations exclude the target benchmark from fitting and are all reported without selecting the best held-out result. The globally selected primary statistic still used all-benchmark screening; its identity is not wholly unseen-benchmark confirmation. All data remain development. A missing baseline is not inferred. Differences are percentage points with paired image/document-bootstrap 95% intervals.", "",
             "| Model | Benchmark | KV target | Rule / mode | Questions | All image tokens | AsymVLM adaptive | AsymVLM uniform | DivPrune adaptation | CDPruner adaptation | Adaptive − uniform | Adaptive − DivPrune | Matched budget |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for row in report.get("quality_table", []):
        q = row["quality"]
        lines.append(f"| {row['model']} | {row['dataset']} | {row['target_kv_reduction_percent']}% | {row['policy']} / {row['mode']} | {row['matched_questions']}/{row['expected_questions']} | "
                     + " | ".join(_fmt(q.get(role, {}).get("score_percent")) for role in ("full", "adaptive", "uniform", "divprune", "cdpruner"))
                     + f" | {_ci(row.get('adaptive_minus_uniform'))} | {_ci(row.get('adaptive_minus_divprune'))} | {'verified' if row['exact_matched_budget_verified'] else 'provisional/unmatched'} |")
    if not report.get("quality_table"):
        lines.append("| — | No paired physical allocation results yet | — | — | 0 | — | — | — | — | — | — | — | not verified |")
    lines += ["", "DocVQA uses ANLS, TextVQA VQA accuracy, ChartQA relaxed correctness. MME uses the category mean of (question accuracy + paired-image accuracy)/2; official summed points remain in JSON.", "",
              "## Table 2: Memory reduction and frequency", "",
              "Allocated KV tensors, total cache ownership, and peak GPU memory are different quantities. Ratios use aggregate bytes; per-input averages are listed separately.", "",
              "| Model | Benchmark | KV target | Rule / mode | All-token KV MiB | Adaptive KV MiB | Owned-cache MiB | Aggregate KV reduction | Mean per-input reduction | Inputs with lower KV | Budgets changed | GPU peak reduction |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in report.get("memory_table", []):
        lines.append(f"| {row['model']} | {row['dataset']} | {row['target_kv_reduction_percent']}% | {row['policy']} / {row['mode']} | "
                     f"{row['mean_full_kv_bytes']/2**20:.2f} | {row['mean_kv_bytes']/2**20:.2f} | {row['mean_total_owned_cache_bytes']/2**20:.2f} | "
                     f"{row['kv_reduction_percent_ratio_of_sums']:.2f}% | {row['mean_per_input_kv_reduction_percent']:.2f}% | {row['inputs_with_kv_reduction_percent']:.1f}% | "
                     f"{row['inputs_with_changed_budget_percent']:.1f}% | {_fmt(row['gpu_peak_reduction_percent_ratio_of_sums'])}% |")
    if not report.get("memory_table"):
        lines.append("| — | No matched physical memory results yet | — | — | — | — | — | — | — | — | — | — |")
    lines += ["", "Timings are not isolated and omit offline policy fitting/allocation; they do not establish serving-throughput gains. Short answers do not activate text eviction.", "",
              "## Does assigning the right input a larger budget help?", "",
              "Shuffled controls preserve the same budget multiset within equal-image-size, same-benchmark groups. A shuffle that changes no assignments is uninformative, not a policy win. Physical selection uses screening only; evaluation is never used to choose a rule.", "",
              "| Model | Benchmark | Target | Rule | Adaptive − uniform | Adaptive − mean shuffled | Shuffle-movable inputs |",
              "|---|---|---:|---|---|---|---:|"]
    for row in report.get("physical_selection", {}).get("details", []):
        lines.append(f"| {row['model']} | {row['dataset']} | {row['target']}% | {row['candidate']} | {_ci(row['adaptive_minus_uniform'])} | {_ci(row['adaptive_minus_mean_shuffled'])} | {100*row['shuffle_movable_fraction']:.1f}% |")
    if not report.get("physical_selection", {}).get("details"):
        lines.append("| — | Awaiting complete physical controls | — | — | — | — | — |")
    lines += ["", "## Phi stopping repair: separate diagnostic replay", "",
              "These comparisons rerun the original exact image counts and selectors. They diagnose the old stopping bug; old Phi predictions are excluded from primary policy comparisons. Partial image-pair cohorts are explicitly incomplete.", "",
              "| Benchmark | Original method | Questions | Before score | Corrected score | Before generated tokens | Corrected generated tokens | Before reaching 128 tokens | Corrected reaching cap |",
              "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    displayed = [row for row in report.get("phi_stopping_replay", {}).get("comparisons", [])
                 if row["stage"] == "evaluation" and (row["percent"] == 60 or row["allocation"] == "full")]
    for row in displayed:
        lines.append(f"| {row['dataset']} | {row['method']} | {row['questions']}/{row['expected_questions']} | {_fmt(row['before_score_percent'])} | {_fmt(row['corrected_score_percent'])} | "
                     f"{row['before_mean_generated_tokens']:.1f} | {row['corrected_mean_generated_tokens']:.1f} | {row['before_reaching_128_tokens_percent']:.1f}% | {row['corrected_max_token_cap_percent']:.1f}% |")
    if not displayed:
        lines.append("| — | Awaiting exact-count corrected evaluation replay | 0 | — | — | — | — | — | — |")
    lines += ["", "## Execution audit", ""]
    for model, stages in report.get("models", {}).items():
        for stage, status in stages.items():
            lines.append(f"- {model}/{stage}: {status['status']}; {status.get('valid_physical_rows', 0)}/{status.get('expected_physical_rows', '?')} physical rows" + (f"; {status['error']}" if status.get("error") else ""))
            if status.get("infeasible_panels"):
                lines.append(f"  - Mathematically infeasible targets: `{json.dumps(status['infeasible_panels'], sort_keys=True)}`")
            if status.get("unavailable_panels"):
                lines.append(f"  - Unavailable fitted-policy panels: `{json.dumps(status['unavailable_panels'], sort_keys=True)}`")
    return "\n".join(lines) + "\n"


def build_report(root, *, study=None, stages=("screening", "benefit_screening", "evaluation"), repetitions=BOOTSTRAP_REPETITIONS):
    root = Path(root)
    study = study if study is not None else _sealed(root / "study.json")
    manifest = load_manifest(study["manifest"])
    report = {"format": "mlsys_adaptive_report_v4", "updated_utc": datetime.now(timezone.utc).isoformat(),
              "manifest_sha256": manifest.file_sha256, "status": "partial", "models": {},
              "quality_table": [], "memory_table": [], "allocation_diagnostics": [],
              "evidence": "physical_development_not_fresh_confirmation", "old_phi_outputs_included": False}
    screening, schedules, benefit, benefit_schedules = {}, {}, {}, {}
    for model, spec in sorted(study["models"].items()):
        report["models"][model] = {}
        for stage in stages:
            try:
                summary, rows = scan_stage(root, model, spec, manifest, stage, world_size=study.get("world_size", 16))
                report["models"][model][stage] = summary
                if stage == "evaluation":
                    quality, memory, diagnostics = comparison_tables(rows, manifest.splits["evaluation"], repetitions=repetitions)
                    for key, values in (("quality_table", quality), ("memory_table", memory), ("allocation_diagnostics", diagnostics)):
                        report[key].extend({"model": model, **value} for value in values)
                elif stage in {"screening", "benefit_screening"}:
                    destination, planned = (screening, schedules) if stage == "screening" else (benefit, benefit_schedules)
                    destination[model] = rows
                    path = root / model / "schedules" / f"{stage}.json"
                    if path.exists():
                        planned[model] = _sealed(path)
            except (ValueError, KeyError, TypeError, OSError) as error:
                report["models"][model][stage] = {"status": "invalid", "error": str(error)}
    report["physical_selection"] = physical_selection_records(screening, manifest, schedules, repetitions=repetitions)
    benefit_report = physical_selection_records(benefit, manifest, benefit_schedules, repetitions=repetitions)
    for key in ("records", "details", "waiting"):
        report["physical_selection"][key].extend(benefit_report[key])
    states = [status["status"] for stages_by_model in report["models"].values() for stage, status in stages_by_model.items()
              if stage != "benefit_screening" or status["status"] != "awaiting_schedule"]
    report["main_matrix_complete"] = bool(states) and all(status in {"complete", "skipped_by_predeclared_condition"} for status in states)
    report["feasible_main_work_complete"] = bool(states) and all(status in {"complete", "complete_feasible_subset", "skipped_by_predeclared_condition"} for status in states)
    report["status"] = "invalid" if "invalid" in states else "complete" if report["main_matrix_complete"] else "complete_feasible_subset" if report["feasible_main_work_complete"] else "partial"
    report["execution_states"] = execution_states(root)
    report["phi_stopping_replay"] = phi_replay_diagnostics(study, repetitions=repetitions)
    report["execution_complete"] = report["feasible_main_work_complete"] and report["phi_stopping_replay"]["status"] == "complete"
    if report["phi_stopping_replay"]["status"] == "invalid":
        report["status"] = "invalid"
    elif report["feasible_main_work_complete"] and not report["execution_complete"]:
        report["status"] = "main_feasible_matrix_complete_replays_pending"
    return report


def write_report(study, *, repetitions=BOOTSTRAP_REPETITIONS):
    """Workflow entry point: accept the frozen study dictionary or its run root."""
    if isinstance(study, (str, Path)):
        root, payload = Path(study), None
    elif isinstance(study, dict):
        root, payload = Path(study["output_root"]), study
        if study.get("sha256") != digest({key: value for key, value in study.items() if key != "sha256"}):
            raise ValueError("workflow supplied an unsealed or changed study")
    else:
        raise TypeError("write_report requires a sealed study dictionary or run-root path")
    report = build_report(root, study=payload, repetitions=repetitions)
    atomic_json(root / "reports/results.json", report)
    _atomic_text(root / "reports/tables.md", render_markdown(report))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval", type=float, default=45)
    args = parser.parse_args(argv)
    while True:
        report = write_report(args.root)
        print(json.dumps({"event": "adaptive_v4_report", "status": report["status"], "updated_utc": report["updated_utc"]}), flush=True)
        if not args.watch or args.once:
            break
        time.sleep(min(60, max(1, args.interval)))


if __name__ == "__main__":
    main()
