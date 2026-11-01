"""Audit physical adaptive-vision outputs without reading or fitting any labels.

Reads only frozen policies, answer-free screening identities/assignments, and the
new runner's scored output rows. Partial outputs have no confidence intervals or
promotion decisions. Completed outputs remain previously examined development
evidence, never independent confirmation. This module never launches GPU work.
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
import time

from .adaptive_vision_policy import AdaptiveVisionPolicy
from .state import atomic_json, digest, file_digest
from learned_score.mlsys_eval.statistics import paired_cluster_bootstrap


MODELS = ("qwen25", "llava_next")
WORLD_SIZE = 8
EXPECTED_REQUESTS = 298
LABELS = {
    "adaptive": "AsymVLM adaptive vision + fixed text cache",
    "fixed": "AsymVLM fixed 70% vision + fixed text cache",
    "full_text": "All image tokens + fixed text cache",
    "full": "All tokens, no eviction",
}


def _read(path):
    return json.loads(Path(path).read_text())


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _cells(policy):
    tag = f"normalized_entropy_continuous_{digest(policy.payload)[:12]}"
    return {"adaptive": f"scorer_adaptive_{tag}_text512",
            "fixed": "scorer_fixed_control_v70_text512",
            "full_text": "full_all_v100_text512", "full": "full_all_v100_textoff"}


def _expected(root, model):
    directory = root / model
    policy = AdaptiveVisionPolicy.load(directory / "calibration/policy.json")
    identities = _read(directory / "calibration/screening_rows.json")
    if not isinstance(identities, list) or len(identities) != EXPECTED_REQUESTS:
        raise ValueError(f"{model}: expected exactly {EXPECTED_REQUESTS} screening requests")
    expected = {item["row_hash"]: item for item in identities}
    if len(expected) != len(identities):
        raise ValueError("duplicate screening identity")
    for item in identities:
        if policy.validate_evaluation_identity(**{key: item[key] for key in ("row_hash", "group_id", "image_sha256", "split")}):
            raise ValueError("screening overlaps policy calibration")
    assignments = _read(directory / "budget_assignments.json")
    if (assignments.get("sha256") != digest({key: value for key, value in assignments.items() if key != "sha256"})
            or assignments.get("format") != "mlsys_adaptive_budget_assignments_v1"
            or assignments.get("policy_sha256") != digest(policy.payload)
            or assignments.get("manifest_sha256") != policy.calibration_manifest_sha256
            or assignments.get("target_retention") != 0.7 or assignments.get("labels_used") is not False
            or set(assignments.get("rows", {})) != set(expected)):
        raise ValueError("assignment provenance or request set mismatch")
    for request, entry in expected.items():
        assigned = assignments["rows"][request]
        if any(assigned.get(key) != entry[key] for key in ("row_hash", "group_id", "image_sha256", "dataset")):
            raise ValueError("assignment identity mismatch")
        count, kept = assigned.get("input_visual_tokens"), assigned.get("selected_count")
        if type(count) is not int or type(kept) is not int or not 1 <= kept <= count:
            raise ValueError("invalid assigned token count")
    return policy, expected, assignments


def _audit_metadata(metadata, policy, root, model, cells):
    directory = root / model
    if (metadata.get("format") != "mlsys_adaptive_vision_benchmark_v1"
            or metadata.get("manifest_sha256") != policy.calibration_manifest_sha256
            or metadata.get("checkpoint_sha256") != policy.checkpoint_sha256
            or metadata.get("budget_policy") != policy.payload
            or metadata.get("budget_policy_sha256") != digest(policy.payload)
            or metadata.get("budget_policy_file_sha256") != file_digest(directory / "calibration/policy.json")
            or metadata.get("request_subset_sha256") != file_digest(directory / "calibration/screening_rows.json")
            or metadata.get("budget_assignments_sha256") != file_digest(directory / "budget_assignments.json")
            or metadata.get("repetitions") != 1 or metadata.get("world_size") != WORLD_SIZE
            or metadata.get("stage") != "development" or metadata.get("confirmation_claim") is not False
            or metadata.get("fixed_output_tokens") is not False
            or metadata.get("aggregate_visual_tokens_reconciled") is not True
            or metadata.get("aggregate_token_target") != 0.7
            or len(metadata.get("cells", [])) != 4 or set(metadata["cells"]) != set(cells.values())
            or metadata.get("backbone", {}).get("model") != policy.model
            or metadata.get("backbone", {}).get("adapter") != policy.adapter):
        raise ValueError(f"{model}: runtime metadata does not match frozen development experiment")
    if not metadata.get("source_sha256"):
        raise ValueError("runtime metadata must record source hashes")


def scan_model(root, model):
    root = Path(root)
    policy, expected, assignments = _expected(root, model)
    cells = _cells(policy)
    by_method = {method: role for role, method in cells.items()}
    physical = root / "physical" / model
    metadata_digests, contexts, completions = set(), set(), []
    present_ranks = []
    for rank in range(WORLD_SIZE):
        path = physical / f"rank{rank}.metadata.json"
        marker = physical / f"rank{rank}.complete.json"
        if not path.exists():
            if marker.exists():
                raise ValueError("completion marker without metadata")
            continue
        metadata = _read(path)
        _audit_metadata(metadata, policy, root, model, cells)
        metadata_digests.add(digest(metadata))
        contexts.add(digest({key: value for key, value in metadata.items() if key != "hardware"}))
        present_ranks.append(rank)
        if marker.exists():
            complete = _read(marker)
            if (complete.get("identity_sha256") != digest(metadata) or complete.get("cells") != 4
                    or complete.get("repetitions") != 1 or type(complete.get("requests")) is not int
                    or complete["requests"] <= 0 or complete.get("request_results") != complete["requests"] * 4):
                raise ValueError("rank completion marker mismatch")
            completions.append(complete)
    if len(contexts) > 1:
        raise ValueError("rank configurations or source hashes differ")
    rows, failed, seen = [], [], set()
    for path in sorted((physical / "rows").glob("*.json")):
        record = _read(path)
        request, method = record.get("request_id"), record.get("method")
        if request not in expected or method not in by_method:
            raise ValueError(f"unexpected output request or method: {path.name}")
        if path.name != f"{request}.{method}.repeat0.json" or (request, method) in seen:
            raise ValueError("duplicate output or unexpected repeat filename")
        seen.add((request, method))
        item, assigned = expected[request], assignments["rows"][request]
        if (record.get("format") != "mlsys_adaptive_request_result_v1"
                or record.get("identity_sha256") not in metadata_digests
                or record.get("model") != policy.model or record.get("stage") != "development"
                or record.get("split") != item["split"] or record.get("dataset") != item["dataset"]
                or record.get("group_id") != item["group_id"] or record.get("image_sha256") != item["image_sha256"]
                or record.get("repetition") != 0 or record.get("repeat") != 1
                or record.get("calibration_reuse") is not False):
            raise ValueError("physical row identity/context mismatch")
        role = by_method[method]
        mode = "adaptive" if role == "adaptive" else "fixed_control" if role == "fixed" else "all"
        if (record.get("vision_budget_mode") != mode
                or record.get("generated_budget") != (None if role == "full" else 512)):
            raise ValueError("row method configuration mismatch")
        if record.get("status") == "failed":
            failed.append({"request_id": request, "method": method, "error": record.get("error")})
            continue
        if record.get("status") != "ok" or record.get("quality_status") != "scored":
            raise ValueError("unexpected physical result status")
        quality = _number(record.get("quality_score"), "quality_score")
        if not 0 <= quality <= 1 or record.get("quality_metric") != ("anls" if item["dataset"] == "DocVQA" else "vqa_score"):
            raise ValueError("invalid quality metric or normalized score")
        n = assigned["input_visual_tokens"]
        kept = assigned["selected_count"] if role == "adaptive" else max(1, n * 70 // 100) if role == "fixed" else n
        if record.get("input_visual_tokens") != n or record.get("retained_visual_tokens") != kept:
            raise ValueError("physical image counts differ from frozen exact assignment")
        if role == "adaptive":
            decision = record.get("adaptive_decision") or {}
            if (record.get("budget_policy_sha256") != digest(policy.payload)
                    or record.get("aggregate_visual_tokens_reconciled") is not True
                    or record.get("aggregate_reconciled_selected_count") != kept
                    or decision.get("fallback") != assigned["fallback"]):
                raise ValueError("adaptive row does not record the frozen policy/reconciliation")
        rows.append({**record, "configuration_id": method})
    complete = (len(completions) == WORLD_SIZE and len(rows) == len(expected) * 4 and not failed)
    if len(completions) == WORLD_SIZE:
        if sum(item["requests"] for item in completions) != len(expected):
            raise ValueError("rank completion request totals mismatch")
        if not complete:
            raise ValueError("all ranks claim completion but physical output matrix is incomplete")
    counts = Counter(row["request_id"] for row in rows)
    summary = {"model": policy.model, "status": "complete" if complete else "partial",
               "expected_requests": len(expected), "expected_results": len(expected) * 4,
               "completed_results": len(rows), "completed_four_method_requests": sum(n == 4 for n in counts.values()),
               "metadata_ranks": present_ranks, "completed_ranks": len(completions),
               "failed_results": failed, "cells": cells, "policy_sha256": digest(policy.payload),
               "budget_assignments_sha256": assignments["sha256"]}
    return summary, rows, expected, assignments


def _dataset_report(rows, expected, assignments, cells, *, complete, bootstrap_repetitions):
    report = {}
    for dataset in sorted({entry["dataset"] for entry in expected.values()}):
        subset = [row for row in rows if row["dataset"] == dataset]
        expected_ids = {key for key, entry in expected.items() if entry["dataset"] == dataset}
        quality = {}
        for role, method in cells.items():
            values = [row for row in subset if row["method"] == method]
            quality[role] = {"label": LABELS[role], "completed_requests": len(values),
                             "quality_percent": 100 * sum(row["quality_score"] for row in values) / len(values) if values else None,
                             "retained_visual_tokens": sum(row["retained_visual_tokens"] for row in values)}
        adaptive = [row for row in subset if row["method"] == cells["adaptive"]]
        distribution = Counter(str(row["retained_visual_tokens"]) for row in adaptive)
        percentages = Counter(str(round(100 * row["retained_visual_tokens"] / row["input_visual_tokens"], 3)) for row in adaptive)
        fallback = sum(row["adaptive_decision"]["fallback"] for row in adaptive)
        allocation = {"completed_requests": len(adaptive), "selected_count_histogram": dict(sorted(distribution.items())),
                      "actual_retention_percent_histogram": dict(sorted(percentages.items())),
                      "full_fallback_requests": fallback,
                      "full_fallback_fraction": fallback / len(adaptive) if adaptive else None,
                      "exact_aggregate_match_verified": False}
        comparisons = {}
        if complete:
            if any(quality[role]["completed_requests"] != len(expected_ids) for role in cells):
                raise ValueError("complete dataset has missing method outputs")
            adaptive_tokens, static_tokens = quality["adaptive"]["retained_visual_tokens"], quality["fixed"]["retained_visual_tokens"]
            target = assignments["datasets"][dataset]["static_visual_tokens"]
            if adaptive_tokens != static_tokens or adaptive_tokens != target:
                raise ValueError("physical adaptive/static aggregate visual-token totals differ")
            allocation.update(exact_aggregate_match_verified=True, adaptive_visual_tokens=adaptive_tokens,
                              static_visual_tokens=static_tokens, token_error_fraction=0.0)
            for reference in ("fixed", "full_text", "full"):
                comparisons[reference] = paired_cluster_bootstrap(
                    subset, cells[reference], method=cells["adaptive"], repetitions=bootstrap_repetitions,
                    seed=20261015)
        report[dataset] = {"status": "completed_development" if complete else "partial_not_comparable",
                           "expected_requests": len(expected_ids), "methods": quality, "allocation": allocation,
                           "paired_comparisons": comparisons,
                           "confidence_intervals": "document_group_bootstrap" if complete else "not_computed_for_partial_results"}
    return report


def collect(root, *, bootstrap_repetitions=10000):
    root = Path(root)
    scanned = {model: scan_model(root, model) for model in MODELS}
    all_complete = all(value[0]["status"] == "complete" for value in scanned.values())
    report = {"format": "mlsys_adaptive_physical_report_v1", "updated_at": datetime.now(timezone.utc).isoformat(),
              "status": "complete" if all_complete else "partial", "independent_confirmation": False,
              "evidence_label": "physical previously-examined development screening; not fresh confirmation",
              "promotion_claim": False, "models": {}}
    for model, (summary, rows, expected, assignments) in scanned.items():
        summary["datasets"] = _dataset_report(rows, expected, assignments, summary["cells"],
                                              complete=all_complete, bootstrap_repetitions=bootstrap_repetitions)
        report["models"][model] = summary
    return report


def markdown(report):
    lines = ["# Adaptive vision: physical development screening", "", report["evidence_label"], "",
             "No independent-confirmation or promotion claim. Text-cache allowance is fixed at 512 generated tokens.", ""]
    if report["status"] != "complete":
        lines.extend(["**Incomplete: descriptive partial scores only. Methods may cover different requests; do not interpret differences. No confidence intervals.**", ""])
    lines.extend(["| Model | Dataset | Completed / expected results | Adaptive vision | Fixed 70% vision | All vision + text cache | All tokens, no eviction |",
                  "|---|---|---:|---:|---:|---:|---:|"])
    def score(value):
        return "—" if value is None else f"{value:.2f}%"
    for summary in report["models"].values():
        for dataset, result in summary["datasets"].items():
            methods = result["methods"]
            done = sum(method["completed_requests"] for method in methods.values())
            lines.append(f"| {summary['model']} | {dataset} | {done}/{result['expected_requests'] * 4} | "
                         + " | ".join(score(methods[role]["quality_percent"]) for role in ("adaptive", "fixed", "full_text", "full")) + " |")
    if report["status"] == "complete":
        lines.extend(["", "| Model | Dataset | Comparison | Adaptive minus reference (pp) | Paired 95% CI (pp) |",
                      "|---|---|---|---:|---:|"])
        for summary in report["models"].values():
            for dataset, result in summary["datasets"].items():
                for role, comparison in result["paired_comparisons"].items():
                    lines.append(f"| {summary['model']} | {dataset} | {LABELS[role]} | {100 * comparison['delta']:+.2f} | "
                                 f"[{100 * comparison['ci_lower']:+.2f}, {100 * comparison['ci_upper']:+.2f}] |")
        lines.extend(["", "Exact adaptive/static aggregate visual-token equality verified separately for each model and dataset."])
    return "\n".join(lines) + "\n"


def _atomic_text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    with os.fdopen(fd, "w") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("reproduction_results/mlsys_adaptive_vision_20261015_v1"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--seconds", type=float, default=43200)
    args = parser.parse_args(argv)
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    destination = args.root / "reporting"
    deadline = time.monotonic() + args.seconds
    while True:
        try:
            report = collect(args.root)
        except Exception as error:
            atomic_json(destination / "status.json", {"status": "audit_failed", "error": str(error),
                                                     "updated_at": datetime.now(timezone.utc).isoformat()})
            raise
        atomic_json(destination / "report.json", report)
        _atomic_text(destination / "tables.md", markdown(report))
        status = {"status": report["status"], "updated_at": report["updated_at"],
                  "models": {model: {key: value for key, value in item.items() if key != "datasets"}
                             for model, item in report["models"].items()}}
        atomic_json(destination / "status.json", status)
        print(json.dumps({"status": report["status"], "completed_results": {
            model: item["completed_results"] for model, item in report["models"].items()}}), flush=True)
        if report["status"] == "complete" or not args.watch:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            atomic_json(destination / "watch_deadline.json", {"status": "deadline_reached_with_partial_results",
                                                             "updated_at": datetime.now(timezone.utc).isoformat()})
            return
        time.sleep(min(30, remaining))


if __name__ == "__main__":
    main()
