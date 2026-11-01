"""Physical exact-count experiments from immutable, answer-free v4 schedules.

The scheduler decides which counts to measure. This runner never fits a policy,
interpolates quality, or chooses counts from answers. Frozen selectors and the
protected-prefix runtime are reused; Phi uses the versioned tokenizer-EOS fix.
"""

import argparse
import copy
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import socket
import time

from . import adaptive_experiment_v3 as prior
from . import phi_adaptive_v4 as phi
from .state import atomic_json, digest, file_digest


FORMAT = "mlsys_adaptive_experiment_v4"
MAX_NEW_TOKENS = 128
GENERATED_BUDGET = 512
SAFE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


def source_hashes(adapter):
    result = prior.source_hashes(adapter)
    result["mlsys_sprint/adaptive_experiment_v4.py"] = file_digest(__file__)
    if adapter == "phi3_vision":
        result.update(phi.source_hashes())
    return result


def validate_schedule(payload, *, manifest_sha256, checkpoint_sha256, split, allowed_ids):
    if not isinstance(payload, dict) or payload.get("sha256") != digest(
            {key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError("schedule content hash mismatch")
    for key, expected in (("manifest_sha256", manifest_sha256),
                          ("checkpoint_sha256", checkpoint_sha256), ("split", split)):
        if payload.get(key) != expected:
            raise ValueError(f"schedule {key} mismatch")
    if not isinstance(payload.get("cache_contract"), dict):
        raise ValueError("schedule must bind the measured cache contract")
    rows = payload.get("rows")
    if not isinstance(rows, dict) or not rows or set(rows) - set(allowed_ids):
        raise ValueError("schedule contains no rows or unknown/out-of-split identities")
    for row_hash, row in rows.items():
        if not isinstance(row, dict) or not isinstance(row.get("cells"), list) or not row["cells"]:
            raise ValueError(f"scheduled request {row_hash} has no physical cells")
        keys = set()
        for cell in row["cells"]:
            if not isinstance(cell, dict):
                raise ValueError("schedule cell must be an object")
            key = cell.get("key")
            if not isinstance(key, str) or not SAFE_KEY.fullmatch(key) or key in keys:
                raise ValueError("unsafe or duplicate schedule cell key")
            keys.add(key)
            if cell.get("selector") not in {"scorer", "full", "divprune", "cdpruner"}:
                raise ValueError("unsupported scheduled selector")
            if type(cell.get("selected_count")) is not int or cell["selected_count"] < 1:
                raise ValueError("scheduled visual count must be a positive integer")
            if any(name in cell for name in ("answer", "quality_score", "prediction", "references")):
                raise ValueError("physical schedules may not contain answer labels")
    return payload


def validate_cell_count(cell, feature, row_schedule=None):
    n, p = feature["input_visual_tokens"], feature["input_text_tokens"]
    count = cell["selected_count"]
    if type(count) is not int or not 1 <= count <= n or (cell["selector"] == "full" and count != n):
        raise ValueError("scheduled count is outside actual image geometry")
    for key, actual in (("input_visual_tokens", n), ("input_text_tokens", p)):
        if row_schedule is not None and key in row_schedule and row_schedule[key] != actual:
            raise ValueError(f"schedule/runtime {key} mismatch")
    return count


def selector_for(cell, scorer, feature, observed=None):
    validate_cell_count(cell, feature)
    # External allocation is already frozen. The selector receives only its
    # exact K; calling this internal uniform branch does not change allocation.
    internal = prior.Cell(cell["selector"], "full" if cell["selector"] == "full" else "uniform", 0)
    assigned = {"uniform_selected_count": cell["selected_count"]}
    return prior.select_for(internal, scorer, feature, assigned, observed), internal


def expected_cache_bytes(feature, count, contract):
    return (count + feature["input_text_tokens"] + contract["reserved_generated_slots"]) * contract["total_owned_bytes_per_slot"]


def generation_trace(adapter, ids, runtime, max_new_tokens=MAX_NEW_TOKENS):
    if adapter.name == "phi3_vision":
        return phi.annotate_generation(adapter, ids, runtime, max_new_tokens)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("generation tracing requires one unpadded sequence")
    values = ids.detach().cpu()[0].tolist()
    eos = getattr(getattr(adapter.model, "generation_config", None), "eos_token_id", None)
    if eos is None:
        eos = getattr(adapter.model.config, "eos_token_id", None)
    eos = [eos] if isinstance(eos, int) else list(eos or [])
    if not values or len(values) > max_new_tokens or runtime.get("stop_on_eos") is not True:
        raise ValueError("invalid natural-stopping generation")
    if any(value in eos for value in values[:-1]):
        raise RuntimeError("generation continued after configured EOS")
    ended = values[-1] in eos
    if not ended and len(values) != max_new_tokens:
        raise RuntimeError("generation ended without EOS or declared safety cap")
    return {**runtime, "generated_token_ids": values,
            "stop_reason": "eos_token" if ended else "max_new_tokens", "ended_with_eos": ended,
            "effective_eos_token_ids": eos, "first_eos_position": len(values) - 1 if ended else None}


def compare_features(live, saved, checkpoint_sha256):
    differences = {key: abs(live[key] - saved[key]) for key in ("normalized_entropy", "nearest_neighbor_diversity")}
    a, b = live["top_mass_by_percent"], saved["top_mass_by_percent"]
    if len(a) != 101 or len(b) != 101:
        raise ValueError("feature mass curve has wrong length")
    mass_difference = max(abs(x - y) for x, y in zip(a, b))
    if (saved["checkpoint_sha256"] != checkpoint_sha256
            or any(saved[key] != live[key] for key in ("input_visual_tokens", "input_text_tokens"))
            or max(*differences.values(), mass_difference) > 1e-6):
        raise ValueError(f"live features differ from frozen features: {differences}; mass={mass_difference}")


def reuse_physical_measurement(source, record):
    """Alias only identical request/selector/K under the same frozen identity."""
    for key in ("identity_sha256", "request_id", "selector", "selected_count"):
        if source.get(key) != record.get(key):
            raise ValueError(f"physical reuse {key} mismatch")
    if source.get("status") != "ok" or not source.get("physical_measurement_performed"):
        raise ValueError("physical reuse requires a successful directly measured source")
    result = copy.deepcopy(source)
    result.update(record)
    result.update(physical_measurement_performed=False,
        physical_reuse_of_cell=source["method"], physical_source_row_sha256=digest(source),
        physical_source_host=source["host"], physical_source_job_id=source["job_id"],
        timing_reused_not_an_independent_observation=True)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("features", "physical"))
    parser.add_argument("--split", required=True, choices=("fit", "screening", "evaluation"))
    for name in ("manifest", "checkpoint", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--training-manifest", type=Path,
                        default=Path("reproduction_results/mlsys_20261012/manifest.json"))
    parser.add_argument("--adapter", required=True, choices=("qwen2_5_vl", "llava_next", "phi3_vision"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--num-crops", type=int, choices=(16,), default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if (not 0 <= args.rank < args.world_size or args.warmup < 0
            or (args.limit is not None and args.limit < 1)):
        parser.error("invalid rank or request/warmup limit")
    if (args.phase == "physical") != (args.schedule is not None):
        parser.error("physical phase requires a schedule; features phase forbids one")
    args.backbone_dtype = "bfloat16"
    args.attention_implementation = "eager" if args.adapter == "phi3_vision" else "sdpa"
    args.dataset_splits = (args.split,) if args.adapter == "phi3_vision" else ("scorer_dev",)
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from .adaptive_data_v3 import load_manifest
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    manifest = load_manifest(args.manifest)
    entries = sorted(manifest.splits[args.split], key=lambda item: (item["dataset"], item["row_hash"]))
    checkpoint_sha = file_digest(args.checkpoint)
    schedule = None
    if args.schedule:
        schedule = validate_schedule(json.loads(args.schedule.read_text()),
            manifest_sha256=manifest.file_sha256, checkpoint_sha256=checkpoint_sha,
            split=args.split, allowed_ids={item["row_hash"] for item in entries})
        entries = [item for item in entries if item["row_hash"] in schedule["rows"]]
    if args.limit:
        entries = entries[:args.limit]
    entries = entries[args.rank::args.world_size]
    context = phi.installed() if args.adapter == "phi3_vision" else nullcontext()
    with context:
        adapter, dataset, backbone, scorer = prior.load_runtime(args, manifest)
    contract = prior.contract_from_config(adapter.model.config)
    if schedule is not None and schedule["cache_contract"] != contract:
        raise ValueError("schedule/runtime cache contract mismatch")
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    identity = {"format": FORMAT, "manifest_sha256": manifest.file_sha256,
        "checkpoint_sha256": checkpoint_sha, "backbone": backbone,
        "phase": args.phase, "split": args.split, "world_size": args.world_size,
        "limit": args.limit, "warmup": args.warmup, "source_sha256": source_hashes(args.adapter),
        "cache_contract": contract, "hardware": torch.cuda.get_device_name(adapter.device),
        "max_new_tokens": MAX_NEW_TOKENS, "generated_budget": GENERATED_BUDGET,
        "schedule_sha256": file_digest(args.schedule) if args.schedule else None,
        "schedule_content_sha256": schedule["sha256"] if schedule else None,
        "feature_root": str(args.feature_root) if args.feature_root else None,
        "baseline_scope": "projected_feature_adaptations_not_official_pipeline",
        "timing_isolated": False, "text_eviction_claim": False}
    prior._immutable(args.output / f"rank{args.rank}.metadata.json", identity, resume=args.resume)
    identity_sha = digest(identity)
    completed, started = 0, time.monotonic()
    for index, entry in enumerate(entries):
        row = dataset.row_for_entry(entry)
        if hasattr(row.get("image"), "seek"):
            row["image"].seek(0)
        with torch.no_grad():
            prepared = adapter.prepare_sample(row, include_answer=False)
            features = prior.feature_summary(prepared, scorer)
        del prepared
        feature = {"format": "mlsys_answer_free_features_v3", "measurement_version": "v4",
            "request_id": entry["row_hash"], "group_id": entry["group_id"],
            "image_sha256": entry["image_sha256"], "dataset": entry["dataset"],
            "split": args.split, "identity_sha256": identity_sha,
            "checkpoint_sha256": checkpoint_sha, "cache_contract": contract,
            "answers_consumed_by_feature_extractor": False, **features}
        if args.feature_root:
            saved = json.loads((args.feature_root / "features" / (entry["row_hash"] + ".json")).read_text())
            try:
                compare_features(features, saved, checkpoint_sha)
            except ValueError as error:
                atomic_json(args.output / f"rank{args.rank}.feature_mismatch.json", {
                    "request_id": entry["row_hash"], "live": features, "saved": saved,
                    "error": str(error), "host": socket.gethostname()})
                raise
        prior._immutable(args.output / "features" / (entry["row_hash"] + ".json"), feature, resume=args.resume)
        if args.phase == "features":
            assert_frozen(adapter.model, versions)
            assert_frozen(scorer, scorer_versions)
            completed += 1
            continue
        row_schedule = schedule["rows"][entry["row_hash"]]
        cells = row_schedule["cells"]
        offset = int(entry["row_hash"][:8], 16) % len(cells)
        references = prior.benchmark.parse_references(row["answer"])
        metric = entry.get("quality_metric", entry.get("metric"))
        physical_measurements = {}
        for cell in cells[offset:] + cells[:offset]:
            count = validate_cell_count(cell, feature, row_schedule)
            physical_key = (cell["selector"], count)
            path = args.output / "rows" / f"{entry['row_hash']}.{cell['key']}.json"
            attempt = 0
            if path.exists():
                previous = json.loads(path.read_text())
                if not args.resume or previous.get("identity_sha256") != identity_sha:
                    raise ValueError("physical-result resume identity mismatch")
                if previous["status"] == "ok":
                    if previous.get("physical_measurement_performed"):
                        physical_measurements.setdefault(physical_key, previous)
                    completed += 1
                    continue
                if previous.get("attempt", 0) >= 1:
                    raise RuntimeError("physical result exhausted one retry")
                attempt = previous.get("attempt", 0) + 1
            observed = {}
            record = {"format": FORMAT, "identity_sha256": identity_sha,
                "request_id": entry["row_hash"], "group_id": entry["group_id"],
                "image_sha256": entry["image_sha256"], "dataset": entry["dataset"],
                "category": entry.get("category"), "model": backbone["model"],
                "method": cell["key"], "selector": cell["selector"],
                "phase": args.phase, "split": args.split, "quality_metric": metric,
                "schedule_cell": cell, "target": cell.get("target"),
                "policy": cell.get("policy"), "mode": cell.get("mode"),
                "selected_count": count, "schedule_content_sha256": schedule["sha256"],
                "evidence_status": entry.get("evidence_status", entry.get("evaluation_stage", "development")),
                "host": socket.gethostname(), "job_id": os.getenv("PBS_JOBID"),
                "hardware": identity["hardware"], "timing_isolated": False,
                "attempt": attempt, "max_new_tokens": MAX_NEW_TOKENS, "generated_budget": GENERATED_BUDGET}
            if physical_key in physical_measurements:
                record = reuse_physical_measurement(physical_measurements[physical_key], record)
                atomic_json(path, record)
                completed += 1
                continue
            try:
                select, internal = selector_for(cell, scorer, feature, observed)
                if index == 0 and args.warmup:
                    for _ in range(args.warmup):
                        _, ids, _, metrics = prior.measured(adapter, row, select, internal, 8)
                        generation_trace(adapter, ids, metrics, 8)
                prediction, ids, timing, runtime = prior.measured(adapter, row, select, internal, MAX_NEW_TOKENS)
                runtime = generation_trace(adapter, ids, runtime)
                expected = expected_cache_bytes(feature, count, contract)
                prior.check_cache(runtime, expected)
                assert_frozen(adapter.model, versions)
                assert_frozen(scorer, scorer_versions)
                record.update(status="ok", prediction=prediction,
                    quality_score=prior.score_prediction(prediction, references, metric),
                    generated_tokens=int(ids.shape[-1]), generated_token_ids=runtime["generated_token_ids"],
                    stop_reason=runtime["stop_reason"], expected_cache_bytes=expected,
                    metrics=runtime, physical_measurement_performed=True,
                    physical_reuse_of_cell=None, timing_reused_not_an_independent_observation=False,
                    **timing, **observed)
                del ids
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error), quality_score=None)
                atomic_json(path, record)
                raise
            atomic_json(path, record)
            physical_measurements[physical_key] = record
            completed += 1
        print(json.dumps({"event": "adaptive_v4_progress", "phase": args.phase,
            "rank": args.rank, "requests": index + 1, "results": completed,
            "elapsed_seconds": time.monotonic() - started}), flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json", {
        "identity_sha256": identity_sha, "requests": len(entries), "results": completed,
        "elapsed_seconds": time.monotonic() - started})


if __name__ == "__main__":
    main()
