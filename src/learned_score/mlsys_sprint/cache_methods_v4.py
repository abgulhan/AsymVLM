"""Atomic, development-only physical pilot for deterministic cache history.

This driver cannot open the reserved final split. Each page is published only
after every declared cell finishes in the same process. Incomplete attempts are
kept for diagnosis, never assembled into a completed page.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re
import uuid

from .followup_common import (load_followup_protocol, load_model, model_spec,
                              require_cuda, verify_native_gate, write_immutable)
from .long_document_benchmark import execution_context
from .state import digest, file_digest


FORMAT = "mlsys_cache_history_page_v4"
NEW_POLICIES = {"protected_recency", "protected_grid", "protected_weyl", "protected_priority_random"}
LEGACY_POLICIES = {"h2o_recent_adjusted", "rkv"}
CHECKS = ("no_eviction_exact_ids", "reference_optimized_exact_ids", "physical_capacity_and_prefix")


def entries_for(protocol):
    entries = protocol.get("entries", protocol.get("development_entries"))
    if not isinstance(entries, list) or len(entries) != 24:
        raise ValueError("v4 physical pilot requires exactly 24 declared development pages")
    if len({r["row_hash"] for r in entries}) != 24 or len({r["group_id"] for r in entries}) != 24:
        raise ValueError("pilot requires distinct page and source identities")
    return entries


def cell_objects(protocol):
    from .cache_protocol_v4 import CacheCell
    fields = set(CacheCell.__dataclass_fields__)
    cells = tuple(CacheCell(**{k: v for k, v in row.items() if k in fields}) for row in protocol["cells"])
    if len({c.key for c in cells}) != len(cells) or not cells:
        raise ValueError("duplicate or absent physical cells")
    return cells


def load_inputs(path):
    from .cache_protocol_v4 import load_protocol
    protocol = load_protocol(path)
    base = load_followup_protocol(protocol["base_protocol"])
    allowed = set(base["development_request_ids"])
    if any(e["row_hash"] not in allowed for e in entries_for(protocol)):
        raise ValueError("pilot tries to access a page outside declared development")
    return protocol, base


def metadata_for(protocol, base, model):
    if model not in protocol.get("active_models", ["qwen25"]):
        raise ValueError("model is not authorized for this pilot stage")
    spec = model_spec(base, model)
    return {"format": "mlsys_cache_history_panel_v4", "protocol_sha256": protocol["sha256"],
            "stage": protocol["stage"], "model_name": model, "model": spec["model_id"],
            "model_spec": spec, "split": "scorer_dev", "dataset_kind": "olmocr",
            "manifest": base["manifest"], "manifest_sha256": file_digest(base["manifest"]),
            "cells": [dict(asdict(c), key=c.key) for c in cell_objects(protocol)],
            "request_ids": [e["row_hash"] for e in entries_for(protocol)],
            "max_new_tokens": 4096, "quality_run_only": True, "timing_isolated": False,
            "all_results_development": True, "final_labels_opened": False,
            "checkpoint_sha256": spec["checkpoint_sha256"],
            "quality_floor_relative_to_full": .95,
            "budget_definition": "physical_M=original_prefix+B+1; retained=M-staging"}


def validate_page(rows, entry, metadata):
    cells = {c["key"]: c for c in metadata["cells"]}
    if len(rows) != len(cells) or {r.get("method") for r in rows} != set(cells):
        raise ValueError("page must include every physical cell exactly once")
    contexts, attempts, prefixes, visuals = set(), set(), set(), set()
    for row in rows:
        cell = cells[row["method"]]
        required = {"format": FORMAT, "identity_sha256": digest(metadata), "run_context": digest(metadata),
                    "model": metadata["model"],
                    "model_name": metadata["model_name"], "protocol_sha256": metadata["protocol_sha256"],
                    "stage": metadata["stage"], "split": "scorer_dev", "status": "ok",
                    "request_id": entry["row_hash"], "group_id": entry["group_id"],
                    "image_sha256": entry["image_sha256"], "natural_stop_enabled": True,
                    "quality_run_only": True, "timing_isolated": False, "final_labels_opened": False,
                    "cache_policy": cell["policy"], "random_seed": cell["seed"], "cell": cell,
                    "max_new_tokens": 4096, "visual_retention": 1., "selector": "full"}
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError("physical row differs from declared cell or page identity")
        ids = row.get("generated_ids")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 4096
                or any(type(t) is not int or t < 0 for t in ids)
                or row.get("generated_tokens") != len(ids) or row.get("generated_ids_sha256") != digest(ids)
                or not isinstance(row.get("prediction"), str)):
            raise ValueError("invalid generated token IDs or prediction")
        prefix = row["input_prefix_tokens"]
        if (type(prefix) is not int or prefix < 1 or row["retained_prefix_tokens"] != prefix
                or row["retained_visual_tokens"] != row["input_visual_tokens"]):
            raise ValueError("initial pilot must retain all image/prompt tokens")
        capacity = prefix + cell["budget"] + 1
        cache = row["metrics"]["cache"]
        if (row.get("kv_budget_slots") != capacity
                or row.get("staging_slots") != cell["staging"]
                or row.get("kv_retained_slots") != capacity - cell["staging"]
                or cache.get("capacity_tokens_per_layer") != capacity
                or not 0 < cache.get("peak_active_tokens", 0) <= capacity
                or row.get("kv_budget_bytes") != cache.get("kv_capacity_bytes")
                or cache.get("protects_entire_prefix") != (cell["policy"] in NEW_POLICIES)):
            raise ValueError("physical cache exceeds or differs from its declared capacity")
        contexts.add(digest(row["execution_context"]))
        attempts.add(row["attempt"])
        prefixes.add(prefix)
        visuals.add(row["selected_visual_indices_sha256"])
    if any(len(values) != 1 for values in (contexts, attempts, prefixes, visuals)):
        raise ValueError("page cells were spliced across runs or inputs")


def collect_panel(protocol_path, directory=None, *, require_complete=True, model="qwen25"):
    protocol, base = load_inputs(protocol_path)
    metadata = metadata_for(protocol, base, model)
    directory = Path(directory) if directory is not None else Path(protocol_path).parent / "documents" / model
    expected_directory = (Path(protocol_path).parent / "documents" / model).resolve()
    if directory.resolve() != expected_directory:
        raise ValueError("panel directory is not the protocol's authorized output")
    metadata_path = directory / "metadata.json"
    if directory.is_symlink() or metadata_path.is_symlink():
        raise ValueError("panel metadata and directory must not be symlinks")
    artifacts, rows, completed = {}, {}, 0
    if metadata_path.exists():
        if json.loads(metadata_path.read_text()) != metadata:
            raise ValueError("published panel metadata changed")
        artifacts[str(metadata_path)] = file_digest(metadata_path)
    entries = entries_for(protocol)
    for entry in entries:
        marker_path = directory / f"{entry['row_hash']}.complete.json"
        if marker_path.is_symlink():
            raise ValueError("page completion marker must not be a symlink")
        if not marker_path.exists():
            continue
        marker = json.loads(marker_path.read_text())
        expected = {"format": "mlsys_cache_history_page_complete_v4", "passed": True,
                    "protocol_sha256": protocol["sha256"], "metadata_sha256": digest(metadata),
                    "request_id": entry["row_hash"], "final_labels_opened": False}
        if any(marker.get(k) != v for k, v in expected.items()) or not metadata_path.exists():
            raise ValueError("invalid physical page completion marker")
        files = marker.get("artifacts", {})
        if len(files) != len(metadata["cells"]):
            raise ValueError("page seal does not bind all physical cells")
        attempt = marker.get("attempt")
        if not isinstance(attempt, str) or not re.fullmatch(r"[0-9a-f]{32}", attempt):
            raise ValueError("page attempt must be one UUID hex identity")
        attempt_path = directory / "attempts" / entry["row_hash"] / attempt
        if any(p.is_symlink() for p in (attempt_path, attempt_path.parent, attempt_path.parent.parent)):
            raise ValueError("page attempt directories must not be symlinks")
        attempt_root = attempt_path.resolve()
        page = []
        for filename, expected_hash in files.items():
            path = Path(filename)
            if path.is_symlink() or path.resolve().parent != attempt_root or path.suffix != ".json":
                raise ValueError("sealed artifact escapes its page attempt")
            if file_digest(path) != expected_hash:
                raise ValueError("sealed prediction changed")
            row = json.loads(path.read_text())
            if row.get("attempt") != marker["attempt"]:
                raise ValueError("attempt identity differs")
            page.append(row)
            artifacts[str(path)] = expected_hash
        validate_page(page, entry, metadata)
        for row in page:
            rows[(row["request_id"], row["method"])] = row
        artifacts[str(marker_path)] = file_digest(marker_path)
        completed += 1
    complete = completed == len(entries)
    if require_complete and not complete:
        raise ValueError(f"official pilot scoring requires all 24 sealed pages, found {completed}")
    return {"metadata": metadata, "entries": entries, "cells": metadata["cells"], "rows": rows,
            "complete": complete, "completed_pages": completed, "context": digest(metadata),
            "stage": protocol["stage"], "integrity_artifacts": artifacts, "directory": str(directory)}


def _generate(adapter, pruned, prepared, cell, tokens, **kwargs):
    if cell.policy in NEW_POLICIES:
        from learned_score.mlsys_runtime.matched_generation_v4 import generate_matched_v4
        return generate_matched_v4(adapter.model, pruned, max_new_tokens=tokens,
                                  full_prefix_tokens=int(prepared.inputs_embeds.shape[1]),
                                  budget_profile=cell.budget, policy=cell.policy, staging=cell.staging,
                                  recent_mode=cell.recent_mode, seed=cell.seed or 42, phase=cell.phase,
                                  visual_source_positions=prepared.visual_positions, **kwargs)
    if cell.policy in LEGACY_POLICIES:
        from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
        return generate_matched_v3(adapter.model, pruned, max_new_tokens=tokens,
                                  full_prefix_tokens=int(prepared.inputs_embeds.shape[1]),
                                  budget_profile=cell.budget, policy=cell.policy, staging=cell.staging,
                                  visual_source_positions=prepared.visual_positions, **kwargs)
    raise ValueError("unsupported pilot policy")


def measured_cell(adapter, sample, cell, *, max_new_tokens=4096, fixed_output=False,
                  optimized=True, record_events=False):
    import torch
    from learned_score.mlsys_eval.timing import TimingContract
    if hasattr(sample.get("image"), "seek"):
        sample["image"].seek(0)
    device = adapter.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timer = TimingContract(synchronize=lambda: torch.cuda.synchronize(device))
    with timer.stage("preprocess"):
        prepared = adapter.prepare_sample(sample)
    with timer.stage("selector"):
        selected = torch.arange(prepared.visual_positions.numel(), device=device)
        pruned = adapter.physically_prune(prepared, selected, "original")
    prefix = int(prepared.inputs_embeds.shape[1])
    observed = {"input_prefix_tokens": prefix, "retained_prefix_tokens": int(pruned.inputs_embeds.shape[1]),
                "input_visual_tokens": int(selected.numel()), "retained_visual_tokens": int(selected.numel()),
                "selected_visual_indices_sha256": digest(selected.detach().cpu().tolist())}
    preprocessing_peak = torch.cuda.max_memory_allocated(device)
    result = _generate(adapter, pruned, prepared, cell, max_new_tokens,
                       optimized=optimized, record_events=record_events, stop_on_eos=not fixed_output,
                       on_first_token=timer.mark_first_token)
    runtime = result.metrics
    runtime.update({"cache_policy": cell.policy, "timing_isolated": False,
                    "original_visual_tokens": int(selected.numel()), "retained_visual_tokens": int(selected.numel())})
    timer.stages["prefill"] = runtime["prefill_ttft_seconds"]
    timer.stages["compression"] = runtime.get("compaction_cpu_seconds") or 0.
    timer.stages["decode"] = max(0., runtime["decode_seconds"] - timer.stages["compression"])
    with timer.stage("postprocess"):
        ids = result.ids.detach().cpu()
        prediction = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
    timing = timer.finish()
    timing["timing_mode"] = "quality_shared_node_synchronized_not_isolated"
    runtime["full_request_peak_allocated_bytes"] = max(preprocessing_peak, torch.cuda.max_memory_allocated(device))
    runtime["full_request_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    capacity = prefix + cell.budget + 1
    budget = {"kv_budget_slots": capacity, "kv_retained_slots": capacity - cell.staging,
              "staging_slots": cell.staging, "kv_budget_bytes": runtime["cache"]["kv_capacity_bytes"],
              "generated_budget": capacity - cell.staging - prefix if cell.policy in NEW_POLICIES else None}
    del result, pruned, selected, prepared
    return prediction, ids, timing, runtime, observed, budget


def smoke_cells(protocol):
    unique = {}
    for cell in cell_objects(protocol):
        unique.setdefault((cell.policy, cell.budget, cell.staging, cell.recent_mode, cell.phase), cell)
    return tuple(unique.values())


def smoke_checks(adapter, sample, protocol):
    import torch
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    prepared = adapter.prepare_sample(sample)
    selected = torch.arange(prepared.visual_positions.numel(), device=adapter.device)
    pruned = adapter.physically_prune(prepared, selected, "original")
    prefix = int(prepared.inputs_embeds.shape[1])
    native = generate_dynamic(adapter.model, pruned, 64, stop_on_eos=False)
    expected = native.ids.cpu()
    del native
    checks = []
    for cell in smoke_cells(protocol):
        early, forced, healthy = [], [], True
        for optimized in (False, True):
            result = _generate(adapter, pruned, prepared, cell, 64, optimized=optimized, stop_on_eos=False)
            early.append(result.ids.cpu())
            healthy &= result.metrics["cache"]["compaction_events"] == 0
            del result
            result = _generate(adapter, pruned, prepared, cell, max(512, cell.budget + 256),
                               optimized=optimized, stop_on_eos=False)
            forced.append(result.ids.cpu())
            stats = result.metrics["cache"]
            cap = prefix + cell.budget + 1
            healthy &= (stats["compaction_events"] > 0 and stats["capacity_tokens_per_layer"] == cap
                        and stats["peak_active_tokens"] <= cap
                        and stats["protects_entire_prefix"] == (cell.policy in NEW_POLICIES))
            if cell.policy in NEW_POLICIES:
                wanted = set(pruned.retained_source_positions.cpu().tolist())
                healthy &= all(wanted.issubset(set(head.tolist())) for layer in range(len(result.cache))
                               for head in result.cache.layer_source_positions(layer)[0].cpu())
            healthy &= all(bool(torch.isfinite(t).all()) for t in result.cache.key_cache + result.cache.value_cache)
            del result
        values = (all(torch.equal(expected, ids) for ids in early), torch.equal(forced[0], forced[1]), bool(healthy))
        checks.extend({"cell": cell.key, "check": key, "passed": bool(passed)} for key, passed in zip(CHECKS, values))
        if not all(values):
            raise RuntimeError(f"v4 physical correctness failed for {cell.key}: {values}")
    return checks


def smoke_path(protocol_path, model):
    return Path(protocol_path).parent / "smoke" / f"{model}.complete.json"


def verify_smoke(protocol_path, protocol, base, model):
    path = smoke_path(protocol_path, model)
    value = json.loads(path.read_text())
    if (value.get("protocol_sha256") != protocol["sha256"] or value.get("model") != model
            or value.get("passed") is not True or value.get("request_ids") != base["native_request_ids"]):
        raise ValueError("v4 smoke gate missing or mismatched")
    artifacts = value.get("artifacts", {})
    if len(artifacts) != 8:
        raise ValueError("eight physical smoke pages are required")
    for filename, sha in artifacts.items():
        if file_digest(filename) != sha:
            raise ValueError("physical smoke artifact changed")
    return value


def seal_smoke(args, protocol, base):
    path = smoke_path(args.protocol, args.model)
    artifacts = {}
    expected_checks = {(cell.key, check) for cell in smoke_cells(protocol) for check in CHECKS}
    for rank, rid in enumerate(base["native_request_ids"]):
        shard = path.parent / f"{args.model}.rank{rank}.complete.json"
        value = json.loads(shard.read_text())
        checks = value.get("checks", [])
        if (value.get("protocol_sha256") != protocol["sha256"] or value.get("model") != args.model
                or value.get("request_id") != rid or value.get("passed") is not True
                or len(checks) != len(expected_checks)
                or {(r["cell"], r["check"]) for r in checks} != expected_checks
                or any(r.get("passed") is not True for r in checks)):
            raise ValueError("incomplete or mismatched physical smoke shard")
        artifacts[str(shard)] = file_digest(shard)
    write_immutable(path, {"format": "mlsys_cache_history_smoke_v4", "passed": True,
                           "protocol_sha256": protocol["sha256"], "model": args.model,
                           "request_ids": base["native_request_ids"], "artifacts": artifacts,
                           "native_gate": verify_native_gate(base, args.model), "final_labels_opened": False})


def run_smoke(args, protocol, base):
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    path = smoke_path(args.protocol, args.model).parent / f"{args.model}.rank{args.smoke_index}.complete.json"
    rid = base["native_request_ids"][args.smoke_index]
    if path.exists():
        value = json.loads(path.read_text())
        if (value.get("protocol_sha256") != protocol["sha256"] or value.get("request_id") != rid
                or value.get("model") != args.model or value.get("passed") is not True):
            raise ValueError("existing smoke shard does not match this run")
        return
    adapter, dataset, _, _, entries = load_model(base, args.model, args.device, with_scorer=False)
    versions = frozen_versions(adapter.model)
    entry = next(e for e in entries if e["row_hash"] == rid)
    checks = smoke_checks(adapter, read_entry(dataset, entry), protocol)
    assert_frozen(adapter.model, versions)
    write_immutable(path, {"protocol_sha256": protocol["sha256"], "model": args.model,
                           "request_id": rid, "checks": checks, "passed": True,
                           "final_labels_opened": False, "execution_context": execution_context()})
    print(json.dumps({"event": "v4_smoke_complete", "model": args.model, "index": args.smoke_index}), flush=True)


def run_page(args, protocol, base):
    import torch
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    verify_smoke(args.protocol, protocol, base, args.model)
    panel = collect_panel(args.protocol, args.output, require_complete=False, model=args.model)
    entry = entries_for(protocol)[args.request_index]
    if any(key[0] == entry["row_hash"] for key in panel["rows"]):
        return
    adapter, dataset, _, _, _ = load_model(base, args.model, args.device, with_scorer=False)
    versions = frozen_versions(adapter.model)
    metadata, directory = panel["metadata"], Path(panel["directory"])
    write_immutable(directory / "metadata.json", metadata)
    sample = read_entry(dataset, entry)
    cells = cell_objects(protocol)
    for cell in cells:
        measured_cell(adapter, sample, cell, max_new_tokens=8)
    provenance, attempt, artifacts, rows = execution_context(), uuid.uuid4().hex, {}, []
    offset = int(entry["row_hash"][:8], 16) % len(cells)
    for cell in cells[offset:] + cells[:offset]:
        torch.cuda.empty_cache()
        prediction, ids, timing, runtime, observed, budget = measured_cell(adapter, sample, cell)
        assert_frozen(adapter.model, versions)
        tokens = ids[0].tolist()
        record = {"format": FORMAT, "identity_sha256": digest(metadata), "run_context": digest(metadata),
                  "protocol_sha256": protocol["sha256"], "model": metadata["model"], "model_name": args.model,
                  "dataset": "olmOCR-Bench", "dataset_kind": "olmocr", "split": "scorer_dev", "stage": protocol["stage"],
                  "request_id": entry["row_hash"], "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                  "method": cell.key, "method_label": cell.policy.replace("_", " "), "cache_policy": cell.policy,
                  "cache_method": cell.policy, "random_seed": cell.seed, "cell": dict(asdict(cell), key=cell.key),
                  "selector": "full", "visual_retention": 1., "comparison_group": f"v100_b{cell.budget}",
                  "budget_profile": f"base{cell.budget}", "max_new_tokens": 4096, "batch_size": 1,
                  "serving_measurement": False, "repetition": 0, "repeat": 1, "attempt": attempt,
                  "execution_context": provenance, "hardware": torch.cuda.get_device_name(adapter.device),
                  "quality_run_only": True, "timing_isolated": False, "final_labels_opened": False,
                  "status": "ok", "prediction": prediction, "quality_score": None,
                  "quality_status": "awaiting_official_score", "generated_tokens": len(tokens),
                  "generated_ids": tokens, "generated_ids_sha256": digest(tokens), "natural_stop_enabled": True,
                  "metrics": runtime, "peak_allocated_bytes": runtime["full_request_peak_allocated_bytes"],
                  "peak_reserved_bytes": runtime["full_request_peak_reserved_bytes"], **timing, **observed, **budget}
        path = directory / "attempts" / entry["row_hash"] / attempt / f"{cell.key}.json"
        write_immutable(path, record)
        artifacts[str(path.resolve())] = file_digest(path)
        rows.append(record)
        print(json.dumps({"event": "v4_pilot_cell", "model": args.model, "index": args.request_index,
                          "cell": cell.key, "tokens": len(tokens), "seconds": timing["end_to_end_seconds"]}), flush=True)
    validate_page(rows, entry, metadata)
    write_immutable(directory / f"{entry['row_hash']}.complete.json", {
        "format": "mlsys_cache_history_page_complete_v4", "passed": True,
        "protocol_sha256": protocol["sha256"], "metadata_sha256": digest(metadata),
        "request_id": entry["row_hash"], "attempt": attempt, "artifacts": artifacts, "final_labels_opened": False})
    collect_panel(args.protocol, directory, require_complete=False, model=args.model)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen25", "llava_next"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--request-index", type=int, choices=range(24), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-index", type=int, choices=range(8), default=0)
    parser.add_argument("--seal-smoke", action="store_true")
    args = parser.parse_args(argv)
    protocol, base = load_inputs(args.protocol)
    metadata_for(protocol, base, args.model)
    if args.seal_smoke:
        seal_smoke(args, protocol, base)
    else:
        require_cuda(args.device)
        if args.smoke:
            run_smoke(args, protocol, base)
        else:
            if args.output is None:
                args.output = args.protocol.parent / "documents" / args.model
            run_page(args, protocol, base)


if __name__ == "__main__":
    main()
