"""Prepare source-pinned cache-history development stages; dry-run by default."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

from .cache_protocol_v4 import (ACTIVE_MODELS, FORMAT, SEEDS, build_stage_protocol,
                                cells_for_stage, load_protocol, seal,
                                select_development_entries, validate_protocol, _read_sealed)
from .followup_common import write_immutable
from .state import file_digest


DEFAULT_PARENT = Path("reproduction_results/mlsys_cache_methods_20261013_v3/protocol.json")
DEFAULT_OUTPUT = Path("reproduction_results/mlsys_cache_methods_20261014_v4")
DRIVER = "learned_score.mlsys_sprint.cache_methods_v4"


def _absolute(path, root):
    path = Path(path)
    return (path if path.is_absolute() else Path(root) / path).resolve()


def source_paths(root):
    names = ("learned_score/mlsys_runtime/matched_cache_v4.py",
             "learned_score/mlsys_runtime/matched_generation_v4.py",
             "learned_score/mlsys_sprint/cache_protocol_v4.py",
             "learned_score/mlsys_sprint/prepare_cache_methods_v4.py",
             "learned_score/mlsys_sprint/cache_methods_v4.py",
             "learned_score/mlsys_sprint/cache_coordinator_v4.py",
             "learned_score/mlsys_next/cache_report_v4.py")
    return [Path(root) / name for name in names]


def _reference_report(parent_path, root):
    latest = json.loads((parent_path.parent / "evaluation/latest.json").read_text())
    if latest.get("complete") is not True or latest.get("final_pages_opened") is not False:
        raise ValueError("v4 requires the complete existing development report")
    report_path = _absolute(latest["models"]["qwen25"]["official_report"], root)
    report = json.loads(report_path.read_text())
    if report.get("final_labels_closed") is not True:
        raise ValueError("reference report must keep final labels closed")
    score_paths = sorted(report_path.parent.glob("*/official_scores.json"))
    if len(score_paths) != 10:
        raise ValueError("reference report must bind all ten original physical cells")
    return report_path, [report_path, *score_paths]


def _native_summary(base):
    """Reuse existing label-free audit; do not rerun or alter its observations."""
    result, paths = {}, []
    for name, filename in base.get("native_gates", {}).items():
        path = Path(filename)
        gate = json.loads(path.read_text())
        if gate.get("passed") is not True or gate.get("final_pages_opened") is not False:
            raise ValueError("existing native gate is missing or unsuccessful")
        paths.append(path)
        pages = []
        for page_name, expected in gate.get("page_artifacts", {}).items():
            page_path = Path(page_name)
            if file_digest(page_path) != expected:
                raise ValueError("existing native audit page changed")
            record = json.loads(page_path.read_text())
            if record.get("passed") is not True:
                raise ValueError("existing native audit page failed")
            outcome = record.get("result", {})
            pages.append({"request_id": record["request_id"], "passed": True,
                          "native_generated_tokens": outcome.get("native_generated_tokens"),
                          "native_stop_reason": outcome.get("native_stop_reason"),
                          "native_repeated_eight_word_fraction": outcome.get("native_repeated_eight_word_fraction")})
            paths.append(page_path)
        result[name] = {"passed": True, "pages": pages,
                        "scope": "outcome-selected native parity diagnostics; not quality validation"}
    return result, paths


def build_protocol(root, parent_path, output, *, created=None, extra_sources=()):
    root = Path(root).resolve()
    parent_path, output = _absolute(parent_path, root), _absolute(output, root)
    if not output.is_relative_to(root / "reproduction_results") or output == parent_path.parent:
        raise ValueError("new v4 output must be a separate reproduction_results directory")
    parent = _read_sealed(parent_path)
    if (parent.get("format") != "mlsys_matched_cache_protocol_v3"
            or parent.get("split") != "scorer_dev" or parent.get("final_pages_opened") is not False):
        raise ValueError("expected unchanged v3 development parent")
    for filename, expected in parent["input_hashes"].items():
        if file_digest(filename) != expected:
            raise ValueError(f"frozen v3 input changed: {filename}")
    base_path = Path(parent["base_protocol"])
    if file_digest(base_path) != parent["base_sha256"]:
        raise ValueError("original model-loading protocol changed")
    base = _read_sealed(base_path)
    manifest_path = Path(parent["manifest"])
    manifest = json.loads(manifest_path.read_text())
    development = manifest["splits"]["scorer_dev"]
    if set(parent["development_request_ids"]) != {entry["row_hash"] for entry in development}:
        raise ValueError("parent development identities differ from the manifest")
    entries = select_development_entries(development)
    # Only final identities are inspected here, never their benchmark labels.
    final_sources = {entry["group_id"] for entry in manifest["splits"].get("final_test", [])}
    if final_sources.intersection(entry["group_id"] for entry in entries):
        raise ValueError("selected development source overlaps reserved final identities")
    reference, reference_paths = _reference_report(parent_path, root)
    native_summary, native_paths = _native_summary(base)
    sources = [*source_paths(root), *(_absolute(p, root) for p in extra_sources)]
    for path in sources:
        if not path.is_file():
            raise ValueError(f"v4 source is not ready to freeze: {path}")
    evaluator_root = root / ".data/mlsys_olmocr_evaluator"
    cache_patch = root / "reproduction_results/mlsys_20261012/olmocr_evaluator_cache_patch.json"
    pinned_paths = [parent_path, base_path, manifest_path, cache_patch, *reference_paths, *native_paths, *sources]
    for model in parent["models"]:
        checkpoint = Path(model["checkpoint"])
        if file_digest(checkpoint) != model["checkpoint_sha256"]:
            raise ValueError("frozen scorer checkpoint changed")
        pinned_paths.append(checkpoint)
    inputs = {**parent["input_hashes"], **{str(p): file_digest(p) for p in pinned_paths}}
    coverage = Counter(category for entry in entries for category in set(entry["categories"]))
    value = {"format": FORMAT, "workspace": str(root), "output_root": str(output),
             "created_utc_epoch": int(time.time() if created is None else created),
             "parent_protocol": str(parent_path), "parent_sha256": file_digest(parent_path),
             "base_protocol": str(base_path), "base_sha256": file_digest(base_path),
             "manifest": str(manifest_path), "manifest_sha256": file_digest(manifest_path),
             "evaluator_root": str(evaluator_root), "cache_patch": str(cache_patch),
             "cache_patch_sha256": file_digest(cache_patch),
             "models": parent["models"], "active_models": list(ACTIVE_MODELS),
             "scorer_training_manifest": parent["scorer_training_manifest"],
             "stage": "stage1", "stage_selection": None, "split": "scorer_dev",
             "entries": entries, "development_entries": entries,
             "development_request_ids": [entry["row_hash"] for entry in entries],
             "development_pages": 24, "independent_sources": 24,
             "category_counts": dict(sorted(coverage.items())),
             "source_selection": "metadata-only least-represented available category, SHA ties, one page per source",
             "native_request_ids": base["native_request_ids"],
             "micro_request_ids": sorted(entry["row_hash"] for entry in entries)[:2],
             "micro_selection": "first two SHA-ordered selected development identities; outcomes unused",
             "smoke_coverage": "implementation families; random seed42 representative; all three seeds physically evaluated in quality runs",
             "existing_native_audit": native_summary,
             "llava_audit_note": "native parity already passed; FP16 then BF16 loading requires separate diagnosis, not silent correction",
             "reference_report": str(reference), "reference_report_sha256": file_digest(reference),
             "reference_scope": "existing same-page development scores only; neither final annotations nor new quality labels read during preparation",
             "cells": [asdict(cell) for cell in cells_for_stage("stage1")],
             "max_new_tokens": 4096, "natural_eos": True, "visual_retention": 1.0,
             "random_seeds": list(SEEDS), "relative_quality_floor": .95,
             "relative_quality_floor_definition": "candidate official score divided by same-page all-token official score >= 0.95; not a significance claim",
             "budget_definition": "M=original_full_prefix+B+1; staging is inside M",
             "stage_order": ["stage1", "stage2", "stage3"],
             "final_labels_opened": False, "final_pages_opened": False,
             "all_results_development": True, "fresh_confirmation_authorized": False,
             "input_hashes": inputs}
    return seal(validate_protocol(value))


def json_file_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()


def build_plan(protocol, output, root, name):
    validate_protocol(protocol)
    root, output = Path(root).resolve(), _absolute(output, root)
    if not output.is_relative_to(root / "reproduction_results"):
        raise ValueError("v4 stage output must remain inside reproduction_results")
    if name not in protocol["active_models"]:
        raise ValueError("model is not enabled by this v4 stage")
    path = output / "protocol.json"
    command = [str(root / ".conda/asymvlm/bin/python"), "-u", "-m", DRIVER,
               "--protocol", str(path), "--model", name,
               "--output", str(output / "documents" / name)]
    tasks, smoke_ids = [], []
    for index, _ in enumerate(protocol["native_request_ids"]):
        ident = f"{name}-v4-{protocol['stage']}-smoke-{index}"
        smoke_ids.append(ident)
        tasks.append({"id": ident, "resource": "gpu", "tag": "cache_v4_smoke",
                      "minimum_start_seconds": 600, "max_attempts": 2,
                      "argv": [*command, "--smoke", "--smoke-index", str(index)],
                      "artifacts": [{"path": str(output / "smoke" / f"{name}.rank{index}.complete.json"), "passed": True}]})
    gate = f"{name}-v4-{protocol['stage']}-seal"
    tasks.append({"id": gate, "resource": "cpu", "tag": "cache_v4_seal",
                  "minimum_start_seconds": 30, "depends_on": smoke_ids,
                  "argv": [*command, "--seal-smoke"],
                  "artifacts": [{"path": str(output / "smoke" / f"{name}.complete.json"), "passed": True}]})
    for index, rid in enumerate(protocol["development_request_ids"]):
        tasks.append({"id": f"{name}-v4-{protocol['stage']}-page-{index:03d}",
                      "resource": "gpu", "tag": "cache_v4_quality", "minimum_start_seconds": 900,
                      "max_attempts": 2, "depends_on": [gate],
                      "argv": [*command, "--request-index", str(index)],
                      "artifacts": [{"path": str(output / "documents" / name / f"{rid}.complete.json"), "passed": True}]})
    return seal({"format": "mlsys_task_plan_v1", "cwd": str(root), "unexpected_retries": 1,
                 "protocol_sha256": protocol["sha256"], "model_name": name,
                 "input_hashes": {**protocol["input_hashes"], str(path): json_file_digest(protocol)},
                 "tasks": tasks, "all_results_development": True, "final_pages_opened": False})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-protocol", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output", "--output-root", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("stage1", "stage2", "stage3"), default="stage1")
    parser.add_argument("--selection", type=Path)
    parser.add_argument("--extra-source", action="append", type=Path, default=[])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--write", action="store_true")
    modes.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output, parent = _absolute(args.output, root), _absolute(args.parent_protocol, root)
    if args.stage == "stage1":
        if args.selection:
            parser.error("stage1 cannot consume a selection")
        existing = output / "protocol.json"
        created = json.loads(existing.read_text())["created_utc_epoch"] if existing.exists() else int(time.time())
        protocol = build_protocol(root, parent, output, created=created, extra_sources=args.extra_source)
    else:
        if args.selection is None:
            parser.error("later stages require --selection")
        if output == parent.parent:
            parser.error("later stages must use a new directory, never overwrite their parent")
        selection_path = _absolute(args.selection, root)
        decision = json.loads(selection_path.read_text())
        protocol = build_stage_protocol(load_protocol(parent), args.stage, decision["selection"],
                                        selection_path=selection_path)
    plans = {name: build_plan(protocol, output, root, name) for name in protocol["active_models"]}
    if args.write:
        write_immutable(output / "protocol.json", protocol)
        for name, plan in plans.items():
            write_immutable(output / "plans" / name / "plan.json", plan)
    print(json.dumps({"written": bool(args.write), "output": str(output), "stage": protocol["stage"],
                      "protocol_sha256": protocol["sha256"], "pages": 24,
                      "category_counts": protocol["category_counts"], "cells": len(protocol["cells"]),
                      "tasks": {name: len(plan["tasks"]) for name, plan in plans.items()},
                      "final_labels_opened": False}, sort_keys=True))


if __name__ == "__main__":
    main()
