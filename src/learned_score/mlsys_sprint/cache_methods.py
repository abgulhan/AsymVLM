"""Same-total-KV competing methods on unopened-final, development OCR pages.

One task is one page and all its cache methods, in a single model process.
An interrupted page is rerun in full; partial attempts are never spliced into
fresh controls. Quality-run timings are diagnostics, not isolated speedups.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import uuid

from . import benchmark
from .followup_common import (load_followup_protocol, load_model, model_spec,
                              protocol_manifest, require_cuda, verify_native_gate,
                              write_immutable)
from .long_document_benchmark import execution_context
from .state import digest, file_digest


FORMAT = "mlsys_matched_cache_protocol_v1"
RUNTIME_METHODS = {"asymvlm": "asymvlm", "h2o": "h2o_gqa_native_positions",
                   "streamingllm": "streaming_llm_rolling"}
LABELS = {"none": "All tokens, no cache eviction",
          "asymvlm": "AsymVLM (protected image/prompt; recent generated text)",
          "h2o": "H2O (all-layer attention; GQA adaptation)",
          "streamingllm": "StreamingLLM (4 sinks + recent; rolling positions)"}


def cache_methods(name):
    if name == "qwen25":
        return ("asymvlm", "h2o")
    if name == "llava_next":
        return ("asymvlm", "h2o", "streamingllm")
    raise ValueError("model is outside the verified cache comparison")


@dataclass(frozen=True)
class CacheCell:
    cache_method: str
    visual_retention: float
    profile: int | None

    @property
    def selector(self):
        return "full" if self.visual_retention == 1. else "scorer"

    @property
    def key(self):
        if self.cache_method == "none":
            return "full_v100_off"
        return f"{self.cache_method}_v{int(self.visual_retention * 100)}_b{self.profile}"

    @property
    def group(self):
        return "unrestricted" if self.profile is None else f"v{int(self.visual_retention * 100)}_b{self.profile}"


def cells_for(name):
    return (CacheCell("none", 1., None),) + tuple(
        CacheCell(method, vision, budget) for vision, budget in ((1., 512), (1., 1024), (.9, 512))
        for method in cache_methods(name))


def matched_budget(full_prefix, retained_prefix, profile, max_new_tokens=4096):
    if (type(full_prefix) is not int or type(retained_prefix) is not int
            or not 1 <= retained_prefix <= full_prefix or profile not in (512, 1024)):
        raise ValueError("invalid original/retained prefix or budget profile")
    steady = full_prefix + profile
    # A common C+1 capacity must be reachable under the declared output cap.
    # Never silently clamp one visual-retention arm to a smaller allocation.
    if steady + 1 > retained_prefix + max_new_tokens - 1:
        raise ValueError("output cap cannot support the common total KV allowance")
    return {"kv_retained_slots": steady, "kv_budget_slots": steady + 1,
            "generated_budget": steady - retained_prefix}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("sha256") != digest({k: v for k, v in protocol.items() if k != "sha256"}):
        raise ValueError("cache comparison protocol digest mismatch")
    if (protocol.get("format") != FORMAT or protocol.get("split") != "scorer_dev"
            or protocol.get("final_pages_opened") is not False or protocol.get("slack") != 1
            or protocol.get("max_new_tokens") != 4096):
        raise ValueError("unsupported cache comparison protocol")
    if file_digest(protocol["base_protocol"]) != protocol["base_sha256"]:
        raise ValueError("base protocol changed")
    for filename, expected in protocol["input_hashes"].items():
        if file_digest(filename) != expected:
            raise ValueError(f"cache comparison input changed: {filename}")
    base = load_followup_protocol(protocol["base_protocol"])
    if protocol["models"] != base["models"]:
        raise ValueError("cache comparison models differ from frozen base")
    return protocol, base


def smoke_gate(protocol, name):
    path = Path(protocol["smoke_gates"][name])
    value = json.loads(path.read_text())
    if (value.get("format") != "mlsys_matched_cache_smoke_v1" or value.get("passed") is not True
            or value.get("protocol_sha256") != protocol["sha256"] or value.get("model_name") != name
            or value.get("cache_methods") != list(cache_methods(name))):
        raise ValueError("cache comparison smoke gate absent/failed/mismatched")
    base = json.loads(Path(protocol["base_protocol"]).read_text())
    expected = {(rid, vision, method, check) for rid in base["native_request_ids"]
                for vision in (1., .9) for method in cache_methods(name)
                for check in ("native_no_eviction_exact_ids_64", "forced_eviction_capacity_and_policy_48")}
    checks = value.get("checks", [])
    observed = {(c.get("request_id"), c.get("vision"), c.get("method"), c.get("check")) for c in checks}
    if observed != expected or len(checks) != len(expected) or not all(c.get("passed") is True for c in checks):
        raise ValueError("smoke gate does not cover the complete passing physical check matrix")
    return {"path": str(path), "sha256": file_digest(path), "passed": True}


def metadata_for(protocol, base, name, backbone):
    manifest, _ = protocol_manifest(base)
    return {"format": "mlsys_matched_cache_panel_v1", "protocol_sha256": protocol["sha256"],
            "model_name": name, "manifest_sha256": manifest.file_sha256,
            "dataset_kind": "olmocr", "split": "scorer_dev", "stage": "development",
            "backbone": backbone, "checkpoint_sha256": model_spec(base, name)["checkpoint_sha256"],
            "scorer_training_manifest_sha256": file_digest(base["scorer_training_manifest"]),
            "cache_methods": list(cache_methods(name)), "cells": [c.key for c in cells_for(name)],
            "unsupported_cache_methods": ({"streamingllm": "Qwen image mRoPE has no validated upstream rolling-position equivalent"}
                                          if name == "qwen25" else {}),
            "native_gate": verify_native_gate(base, name), "smoke_gate": smoke_gate(protocol, name),
            "max_new_tokens": 4096, "slack": 1, "repetitions": 1, "batch_size": 1,
            "fixed_output_tokens": False, "all_results_development": True, "final_pages_opened": False,
            "budget_definition": "C=original_full_prefix+B; allocated_KV_slots=C+1, shared_across_methods_and_vision_retention",
            "pairing_unit": "same_page_same_model_process", "quality_run_only": True,
            "asymvlm_text_policy": "protected_prefix_recent_only_not_original_importance_eviction"}


def validate_page(rows, entry, metadata):
    """Fail closed on incomplete pairing, budgets, identities, or method drift."""
    cells = {cell.key: cell for cell in cells_for(metadata["model_name"])}
    if len(rows) != len(cells) or {r["method"] for r in rows} != set(cells):
        raise ValueError("page does not contain exactly the predeclared cache cells")
    contexts, full_prefixes, kept, allocations, attempts = set(), set(), {}, {}, set()
    for row in rows:
        cell = cells[row["method"]]
        required = {"format": "mlsys_request_result_v1", "identity_sha256": digest(metadata),
                    "model": metadata["backbone"]["model"], "dataset": "olmOCR-Bench", "split": "scorer_dev",
                    "request_id": entry["row_hash"], "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                    "stage": "development", "status": "ok", "cache_method": cell.cache_method,
                    "visual_retention": cell.visual_retention, "selector": cell.selector,
                    "max_new_tokens": 4096, "natural_stop_enabled": True,
                    "repetition": 0, "quality_score": None, "quality_status": "awaiting_official_score",
                    "comparison_group": cell.group, "budget_profile": "unrestricted" if cell.profile is None else f"base{cell.profile}"}
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError("page row violates frozen identity/configuration")
        execution = row.get("execution_context", {})
        if not execution.get("hostname") or execution.get("cluster") not in ("clusterb", "clustera"):
            raise ValueError("page lacks verified cluster/process provenance")
        contexts.add(digest(execution))
        attempts.add(row.get("attempt"))
        if not isinstance(row["prediction"], str) or not 1 <= row["generated_tokens"] <= 4096:
            raise ValueError("invalid generated prediction")
        full, prefix, original, retained = (row[k] for k in (
            "input_prefix_tokens", "retained_prefix_tokens", "input_visual_tokens", "retained_visual_tokens"))
        if retained != max(1, int(original * cell.visual_retention)) or full - prefix != original - retained:
            raise ValueError("incorrect visual-token retention count")
        full_prefixes.add(full)
        selected_key = (row["selected_visual_indices_sha256"], retained, prefix)
        kept.setdefault(cell.visual_retention, set()).add(selected_key)
        cache = row["metrics"]["cache"]
        if cell.cache_method != "none":
            budget = matched_budget(full, prefix, cell.profile)
            if (row["kv_budget_slots"] != budget["kv_budget_slots"] or row["kv_retained_slots"] != budget["kv_retained_slots"]
                    or cache["capacity_tokens_per_layer"] != budget["kv_budget_slots"]
                    or row["kv_budget_bytes"] != cache["kv_capacity_bytes"]
                    or cache["active_tokens"] > budget["kv_budget_slots"]
                    or cache["protects_entire_prefix"] != (cell.cache_method == "asymvlm")
                    or row["metrics"].get("exploratory_mrope_flattening") is not False
                    or row["metrics"].get("method") != RUNTIME_METHODS[cell.cache_method]):
                raise ValueError("physical cache differs from declared matched budget/policy")
            expected_generated = budget["generated_budget"] if cell.cache_method == "asymvlm" else None
            if row["generated_budget"] != expected_generated:
                raise ValueError("misreported generated-text allowance")
            allocations.setdefault(cell.profile, set()).add((row["kv_budget_slots"], row["kv_budget_bytes"]))
        elif any(row[k] is not None for k in ("kv_budget_slots", "kv_retained_slots", "kv_budget_bytes", "generated_budget")):
            raise ValueError("unrestricted reference is incorrectly budget-labeled")
    if (len(contexts) != 1 or len(full_prefixes) != 1 or any(len(v) != 1 for v in kept.values())
            or len(attempts) != 1 or not isinstance(next(iter(attempts)), str)):
        raise ValueError("page methods crossed processes or used different visual selections")
    if any(len(v) != 1 for v in allocations.values()):
        raise ValueError("cache capacities differ across methods or visual retention arms")


def collect_panel(protocol_path, directory, require_complete=True):
    protocol, base = load_protocol(protocol_path)
    manifest, entries = protocol_manifest(base)
    directory = Path(directory)
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    name = metadata["model_name"]
    if metadata != metadata_for(protocol, base, name, metadata["backbone"]):
        raise ValueError("panel metadata differs from frozen protocol")
    from learned_score.mlsys_scorer.checkpoints import load_checkpoint
    checkpoint_path = model_spec(base, name)["checkpoint"]
    checkpoint = load_checkpoint(checkpoint_path, device="cpu")
    benchmark.require_completed_checkpoint(checkpoint_path, checkpoint)
    benchmark.validate_checkpoint_identity(checkpoint, metadata["scorer_training_manifest_sha256"], metadata["backbone"])
    context = digest({"directory": str(directory.absolute()), "identity_sha256": digest(metadata)})
    rows, artifacts = {}, {str(metadata_path): file_digest(metadata_path)}
    for rank, entry in enumerate(entries):
        seal_path, complete_path = (directory / f"rank{rank}.{suffix}.json" for suffix in ("integrity", "complete"))
        if not seal_path.exists():
            continue
        seal = json.loads(seal_path.read_text())
        if (seal.get("format") != "mlsys_matched_cache_page_integrity_v1" or seal.get("rank") != rank
                or seal.get("protocol_sha256") != protocol["sha256"] or seal.get("identity_sha256") != digest(metadata)
                or seal.get("metadata_sha256") != file_digest(metadata_path)
                or seal.get("completion_sha256") != file_digest(complete_path)):
            raise ValueError("page integrity seal does not match metadata/completion")
        completion = json.loads(complete_path.read_text())
        if completion != {"identity_sha256": digest(metadata), "request_id": entry["row_hash"],
                          "cells": len(cells_for(name)), "rank": rank, "attempt": seal["attempt"]}:
            raise ValueError("page completion differs from its assignment")
        attempt = seal["attempt"]
        if len(attempt) != 32 or any(x not in "0123456789abcdef" for x in attempt):
            raise ValueError("invalid page attempt identifier")
        expected = {f"attempts/rank{rank}/{attempt}/{c.key}.json" for c in cells_for(name)}
        if set(seal["row_sha256"]) != expected:
            raise ValueError("page seal does not bind exactly one complete attempt")
        page = []
        for relative, sha in seal["row_sha256"].items():
            path = directory / relative
            if file_digest(path) != sha:
                raise ValueError("sealed page row changed")
            page.append(json.loads(path.read_text()))
        validate_page(page, entry, metadata)
        if any(row["attempt"] != attempt for row in page):
            raise ValueError("row attempt differs from sealed attempt")
        for row in page:
            rows[(row["request_id"], row["method"], 0)] = {**row, "run_context": context}
        artifacts[str(seal_path)] = file_digest(seal_path)
    complete = len(rows) == 100 * len(cells_for(name))
    if require_complete and not complete:
        raise ValueError("official scoring requires all100 development pages and every cache cell")
    return {"metadata": metadata, "entries": entries, "rows": rows, "complete": complete,
            "context": context, "integrity_artifacts": artifacts, "directory": str(directory),
            "label": f"{name} matched-total-KV cache methods (development)",
            "checkpoint": {"variant": "rank32", "seed": 42, "sha256": metadata["checkpoint_sha256"],
                           "identity": checkpoint["identity"]}}


def recover_completed_page(protocol, base, directory, rank):
    """Recover a crash between completion and integrity publication, not cells."""
    directory = Path(directory)
    complete_path = directory / f"rank{rank}.complete.json"
    seal_path = directory / f"rank{rank}.integrity.json"
    if not complete_path.exists() or seal_path.exists():
        return
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    name = metadata["model_name"]
    if metadata != metadata_for(protocol, base, name, metadata["backbone"]):
        raise ValueError("unsealed completed page has incorrect metadata")
    entry = protocol_manifest(base)[1][rank]
    marker = json.loads(complete_path.read_text())
    attempt = marker.get("attempt", "")
    if (len(attempt) != 32 or any(x not in "0123456789abcdef" for x in attempt)
            or marker != {"identity_sha256": digest(metadata), "request_id": entry["row_hash"],
                          "cells": len(cells_for(name)), "rank": rank, "attempt": attempt}):
        raise ValueError("unsealed completion marker is malformed")
    names = [f"attempts/rank{rank}/{attempt}/{cell.key}.json" for cell in cells_for(name)]
    page = [json.loads((directory / relative).read_text()) for relative in names]
    validate_page(page, entry, metadata)
    if any(row["attempt"] != attempt for row in page):
        raise ValueError("unsealed row attempt differs from completion")
    write_immutable(seal_path, {"format": "mlsys_matched_cache_page_integrity_v1", "rank": rank,
                               "protocol_sha256": protocol["sha256"], "identity_sha256": digest(metadata),
                               "metadata_sha256": file_digest(metadata_path),
                               "completion_sha256": file_digest(complete_path), "attempt": attempt,
                               "row_sha256": {relative: file_digest(directory / relative) for relative in names}})


def measured_cell(adapter, sample, scorer, cell, *, max_new_tokens=4096, fixed_output=False):
    import torch
    from learned_score.mlsys_eval.timing import TimingContract
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation import generate_matched
    if hasattr(sample.get("image"), "seek"):
        sample["image"].seek(0)
    device = adapter.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    observed = {}
    timer = TimingContract(synchronize=lambda: torch.cuda.synchronize(device))
    with timer.stage("preprocess"):
        prepared = adapter.prepare_sample(sample)
    with timer.stage("selector"):
        selected = benchmark.make_selector(cell, scorer, "rank32", observed)(prepared)
        pruned = adapter.physically_prune(prepared, selected, "original")
        observed["selected_visual_indices_sha256"] = digest(selected.detach().cpu().tolist())
        observed["retained_visual_tokens"] = len(selected)
        observed["retained_prefix_tokens"] = int(pruned.inputs_embeds.shape[1])
    preprocessing_peak = torch.cuda.max_memory_allocated(device)
    if cell.cache_method == "none":
        budget = {"kv_retained_slots": None, "kv_budget_slots": None, "generated_budget": None}
        result = generate_dynamic(adapter.model, pruned, max_new_tokens=max_new_tokens,
                                  stop_on_eos=not fixed_output, on_first_token=timer.mark_first_token)
        cache = result.metrics["cache"]
        cache.update({"peak_active_kv_bytes": cache["kv_active_bytes"],
                      "auxiliary_persistent_bytes": cache["source_metadata_bytes"],
                      "allocated_cache_and_auxiliary_bytes": cache["total_owned_bytes"],
                      "attention_statistics_scratch_peak_bytes_estimate": 0,
                      "rolling_rotary_scratch_peak_bytes_estimate": 0})
    else:
        # Small warmup generation does not represent a matched-budget result.
        budget = matched_budget(observed["input_prefix_tokens"], observed["retained_prefix_tokens"], cell.profile)
        result = generate_matched(adapter.model, pruned, max_new_tokens=max_new_tokens,
                                  generated_budget=budget["generated_budget"],
                                  total_retained_tokens=budget["kv_retained_slots"],
                                  method=RUNTIME_METHODS[cell.cache_method], slack=1,
                                  stop_on_eos=not fixed_output, on_first_token=timer.mark_first_token)
    runtime = result.metrics
    timer.stages["prefill"] = runtime["prefill_ttft_seconds"]
    timer.stages["compression"] = runtime["compaction_cpu_seconds"]
    timer.stages["decode"] = max(0., runtime["decode_seconds"] - runtime["compaction_cpu_seconds"])
    with timer.stage("postprocess"):
        ids = result.ids.detach().cpu()
        prediction = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
    timing = timer.finish()
    runtime["full_request_peak_allocated_bytes"] = max(preprocessing_peak, torch.cuda.max_memory_allocated(device))
    runtime["original_visual_tokens"] = observed["input_visual_tokens"]
    runtime["retained_visual_tokens"] = observed["retained_visual_tokens"]
    budget["kv_budget_bytes"] = None if cell.cache_method == "none" else runtime["cache"]["kv_capacity_bytes"]
    if cell.cache_method != "asymvlm":
        budget["generated_budget"] = None
    del result, pruned, selected, prepared
    return prediction, ids, timing, runtime, observed, budget


def run_smoke(args, protocol, base):
    import torch
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation import generate_matched
    adapter, dataset, _, scorer, entries = load_model(base, args.model, args.device, with_scorer=True)
    scorer.requires_grad_(False)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    checks = []
    for entry in entries:
        if entry["row_hash"] not in base["native_request_ids"]:
            continue
        sample = read_entry(dataset, entry)
        prepared = adapter.prepare_sample(sample)
        for retention in (1., .9):
            cell = CacheCell("asymvlm", retention, 512)
            chosen = benchmark.make_selector(cell, scorer, "rank32")(prepared)
            pruned = adapter.physically_prune(prepared, chosen, "original")
            full_prefix = int(prepared.inputs_embeds.shape[1])
            prefix = int(pruned.inputs_embeds.shape[1])
            reference = generate_dynamic(adapter.model, pruned, 64, stop_on_eos=False)
            expected = reference.ids.cpu()
            del reference
            for method in cache_methods(args.model):
                result = generate_matched(adapter.model, pruned, 64, 128,
                                          RUNTIME_METHODS[method], total_retained_tokens=full_prefix + 128,
                                          stop_on_eos=False)
                same = torch.equal(expected, result.ids.cpu())
                checks.append({"request_id": entry["row_hash"], "vision": retention, "method": method,
                               "check": "native_no_eviction_exact_ids_64", "passed": same})
                del result
                if not same:
                    raise RuntimeError(f"native no-eviction parity failed: {args.model}/{method}/{retention}")
                # Force eviction on BOTH visual-retention arms, without using quality labels.
                forced = generate_matched(adapter.model, pruned, 48, 8, RUNTIME_METHODS[method],
                                          total_retained_tokens=prefix + 8, stop_on_eos=False)
                stats = forced.metrics["cache"]
                finite = all(bool(torch.isfinite(tensor).all())
                             for tensor in forced.cache.key_cache + forced.cache.value_cache)
                prefix_intact = True
                if method == "asymvlm":
                    expected_sources = pruned.retained_source_positions
                    prefix_intact = all(torch.equal(forced.cache.layer_source_positions(layer)[0, head, :prefix], expected_sources)
                                        for layer in range(len(forced.cache))
                                        for head in range(forced.cache.key_cache[layer].shape[1]))
                passed = (stats["compaction_events"] > 0 and stats["capacity_tokens_per_layer"] == prefix + 9
                          and stats["peak_active_tokens"] <= prefix + 9
                          and stats["protects_entire_prefix"] == (method == "asymvlm")
                          and finite and prefix_intact)
                checks.append({"request_id": entry["row_hash"], "vision": retention, "method": method,
                               "check": "forced_eviction_capacity_and_policy_48", "passed": passed,
                               "finite_cache": finite, "protected_prefix_intact": prefix_intact,
                               "generation_seconds": forced.metrics["generation_seconds"]})
                del forced
                if not passed:
                    raise RuntimeError("forced cache eviction smoke failed")
                assert_frozen(adapter.model, versions)
                assert_frozen(scorer, scorer_versions)
            print(json.dumps({"event": "cache_smoke", "model": args.model,
                              "request_id": entry["row_hash"], "visual_retention": retention}), flush=True)
            del pruned, chosen
        del prepared
    # Exercise the exact production measurement wrapper before releasing any
    # quality tasks. These short outputs are never exported as quality evidence.
    for cell in cells_for(args.model):
        _, ids, timing, _, _, _ = measured_cell(adapter, sample, scorer, cell, max_new_tokens=8)
        if not 1 <= ids.shape[-1] <= 8 or timing["end_to_end_seconds"] <= 0 or timing["ttft_seconds"] <= 0:
            raise RuntimeError("production request measurement wrapper failed smoke")
        del ids
    assert_frozen(adapter.model, versions)
    assert_frozen(scorer, scorer_versions)
    result = {"format": "mlsys_matched_cache_smoke_v1", "protocol_sha256": protocol["sha256"],
              "model_name": args.model, "cache_methods": list(cache_methods(args.model)),
              "passed": True, "checks": checks, "execution_context": execution_context(),
              "native_gate": verify_native_gate(base, args.model), "final_pages_opened": False}
    write_immutable(protocol["smoke_gates"][args.model], result)


def run_page(args, protocol, base):
    import torch
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    smoke_gate(protocol, args.model)
    recover_completed_page(protocol, base, args.output, args.rank)
    if (args.output / f"rank{args.rank}.integrity.json").exists():
        panel = collect_panel(args.protocol, args.output, require_complete=False)
        if panel["metadata"]["model_name"] != args.model:
            raise ValueError("existing output panel belongs to another model")
        if any(key[0] == panel["entries"][args.rank]["row_hash"] for key in panel["rows"]):
            return
        raise ValueError("existing page seal cannot be verified")
    adapter, dataset, backbone, scorer, entries = load_model(base, args.model, args.device, with_scorer=True)
    scorer.requires_grad_(False)
    metadata = metadata_for(protocol, base, args.model, backbone)
    metadata_path = args.output / "metadata.json"
    write_immutable(metadata_path, metadata)
    identity = digest(metadata)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    entry, cells = entries[args.rank], cells_for(args.model)
    sample = read_entry(dataset, entry)
    for cell in cells:
        measured_cell(adapter, sample, scorer, cell, max_new_tokens=8)
    provenance, attempt = execution_context(), uuid.uuid4().hex
    hardware = torch.cuda.get_device_name(adapter.device)
    metric = protocol_manifest(base)[0].metadata["official_metric"]
    page, hashes = [], {}
    offset = int(entry["row_hash"][:8], 16) % len(cells)
    for cell in cells[offset:] + cells[:offset]:
        torch.cuda.empty_cache()
        prediction, ids, timing, runtime, observed, budget = measured_cell(adapter, sample, scorer, cell)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        record = {"format": "mlsys_request_result_v1", "identity_sha256": identity,
                  "model": backbone["model"], "dataset": "olmOCR-Bench", "split": "scorer_dev",
                  "stage": "development", "request_id": entry["row_hash"], "group_id": entry["group_id"],
                  "image_sha256": entry["image_sha256"], "method": cell.key,
                  "method_label": LABELS[cell.cache_method], "cache_method": cell.cache_method,
                  "selector": cell.selector, "visual_retention": cell.visual_retention,
                  "comparison_group": cell.group, "budget_profile": "unrestricted" if cell.profile is None else f"base{cell.profile}",
                  "quality_metric": metric, "max_new_tokens": 4096, "hardware": hardware, "batch_size": 1,
                  "serving_measurement": False, "repetition": 0, "repeat": 1, "attempt": attempt,
                  "execution_context": provenance, "timing_boundary": "image_preprocessing_to_detokenized_output",
                  "includes_output_detokenization": True, "quality_run_only": True,
                  "status": "ok", "prediction": prediction, "quality_score": None,
                  "quality_status": "awaiting_official_score", "generated_tokens": int(ids.shape[-1]),
                  "natural_stop_enabled": True, "metrics": runtime,
                  "peak_allocated_bytes": runtime["full_request_peak_allocated_bytes"],
                  **timing, **observed, **budget}
        relative = f"attempts/rank{args.rank}/{attempt}/{cell.key}.json"
        write_immutable(args.output / relative, record)
        hashes[relative] = file_digest(args.output / relative)
        page.append(record)
        print(json.dumps({"event": "cache_method_page", "model": args.model, "rank": args.rank,
                          "cell": cell.key, "generated_tokens": record["generated_tokens"],
                          "seconds": record["end_to_end_seconds"]}), flush=True)
        del ids
    validate_page(page, entry, metadata)
    complete = args.output / f"rank{args.rank}.complete.json"
    write_immutable(complete, {"identity_sha256": identity, "request_id": entry["row_hash"],
                              "cells": len(cells), "rank": args.rank, "attempt": attempt})
    write_immutable(args.output / f"rank{args.rank}.integrity.json",
                    {"format": "mlsys_matched_cache_page_integrity_v1", "rank": args.rank,
                     "protocol_sha256": protocol["sha256"], "identity_sha256": identity,
                     "metadata_sha256": file_digest(metadata_path), "completion_sha256": file_digest(complete),
                     "attempt": attempt, "row_sha256": hashes})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen25", "llava_next"), required=True)
    parser.add_argument("--rank", type=int, choices=range(100), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    protocol, base = load_protocol(args.protocol)
    verify_native_gate(base, args.model)
    require_cuda(args.device)
    (run_smoke if args.smoke else run_page)(args, protocol, base)


if __name__ == "__main__":
    main()
