"""Versioned, same-physical-KV policy comparisons; frozen v1/v2 are untouched.

The publication unit is one complete page, generated in one process. A failed
attempt is never spliced with another attempt. Official scoring may only consume
all 100 sealed development pages. Quality-run timings are not isolated timings.
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


FORMAT = "mlsys_matched_cache_protocol_v3"
RUNTIME_REVISION = "matched_policy_quality_systems_v3"
DRIVER = "learned_score.mlsys_sprint.cache_methods_v3"
PRIMARY_CELL_COUNT = 10
POLICIES = ("protected_recency", "h2o", "h2o_recent_adjusted", "protected_h2o",
            "rkv", "protected_rkv", "protected_random")
PROTECTED = frozenset({"protected_recency", "protected_h2o", "protected_rkv", "protected_random"})
RANDOM_SEEDS = (42, 43, 44)
LABELS = {"fullref": "All tokens, no cache eviction",
          "protected_recency": "AsymVLM: protected image/prompt + recent generated text",
          "h2o": "H2O: original half-heavy/half-recent allocation",
          "h2o_recent_adjusted": "H2O: 256 recent tokens",
          "protected_h2o": "H2O + protected image/prompt, 256 recent tokens (ablation)",
          "rkv": "R-KV: attention and key redundancy",
          "protected_rkv": "R-KV + protected image/prompt (ablation)",
          "protected_random": "Protected image/prompt + random generated tokens (ablation)"}


def staging_for(policy):
    if policy not in POLICIES:
        raise ValueError("unknown cache policy")
    return 128 if policy in ("rkv", "protected_rkv", "protected_random") else 1


@dataclass(frozen=True)
class CacheCell:
    cache_policy: str
    visual_retention: float = 1.
    profile: int | None = 512
    random_seed: int | None = None
    staging: int | None = None

    @property
    def selector(self):
        return "full" if self.visual_retention == 1. else "scorer"

    @property
    def staging_slots(self):
        return 0 if self.cache_policy == "fullref" else self.staging or staging_for(self.cache_policy)

    @property
    def key(self):
        if self.cache_policy == "fullref":
            return "full_v100_off"
        suffix = "" if self.random_seed is None else f"_seed{self.random_seed}"
        return f"{self.cache_policy}_v{round(self.visual_retention * 100)}_b{self.profile}_s{self.staging_slots}{suffix}"

    @property
    def group(self):
        return ("unrestricted" if self.profile is None else
                f"v{round(self.visual_retention * 100)}_b{self.profile}")


def cells_for(name, protocol=None):
    if name not in ("qwen25", "llava_next"):
        raise ValueError("unsupported model")
    stage = (protocol or {}).get("stage", "primary")
    if stage != "primary":
        raise ValueError("secondary/final cells require a separately approved versioned protocol")
    cells = [CacheCell("fullref", profile=None)]
    for policy in POLICIES:
        seeds = RANDOM_SEEDS if policy == "protected_random" else (None,)
        cells.extend(CacheCell(policy, random_seed=seed) for seed in seeds)
    return tuple(cells)


def matched_budget(full_prefix, retained_prefix, profile, staging=1, max_new_tokens=4096):
    if (type(full_prefix) is not int or type(retained_prefix) is not int
            or not 1 <= retained_prefix <= full_prefix or profile not in (256, 512, 1024)
            or staging not in (1, 128)):
        raise ValueError("invalid prefix, budget profile, or staging allowance")
    capacity = full_prefix + profile + 1
    retained = capacity - staging
    if retained < retained_prefix:
        raise ValueError("protected prefix cannot fit inside retained capacity")
    # Short smoke runs may not reach capacity; production callers separately
    # enforce reachability. Never clip the physical allocation to output length.
    return {"kv_budget_slots": capacity, "kv_retained_slots": retained,
            "staging_slots": staging, "generated_budget": retained - retained_prefix}


def load_protocol(path):
    protocol = json.loads(Path(path).read_text())
    if protocol.get("sha256") != digest({k: v for k, v in protocol.items() if k != "sha256"}):
        raise ValueError("v3 protocol digest mismatch")
    required = {"format": FORMAT, "split": "scorer_dev", "final_pages_opened": False,
                "max_new_tokens": 4096, "driver_module": DRIVER,
                "runtime_revision": RUNTIME_REVISION, "stage": "primary",
                "budget_profiles": [512], "visual_retentions": [1.0],
                "random_seeds": list(RANDOM_SEEDS)}
    if any(protocol.get(k) != v for k, v in required.items()):
        raise ValueError("unsupported v3 protocol; final/secondary execution is not authorized here")
    if set(protocol.get("active_models", [])) != {"qwen25", "llava_next"} or len(protocol["active_models"]) != 2:
        raise ValueError("v3 primary must declare both models")
    for field, hash_field in (("base_protocol", "base_sha256"),
                              ("parent_cache_protocol", "parent_cache_sha256")):
        if file_digest(protocol[field]) != protocol[hash_field]:
            raise ValueError(f"frozen {field} changed")
    for filename, expected in protocol["input_hashes"].items():
        if file_digest(filename) != expected:
            raise ValueError(f"v3 input changed: {filename}")
    base = load_followup_protocol(protocol["base_protocol"])
    if protocol["models"] != base["models"]:
        raise ValueError("v3 models differ from the frozen base")
    return protocol, base


SMOKE_CHECKS = ("native_no_eviction_exact_ids_64", "forced_reference_optimized_exact_ids_512",
                "forced_capacity_and_policy_512")


def expected_smoke_checks(base):
    return {(rid, policy, check) for rid in base["native_request_ids"]
            for policy in POLICIES for check in SMOKE_CHECKS}


def smoke_gate(protocol, name):
    path = Path(protocol["smoke_gates"][name])
    record = json.loads(path.read_text())
    base = json.loads(Path(protocol["base_protocol"]).read_text())
    required = {"format": "mlsys_matched_cache_smoke_v3", "passed": True,
                "model_name": name, "protocol_sha256": protocol["sha256"],
                "runtime_revision": RUNTIME_REVISION, "final_pages_opened": False}
    if any(record.get(k) != v for k, v in required.items()):
        raise ValueError("v3 native/reference/optimized smoke gate absent or mismatched")
    checks = record.get("checks", [])
    observed = {(r.get("request_id"), r.get("policy"), r.get("check")) for r in checks}
    if (observed != expected_smoke_checks(base) or len(checks) != len(observed)
            or any(r.get("passed") is not True for r in checks)):
        raise ValueError("v3 smoke gate is incomplete or failed")
    for filename, sha in record.get("artifacts", {}).items():
        if file_digest(filename) != sha:
            raise ValueError("v3 smoke artifact changed")
    return {"path": str(path), "sha256": file_digest(path), "passed": True}


def metadata_for(protocol, base, name, backbone):
    manifest, _ = protocol_manifest(base)
    return {"format": "mlsys_matched_cache_panel_v3", "protocol_sha256": protocol["sha256"],
            "runtime_revision": RUNTIME_REVISION, "model_name": name, "backbone": backbone,
            "manifest_sha256": manifest.file_sha256, "dataset_kind": "olmocr", "split": "scorer_dev",
            "stage": "development", "checkpoint_sha256": model_spec(base, name)["checkpoint_sha256"],
            "scorer_training_manifest_sha256": file_digest(base["scorer_training_manifest"]),
            "cache_policies": list(POLICIES), "cells": [c.key for c in cells_for(name, protocol)],
            "native_gate": verify_native_gate(base, name), "smoke_gate": smoke_gate(protocol, name),
            "max_new_tokens": 4096, "repetitions": 1, "batch_size": 1,
            "random_seeds": list(RANDOM_SEEDS), "fixed_output_tokens": False,
            "all_results_development": True, "final_pages_opened": False,
            "budget_definition": "physical_M=original_full_prefix+B+1; retained=M-staging",
            "pairing_unit": "same_page_same_model_process", "quality_run_only": True,
            "timing_isolated": False, "asymvlm_text_policy": "protected_prefix_recent_only",
            "protected_competitors_are_ablations": True,
            "rkv_reference_order": "HuggingFace_topk_score_order_not_chronological"}


def validate_page(rows, entry, metadata):
    cells = {c.key: c for c in cells_for(metadata["model_name"])}
    if len(rows) != len(cells) or {r.get("method") for r in rows} != set(cells):
        raise ValueError("page must contain exactly all ten predeclared cells")
    contexts, attempts, prefixes, visual_sets, allocations = set(), set(), set(), set(), set()
    for row in rows:
        cell = cells[row["method"]]
        required = {"format": "mlsys_request_result_v1", "identity_sha256": digest(metadata),
                    "model": metadata["backbone"]["model"], "dataset": "olmOCR-Bench", "split": "scorer_dev",
                    "stage": "development", "request_id": entry["row_hash"], "group_id": entry["group_id"],
                    "image_sha256": entry["image_sha256"], "status": "ok", "cache_policy": cell.cache_policy,
                    "random_seed": cell.random_seed, "visual_retention": cell.visual_retention,
                    "selector": cell.selector, "max_new_tokens": 4096, "natural_stop_enabled": True,
                    "repetition": 0, "quality_score": None, "quality_status": "awaiting_official_score",
                    "comparison_group": cell.group, "budget_profile": "unrestricted" if cell.profile is None else f"base{cell.profile}",
                    "quality_run_only": True, "timing_isolated": False}
        if any(row.get(k) != v for k, v in required.items()):
            raise ValueError("row violates frozen identity, policy or configuration")
        provenance = row.get("execution_context", {})
        if not provenance.get("hostname") or provenance.get("cluster") not in ("clusterb", "clustera"):
            raise ValueError("page lacks assigned-cluster provenance")
        contexts.add(digest(provenance))
        attempts.add(row.get("attempt"))
        ids = row.get("generated_ids")
        if (not isinstance(ids, list) or not 1 <= len(ids) <= 4096
                or any(type(token) is not int or token < 0 for token in ids)
                or row.get("generated_tokens") != len(ids)
                or row.get("generated_ids_sha256") != digest(ids)
                or not isinstance(row.get("prediction"), str)):
            raise ValueError("generated IDs are absent, changed, or inconsistent")
        full, prefix, original, retained = (row[k] for k in (
            "input_prefix_tokens", "retained_prefix_tokens", "input_visual_tokens", "retained_visual_tokens"))
        if (any(type(v) is not int or v < 1 for v in (full, prefix, original, retained))
                or retained != int(original * cell.visual_retention)
                or full - prefix != original - retained):
            raise ValueError("invalid visual retention/prefix counts")
        prefixes.add(full)
        visual_sets.add((row["selected_visual_indices_sha256"], retained, prefix))
        metrics, cache = row["metrics"], row["metrics"]["cache"]
        if (metrics.get("matched_runtime_revision") != RUNTIME_REVISION
                or metrics.get("cache_policy") != cell.cache_policy
                or metrics.get("timing_isolated") is not False):
            raise ValueError("runtime provenance or timing isolation was misreported")
        if cell.cache_policy == "fullref":
            if any(row.get(k) is not None for k in ("kv_budget_slots", "kv_retained_slots", "kv_budget_bytes", "generated_budget")):
                raise ValueError("unrestricted reference mislabeled as bounded")
            if cache.get("compaction_events") != 0:
                raise ValueError("unrestricted reference performed eviction")
            continue
        budget = matched_budget(full, prefix, cell.profile, cell.staging_slots)
        protected = cell.cache_policy in PROTECTED
        if (row["kv_budget_slots"] != budget["kv_budget_slots"]
                or row["kv_retained_slots"] != budget["kv_retained_slots"]
                or row.get("staging_slots") != cell.staging_slots
                or cache.get("capacity_tokens_per_layer") != budget["kv_budget_slots"]
                or row["kv_budget_bytes"] != cache.get("kv_capacity_bytes")
                or cache.get("active_tokens", -1) < 1
                or cache["active_tokens"] > budget["kv_budget_slots"]
                or cache.get("peak_active_tokens", -1) < cache["active_tokens"]
                or cache["peak_active_tokens"] > budget["kv_budget_slots"]
                or cache.get("protects_entire_prefix") != protected
                or row.get("generated_budget") != (budget["generated_budget"] if protected else None)):
            raise ValueError("physical capacity, protection or staging disagrees with declared budget")
        allocations.add((row["kv_budget_slots"], row["kv_budget_bytes"]))
    if (len(contexts) != 1 or len(attempts) != 1 or not valid_attempt(next(iter(attempts)))
            or len(prefixes) != 1 or len(visual_sets) != 1 or len(allocations) != 1):
        raise ValueError("page crossed processes/attempts or visual selections/physical capacities differ")


def valid_attempt(value):
    return isinstance(value, str) and len(value) == 32 and all(c in "0123456789abcdef" for c in value)


def completion_record(metadata, entry, rank, attempt):
    return {"identity_sha256": digest(metadata), "request_id": entry["row_hash"],
            "cells": PRIMARY_CELL_COUNT, "rank": rank, "attempt": attempt}


def row_paths(rank, attempt, name):
    if not valid_attempt(attempt):
        raise ValueError("invalid attempt identifier")
    return [f"attempts/rank{rank}/{attempt}/{c.key}.json" for c in cells_for(name)]


def recover_completed_page(protocol, base, directory, rank):
    directory = Path(directory)
    completion, seal_path = (directory / f"rank{rank}.{suffix}.json" for suffix in ("complete", "integrity"))
    if not completion.exists() or seal_path.exists():
        return
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata != metadata_for(protocol, base, metadata["model_name"], metadata["backbone"]):
        raise ValueError("unsealed metadata differs from frozen protocol")
    entry = protocol_manifest(base)[1][rank]
    marker = json.loads(completion.read_text())
    attempt = marker.get("attempt")
    if marker != completion_record(metadata, entry, rank, attempt) or not valid_attempt(attempt):
        raise ValueError("invalid unsealed completion")
    names = row_paths(rank, attempt, metadata["model_name"])
    rows = [json.loads((directory / name).read_text()) for name in names]
    validate_page(rows, entry, metadata)
    if any(row["attempt"] != attempt for row in rows):
        raise ValueError("completion and rows belong to different attempts")
    write_immutable(seal_path, {"format": "mlsys_matched_cache_page_integrity_v3", "rank": rank,
                               "protocol_sha256": protocol["sha256"], "identity_sha256": digest(metadata),
                               "metadata_sha256": file_digest(metadata_path),
                               "completion_sha256": file_digest(completion), "attempt": attempt,
                               "row_sha256": {name: file_digest(directory / name) for name in names}})


def collect_panel(protocol_path, directory, require_complete=True):
    protocol, base = load_protocol(protocol_path)
    _, entries = protocol_manifest(base)
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
        seal_path, marker_path = (directory / f"rank{rank}.{suffix}.json" for suffix in ("integrity", "complete"))
        if not seal_path.exists():
            continue
        seal = json.loads(seal_path.read_text())
        attempt = seal.get("attempt")
        if (seal.get("format") != "mlsys_matched_cache_page_integrity_v3" or seal.get("rank") != rank
                or seal.get("protocol_sha256") != protocol["sha256"]
                or seal.get("identity_sha256") != digest(metadata)
                or seal.get("metadata_sha256") != file_digest(metadata_path)
                or seal.get("completion_sha256") != file_digest(marker_path)):
            raise ValueError("page integrity seal is invalid")
        if json.loads(marker_path.read_text()) != completion_record(metadata, entry, rank, attempt):
            raise ValueError("completion differs from frozen page assignment")
        expected = set(row_paths(rank, attempt, name))
        if set(seal.get("row_sha256", {})) != expected:
            raise ValueError("seal does not bind exactly one complete attempt")
        page = []
        for relative, sha in seal["row_sha256"].items():
            path = directory / relative
            if file_digest(path) != sha:
                raise ValueError("sealed row changed")
            page.append(json.loads(path.read_text()))
        validate_page(page, entry, metadata)
        if any(row["attempt"] != attempt for row in page):
            raise ValueError("row attempt does not match sealed attempt")
        rows.update({(r["request_id"], r["method"], 0): {**r, "run_context": context} for r in page})
        artifacts[str(seal_path)] = file_digest(seal_path)
        artifacts[str(marker_path)] = file_digest(marker_path)
        artifacts.update({str(directory / rel): sha for rel, sha in seal["row_sha256"].items()})
    complete = len(rows) == 100 * PRIMARY_CELL_COUNT
    if require_complete and not complete:
        raise ValueError("official scoring requires all 100 sealed development pages")
    return {"metadata": metadata, "entries": entries, "rows": rows, "complete": complete,
            "context": context, "integrity_artifacts": artifacts, "directory": str(directory),
            "label": f"{name}: matched physical KV methods, development",
            "checkpoint": {"variant": "rank32", "seed": 42, "sha256": metadata["checkpoint_sha256"],
                           "identity": checkpoint["identity"]}}


def measured_cell(adapter, sample, scorer, cell, *, max_new_tokens=4096, fixed_output=False,
                  optimized=True, record_events=False, trace=False, teacher_ids=None):
    import torch
    from learned_score.mlsys_eval.timing import TimingContract
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
    if hasattr(sample.get("image"), "seek"):
        sample["image"].seek(0)
    device, observed = adapter.device, {}
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    timer = TimingContract(synchronize=lambda: torch.cuda.synchronize(device))
    with timer.stage("preprocess"):
        prepared = adapter.prepare_sample(sample)
    with timer.stage("selector"):
        selected = benchmark.make_selector(cell, scorer, "rank32", observed)(prepared)
        pruned = adapter.physically_prune(prepared, selected, "original")
        observed.update({"selected_visual_indices_sha256": digest(selected.detach().cpu().tolist()),
                         "retained_visual_tokens": int(selected.numel()),
                         "retained_prefix_tokens": int(pruned.inputs_embeds.shape[1])})
    preprocessing_peak = torch.cuda.max_memory_allocated(device)
    if cell.cache_policy == "fullref" and not trace and teacher_ids is None:
        result = generate_dynamic(adapter.model, pruned, max_new_tokens=max_new_tokens,
                                  stop_on_eos=not fixed_output, on_first_token=timer.mark_first_token)
        budget = {"kv_budget_slots": None, "kv_retained_slots": None, "kv_budget_bytes": None,
                  "staging_slots": 0, "generated_budget": None}
    else:
        budget = matched_budget(observed["input_prefix_tokens"], observed["retained_prefix_tokens"],
                                cell.profile, cell.staging_slots) if cell.profile else {
                                    "kv_budget_slots": None, "kv_retained_slots": None,
                                    "kv_budget_bytes": None, "staging_slots": 0, "generated_budget": None}
        result = generate_matched_v3(adapter.model, pruned, max_new_tokens=max_new_tokens,
                                     full_prefix_tokens=observed["input_prefix_tokens"],
                                     budget_profile=cell.profile or max_new_tokens,
                                     policy="protected_recency" if cell.cache_policy == "fullref" else cell.cache_policy,
                                     staging=cell.staging_slots or None, seed=cell.random_seed or 42,
                                     optimized=optimized, record_events=record_events, trace=trace,
                                     teacher_ids=teacher_ids, stop_on_eos=not fixed_output,
                                     visual_source_positions=prepared.visual_positions,
                                     on_first_token=timer.mark_first_token)
    runtime = result.metrics
    if cell.cache_policy != "fullref" and runtime.get("matched_runtime_revision") != RUNTIME_REVISION:
        raise ValueError("generation used an unexpected runtime revision")
    runtime.update({"matched_runtime_revision": RUNTIME_REVISION, "cache_policy": cell.cache_policy,
                    "timing_isolated": False, "original_visual_tokens": observed["input_visual_tokens"],
                    "retained_visual_tokens": observed["retained_visual_tokens"]})
    timer.stages["prefill"] = runtime["prefill_ttft_seconds"]
    compression = runtime.get("compaction_cpu_seconds") or 0.
    timer.stages["compression"] = compression
    timer.stages["decode"] = max(0., runtime["decode_seconds"] - compression)
    with timer.stage("postprocess"):
        ids = result.ids.detach().cpu()
        prediction = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
    timing = timer.finish()
    timing["timing_mode"] = "quality_shared_node_synchronized_not_isolated"
    runtime["full_request_peak_allocated_bytes"] = max(preprocessing_peak, torch.cuda.max_memory_allocated(device))
    runtime["full_request_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    if cell.cache_policy != "fullref":
        budget["kv_budget_bytes"] = runtime["cache"]["kv_capacity_bytes"]
    if cell.cache_policy not in PROTECTED:
        budget["generated_budget"] = None
    del result, pruned, selected, prepared
    return prediction, ids, timing, runtime, observed, budget


def smoke_page(adapter, dataset, scorer, entry):
    import torch
    from learned_score.mlsys_scorer.backbone import read_entry
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
    prepared = adapter.prepare_sample(read_entry(dataset, entry))
    chosen = benchmark.make_selector(CacheCell("protected_recency"), scorer, "rank32")(prepared)
    pruned = adapter.physically_prune(prepared, chosen, "original")
    prefix = int(prepared.inputs_embeds.shape[1])
    native = generate_dynamic(adapter.model, pruned, 64, stop_on_eos=False)
    expected = native.ids.cpu()
    del native
    checks = []
    for policy in POLICIES:
        no_evict_ids, forced_ids = [], []
        healthy = True
        for optimized in (False, True):
            result = generate_matched_v3(adapter.model, pruned, 64, full_prefix_tokens=prefix,
                                         budget_profile=512, policy=policy, seed=42,
                                         optimized=optimized, stop_on_eos=False)
            no_evict_ids.append(result.ids.cpu())
            healthy &= result.metrics["cache"]["compaction_events"] == 0
            del result
            forced = generate_matched_v3(adapter.model, pruned, 512, full_prefix_tokens=prefix,
                                         budget_profile=256, policy=policy, seed=42,
                                         optimized=optimized, stop_on_eos=False)
            forced_ids.append(forced.ids.cpu())
            stats = forced.metrics["cache"]
            healthy &= (stats["compaction_events"] > 0 and stats["capacity_tokens_per_layer"] == prefix + 257
                        and stats["peak_active_tokens"] <= prefix + 257
                        and stats["protects_entire_prefix"] == (policy in PROTECTED))
            healthy &= all(bool(torch.isfinite(t).all()) for t in forced.cache.key_cache + forced.cache.value_cache)
            if policy in PROTECTED:
                wanted = set(pruned.retained_source_positions.cpu().tolist())
                healthy &= all(wanted.issubset(set(head.tolist()))
                               for layer in range(len(forced.cache))
                               for head in forced.cache.layer_source_positions(layer)[0].cpu())
            del forced
        values = (all(torch.equal(expected, ids) for ids in no_evict_ids),
                  torch.equal(forced_ids[0], forced_ids[1]), bool(healthy))
        for check, passed in zip(SMOKE_CHECKS, values):
            checks.append({"request_id": entry["row_hash"], "policy": policy, "check": check, "passed": passed})
        if not all(values):
            raise RuntimeError(f"v3 physical parity/capacity smoke failed: {policy}/{entry['row_hash']}")
    return checks


def seal_smoke(protocol, base, name):
    path = Path(protocol["smoke_gates"][name])
    checks, artifacts = [], {}
    for rank, rid in enumerate(base["native_request_ids"]):
        shard = path.parent / f"{name}.rank{rank}.complete.json"
        record = json.loads(shard.read_text())
        if (record.get("protocol_sha256") != protocol["sha256"] or record.get("model_name") != name
                or record.get("request_id") != rid or record.get("runtime_revision") != RUNTIME_REVISION
                or record.get("passed") is not True):
            raise ValueError("smoke shard identity mismatch")
        checks.extend(record["checks"])
        artifacts[str(shard)] = file_digest(shard)
    write_immutable(path, {"format": "mlsys_matched_cache_smoke_v3", "protocol_sha256": protocol["sha256"],
                           "runtime_revision": RUNTIME_REVISION, "model_name": name,
                           "passed": True, "checks": checks, "artifacts": artifacts,
                           "native_gate": verify_native_gate(base, name), "final_pages_opened": False})
    smoke_gate(protocol, name)


def run_smoke(args, protocol, base):
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions
    adapter, dataset, _, scorer, entries = load_model(base, args.model, args.device, with_scorer=True)
    scorer.requires_grad_(False)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    by_id = {r["row_hash"]: r for r in entries}
    ranks = range(8) if args.smoke_rank is None else (args.smoke_rank,)
    for rank in ranks:
        rid = base["native_request_ids"][rank]
        shard = Path(protocol["smoke_gates"][args.model]).parent / f"{args.model}.rank{rank}.complete.json"
        if shard.exists():
            existing = json.loads(shard.read_text())
            if existing.get("protocol_sha256") != protocol["sha256"] or existing.get("passed") is not True:
                raise ValueError("existing smoke shard changed")
            continue
        checks = smoke_page(adapter, dataset, scorer, by_id[rid])
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        write_immutable(shard, {"protocol_sha256": protocol["sha256"], "model_name": args.model,
                                "request_id": rid, "runtime_revision": RUNTIME_REVISION,
                                "passed": True, "checks": checks, "execution_context": execution_context()})
        print(json.dumps({"event": "v3_smoke_page", "model": args.model, "rank": rank}), flush=True)
    if args.smoke_rank is None:
        seal_smoke(protocol, base, args.model)


def run_page(args, protocol, base):
    import torch
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    smoke_gate(protocol, args.model)
    recover_completed_page(protocol, base, args.output, args.rank)
    if (args.output / f"rank{args.rank}.integrity.json").exists():
        panel = collect_panel(args.protocol, args.output, require_complete=False)
        if panel["metadata"]["model_name"] != args.model:
            raise ValueError("output belongs to another model")
        return
    adapter, dataset, backbone, scorer, entries = load_model(base, args.model, args.device, with_scorer=True)
    scorer.requires_grad_(False)
    metadata = metadata_for(protocol, base, args.model, backbone)
    write_immutable(args.output / "metadata.json", metadata)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    entry, cells = entries[args.rank], cells_for(args.model, protocol)
    sample = read_entry(dataset, entry)
    # Warmup measurements are discarded; they are never appended to quality rows.
    for cell in cells:
        measured_cell(adapter, sample, scorer, cell, max_new_tokens=8)
    provenance, attempt = execution_context(), uuid.uuid4().hex
    hardware = torch.cuda.get_device_name(adapter.device)
    metric = protocol_manifest(base)[0].metadata["official_metric"]
    page = []
    offset = int(entry["row_hash"][:8], 16) % len(cells)
    for cell in cells[offset:] + cells[:offset]:
        torch.cuda.empty_cache()
        prediction, ids, timing, runtime, observed, budget = measured_cell(adapter, sample, scorer, cell)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        token_ids = ids[0].tolist()
        record = {"format": "mlsys_request_result_v1", "identity_sha256": digest(metadata),
                  "model": backbone["model"], "dataset": "olmOCR-Bench", "split": "scorer_dev",
                  "stage": "development", "request_id": entry["row_hash"], "group_id": entry["group_id"],
                  "image_sha256": entry["image_sha256"], "method": cell.key,
                  "method_label": LABELS[cell.cache_policy], "cache_policy": cell.cache_policy,
                  "cache_method": cell.cache_policy, "random_seed": cell.random_seed,
                  "selector": cell.selector, "visual_retention": cell.visual_retention,
                  "comparison_group": cell.group, "budget_profile": "unrestricted" if cell.profile is None else f"base{cell.profile}",
                  "quality_metric": metric, "max_new_tokens": 4096, "hardware": hardware,
                  "batch_size": 1, "serving_measurement": False, "repetition": 0, "repeat": 1,
                  "attempt": attempt, "execution_context": provenance,
                  "timing_boundary": "image_preprocessing_to_detokenized_output",
                  "includes_output_detokenization": True, "quality_run_only": True, "timing_isolated": False,
                  "status": "ok", "prediction": prediction, "quality_score": None,
                  "quality_status": "awaiting_official_score", "generated_tokens": len(token_ids),
                  "generated_ids": token_ids, "generated_ids_sha256": digest(token_ids),
                  "natural_stop_enabled": True, "metrics": runtime,
                  "peak_allocated_bytes": runtime["full_request_peak_allocated_bytes"],
                  **timing, **observed, **budget}
        write_immutable(args.output / f"attempts/rank{args.rank}/{attempt}/{cell.key}.json", record)
        page.append(record)
        print(json.dumps({"event": "cache_v3_page", "model": args.model, "rank": args.rank,
                          "cell": cell.key, "generated_tokens": len(token_ids),
                          "seconds": record["end_to_end_seconds"]}), flush=True)
    validate_page(page, entry, metadata)
    write_immutable(args.output / f"rank{args.rank}.complete.json", completion_record(metadata, entry, args.rank, attempt))
    recover_completed_page(protocol, base, args.output, args.rank)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("qwen25", "llava_next"), required=True)
    parser.add_argument("--rank", type=int, choices=range(100), default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-rank", type=int, choices=range(8))
    parser.add_argument("--seal-smoke", action="store_true")
    args = parser.parse_args(argv)
    protocol, base = load_protocol(args.protocol)
    verify_native_gate(base, args.model)
    if args.seal_smoke:
        seal_smoke(protocol, base, args.model)
        return
    require_cuda(args.device)
    (run_smoke if args.smoke or args.smoke_rank is not None else run_page)(args, protocol, base)


if __name__ == "__main__":
    main()
