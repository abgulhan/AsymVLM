"""Development-only, four-hour vision/text pilot at exact physical KV capacity.

Vision counts come from the already frozen v12 rule, never from olmOCR labels.
This driver reuses the audited v3 cache/runtime and the v4 metadata-only page
selection. Neither historical source nor historical result files are changed.
"""

import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
from statistics import mean
import time
import uuid

from .cache_methods_v3 import RUNTIME_REVISION, matched_budget
from .cache_protocol_v4 import _read_sealed, seal, select_development_entries
from .followup_common import (load_followup_protocol, load_model, model_spec,
                              require_cuda, verify_native_gate, write_immutable)
from .state import digest, file_digest


FORMAT = "mlsys_compression_text_v12"
ASSIGNMENTS_FORMAT = "mlsys_compression_text_assignments_v12"
MODELS = ("qwen25", "llava_next")
PROFILES = (512, 256)
METHODS = ("allvision_recency", "static_recency", "adaptive_recency", "adaptive_rkv")
MAX_SECONDS = 4 * 60 * 60
MAX_NEW_TOKENS = 4096


def budget_for(full_prefix, retained_prefix, profile, policy):
    """The staging allowance is inside M, never additional free KV memory."""
    if policy not in ("protected_recency", "rkv"):
        raise ValueError("pilot permits protected recency and the existing R-KV adaptation")
    value = matched_budget(full_prefix, retained_prefix, profile,
                           staging=1 if policy == "protected_recency" else 128)
    return {**value, "protects_entire_prefix": policy == "protected_recency",
            "protected_prefix_tokens": retained_prefix if policy == "protected_recency" else 0,
            "available_generated_history_slots": (value["generated_budget"]
                                                   if policy == "protected_recency" else None),
            "retained_slots_beyond_initial_prefix": value["generated_budget"]}


def uniform_exact_counts(rows, total):
    """Hamilton rounding of one shared retention, with stable identity ties."""
    rows = list(rows)
    sizes = [row["input_visual_tokens"] for row in rows]
    if (not sizes or any(type(n) is not int or n < 1 for n in sizes)
            or type(total) is not int or not len(rows) <= total <= sum(sizes)):
        raise ValueError("infeasible exact aggregate visual count")
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate assignment identity")
    # Find a common retention respecting at least one token per request.
    low, high = 0., 1.
    for _ in range(80):
        ratio = (low + high) / 2
        if sum(max(1., ratio * n) for n in sizes) > total:
            high = ratio
        else:
            low = ratio
    raw = [max(1., low * n) for n in sizes]
    counts = [min(n, math.floor(k)) for n, k in zip(sizes, raw)]
    order = sorted(range(len(rows)), key=lambda i: (-(raw[i] - counts[i]), rows[i]["request_id"]))
    remaining = total - sum(counts)
    for i in order:
        if remaining and counts[i] < sizes[i]:
            counts[i] += 1
            remaining -= 1
    if remaining or sum(counts) != total:
        raise ValueError("uniform exact-budget reconciliation failed")
    return {row["request_id"]: count for row, count in zip(rows, counts)}


def build_config(parent_v4_protocol, frozen_policy, output_root, *, started_epoch=None,
                 deadline_epoch=None):
    parent_path, policy_path = Path(parent_v4_protocol).resolve(), Path(frozen_policy).resolve()
    parent, frozen = _read_sealed(parent_path), _read_sealed(policy_path)
    if (parent.get("format") != "mlsys_cache_history_protocol_v4"
            or parent.get("split") != "scorer_dev" or parent.get("final_pages_opened") is not False):
        raise ValueError("use the existing v4 development-only protocol")
    if frozen.get("format") != "mlsys_compression_selection_v12" or set(frozen.get("models", {})) != set(MODELS):
        raise ValueError("both v12 vision rules must be frozen before the text pilot")
    manifest_path = Path(parent["manifest"])
    manifest = json.loads(manifest_path.read_text())
    entries = select_development_entries(manifest["splits"]["scorer_dev"])
    if entries != parent["development_entries"]:
        raise ValueError("v4 deterministic development identities changed")
    final_groups = {e["group_id"] for e in manifest["splits"].get("final_test", [])}
    if final_groups & {e["group_id"] for e in entries}:
        raise ValueError("development sources overlap reserved final sources")
    started = time.time() if started_epoch is None else float(started_epoch)
    deadline = started + MAX_SECONDS if deadline_epoch is None else float(deadline_epoch)
    if not 0 < deadline - started <= MAX_SECONDS:
        raise ValueError("the text pilot has a hard maximum four-hour window")
    base_path = Path(parent["base_protocol"])
    base = load_followup_protocol(base_path)
    for name in MODELS:
        verify_native_gate(base, name)
    from learned_score.mlsys_runtime import matched_generation_v3, matched_cache_v3, dynamic
    from . import compression_features_v12, compression_policy_v12
    sources = [Path(__file__), Path(matched_generation_v3.__file__),
               Path(matched_cache_v3.__file__), Path(dynamic.__file__),
               Path(compression_features_v12.__file__), Path(compression_policy_v12.__file__)]
    pins = {**parent["input_hashes"], **base["input_hashes"],
            **{str(path.resolve()): file_digest(path) for path in
               [parent_path, policy_path, manifest_path, base_path, *sources]}}
    value = {"format": FORMAT, "parent_v4_protocol": str(parent_path),
             "frozen_policy": str(policy_path), "frozen_policy_sha256": frozen["sha256"],
             "base_protocol": str(base_path), "manifest": str(manifest_path),
             "entries": entries, "active_models": list(MODELS), "profiles": list(PROFILES),
             "started_epoch": started, "deadline_epoch": deadline,
             "output_root": str(Path(output_root).resolve()), "input_hashes": pins,
             "split": "scorer_dev", "final_pages_opened": False,
             "quality_stage": "previously_examined_development_pilot",
             "max_new_tokens": MAX_NEW_TOKENS, "natural_stop_enabled": True,
             "evaluator_root": parent["evaluator_root"], "cache_patch": parent["cache_patch"],
             "budget_definition": "M=original_full_prefix+B+1; staging inside M",
             "runtime_revision": RUNTIME_REVISION, "four_hour_cap_includes_assignment_preparation": True}
    return seal(value)


def load_config(path):
    value = _read_sealed(path)
    required = {"format": FORMAT, "active_models": list(MODELS), "profiles": list(PROFILES),
                "split": "scorer_dev", "final_pages_opened": False,
                "max_new_tokens": MAX_NEW_TOKENS, "natural_stop_enabled": True,
                "runtime_revision": RUNTIME_REVISION}
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError("invalid development-only text pilot configuration")
    if not 0 < value["deadline_epoch"] - value["started_epoch"] <= MAX_SECONDS:
        raise ValueError("text pilot exceeds four hours")
    if len(value["entries"]) != 24 or len({e["group_id"] for e in value["entries"]}) != 24:
        raise ValueError("pilot requires 24 distinct development source documents")
    for filename, sha in value["input_hashes"].items():
        if file_digest(filename) != sha:
            raise ValueError(f"frozen text pilot input changed: {filename}")
    return value


def build_tasks(config, config_path):
    """Complete B512 before B256; prioritize Qwen within each capacity."""
    return [{"model": model, "profile": profile, "rank": rank,
             "command": ["-m", "learned_score.mlsys_sprint.compression_text_v12", "worker",
                         "--config", str(config_path), "--model", model,
                         "--profile", str(profile), "--rank", str(rank)],
             "deadline_epoch": config["deadline_epoch"]}
            for profile in PROFILES for model in MODELS for rank in range(len(config["entries"]))]


@contextmanager
def pilot_window(config):
    """Interrupt a running worker too, not just its next queued request."""
    remaining = config["deadline_epoch"] - time.time()
    if remaining <= 0:
        raise TimeoutError("four-hour text pilot window expired")
    def expired(_signal, _frame):
        raise TimeoutError("four-hour text pilot window expired during execution")
    previous = signal.signal(signal.SIGALRM, expired)
    old_timer = signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        if old_timer[0]:
            signal.setitimer(signal.ITIMER_REAL, *old_timer)


def assignment_path(config, model):
    return Path(config["output_root"]) / "assignments" / f"{model}.json"


def validate_assignments(value, config, model):
    if (value.get("format") != ASSIGNMENTS_FORMAT or value.get("model") != model
            or value.get("config_sha256") != config["sha256"]
            or value.get("frozen_vision_policy_sha256") != config["frozen_policy_sha256"]
            or value.get("answers_consumed") is not False):
        raise ValueError("assignments must come from the frozen answer-free vision rule")
    rows = value["rows"]
    wanted = {e["row_hash"]: e["group_id"] for e in config["entries"]}
    if len(rows) != 24 or {r["request_id"] for r in rows} != set(wanted):
        raise ValueError("assignments must cover the exact 24 development pages")
    for row in rows:
        if row["group_id"] != wanted[row["request_id"]]:
            raise ValueError("assignment source identity changed")
        n = row["input_visual_tokens"]
        if type(n) is not int or n < 1 or any(type(row[k]) is not int or not 1 <= row[k] <= n
                                             for k in ("static_count", "adaptive_count")):
            raise ValueError("invalid exact retained count")
    if sum(r["static_count"] for r in rows) != sum(r["adaptive_count"] for r in rows):
        raise ValueError("static and adaptive aggregate visual counts differ")
    return value


def prepare_assignments(config, model, device="cuda:0"):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.backbone import read_entry, frozen_versions, assert_frozen
    from .compression_features_v12 import compute_feature_curves
    from .compression_policy_v12 import infer, answer_free
    path = assignment_path(config, model)
    if path.exists():
        return validate_assignments(_read_sealed(path), config, model)
    base = load_followup_protocol(config["base_protocol"])
    adapter, dataset, backbone, scorer, _ = load_model(base, model, device, with_scorer=True)
    scorer.requires_grad_(False)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    frozen = _read_sealed(config["frozen_policy"])
    rule = frozen["models"][model]["rule"]
    rows = []
    for entry in config["entries"]:
        if time.time() >= config["deadline_epoch"]:
            raise TimeoutError("text pilot expired during answer-free assignment preparation")
        sample = read_entry(dataset, entry)
        with torch.no_grad():
            prepared = adapter.prepare_sample(sample)
            vision = prepared.inputs_embeds[0, prepared.visual_positions]
            question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
            scores = scorer(vision, question).float()
            projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
            curves = compute_feature_curves(scores, vision, projected)
        rows.append({"request_id": entry["row_hash"], "row_hash": entry["row_hash"],
                     "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                     "input_visual_tokens": len(vision),
                     "input_text_tokens": int(prepared.inputs_embeds.shape[1] - len(vision)),
                     "curves": curves,
                     "full_ranking": torch.argsort(scores.float(), descending=True, stable=True).cpu().tolist()})
        del prepared, vision, question, scores, projected
    predictions = infer(rule, answer_free(rows))
    counts = predictions["counts"] if isinstance(predictions, dict) else predictions
    if isinstance(counts, dict):
        counts = [counts[row["request_id"]] for row in rows]
    if len(counts) != len(rows):
        raise ValueError("frozen vision inference returned an incomplete assignment")
    controls = uniform_exact_counts(rows, sum(counts))
    for row, count in zip(rows, counts):
        row.update(adaptive_count=int(count), static_count=controls[row["request_id"]],
                   fallback_reason=predictions["fallbacks"].get(row["request_id"]))
        ranking = row.pop("full_ranking")
        row["selected_indices_sha256"] = {
            key: digest(sorted(ranking[:row[key + "_count"]])) for key in ("static", "adaptive")}
    assert_frozen(adapter.model, versions)
    assert_frozen(scorer, scorer_versions)
    value = seal({"format": ASSIGNMENTS_FORMAT, "model": model,
                  "config_sha256": config["sha256"],
                  "frozen_vision_policy_sha256": config["frozen_policy_sha256"],
                  "rows": rows, "answers_consumed": False, "backbone": backbone,
                  "checkpoint_sha256": model_spec(base, model)["checkpoint_sha256"],
                  "prompt": "unchanged_olmocr_prompt_no_VQA_suffix", "labels_used_for_tuning": False,
                  "eligible_for_joint_pilot": any(r["adaptive_count"] < r["input_visual_tokens"] for r in rows),
                  "visual_pruning_active_fraction": mean(r["adaptive_count"] < r["input_visual_tokens"] for r in rows),
                  "fallback_fraction": len(predictions["fallbacks"]) / len(rows)})
    validate_assignments(value, config, model)
    write_immutable(path, value)
    return value


def selected_indices(prepared, scorer, keep):
    import torch
    from .adaptive_benchmark_v1 import select_exact_count
    vision = prepared.inputs_embeds[0, prepared.visual_positions]
    if type(keep) is not int or not 1 <= keep <= len(vision):
        raise ValueError("frozen assignment does not fit the prepared image")
    if keep == len(vision):
        return torch.arange(keep, dtype=torch.long, device=vision.device)
    question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
    return select_exact_count(scorer(vision, question), vision, keep)


def measured(adapter, sample, scorer, keep, profile, policy, *, max_new_tokens=MAX_NEW_TOKENS,
             stop_on_eos=True, optimized=True):
    """One physically evaluated cell; no learned model or cache code is modified."""
    import torch
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
    if hasattr(sample.get("image"), "seek"):
        sample["image"].seek(0)
    torch.cuda.synchronize(adapter.device)
    torch.cuda.reset_peak_memory_stats(adapter.device)
    started = time.perf_counter()
    with torch.no_grad():
        prepared = adapter.prepare_sample(sample)
        selected = selected_indices(prepared, scorer, keep)
        pruned = adapter.physically_prune(prepared, selected, "original")
        full, prefix = int(prepared.inputs_embeds.shape[1]), int(pruned.inputs_embeds.shape[1])
        n = len(prepared.visual_positions)
        preprocessing_peak = torch.cuda.max_memory_allocated(adapter.device)
        if full - prefix != n - keep:
            raise ValueError("physical pruning changed protected prompt/structural tokens")
        if policy == "fullref":
            result = generate_dynamic(adapter.model, pruned, max_new_tokens=max_new_tokens,
                                      stop_on_eos=stop_on_eos)
            budget = {"kv_budget_slots": None, "kv_retained_slots": None,
                      "staging_slots": 0, "available_generated_history_slots": None}
        else:
            budget = budget_for(full, prefix, profile, policy)
            result = generate_matched_v3(adapter.model, pruned, max_new_tokens,
                full_prefix_tokens=full, budget_profile=profile, policy=policy,
                staging=budget["staging_slots"], stop_on_eos=stop_on_eos, optimized=optimized,
                visual_source_positions=prepared.visual_positions)
        ids = result.ids[0].cpu().tolist()
        text = adapter.processor.decode(ids, skip_special_tokens=True).strip()
        metrics = dict(result.metrics)
        metrics["full_request_peak_allocated_bytes"] = max(preprocessing_peak, torch.cuda.max_memory_allocated(adapter.device))
        metrics["full_request_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(adapter.device)
        metrics["quality_shared_node_timing_not_isolated"] = True
        metrics["request_seconds"] = time.perf_counter() - started
        record = {"input_prefix_tokens": full, "retained_prefix_tokens": prefix,
                  "input_visual_tokens": n, "retained_visual_tokens": keep,
                  "selected_visual_indices_sha256": digest(selected.cpu().tolist()),
                  "prediction": text, "generated_ids": ids, "generated_ids_sha256": digest(ids),
                  "generated_tokens": len(ids), "max_new_tokens": max_new_tokens,
                  "natural_stop_enabled": stop_on_eos, "cache_policy": policy,
                  "metrics": metrics, "kv_budget_bytes": metrics["cache"]["kv_capacity_bytes"],
                  "quality_score": None, "quality_status": "awaiting_official_score", **budget}
    del result, prepared, pruned, selected
    return record


def validate_row(row):
    ids = row["generated_ids"]
    if (not ids or any(type(token) is not int or token < 0 for token in ids)
            or len(ids) != row["generated_tokens"] or len(ids) > row["max_new_tokens"]
            or digest(ids) != row["generated_ids_sha256"]):
        raise ValueError("generated token IDs changed or are incomplete")
    if row["input_prefix_tokens"] - row["retained_prefix_tokens"] != row["input_visual_tokens"] - row["retained_visual_tokens"]:
        raise ValueError("protected prompt/structural tokens were removed")
    cache = row["metrics"]["cache"]
    if row["cache_policy"] == "fullref":
        if cache.get("compaction_events", 0) != 0:
            raise ValueError("unrestricted baseline evicted cache entries")
        return
    expected = budget_for(row["input_prefix_tokens"], row["retained_prefix_tokens"],
                          row["profile"], row["cache_policy"])
    if any(row.get(k) != value for k, value in expected.items()):
        raise ValueError("reported budget disagrees with hard-cap bookkeeping")
    if (cache["capacity_tokens_per_layer"] != expected["kv_budget_slots"]
            or cache["kv_capacity_bytes"] != row["kv_budget_bytes"]
            or not 0 < cache["active_tokens"] <= cache["peak_active_tokens"] <= expected["kv_budget_slots"]
            or cache["protects_entire_prefix"] != expected["protects_entire_prefix"]):
        raise ValueError("physical cache violates the declared capacity or protection")


def cached_baseline(path, identity, generate):
    """Serialize same-page reuse across capacity workers; never replace a result."""
    import fcntl
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.with_suffix(".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            cached = _read_sealed(path)
            if cached["identity"] != identity:
                raise ValueError("unrestricted baseline hash identity differs; cannot reuse")
            return {**cached["result"], "reused_baseline_sha256": file_digest(path)}
        row = generate()
        write_immutable(path, seal({"identity": identity, "result": row}))
        return row


def canary(config, model, device="cuda:0"):
    """New physical pruning integration: full-reference parity, cap and EOS checks."""
    from learned_score.mlsys_scorer.backbone import read_entry, frozen_versions, assert_frozen
    base = load_followup_protocol(config["base_protocol"])
    assignments = validate_assignments(_read_sealed(assignment_path(config, model)), config, model)
    if assignments.get("eligible_for_joint_pilot") is not True:
        raise ValueError("frozen vision rule retains all images; joint pilot is ineligible, do not retune on olmOCR")
    adapter, dataset, _, scorer, _ = load_model(base, model, device, with_scorer=True)
    scorer.requires_grad_(False)
    versions, sversions = frozen_versions(adapter.model), frozen_versions(scorer)
    entry = config["entries"][0]
    assigned = next(row for row in assignments["rows"] if row["request_id"] == entry["row_hash"])
    sample = read_entry(dataset, entry)
    checks = []
    for keep in sorted({assigned["input_visual_tokens"], assigned["adaptive_count"]}):
        reference = measured(adapter, sample, scorer, keep, None, "fullref", max_new_tokens=16, stop_on_eos=False)
        for policy in ("protected_recency", "rkv"):
            row = measured(adapter, sample, scorer, keep, 256, policy, max_new_tokens=16, stop_on_eos=False)
            row["profile"] = 256
            validate_row(row)
            if row["generated_ids"] != reference["generated_ids"]:
                raise ValueError("no-eviction native token parity failed")
            checks.append({"keep": keep, "policy": policy, "native_no_eviction_exact_ids": True})
    for policy in ("protected_recency", "rkv"):
        rows = [measured(adapter, sample, scorer, assigned["input_visual_tokens"], 256,
                         policy, max_new_tokens=320, stop_on_eos=False, optimized=optimized)
                for optimized in (False, True)]
        if rows[0]["generated_ids"] != rows[1]["generated_ids"]:
            raise ValueError("reference/optimized forced-eviction parity failed")
        for row in rows:
            row["profile"] = 256
            validate_row(row)
            if row["metrics"]["cache"]["compaction_events"] < 1:
                raise ValueError("forced-eviction canary never exercised eviction")
        checks.append({"policy": policy, "forced_eviction_exact_ids": True})
    assert_frozen(adapter.model, versions)
    assert_frozen(scorer, sversions)
    value = seal({"config_sha256": config["sha256"], "model": model,
                  "assignments_sha256": file_digest(assignment_path(config, model)),
                  "passed": True, "checks": checks, "quality_labels_used": False})
    write_immutable(Path(config["output_root"]) / "canary" / f"{model}.json", value)
    return value


def run_worker(config, model, rank, profile, device="cuda:0"):
    import torch
    from learned_score.mlsys_scorer.backbone import read_entry, frozen_versions, assert_frozen
    from .long_document_benchmark import execution_context
    if model not in MODELS or profile not in PROFILES or not 0 <= rank < 24:
        raise ValueError("invalid declared pilot task")
    out = Path(config["output_root"]) / model / f"b{profile}" / f"rank{rank}"
    if (out / "complete.json").exists():
        return collect_page(config, model, rank, profile)
    assignments = validate_assignments(_read_sealed(assignment_path(config, model)), config, model)
    if assignments.get("eligible_for_joint_pilot") is not True:
        raise ValueError("frozen vision rule retains all images; joint pilot is ineligible")
    gate = _read_sealed(Path(config["output_root"]) / "canary" / f"{model}.json")
    if (gate.get("config_sha256") != config["sha256"] or gate.get("passed") is not True
            or gate.get("assignments_sha256") != file_digest(assignment_path(config, model))):
        raise ValueError("matching physical canary is required before quality generation")
    base = load_followup_protocol(config["base_protocol"])
    adapter, dataset, backbone, scorer, _ = load_model(base, model, device, with_scorer=True)
    scorer.requires_grad_(False)
    versions, sversions = frozen_versions(adapter.model), frozen_versions(scorer)
    entry = config["entries"][rank]
    assigned = next(row for row in assignments["rows"] if row["request_id"] == entry["row_hash"])
    sample = read_entry(dataset, entry)
    attempt = uuid.uuid4().hex
    context = execution_context()
    baseline_identity = {"config_sha256": config["sha256"], "model": model,
                         "request_id": entry["row_hash"], "image_sha256": entry["image_sha256"],
                         "checkpoint_sha256": model_spec(base, model)["checkpoint_sha256"],
                         "backbone": backbone, "max_new_tokens": MAX_NEW_TOKENS,
                         "natural_stop_enabled": True, "input_hashes": config["input_hashes"],
                         "gpu": torch.cuda.get_device_name(adapter.device),
                         "torch_version": str(torch.__version__), "cuda_version": torch.version.cuda}
    baseline_path = (Path(config["output_root"]) / "baselines" / model /
                     entry["row_hash"] / f"{digest(baseline_identity)}.json")
    rows = []
    for method in ("fullref", *METHODS):
        if time.time() >= config["deadline_epoch"]:
            raise TimeoutError("four-hour text pilot limit reached")
        policy = "fullref" if method == "fullref" else "rkv" if method == "adaptive_rkv" else "protected_recency"
        keep = (assigned["input_visual_tokens"] if method in ("fullref", "allvision_recency") else
                assigned["static_count"] if method == "static_recency" else assigned["adaptive_count"])
        def generate():
            torch.cuda.empty_cache()
            row = measured(adapter, sample, scorer, keep, profile if method != "fullref" else None, policy)
            row["source_execution_context"] = context
            return row
        row = cached_baseline(baseline_path, baseline_identity, generate) if method == "fullref" else generate()
        row.update(format=FORMAT, config_sha256=config["sha256"], model=backbone["model"],
                   model_tag=model, method=method, profile=profile, request_id=entry["row_hash"],
                   group_id=entry["group_id"], image_sha256=entry["image_sha256"],
                   dataset="olmOCR-Bench", split="scorer_dev", stage="development", status="ok",
                   repetition=0, attempt=attempt, execution_context=context,
                   hardware=baseline_identity["gpu"],
                   assignments_sha256=file_digest(assignment_path(config, model)),
                   quality_run_only=True, timing_isolated=False)
        if row["input_visual_tokens"] != assigned["input_visual_tokens"]:
            raise ValueError("prepared image size differs from frozen assignments")
        selection_kind = "static" if method == "static_recency" else "adaptive"
        if (method not in ("fullref", "allvision_recency") and row["selected_visual_indices_sha256"] !=
                assigned["selected_indices_sha256"][selection_kind]):
            raise ValueError("physical selected-token identities differ from answer-free assignment extraction")
        validate_row(row)
        rows.append(row)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, sversions)
        write_immutable(out / "attempts" / attempt / f"{method}.json", row)
    allocations = {(r["kv_budget_slots"], r["kv_budget_bytes"]) for r in rows if r["method"] != "fullref"}
    if len(allocations) != 1:
        raise ValueError("bounded methods do not have equal physical KV capacity")
    if rows[-1]["selected_visual_indices_sha256"] != rows[-2]["selected_visual_indices_sha256"]:
        raise ValueError("adaptive R-KV must use the identical adaptive image selection")
    write_immutable(out / "complete.json", seal({"config_sha256": config["sha256"],
                    "model": model, "rank": rank, "profile": profile, "attempt": attempt,
                    "row_hashes": {str(Path("attempts") / attempt / f"{r['method']}.json"):
                                   file_digest(out / "attempts" / attempt / f"{r['method']}.json") for r in rows}}))
    return rows


def collect_page(config, model, rank, profile):
    root = Path(config["output_root"]) / model / f"b{profile}" / f"rank{rank}"
    marker = _read_sealed(root / "complete.json")
    if any(marker.get(k) != v for k, v in {"config_sha256": config["sha256"], "model": model,
                                          "rank": rank, "profile": profile}.items()):
        raise ValueError("completed page identity differs")
    rows = []
    for relative, sha in marker["row_hashes"].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()) or file_digest(path) != sha:
            raise ValueError("completed row hash changed")
        row = json.loads(path.read_text())
        if (row.get("config_sha256") != config["sha256"] or row.get("attempt") != marker["attempt"]
                or row.get("request_id") != config["entries"][rank]["row_hash"]):
            raise ValueError("completed row identity changed")
        validate_row(row)
        rows.append(row)
    if len(rows) != 5 or {row["method"] for row in rows} != {"fullref", *METHODS}:
        raise ValueError("incomplete paired page matrix")
    bounded = [r for r in rows if r["method"] != "fullref"]
    if len({(r["kv_budget_slots"], r["kv_budget_bytes"]) for r in bounded}) != 1:
        raise ValueError("completed methods have unequal physical capacity")
    indexed = {row["method"]: row for row in rows}
    if indexed["adaptive_recency"]["selected_visual_indices_sha256"] != indexed["adaptive_rkv"]["selected_visual_indices_sha256"]:
        raise ValueError("completed adaptive selections differ")
    return rows


def report(config, model, profile, output):
    """Official scoring only after all 24 predetermined pages complete."""
    from learned_score.mlsys_eval.common import load_manifest
    from learned_score.mlsys_eval.olmocr_score import score_candidate, verify_evaluator
    from learned_score.mlsys_next.cache_report_v4 import development_subset
    from learned_score.mlsys_next.cache_method_report_v3 import paired_difference
    rows = [row for rank in range(24) for row in collect_page(config, model, rank, profile)]
    manifest = load_manifest(config["manifest"])
    subset = development_subset({"split": "scorer_dev", "entries": config["entries"]}, manifest)
    verify_evaluator(config["evaluator_root"], subset, config["cache_patch"])
    context = digest({"config": config["sha256"], "model": model, "profile": profile,
                      "rows": rows})
    scores, grouped = {}, {}
    for method in ("fullref", *METHODS):
        grouped[method] = [r for r in rows if r["method"] == method]
        # The official adapter requires one action label. Exact per-page counts
        # remain in the physical rows; None explicitly means adaptive/fixed-rule.
        records = [{**r, "run_context": context, "visual_retention": None,
                    "generated_budget": None} for r in grouped[method]]
        score_path = Path(output) / context / method / "official_scores.json"
        if score_path.exists():
            value = json.loads(score_path.read_text())
            if value.get("run_context") != context:
                raise ValueError("official score belongs to another sealed context")
            scores[method] = value
        else:
            scores[method] = score_candidate(subset, records, score_path.parent, split="scorer_dev")
    quality = []
    baseline_bytes = mean(r["kv_budget_bytes"] for r in grouped["fullref"])
    baseline_by_id = {r["request_id"]: r["kv_budget_bytes"] for r in grouped["fullref"]}
    for method, physical in grouped.items():
        memory = mean(r["kv_budget_bytes"] for r in physical)
        quality.append({"method": method, "score_percent": 100 * scores[method]["official_macro_score"],
                        "versus_full": paired_difference(scores[method], scores["fullref"], repetitions=10000),
                        "versus_allvision_bounded": paired_difference(scores[method], scores["allvision_recency"], repetitions=10000),
                        "mean_allocated_kv_mib": memory / 2**20,
                        "kv_reduction_vs_measured_full_percent": 100 * (1 - memory / baseline_bytes),
                        "requests_with_memory_reduction_percent": 100 * mean(
                            r["kv_budget_bytes"] < baseline_by_id[r["request_id"]] for r in physical),
                        "mean_occupied_kv_mib": mean(r["metrics"]["cache"]["kv_active_bytes"] for r in physical) / 2**20,
                        "mean_auxiliary_cache_mib": mean(r["metrics"]["cache"].get("auxiliary_persistent_bytes",
                            r["metrics"]["cache"].get("source_metadata_bytes", 0)) for r in physical) / 2**20,
                        "mean_peak_gpu_allocated_mib": mean(r["metrics"]["full_request_peak_allocated_bytes"]
                                                             for r in physical) / 2**20,
                        "mean_retained_prefix_tokens": mean(r["retained_prefix_tokens"] for r in physical),
                        "mean_retained_image_tokens": mean(r["retained_visual_tokens"] for r in physical),
                        "mean_generated_history_slots": (None if method in ("fullref", "adaptive_rkv") else
                              mean(r["available_generated_history_slots"] for r in physical)),
                        "eviction_active_fraction": mean(r["metrics"]["cache"].get("compaction_events", 0) > 0 for r in physical),
                        "visual_pruning_active_fraction": mean(r["retained_visual_tokens"] < r["input_visual_tokens"] for r in physical),
                        "mean_output_tokens": mean(r["generated_tokens"] for r in physical),
                        "output_limit_fraction": mean(r["generated_tokens"] == MAX_NEW_TOKENS for r in physical)})
    adaptive = next(row for row in quality if row["method"] == "adaptive_recency")
    value = {"format": FORMAT, "config_sha256": config["sha256"], "model": model,
             "profile": profile, "complete": True, "paired_source_count": 24,
             "quality": quality, "independent_confirmation": False,
             "paired_scope": [entry["row_hash"] for entry in config["entries"]],
             "same_capacity_pruning_reallocates_memory_does_not_reduce_total_capacity": True,
             "rkv_is_existing_unprotected_prefix_adaptation": True,
             "advance_combined_evaluation": (adaptive["versus_allvision_bounded"]["delta_pp"] >= -1 - 1e-9
                    and adaptive["visual_pruning_active_fraction"] > 0 and adaptive["eviction_active_fraction"] > 0)}
    write_immutable(Path(output) / context / "report.json", value)
    return value


def prepare_model(config, model, device="cuda:0"):
    """One queue task publishes a terminal preparation or ineligibility record."""
    path = Path(config["output_root"]) / "preparation" / f"{model}.complete.json"
    if path.exists():
        value = _read_sealed(path)
        if (value.get("config_sha256") != config["sha256"] or value.get("model") != model
                or value.get("assignments_sha256") != file_digest(assignment_path(config, model))):
            raise ValueError("completed preparation identity changed")
        return value
    assigned = prepare_assignments(config, model, device)
    eligible = assigned["eligible_for_joint_pilot"]
    gate = canary(config, model, device) if eligible else None
    value = seal({"config_sha256": config["sha256"], "model": model,
                  "assignments_sha256": file_digest(assignment_path(config, model)),
                  "eligible_for_joint_pilot": eligible,
                  "visual_pruning_active_fraction": assigned["visual_pruning_active_fraction"],
                  "fallback_fraction": assigned["fallback_fraction"],
                  "canary_sha256": gate["sha256"] if gate else None,
                  "status": "ready" if eligible else "ineligible_frozen_vision_rule_keeps_all",
                  "labels_used_for_tuning": False})
    write_immutable(path, value)
    return value


def _expire_queued_plans(study):
    """Stop only unlocked, uncompleted tasks in this pilot's own namespaces."""
    import fcntl
    from .state import atomic_json
    from .worker import read_plan
    root = Path(study["output_root"]) / "plans"
    for model in MODELS:
        for stage in ("text-prepare", "text-512", "text-256"):
            directory = root / f"{stage}-{model}"
            if not (directory / "plan.json").exists():
                continue
            plan = read_plan(directory / "plan.json")
            state_root = directory / "state"
            state_root.mkdir(parents=True, exist_ok=True)
            for task in plan["tasks"]:
                with (state_root / (task["id"] + ".lock")).open("a+") as lock:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    path = state_root / (task["id"] + ".json")
                    state = json.loads(path.read_text()) if path.exists() else {}
                    if state.get("status") in ("complete", "failed", "blocked"):
                        continue
                    if state and state.get("plan_sha256") != plan["sha256"]:
                        raise ValueError("text queue state belongs to another plan")
                    atomic_json(path, {**state, "status": "blocked", "reason": "text_pilot_four_hour_deadline",
                                       "plan_sha256": plan["sha256"]})


def advance(study, selection_path):
    """Publish preparation/physical queues and complete-cohort reports; no jobs.

    Called by the v12 CPU coordinator. GPU work uses the existing generic task
    worker; only the normal dispatchers may acquire allocations or launch it.
    """
    from .compression_study_v12 import verify
    from .compression_execution_v12 import publish, make_task, python
    from .state import atomic_json
    verify(study)
    selection_path = Path(selection_path)
    if not selection_path.exists():
        return {"status": "waiting_for_frozen_vision_selection", "launched": False}
    frozen = _read_sealed(selection_path)
    if frozen.get("study_sha256") != study["sha256"]:
        raise ValueError("text pilot selection belongs to another study")
    if frozen.get("rule_kind") is None:
        return {"status": "ineligible_no_shared_vision_rule", "launched": False}
    output = Path(study["output_root"]) / "text"
    config_path = output / "config.json"
    if not config_path.exists():
        now = time.time()
        if now >= study["deadline_utc_epoch"]:
            return {"status": "study_deadline_before_text_pilot", "launched": False}
        parent = Path(study["cwd"]) / "reproduction_results/mlsys_cache_methods_20261014_v4/protocol.json"
        parent_value = _read_sealed(parent)
        for model in MODELS:
            spec = model_spec(parent_value, model)
            if (any(study["models"][model].get(key) != spec.get(key) for key in
                    ("adapter", "model_path", "revision", "max_visual_patches"))
                    or file_digest(study["models"][model]["checkpoint"]) != spec["checkpoint_sha256"]):
                raise ValueError("text and vision model/scorer identities must match")
        config = build_config(parent, selection_path, output, started_epoch=now,
                              deadline_epoch=min(now + MAX_SECONDS, study["deadline_utc_epoch"]))
        write_immutable(config_path, config)
    config = load_config(config_path)
    expired = time.time() >= config["deadline_epoch"]
    if expired:
        _expire_queued_plans(study)
    status = {"status": "deadline_reached" if expired else "running", "launched": False,
              "config": str(config_path), "deadline_epoch": config["deadline_epoch"],
              "models": {}, "plans": [], "all_results_development": True}
    prep = {}
    pins = {str(config_path): file_digest(config_path), **config["input_hashes"]}
    for model in MODELS:
        marker = output / "preparation" / f"{model}.complete.json"
        if not marker.exists():
            status["models"][model] = {"status": "awaiting_preparation"}
            if not expired:
                argv = python(study, "compression_text_v12") + ["prepare", "--config", str(config_path), "--model", model]
                task = make_task(study, "text-prepare-" + model, argv, [marker],
                                 model=model, minimum_start_seconds=1200)
                status["plans"].append(str(publish(study, "text-prepare-" + model, [task], pins)))
            continue
        prepared = _read_sealed(marker)
        if (prepared.get("config_sha256") != config["sha256"] or prepared.get("model") != model
                or prepared.get("assignments_sha256") != file_digest(assignment_path(config, model))):
            raise ValueError("text preparation marker changed")
        prep[model] = prepared
        status["models"][model] = {"status": prepared["status"],
                                    "visual_pruning_active_fraction": prepared["visual_pruning_active_fraction"],
                                    "fallback_fraction": prepared["fallback_fraction"], "profiles": {}}
    first_capacity_complete = len(prep) == len(MODELS) and all(
        not prep[model]["eligible_for_joint_pilot"] or all(
            (output / model / "b512" / f"rank{rank}" / "complete.json").exists() for rank in range(24))
        for model in MODELS)
    for model, prepared in prep.items():
        if not prepared["eligible_for_joint_pilot"]:
            continue
        for profile in PROFILES:
            ready = [rank for rank in range(24) if
                     (output / model / f"b{profile}" / f"rank{rank}" / "complete.json").exists()]
            state = {"completed_paired_pages": len(ready), "required_pages": 24}
            status["models"][model]["profiles"][str(profile)] = state
            if len(ready) == 24:
                index = output / "reports" / model / f"b{profile}.json"
                if index.exists():
                    result = json.loads(index.read_text())
                    if result.get("config_sha256") != config["sha256"]:
                        raise ValueError("text report belongs to another config")
                else:
                    result = report(config, model, profile, output / "reports" / model / f"b{profile}")
                    write_immutable(index, result)
                state.update(status="complete", report=str(index),
                             advance_combined_evaluation=result["advance_combined_evaluation"])
                continue
            if expired or (profile == 256 and not first_capacity_complete):
                state["status"] = "deadline_reached" if expired else "waiting_for_complete_512_comparisons"
                continue
            tasks = []
            for rank in range(24):
                argv = python(study, "compression_text_v12") + ["worker", "--config", str(config_path),
                        "--model", model, "--profile", str(profile), "--rank", str(rank)]
                marker = output / model / f"b{profile}" / f"rank{rank}" / "complete.json"
                tasks.append(make_task(study, f"text-{profile}-{model}-{rank}", argv, [marker],
                                       model=model, minimum_start_seconds=900))
            phase_pins = {**pins, str(assignment_path(config, model)): file_digest(assignment_path(config, model)),
                          str(output / "canary" / f"{model}.json"): file_digest(output / "canary" / f"{model}.json")}
            status["plans"].append(str(publish(study, f"text-{profile}-{model}", tasks, phase_pins)))
            state["status"] = "queued"
    if len(prep) == 2 and all(not p["eligible_for_joint_pilot"] for p in prep.values()):
        status["status"] = "ineligible_frozen_vision_rules_keep_all"
    elif len(prep) == 2 and all(not prepared["eligible_for_joint_pilot"] or all(
            state.get("status") == "complete" for state in status["models"][model]["profiles"].values())
            for model, prepared in prep.items()):
        status["status"] = "complete"
    atomic_json(output / "status.json", status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("config")
    build.add_argument("--parent-v4-protocol", required=True, type=Path)
    build.add_argument("--frozen-policy", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    for name in ("prepare", "prepare-assignments", "canary", "worker", "report"):
        sub = commands.add_parser(name)
        sub.add_argument("--config", required=True, type=Path)
        sub.add_argument("--model", choices=MODELS, required=True)
        sub.add_argument("--device", default="cuda:0")
        if name in ("worker", "report"):
            sub.add_argument("--profile", type=int, choices=PROFILES, required=True)
        if name == "worker":
            sub.add_argument("--rank", type=int, required=True)
        if name == "report":
            sub.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "config":
        config = build_config(args.parent_v4_protocol, args.frozen_policy, args.output)
        write_immutable(args.output / "config.json", config)
        return
    config = load_config(args.config)
    if args.command == "report":
        print(json.dumps(report(config, args.model, args.profile, args.output)), flush=True)
        return
    require_cuda(args.device)
    with pilot_window(config):
        if args.command == "prepare":
            value = prepare_model(config, args.model, args.device)
        elif args.command == "prepare-assignments":
            value = prepare_assignments(config, args.model, args.device)
        elif args.command == "canary":
            value = canary(config, args.model, args.device)
        else:
            value = run_worker(config, args.model, args.rank, args.profile, args.device)
    print(json.dumps({"event": "text_pilot_complete", "command": args.command,
                      "model": args.model, "rows": len(value) if isinstance(value, list) else None}), flush=True)


if __name__ == "__main__":
    main()
