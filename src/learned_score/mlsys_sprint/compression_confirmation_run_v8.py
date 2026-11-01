"""Fail-closed fresh confirmation execution after development selection freezes.

This module publishes plans; it never submits jobs or fits policies. Blocked
identity audits cause no writes. The fresh-cohort auditor owns manifest
publication. A separately sealed gate pins selection and sources before any
answer-free feature task can be published. Confirmation labels cannot change
counts, methods, cohorts, or the screening-selected static retention levels.
"""

from collections import defaultdict
from pathlib import Path
import time

from . import adaptive_experiment_v3 as feature_runtime
from . import compression_confirmation_v8 as freshness
from . import compression_matrix_v8 as matrix
from . import compression_policy_v8 as policies
from . import compression_report_v8 as reporting
from .followup_common import write_immutable
from .state import digest, file_digest


FORMAT = "mlsys_compression_confirmation_run_v8"
FEATURE_FIELDS = frozenset({
    "format", "measurement_version", "request_id", "group_id", "image_sha256", "dataset", "split",
    "identity_sha256", "checkpoint_sha256", "cache_contract", "answers_consumed_by_feature_extractor",
    "input_visual_tokens", "input_text_tokens", "normalized_entropy", "nearest_neighbor_diversity",
    "top_mass_by_percent",
})
read, seal = matrix.read, matrix.seal


def checked(value):
    return matrix.planning._checked(value)


def sources():
    return {str(Path(module.__file__).resolve()): file_digest(module.__file__)
            for module in (matrix, policies, reporting, freshness)} | {
                str(Path(__file__).resolve()): file_digest(__file__)}


def feature_sources(adapter):
    """Exact v4 Qwen/LLaVA source contract without importing its Phi GPU shim."""
    if adapter not in {"qwen2_5_vl", "llava_next"}:
        raise ValueError("confirmation supports only the two frozen study models")
    return {**feature_runtime.source_hashes(adapter),
            "mlsys_sprint/adaptive_experiment_v4.py": file_digest(Path(__file__).with_name("adaptive_experiment_v4.py"))}


def admission_reasons(study, selection, audit):
    """Pure admission check. Eligibility is not a final image-freshness proof."""
    if selection is None:
        return ["development_selection_not_frozen"]
    checked(selection)
    if (selection.get("study_sha256") != study["sha256"]
            or selection.get("confirmation_labels_consumed") is not False
            or type(selection.get("created_utc_epoch")) not in (int, float)):
        raise ValueError("selection is not a pre-label freeze for this study")
    if not isinstance(audit, dict):
        return ["fresh_identity_audit_missing"]
    freshness.checked(audit)
    if audit.get("study_sha256") != study["sha256"]:
        raise ValueError("fresh audit belongs to another study")
    reasons = []
    for key, expected in (("eligible", True), ("labels_loaded", False), ("annotations_opened", False)):
        if audit.get(key) is not expected:
            reasons.append("audit_" + key + "_not_verified")
    for key in ("blocking_reasons", "unknown_artifacts", "unsupported_artifacts"):
        if audit.get(key) != []:
            reasons.append("audit_" + key)
    if audit.get("minimum_non_mme_groups") != study["confirmation"]["non_mme_target_groups"]:
        reasons.append("confirmation_cohort_size_contract_changed")
    return reasons


def _methods(model):
    return ("scorer", "divprune") + (("cdpruner",) if model == "llava_next" else ())


def _bindings(study, selection, datasets):
    """Extract only frozen policy identities and static keeps, not outcomes."""
    result = {}
    expected = {model + "|" + dataset for model in matrix.MODELS for dataset in datasets}
    if set(selection.get("domains", {})) != expected:
        raise ValueError("fresh cohort datasets differ from frozen development domains")
    for key in sorted(expected):
        model, dataset = key.split("|", 1)
        choice = selection["domains"][key]
        if choice is None:
            policy_sha, policy_file = None, None
        else:
            policy = policies.load_policy(choice["policy"])
            if (policy["identity"].get("model") != model or policy["identity"].get("dataset") != dataset
                    or choice.get("family") != selection["selected_family"]
                    or policy["statistic"] != choice["family"]):
                raise ValueError("selected policy model, dataset, or scalar family changed")
            policy_sha, policy_file = policy["sha256"], choice["policy_file"]
        keeps = {}
        for selector in _methods(model):
            baseline = selection.get("baselines", {}).get(key + "|" + selector)
            if (not isinstance(baseline, dict) or baseline.get("selector") != selector
                    or type(baseline.get("keep_percent")) is not int
                    or baseline["keep_percent"] not in policies.KEEP_PERCENTAGES):
                raise ValueError("missing or invalid frozen static baseline keep")
            keeps[selector] = baseline["keep_percent"]
        result[key] = {"policy_sha256": policy_sha, "policy_file": policy_file,
                       "static_keep_percent": keeps}
    return result


def _manifest_entries(path):
    from learned_score.mlsys_eval.common import load_manifest
    manifest = load_manifest(path)
    metadata = manifest.get("metadata", {})
    if (metadata.get("fresh_confirmation") is not True
            or metadata.get("selection_uses_labels") is not False
            or any(manifest.get("splits", {}).get(s) for s in ("fit", "screening", "phi_train"))):
        raise ValueError("confirmation requires a separate fresh evaluation-only manifest")
    entries = freshness.identity_rows(manifest)
    if not entries or len({row["row_hash"] for row in entries}) != len(entries):
        raise ValueError("fresh manifest must contain a unique nonempty cohort")
    return manifest, entries


def _verify_pins(pins):
    for path, expected in pins.items():
        if file_digest(path) != expected:
            raise ValueError("frozen confirmation source/input changed: " + str(path))


def _gate_check(study, selection, gate):
    checked(gate)
    if (gate.get("format") != FORMAT + "_gate" or gate.get("study_sha256") != study["sha256"]
            or gate.get("selection_sha256") != selection["sha256"]
            or gate.get("fresh_identity_audit_passed") is not True
            or gate.get("policies_frozen_before_labels") is not True
            or gate.get("created_utc_epoch", -1) < selection["created_utc_epoch"]):
        raise ValueError("confirmation gate does not bind the pre-label selection")
    return gate


def freeze_gate(study, selection, audit):
    """Final freshness audit and immutable source freeze; never called if blocked."""
    if admission_reasons(study, selection, audit):
        raise ValueError("fresh confirmation admission is blocked")
    root = Path(study["output_root"])
    gate_path = root / "confirmation/gate.json"
    selection_path = root / "selection.json"
    if matrix.checked(selection_path) != selection:
        raise ValueError("selection argument differs from its immutable publication")
    if gate_path.exists():
        gate = _gate_check(study, selection, matrix.checked(gate_path))
        if gate["audit_sha256"] != audit["sha256"] or gate["source_hashes"] != sources():
            raise ValueError("confirmation freshness audit or frozen sources changed")
        _verify_pins(gate["input_hashes"])
        return gate
    for model in matrix.MODELS:
        for stage in ("confirmation-features", "confirmation"):
            if any((root / model / stage).rglob("*")) or (root / f"plans/{stage}-{model}/plan.json").exists():
                raise ValueError("confirmation artifacts predate the selection/freshness gate")
    manifest_path, proof = freshness.verified_manifest(study, audit)
    manifest_path = Path(manifest_path)
    freshness.checked(proof)
    manifest, entries = _manifest_entries(manifest_path)
    if (proof.get("study_sha256") != study["sha256"] or proof.get("audit_sha256") != audit["sha256"]
            or proof.get("manifest_path") != str(manifest_path)
            or proof.get("manifest_sha256") != file_digest(manifest_path)
            or proof.get("manifest_content_sha256") != manifest["manifest_sha256"]
            or proof.get("fresh_identity_audit_passed") is not True
            or proof.get("image_exclusions_verified") is not True
            or proof.get("labels_loaded") is not False or proof.get("annotations_opened") is not False):
        raise ValueError("final manifest lacks a sealed image-disjoint freshness proof")
    bindings = _bindings(study, selection, sorted({r["dataset"] for r in entries}))
    proof_path = root / "confirmation/fresh_manifest_proof.json"
    write_immutable(proof_path, proof)
    paths = [root / "study.json", root / "pipeline.json", selection_path, manifest_path, proof_path]
    for model in matrix.MODELS:
        path = root / model / "policies/index.json"
        if file_digest(path) != selection["fit_index_hashes"][model]:
            raise ValueError("selected fit policy index changed")
        paths.append(path)
    for binding in bindings.values():
        artifact = binding["policy_file"]
        if artifact is not None:
            if file_digest(artifact["path"]) != artifact["sha256"]:
                raise ValueError("selected policy bundle changed")
            bundle = matrix.checked(artifact["path"])
            if binding["policy_sha256"] not in {p["sha256"] for p in bundle["policies"]}:
                raise ValueError("selected policy is absent from its pinned bundle")
            paths.append(Path(artifact["path"]))
    pins = {**study["input_hashes"], **proof.get("input_hashes", {}), **sources(),
            **{str(path): file_digest(path) for path in paths}}
    _verify_pins(pins)
    now = time.time()
    if now < selection["created_utc_epoch"]:
        raise ValueError("selection timestamp is in the future")
    gate = seal({"format": FORMAT + "_gate", "study_sha256": study["sha256"],
        "selection_sha256": selection["sha256"], "audit_sha256": audit["sha256"],
        "manifest_path": str(manifest_path), "manifest_sha256": file_digest(manifest_path),
        "manifest_content_sha256": manifest["manifest_sha256"], "proof_sha256": proof["sha256"],
        "source_hashes": sources(), "input_hashes": pins, "bindings": bindings,
        "created_utc_epoch": now, "selection_created_utc_epoch": selection["created_utc_epoch"],
        "fresh_identity_audit_passed": True, "policies_frozen_before_labels": True,
        "cohort": {dataset: sorted(r["row_hash"] for r in entries if r["dataset"] == dataset)
                   for dataset in sorted({r["dataset"] for r in entries})}})
    write_immutable(gate_path, gate)
    return gate


def build_feature_plan(study, model, gate):
    """Build (without writing) a generic v4 answer-free feature extraction plan."""
    checked(gate)
    if model not in matrix.MODELS or gate.get("study_sha256") != study["sha256"]:
        raise ValueError("feature plan model/study mismatch")
    if gate.get("fresh_identity_audit_passed") is not True or gate.get("policies_frozen_before_labels") is not True:
        raise ValueError("feature plan requires an admitted pre-label gate")
    root, spec = Path(study["output_root"]), study["models"][model]
    destination = root / model / "confirmation-features"
    tasks = []
    for rank in range(study["world_size"]):
        argv = matrix.previous.python(study, "adaptive_experiment_v4") + ["--phase", "features",
            "--split", "evaluation", "--manifest", gate["manifest_path"], "--checkpoint", spec["checkpoint"],
            "--adapter", spec["adapter"], "--model", spec["model_path"], "--model-revision", spec["revision"],
            "--output", str(destination), "--rank", str(rank), "--world-size", str(study["world_size"])]
        task = matrix.previous.task(study, f"confirmation-features-{model}-{rank}", argv,
            [destination / f"rank{rank}.complete.json"], resume=destination / f"rank{rank}.metadata.json")
        task.update(id=f"compression-v8-confirmation-features-{model.replace('_', '-')}-{rank}",
                    tag="compression-v8", model_tag=model)
        tasks.append(task)
    pins = dict(gate["input_hashes"])
    for relative, sha in feature_sources(spec["adapter"]).items():
        pins[str(Path(study["cwd"]) / "learned_score" / relative)] = sha
    return seal({"format": FORMAT + "_feature_plan", "cwd": study["cwd"], "tasks": tasks,
        "input_hashes": pins, "gate_sha256": gate["sha256"], "unexpected_retries": 1,
        "created_utc_epoch": gate["created_utc_epoch"], "deadline_utc_epoch": study["deadline_utc_epoch"]})


def collect_features(study, model, gate):
    """Read only complete, exact-cohort, answer-free feature shards."""
    directory = Path(study["output_root"]) / model / "confirmation-features"
    world = study["world_size"]
    if not all((directory / f"rank{rank}.complete.json").is_file() for rank in range(world)):
        return None
    manifest, entries = _manifest_entries(gate["manifest_path"])
    if file_digest(gate["manifest_path"]) != gate["manifest_sha256"]:
        raise ValueError("fresh manifest changed after the confirmation gate")
    ordered = sorted(entries, key=lambda r: (r["dataset"], r["row_hash"]))
    checkpoint = file_digest(study["models"][model]["checkpoint"])
    result, contexts, pins = {}, set(), {}
    first_metadata = None
    for rank in range(world):
        meta_path, marker_path = (directory / f"rank{rank}.{kind}.json" for kind in ("metadata", "complete"))
        meta, marker = read(meta_path), read(marker_path)
        expected = {"format": "mlsys_adaptive_experiment_v4", "phase": "features", "split": "evaluation",
            "manifest_sha256": gate["manifest_sha256"], "checkpoint_sha256": checkpoint,
            "world_size": world, "limit": None, "max_new_tokens": 128, "generated_budget": 512,
            "schedule_sha256": None, "schedule_content_sha256": None, "feature_root": None,
            "source_sha256": feature_sources(study["models"][model]["adapter"])}
        assigned = ordered[rank::world]
        if (any(meta.get(k) != v for k, v in expected.items()) or marker.get("identity_sha256") != digest(meta)
                or marker.get("requests") != len(assigned) or marker.get("results") != len(assigned)):
            raise ValueError("confirmation feature shard differs from the frozen cohort/runtime")
        contract = meta["cache_contract"]
        if (contract.get("reserved_generated_slots") != 127 or contract.get("generated_budget") != 512
                or contract.get("scratch_bytes") != 0 or contract.get("compaction_possible") is not False
                or type(contract.get("total_owned_bytes_per_slot")) is not int
                or contract["total_owned_bytes_per_slot"] < 1):
            raise ValueError("unsupported confirmation cache geometry")
        contexts.add(digest(matrix.runtime_contract(meta)))
        first_metadata = first_metadata or meta
        pins.update({str(meta_path): file_digest(meta_path), str(marker_path): file_digest(marker_path)})
        for entry in assigned:
            path = directory / "features" / (entry["row_hash"] + ".json")
            feature = read(path)
            expected_feature = {"format": "mlsys_answer_free_features_v3", "measurement_version": "v4",
                "request_id": entry["row_hash"], "group_id": entry["group_id"], "dataset": entry["dataset"],
                "image_sha256": entry["image_sha256"], "split": "evaluation", "identity_sha256": digest(meta),
                "checkpoint_sha256": checkpoint, "cache_contract": contract,
                "answers_consumed_by_feature_extractor": False}
            if set(feature) != FEATURE_FIELDS or any(feature.get(k) != v for k, v in expected_feature.items()):
                raise ValueError("confirmation feature identity or answer-free schema changed")
            for key, minimum in (("input_visual_tokens", 1), ("input_text_tokens", 0)):
                if type(feature[key]) is not int or feature[key] < minimum:
                    raise ValueError("invalid confirmation feature geometry")
            policies.features_from_cached(feature)
            result[entry["row_hash"]] = feature
            pins[str(path)] = file_digest(path)
    if len(contexts) != 1 or {p.stem for p in (directory / "features").glob("*.json")} != set(result):
        raise ValueError("mixed feature runtimes or unexpected confirmation request identities")
    return first_metadata, result, entries, pins


def build_exact_schedule(study, model, selection, metadata, feature_rows, entries, gate):
    """Pure answer-free counts: selected policy, full, exact-total control, statics."""
    _gate_check(study, selection, gate)
    by_id = {entry["row_hash"]: entry for entry in entries}
    expected = {request for requests in gate["cohort"].values() for request in requests}
    if len(by_id) != len(entries) or set(feature_rows) != set(by_id) or set(by_id) != expected:
        raise ValueError("exact confirmation schedule requires the entire predeclared cohort")
    bindings = _bindings(study, selection, sorted({entry["dataset"] for entry in entries}))
    if bindings != gate["bindings"]:
        raise ValueError("selected policies or static retention levels changed after gate")
    grouped = defaultdict(list)
    for request, feature in sorted(feature_rows.items()):
        entry = by_id[request]
        if set(feature) != FEATURE_FIELDS or feature.get("answers_consumed_by_feature_extractor") is not False:
            raise ValueError("scheduling accepts only audited answer-free cached features")
        if any(feature.get(k) != entry[k] for k in ("group_id", "dataset", "image_sha256")):
            raise ValueError("feature differs from fresh identity manifest")
        grouped[entry["dataset"]].append({**matrix.planning._geometry(feature),
            "benefit_features": policies.features_from_cached(feature),
            "quality_metric": entry.get("quality_metric", entry.get("metric")), "category": entry.get("category") or None})
    rows, assignments, budgets = {}, {}, {}
    for dataset, group in sorted(grouped.items()):
        key = model + "|" + dataset
        choice, binding = selection["domains"][key], bindings[key]
        if choice is None:
            assigned = {"counts": {r["row_hash"]: r["input_visual_tokens"] for r in group},
                "rows": {r["row_hash"]: {"fallback": True, "fallback_reasons": ["no_selected_compressed_policy"]} for r in group}}
        else:
            assigned = policies.predict_counts(choice["policy"], group)
        total = sum(assigned["counts"].values())
        uniform = policies.uniform_at_visual_total(group, total)
        if not uniform["feasible"] or uniform["token_error"] != 0:
            raise ValueError("exact confirmation uniform total is infeasible")
        budgets[dataset] = {"adaptive_visual_tokens": total, "matched_uniform_visual_tokens": uniform["total_visual_tokens"],
                            "token_error": 0, "full_visual_tokens": sum(r["input_visual_tokens"] for r in group)}
        assignments[dataset] = assigned
        for row in group:
            request, n = row["row_hash"], row["input_visual_tokens"]
            cells = [{"key": "full", "selector": "full", "selected_count": n, "mode": "full", "policy": "full", "full_fallback": False}]
            for method, count in (("adaptive", assigned["counts"][request]), ("matched_uniform", uniform["counts"][request])):
                cells.append({"key": method, "selector": "full" if count == n else "scorer", "selected_count": count,
                    "mode": method, "policy": "frozen_development_selection" if method == "adaptive" else "exact_visual_total",
                    "policy_sha256": binding["policy_sha256"] if method == "adaptive" else None,
                    "full_fallback": bool(assigned["rows"][request].get("fallback")) if method == "adaptive" else False})
            for selector, keep in binding["static_keep_percent"].items():
                count = policies.physical_count_menu(n)[keep]
                cells.append({"key": "static_" + selector, "selector": "full" if count == n else selector,
                    "selected_count": count, "keep_percent": keep, "mode": "uniform", "policy": "frozen_screening_static",
                    "full_fallback": False})
            rows[request] = {**matrix.planning._geometry(feature_rows[request]), "cells": cells}
    return seal({"format": FORMAT + "_schedule", "stage": "confirmation", "study_sha256": study["sha256"],
        "model": model, "split": "evaluation", "candidate": "current_rank32", "manifest_sha256": gate["manifest_sha256"],
        "checkpoint_sha256": metadata["checkpoint_sha256"], "cache_contract": metadata["cache_contract"],
        "features_sha256": digest(feature_rows), "selection_sha256": selection["sha256"], "gate_sha256": gate["sha256"],
        "answers_used_to_choose_counts": False, "confirmation_labels_used": False,
        "assignments": assignments, "budgets": budgets, "rows": rows})


def _publish_features(study, model, gate):
    from .worker import read_plan, verify_sources
    root = Path(study["output_root"])
    plan = build_feature_plan(study, model, gate)
    gate_path = root / "confirmation/gate.json"
    plan = seal({**{k: v for k, v in plan.items() if k != "sha256"},
                 "input_hashes": {**plan["input_hashes"], str(gate_path): file_digest(gate_path)}})
    path = root / f"plans/confirmation-features-{model}/plan.json"
    write_immutable(path, plan)
    verify_sources(read_plan(path))
    return path


def _publish_physical(study, model, selection, gate, collected):
    metadata, feature_rows, entries, feature_pins = collected
    root = Path(study["output_root"])
    master = build_exact_schedule(study, model, selection, metadata, feature_rows, entries, gate)
    paths = {name: root / model / "schedules" / ("confirmation" + suffix + ".json")
             for name, suffix in (("master", ""), ("execution", "-execution"), ("reuse", "-reuse"))}
    # Fresh observations are never substituted with historical physical labels.
    execution, reuse = matrix.split_reuse(master, {})
    for name, value in (("master", master), ("execution", execution), ("reuse", reuse)):
        write_immutable(paths[name], value)
    # The existing physical publisher pins every extra path, while preserving its
    # exact runtime arguments. Features, gate, policy bundles and source pins are
    # therefore checked by workers as well as by the final CPU collector.
    extra = {**gate["input_hashes"], **feature_pins,
             str(root / "confirmation/gate.json"): file_digest(root / "confirmation/gate.json")}
    paths.update({"confirmation_pin_" + str(i): Path(path) for i, path in enumerate(sorted(extra))})
    _verify_pins(extra)
    return matrix.publish_execution(study, model, "confirmation", "evaluation", paths, execution,
        feature_root=root / model / "confirmation-features", manifest=gate["manifest_path"])


def _report(study, selection, gate, measured):
    root = Path(study["output_root"])
    expected = {(model, dataset): requests for model in matrix.MODELS for dataset, requests in gate["cohort"].items()}
    rows = [row for model in matrix.MODELS for row in measured[model]]
    # Matrix collection guarantees complete physical cells; reporting rechecks
    # every paired cohort and native MME pair/category geometry independently.
    confirmation_audit = {key: True for key in reporting.CONFIRMATION_CHECKS}
    report = reporting.summarize(rows, evidence="confirmation", expected_cohorts=expected,
                                 confirmation_audit=confirmation_audit)
    expected_panels = {(model, dataset, method) for model in matrix.MODELS for dataset in gate["cohort"]
                       for method in ("full", "adaptive", "matched_uniform", *("static_" + s for s in _methods(model)))}
    if ({(p["model"], p["dataset"], p["method"]) for p in report["panels"]} != expected_panels
            or any(p.get("cohort_complete") is not True for p in report["panels"])):
        raise ValueError("confirmation report lacks the exact frozen methods/cohorts")
    report_path = root / "reports/confirmation.json"
    write_immutable(report_path, report)
    schedules = {}
    for model in matrix.MODELS:
        path = root / model / "schedules/confirmation.json"
        schedules[model] = {"path": str(path), "sha256": file_digest(path), "content_sha256": matrix.checked(path)["sha256"]}
    result_path = root / "confirmation/result.json"
    result = seal({"format": FORMAT + "_result", "status": "complete", "complete": True,
        "study_sha256": study["sha256"], "selection_sha256": selection["sha256"], "gate_sha256": gate["sha256"],
        "gate_path": str(root / "confirmation/gate.json"), "manifest_path": gate["manifest_path"],
        "manifest_sha256": gate["manifest_sha256"], "report_path": str(report_path),
        "report_sha256": file_digest(report_path), "result_path": str(result_path), "schedules": schedules,
        "confirmation_audit": confirmation_audit})
    write_immutable(result_path, result)
    return result


def advance(study, selection, audit):
    """Advance only admitted stages; return blocked/pending/complete sealed state.

    This is the sole mutating entrypoint. It publishes immutable local plans and
    reports only; the existing scheduler owns execution and the deadline.
    """
    reasons = admission_reasons(study, selection, audit)
    if reasons:
        return seal({"format": FORMAT + "_status", "status": "blocked", "complete": False,
            "fresh_identity_audit_passed": False, "blocking_reasons": reasons, "plans": {},
            "study_sha256": study["sha256"], "audit_sha256": audit.get("sha256") if isinstance(audit, dict) else None})
    matrix.verify(study)
    root = Path(study["output_root"])
    expired = time.time() >= study["deadline_utc_epoch"]
    if expired and not (root / "confirmation/gate.json").exists():
        return seal({"format": FORMAT + "_status", "status": "deadline_expired", "complete": False, "plans": {}})
    gate = freeze_gate(study, selection, audit)
    plans, measured = {}, {}
    for model in matrix.MODELS:
        feature_plan = root / f"plans/confirmation-features-{model}/plan.json"
        if not expired or feature_plan.exists():
            plans["confirmation-features-" + model] = str(_publish_features(study, model, gate))
        collected = collect_features(study, model, gate)
        if collected is None:
            continue
        physical_plan = root / f"plans/confirmation-{model}/plan.json"
        if not expired or physical_plan.exists():
            plans["confirmation-" + model] = str(_publish_physical(study, model, selection, gate, collected))
            result = matrix.collect(study, model, "confirmation")
            if result is not None:
                measured[model] = result
    if len(measured) == len(matrix.MODELS):
        return _report(study, selection, gate, measured)
    return seal({"format": FORMAT + "_status", "status": "deadline_expired" if expired else "confirmation_pending",
        "complete": False, "study_sha256": study["sha256"], "selection_sha256": selection["sha256"],
        "gate_sha256": gate["sha256"], "fresh_identity_audit_passed": True, "plans": plans,
        "completed_models": sorted(measured)})
