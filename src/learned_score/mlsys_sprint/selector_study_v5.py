"""Bounded, restartable selector/allocator study with screening-only selection.

Historical outputs are read-only. Missing tasks or failed candidates never
become wins. A 24-hour deadline stops new work; it does not make a study complete.
"""

import argparse
import copy
from collections import OrderedDict
import fcntl
import json
from pathlib import Path
import time

from .state import atomic_json, digest, file_digest
from .followup_common import write_immutable

ROOT = Path("reproduction_results/mlsys_selector_20261017_v5")
OLD = Path("reproduction_results/mlsys_adaptive_20261016_v4")
MODELS = ("qwen25", "llava_next", "phi35")
INITIAL = MODELS[:2]
RECIPES = ("saliency_control", "query_saliency", "answer_soft", "query_answer_soft", "answer_hard")
CANDIDATES = ("current_rank32", *RECIPES, "rank32_coverage")
WORLD = 8
TARGETS = (40, 50, 60, 70)
_REPORT_CACHE = OrderedDict()
_QUERY_CACHE = {}


def cached_report(rows, **kwargs):
    """Do not bootstrap identical completed physical observations every tick."""
    from .selector_report_v5 import report
    key = digest({"rows": rows, "options": kwargs})
    if key not in _REPORT_CACHE:
        _REPORT_CACHE[key] = report(rows, **kwargs)
        if len(_REPORT_CACHE) > 64:
            _REPORT_CACHE.popitem(last=False)
    else:
        _REPORT_CACHE.move_to_end(key)
    return copy.deepcopy(_REPORT_CACHE[key])


def read(path):
    return json.loads(Path(path).read_text())


def seal(value):
    return {**value, "sha256": digest(value)}


def checked(path):
    value = read(path)
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"sealed artifact changed: {path}")
    return value


def source_hashes():
    names = ("selector_study_v5.py", "selector_experiment_v5.py", "selector_models_v5.py",
             "selector_training_v5.py", "selector_controls_v5.py", "baseline_audit_v5.py",
             "selector_allocation_study_v5.py", "selector_report_v5.py", "selector_operations_v5.py")
    return {str(Path(__file__).with_name(name)): file_digest(Path(__file__).with_name(name)) for name in names}


def training_manifest(original, development):
    """Reuse the previously fixed 256+256 official-train pilot, without answers."""
    from learned_score.mlsys_eval.common import seal_manifest
    from .adaptive_data_v3 import _split_keys, dataset_family
    original_rows = {r["row_hash"]: r for r in original["splits"]["scorer_train"]}
    chosen = sorted(development["splits"]["phi_train"], key=lambda r: r["row_hash"])
    forbidden = set()
    for split in ("fit", "screening", "evaluation"):
        for row in development["splits"][split]:
            forbidden.update(_split_keys(row))
    if len(chosen) != 512 or len({r["row_hash"] for r in chosen}) != 512:
        raise ValueError("pilot must contain 512 unique training requests")
    counts = {}
    rows = []
    for entry in chosen:
        row = original_rows.get(entry["row_hash"])
        if row is None or row["source_split"] != "train" or _split_keys(row) & forbidden:
            raise ValueError("pilot is not original official training, or overlaps development")
        family = dataset_family(row["dataset"])
        counts[family] = counts.get(family, 0) + 1
        rows.append(dict(row))
    if counts != {"DocVQA": 256, "TextVQA": 256}:
        raise ValueError("pilot requires exactly 256 official examples from each training dataset")
    return seal_manifest({"format": "mlsys_selector_training_manifest_v5", "dataset": "mixed",
        "metadata": {**original.get("metadata", {}), "pilot_selection": "previous identity-only phi_train512",
                     "training_rows": 512, "development_only": True, "fresh_confirmation": False},
        "splits": {"scorer_train": rows, **{key: development["splits"][key]
                    for key in ("fit", "screening", "evaluation")}}})


def plan_names():
    names = [f"train-{m}" for m in INITIAL]
    names += [f"screen-{m}-{c}" for m in INITIAL for c in CANDIDATES]
    names += [f"allocation-{split}-{m}" for split in ("screening", "evaluation") for m in MODELS]
    names += [f"replication-{m}" for m in MODELS]
    names += [f"replication-screen-{m}" for m in MODELS]
    names += [f"final-{m}-{role}" for m in MODELS for role in ("current_rank32", "saliency_control", "winner")]
    names += [f"query-{m}-{role}" for m in INITIAL for role in ("current_rank32", "query_saliency")]
    return names


def prepare(cwd):
    from .selector_operations_v5 import build_config
    from learned_score.mlsys_scorer.protocol import load_manifest
    cwd = Path(cwd).resolve()
    output = cwd / ROOT
    if (output / "study.json").exists():
        study = checked(output / "study.json")
        verify_study(study)
        return study
    old = checked(cwd / OLD / "study.json")
    old_manifest = read(old["manifest"])
    original_path = cwd / "reproduction_results/mlsys_20261012/manifest.json"
    manifest_path = output / "data/training512.json"
    write_immutable(manifest_path, training_manifest(read(original_path), old_manifest))
    load_manifest(manifest_path, expected_training_count=512)
    from .baseline_audit_v5 import run_audit
    audit_path = output / "audits/divprune_rule.json"
    write_immutable(audit_path, run_audit())
    now = int(time.time())
    study = seal({"format": "mlsys_selector_study_v5", "cwd": str(cwd), "output_root": str(output),
        "created_utc_epoch": now, "deadline_utc_epoch": now + 86400, "models": old["models"],
        "manifest": old["manifest"], "manifest_sha256": old["manifest_sha256"],
        "training_manifest": str(manifest_path), "training_manifest_sha256": file_digest(manifest_path),
        "original_training_manifest": str(original_path), "old_study": str(cwd / OLD / "study.json"),
        "baseline_audit": str(audit_path), "baseline_audit_sha256": file_digest(audit_path),
        "old_study_sha256": file_digest(cwd / OLD / "study.json"), "world_size": WORLD,
        "candidates": list(CANDIDATES), "screening_targets": [60, 70], "targets": list(TARGETS),
        "allocation_ranker": "current_rank32", "allocation_statistic": "normalized_entropy",
        "allocation_step_pp": [1, 5], "operating_retention_margin": .005, "risk_penalty": .5,
        "refinement_seed": 42, "replication_seed": 43, "source_sha256": source_hashes(),
        "selection": {"mean_gain_over_frozen_and_control_each_model_pp": 0,
                      "model_balanced_mean_gain_over_frozen_pp": .5,
                      "worst_panel_gain_over_both_pp": -1,
                      "tie_break": ["worst_panel", "selector_seconds", "lexical_candidate"],
                      "labels": "screening only; no forced winner"},
        "evidence_status": "development; reused historical splits, not fresh confirmation",
        "training_disclosure": "Qwen/LLaVA warmstarts saw5600 rows; Phi warmstart saw512; all refinements same512",
        "baseline_scope": "audited DivPrune and reference-CLIP CDPruner selector adaptations, not official pipelines",
        "cache_contract": "exact cohort allocated KV bytes, original protected prompt and image positions, reserve127",
        "text_eviction_claim": False})
    write_immutable(output / "study.json", study)
    for role in ("clusterb", "clustera"):
        # Only LLaVA has validated cross-cluster parity. Qwen and Phi remain on
        # ClusterB until a separate physical parity audit authorizes ClusterA.
        names = plan_names() if role == "clusterb" else [n for n in plan_names() if "llava_next" in n]
        config = build_config(cwd, [output / f"plans/{n}/plan.json" for n in names],
            study["deadline_utc_epoch"], role=role, created_utc_epoch=now)
        write_immutable(output / f"coordinator/{role}-config.json", config)
    return study


def verify_study(study):
    pins = {**study["source_sha256"], study["manifest"]: study["manifest_sha256"],
            study["training_manifest"]: study["training_manifest_sha256"],
            study["old_study"]: study["old_study_sha256"], study["baseline_audit"]: study["baseline_audit_sha256"]}
    for path, expected in pins.items():
        if file_digest(path) != expected:
            raise ValueError(f"frozen study source/input changed: {path}")


def publish(study, name, tasks, inputs=()):
    from .worker import read_plan, verify_sources
    path = Path(study["output_root"]) / f"plans/{name}/plan.json"
    if path.exists():
        verify_sources(read_plan(path))
        return path
    hashes = dict(study["source_sha256"])
    if name.startswith(("train-", "replication-")):
        from .selector_training_v5 import source_hashes as training_sources
        hashes.update(training_sources())
    for item in inputs:
        hashes[str(item)] = file_digest(item)
    plan = seal({"format": "mlsys_selector_stage_v5", "cwd": study["cwd"], "tasks": tasks,
                 "input_hashes": hashes, "unexpected_retries": 1,
                 "created_utc_epoch": time.time(), "deadline_utc_epoch": study["deadline_utc_epoch"]})
    write_immutable(path, plan)
    verify_sources(read_plan(path))
    return path


def task(study, name, argv, artifacts, *, resume=None):
    from .adaptive_workflow_v4 import environment
    result = {"id": "selector-v5-" + name.replace("_", "-"), "tag": "selector-v5", "resource": "gpu",
              "minimum_start_seconds": 300, "depends_on": [], "argv": argv,
              "env": environment(Path(study["cwd"])), "artifacts": [{"path": str(p)} for p in artifacts]}
    if resume:
        result.update(resume_checkpoint=str(resume), resume_args=["--resume"])
    return result


def python(study, module):
    return [str(Path(study["cwd"]) / ".conda/asymvlm/bin/python"), "-u", "-m", "learned_score.mlsys_sprint." + module]


def training_dir(study, model, recipe, seed=42):
    return Path(study["output_root"]) / model / "training" / f"{recipe}-seed{seed}"


def trained(study, model, recipe, seed=42):
    directory = training_dir(study, model, recipe, seed)
    if not (directory / "complete.json").exists():
        return False
    marker = read(directory / "complete.json")
    if marker.get("steps") != 512 or marker.get("checkpoint_sha256") != file_digest(directory / "latest.pt"):
        raise ValueError("training completion does not match all512 steps and checkpoint")
    return True


def checkpoint_path(study, model, candidate, seed=42):
    if candidate in ("current_rank32", "rank32_coverage"):
        return Path(study["models"][model]["checkpoint"])
    return training_dir(study, model, candidate, seed) / "latest.pt"


def training_task(study, model, recipe, seed=42):
    spec = study["models"][model]
    output = training_dir(study, model, recipe, seed)
    argv = python(study, "selector_training_v5") + ["--manifest", study["training_manifest"],
        "--warmstart", spec["checkpoint"], "--recipe", recipe, "--output", str(output),
        "--adapter", spec["adapter"], "--model", spec["model_path"], "--model-revision", spec["revision"],
        "--seed", str(seed)]
    if "saliency" in recipe and model in INITIAL:
        branch = "training_v1" if model == "qwen25" else "llava_memory_v2"
        cache = Path(study["cwd"]) / f"reproduction_results/mlsys_20261012/{branch}/{model}/saliency"
        argv += ["--cache", str(cache), "--cache-manifest", study["original_training_manifest"]]
    # Trainer is automatically restartable; no unsupported --resume flag.
    return task(study, f"train-{model}-{recipe}-{seed}", argv, [output / "complete.json", output / "latest.pt"])


def selector_schedule(study, model, candidate, split, targets, *, seed=42, role=None):
    from . import adaptive_planning_v4 as planning
    path = Path(study["output_root"]) / model / "schedules" / f"{role or split}-{candidate}-seed{seed}.json"
    if path.exists():
        saved = checked(path)
        expected = {"study_sha256": study["sha256"], "candidate": candidate, "model": model,
                    "split": split, "seed": seed,
                    "checkpoint_sha256": file_digest(checkpoint_path(study, model, candidate, seed))}
        if (any(saved.get(key) != value for key, value in expected.items())
                or set(saved["budgets"]) != {str(t) for t in targets}):
            raise ValueError("existing selector schedule has a different experiment contract")
        return path
    old = checked(study["old_study"])
    _, _, _, metadata, features = planning._features(old, model, split)
    budgets, infeasible = planning._budgets(features, targets, metadata["cache_contract"])
    rows = {}
    for request, feature in sorted(features.items()):
        cells = []
        if candidate == "current_rank32":
            cells.append({"key": "all_tokens", "selector": "full", "selected_count": feature["input_visual_tokens"],
                          "mode": "full", "target": None, "policy": "none"})
        for target in targets:
            budget = budgets[str(target)][feature["dataset"]]
            if not budget["feasible"]:
                continue
            selectors = ["scorer"]
            if candidate == "current_rank32":
                selectors += ["divprune"] + (["cdpruner"] if model == "llava_next" else [])
            for selector in selectors:
                cells.append({"key": f"{selector}_uniform_t{target}", "selector": selector,
                    "selected_count": budget["counts"][request], "target": target, "policy": "uniform",
                    "mode": "uniform"})
        if cells:
            rows[request] = {**planning._geometry(feature), "cells": cells}
    payload = seal({"format": "mlsys_selector_schedule_v5", "candidate": candidate, "model": model,
        "split": split, "seed": seed, "manifest_sha256": study["manifest_sha256"],
        "checkpoint_sha256": file_digest(checkpoint_path(study, model, candidate, seed)),
        "cache_contract": metadata["cache_contract"], "rows": rows, "budgets": budgets,
        "infeasible_panels": infeasible, "features_sha256": digest(features),
        "study_sha256": study["sha256"], "answers_consumed": False})
    write_immutable(path, payload)
    return path


def publish_physical(study, name, model, candidate, split, schedule_path, destination, *, seed=42,
                     query_control="correct"):
    spec = study["models"][model]
    checkpoint = checkpoint_path(study, model, candidate, seed)
    tasks = []
    for rank in range(WORLD):
        argv = python(study, "selector_experiment_v5") + ["--phase", "physical", "--candidate", candidate,
            "--warmstart", spec["checkpoint"], "--checkpoint", str(checkpoint), "--split", split,
            "--manifest", study["manifest"], "--schedule", str(schedule_path), "--adapter", spec["adapter"],
            "--model", spec["model_path"], "--model-revision", spec["revision"], "--output", str(destination),
            "--rank", str(rank), "--world-size", str(WORLD), "--query-control", query_control,
            "--feature-root", str(Path(study["cwd"]) / OLD / model / f"{split}_features")]
        if candidate == "rank32_coverage":
            argv.append("--coverage")
        if model == "llava_next":
            # Equal reference-model residency across all compared LLaVA cells.
            argv.append("--reference-clip")
        tasks.append(task(study, f"{name}-{rank}", argv, [destination / f"rank{rank}.complete.json"],
                          resume=destination / f"rank{rank}.metadata.json"))
    from .selector_experiment_v5 import source_hashes as runtime_sources
    inputs = [Path(study["cwd"]) / "learned_score" / name for name in runtime_sources(spec["adapter"])]
    inputs += [Path(study["manifest"]), checkpoint, Path(spec["checkpoint"]), Path(schedule_path)]
    publish(study, name, tasks, inputs)


def scan_physical(study, model, candidate, directory, schedule_path):
    """Audit completed cohorts only; partial shards never imply matched budgets."""
    from .adaptive_report_v4 import audit_cache, audit_stopping
    from .selector_experiment_v5 import source_hashes as runtime_sources
    schedule = checked(schedule_path)
    if not all((directory / f"rank{rank}.complete.json").is_file() for rank in range(WORLD)):
        return None
    expected = {(request, cell["key"]): (cell, row) for request, row in schedule["rows"].items() for cell in row["cells"]}
    known = {}
    sources = runtime_sources(study["models"][model]["adapter"])
    ordered = sorted(schedule["rows"], key=lambda request: (schedule["rows"][request]["dataset"], request))
    for rank in range(WORLD):
        meta = read(directory / f"rank{rank}.metadata.json")
        marker = read(directory / f"rank{rank}.complete.json")
        assigned = ordered[rank::WORLD]
        results = sum(len(schedule["rows"][r]["cells"]) for r in assigned)
        if (meta.get("format") != "mlsys_selector_experiment_v5"
                or meta.get("schedule_content_sha256") != schedule["sha256"]
                or meta.get("checkpoint_sha256") != schedule["checkpoint_sha256"]
                or meta.get("manifest_sha256") != study["manifest_sha256"]
                or meta.get("source_sha256") != sources
                or meta.get("backbone", {}).get("selector_v5", {}).get("candidate") != candidate
                or meta.get("backbone", {}).get("selector_v5", {}).get("warmstart_sha256")
                   != file_digest(study["models"][model]["checkpoint"])
                or marker.get("identity_sha256") != digest(meta)
                or marker.get("requests") != len(assigned) or marker.get("results") != results):
            raise ValueError("physical completion does not match its exact scorer/schedule/shard")
        for request in assigned:
            known[request] = meta
    found = {}
    for path in sorted((directory / "rows").glob("*.json")):
        row = read(path)
        key = row.get("request_id"), row.get("method")
        if key not in expected or key in found:
            raise ValueError("unexpected/duplicate physical record")
        cell, planned = expected[key]
        meta = known[key[0]]
        if (row.get("status") != "ok" or row.get("identity_sha256") != digest(meta)
                or row.get("schedule_cell") != cell or row.get("schedule_content_sha256") != schedule["sha256"]):
            raise ValueError("physical record changed or did not complete successfully")
        audit_cache(row, schedule["cache_contract"], cell, planned)
        audit_stopping(row, meta, is_phi=model == "phi35")
        found[key] = {**row, "candidate": candidate, "model_tag": model}
    if set(found) != set(expected):
        raise ValueError("completion markers exist but physical result matrix is incomplete")
    return list(found.values())


def expected_panels(study):
    from .adaptive_data_v3 import dataset_family
    result = []
    for model in INITIAL:
        path = Path(study["output_root"]) / model / "schedules/screening-current_rank32-seed42.json"
        for target, groups in checked(path)["budgets"].items():
            for dataset, budget in groups.items():
                if budget["feasible"]:
                    result.append((model, dataset_family(dataset), int(target)))
    return result


def tick(study):
    from .selector_allocation_study_v5 import build_schedules
    from .selector_report_v5 import report
    from .adaptive_report_v1 import _atomic_text
    verify_study(study)
    output = Path(study["output_root"])
    old = checked(study["old_study"])
    for model in INITIAL:
        publish(study, f"train-{model}", [training_task(study, model, recipe) for recipe in RECIPES],
                [study["training_manifest"], study["models"][model]["checkpoint"]])
    all_rows, screening_complete = [], True
    for model in INITIAL:
        for candidate in CANDIDATES:
            if candidate in RECIPES and not trained(study, model, candidate):
                screening_complete = False
                continue
            name = f"screen-{model}-{candidate}"
            schedule = selector_schedule(study, model, candidate, "screening", (60, 70))
            directory = output / model / "screening" / candidate
            publish_physical(study, name, model, candidate, "screening", schedule, directory)
            rows = scan_physical(study, model, candidate, directory, schedule)
            screening_complete &= rows is not None
            all_rows.extend(rows or [])
    # Allocation is deliberately independent of scorer selection and always runs.
    allocation_rows, allocation_complete = [], True
    for split in ("screening", "evaluation"):
        for model in MODELS:
            build_schedules(old, study, model, split)
            schedule = output / model / "schedules" / f"allocation-{split}.json"
            directory = output / model / f"allocation-{split}" / "current_rank32"
            publish_physical(study, f"allocation-{split}-{model}", model, "current_rank32", split, schedule, directory)
            rows = scan_physical(study, model, "current_rank32", directory, schedule)
            allocation_complete &= rows is not None
            if rows:
                allocation_rows.extend({**row, "allocation_split": split} for row in rows)
    screening = cached_report(all_rows, complete=screening_complete, expected_panels=expected_panels(study), stage="screening")
    atomic_json(output / "reports/screening.json", screening)
    _atomic_text(output / "reports/selector_table.md", screening["tables_markdown"])
    # Never pool screening and evaluation requests into one allocation estimate.
    for split in ("screening", "evaluation"):
        subset = [r for r in allocation_rows if r["allocation_split"] == split]
        allocation_report = cached_report(subset, complete=allocation_complete, stage="allocation")
        atomic_json(output / f"reports/allocation-{split}.json", allocation_report)
        _atomic_text(output / f"reports/allocation-{split}.md", allocation_report["tables_markdown"])
    # Query diagnostics are independently predeclared, not conditional on
    # finding a winning recipe and never inputs to candidate selection.
    query_complete = advance_query_diagnostics(study)
    decision_path = output / "selection.json"
    if screening_complete:
        if screening["selector_decision"]["status"] not in {"selected_for_development", "no_candidate_qualified"}:
            raise ValueError("all screening shards arrived, but the selection audit did not pass")
        write_immutable(decision_path, seal({"format": "mlsys_selector_selection_v5", "study_sha256": study["sha256"],
            "decision": screening["selector_decision"], "screening_report_sha256": digest(screening),
            "evaluation_labels_used": False}))
    decision = checked(decision_path)["decision"] if decision_path.exists() else None
    downstream_complete = advance_selected(study, decision) if decision is not None else False
    status = {"time": time.time(), "screening_complete": screening_complete,
        "allocation_complete": allocation_complete, "downstream_complete": downstream_complete,
        "query_diagnostics_complete": query_complete,
        "decision": decision, "complete": screening_complete and allocation_complete and downstream_complete and query_complete,
        "deadline_utc_epoch": study["deadline_utc_epoch"], "evidence_status": study["evidence_status"]}
    atomic_json(output / "coordinator/status.json", status)
    return status


def advance_query_diagnostics(study):
    """Always compare correct and shuffled queries, without selecting on it."""
    from .selector_report_v5 import report
    output = Path(study["output_root"])
    complete = True
    for model in INITIAL:
        for candidate in ("current_rank32", "query_saliency"):
            if candidate == "query_saliency" and not trained(study, model, candidate):
                complete = False
                continue
            schedule = selector_schedule(study, model, candidate, "screening", (60, 70))
            directory = output / model / "query-shuffled" / candidate
            publish_physical(study, f"query-{model}-{candidate}", model, candidate, "screening", schedule, directory,
                             query_control="shuffled")
            rows = scan_physical(study, model, candidate, directory, schedule)
            complete &= rows is not None
            if rows is not None:
                result = cached_report(rows, complete=True, stage="query_diagnostic")
                correct = scan_physical(study, model, candidate, output / model / "screening" / candidate, schedule)
                complete &= correct is not None
                result.update(diagnostic_only=True, candidate_selection_uses_this_report=False,
                              original_query_used_for_generation=True,
                              paired_correct_query_complete=correct is not None,
                              paired_query_effects=query_effects(correct, rows) if correct is not None else [])
                atomic_json(output / "reports" / f"query-shuffled-{model}-{candidate}.json", result)
    return complete


def query_effects(correct, shuffled, *, repetitions=2000):
    """Paired quality and patch-set diagnostics; never candidate-selection input."""
    from collections import defaultdict
    from itertools import combinations
    from statistics import mean
    from .selector_report_v5 import _compare
    cache_key = digest({"correct": correct, "shuffled": shuffled, "repetitions": repetitions})
    if cache_key in _QUERY_CACHE:
        return copy.deepcopy(_QUERY_CACHE[cache_key])
    groups = defaultdict(lambda: [[], []])
    for side, rows in enumerate((correct, shuffled)):
        for row in rows:
            if row["selector"] == "scorer":
                groups[(row["dataset"], row["target"])][side].append(row)
    effects = []
    for (dataset, target), (original, replaced) in sorted(groups.items()):
        comparison = _compare(original, replaced, same_counts=True, repetitions=repetitions)
        a, b = ({r["request_id"]: r for r in rows} for rows in (original, replaced))
        overlap, image_groups = [], defaultdict(list)
        for request, row in a.items():
            if request in b:
                first, second = (set(item["selected_patch_indices"]) for item in (row, b[request]))
                overlap.append(len(first & second) / len(first | second))
            image_groups[row["image_sha256"]].append(row)
        same_image = []
        for rows in image_groups.values():
            for first, second in combinations(rows, 2):
                if first["input_visual_tokens"] != second["input_visual_tokens"]:
                    raise ValueError("same image has incompatible physical patch geometry")
                left, right = set(first["selected_patch_indices"]), set(second["selected_patch_indices"])
                same_image.append(len(left & right) / len(left | right))
        effects.append({"dataset": dataset, "target": target,
            "effect_direction": "correct scorer query minus shuffled scorer query; generation query unchanged",
            "paired_quality": comparison,
            "correct_vs_shuffled_patch_jaccard_mean": mean(overlap) if overlap else None,
            "same_image_different_query_pair_count": len(same_image),
            "same_image_different_query_patch_jaccard_mean": mean(same_image) if same_image else None})
    _QUERY_CACHE[cache_key] = effects
    return copy.deepcopy(effects)


def advance_selected(study, decision):
    """Conditional refinement replication and fixed-winner development transfer."""
    from .selector_report_v5 import report
    from .adaptive_report_v1 import _atomic_text
    output = Path(study["output_root"])
    winner = decision.get("selected_candidate")
    pending_names = [n for n in plan_names() if n.startswith(("replication-", "final-"))]
    if winner is None:
        for name in pending_names:
            publish(study, name, [])
        return True
    if winner not in CANDIDATES or winner in ("current_rank32", "saliency_control"):
        raise ValueError("screening returned an invalid improvement candidate")
    complete, final_rows = True, []
    for model in MODELS:
        recipes = [(winner, 43)] if model in INITIAL and winner in RECIPES else []
        if model == "phi35":
            recipes = [("saliency_control", 42)] + ([(winner, 42)] if winner in RECIPES else [])
        publish(study, f"replication-{model}", [training_task(study, model, r, seed) for r, seed in recipes],
                [study["training_manifest"], study["models"][model]["checkpoint"]])
        ready = all(trained(study, model, r, seed) for r, seed in recipes)
        if not ready:
            complete = False
        # Replication is reported, not used to retune or silently replace the winner.
        if ready:
            seed = 43 if model in INITIAL and winner in RECIPES else 42
            schedule = selector_schedule(study, model, winner, "screening", (60, 70), seed=seed, role="replication")
            directory = output / model / "replication" / winner
            publish_physical(study, f"replication-screen-{model}", model, winner, "screening", schedule, directory, seed=seed)
            rows = scan_physical(study, model, winner, directory, schedule)
            complete &= rows is not None
            if rows is not None:
                atomic_json(output / "reports" / f"replication-{model}.json",
                            cached_report(rows, complete=True, stage="replication"))
        for role, candidate in (("current_rank32", "current_rank32"), ("saliency_control", "saliency_control"), ("winner", winner)):
            if candidate in RECIPES and not trained(study, model, candidate):
                complete = False
                continue
            schedule = selector_schedule(study, model, candidate, "evaluation", TARGETS, role="final")
            directory = output / model / "evaluation" / candidate
            publish_physical(study, f"final-{model}-{role}", model, candidate, "evaluation", schedule, directory)
            rows = scan_physical(study, model, candidate, directory, schedule)
            complete &= rows is not None
            final_rows.extend(rows or [])
    summary = cached_report(final_rows, complete=complete, stage="evaluation")
    summary["frozen_screening_decision"] = decision
    atomic_json(output / "reports/evaluation.json", summary)
    _atomic_text(output / "reports/evaluation.md", summary["tables_markdown"])
    return complete


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args(argv)
    study = prepare(args.cwd)
    if args.prepare:
        print(json.dumps(study), flush=True)
        return
    output = Path(study["output_root"])
    with (output / "coordinator/workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study["deadline_utc_epoch"]:
            try:
                status = tick(study)
            except Exception as error:
                atomic_json(output / "coordinator/error.json", {"time": time.time(), "error": str(error), "type": type(error).__name__})
                raise
            print(json.dumps(status), flush=True)
            if args.once or status["complete"]:
                return
            time.sleep(min(30, max(0, study["deadline_utc_epoch"] - time.time())))
        atomic_json(output / "coordinator/deadline.json", {"complete": False, "reason": "24-hour study window expired"})


if __name__ == "__main__":
    main()
