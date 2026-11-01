"""Publish answer-free v5 budget-grid controls from frozen v4 calibration.

Only completed v4 feature metadata, features, and already fitted entropy
policies are consumed. No physical screening/evaluation result is opened and
no policy is refitted. New assignments and schedules are immutable artifacts.
"""

from collections import defaultdict
from pathlib import Path

from . import adaptive_allocation_v4 as allocation
from . import adaptive_planning_v4 as old
from . import selector_controls_v5 as controls
from .followup_common import write_immutable
from .state import digest, file_digest


FORMAT = "mlsys_selector_allocation_study_v5"
TARGETS = (40, 50, 60, 70)
STEPS = (1, 5)
SEEDS = (42, 43, 44)
CANDIDATE = "current_rank32"
STATISTIC = "normalized_entropy"
_VERIFIED_FILES = {}


def sources():
    return {**old._sources(), str(Path(__file__).resolve()): file_digest(__file__),
            str(Path(controls.__file__).resolve()): file_digest(controls.__file__)}


def _validate_studies(old_study, new_study, model, split):
    old._checked(old_study)
    old._checked(new_study)
    if split not in ("screening", "evaluation"):
        raise ValueError("allocation controls require screening or evaluation, never fit")
    if model not in old_study["models"] or model not in new_study["models"]:
        raise ValueError("unknown allocation study model")
    for field in ("manifest", "manifest_sha256"):
        if new_study[field] != old_study[field]:
            raise ValueError("allocation controls must use the audited old manifest")
    root = Path(new_study["output_root"]).resolve()
    previous = Path(old_study["output_root"]).resolve()
    if root == previous or previous in root.parents or root in previous.parents:
        raise ValueError("new allocation output must be separate from preserved v4 outputs")
    if tuple(old_study["kv_reduction_percent"]) != TARGETS:
        raise ValueError("old study does not use the four declared compression targets")
    if "kv_reduction_percent" in new_study and tuple(new_study["kv_reduction_percent"]) != TARGETS:
        raise ValueError("new study target contract changed")
    return root / model


def _study_context(old_study, new_study, model, split):
    output_root = _validate_studies(old_study, new_study, model, split)
    manifest, output, checkpoint, metadata, features = old._features(old_study, model, split)
    if file_digest(new_study["models"][model]["checkpoint"]) != checkpoint:
        raise ValueError("old entropy calibration requires the frozen current scorer checkpoint")
    index_path = output / "policies/fit_index.json"
    index = old._checked(old._read(index_path))
    expected = {"study_sha256": old_study["sha256"], "checkpoint_sha256": checkpoint,
                "manifest_sha256": old_study["manifest_sha256"], "model": model,
                "source_sha256": old._sources(), "learned": False,
                "labels_consumed": "fit physical responses only"}
    if any(index.get(key) != value for key, value in expected.items()):
        raise ValueError("frozen fit-policy index provenance differs")
    return output_root, index_path, index, metadata, features


def _verified_digest(path):
    """Hash on first use/change; unchanged inode/size/mtime/ctime is a CPU cache."""
    path = Path(path).resolve()
    stat = path.stat()
    signature = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    cached = _VERIFIED_FILES.get(str(path))
    if cached is not None and cached[0] == signature:
        return cached[1]
    value = file_digest(path)
    # Do not cache a hash of a file changed while being read.
    after = path.stat()
    if signature != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns):
        raise ValueError(f"allocation input changed during verification: {path}")
    _VERIFIED_FILES[str(path)] = (signature, value)
    return value


def _feature_inventory(root):
    root = Path(root)
    return sorted(str(path.resolve()) for pattern in ("rank*.metadata.json", "rank*.complete.json", "features/*.json")
                  for path in root.glob(pattern))


def _load_existing(path, old_study, new_study, model, split):
    schedule = old._checked(old._read(path))
    controls_meta = schedule.get("v5_controls", {})
    expected = {"format": old.FORMAT + "_schedule", "stage": f"allocation-{split}", "model": model,
                "split": split, "study_sha256": new_study["sha256"], "candidate": CANDIDATE,
                "manifest_sha256": new_study["manifest_sha256"], "source_sha256": sources()}
    if (any(schedule.get(key) != value for key, value in expected.items())
            or controls_meta.get("old_study_sha256") != old_study["sha256"]
            or controls_meta.get("budget_steps_pp") != list(STEPS)
            or controls_meta.get("operating_retention_margin") != .005
            or controls_meta.get("shuffle_seeds") != list(SEEDS)
            or controls_meta.get("quality_labels_opened") is not False):
        raise ValueError("existing allocation schedule study/source/control provenance changed")
    pins = controls_meta.get("input_file_sha256", {})
    feature_root = Path(old_study["output_root"]) / model / f"{split}_features"
    inventory = _feature_inventory(feature_root)
    checkpoint = str(Path(new_study["models"][model]["checkpoint"]).resolve())
    required = {checkpoint, str(Path(new_study["manifest"]).resolve()), controls_meta["fit_index_path"], *inventory}
    if (not pins or not required <= set(pins)
            or inventory != controls_meta.get("feature_file_inventory")
            or pins[checkpoint] != schedule["checkpoint_sha256"]
            or pins[str(Path(new_study["manifest"]).resolve())] != new_study["manifest_sha256"]):
        raise ValueError("existing allocation input inventory changed")
    for filename, expected_hash in pins.items():
        if _verified_digest(filename) != expected_hash:
            raise ValueError(f"frozen allocation input changed: {filename}")
    for artifact in schedule["assignment_artifacts"]:
        if _verified_digest(artifact["path"]) != artifact["file_sha256"]:
            raise ValueError("frozen allocation assignment changed")
    return schedule


def build_schedules(old_study, new_study, model, split):
    """Write immutable schedules/assignments and return the sealed schedule.

Output: ``new_root/model/schedules/allocation-SPLIT.json`` plus
``new_root/model/assignments/allocation-SPLIT/TARGET_DATASETDIGEST/``.
The v4 schedule wire format is retained for the existing physical executor.
"""
    output = _validate_studies(old_study, new_study, model, split)
    schedule_path = output / "schedules" / f"allocation-{split}.json"
    if schedule_path.exists():
        return _load_existing(schedule_path, old_study, new_study, model, split)
    output, index_path, index, metadata, features = _study_context(old_study, new_study, model, split)
    index_path = index_path.resolve()
    inventory = _feature_inventory(Path(old_study["output_root"]) / model / f"{split}_features")
    policy_files = [record["path"] for record in index["policies"]
                    if record["candidate"] == STATISTIC and record["target"] in TARGETS
                    and record["held_out_benchmark"] is None]
    input_paths = {str(Path(path).resolve()) for path in (
        new_study["manifest"], new_study["models"][model]["checkpoint"], index_path, *inventory, *policy_files)}
    input_pins = {path: _verified_digest(path) for path in sorted(input_paths)}
    if input_pins[str(Path(new_study["manifest"]).resolve())] != new_study["manifest_sha256"]:
        raise ValueError("allocation manifest hash changed")
    budgets, infeasible = old._budgets(features, TARGETS, metadata["cache_contract"])
    schedule = old._schedule_base(new_study, model, split, metadata, features, budgets, infeasible,
                                 stage=f"allocation-{split}")
    schedule.update(candidate=CANDIDATE, source_sha256=sources(),
                    v5_controls={"format": FORMAT, "candidate": CANDIDATE,
                                 "old_study_sha256": old_study["sha256"],
                                 "fit_index_path": str(index_path), "fit_index_file_sha256": file_digest(index_path),
                                 "fit_index_sha256": index["sha256"], "statistic": STATISTIC,
                                 "input_file_sha256": input_pins, "feature_file_inventory": inventory,
                                 "budget_steps_pp": list(STEPS), "operating_retention_margin": .005,
                                 "shuffle_seeds": list(SEEDS), "quality_labels_opened": False,
                                 "calibration_reused_only_for_identical_ranker": True,
                                 "grid_definition": "nominal choices; token-wise reconciliation may leave the nominal grid",
                                 "scope": "development; exact-count physical inference required"},
                    assignment_artifacts=[], unavailable_panels=[])
    grouped = defaultdict(list)
    for request, feature in sorted(features.items()):
        grouped[feature["dataset"]].append(feature)
        schedule["rows"][request] = {**old._geometry(feature), "cells": [
            {"key": "full_all_tokens", "selector": "full", "selected_count": feature["input_visual_tokens"],
             "target": 0, "policy": "full", "mode": "full", "candidate": CANDIDATE}]}

    def publish_assignment(assignment, *, target, dataset, mode):
        path = output / "assignments" / f"allocation-{split}" / f"t{target}_{digest(dataset)[:16]}" / f"{mode}.json"
        write_immutable(path, assignment)
        schedule["assignment_artifacts"].append({"target": target, "dataset": dataset, "mode": mode,
                "path": str(path), "file_sha256": file_digest(path),
                "assignments_sha256": assignment["assignments_sha256"]})

    def add_cells(assignment, *, target, mode, step, policy, seed=None):
        for request, assigned in assignment["rows"].items():
            cell = {"key": f"{mode}_t{target}", "selector": "scorer", "selected_count": assigned["adaptive_selected_count"],
                    "target": target, "policy": STATISTIC, "mode": mode, "candidate": CANDIDATE,
                    "uniform_selected_count": assigned["uniform_selected_count"],
                    "assignments_sha256": assignment["assignments_sha256"], "policy_sha256": policy["policy_sha256"],
                    "budget_step_pp": step, "operating_retention_margin": .005,
                    "support_failures": assigned.get("support_failures", []),
                    "out_of_support": assigned.get("out_of_support", False),
                    "fallback": assigned.get("fallback", False), "fallback_kind": assigned.get("fallback_kind"),
                    "changed_from_uniform": assigned["changed_from_uniform"]}
            if seed is not None:
                cell.update(mode="shuffle", allocation_source_mode=f"continuous_{step}pp",
                            shuffle_seed=seed, shuffle_movable_fraction=assignment["movable_fraction"],
                            shuffle_changed_fraction=assignment["changed_fraction"],
                            shuffle_informative=assignment["informative_control"] and assignment["movable_fraction"] >= .2,
                            source_adaptive_selected_count=assigned["source_adaptive_selected_count"])
            schedule["rows"][request]["cells"].append(cell)

    for target in TARGETS:
        policy, benefit = old._load_policy(index, STATISTIC, target)
        if policy is None or benefit is not None or policy["statistic"] != STATISTIC:
            raise ValueError("missing frozen single-statistic entropy policy")
        for dataset, cohort in sorted(grouped.items()):
            budget = budgets[str(target)][dataset]
            if not budget["feasible"]:
                continue
            inputs = []
            for feature in cohort:
                request = feature["request_id"]
                uniform = budget["counts"][request]
                schedule["rows"][request]["cells"].append(
                    {"key": f"uniform_t{target}", "selector": "scorer", "selected_count": uniform,
                     "target": target, "policy": STATISTIC, "mode": "uniform", "candidate": CANDIDATE,
                     "uniform_selected_count": uniform, "policy_sha256": None,
                     "changed_from_uniform": False, "fallback": False, "fallback_kind": None})
                inputs.append({**old._geometry(feature), "statistic_value":
                               allocation.statistic_from_features(feature, STATISTIC, uniform)})
            for step in STEPS:
                mode = f"continuous_{step}pp"
                assignment = controls.allocate(policy, inputs, budget["counts"], budget_step_pp=step,
                                               operating_retention_margin=.005)
                publish_assignment(assignment, target=target, dataset=dataset, mode=mode)
                add_cells(assignment, target=target, mode=mode, step=step, policy=policy)
                for seed in SEEDS:
                    mode = f"shuffle_{step}pp_seed{seed}"
                    shuffled = controls.shuffle_assignments(assignment, seed=seed)
                    publish_assignment(shuffled, target=target, dataset=dataset, mode=mode)
                    add_cells(shuffled, target=target, mode=mode, step=step, policy=policy, seed=seed)
    schedule = old._seal(schedule)
    write_immutable(schedule_path, schedule)
    return schedule
