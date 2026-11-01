"""Six-hour, fit-gated recovery experiment; completed studies stay read-only.

The allocation ranker is always the original Rank32. Schedules are frozen
before screening labels are read for physical-cache reuse or reporting.
"""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import time

from . import adaptive_allocation_v4 as allocation
from . import adaptive_planning_v4 as planning
from . import selector_controls_v5 as controls
from . import selector_study_v5 as previous
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest


ROOT = Path("reproduction_results/mlsys_recovery_20261017_v6")
START = 1792238634  # 2026-10-17 12:03:54 UTC; never reset on resume.
DEADLINE = START + 6 * 3600
MODELS = ("qwen25", "llava_next")
TARGETS = (60, 70)
SEEDS = (42, 43, 44)
WORLD = 8
FORMAT = "mlsys_recovery_study_v6"


def seal(value):
    return {**value, "sha256": digest(value)}


def read(path):
    return json.loads(Path(path).read_text())


def checked(path):
    return planning._checked(read(path))


def sources():
    names = ("recovery_study_v6.py", "recovery_allocation_v6.py", "recovery_training_v6.py",
             "recovery_training_physical_v6.py", "recovery_training_task_v6.py",
             "recovery_operations_v6.py", "recovery_report_v6.py")
    return {str(Path(__file__).with_name(name).resolve()): file_digest(Path(__file__).with_name(name))
            for name in names}


def plan_names():
    return [f"{stage}-{model}" for stage in ("training", "allocation", "stress") for model in MODELS]


def prepare(cwd):
    """Freeze sources, the historical cohort, and the original six-hour clock."""
    from .recovery_operations_v6 import build_config
    cwd = Path(cwd).resolve()
    output = cwd / ROOT
    path = output / "study.json"
    if path.exists():
        study = checked(path)
        verify(study)
        ensure_configs(study)
        return study
    old_path = cwd / "reproduction_results/mlsys_adaptive_20261016_v4/study.json"
    v5_path = cwd / "reproduction_results/mlsys_selector_20261017_v5/study.json"
    old, v5 = checked(old_path), checked(v5_path)
    previous.verify_study(v5)
    study = seal({"format": FORMAT, "cwd": str(cwd), "output_root": str(output),
        "created_utc_epoch": START, "deadline_utc_epoch": DEADLINE,
        "new_work_cutoff_utc_epoch": START + 5 * 3600 + 15 * 60,
        "training_cutoff_utc_epoch": START + 75 * 60,
        "stress_admission_cutoff_utc_epoch": START + 4 * 3600,
        "models": {model: old["models"][model] for model in MODELS},
        "manifest": old["manifest"], "manifest_sha256": old["manifest_sha256"],
        "old_study": str(old_path), "old_study_sha256": file_digest(old_path),
        "v5_study": str(v5_path), "v5_study_sha256": file_digest(v5_path),
        "source_sha256": sources(), "world_size": WORLD, "targets": list(TARGETS),
        "screening_families": ["DocVQA", "TextVQA"], "rows_per_family": 60,
        "stress_family": "ChartQA", "stress_target": 70,
        "allocation_ranker": "current_rank32", "new_allocation": "removal_harm",
        "scalar_inputs": list(allocation.STATISTICS), "scalar_outputs": 1,
        "risk_penalty": .5, "operating_retention_margin": .005, "budget_step_pp": 1,
        "shuffle_seeds": list(SEEDS), "reserved_generated_slots": 127,
        "selection": {"minimum_model_mean_vs_uniform_pp": 0.,
                      "minimum_model_mean_vs_shuffle_pp": 0.,
                      "minimum_balanced_gain_pp": .5, "minimum_panel_gain_pp": -1.},
        "evidence_status": "development only; previously inspected screening cohort; no fresh confirmation",
        "allocation_training_separation": "original ranker only; repaired ranker never enters allocation",
        "cache_claim": "allocated KV capacity, not observed live use or GPU peak memory"})
    write_immutable(path, study)
    ensure_configs(study)
    return study


def ensure_configs(study):
    """Recover a preparation interrupted between study and config publication."""
    from .recovery_operations_v6 import build_config
    cwd, output = Path(study["cwd"]), Path(study["output_root"])
    for role in ("clusterb", "clustera"):
        names = plan_names() if role == "clusterb" else [n for n in plan_names() if n.endswith("llava_next")]
        config = build_config(cwd, [output / f"plans/{n}/plan.json" for n in names], DEADLINE,
                              role=role, created_utc_epoch=START)
        write_immutable(output / f"coordinator/{role}-config.json", config)


def verify(study):
    planning._checked(study)
    if (study.get("format") != FORMAT or study.get("created_utc_epoch") != START
            or study.get("deadline_utc_epoch") != DEADLINE):
        raise ValueError("recovery identity or original deadline changed")
    pins = {**study["source_sha256"], study["manifest"]: study["manifest_sha256"],
            study["old_study"]: study["old_study_sha256"], study["v5_study"]: study["v5_study_sha256"]}
    for path, expected in pins.items():
        if file_digest(path) != expected:
            raise ValueError(f"frozen recovery source/input changed: {path}")


def fit(study):
    """Read fit labels only. CPU admission is frozen before physical schedules."""
    from .recovery_allocation_v6 import fit_recovery, admission_report
    verify(study)
    output = Path(study["output_root"])
    admission_path = output / "policies/admission.json"
    if admission_path.exists():
        admission = checked(admission_path)
        for path, sha in admission["fit_file_hashes"].items():
            if file_digest(path) != sha:
                raise ValueError("frozen recovery fit artifact changed")
        return admission
    old = checked(study["old_study"])
    results, pins = {}, {}
    for model in MODELS:
        _, metadata, features, schedule, qualities, physical_hash = planning._physical_fit(old, model)
        results[model] = []
        for target in TARGETS:
            rows = planning._fit_rows(features, schedule, qualities, target, "normalized_entropy")
            rows = [{**row, "benefit_features": {
                stat: allocation.statistic_from_features(features[row["row_hash"]], stat,
                                                         row["uniform_selected_count"])
                for stat in allocation.STATISTICS}} for row in rows]
            identity = {"study_sha256": study["sha256"], "model": model, "target": target,
                        "manifest_sha256": study["manifest_sha256"],
                        "checkpoint_sha256": metadata["checkpoint_sha256"],
                        "physical_rows_sha256": physical_hash, "held_out_benchmark": None}
            path = output / model / f"policies/removal_harm_t{target}.json"
            if path.exists():
                result = checked(path)
                if result.get("identity") != identity:
                    raise ValueError("recovery fit has a different input identity")
            else:
                result = fit_recovery(rows, identity=identity, ridge=10., min_groups=20, shrinkage=30.)
                write_immutable(path, result)
            pins[str(path)] = file_digest(path)
            results[model].append(result)
    decision = admission_report(results)
    admission = seal({"format": FORMAT + "_admission", "study_sha256": study["sha256"],
                      "decision": decision, "fit_file_hashes": pins,
                      "screening_labels_consumed": False})
    write_immutable(admission_path, admission)
    return admission


def cohort(features, families):
    from .selector_report_v5 import _dataset
    result = {key: value for key, value in features.items() if _dataset(value["dataset"]) in families}
    counts = {family: sum(_dataset(row["dataset"]) == family for row in result.values()) for family in families}
    if any(count != 60 for count in counts.values()):
        raise ValueError(f"recovery requires the existing complete 60-row cohorts: {counts}")
    return result


def assign(policy, scalar, features, budget):
    rows = []
    for feature in features:
        count = budget["counts"][feature["request_id"]]
        inputs = {stat: allocation.statistic_from_features(feature, stat, count) for stat in allocation.STATISTICS}
        value = allocation.predict_benefit_scalar(scalar, inputs) if scalar else inputs[policy["statistic"]]
        rows.append({**planning._geometry(feature), "statistic_value": value})
    return controls.allocate(policy, rows, budget["counts"], budget_step_pp=1, operating_retention_margin=.005)


def schedule(study, model, *, stress=False):
    """No screening labels are read while selecting counts or freezing rules."""
    verify(study)
    output = Path(study["output_root"])
    stage = "stress" if stress else "allocation"
    path = output / model / f"schedules/{stage}.json"
    if path.exists():
        saved = checked(path)
        if saved["study_sha256"] != study["sha256"]:
            raise ValueError("existing schedule belongs to another recovery study")
        return path
    admission = checked(output / "policies/admission.json")
    if not admission_passed(admission):
        raise ValueError("CPU removal-harm gate did not admit physical allocation")
    old = checked(study["old_study"])
    _, old_output, _, metadata, all_features = planning._features(old, model, "screening")
    features = cohort(all_features, ("ChartQA",) if stress else ("DocVQA", "TextVQA"))
    targets = (70,) if stress else TARGETS
    budgets, infeasible = planning._budgets(features, targets, metadata["cache_contract"])
    if infeasible:
        raise ValueError("predeclared recovery panel is infeasible")
    indexes = {"entropy": checked(old_output / "policies/fit_index.json"),
               "endpoint": checked(old_output / "policies/learned_fit_index.json")}
    for index in indexes.values():
        if (index["study_sha256"] != old["sha256"] or index["manifest_sha256"] != study["manifest_sha256"]
                or index["checkpoint_sha256"] != metadata["checkpoint_sha256"]
                or index["source_sha256"] != planning._sources()):
            raise ValueError("comparison policy provenance differs from the frozen original ranker")
    payload = planning._schedule_base(study, model, "screening", metadata, features, budgets, infeasible, stage=stage)
    payload.update(candidate="current_rank32", seed=42, source_sha256=sources(), assignment_artifacts=[])
    for request, feature in sorted(features.items()):
        payload["rows"][request] = {**planning._geometry(feature), "cells": [
            {"key": "all_tokens", "selector": "full", "selected_count": feature["input_visual_tokens"],
             "target": 0, "policy": "full", "mode": "full"}]}
    groups = defaultdict(list)
    for feature in features.values():
        groups[feature["dataset"]].append(feature)

    def append_assignment(value, target, dataset, mode, policy, seed=None):
        tag = f"{mode}_seed{seed}" if seed is not None else mode
        artifact = output / model / "assignments" / stage / f"t{target}_{digest(dataset)[:12]}_{tag}.json"
        write_immutable(artifact, value)
        payload["assignment_artifacts"].append({"path": str(artifact), "sha256": file_digest(artifact)})
        for request, row in value["rows"].items():
            cell = {"key": f"{tag}_t{target}", "selector": "scorer", "selected_count": row["adaptive_selected_count"],
                    "target": target, "policy": mode, "mode": tag,
                    "uniform_selected_count": row["uniform_selected_count"],
                    "assignments_sha256": value["assignments_sha256"], "policy_sha256": policy["policy_sha256"],
                    "fallback": row.get("fallback", False), "fallback_kind": row.get("fallback_kind"),
                    "support_failures": row.get("support_failures", []), "out_of_support": row.get("out_of_support", False),
                    "changed_from_uniform": row["changed_from_uniform"]}
            if seed is not None:
                cell.update(shuffle_seed=seed, shuffle_informative=value["informative_control"],
                            shuffle_movable_fraction=value["movable_fraction"],
                            shuffle_changed_fraction=value["changed_fraction"])
            payload["rows"][request]["cells"].append(cell)

    for target in targets:
        result = checked(output / model / f"policies/removal_harm_t{target}.json")
        fitted = {"removal_harm": (result["policy"], result["crossfit"]["model"]),
                  "entropy": planning._load_policy(indexes["entropy"], "normalized_entropy", target),
                  "endpoint": planning._load_policy(indexes["endpoint"], "learned_benefit_scalar", target)}
        if any(policy is None for policy, _ in fitted.values()):
            raise ValueError("missing frozen comparison policy")
        for dataset, group in sorted(groups.items()):
            budget = budgets[str(target)][dataset]
            for feature in group:
                count = budget["counts"][feature["request_id"]]
                for selector in ("scorer", "divprune", *(("cdpruner",) if model == "llava_next" else ())):
                    payload["rows"][feature["request_id"]]["cells"].append({
                        "key": f"{selector}_uniform_t{target}", "selector": selector, "selected_count": count,
                        "target": target, "policy": "uniform", "mode": "uniform", "uniform_selected_count": count,
                        "changed_from_uniform": False, "fallback": False})
            for mode, (policy, scalar) in fitted.items():
                value = assign(policy, scalar, group, budget)
                append_assignment(value, target, dataset, mode, policy)
                if mode == "removal_harm":
                    for seed in SEEDS:
                        shuffled = controls.shuffle_assignments(value, seed=seed)
                        append_assignment(shuffled, target, dataset, "shuffled", policy, seed)
    write_immutable(path, seal(payload))
    return path


def admission_passed(admission):
    decision = admission["decision"]
    # The fit module owns the scientific admission schema, not runtime code.
    return decision.get("passed") is True


def cache_inventory(study, model):
    """Only completed, audited current-ranker v5 results may satisfy new cells."""
    v5 = checked(study["v5_study"])
    previous.verify_study(v5)
    root = Path(v5["output_root"]) / model
    entries = {}
    for stage, schedule_name in (("screening", "screening-current_rank32-seed42"),
                                 ("allocation-screening", "allocation-screening")):
        directory = root / stage / "current_rank32"
        rows = previous.scan_physical(v5, model, "current_rank32", directory, root / f"schedules/{schedule_name}.json")
        if rows is None:
            raise ValueError("baseline cache is not a complete audited stage")
        for rank in range(WORLD):
            meta = read(directory / f"rank{rank}.metadata.json")
            context = meta["backbone"]["selector_v5"]
            if context["query_control"] != "correct" or context["coverage_fraction"] != 0:
                raise ValueError("physical reuse requires the unmodified correct-query scorer")
        for row in rows:
            key = (row["request_id"], row["selector"], row["selected_count"])
            path = directory / "rows" / f"{row['request_id']}.{row['method']}.json"
            old = entries.get(key)
            if old and any(old["row"].get(field) != row.get(field) for field in
                           ("prediction", "generated_token_ids", "quality_score", "selected_patch_indices")):
                raise ValueError("identical frozen physical counts disagree across baseline runs")
            if old is None or row.get("physical_measurement_performed"):
                entries[key] = {"path": str(path), "sha256": file_digest(path), "row": row}
    return entries


def reuse_and_missing(master, inventory):
    """Cache membership changes execution work only, never assigned budgets."""
    execution = copy.deepcopy({key: value for key, value in master.items() if key != "sha256"})
    execution["rows"] = {}
    reuse = {}
    for request, row in master["rows"].items():
        missing = []
        for cell in row["cells"]:
            source = inventory.get((request, cell["selector"], cell["selected_count"]))
            if source is None:
                # Competitor baselines are cached-only in this bounded sprint.
                if cell["selector"] in ("divprune", "cdpruner"):
                    reuse[f"{request}.{cell['key']}"] = {"status": "not_run_no_exact_cached_baseline"}
                else:
                    missing.append(cell)
            else:
                reuse[f"{request}.{cell['key']}"] = {"status": "exact_cached_physical",
                    "path": source["path"], "sha256": source["sha256"]}
        if missing:
            execution["rows"][request] = {**row, "cells": missing}
    execution["master_schedule_sha256"] = master["sha256"]
    return seal(execution), seal({"master_schedule_sha256": master["sha256"], "cells": reuse})


def publish_physical(study, model, master_path, *, stress=False):
    stage = "stress" if stress else "allocation"
    root = Path(study["output_root"])
    execution_path = root / model / f"schedules/{stage}-execution.json"
    reuse_path = root / model / f"schedules/{stage}-reuse.json"
    master = checked(master_path)
    if execution_path.exists():
        execution, reuse = checked(execution_path), checked(reuse_path)
        if any(item["master_schedule_sha256"] != master["sha256"] for item in (execution, reuse)):
            raise ValueError("execution/reuse schedule no longer matches frozen assignments")
    else:
        execution, reuse = reuse_and_missing(master, cache_inventory(study, model))
        write_immutable(reuse_path, reuse)
        write_immutable(execution_path, execution)
    destination = root / model / stage / "current_rank32"
    spec = study["models"][model]
    tasks = []
    if execution["rows"]:
        for rank in range(WORLD):
            argv = previous.python(study, "selector_experiment_v5") + ["--phase", "physical",
                "--candidate", "current_rank32", "--warmstart", spec["checkpoint"], "--checkpoint", spec["checkpoint"],
                "--split", "screening", "--manifest", study["manifest"], "--schedule", str(execution_path),
                "--adapter", spec["adapter"], "--model", spec["model_path"], "--model-revision", spec["revision"],
                "--output", str(destination), "--rank", str(rank), "--world-size", str(WORLD),
                "--query-control", "correct", "--feature-root",
                str(Path(study["old_study"]).parent / model / "screening_features")]
            if model == "llava_next":
                argv.append("--reference-clip")
            task = previous.task(study, f"{stage}-{model}-{rank}", argv,
                                 [destination / f"rank{rank}.complete.json"],
                                 resume=destination / f"rank{rank}.metadata.json")
            task.update(id=f"recovery-v6-{stage}-{model}-{rank}", tag="recovery-v6", model_tag=model)
            tasks.append(task)
    return publish_plan(study, f"{stage}-{model}", tasks,
                        [execution_path, reuse_path, master_path, spec["checkpoint"]])


def publish_plan(study, name, tasks, inputs=()):
    from .selector_experiment_v5 import source_hashes
    from .worker import read_plan, verify_sources
    if name not in plan_names():
        raise ValueError("undeclared recovery queue")
    path = Path(study["output_root"]) / f"plans/{name}/plan.json"
    if path.exists():
        verify_sources(read_plan(path))
        if read(path)["tasks"] != tasks:
            raise ValueError("published recovery tasks changed")
        return path
    pins = dict(study["source_sha256"])
    model = next(model for model in MODELS if name.endswith(model))
    for relative, sha in source_hashes(study["models"][model]["adapter"]).items():
        pins[str(Path(study["cwd"]) / "learned_score" / relative)] = sha
    for item in (study["manifest"], *inputs):
        pins[str(item)] = file_digest(item)
    write_immutable(path, seal({"format": FORMAT + "_plan", "cwd": study["cwd"], "tasks": tasks,
        "input_hashes": pins, "unexpected_retries": 1, "created_utc_epoch": START,
        "deadline_utc_epoch": min(DEADLINE, study["training_cutoff_utc_epoch"]) if name.startswith("training")
                              else study["new_work_cutoff_utc_epoch"]}))
    verify_sources(read_plan(path))
    return path


def publish_training(study):
    """Only the two Qwen gradient pilots can occupy GPUs in this short lane."""
    root = Path(study["output_root"])
    v5 = checked(study["v5_study"])
    manifest = root / "data/training64.json"
    # This is deterministic and refuses to overwrite a different selection.
    from .recovery_training_v6 import create_manifest
    create_manifest(v5["training_manifest"], manifest)
    for model in MODELS:
        path = root / f"plans/training-{model}/plan.json"
        if path.exists():
            from .worker import read_plan, verify_sources
            verify_sources(read_plan(path))
            continue
        tasks = []
        if model == "qwen25" and time.time() < study["training_cutoff_utc_epoch"]:
            for gate in ("detached", "corrected_kth"):
                output = root / model / "training" / f"{gate}-seed42"
                argv = previous.python(study, "recovery_training_task_v6") + [
                    "--study", str(root / "study.json"), "--model", model,
                    "--adapter", study["models"][model]["adapter"], "--gate", gate]
                task = previous.task(study, f"training-{model}-{gate}", argv, [output / "outcome.json"])
                task.update(id=f"recovery-v6-training-{model}-{gate}", tag="recovery-v6", model_tag=model)
                tasks.append(task)
        publish_plan(study, f"training-{model}", tasks,
                     [manifest, v5["training_manifest"], v5["original_training_manifest"],
                      study["models"][model]["checkpoint"]])
        if not tasks:
            write_immutable(root / model / "training/skipped.json", {
                "reason": "bounded_Qwen_priority" if model != "qwen25" else "training_window_expired",
                "deadline": study["training_cutoff_utc_epoch"], "success": False})


def tick(study):
    """CPU orchestration only: external resource launch remains a separate action."""
    from .recovery_report_v6 import write_report
    verify(study)
    publish_training(study)
    admission = fit(study)
    root = Path(study["output_root"])
    for model in MODELS:
        if admission_passed(admission) and time.time() < study["new_work_cutoff_utc_epoch"]:
            publish_physical(study, model, schedule(study, model))
        elif not (root / f"plans/allocation-{model}/plan.json").exists():
            publish_plan(study, f"allocation-{model}", [])
    result = write_report(study)
    if (result["decision"]["passed"] and time.time() < study["stress_admission_cutoff_utc_epoch"]):
        for model in MODELS:
            publish_physical(study, model, schedule(study, model, stress=True), stress=True)
    elif (not admission_passed(admission) or all(s["complete"] for s in result["execution"].values())
          or time.time() >= study["stress_admission_cutoff_utc_epoch"]):
        for model in MODELS:
            if not (root / f"plans/stress-{model}/plan.json").exists():
                publish_plan(study, f"stress-{model}", [])
    return result


def watch(study):
    """Persistent local coordinator; no qsub, SSH, or GPU attachment occurs here."""
    import fcntl
    root = Path(study["output_root"]) / "coordinator"
    root.mkdir(parents=True, exist_ok=True)
    with (root / "driver.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study["deadline_utc_epoch"]:
            result = tick(study)
            event = {"time": time.time(), "decision": result["decision"],
                     "execution": {model: state["status"] for model, state in result["execution"].items()}}
            with (root / "driver.jsonl").open("a") as stream:
                stream.write(json.dumps(event, sort_keys=True) + "\n")
            print(json.dumps(event, sort_keys=True), flush=True)
            time.sleep(min(60, max(0, study["deadline_utc_epoch"] - time.time())))
        from .recovery_report_v6 import write_report
        return write_report(study)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--phase", choices=("prepare", "fit", "publish", "report", "tick", "watch"), required=True)
    args = parser.parse_args(argv)
    study = prepare(args.cwd)
    if args.phase == "prepare":
        print(json.dumps({"study": str(Path(study["output_root"]) / "study.json"), "deadline": DEADLINE}))
    elif args.phase == "fit":
        print(json.dumps(fit(study)["decision"]))
    elif args.phase == "publish":
        admission = fit(study)
        if admission_passed(admission) and time.time() < study["new_work_cutoff_utc_epoch"]:
            for model in MODELS:
                print(publish_physical(study, model, schedule(study, model)))
        else:
            print(json.dumps({"physical_publication": "not_admitted_or_deadline", "decision": admission["decision"]}))
    elif args.phase == "report":
        from .recovery_report_v6 import write_report
        print(json.dumps(write_report(study)))
    elif args.phase == "tick":
        print(json.dumps(tick(study)))
    else:
        watch(study)


if __name__ == "__main__":
    main()
