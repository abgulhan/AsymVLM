"""Versioned physical allocation repair, with bounded restartable execution.

GPU plans and schedules are immutable. CPU publication may advance only after
the preceding physical stage has completed; evaluation outcomes never select a
policy. Historical v3 artifacts are read-only diagnostic references.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import pwd
import time

from .state import atomic_json, digest, file_digest
from .followup_common import write_immutable
from . import sprint48


DEFAULT_ROOT = Path("reproduction_results/mlsys_adaptive_20261016_v4")
OLD_ROOT = Path("reproduction_results/mlsys_adaptive_20261016_v3")
DEADLINE = 1792204506  # 2026-10-17 02:35:06 UTC; bounded overnight study.
WORLD_SIZE = 16
MODELS = ("qwen25", "llava_next", "phi35")
TARGETS = (40, 50, 60, 70)
STATISTICS = ("normalized_entropy", "local_shell", "nearest_neighbor_diversity")
SEEDS = (42, 43, 44)
STAGES = ("fit_features", "screening_features", "evaluation_features", "local_calibration",
          "screening", "benefit_screening", "evaluation")


def seal(value):
    return {**value, "sha256": digest(value)}


def read_json(path):
    return json.loads(Path(path).read_text())


def environment(root):
    return {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_MODULES_CACHE": str(root / ".cache/huggingface/modules"),
            "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "1"}


def prepare(root, output):
    root, output = root.resolve(), output.resolve()
    if output != root / DEFAULT_ROOT:
        raise ValueError("v4 is restricted to its new explicit output directory")
    from .adaptive_data_v4 import load_manifest
    manifest = output / "data/manifest.json"
    load_manifest(manifest)
    old = read_json(root / OLD_ROOT / "study.json")
    models = old["models"]
    study_path = output / "study.json"
    if study_path.exists():
        study = read_json(study_path)
        if study.get("sha256") != digest({k: v for k, v in study.items() if k != "sha256"}):
            raise ValueError("modified study")
    else:
        now = int(time.time())
        if not now < DEADLINE:
            raise ValueError("approved study window has expired")
        study = seal({"format": "mlsys_adaptive_study_v4", "cwd": str(root), "output_root": str(output),
            "created_utc_epoch": now, "deadline_utc_epoch": DEADLINE, "models": models,
            "manifest": str(manifest), "manifest_sha256": file_digest(manifest), "world_size": WORLD_SIZE,
            "old_manifest": str(root / OLD_ROOT / "data/manifest.json"),
            "statistic_candidates": list(STATISTICS), "kv_reduction_percent": list(TARGETS),
            "local_physical_offsets_percentage_points": [-10, -5, 0, 5, 10], "shuffle_seeds": list(SEEDS),
            "selection_rule": "physical screening only; paired uniform and same-histogram shuffled controls; model-balanced robustness",
            "conditional_learned_benefit": "three input statistics, one cross-fitted benefit scalar; only after entropy fails physical control",
            "evidence_status": "development; original evaluation retained, not fresh confirmation",
            "baseline_scope": "projected_feature_selector_adaptations", "text_eviction_claim": False,
            "cache_objective": "exact aggregate allocated KV bytes; reserve 127 generated slots for every method",
            "infeasible_target_policy": "report explicitly; never silently relax target",
            "phi_old_results": "excluded from promotion; corrected replay isolates only EOS stopping"})
        write_immutable(study_path, study)
    shared = ["scripts/mlsys_adaptive_v4_clustera.pbs", "scripts/mlsys_adaptive_v4_clustera_worker.sh",
              "scripts/mlsys_adaptive_v4_idle_worker.sh", "learned_score/mlsys_sprint/adaptive_operations_v4.py",
              "learned_score/mlsys_sprint/sprint48.py", "learned_score/mlsys_sprint/worker.py",
              "learned_score/mlsys_sprint/overnight.py"]
    inputs = {str(root / path): file_digest(root / path) for path in shared}
    for cluster in ("clusterb", "clustera"):
        names = MODELS if cluster == "clusterb" else ("llava_next",)
        plans = [str(output / "plans/phi-stopping-parity/plan.json")] if cluster == "clusterb" else []
        # Feature extraction and dense local labels precede diagnostic replays.
        plans += [str(output / f"plans/{model}-{stage}/plan.json") for stage in STAGES for model in names]
        if cluster == "clusterb":
            plans += [str(output / f"plans/phi-corrected-{stage}/plan.json") for stage in ("calibration", "screening", "evaluation")]
        config = seal({"format": sprint48.FORMAT, "cwd": str(root), "user": pwd.getpwuid(os.getuid()).pw_name,
            "created_utc_epoch": study["created_utc_epoch"], "deadline_utc_epoch": study["deadline_utc_epoch"],
            "maximum_submissions": 24, "maximum_debug_nodes": 2,
            "pbs_script": "scripts/mlsys_adaptive_v4_clustera.pbs", "plan_paths": plans,
            "state_root": str(output / f"coordinator/{cluster}"), "coordinator_role": cluster,
            "input_hashes": inputs, "resource_map": {"adaptive-v4": "gpu"}})
        write_immutable(output / f"coordinator/{cluster}-config.json", config)
        sprint48.read_config(output / f"coordinator/{cluster}-config.json")
    return study


def ready_stage(directory, world_size=WORLD_SIZE):
    directory = Path(directory)
    if len(list(directory.glob("rank*.complete.json"))) != world_size:
        return False
    for rank in range(world_size):
        marker_path, identity_path = directory / f"rank{rank}.complete.json", directory / f"rank{rank}.metadata.json"
        if not marker_path.is_file() or not identity_path.is_file():
            return False
        marker, identity = read_json(marker_path), read_json(identity_path)
        if marker.get("identity_sha256") != digest(identity):
            raise ValueError(f"completion identity differs: {marker_path}")
    return True


def publish_plan(study, name, tasks, inputs):
    path = Path(study["output_root"]) / f"plans/{name}/plan.json"
    if path.exists():
        return False
    plan = seal({"format": "mlsys_adaptive_stage_v4", "cwd": study["cwd"], "tasks": tasks,
        "input_hashes": inputs, "unexpected_retries": 1, "created_utc_epoch": time.time(),
        "deadline_utc_epoch": study["deadline_utc_epoch"]})
    write_immutable(path, plan)
    from .worker import read_plan, verify_sources
    verify_sources(read_plan(path))
    return True


def publish_phi_gate(study):
    root, output = Path(study["cwd"]), Path(study["output_root"])
    if (output / "plans/phi-stopping-parity/plan.json").exists():
        return False
    from .adaptive_data_v3 import load_manifest, dataset_family
    from .adaptive_experiment_v3 import source_hashes
    from .phi_adaptive_v4 import source_hashes as phi_hashes
    manifest = load_manifest(study["old_manifest"])
    entries = sorted(manifest.splits["fit"], key=lambda row: row["row_hash"])
    ids = [next(row["row_hash"] for row in entries if dataset_family(row["dataset"]) == name)
           for name in ("DocVQA", "TextVQA")]
    requests = output / "phi_corrected/parity_requests.json"
    write_immutable(requests, {"row_hashes": ids})
    spec = study["models"]["phi35"]
    physical = output / "phi_corrected/parity"
    argv = [str(root / ".conda/asymvlm/bin/python"), "-u", "-m", "learned_score.mlsys_sprint.phi_adaptive_v4",
            "--phase", "parity", "--manifest", study["old_manifest"], "--checkpoint", spec["checkpoint"],
            "--output", str(physical), "--request-ids", str(requests), "--limit", "2", "--max-new-tokens", "128",
            "--retentions", "10", "40", "70", "100", "--model", spec["model_path"]]
    sources = {str(root / "learned_score" / path): sha for path, sha in
               {**source_hashes("phi3_vision"), **phi_hashes()}.items()}
    for path in (Path(study["old_manifest"]), requests, Path(spec["checkpoint"]), *Path(spec["model_path"]).glob("*.py")):
        sources[str(path)] = file_digest(path)
    task = {"id": "adaptive-v4-phi-stopping-parity", "tag": "adaptive-v4", "resource": "gpu",
            "minimum_start_seconds": 600, "depends_on": [], "argv": argv, "env": environment(root),
            "artifacts": [{"path": str(physical / "rank0.complete.json")}],
            "resume_checkpoint": str(physical / (ids[0] + ".json")), "resume_args": ["--resume"]}
    return publish_plan(study, "phi-stopping-parity", [task], sources)


def phi_gate_passed(study):
    path = Path(study["output_root"]) / "phi_corrected/parity/rank0.complete.json"
    if not path.exists():
        return False
    marker = read_json(path)
    if marker.get("status") != "passed" or marker.get("requests") != 2:
        raise ValueError("Phi stopping gate did not pass both fit-only examples")
    from .phi_adaptive_v4 import source_hashes, generation_trace
    requests = read_json(path.parent.parent / "parity_requests.json")["row_hashes"]
    if len(requests) != 2 or len(set(requests)) != 2:
        raise ValueError("Phi stopping gate requires two distinct fitting examples")
    fit_ids = {row["row_hash"] for row in read_json(study["old_manifest"])["splits"]["fit"]}
    if not set(requests) <= fit_ids:
        raise ValueError("Phi stopping gate contains non-fitting examples")
    for request in requests:
        proof = read_json(path.parent / f"{request}.json")
        provenance = {k: v for k, v in proof.items() if k not in
                      {"identity_sha256", "status", "request_id", "input_visual_tokens", "outcomes"}}
        if (proof.get("status") != "passed" or proof.get("request_id") != request
                or proof.get("identity_sha256") != digest(provenance)
                or marker.get("identity_sha256") != digest(provenance)
                or proof.get("manifest_sha256") != file_digest(study["old_manifest"])
                or proof.get("checkpoint_sha256") != file_digest(study["models"]["phi35"]["checkpoint"])
                or proof.get("source_sha256") != source_hashes() or proof.get("split") != "fit"
                or proof.get("max_new_tokens") != 128 or proof.get("retentions_percent") != [10, 40, 70, 100]
                or marker.get("stopping") != proof.get("backbone", {}).get("stopping")):
            raise ValueError("Phi stopping proof identity changed")
        outcomes = proof.get("outcomes", [])
        if [item.get("retention_percent") for item in outcomes] != [10, 40, 70, 100]:
            raise ValueError("Phi stopping proof lacks required physical retentions")
        for item in outcomes:
            bounded = item["bounded"]["generated_token_ids"]
            if (not item.get("native_ids_equal") or bounded != item["native_unbounded"]["generated_token_ids"]
                    or item["bounded"] != generation_trace(bounded, 128)
                    or item["native_unbounded"] != generation_trace(bounded, 128)
                    or item["selected_visual_tokens"] != max(1, proof["input_visual_tokens"] * item["retention_percent"] // 100)
                    or item["cache"].get("compaction_events") != 0 or item["cache"].get("evicted_tokens") != 0):
                raise ValueError("Phi native/bounded stopping proof is inconsistent")
            if item["retention_percent"] == 100 and (
                    item.get("native_generate_ids_equal") is not True
                    or item["native_generate"] != generation_trace(bounded, 128)):
                raise ValueError("Phi all-token native generate parity is missing")
    return True


def publish_phi_replay(study, stage):
    root, output = Path(study["cwd"]), Path(study["output_root"])
    if (output / f"plans/phi-corrected-{stage}/plan.json").exists() or not phi_gate_passed(study):
        return False
    from .adaptive_experiment_v3 import source_hashes
    from .phi_adaptive_v4 import source_hashes as phi_hashes
    spec = study["models"]["phi35"]
    split = "fit" if stage == "calibration" else stage
    physical = output / "phi_corrected" / stage
    sources = {str(root / "learned_score" / path): sha for path, sha in
               {**source_hashes("phi3_vision"), **phi_hashes()}.items()}
    for path in (Path(study["old_manifest"]), Path(spec["checkpoint"]), output / "phi_corrected/parity/rank0.complete.json"):
        sources[str(path)] = file_digest(path)
    tasks = []
    for rank in range(WORLD_SIZE):
        argv = [str(root / ".conda/asymvlm/bin/python"), "-u", "-m", "learned_score.mlsys_sprint.phi_adaptive_v4",
            "--experiment", "--phase", stage, "--split", split, "--manifest", study["old_manifest"],
            "--checkpoint", spec["checkpoint"], "--adapter", spec["adapter"], "--model", spec["model_path"],
            "--model-revision", spec["revision"], "--output", str(physical), "--rank", str(rank), "--world-size", str(WORLD_SIZE)]
        if stage != "calibration":
            assignments = root / OLD_ROOT / "phi35" / f"{stage}_assignments.json"
            argv += ["--assignments", str(assignments), "--feature-root", str(root / OLD_ROOT / "phi35" / f"{stage}_features")]
            sources[str(assignments)] = file_digest(assignments)
        tasks.append({"id": f"adaptive-v4-phi-corrected-{stage}-{rank}", "tag": "adaptive-v4", "resource": "gpu",
            "minimum_start_seconds": 300, "depends_on": [], "argv": argv, "env": environment(root),
            "resume_checkpoint": str(physical / f"rank{rank}.metadata.json"), "resume_args": ["--resume"],
            "artifacts": [{"path": str(physical / f"rank{rank}.complete.json")}]})
    return publish_plan(study, f"phi-corrected-{stage}", tasks, sources)


def publish_stage(study, model, stage):
    root, output = Path(study["cwd"]), Path(study["output_root"])
    if stage not in STAGES or model not in MODELS:
        raise ValueError("unknown declared experiment stage")
    if (output / f"plans/{model}-{stage}/plan.json").exists():
        return False
    if model == "phi35" and not phi_gate_passed(study):
        return False
    from .adaptive_experiment_v4 import source_hashes, parse_args
    spec = study["models"][model]
    feature_phase = stage.endswith("_features")
    split = stage.removesuffix("_features") if feature_phase else (
        "fit" if stage == "local_calibration" else "screening" if stage == "benefit_screening" else stage)
    schedule = output / model / "schedules" / f"{stage}.json"
    if not feature_phase and not schedule.is_file():
        return False
    physical = output / model / stage
    sources = {str(root / "learned_score" / path): sha for path, sha in source_hashes(spec["adapter"]).items()}
    for path in (Path(study["manifest"]), Path(spec["checkpoint"])):
        sources[str(path)] = file_digest(path)
    if not feature_phase:
        sources[str(schedule)] = file_digest(schedule)
    if model == "phi35":
        proof = output / "phi_corrected/parity/rank0.complete.json"
        sources[str(proof)] = file_digest(proof)
    tasks = []
    for rank in range(WORLD_SIZE):
        argv = [str(root / ".conda/asymvlm/bin/python"), "-u", "-m", "learned_score.mlsys_sprint.adaptive_experiment_v4",
            "--phase", "features" if feature_phase else "physical", "--split", split,
            "--manifest", study["manifest"], "--checkpoint", spec["checkpoint"],
            "--adapter", spec["adapter"], "--model", spec["model_path"], "--model-revision", spec["revision"],
            "--output", str(physical), "--rank", str(rank), "--world-size", str(WORLD_SIZE)]
        if not feature_phase:
            argv += ["--schedule", str(schedule), "--feature-root", str(output / model / f"{split}_features")]
        parse_args(argv[4:])
        tasks.append({"id": f"adaptive-v4-{model}-{stage.replace('_', '-')}-{rank}", "tag": "adaptive-v4", "resource": "gpu",
            "minimum_start_seconds": 300, "depends_on": [], "argv": argv, "env": environment(root),
            "resume_checkpoint": str(physical / f"rank{rank}.metadata.json"), "resume_args": ["--resume"],
            "artifacts": [{"path": str(physical / f"rank{rank}.complete.json")}]})
    return publish_plan(study, f"{model}-{stage}", tasks, sources)


def skip_benefit(study, model, decision):
    """Resolve the declared optional queue without fabricating physical results."""
    output = Path(study["output_root"])
    marker = output / model / "benefit_screening/skipped.json"
    write_immutable(marker, {"status": "not_triggered", "selection_sha256": decision["sha256"],
        "reason": "entropy met the preregistered physical robustness condition"})
    return publish_plan(study, f"{model}-benefit_screening", [], {str(marker): file_digest(marker)})


def audited_rows(study, stage):
    from .adaptive_data_v4 import load_manifest
    from .adaptive_report_v4 import scan_stage
    output = Path(study["output_root"])
    manifest = load_manifest(study["manifest"])
    rows, schedules = {}, {}
    for model in MODELS:
        schedule = output / model / "schedules" / f"{stage}.json"
        summary, measured = scan_stage(output, model, study["models"][model], manifest,
            stage=stage, schedule_path=schedule, world_size=WORLD_SIZE)
        if summary.get("status") not in {"complete", "complete_feasible_subset"}:
            raise ValueError(f"stage markers exist but physical audit incomplete: {model}/{stage}: {summary}")
        rows[model], schedules[model] = measured, read_json(schedule)
    return manifest, rows, schedules


def select_screening(study, *, include_benefit=False):
    from .adaptive_report_v4 import physical_selection_records
    from .adaptive_allocation_v4 import select_physical_candidate
    output = Path(study["output_root"])
    destination = output / ("final_selection.json" if include_benefit else "screening_selection.json")
    if destination.exists():
        saved = read_json(destination)
        if (saved.get("sha256") != digest({k: v for k, v in saved.items() if k != "sha256"})
                or saved.get("study_sha256") != study["sha256"]):
            raise ValueError("physical selection identity changed")
        return saved
    manifest, rows, schedules = audited_rows(study, "screening")
    panels = physical_selection_records(rows, manifest, schedules)
    if panels["waiting"]:
        raise ValueError(f"incomplete physical selection panels: {panels['waiting']}")
    records = panels["records"]
    if include_benefit:
        _, extra_rows, extra_schedules = audited_rows(study, "benefit_screening")
        # Benefit schedules contain their own uniform and shuffled comparisons;
        # selecting with merged panel records keeps candidate coverage explicit.
        extra = physical_selection_records(extra_rows, manifest, extra_schedules)
        if extra["waiting"]:
            raise ValueError(f"incomplete learned-benefit panels: {extra['waiting']}")
        records += extra["records"]
    selected = select_physical_candidate(records)
    result = seal({**{k: v for k, v in selected.items() if k != "sha256"},
                   "study_sha256": study["sha256"], "physical_panel_records_sha256": digest(records)})
    write_immutable(destination, result)
    write_immutable(destination.with_name(destination.stem + "_panels.json"),
                    seal({"records": records, "selection_sha256": result["sha256"]}))
    return result


def tick(study):
    """Publish ready work only, never inspect evaluation quality for decisions."""
    from . import adaptive_planning_v4 as planning
    output = Path(study["output_root"])
    published = []
    if publish_phi_gate(study):
        published.append("phi-stopping-parity")
    for stage in ("fit_features", "screening_features", "evaluation_features"):
        for model in MODELS:
            if publish_stage(study, model, stage):
                published.append(f"{model}-{stage}")
    for model in MODELS:
        base = output / model
        if ready_stage(base / "fit_features"):
            planning.build_local_schedule(study, model)
            if publish_stage(study, model, "local_calibration"):
                published.append(f"{model}-local_calibration")
        if ready_stage(base / "local_calibration"):
            planning.fit_candidates(study, model)
            if ready_stage(base / "screening_features"):
                planning.build_screening_schedule(study, model)
                if publish_stage(study, model, "screening"):
                    published.append(f"{model}-screening")
    for stage in ("calibration", "screening", "evaluation"):
        if publish_phi_replay(study, stage):
            published.append(f"phi-corrected-{stage}")
    decision = None
    if all(ready_stage(output / model / "screening") for model in MODELS):
        initial = select_screening(study)
        if initial["learned_benefit_branch_needed"]:
            for model in MODELS:
                planning.fit_candidates(study, model, learned=True)
                planning.build_screening_schedule(study, model, learned=True)
                if publish_stage(study, model, "benefit_screening"):
                    published.append(f"{model}-benefit_screening")
            if all(ready_stage(output / model / "benefit_screening") for model in MODELS):
                decision = select_screening(study, include_benefit=True)
        else:
            for model in MODELS:
                skip_benefit(study, model, initial)
            write_immutable(output / "final_selection.json", initial)
            decision = initial
    if decision is not None:
        # Even an unpromoted candidate is physically measured and reported as
        # exploratory. No static fallback can manufacture an adaptive win.
        for model in MODELS:
            if ready_stage(output / model / "evaluation_features"):
                planning.build_evaluation_schedule(study, model, decision["selected_candidate"])
                if publish_stage(study, model, "evaluation"):
                    published.append(f"{model}-evaluation")
    readiness = {model: {stage: ready_stage(output / model / stage) for stage in STAGES} for model in MODELS}
    replay = {stage: ready_stage(output / "phi_corrected" / stage) for stage in ("calibration", "screening", "evaluation")}
    completed = decision is not None and all(readiness[model]["evaluation"] for model in MODELS) and all(replay.values())
    if completed:
        audited_rows(study, "evaluation")
    status = {"time": time.time(), "study_sha256": study["sha256"], "published": published,
        "readiness": readiness, "phi_stopping_parity": phi_gate_passed(study), "phi_corrected_replay": replay,
        "selected_candidate": decision["selected_candidate"] if decision else None,
        "promoted_candidate": decision.get("promoted_candidate") if decision else None,
        "complete": completed, "evidence_status": study["evidence_status"]}
    atomic_json(output / "coordinator/status.json", status)
    return status


def run(study, *, once=False):
    output = Path(study["output_root"])
    (output / "coordinator").mkdir(parents=True, exist_ok=True)
    with (output / "coordinator/workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source_files = ("adaptive_workflow_v4.py", "adaptive_planning_v4.py", "adaptive_allocation_v4.py",
                        "adaptive_report_v4.py", "adaptive_data_v4.py")
        source_hashes = {str(Path(__file__).with_name(name)): file_digest(Path(__file__).with_name(name)) for name in source_files}
        identity = seal({"study_sha256": study["sha256"], "source_sha256": source_hashes})
        write_immutable(output / "coordinator/workflow_identity.json", identity)
        last_report = 0
        while time.time() < study["deadline_utc_epoch"]:
            try:
                if any(file_digest(path) != expected for path, expected in source_hashes.items()):
                    raise ValueError("CPU workflow source changed after launch")
                status = tick(study)
                if status["complete"] or time.time() - last_report >= 120:
                    from .adaptive_report_v4 import write_report
                    write_report(study)
                    last_report = time.time()
            except Exception as error:
                atomic_json(output / "coordinator/error.json", {"time": time.time(), "type": type(error).__name__, "error": str(error)})
                raise
            print(json.dumps(status), flush=True)
            if once or status["complete"]:
                return status
            time.sleep(min(30, max(0, study["deadline_utc_epoch"] - time.time())))
        from .adaptive_report_v4 import write_report
        write_report(study)
        status = {"status": "deadline_reached_partial_results_preserved", "time": time.time(), "complete": False}
        atomic_json(output / "coordinator/deadline.json", status)
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "phi-gate", "phi-replay", "features", "once", "run"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    output = (args.root / args.output).resolve()
    study = prepare(args.root, output) if args.command == "prepare" else read_json(output / "study.json")
    if args.command == "phi-gate":
        publish_phi_gate(study)
    elif args.command == "phi-replay":
        for stage in ("calibration", "screening", "evaluation"):
            publish_phi_replay(study, stage)
    elif args.command == "features":
        for stage in ("fit_features", "screening_features", "evaluation_features"):
            for model in MODELS:
                publish_stage(study, model, stage)
    elif args.command in {"once", "run"}:
        run(study, once=args.command == "once")
    print(json.dumps({"event": args.command, "output": str(output), "deadline_utc_epoch": study["deadline_utc_epoch"]}), flush=True)


if __name__ == "__main__":
    main()
