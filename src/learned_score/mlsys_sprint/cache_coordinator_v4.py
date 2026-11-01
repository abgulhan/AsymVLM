"""Restartable v4 pilot coordination without modifying earlier experiments.

GPU dispatch and the bounded, nonce-journalled PBS scheduler are reused as
immutable utilities. The v4 ClusterA coordinator yields to the running v3
scheduler; it never cancels another allocation or opens reserved final labels.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .cache_protocol_v4 import build_stage_protocol, load_protocol
from .followup_common import write_immutable
from .prepare_cache_methods_v4 import build_plan
from .state import atomic_json, digest, file_digest
from . import sprint48


STAGES = ("stage1", "stage2", "stage3")


def stage_directory(root, stage):
    if stage not in STAGES:
        raise ValueError("unknown development stage")
    return Path(root) if stage == "stage1" else Path(root) / stage


def make_configs(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    root, cwd = protocol_path.parent, Path(protocol["workspace"])
    created = protocol["created_utc_epoch"]
    paths = [str(stage_directory(root, stage) / "plans/qwen25/plan.json") for stage in STAGES]
    scripts = ("scripts/mlsys_cache_methods_v4_clustera.pbs", "scripts/mlsys_cache_methods_v4_clustera_worker.sh",
               "scripts/mlsys_cache_methods_v4_report.sh", "scripts/mlsys_cache_methods_v4_interactive.sh")
    pins = {**protocol["input_hashes"], str(protocol_path): file_digest(protocol_path),
            **{str(cwd / p): file_digest(cwd / p) for p in scripts}}
    result = {}
    for role in ("cpu", "clustera", "clusterb"):
        config = {"format": sprint48.FORMAT, "coordinator_role": role, "cwd": str(cwd),
                  "created_utc_epoch": created, "deadline_utc_epoch": created + 48 * 3600,
                  "maximum_debug_nodes": 2, "maximum_submissions": 8,
                  "pbs_script": scripts[0], "plan_paths": paths, "input_hashes": pins,
                  "state_root": str(root / "coordinator" / f"{role}_state"), "user": os.environ.get("USER", "user"),
                  "clusterb_jobs": {}, "v4_protocol": str(protocol_path),
                  "predecessor_clustera_lock": str(Path(protocol["parent_protocol"]).parent /
                                                  "coordinator/clustera_state/scheduler.lock")}
        config["sha256"] = digest(config)
        destination = root / "coordinator" / f"{role}-config.json"
        write_immutable(destination, config)
        result[role] = str(destination)
    return result


def failed_tasks(root):
    failed = []
    for stage in STAGES:
        directory = stage_directory(root, stage) / "plans/qwen25/state"
        for path in directory.glob("*.json"):
            value = json.loads(path.read_text())
            if value.get("status") in {"failed", "blocked"}:
                failed.append({"stage": stage, "task": path.stem, "status": value["status"]})
    return failed


def publish_next(previous_path, stage, decision_path):
    previous_path, decision_path = Path(previous_path), Path(decision_path)
    previous = load_protocol(previous_path)
    decision = json.loads(decision_path.read_text())
    protocol = build_stage_protocol(previous, stage, decision["selection"], selection_path=decision_path)
    # Stage1 is the root; later protocols keep the same experiment output root.
    root = Path(previous["output_root"])
    output = stage_directory(root, stage)
    write_immutable(output / "protocol.json", protocol)
    for model in protocol["active_models"]:
        plan = build_plan(protocol, output, Path(protocol["workspace"]), model)
        write_immutable(output / "plans" / model / "plan.json", plan)
    return output / "protocol.json"


def refresh(protocol_path, *, repetitions=10000):
    from .cache_methods_v4 import collect_panel
    from learned_score.mlsys_next.cache_report_v4 import report_stage
    protocol_path = Path(protocol_path).resolve()
    protocol = load_protocol(protocol_path)
    root = protocol_path.parent
    state = {"format": "mlsys_cache_history_workflow_v4", "protocol_sha256": protocol["sha256"],
             "updated_epoch": time.time(), "final_labels_opened": False, "complete": False,
             "relative_quality_floor": .95, "stages": {}}
    failures = failed_tasks(root)
    if failures:
        state.update(status="failed_tasks_require_inspection", failed_tasks=failures)
    else:
        for index, stage in enumerate(STAGES):
            directory = stage_directory(root, stage)
            path = directory / "protocol.json"
            if not path.exists():
                state["stages"][stage] = {"status": "not_published"}
                break
            panel = collect_panel(path, require_complete=False)
            state["stages"][stage] = {"completed_pages": panel["completed_pages"], "expected_pages": 24,
                                      "status": "generating" if not panel["complete"] else "scoring"}
            if not panel["complete"]:
                break
            report = report_stage(path, directory, stage, repetitions=repetitions)
            state["stages"][stage].update(status="complete", report_complete=True)
            if stage != "stage3":
                decision_path = directory / "selection.json"
                if not decision_path.exists():
                    raise ValueError("complete stage report did not publish its immutable selection")
                next_path = stage_directory(root, STAGES[index + 1]) / "protocol.json"
                if not next_path.exists():
                    publish_next(path, STAGES[index + 1], decision_path)
            else:
                state.update(complete=True, status="development_pilot_complete",
                             confirmation_status="requires_separate_identity_audited_source_panel")
        state.setdefault("status", "running")
    atomic_json(root / "coordinator/workflow-status.json", state)
    return state


def watch(protocol_path, *, once=False, repetitions=10000):
    protocol_path = Path(protocol_path).resolve()
    configs = make_configs(protocol_path)
    root = protocol_path.parent
    state_root = root / "coordinator"
    with (state_root / "workflow.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = sprint48.read_config(configs["cpu"])
        child = None
        log = None
        if not once:
            log = (state_root / "cpu.log").open("a")
            child = subprocess.Popen([sys.executable, "-u", "-m", "learned_score.mlsys_sprint.sprint48", "dispatch",
                                      "--config", configs["cpu"], "--resource", "cpu", "--seconds", "172800",
                                      "--worker-id", "cache-v4-cpu"], stdout=log, stderr=subprocess.STDOUT)
        try:
            while time.time() < config["deadline_utc_epoch"]:
                if child is not None and child.poll() is not None and child.returncode != 0:
                    raise RuntimeError(f"v4 CPU dispatcher exited {child.returncode}")
                status = refresh(protocol_path, repetitions=repetitions)
                print(json.dumps(status), flush=True)
                if once or status["complete"] or status.get("failed_tasks"):
                    return status
                time.sleep(30)
            status = {"status": "deadline", "complete": False, "final_labels_opened": False}
            atomic_json(state_root / "workflow-status.json", status)
            return status
        finally:
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    # Do not kill subprocess trees or unrelated workers.
                    atomic_json(state_root / "cpu-shutdown-pending.json", {"pid": child.pid, "status": "termination_requested"})
            if log is not None:
                log.close()


def clustera(config_path, *, dry_run=False):
    from .sprint48_scheduler_v3 import scheduler
    config = sprint48.read_config(config_path)
    lock_path = Path(config["predecessor_clustera_lock"])
    # The existing v3 scheduler keeps this lock for its entire lifetime. Wait
    # without stopping it; once it exits, hold the lock while v4 submits so a
    # predecessor restart cannot race the same two-node allowance.
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as predecessor:
        while time.time() < config["deadline_utc_epoch"]:
            try:
                fcntl.flock(predecessor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                atomic_json(Path(config["state_root"]) / "scheduler_status_v4.json",
                            {"status": "yielding_to_existing_v3_scheduler", "updated_epoch": time.time()})
                if dry_run:
                    return "yielding_to_existing_v3_scheduler"
                time.sleep(30)
        else:
            return "deadline"
        return scheduler(config, dry_run=dry_run)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "watch"):
        item = sub.add_parser(name)
        item.add_argument("--protocol", type=Path, required=True)
        if name == "watch":
            item.add_argument("--once", action="store_true")
            item.add_argument("--repetitions", type=int, default=10000)
    item = sub.add_parser("clustera")
    item.add_argument("--config", type=Path, required=True)
    item.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = make_configs(args.protocol)
    elif args.command == "watch":
        result = watch(args.protocol, once=args.once, repetitions=args.repetitions)
    else:
        result = clustera(args.config, dry_run=args.dry_run)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
