"""Cooperative, restartable work queue shared by dedicated PBS allocations.

Each process owns one GPU. POSIX locks protect tasks, including after crashes;
completed artifacts are hashed and checked on resume. No scheduler operations
or other users' processes are touched by this module.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .state import atomic_json, digest, file_digest, resolve_inside


def read_plan(path):
    plan = json.loads(Path(path).read_text())
    if plan.get("sha256") != digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise ValueError("task plan changed after creation")
    ids = [task["id"] for task in plan["tasks"]]
    if plan.get("unexpected_retries", 1) not in (0, 1):
        raise ValueError("unexpected failures permit at most one retry")
    if len(ids) != len(set(ids)) or any(not s or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in s) for s in ids):
        raise ValueError("invalid or duplicate task IDs")
    for task in plan["tasks"]:
        if not isinstance(task["argv"], list) or not task["argv"] or not all(isinstance(x, str) for x in task["argv"]):
            raise ValueError("commands must be argument lists")
        environment = task.get("env", {})
        if not isinstance(environment, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in environment.items()):
            raise ValueError("task environment must map strings to strings")
        if any(dep not in ids for dep in task.get("depends_on", [])):
            raise ValueError("unknown task dependency")
        if task.get("minimum_start_seconds", 10) < 0:
            raise ValueError("minimum_start_seconds must be nonnegative")
        if "resume_checkpoint" in task:
            if not isinstance(task["resume_checkpoint"], str) or not task["resume_checkpoint"]:
                raise ValueError("resume_checkpoint must be a path string")
            if not isinstance(task.get("resume_args"), list) or not task["resume_args"] or not all(isinstance(x, str) for x in task["resume_args"]):
                raise ValueError("resume_checkpoint requires an explicit resume_args argument list")
    # Cycles must fail, not occupy idle GPUs indefinitely.
    visited = set()
    while len(visited) < len(ids):
        ready = {t["id"] for t in plan["tasks"] if set(t.get("depends_on", [])) <= visited}
        if ready <= visited:
            raise ValueError("cyclic task dependencies")
        visited |= ready
    return plan


def verify_sources(plan):
    for path, expected in plan.get("input_hashes", {}).items():
        if file_digest(Path(plan["cwd"]) / path) != expected:
            raise ValueError(f"experiment input changed: {path}")


def verify_artifacts(task, cwd):
    hashes = {}
    for item in task.get("artifacts", []):
        path = Path(cwd) / item["path"]
        if not path.is_file():
            raise ValueError(f"missing task artifact: {path}")
        if item.get("passed") and json.loads(path.read_text()).get("passed") is not True:
            raise ValueError(f"correctness gate failed: {path}")
        hashes[str(path)] = file_digest(path)
    return hashes


def _state(path):
    return json.loads(path.read_text()) if path.exists() else {}


def _completed(task, state, plan):
    if state.get("status") != "complete" or state.get("plan_sha256") != plan["sha256"]:
        return False
    if state.get("artifact_hashes") != verify_artifacts(task, plan["cwd"]):
        raise ValueError(f"completed task artifacts were modified: {task['id']}")
    return True


def _command(task, cwd):
    """Append explicit resume arguments only when this task's checkpoint exists."""
    argv = list(task["argv"])
    checkpoint = task.get("resume_checkpoint")
    if checkpoint is None:
        return argv, None
    path = resolve_inside(checkpoint, cwd)
    if path.exists():
        if not path.is_file():
            raise ValueError(f"resume checkpoint is not a file: {path}")
        argv.extend(value.replace("{resume_checkpoint}", str(path)) for value in task["resume_args"])
        return argv, {"path": str(path), "sha256": file_digest(path)}
    return argv, None


def run_worker(plan_path, state_root, *, seconds=3000, worker_id="local", tags=None):
    if seconds <= 0:
        raise ValueError("worker deadline must be positive")
    plan = read_plan(plan_path)
    verify_sources(plan)
    state_root = Path(state_root)
    state_root.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + seconds
    selected = [t for t in plan["tasks"] if not tags or t.get("tag") in tags]
    tasks_by_id = {task["id"]: task for task in plan["tasks"]}
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True

    old_handlers = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopped and time.monotonic() < deadline:
            unfinished = False
            progressed = False
            for task in selected:
                state_path = state_root / (task["id"] + ".json")
                with (state_root / (task["id"] + ".lock")).open("a+") as lock:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        unfinished = True
                        continue
                    state = _state(state_path)
                    if _completed(task, state, plan):
                        continue
                    if state and state.get("plan_sha256") != plan["sha256"]:
                        raise ValueError("state directory belongs to another plan")
                    if state.get("status") in {"failed", "blocked"}:
                        continue
                    dependency_ids = task.get("depends_on", [])
                    dependencies = [_state(state_root / (dep + ".json")) for dep in dependency_ids]
                    if any(d and d.get("plan_sha256") != plan["sha256"] for d in dependencies):
                        raise ValueError("dependency state belongs to another plan")
                    if any(d.get("status") in {"failed", "blocked"} for d in dependencies):
                        atomic_json(state_path, {"status": "blocked", "reason": "failed_dependency", "plan_sha256": plan["sha256"]})
                        continue
                    # Tags may exclude a prerequisite from this worker's own
                    # task list. Always recheck its artifacts before trusting it.
                    if not all(_completed(tasks_by_id[dep], d, plan) for dep, d in zip(dependency_ids, dependencies)):
                        unfinished = True
                        continue
                    if stopped or deadline - time.monotonic() < task.get("minimum_start_seconds", 10):
                        continue
                    unfinished = True
                    verify_sources(plan)
                    failures = int(state.get("failures", 0))
                    # Parent and child must both exit before the inherited lock
                    # releases. Count the unexpected parent death on reclaim.
                    if state.get("status") == "running":
                        failures += 1
                    if failures > plan.get("unexpected_retries", 1):
                        atomic_json(state_path, {"status": "failed", "reason": "retry_exhausted", "failures": failures, "plan_sha256": plan["sha256"]})
                        continue
                    log_path = state_root / (task["id"] + ".log")
                    argv, resumed_from = _command(task, plan["cwd"])
                    record = {"status": "running", "plan_sha256": plan["sha256"], "failures": failures,
                              "worker": worker_id, "host": socket.gethostname(), "pbs_job_id": os.getenv("PBS_JOBID"),
                              "started": time.time(), "argv": argv, "log": str(log_path), "resumed_from": resumed_from}
                    atomic_json(state_path, record)
                    environment = dict(os.environ, **task.get("env", {}))
                    interrupted = False
                    launch_error = None
                    process = None
                    with log_path.open("ab") as stream:
                        try:
                            process = subprocess.Popen(argv, cwd=plan["cwd"], env=environment,
                                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                                       # The child holds the same flock if its parent
                                                       # crashes, preventing a simultaneous retry.
                                                       pass_fds=(lock.fileno(),))
                        except OSError as error:
                            launch_error = f"{type(error).__name__}: {error}"
                        while process is not None and process.poll() is None:
                            if stopped or time.monotonic() >= deadline:
                                interrupted = True
                                try:
                                    os.killpg(process.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                                try:
                                    process.wait(timeout=20)
                                except subprocess.TimeoutExpired:
                                    try:
                                        os.killpg(process.pid, signal.SIGKILL)
                                    except ProcessLookupError:
                                        pass
                                    process.wait()
                                break
                            time.sleep(0.25)
                    record.update(finished=time.time(), returncode=process.returncode if process is not None else None)
                    if interrupted:
                        record.update(status="interrupted", reason="worker_deadline_or_signal")
                    elif launch_error is not None or process.returncode != 0:
                        failures += 1
                        record.update(status="failed" if failures > plan.get("unexpected_retries", 1) else "retry", failures=failures)
                        if launch_error is not None:
                            record["reason"] = "process_launch_failed: " + launch_error
                    else:
                        try:
                            verify_sources(plan)
                            record.update(artifact_hashes=verify_artifacts(task, plan["cwd"]), status="complete")
                        except (ValueError, OSError) as error:
                            record.update(status="failed", reason=str(error))
                    atomic_json(state_path, record)
                    progressed = True
                    if interrupted:
                        break
            if not unfinished:
                break
            if not progressed:
                time.sleep(min(2, max(0, deadline - time.monotonic())))
        states = {t["id"]: _state(state_root / (t["id"] + ".json")).get("status", "pending") for t in selected}
        print(json.dumps({"worker": worker_id, "states": states}), flush=True)
        return 1 if any(s in {"failed", "blocked"} for s in states.values()) else 0
    finally:
        for sig, handler in old_handlers.items():
            signal.signal(sig, handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--worker-id", default="local")
    parser.add_argument("--tags", nargs="+")
    args = parser.parse_args()
    raise SystemExit(run_worker(args.plan, args.state_root, seconds=args.seconds, worker_id=args.worker_id, tags=args.tags))


if __name__ == "__main__":
    main()
