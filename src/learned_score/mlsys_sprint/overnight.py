"""Bounded overnight continuation of immutable MLSys queues.

Scheduler runs in an authenticated login shell on the cluster. It submits at most one
owned debug allocation at a time, never cancels jobs, and stops on uncertain
submission outcomes. Dispatch shares the original worker's task locks/state;
it can move between queues instead of occupying a GPU while dependencies run.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import time

from .state import atomic_json, digest, file_digest, resolve_inside
from .worker import _command, _completed, _state, read_plan, verify_artifacts, verify_sources


TERMINAL = {"complete", "failed", "blocked"}


def configuration(path):
    config = json.loads(Path(path).read_text())
    if config.get("sha256") != digest({k: v for k, v in config.items() if k != "sha256"}):
        raise ValueError("overnight configuration changed")
    if not 1 <= config["maximum_submissions"] <= 12:
        raise ValueError("overnight submission cap must be 1..12")
    if config["deadline_utc_epoch"] <= config["created_utc_epoch"] or config["deadline_utc_epoch"] - config["created_utc_epoch"] > 12 * 3600:
        raise ValueError("overnight window must be at most twelve hours")
    for name in ("status_path", "journal_path", "scheduler_lock", "pbs_script"):
        resolve_inside(config[name], config["cwd"])
    for source, expected in config["input_hashes"].items():
        if file_digest(Path(config["cwd"]) / source) != expected:
            raise ValueError(f"overnight source changed: {source}")
    return config


def queues(config, cluster):
    result = []
    for item in config["queues"]:
        if cluster not in item.get("clusters", ["clusterb", "clustera"]):
            continue
        path = resolve_inside(item["plan"], config["cwd"])
        plan = read_plan(path)
        if plan["sha256"] != item["sha256"]:
            raise ValueError("queue plan differs from overnight configuration")
        if Path(plan["cwd"]).resolve() != Path(config["cwd"]).resolve():
            raise ValueError("queue uses a different workspace")
        verify_sources(plan)
        state = resolve_inside(item["state_root"], config["cwd"])
        tags = item.get("tags", {}).get(cluster)
        result.append((item, plan, state, tags))
    return result


def dependency_ready(task, plan, state_root):
    mapping = {t["id"]: t for t in plan["tasks"]}
    for dep in task.get("depends_on", []):
        state = _state(state_root / (dep + ".json"))
        if state and state.get("plan_sha256") != plan["sha256"]:
            raise ValueError("dependency state belongs to another plan")
        if state.get("status") in {"failed", "blocked"}:
            return False, True
        if not _completed(mapping[dep], state, plan):
            return False, False
    return True, False


def claim(queue_list, *, dry_run=False, remaining_seconds=None):
    """Return the first unlocked runnable task. Caller owns returned lock."""
    for item, plan, state_root, tags in queue_list:
        state_root.mkdir(parents=True, exist_ok=True)
        for task in plan["tasks"]:
            if tags is not None and task.get("tag") not in tags:
                continue
            if remaining_seconds is not None and remaining_seconds < task.get("minimum_start_seconds", 10):
                continue
            lock = (state_root / (task["id"] + ".lock")).open("a+")
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                continue
            try:
                state_path = state_root / (task["id"] + ".json")
                state = _state(state_path)
                if state and state.get("plan_sha256") != plan["sha256"]:
                    raise ValueError("state belongs to another plan")
                if _completed(task, state, plan) or state.get("status") in {"failed", "blocked"}:
                    lock.close()
                    continue
                ready, blocked = dependency_ready(task, plan, state_root)
                if blocked and not dry_run:
                    atomic_json(state_path, {"status": "blocked", "reason": "failed_dependency", "plan_sha256": plan["sha256"]})
                if not ready:
                    lock.close()
                    continue
                return task, plan, state_root, state, lock
            except BaseException:
                lock.close()
                raise
    return None


def run_claim(selected, deadline, worker_id, stopping):
    task, plan, state_root, state, lock = selected
    state_path = state_root / (task["id"] + ".json")
    try:
        if deadline - time.monotonic() < task.get("minimum_start_seconds", 10):
            return
        verify_sources(plan)
        failures = int(state.get("failures", 0)) + int(state.get("status") == "running")
        if failures > plan.get("unexpected_retries", 1):
            atomic_json(state_path, {"status": "failed", "reason": "retry_exhausted", "failures": failures, "plan_sha256": plan["sha256"]})
            return
        argv, resumed = _command(task, plan["cwd"])
        log = state_root / (task["id"] + ".log")
        record = dict(status="running", plan_sha256=plan["sha256"], failures=failures,
                      worker=worker_id, host=socket.gethostname(), pbs_job_id=os.getenv("PBS_JOBID"),
                      started=time.time(), argv=argv, log=str(log), resumed_from=resumed)
        atomic_json(state_path, record)
        interrupted = False
        process = None
        error = None
        with log.open("ab") as stream:
            try:
                process = subprocess.Popen(argv, cwd=plan["cwd"], env=dict(os.environ, **task.get("env", {})),
                                           stdout=stream, stderr=subprocess.STDOUT,
                                           start_new_session=True, pass_fds=(lock.fileno(),))
            except OSError as exc:
                error = f"process_launch_failed: {exc}"
            while process is not None and process.poll() is None:
                if stopping() or time.monotonic() >= deadline:
                    interrupted = True
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                        process.wait(timeout=20)
                    except ProcessLookupError:
                        pass
                    except subprocess.TimeoutExpired:
                        try:
                            os.killpg(process.pid, signal.SIGKILL)
                        except ProcessLookupError:
                            pass
                        process.wait()
                    break
                time.sleep(.25)
        record.update(finished=time.time(), returncode=process.returncode if process else None)
        if interrupted:
            record.update(status="interrupted", reason="worker_deadline_or_signal")
        elif error is not None or process.returncode != 0:
            failures += 1
            record.update(status="failed" if failures > plan.get("unexpected_retries", 1) else "retry", failures=failures)
            if error:
                record["reason"] = error
        else:
            try:
                verify_sources(plan)
                record.update(artifact_hashes=verify_artifacts(task, plan["cwd"]), status="complete")
            except (ValueError, OSError) as exc:
                record.update(status="failed", reason=str(exc))
        atomic_json(state_path, record)
        print(json.dumps({"task": task["id"], "status": record["status"], "worker": worker_id}), flush=True)
    finally:
        lock.close()


def dispatch(config, cluster, seconds, worker_id):
    if seconds <= 0:
        raise ValueError("worker deadline must be positive")
    stop = [False]
    def handler(signum, frame):
        stop[0] = True
    previous = {s: signal.signal(s, handler) for s in (signal.SIGTERM, signal.SIGINT)}
    deadline = time.monotonic() + min(seconds, max(0, config["deadline_utc_epoch"] - time.time()))
    queue_list = queues(config, cluster)
    try:
        while not stop[0] and time.monotonic() < deadline:
            selected = claim(queue_list, remaining_seconds=deadline - time.monotonic())
            if selected is None:
                # Stage gates may unlock another queue shortly. Keep ranks in
                # an existing allocation available, with no model resident,
                # rather than lose these GPUs for the rest of its one-hour run.
                pending = any(_state(state / (task["id"] + ".json")).get("status", "pending")
                              not in TERMINAL for _, plan, state, tags in queue_list
                              for task in plan["tasks"] if tags is None or task.get("tag") in tags)
                if not pending or deadline - time.monotonic() < 10:
                    return
                time.sleep(min(5, max(0, deadline - time.monotonic())))
                continue
            run_claim(selected, deadline, worker_id, lambda: stop[0])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def owned_jobs(payload, user):
    jobs = payload.get("Jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError("malformed PBS response")
    return {job: value for job, value in jobs.items()
            if value.get("Job_Owner", "").split("@", 1)[0] == user
            and value.get("Job_Name", "").startswith("mlsys-")
            and value.get("job_state") not in {"F", "X", "C"}}


def assigned_debug_nodes(payload, user):
    """All workflows of the account count toward the two-running-node budget."""
    count = 0
    for value in payload.get("Jobs", {}).values():
        if (value.get("Job_Owner", "").split("@", 1)[0] != user or value.get("queue") != "debug"
                or value.get("job_state") in {"F", "X", "C"}
                or not (value.get("exec_host") or value.get("job_state") in {"R", "E", "S"})):
            continue
        resources = value.get("Resource_List", {})
        nodes = resources.get("nodect")
        if nodes is None:
            chunks = str(resources.get("select", "")).split("+")
            counts = [chunk.split(":", 1)[0] for chunk in chunks]
            if not all(item.isdigit() for item in counts):
                raise ValueError("cannot verify other debug allocation node count")
            nodes = sum(int(item) for item in counts)
        if int(nodes) < 1:
            raise ValueError("invalid assigned debug node count")
        count += int(nodes)
    return count


def propagate_blocked(queue_list):
    """Close impossible dependencies even when no GPU allocation remains."""
    for item, plan, root, tags in queue_list:
        for task in plan["tasks"]:
            path = root / (task["id"] + ".json")
            state = _state(path)
            if state.get("status") in TERMINAL or state.get("status") == "running":
                continue
            if state and state.get("plan_sha256") != plan["sha256"]:
                raise ValueError("state belongs to another plan")
            ready, blocked = dependency_ready(task, plan, root)
            if not blocked:
                continue
            root.mkdir(parents=True, exist_ok=True)
            with (root / (task["id"] + ".lock")).open("a+") as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                current = _state(path)
                if current == state:
                    atomic_json(path, {"status": "blocked", "reason": "failed_dependency", "plan_sha256": plan["sha256"]})


def clusterb_successor(config, gpu):
    """Wait for an exact predecessor and all GPU processes; never interrupt them."""
    settings = config["clusterb"]
    if socket.gethostname().split(".")[0] != settings["host"] or os.getenv("PBS_JOBID") != settings["job"]:
        raise ValueError("successor must run inside the explicitly owned ClusterB allocation")
    item = settings["predecessors"][str(gpu)]
    proc = Path("/proc") / str(item["pid"])
    def predecessor_alive():
        try:
            return proc.stat().st_uid == os.getuid() and (proc / "stat").read_text().split()[21] == item["start_ticks"]
        except FileNotFoundError:
            return False
    # PID reuse is not enough to claim a predecessor. Validate exact start time,
    # command, GPU and PBS ownership before waiting on it.
    if predecessor_alive():
        env = dict(x.split("=", 1) for x in (proc / "environ").read_bytes().decode().split("\0") if "=" in x)
        command = (proc / "cmdline").read_bytes().decode().split("\0")
        if env.get("PBS_JOBID") != settings["job"] or env.get("CUDA_VISIBLE_DEVICES") != str(gpu) or "learned_score.mlsys_sprint.worker" not in command:
            raise ValueError("predecessor ownership or GPU mismatch")
    guard_path = resolve_inside(settings["gpu_lock_root"], config["cwd"]) / f"gpu{gpu}.lock"
    guard_path.parent.mkdir(parents=True, exist_ok=True)
    with guard_path.open("a+") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < config["deadline_utc_epoch"]:
            if predecessor_alive():
                time.sleep(30)
                continue
            # Any compute process, including an orphan or unrelated workload,
            # prevents sharing. We do not infer ownership from utilization.
            check = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=20, check=True)
            if check.stdout.strip():
                time.sleep(30)
                continue
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
            dispatch(config, "clusterb", max(1, config["deadline_utc_epoch"] - time.time()),
                     f"clusterb-overnight-{settings['job']}-{gpu}")
            # Dependency completion elsewhere can make new tasks ready later.
            time.sleep(30)


def queue_snapshot(queue_list):
    result = {}
    for item, plan, state_root, tags in queue_list:
        counts = {}
        for task in plan["tasks"]:
            status = _state(state_root / (task["id"] + ".json")).get("status", "pending")
            counts[status] = counts.get(status, 0) + 1
        result[str(item["plan"])] = counts
    return result


def scheduler(config, *, dry_run=False):
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("scheduler must run through the authenticated ClusterA login session")
    root = Path(config["cwd"])
    lock_path = resolve_inside(config["scheduler_lock"], root)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = root / config["journal_path"]
        journal = _state(journal_path) or {"config_sha256": config["sha256"], "submissions": []}
        if journal["config_sha256"] != config["sha256"] or journal.get("submission_intent"):
            raise ValueError("unresolved submission intent or changed configuration; manual audit required")
        qlist = queues(config, "clustera")
        while time.time() < config["deadline_utc_epoch"]:
            response = subprocess.run(["qstat", "-f", "-F", "json"], capture_output=True, text=True, timeout=45, check=True)
            payload = json.loads(response.stdout)
            jobs = owned_jobs(payload, config["user"])
            debug_nodes = assigned_debug_nodes(payload, config["user"])
            if len(jobs) > 1:
                raise RuntimeError("multiple owned allocations; refuse further submissions")
            if not dry_run:
                propagate_blocked(qlist)
            selected = claim(qlist, dry_run=True)
            ready = selected is not None
            if selected is not None:
                selected[-1].close()
            status = {"time": time.time(), "config_sha256": config["sha256"], "owned_jobs": jobs,
                      "ready_work": ready, "assigned_debug_nodes_all_user_workflows": debug_nodes,
                      "queues": queue_snapshot(qlist), "submissions": journal["submissions"]}
            atomic_json(root / config["status_path"], status)
            print(json.dumps({k: v for k, v in status.items() if k != "owned_jobs"}, sort_keys=True), flush=True)
            if dry_run:
                return
            if not jobs and debug_nodes == 0 and ready and len(journal["submissions"]) < config["maximum_submissions"]:
                if time.time() + 3600 > config["deadline_utc_epoch"]:
                    return  # Do not launch a job that could exceed the window.
                for source, expected in config["input_hashes"].items():
                    if file_digest(root / source) != expected:
                        raise ValueError(f"overnight source changed before submission: {source}")
                journal["submission_intent"] = {"time": time.time(), "script": config["pbs_script"]}
                atomic_json(journal_path, journal)
                # A timeout/rejection leaves the intent in place. Never retry
                # an uncertain submission and accidentally duplicate jobs.
                submitted = subprocess.run(["qsub", config["pbs_script"]], cwd=root,
                                           capture_output=True, text=True, timeout=60, check=True)
                job = submitted.stdout.strip()
                if not re.fullmatch(r"\d+\.clustera-pbs-[A-Za-z0-9.-]+", job):
                    raise ValueError("unrecognized qsub response; manual audit required")
                journal["submissions"].append({"job": job, "submitted": time.time()})
                journal.pop("submission_intent")
                atomic_json(journal_path, journal)
            counts = [counts for counts in status["queues"].values()]
            if not jobs and all(not any(c.get(k, 0) for k in ("pending", "running", "retry", "interrupted")) for c in counts):
                return
            time.sleep(60)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("scheduler", "dispatch", "check", "clusterb-successor"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cluster", choices=("clusterb", "clustera"), default="clustera")
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--worker-id", default="overnight")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpu", type=int, choices=range(8))
    args = parser.parse_args()
    config = configuration(args.config)
    if args.command == "scheduler":
        scheduler(config, dry_run=args.dry_run)
    elif args.command == "dispatch":
        dispatch(config, args.cluster, args.seconds, args.worker_id)
    elif args.command == "clusterb-successor":
        if args.gpu is None:
            parser.error("--gpu is required for clusterb-successor")
        clusterb_successor(config, args.gpu)
    else:
        qlist = queues(config, args.cluster)
        print(json.dumps(queue_snapshot(qlist), indent=2))


if __name__ == "__main__":
    main()
