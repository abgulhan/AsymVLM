"""Bounded 48-hour coordinator for independently frozen follow-up queues.

Only the explicitly configured plan paths may be published. A first-seen plan
hash is pinned in a locked ledger and can never be replaced. Existing frozen
workers retain ownership of task locks, atomic markers and one failure retry.
"""

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import re

from . import overnight
from .state import atomic_json, digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources, _state


FORMAT = "mlsys_48_hour_coordinator_v1"
RESOURCES = {"cpu", "gpu", "isolated_gpu"}


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config.get("format") != FORMAT or config.get("sha256") != digest({k: v for k, v in config.items() if k != "sha256"}):
        raise ValueError("invalid or modified sprint configuration")
    window = config["deadline_utc_epoch"] - config["created_utc_epoch"]
    if not 0 < window <= 48 * 3600 or not 1 <= config["maximum_submissions"] <= 48:
        raise ValueError("sprint exceeds approved duration/submission bounds")
    if len(config["plan_paths"]) != len(set(config["plan_paths"])):
        raise ValueError("duplicate plan publication path")
    for name in ("state_root", "pbs_script", *config["plan_paths"]):
        resolve_inside(config[name] if name in config else name, config["cwd"])
    for path, expected in config["input_hashes"].items():
        if file_digest(Path(config["cwd"]) / path) != expected:
            raise ValueError(f"sprint source changed: {path}")
    return config


def load_queues(config, resource):
    """Discover only approved paths, binding each immutable plan once."""
    if resource not in {"cpu", "gpu", "all"}:
        raise ValueError("unknown worker resource")
    root = Path(config["cwd"])
    state = resolve_inside(config["state_root"], root)
    state.mkdir(parents=True, exist_ok=True)
    result, missing = [], []
    with (state / "catalog.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = _state(state / "catalog.json") or {"configuration_sha256": config["sha256"], "plans": {}}
        if ledger["configuration_sha256"] != config["sha256"]:
            raise ValueError("catalog belongs to a different sprint")
        changed = False
        for relative in config["plan_paths"]:
            path = resolve_inside(relative, root)
            if not path.exists():
                if relative in ledger["plans"]:
                    raise ValueError("published plan was removed")
                missing.append(relative)
                continue
            plan = read_plan(path)
            if Path(plan["cwd"]).resolve() != root.resolve():
                raise ValueError("queue uses another workspace")
            verify_sources(plan)
            expected = ledger["plans"].get(relative)
            if expected is not None and expected != plan["sha256"]:
                raise ValueError(f"published plan changed: {relative}")
            if expected is None:
                ledger["plans"][relative] = plan["sha256"]
                changed = True
            tag_resources = {}
            for task in plan["tasks"]:
                kind = task.get("resource", "gpu")
                if kind not in RESOURCES or not task.get("tag"):
                    raise ValueError("every task requires a valid resource and tag")
                previous = tag_resources.setdefault(task["tag"], kind)
                if previous != kind:
                    raise ValueError("a task tag cannot mix CPU and GPU resource classes")
            tags = [tag for tag, kind in tag_resources.items()
                    if resource == "all" or (resource == "cpu") == (kind == "cpu")]
            item = {"plan": relative, "sha256": plan["sha256"]}
            result.append((item, plan, path.parent / "state", tags))
        if changed or not (state / "catalog.json").exists():
            atomic_json(state / "catalog.json", ledger)
    return result, missing


def all_terminal(queues, missing):
    return not missing and all(
        _state(state / (task["id"] + ".json")).get("status") in overnight.TERMINAL
        for _, plan, state, tags in queues for task in plan["tasks"] if task["tag"] in tags)


def guard(mode, argv):
    """Hold a node lease in both launcher and child, including after a crash."""
    if mode not in {"shared", "exclusive"} or not argv:
        raise ValueError("node guard requires a mode and command")
    path = f"/tmp/mlsys48h-node-{os.getuid()}.lock"
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        if os.fstat(fd).st_uid != os.getuid():
            raise ValueError("node guard is owned by another user")
        fcntl.flock(fd, fcntl.LOCK_EX if mode == "exclusive" else fcntl.LOCK_SH)
        device = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not device.isdigit():
            raise ValueError("one explicit GPU is required")
        query = ["nvidia-smi"]
        if mode == "shared":
            query += ["-i", device]
        query += ["--query-compute-apps=pid", "--format=csv,noheader,nounits"]
        # Other GPU processes are never interrupted. The outer task deadline
        # bounds this wait; node isolation is mandatory for timing claims.
        while subprocess.check_output(query, text=True, timeout=20).strip():
            time.sleep(5)
        if mode == "exclusive":
            os.environ["ASYMVLM_ISOLATED_NODE"] = "1"
            os.environ["ASYMVLM_ISOLATION_LEASE"] = digest(
                [socket.gethostname(), os.getpid(), time.time_ns(), os.environ.get("PBS_JOBID")])
        else:
            os.environ.pop("ASYMVLM_ISOLATED_NODE", None)
            os.environ.pop("ASYMVLM_ISOLATION_LEASE", None)
        os.set_inheritable(fd, True)
        # Inherit the existing worker's task lock as well as this node lease.
        process = subprocess.Popen(argv, close_fds=False)
        return process.wait()
    finally:
        os.close(fd)


def dispatch(config, resource, seconds, worker_id):
    if seconds <= 0:
        raise ValueError("worker deadline must be positive")
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    deadline = time.monotonic() + min(seconds, max(0, config["deadline_utc_epoch"] - time.time()))
    try:
        while not stop[0] and time.monotonic() < deadline:
            queues, missing = load_queues(config, resource)
            overnight.propagate_blocked(queues)
            selected = overnight.claim(queues, remaining_seconds=deadline - time.monotonic())
            if selected is None:
                if all_terminal(queues, missing):
                    return
                time.sleep(min(5, max(0, deadline - time.monotonic())))
                continue
            task, plan, state_root, state, lock = selected
            if resource == "gpu":
                task = dict(task)
                mode = "exclusive" if task.get("resource") == "isolated_gpu" else "shared"
                task["argv"] = [sys.executable, "-u", "-m", "learned_score.mlsys_sprint.sprint48",
                                "guard", "--mode", mode, "--", *task["argv"]]
            overnight.run_claim((task, plan, state_root, state, lock), deadline, worker_id, lambda: stop[0])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def scheduler(config, dry_run=False):
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("use the existing authenticated ClusterA login session")
    root = Path(config["cwd"])
    state = resolve_inside(config["state_root"], root)
    state.mkdir(parents=True, exist_ok=True)
    with (state / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / "submission_journal.json"
        journal = _state(journal_path) or {"config_sha256": config["sha256"], "submissions": []}
        if journal["config_sha256"] != config["sha256"] or journal.get("submission_intent"):
            raise ValueError("changed configuration or unresolved submission; audit before resuming")
        while time.time() < config["deadline_utc_epoch"]:
            response = subprocess.run(["qstat", "-f", "-F", "json"], capture_output=True, text=True, timeout=45, check=True)
            payload = json.loads(response.stdout)
            jobs = overnight.owned_jobs(payload, config["user"])
            nodes = overnight.assigned_debug_nodes(payload, config["user"])
            if len(jobs) > 1:
                raise RuntimeError("multiple MLSys allocations; refusing further submissions")
            queues, missing = load_queues(config, "gpu")
            overnight.propagate_blocked(queues)
            selected = overnight.claim(queues, dry_run=True)
            ready = selected is not None
            if selected is not None:
                selected[-1].close()
            snapshot = {"time": time.time(), "configuration_sha256": config["sha256"],
                        "owned_jobs": jobs, "assigned_debug_nodes_all_workflows": nodes,
                        "ready_gpu_work": ready, "unpublished_plans": missing,
                        "queues": overnight.queue_snapshot(queues), "submissions": journal["submissions"]}
            atomic_json(state / "scheduler_status.json", snapshot)
            print(json.dumps({k: v for k, v in snapshot.items() if k != "owned_jobs"}), flush=True)
            if dry_run:
                return
            if (not jobs and nodes == 0 and ready and len(journal["submissions"]) < config["maximum_submissions"]
                    and time.time() + 3600 <= config["deadline_utc_epoch"]):
                for path, expected in config["input_hashes"].items():
                    if file_digest(root / path) != expected:
                        raise ValueError(f"submission source changed: {path}")
                journal["submission_intent"] = {"time": time.time(), "script": config["pbs_script"]}
                atomic_json(journal_path, journal)
                result = subprocess.run(["qsub", config["pbs_script"]], cwd=root, capture_output=True,
                                        text=True, timeout=60, check=True)
                job = result.stdout.strip()
                import re
                if not re.fullmatch(r"\d+\.clustera-pbs-[A-Za-z0-9.-]+", job):
                    raise ValueError("uncertain qsub result; no automatic resubmission")
                journal["submissions"].append({"job": job, "submitted": time.time()})
                journal.pop("submission_intent")
                atomic_json(journal_path, journal)
            if not jobs and all_terminal(queues, missing):
                return
            time.sleep(30)


def hms_seconds(value):
    parts = str(value).split(":")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError("invalid PBS walltime")
    return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])


def clusterb_watch(config):
    """Use only named, already-authorized interactive jobs; never cancel/submit."""
    if not socket.gethostname().startswith("clusterb-login-"):
        raise ValueError("ClusterB watcher must run on its login node")
    state = resolve_inside(config["state_root"], config["cwd"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "clusterb.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        processes, started = {}, set()
        try:
            while time.time() < config["deadline_utc_epoch"]:
                response = subprocess.run(["qstat", "-f", "-F", "json"], capture_output=True, text=True, timeout=45, check=True)
                jobs = json.loads(response.stdout).get("Jobs", {})
                for job, expected_name in config.get("clusterb_jobs", {}).items():
                    if job in started or job not in jobs:
                        continue
                    value = jobs[job]
                    if (value.get("Job_Name") != expected_name
                            or value.get("Job_Owner", "").split("@", 1)[0] != config["user"]
                            or str(value.get("interactive", "")).lower() not in {"true", "1"}):
                        raise ValueError("ClusterB allocation ownership differs")
                    if value.get("job_state") != "R":
                        continue
                    nodes = {part.split("/")[0] for part in value.get("exec_host", "").split("+")}
                    if len(nodes) != 1 or not re.fullmatch(r"clusterb-gpu-\d+", next(iter(nodes))):
                        raise ValueError("unexpected ClusterB node assignment")
                    host = next(iter(nodes))
                    remaining = hms_seconds(value["Resource_List"]["walltime"]) - hms_seconds(value.get("resources_used", {}).get("walltime", "00:00:00")) - 180
                    remaining = min(remaining, int(config["deadline_utc_epoch"] - time.time()))
                    if remaining < 600:
                        started.add(job)
                        continue
                    query = ["ssh", "-o", "BatchMode=yes", host, "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits"]
                    if subprocess.check_output(query, text=True, timeout=30).strip():
                        continue
                    # Explicit names/host patterns above prevent remote shell input.
                    command = ("bash " + os.environ.get("ASYMVLM_ROOT", ".") + "/scripts/mlsys_sprint48_clusterb.sh "
                               + job + " " + str(remaining))
                    log = (state / ("clusterb-" + job.split(".")[0] + ".log")).open("ab")
                    process = subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, command], stdout=log, stderr=subprocess.STDOUT)
                    log.close()
                    processes[job] = process
                    started.add(job)
                    atomic_json(state / ("clusterb-" + job.split(".")[0] + ".json"),
                                {"job": job, "host": host, "started": time.time(), "seconds": remaining,
                                 "ssh_pid": process.pid, "configuration_sha256": config["sha256"]})
                q, missing = load_queues(config, "all")
                if all_terminal(q, missing):
                    return
                time.sleep(30)
        finally:
            # Worker deadlines protect remote processes if this watcher exits.
            for process in processes.values():
                if process.poll() is not None:
                    process.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dispatch", "scheduler", "check", "clusterb-watch"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--resource", choices=("cpu", "gpu"), default="gpu")
        p.add_argument("--seconds", type=int, default=3000)
        p.add_argument("--worker-id", default="sprint48")
        p.add_argument("--dry-run", action="store_true")
    p = sub.add_parser("guard")
    p.add_argument("--mode", choices=("shared", "exclusive"), required=True)
    p.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == "guard":
        return guard(args.mode, args.argv[1:] if args.argv[:1] == ["--"] else args.argv)
    config = read_config(args.config)
    if args.command == "scheduler":
        scheduler(config, args.dry_run)
    elif args.command == "clusterb-watch":
        clusterb_watch(config)
    elif args.command == "dispatch":
        dispatch(config, args.resource, args.seconds, args.worker_id)
    else:
        queues, missing = load_queues(config, "all")
        print(json.dumps({"queues": overnight.queue_snapshot(queues), "unpublished_plans": missing}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
