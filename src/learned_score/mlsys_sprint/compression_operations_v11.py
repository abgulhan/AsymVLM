"""Independent, bounded v11 dispatch and owned-idle cluster admission.

Configuration construction is pure. Only explicit watch/clustera/dispatch CLI
modes start work. No historical coordinator is monkey-patched, no allocation
number is hardcoded, and no jobs or unrelated processes are cancelled.
"""

import argparse
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import stat
import subprocess
import time

from . import adaptive_operations_v4 as clusterb
from . import overnight, clustera_queue_policy_v5 as queue_policy, sprint48
from . import sprint48_scheduler_v3 as submissions
from .state import atomic_json, digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources, _state


STUDY_ROOT = Path("reproduction_results/mlsys_compression_20261020_v11")
WORKER = Path("scripts/mlsys_compression_v11_idle_worker.sh")
CLUSTERA_WORKER = Path("scripts/mlsys_compression_v11_clustera_worker.sh")
PBS_SCRIPT = Path("scripts/mlsys_compression_v11_clustera.pbs")
DEFAULT_CONFIG = STUDY_ROOT / "coordinator/clusterb-config.json"
CLUSTERA_CONFIG = STUDY_ROOT / "coordinator/clustera-config.json"
SESSION_PREFIX = "mlsys-compression-v11"
SOURCE_FILES = (
    "learned_score/mlsys_sprint/compression_operations_v11.py",
    "learned_score/mlsys_sprint/adaptive_operations_v4.py",
    "learned_score/mlsys_sprint/cache_dedicated_clusterb_v4.py",
    "learned_score/mlsys_sprint/clustera_queue_policy_v5.py",
    "learned_score/mlsys_sprint/sprint48_scheduler_v3.py",
    "learned_score/mlsys_sprint/sprint48.py",
    "learned_score/mlsys_sprint/overnight.py",
    "learned_score/mlsys_sprint/worker.py",
    "learned_score/mlsys_sprint/state.py",
    "activate_env.sh", str(WORKER), str(CLUSTERA_WORKER), str(PBS_SCRIPT),
    "scripts/mlsys_compression_v11_clusterb_watch.py",
    "scripts/mlsys_compression_v11_clustera_watch.py",
)


def seal(value):
    return {**value, "sha256": digest(value)}


def build_config(cwd, plan_paths, *, role, deadline_utc_epoch,
                 created_utc_epoch=None, user=None, extra_sources=(), maximum_submissions=24,
                 study_root=STUDY_ROOT):
    """Pin explicit future plan paths; publishing them later remains supported."""
    root = Path(cwd).resolve()
    study_root = resolve_inside(study_root, root)
    plans_root = study_root / "plans"
    paths = []
    for value in plan_paths:
        value = Path(value)
        if not value.is_absolute():
            value = root / value if value.parts[:1] == ("reproduction_results",) else plans_root / value
        paths.append(str(resolve_inside(value, plans_root)))
    sources = {str(resolve_inside(p, root)) for p in (*SOURCE_FILES, *extra_sources)}
    config = seal({
        "format": sprint48.FORMAT, "study_operations": "compression_v11",
        "study_root": str(study_root),
        "cwd": str(root), "user": user or getpass.getuser(), "coordinator_role": role,
        "created_utc_epoch": time.time() if created_utc_epoch is None else created_utc_epoch,
        "deadline_utc_epoch": deadline_utc_epoch, "maximum_submissions": maximum_submissions,
        "maximum_debug_nodes": 2, "maximum_nonterminal_debug_nodes": 2,
        "clustera_submission_policy": "disabled_without_joint_admission_coordination",
        "state_root": str(study_root / "coordinator" / role),
        "pbs_script": str(PBS_SCRIPT), "plan_paths": paths,
        "clustera_models": ["llava_next"], "qwen_site": "clusterb_owned_idle",
        "input_hashes": {p: file_digest(p) for p in sorted(sources)},
    })
    return validate_config(config)


def validate_config(config):
    root = Path(config["cwd"]).resolve()
    study_root = resolve_inside(config.get("study_root", root / STUDY_ROOT), root / "reproduction_results")
    if study_root.parent != root / "reproduction_results" or not re.fullmatch(r"mlsys_compression_[0-9]{8}_v11", study_root.name):
        raise ValueError("operations study root must be a separate v11 output directory")
    role = config.get("coordinator_role")
    for name in ("deadline_utc_epoch", "created_utc_epoch"):
        if type(config.get(name)) not in (int, float) or not math.isfinite(config[name]):
            raise ValueError("invalid v11 timestamp")
    window = config["deadline_utc_epoch"] - config["created_utc_epoch"]
    if (config.get("format") != sprint48.FORMAT or config.get("study_operations") != "compression_v11"
            or config.get("sha256") != digest({k: v for k, v in config.items() if k != "sha256"})
            or role not in {"clusterb", "clustera"} or not 0 < window <= 24 * 3600
            or type(config.get("maximum_submissions")) is not int
            or not 1 <= config["maximum_submissions"] <= 48
            or config.get("maximum_debug_nodes") != 2
            or config.get("maximum_nonterminal_debug_nodes") != 2
            or config.get("clustera_submission_policy") != "disabled_without_joint_admission_coordination"
            or config.get("clustera_models") != ["llava_next"]
            or config.get("qwen_site") != "clusterb_owned_idle"
            or not re.fullmatch(r"[a-z_][a-z0-9_-]*", config.get("user", ""))):
        raise ValueError("invalid bounded v11 configuration")
    if (resolve_inside(config["state_root"], root) != study_root / "coordinator" / role
            or resolve_inside(config["pbs_script"], root) != root / PBS_SCRIPT):
        raise ValueError("operations state or PBS script escaped v11")
    paths = config.get("plan_paths", [])
    if not paths or len(paths) != len(set(paths)):
        raise ValueError("declare unique v11 plan paths")
    for value in paths:
        path = resolve_inside(value, study_root / "plans")
        if (path.name != "plan.json" or path.parent.parent != study_root / "plans"
                or not re.fullmatch(r"[A-Za-z0-9_-]+", path.parent.name)):
            raise ValueError("invalid v11 plan path")
    pins = config.get("input_hashes", {})
    if not {str(root / name) for name in SOURCE_FILES} <= set(pins):
        raise ValueError("v11 safety sources are not all pinned")
    for name in pins:
        resolve_inside(name, root)
    script = (root / PBS_SCRIPT).read_text()
    for pattern, expected in (
            (r"^#PBS\s+-N\s+(\S+)\s*$", "mlsys-compression-v11"),
            (r"^#PBS\s+-q\s+(\S+)\s*$", "debug"),
            (r"^#PBS\s+-l\s+select=(\S+)\s*$", "2"),
            (r"^#PBS\s+-l\s+walltime=(\S+)\s*$", "01:00:00")):
        if re.findall(pattern, script, re.MULTILINE) != [expected]:
            raise ValueError("v11 PBS request is not two debug nodes for one hour")
    return config


def read_config(path):
    return validate_config(sprint48.read_config(path))


def validate_published_plans(config):
    validate_config(config)
    for path, expected in config["input_hashes"].items():
        if file_digest(path) != expected:
            raise ValueError("frozen v11 operation input changed: " + path)
    for name in config["plan_paths"]:
        if not Path(name).exists():
            continue
        plan = read_plan(name)
        verify_sources(plan)
        if (Path(plan["cwd"]).resolve() != Path(config["cwd"])
                or plan.get("unexpected_retries", 1) != 1
                or plan.get("deadline_utc_epoch", config["deadline_utc_epoch"]) > config["deadline_utc_epoch"]):
            raise ValueError("plan workspace, retry budget or deadline mismatch")
        for task in plan["tasks"]:
            model = task.get("model_tag")
            if (task.get("resource") != "gpu" or not task.get("tag")
                    or model not in {"qwen25", "llava_next"}
                    or config["coordinator_role"] == "clustera" and model != "llava_next"):
                raise ValueError("unsupported task or Qwen assigned to ClusterA without parity")
            argv = task["argv"]
            if (argv.count("--adapter") != 1 or argv.index("--adapter") + 1 >= len(argv)
                    or argv[argv.index("--adapter") + 1] != {
                    "qwen25": "qwen2_5_vl", "llava_next": "llava_next"}[model]):
                raise ValueError("task model tag disagrees with physical adapter")
    return config


def ready_work(config, maximum=16, seconds=3300):
    validate_published_plans(config)
    queues, missing = sprint48.load_queues(config, "gpu")
    locks = []
    try:
        for _ in range(maximum):
            selected = overnight.claim(queues, dry_run=True, remaining_seconds=seconds)
            if selected is None:
                break
            locks.append(selected[-1])
        count = len(locks)
    finally:
        for lock in locks:
            lock.close()
    return count, sprint48.all_terminal(queues, missing), missing


def dispatch(config, seconds, worker_id):
    """Claim shared standard-worker tasks; release the GPU as soon as idle."""
    validate_published_plans(config)
    if config["user"] != getpass.getuser() or not 0 < seconds <= 10800:
        raise ValueError("worker owner or lifetime mismatch")
    role = config["coordinator_role"]
    host = socket.gethostname().split(".", 1)[0]
    if role == "clustera":
        if (not re.fullmatch(r"x[0-9]+[a-z0-9]+", host)
                or not submissions.JOB_ID.fullmatch(os.environ.get("PBS_JOBID", ""))
                or os.environ.get("PBS_JOBNAME") != "mlsys-compression-v11"
                or os.environ.get("CUDA_VISIBLE_DEVICES") not in {"0", "1", "2", "3"}):
            raise ValueError("ClusterA dispatch requires the v11 compute allocation")
    elif (not clusterb.HOST_PATTERN.fullmatch(host)
          or not clusterb.JOB_PATTERN.fullmatch(os.environ.get("PBS_JOBID", ""))
          or os.environ.get("ASYMVLM_V11_IDLE_LEASE") != config["sha256"]):
        raise ValueError("ClusterB dispatch requires a verified owned-idle GPU lease")
    end = time.monotonic() + min(seconds, max(0, config["deadline_utc_epoch"] - time.time()))
    stopping = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopping[0] and end - time.monotonic() >= 10:
            validate_published_plans(config)
            queues, _ = sprint48.load_queues(config, "gpu")
            selected = overnight.claim(queues, remaining_seconds=end - time.monotonic())
            if selected is None:
                print(json.dumps({"event": "idle_gpu_released", "worker": worker_id}), flush=True)
                return 0
            task, plan, state_root, state, lock = selected
            guarded = {**task, "argv": [str(Path(config["cwd"]) / ".conda/asymvlm/bin/python"),
                "-u", "-m", "learned_score.mlsys_sprint.sprint48", "guard", "--mode", "shared", "--", *task["argv"]]}
            overnight.run_claim((guarded, plan, state_root, state, lock), end, worker_id, lambda: stopping[0])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def admission(payload, queue_text, user):
    decision = queue_policy.admission(payload, queue_text, user)
    nodes = sum(row["nodes"] for key in ("running_jobs", "queued_jobs", "other_jobs") for row in decision[key])
    return {**decision, "eligible": nodes == 0, "reserved_debug_nodes_all_workflows": nodes,
            "maximum_nonterminal_debug_nodes": 2, "request_nodes": 2,
            "reason": "eligible" if nodes == 0 else "waiting_for_all_owned_debug_nodes"}


def observe(config):
    validate_config(config)
    jobs = submissions.execute(["qstat", "-x", "-f", "-F", "json"], timeout=45)
    queue = submissions.execute(["qstat", "-Qf", "debug"], timeout=30)
    if jobs["returncode"] != 0 or queue["returncode"] != 0:
        raise ValueError("cannot verify ClusterA jobs and queue policy")
    payload = json.loads(jobs["stdout"])
    return payload, admission(payload, queue["stdout"], config["user"])


def clustera_coordinated(config):
    """No other active submitter has adopted a joint admission lock in v11.

    Inventory rechecks cannot close races with that submitter. This version
    therefore supports inspection but deliberately never authorizes qsub.
    Enabling submissions needs a separately reviewed coordination integration,
    not an environment variable or resealed configuration toggle.
    """
    validate_config(config)
    return False


def scheduler(config, *, dry_run=False, once=False, clock=time.time, sleep=time.sleep):
    validate_published_plans(config)
    if (config["coordinator_role"] != "clustera" or config["user"] != getpass.getuser()
            or not socket.gethostname().startswith("clustera-login-")):
        raise ValueError("use this owner's authenticated ClusterA login session")
    if dry_run:
        return {**observe(config)[1], "submission_enabled": clustera_coordinated(config),
                "submission_blocker": "joint_admission_coordination_unavailable"}
    if not clustera_coordinated(config):
        return "blocked_joint_admission_coordination_unavailable"
    root = Path(config["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "submission_journal.json"
        journal = _state(path) or {"config_sha256": config["sha256"], "submissions": []}
        if journal.get("config_sha256") != config["sha256"]:
            raise ValueError("submission journal belongs to another config")
        while clock() < config["deadline_utc_epoch"]:
            validate_published_plans(config)
            try:
                payload, decision = observe(config)
            except (ValueError, KeyError, TypeError, OSError) as exc:
                atomic_json(root / "status.json", {"status": "waiting_for_verified_inventory", "error": str(exc)})
                if once:
                    return "inventory_failed"
                sleep(30)
                continue
            if journal.get("submission_intent"):
                matches = submissions.nonce_matches(payload, config, journal["submission_intent"])
                if not matches:
                    atomic_json(root / "status.json", {"status": "uncertain_submission_requires_audit"})
                    return "uncertain_submission_requires_audit"
                submissions.append_accepted(journal, matches[0], reconciled=True)
                atomic_json(path, journal)
            if any(row["job"] not in payload["Jobs"] for row in journal["submissions"]):
                atomic_json(root / "status.json", {"status": "accepted_job_missing_requires_audit"})
                return "accepted_job_missing_requires_audit"
            ready, terminal, missing = ready_work(config)
            allowed = (ready > 0 and decision["eligible"]
                       and len(journal["submissions"]) < config["maximum_submissions"]
                       and journal.get("total_rejections_v2", 0) < submissions.MAX_REJECTIONS
                       and clock() >= journal.get("retry_after_epoch", 0)
                       and clock() + 3600 < config["deadline_utc_epoch"])
            snapshot = {"status": "monitoring", "time": clock(), "admission": decision,
                        "ready_tasks": ready, "all_terminal": terminal, "unpublished_plans": missing,
                        "submissions": journal["submissions"]}
            if allowed:
                # Recheck ownership and every other workflow immediately before qsub.
                validate_published_plans(config)
                _, fresh = observe(config)
                if fresh["eligible"] and ready_work(config, maximum=1)[0]:
                    outcome = submissions.submit(config, journal, path)
                    snapshot["submission_outcome"] = outcome
                    if outcome == "uncertain":
                        atomic_json(root / "status.json", snapshot)
                        return "uncertain_submission_requires_audit"
            atomic_json(root / "status.json", snapshot)
            print(json.dumps(snapshot), flush=True)
            if terminal or once:
                return "complete" if terminal else "once"
            sleep(min(30, max(0, config["deadline_utc_epoch"] - clock())))
    return "deadline"


def session_name(host, gpu):
    if not clusterb.HOST_PATTERN.fullmatch(host) or type(gpu) is not int or gpu not in range(8):
        raise ValueError("invalid ClusterB host/GPU")
    return f"{SESSION_PREFIX}-{host}-gpu{gpu}"


def launch_worker(allocation, gpu, config, config_path, *, run=clusterb.command, now=None):
    validate_published_plans(config)
    now = time.time() if now is None else now
    name = session_name(allocation["host"], gpu)
    if not clusterb.JOB_PATTERN.fullmatch(allocation["job_id"]):
        raise ValueError("invalid ClusterB allocation identity")
    seconds = min(10800, int(allocation["expires"] - now - 600), int(config["deadline_utc_epoch"] - now))
    if seconds < 600:
        raise ValueError("insufficient safe allocation lifetime")
    config_path = resolve_inside(config_path, config["cwd"])
    if read_config(config_path) != config:
        raise ValueError("worker configuration differs from watcher")
    logs = Path(config["study_root"]) / "coordinator/logs"
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}-{int(now)}.log"
    remote = "bash " + shlex.join([str(Path(config["cwd"]) / WORKER), allocation["job_id"], str(gpu), str(seconds), str(config_path)])
    argv = ["ssh", *clusterb.SSH_OPTIONS]
    if not socket.gethostname().startswith("clusterb-login-"):
        argv += ["-J", "clusterb-login-01"]
    argv += [allocation["host"], remote]
    run(["tmux", "new-session", "-d", "-s", name,
         "exec " + shlex.join(argv) + " >> " + shlex.quote(str(log)) + " 2>&1"])
    return {"session": name, "job_id": allocation["job_id"], "host": allocation["host"],
            "gpu": gpu, "seconds": seconds, "started": now, "log": str(log)}


def watch(config_path, *, once=False, dry_run=False, run=clusterb.command, clock=time.time, sleep=time.sleep):
    config = read_config(config_path)
    validate_published_plans(config)
    if config["coordinator_role"] != "clusterb" or config["user"] != getpass.getuser():
        raise ValueError("owned ClusterB configuration required")
    if dry_run:
        allocations = clusterb.owned_allocations(clusterb.query_allocations(config["user"], run),
            config["user"], clock(), config["deadline_utc_epoch"])
        return {"allocations": [{**a, "idle_gpus": clusterb.idle_gpus(a["host"], run)} for a in allocations], "dry_run": True}
    root = Path(config["state_root"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root / "launches.json"
        journal = _state(path) or {"configuration_sha256": config["sha256"], "launches": []}
        if journal.get("configuration_sha256") != config["sha256"]:
            raise ValueError("watcher journal differs from current config")
        while clock() < config["deadline_utc_epoch"]:
            config = read_config(config_path)
            ready, terminal, missing = ready_work(config, seconds=10800)
            snapshot = {"time": clock(), "ready_tasks": ready, "all_terminal": terminal,
                        "unpublished_plans": missing, "launched": [], "errors": []}
            if ready:
                try:
                    allocations = clusterb.owned_allocations(clusterb.query_allocations(config["user"], run),
                        config["user"], clock(), config["deadline_utc_epoch"])
                    sessions = clusterb.tmux_sessions(run)
                    for allocation in allocations:
                        if not ready:
                            break
                        idle = clusterb.idle_gpus(allocation["host"], run)
                        for gpu in idle:
                            name = session_name(allocation["host"], gpu)
                            recent = max((r["started"] for r in journal["launches"] if r["session"] == name), default=0)
                            if not ready or name in sessions or clock() - recent < 60:
                                continue
                            receipt = launch_worker(allocation, gpu, config, config_path, run=run, now=clock())
                            journal["launches"].append(receipt)
                            atomic_json(path, journal)
                            snapshot["launched"].append(receipt)
                            sessions.add(name)
                            ready -= 1
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    snapshot["errors"].append(clusterb.error_detail(exc))
            atomic_json(root / "status.json", snapshot)
            print(json.dumps(snapshot), flush=True)
            if once or terminal:
                return snapshot
            sleep(min(30, max(0, config["deadline_utc_epoch"] - clock())))
    return {"status": "deadline"}


def owned_idle_worker(job_id, gpu, seconds, config_path):
    """Compute-node revalidation and cooperative cross-project GPU lease."""
    config = read_config(config_path)
    validate_published_plans(config)
    host, now = socket.gethostname().split(".", 1)[0], time.time()
    if (config["coordinator_role"] != "clusterb" or config["user"] != getpass.getuser()
            or not clusterb.JOB_PATTERN.fullmatch(job_id) or type(gpu) is not int or gpu not in range(8)
            or not 0 < seconds <= 10800):
        raise ValueError("invalid owned-idle worker request")
    # Normalize stime on this cluster's host timezone, as the watcher does.
    from .cache_dedicated_clusterb_v4 import start_epoch
    payload = json.loads(clusterb.command(["qstat", "-f", "-F", "json", job_id]))
    if job_id in payload.get("Jobs", {}):
        payload["Jobs"][job_id]["stime"] = start_epoch(payload["Jobs"][job_id].get("stime"))
    allocations = clusterb.owned_allocations(payload, config["user"], now, config["deadline_utc_epoch"])
    matches = [a for a in allocations if a["job_id"] == job_id and a["host"] == host]
    if len(matches) != 1:
        raise ValueError("worker host is not in the owned running interactive allocation")
    seconds = min(seconds, matches[0]["seconds"])
    lock_path = f"/tmp/kv_day3_{host}_gpu{gpu}.lock"
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid():
            raise ValueError("GPU lease is not a regular owned file")
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_output = clusterb.command(["nvidia-smi", "--query-gpu=index,uuid,memory.used,utilization.gpu", "--format=csv,noheader,nounits"])
        apps = clusterb.command(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"])
        if gpu not in clusterb.parse_idle_gpus(gpu_output, apps):
            raise ValueError("GPU is occupied; no worker launched")
        os.set_inheritable(fd, True)
        os.environ.update(PBS_JOBID=job_id, CUDA_VISIBLE_DEVICES=str(gpu), ASYMVLM_V11_IDLE_LEASE=config["sha256"])
        return dispatch(config, seconds, f"clusterb-v11-{job_id}-{host}-{gpu}")
    finally:
        os.close(fd)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--root", type=Path, default=STUDY_ROOT,
                        help="Separate v11 output root; used to locate role configuration")
    modes = parser.add_mutually_exclusive_group(required=True)
    for name in ("watch", "once", "clustera", "dispatch", "owned-idle-worker"):
        modes.add_argument("--" + name, action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--worker-id")
    parser.add_argument("--job-id")
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args(argv)
    path = args.config or args.root / "coordinator" / (
        "clustera-config.json" if args.clustera or args.dispatch else "clusterb-config.json")
    if args.owned_idle_worker:
        if args.dry_run or not args.job_id or args.gpu is None:
            parser.error("owned-idle-worker requires job-id and gpu; no dry-run")
        return owned_idle_worker(args.job_id, args.gpu, args.seconds, path)
    if args.dispatch:
        if args.dry_run or not args.worker_id:
            parser.error("dispatch requires worker-id; no dry-run")
        return dispatch(read_config(path), args.seconds, args.worker_id)
    if args.clustera:
        print(json.dumps(scheduler(read_config(path), dry_run=args.dry_run)), flush=True)
    else:
        print(json.dumps(watch(path, once=args.once, dry_run=args.dry_run)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
