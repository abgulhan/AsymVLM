"""Run the frozen queues inside one newly authorized ClusterB interactive job.

This module never submits or cancels allocations. Only its own direct dispatch
children receive a termination request; existing dispatch/guard code owns and
reaps each task's children. The usual shared/exclusive node leases remain in
force, so isolated timing blocks cannot overlap quality workers on this node.
"""

import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time

from . import sprint48
from .state import atomic_json, file_digest, resolve_inside


JOB_NAME = "mlsys-cache-v4"
JOB_PATTERN = r"[0-9]+\.clusterb-pbs-[A-Za-z0-9.-]+"
METADATA_GRACE_SECONDS = 300


class AllocationMetadataPending(ValueError):
    """An otherwise authorized running job has not published its start time."""


def start_epoch(value):
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.isdigit()):
        return float(value)
    if not isinstance(value, str):
        raise ValueError("PBS start time is absent")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        pass
    # PBS's human-format timestamp uses the local cluster timezone. datetime's
    # naive timestamp conversion consults that same local timezone, not UTC by
    # assumption. resources_used supplies a second conservative expiry bound.
    try:
        return datetime.strptime(value, "%a %b %d %H:%M:%S %Y").timestamp()
    except ValueError as exc:
        raise ValueError("unrecognized PBS start time") from exc


def authorized_allocation(payload, job_id, hostname, user, config_deadline, now):
    if not re.fullmatch(JOB_PATTERN, job_id or ""):
        raise ValueError("PBS_JOBID is not an explicit ClusterB job identity")
    job = payload.get("Jobs", {}).get(job_id)
    if not isinstance(job, dict):
        raise ValueError("current PBS job is absent from qstat")
    host = hostname.split(".", 1)[0]
    hosts = {part.split("/", 1)[0].split(".", 1)[0]
             for part in job.get("exec_host", "").split("+") if part}
    if (job.get("Job_Name") != JOB_NAME or job.get("job_state") != "R"
            or job.get("Job_Owner", "").split("@", 1)[0] != user
            or str(job.get("interactive", "")).lower() not in {"true", "1"}
            or not re.fullmatch(r"clusterb-gpu-[0-9]+", host) or hosts != {host}
            or job.get("queue", "").split("@", 1)[0] != "by-node"):
        raise ValueError("allocation name/owner/state/interactive/queue/exec_host mismatch")
    resources = job.get("Resource_List", {})
    walltime = sprint48.hms_seconds(resources.get("walltime"))
    if not 0 < walltime <= 24 * 3600:
        raise ValueError("unexpected dedicated ClusterB walltime")
    started = job.get("stime")
    if started is None or (isinstance(started, str) and not started.strip()):
        raise AllocationMetadataPending("PBS start time is not published yet")
    began = start_epoch(started)
    if not 0 < began <= now + 60:
        raise ValueError("PBS start time is invalid or in the future")
    used = sprint48.hms_seconds(job.get("resources_used", {}).get("walltime", "00:00:00"))
    deadline = min(float(config_deadline), began + walltime - 300, now + walltime - used - 300)
    if deadline - now < 600:
        raise ValueError("insufficient conservative allocation lifetime")
    return {"job_id": job_id, "hostname": host, "owner": user,
            "job_name": JOB_NAME, "start_epoch": began, "walltime_seconds": walltime,
            "deadline_epoch": deadline, "seconds": int(deadline - now),
            "safety_margin_seconds": 300}


def verify_idle(inventory, processes):
    if processes.strip():
        raise ValueError("GPU compute processes already exist; no workers were launched")
    rows = []
    for line in inventory.strip().splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or not all(field.isdigit() for field in fields):
            raise ValueError("unrecognized GPU inventory")
        rows.append(tuple(map(int, fields)))
    if (len(rows) != 8 or {row[0] for row in rows} != set(range(8))
            or any(utilization != 0 or memory > 256 for _, memory, utilization in rows)):
        raise ValueError("expected all eight GPUs idle, with at most256MiB baseline allocation each")
    return [{"gpu": gpu, "memory_mib": memory, "utilization_percent": utilization}
            for gpu, memory, utilization in sorted(rows)]


def worker_command(config_path, seconds, job_id, gpu, python=sys.executable):
    if not re.fullmatch(JOB_PATTERN, job_id) or gpu not in range(8) or seconds < 1:
        raise ValueError("invalid worker identity or deadline")
    return [python, "-u", "-m", "learned_score.mlsys_sprint.sprint48", "dispatch",
            "--config", str(config_path), "--resource", "gpu", "--seconds", str(seconds),
            "--worker-id", f"clusterb-dedicated-cache-v4-{job_id}-{gpu}"]


def queued_or_running(payload, job_id, user):
    """Verify identity even while queued; never infer a replacement job."""
    if not re.fullmatch(JOB_PATTERN, job_id or ""):
        raise ValueError("wait-job requires an exact ClusterB job ID")
    job = payload.get("Jobs", {}).get(job_id)
    if (not isinstance(job, dict) or job.get("Job_Name") != JOB_NAME
            or job.get("Job_Owner", "").split("@", 1)[0] != user
            or str(job.get("interactive", "")).lower() not in {"true", "1"}
            or job.get("queue", "").split("@", 1)[0] != "by-node"):
        raise ValueError("requested allocation is absent or changed ownership/name/interactive/queue")
    state = job.get("job_state")
    if state in {"Q", "H", "W", "T"}:
        return None
    if state != "R":
        raise ValueError(f"requested allocation entered non-runnable state {state!r}; no resubmission")
    hosts = {part.split("/", 1)[0].split(".", 1)[0]
             for part in job.get("exec_host", "").split("+") if part}
    if len(hosts) != 1 or not re.fullmatch(r"clusterb-gpu-[0-9]+", next(iter(hosts))):
        raise ValueError("running allocation lacks one validated ClusterB exec_host")
    return next(iter(hosts))


def attachment_command(root, config_path, job_id, host):
    if not re.fullmatch(JOB_PATTERN, job_id) or not re.fullmatch(r"clusterb-gpu-[0-9]+", host):
        raise ValueError("invalid explicit SSH allocation target")
    remote = shlex.join(["env", f"PBS_JOBID={job_id}", "bash",
                         str(Path(root) / "scripts/mlsys_cache_methods_v4_interactive.sh"),
                         "--run", str(config_path)])
    return ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, remote]


def wait_for_job(config_path, job_id, allocation_pane=None):
    """Login-node watcher for one already-submitted interactive allocation."""
    if not re.fullmatch(r"clusterb-login-[0-9]+", socket.gethostname().split(".", 1)[0]):
        raise ValueError("--wait-job must run on a ClusterB login node")
    if not re.fullmatch(JOB_PATTERN, job_id or ""):
        raise ValueError("--wait-job requires an exact ClusterB job identity")
    if allocation_pane not in (None, "mlsys-cache-v4-allocation"):
        raise ValueError("only the dedicated v4 allocation pane is authorized")
    config_path = Path(config_path).resolve(strict=True)
    config = sprint48.read_config(config_path)
    if config.get("coordinator_role") != "clusterb":
        raise ValueError("watcher requires the existing ClusterB configuration")
    root = Path(config["cwd"]).resolve(strict=True)
    resolve_inside(config_path, root)
    state = resolve_inside(config["state_root"], root)
    directory = state / ("dedicated-clusterb-" + job_id.split(".", 1)[0])
    directory.mkdir(parents=True, exist_ok=True)
    status = {"job_id": job_id, "configuration_sha256": config["sha256"],
              "watcher_pid": os.getpid(), "watcher_sha256": file_digest(__file__),
              "config": str(config_path), "never_submits_or_cancels": True}
    with (directory / "waiter.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if allocation_pane is not None and (directory / "waiter.json").exists():
            previous = json.loads((directory / "waiter.json").read_text())
            if previous.get("status") == "launched_in_owned_interactive_pane":
                if (previous.get("job_id") != job_id or previous.get("configuration_sha256") != config["sha256"]):
                    raise ValueError("existing pane-launch receipt belongs to another allocation")
                return 0  # Never inject a second command after an acknowledged launch.
        metadata_started = None
        try:
            while time.time() < config["deadline_utc_epoch"]:
                response = subprocess.run(["qstat", "-f", "-F", "json", job_id], check=True,
                                          text=True, capture_output=True, timeout=30)
                payload = json.loads(response.stdout)
                host = queued_or_running(payload, job_id, config["user"])
                if host is None:
                    atomic_json(directory / "waiter.json", {**status, "status": "waiting_for_exact_job",
                                                            "checked_epoch": time.time()})
                    time.sleep(min(30., max(0., config["deadline_utc_epoch"] - time.time())))
                    continue
                try:
                    allocation = authorized_allocation(payload, job_id, host, config["user"],
                                                       config["deadline_utc_epoch"], time.time())
                except AllocationMetadataPending:
                    # PBS can publish R/exec_host before stime. Never launch or
                    # infer a fresh 24-hour lifetime from that partial record.
                    if metadata_started is None:
                        metadata_started = time.monotonic()
                    remaining = METADATA_GRACE_SECONDS - (time.monotonic() - metadata_started)
                    if remaining <= 0:
                        raise TimeoutError("PBS start time remained absent beyond metadata grace period")
                    atomic_json(directory / "waiter.json", {
                        **status, "status": "waiting_for_allocation_metadata",
                        "checked_epoch": time.time(), "missing_fields": ["stime"],
                        "metadata_grace_remaining_seconds": remaining})
                    time.sleep(min(30., remaining, max(0., config["deadline_utc_epoch"] - time.time())))
                    continue
                if allocation_pane is not None:
                    target = f"={allocation_pane}:0.0"
                    pane = subprocess.check_output(["tmux", "capture-pane", "-p", "-t", target,
                                                    "-S", "-500"], text=True, timeout=10)
                    # qstat may report R before the interactive shell appears.
                    # Require that this pane contains the exact accepted job
                    # and its assigned host before sending anything to it.
                    if job_id not in pane or not re.search(re.escape(host) + r"[: >]", pane):
                        atomic_json(directory / "waiter.json", {
                            **status, "status": "waiting_for_interactive_shell", "checked_epoch": time.time()})
                        time.sleep(10)
                        continue
                    command = shlex.join(["exec", "bash", str(root / "scripts/mlsys_cache_methods_v4_interactive.sh"),
                                          "--run", str(config_path)])
                    # exec replaces only this owned interactive shell. Once
                    # our workers finish, PBS releases this allocation normally.
                    subprocess.run(["tmux", "send-keys", "-t", target, "-l", command], check=True, timeout=10)
                    subprocess.run(["tmux", "send-keys", "-t", target, "Enter"], check=True, timeout=10)
                    receipt = {**status, **allocation, "status": "launched_in_owned_interactive_pane",
                               "tmux_session": allocation_pane, "command": command, "attached_epoch": time.time()}
                    atomic_json(directory / "waiter.json", receipt)
                    print(json.dumps(receipt), flush=True)
                    return 0
                command = attachment_command(root, config_path, job_id, host)
                with (directory / "attachment.log").open("ab") as log:
                    child = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                receipt = {**status, **allocation, "status": "attached", "ssh_pid": child.pid,
                           "argv": command, "attached_epoch": time.time()}
                atomic_json(directory / "waiter.json", receipt)
                print(json.dumps({"event": "dedicated_clusterb_attached", **receipt}), flush=True)
                code = child.wait()
                receipt.update(status="workers_exited" if code == 0 else "failed",
                               returncode=code, finished_epoch=time.time())
                atomic_json(directory / "waiter.json", receipt)
                atomic_json(directory / "waiter-exit.json", receipt)
                return code
            receipt = {**status, "status": "configuration_deadline",
                       "finished_epoch": time.time(), "returncode": 1}
            atomic_json(directory / "waiter.json", receipt)
            atomic_json(directory / "waiter-exit.json", receipt)
            return 1
        except Exception as exc:
            receipt = {**status, "status": "failed", "returncode": 1,
                       "error_type": type(exc).__name__, "error": str(exc),
                       "finished_epoch": time.time()}
            atomic_json(directory / "waiter.json", receipt)
            atomic_json(directory / "waiter-exit.json", receipt)
            raise


def run(config_path):
    config_path = Path(config_path).resolve(strict=True)
    config = sprint48.read_config(config_path)
    if config.get("coordinator_role") != "clusterb":
        raise ValueError("dedicated node must use the existing ClusterB queue configuration")
    root = Path(config["cwd"]).resolve(strict=True)
    resolve_inside(config_path, root)
    state = resolve_inside(config["state_root"], root)
    job_id = os.environ.get("PBS_JOBID", "")
    response = subprocess.run(["qstat", "-f", "-F", "json", job_id], check=True,
                              text=True, capture_output=True, timeout=30)
    allocation = authorized_allocation(json.loads(response.stdout), job_id, socket.gethostname(),
                                       config["user"], config["deadline_utc_epoch"], time.time())
    inventory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                         "--format=csv,noheader,nounits"], text=True, timeout=20)
    processes_text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                              "--format=csv,noheader,nounits"], text=True, timeout=20)
    devices = verify_idle(inventory, processes_text)
    state.mkdir(parents=True, exist_ok=True)
    directory = state / ("dedicated-clusterb-" + job_id.split(".", 1)[0])
    directory.mkdir(exist_ok=True)
    children, stopped = [], [False]
    handlers = {}
    with (directory / "runner.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for sig in (signal.SIGTERM, signal.SIGINT):
            handlers[sig] = signal.signal(sig, lambda *_: stopped.__setitem__(0, True))
        receipt = {**allocation, "configuration_sha256": config["sha256"], "config": str(config_path),
                   "runner_sha256": file_digest(__file__), "gpu_inventory": devices,
                   "started_epoch": time.time(), "runner_pid": os.getpid(), "worker_pids": []}
        try:
            seconds = int(allocation["deadline_epoch"] - time.time())
            if seconds < 600:
                raise ValueError("allocation expired during setup")
            for gpu in range(8):
                if stopped[0]:
                    break
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS="4",
                           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
                # Only the per-task guard can assert actual exclusive isolation.
                env.pop("ASYMVLM_ISOLATED_NODE", None)
                env.pop("ASYMVLM_ISOLATION_LEASE", None)
                with (directory / f"gpu{gpu}.log").open("ab") as log:
                    child = subprocess.Popen(worker_command(config_path, seconds, job_id, gpu),
                                             cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                             pass_fds=(lock.fileno(),))
                children.append(child)
            receipt["worker_pids"] = [child.pid for child in children]
            atomic_json(directory / "allocation.json", receipt)
            print(json.dumps({"event": "dedicated_clusterb_workers_started", **receipt}), flush=True)
            while any(child.poll() is None for child in children):
                workflow_path = Path(config["v4_protocol"]).parent / "coordinator/workflow-status.json"
                if workflow_path.exists():
                    workflow = json.loads(workflow_path.read_text())
                    if workflow.get("complete") or workflow.get("failed_tasks"):
                        stopped[0] = True
                if stopped[0] or time.time() >= allocation["deadline_epoch"]:
                    for child in children:
                        if child.poll() is None:
                            child.terminate()
                    break
                time.sleep(2)
        finally:
            # No pkill, qdel, process-name search, process-group broadcast, or
            # foreign PID signalling. Workers gracefully reap their own tasks.
            for child in children:
                if child.poll() is None:
                    child.terminate()
            codes = [child.wait() for child in children]
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            atomic_json(directory / "exit.json", {"finished_epoch": time.time(), "job_id": job_id,
                        "returncodes": codes, "stop_requested": stopped[0],
                        "configuration_sha256": config["sha256"],
                        "experiment_completion_requires_task_markers": True})
        return int(len(children) != 8 or any(codes))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wait-job", help="attach automatically to this exact already-submitted job")
    parser.add_argument("--allocation-pane", choices=("mlsys-cache-v4-allocation",),
                        help="launch inside the owned interactive shell so PBS releases it when work ends")
    args = parser.parse_args(argv)
    if args.allocation_pane and not args.wait_job:
        parser.error("--allocation-pane requires --wait-job")
    if args.wait_job:
        return (wait_for_job(args.config, args.wait_job, args.allocation_pane)
                if args.allocation_pane else wait_for_job(args.config, args.wait_job))
    return run(args.config)


if __name__ == "__main__":
    raise SystemExit(main())

