"""Owned-idle ClusterB attachments for the independently frozen adaptive v4 queues.

No allocation submissions, cancellations, or signals to existing sessions. The
new idle worker releases its cross-project GPU lease as soon as no runnable task
is available. Existing workers are left untouched. Run this on the local host
with an authenticated SSH path through clusterb-login-01.
"""

import argparse
import fcntl
import getpass
import json
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from . import overnight, sprint48
from .state import atomic_json, digest, file_digest, resolve_inside


DEFAULT_CONFIG = Path("reproduction_results/mlsys_adaptive_20261016_v4/coordinator/clusterb-config.json")
WORKER = Path("scripts/mlsys_adaptive_v4_idle_worker.sh")
SESSION_PREFIX = "mlsys-adaptive-v4-clusterb"
OPERATIONS_VERSION = "pbs_json_v2"
JOB_PATTERN = re.compile(r"[1-9][0-9]*\.clusterb-pbs-[A-Za-z0-9.-]+")
HOST_PATTERN = re.compile(r"clusterb-gpu-[0-9]+")
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "-o", "ServerAliveInterval=10",
               "-o", "ServerAliveCountMax=2"]


def command(argv, *, timeout=30):
    return subprocess.run(argv, check=True, capture_output=True, text=True, timeout=timeout).stdout


def operation_paths(config):
    coordinator = resolve_inside(config["state_root"], config["cwd"]).parent
    return coordinator / "operations-v2" / config["sha256"][:12], coordinator / "logs"


def operation_identity(config):
    root = Path(config["cwd"]).resolve()
    worker = resolve_inside(WORKER, root)
    return {"format": "mlsys_adaptive_operations_v4", "configuration_sha256": config["sha256"],
            "operations_version": OPERATIONS_VERSION,
            "owner": config["user"], "cwd": str(root), "worker": str(worker),
            "source_sha256": {str(Path(__file__).resolve()): file_digest(__file__),
                              str(worker): file_digest(worker)}}


def verify_identity(config, *, create=False):
    """Pin this separate operations layer without changing any frozen GPU plan."""
    root, _ = operation_paths(config)
    expected = operation_identity(config)
    path = root / "identity.json"
    if not path.exists():
        if not create:
            raise ValueError("operations source identity has not been published")
        root.mkdir(parents=True, exist_ok=True)
        # The watcher already holds its singleton lock before publication.
        atomic_json(path, {**expected, "sha256": digest(expected)})
    recorded = json.loads(path.read_text())
    if recorded != {**expected, "sha256": digest(expected)}:
        raise ValueError("operations sources or frozen configuration changed")
    return recorded


def owned_allocations(payload, user, now, deadline):
    """Only owned, running interactive by-node jobs with known expiry qualify."""
    from .cache_dedicated_clusterb_v4 import start_epoch
    jobs = payload.get("Jobs", {})
    if not isinstance(jobs, dict):
        raise ValueError("invalid ClusterB qstat response")
    result, owners = [], {}
    for job_id, job in jobs.items():
        if (job.get("Job_Owner", "").split("@", 1)[0] != user or job.get("job_state") != "R"
                or str(job.get("interactive", "")).lower() not in {"true", "1"}
                or job.get("queue", "").split("@", 1)[0] != "by-node"):
            continue
        if not JOB_PATTERN.fullmatch(job_id):
            raise ValueError("owned allocation has an unexpected job identity")
        hosts = {item.split("/", 1)[0].split(".", 1)[0] for item in job.get("exec_host", "").split("+") if item}
        if not hosts or any(not HOST_PATTERN.fullmatch(host) for host in hosts):
            raise ValueError("owned allocation has an unexpected execution host")
        wall = sprint48.hms_seconds(job.get("Resource_List", {}).get("walltime"))
        used = sprint48.hms_seconds(job.get("resources_used", {}).get("walltime"))
        start = start_epoch(job.get("stime"))
        if not 0 < wall <= 86400 or not 0 <= used <= wall or not 0 < start <= now + 60:
            raise ValueError("owned allocation has an inconsistent lifetime")
        seconds = min(10800, int(start + wall - now - 600), int(wall - used - 600), int(deadline - now))
        if seconds < 600:
            continue
        for host in sorted(hosts):
            if host in owners and owners[host] != job_id:
                raise ValueError("execution host appears in two owned allocations")
            owners[host] = job_id
            result.append({"job_id": job_id, "host": host, "seconds": seconds,
                           "expires": min(start + wall, now + wall - used)})
    return result


def parse_idle_gpus(gpu_output, app_output):
    """Require exactly zero memory/use and no compute process on that GPU."""
    active = set()
    for line in app_output.splitlines():
        if not line.strip():
            continue
        fields = [value.strip() for value in line.split(",")]
        if len(fields) != 2 or not fields[1].isdigit() or int(fields[1]) < 1:
            raise ValueError("unrecognized compute-process listing")
        active.add(fields[0])
    seen, idle = set(), []
    for line in gpu_output.splitlines():
        fields = [value.strip() for value in line.split(",")]
        if (len(fields) != 4 or not all(fields[i].isdigit() for i in (0, 2, 3))
                or not re.fullmatch(r"GPU-[A-Za-z0-9-]+", fields[1])):
            raise ValueError("unrecognized GPU memory/utilization listing")
        index, uuid, memory, utilization = int(fields[0]), fields[1], int(fields[2]), int(fields[3])
        if index not in range(8) or index in seen or not 0 <= utilization <= 100:
            raise ValueError("duplicate or invalid ClusterB GPU index/utilization")
        seen.add(index)
        if memory == 0 and utilization == 0 and uuid not in active:
            idle.append(index)
    if seen != set(range(8)):
        raise ValueError("ClusterB worker expects the complete eight-GPU inventory")
    return idle


def idle_gpus(host, run=command):
    if not HOST_PATTERN.fullmatch(host):
        raise ValueError("invalid ClusterB probe host")
    # JSON separates command outputs without shell parsing or punctuation markers.
    script = ("import json,subprocess; "
              "print(json.dumps([subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu',"
              "'--format=csv,noheader,nounits'],text=True,timeout=10),"
              "subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',"
              "'--format=csv,noheader,nounits'],text=True,timeout=10)]))")
    output = run(["ssh", *SSH_OPTIONS, "-J", "clusterb-login-01", host,
                  "python3 -c " + shlex.quote(script)])
    gpu, apps = json.loads(output)
    return parse_idle_gpus(gpu, apps)


def query_allocations(user, run=command):
    # PBS human-format stime is in the cluster's timezone, not necessarily the
    # watcher's timezone. Normalize on the login host before using it locally.
    # ClusterB PBS switches back to a human-readable table when -u is supplied,
    # even together with -F json. Filter ownership after a genuine JSON query.
    script = ("import json,subprocess,datetime\n"
              "p=json.loads(subprocess.check_output(['qstat','-f','-F','json'],text=True,timeout=20))\n"
              "owner=" + repr(user) + "\n"
              "fields=('Job_Owner','job_state','interactive','queue','exec_host','Resource_List','resources_used','stime')\n"
              "p={'Jobs':{key:{k:v for k,v in value.items() if k in fields} for key,value in p.get('Jobs',{}).items() "
              "if value.get('Job_Owner','').split('@',1)[0]==owner}}\n"
              "for j in p.get('Jobs',{}).values():\n"
              "    value=j.get('stime')\n"
              "    if isinstance(value,str) and not value.isdigit():\n"
              "        try: parsed=datetime.datetime.fromisoformat(value.replace('Z','+00:00'))\n"
              "        except ValueError: parsed=datetime.datetime.strptime(value,'%a %b %d %H:%M:%S %Y')\n"
              "        j['stime']=parsed.timestamp()\n"
              "print(json.dumps(p))")
    return json.loads(run(["ssh", *SSH_OPTIONS, "clusterb-login-01", "python3 -c " + shlex.quote(script)]))


def error_detail(error):
    stderr = getattr(error, "stderr", "") or ""
    return str(error) + (("; stderr=" + stderr.strip()[:2000]) if stderr else "")


def ready_count(config, *, maximum=16):
    """Peek at runnable tasks, holding their locks only during this local count."""
    queues, missing = sprint48.load_queues(config, "gpu")
    locks = []
    try:
        for _ in range(maximum):
            selected = overnight.claim(queues, dry_run=True,
                remaining_seconds=min(10800, max(0, config["deadline_utc_epoch"] - time.time())))
            if selected is None:
                break
            locks.append(selected[-1])
        count = len(locks)
    finally:
        for lock in locks:
            lock.close()
    return count, sprint48.all_terminal(queues, missing), missing


def dispatch_ready_only(config, seconds, worker_id):
    """Keep the caller's lease for real work only; never wait on stage gates."""
    if not 0 < seconds <= 10800:
        raise ValueError("invalid adaptive idle-worker lifetime")
    verify_identity(config)
    deadline = time.monotonic() + min(seconds, max(0, config["deadline_utc_epoch"] - time.time()))
    completed, stopping = 0, [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopping[0] and deadline - time.monotonic() >= 10:
            queues, _ = sprint48.load_queues(config, "gpu")
            selected = overnight.claim(queues, remaining_seconds=deadline - time.monotonic())
            if selected is None:
                print(json.dumps({"event": "idle_gpu_released", "worker": worker_id, "tasks_attempted": completed}), flush=True)
                return 0
            task, plan, state_root, state, lock = selected
            # Match the frozen dispatcher's shared/exclusive node-lease command.
            mode = "exclusive" if task.get("resource") == "isolated_gpu" else "shared"
            guarded = dict(task)
            guarded["argv"] = [str(Path(config["cwd"]) / ".conda/asymvlm/bin/python"), "-u", "-m",
                               "learned_score.mlsys_sprint.sprint48", "guard", "--mode", mode, "--", *task["argv"]]
            overnight.run_claim((guarded, plan, state_root, state, lock), deadline, worker_id, lambda: stopping[0])
            completed += 1
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0


def session_name(host, gpu):
    if not HOST_PATTERN.fullmatch(host) or type(gpu) is not int or gpu not in range(8):
        raise ValueError("invalid host/GPU for operations session")
    return f"{SESSION_PREFIX}-{host}-gpu{gpu}"


def tmux_sessions(run=command):
    try:
        return set(run(["tmux", "list-sessions", "-F", "#{session_name}"]).splitlines())
    except subprocess.CalledProcessError as exc:
        if "no server running" in (exc.stderr or "") or "no sessions" in (exc.stderr or ""):
            return set()
        raise


def launch_worker(allocation, gpu, config, *, config_path=None, run=command, now=None):
    verify_identity(config)
    now = time.time() if now is None else now
    name = session_name(allocation["host"], gpu)
    if not JOB_PATTERN.fullmatch(allocation["job_id"]):
        raise ValueError("invalid launch job identity")
    seconds = min(10800, int(allocation["expires"] - now - 600), int(config["deadline_utc_epoch"] - now))
    if seconds < 600:
        raise ValueError("allocation has insufficient safe lifetime at launch")
    _, logs = operation_paths(config)
    logs.mkdir(parents=True, exist_ok=True)
    log = logs / f"{name}-{int(now)}.log"
    worker = resolve_inside(WORKER, config["cwd"])
    config_path = resolve_inside(config_path or DEFAULT_CONFIG, config["cwd"])
    argv = ["ssh", *SSH_OPTIONS, "-J", "clusterb-login-01", allocation["host"],
            "bash " + shlex.join([str(worker), allocation["job_id"], str(gpu), str(seconds), str(config_path)])]
    shell = "exec " + shlex.join(argv) + " >> " + shlex.quote(str(log)) + " 2>&1"
    run(["tmux", "new-session", "-d", "-s", name, shell])
    return {"session": name, "job_id": allocation["job_id"], "host": allocation["host"],
            "gpu": gpu, "seconds": seconds, "started": now, "log": str(log),
            "configuration_sha256": config["sha256"]}


def watch(config_path, *, once=False, dry_run=False, run=command, clock=time.time, sleep=time.sleep):
    config = sprint48.read_config(config_path)
    if config.get("coordinator_role") != "clusterb" or config.get("user") != getpass.getuser():
        raise ValueError("watcher requires an owned ClusterB configuration")
    state_root, _ = operation_paths(config)
    state_root.mkdir(parents=True, exist_ok=True)
    with (state_root / "watcher.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify_identity(config, create=True)
        journal_path = state_root / "launches.json"
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {"configuration_sha256": config["sha256"], "launches": []}
        if journal.get("configuration_sha256") != config["sha256"]:
            raise ValueError("operations journal belongs to another configuration")
        cached_payload, checked_at = None, 0
        while clock() < config["deadline_utc_epoch"]:
            now = clock()
            config = sprint48.read_config(config_path)
            verify_identity(config)
            ready, terminal, missing = ready_count(config)
            snapshot = {"time": now, "ready_tasks": ready, "all_terminal": terminal,
                        "unpublished_plans": missing, "launched": [], "errors": [], "dry_run": dry_run}
            if ready:
                try:
                    if cached_payload is None or now - checked_at >= 360:
                        cached_payload = query_allocations(config["user"], run)
                        checked_at = now
                    allocations = owned_allocations(cached_payload, config["user"], now, config["deadline_utc_epoch"])
                    sessions = tmux_sessions(run)
                except (ValueError, OSError, subprocess.SubprocessError) as exc:
                    # Unknown ownership/lifetime or unavailable SSH means no
                    # attachments, not weaker validation or a different host.
                    snapshot["errors"].append({"stage": "allocation_audit", "error": error_detail(exc)})
                    cached_payload, checked_at = None, 0
                    allocations, sessions = [], set()
                for allocation in allocations:
                    if not ready:
                        break
                    host = allocation["host"]
                    if all(session_name(host, gpu) in sessions for gpu in range(8)):
                        continue
                    try:
                        idle = idle_gpus(host, run)
                    except (ValueError, OSError, subprocess.SubprocessError) as exc:
                        snapshot["errors"].append({"host": host, "error": error_detail(exc)})
                        continue
                    for gpu in idle:
                        if not ready:
                            break
                        name = session_name(host, gpu)
                        last = max((item["started"] for item in journal["launches"] if item["session"] == name), default=0)
                        if name in sessions or now - last < 60:
                            continue
                        if dry_run:
                            receipt = {"session": name, "host": host, "gpu": gpu, "would_launch": True}
                        else:
                            try:
                                receipt = launch_worker(allocation, gpu, config, config_path=config_path, run=run, now=now)
                            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                                snapshot["errors"].append({"session": name, "error": error_detail(exc)})
                                continue
                            journal["launches"].append(receipt)
                            atomic_json(journal_path, journal)
                        snapshot["launched"].append(receipt)
                        sessions.add(name)
                        ready -= 1
            atomic_json(state_root / "status.json", snapshot)
            print(json.dumps(snapshot), flush=True)
            if once or terminal:
                return snapshot
            sleep(min(60, max(0, config["deadline_utc_epoch"] - clock())))
        return {"status": "deadline_reached"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    watch(args.config, once=args.once, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
