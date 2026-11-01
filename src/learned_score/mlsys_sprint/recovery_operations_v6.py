"""Six-hour recovery operations; no historical source or queue is modified.

ClusterB attaches only to verified owned idle allocations. ClusterA admits one
two-node debug request only when this owner has no nonterminal debug nodes,
including queued requests from other workflows. This deliberately does not use
v5's one-queued-successor relaxation. Submission/attachment require explicit
CLI modes; config construction and admission inspection never launch work.
"""

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import getpass
import json
import math
from pathlib import Path
import re
import time

from . import adaptive_operations_v4 as previous
from . import cache_clustera_queue_v5 as clustera
from . import clustera_queue_policy_v5 as queue_policy
from . import overnight
from . import sprint48
from . import sprint48_scheduler_v3 as legacy
from .state import digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources


START_UTC_EPOCH = int(datetime(2026, 10, 17, 12, 3, 54, tzinfo=timezone.utc).timestamp())
DEADLINE_UTC_EPOCH = START_UTC_EPOCH + 6 * 3600
TRAINING_CUTOFF_UTC_EPOCH = START_UTC_EPOCH + 75 * 60
NEW_WORK_CUTOFF_UTC_EPOCH = START_UTC_EPOCH + 5 * 3600 + 15 * 60
STUDY_ROOT = Path("reproduction_results/mlsys_recovery_20261017_v6")
DEFAULT_CONFIG = STUDY_ROOT / "coordinator/clusterb-config.json"
CLUSTERA_CONFIG = STUDY_ROOT / "coordinator/clustera-config.json"
WORKER = Path("scripts/mlsys_recovery_v6_idle_worker.sh")
PBS_SCRIPT = Path("scripts/mlsys_recovery_v6_clustera.pbs")
CLUSTERA_WORKER = Path("scripts/mlsys_recovery_v6_clustera_worker.sh")
SESSION_PREFIX = "mlsys-recovery-v6-clusterb"
PLAN_NAMES = frozenset(f"{stage}-{model}" for stage in ("allocation", "training", "stress")
                       for model in ("qwen25", "llava_next"))
SOURCE_FILES = (
    "learned_score/mlsys_sprint/recovery_operations_v6.py",
    "learned_score/mlsys_sprint/adaptive_operations_v4.py",
    "learned_score/mlsys_sprint/cache_dedicated_clusterb_v4.py",
    "learned_score/mlsys_sprint/cache_clustera_queue_v5.py",
    "learned_score/mlsys_sprint/cache_clustera_priority_v4.py",
    "learned_score/mlsys_sprint/clustera_queue_policy_v5.py",
    "learned_score/mlsys_sprint/sprint48.py",
    "learned_score/mlsys_sprint/sprint48_scheduler_v3.py",
    "learned_score/mlsys_sprint/overnight.py",
    "learned_score/mlsys_sprint/worker.py",
    "learned_score/mlsys_sprint/state.py",
    str(WORKER), str(PBS_SCRIPT), str(CLUSTERA_WORKER),
)


def build_config(cwd, plan_paths, deadline_utc_epoch=DEADLINE_UTC_EPOCH, *, role,
                 created_utc_epoch=START_UTC_EPOCH, user=None, extra_sources=()):
    """Build a frozen config; declared plans may be published later, never replaced."""
    cwd = Path(cwd).resolve()
    user = getpass.getuser() if user is None else user
    plans_root = cwd / STUDY_ROOT / "plans"
    paths = [resolve_inside(path, plans_root) for path in plan_paths]
    sources = sorted({*SOURCE_FILES, *(str(path) for path in extra_sources)})
    hashes = {str(resolve_inside(path, cwd)): file_digest(resolve_inside(path, cwd)) for path in sources}
    value = {
        "format": sprint48.FORMAT, "coordinator_role": role, "cwd": str(cwd),
        "user": user, "created_utc_epoch": created_utc_epoch,
        "deadline_utc_epoch": deadline_utc_epoch,
        "state_root": str(cwd / STUDY_ROOT / "coordinator" / role),
        "pbs_script": str(PBS_SCRIPT), "plan_paths": [str(path) for path in paths],
        "maximum_debug_nodes": 2, "maximum_submissions": 6,
        "maximum_nonterminal_debug_nodes": 2,
        "input_hashes": hashes, "resource_map": {"recovery-v6": "gpu"},
        "study_operations": "recovery_v6_owned_idle_six_hours",
        "clustera_models": ["llava_next"], "qwen_site": "clusterb_owned_idle",
    }
    value = {**value, "sha256": digest(value)}
    return validate_config(value)


def validate_config(config):
    root = Path(config["cwd"]).resolve()
    role = config.get("coordinator_role")
    if (config.get("study_operations") != "recovery_v6_owned_idle_six_hours"
            or config.get("format") != sprint48.FORMAT
            or config.get("sha256") != digest({k: v for k, v in config.items() if k != "sha256"})
            or role not in {"clusterb", "clustera"}
            or type(config.get("created_utc_epoch")) not in (int, float)
            or config["created_utc_epoch"] != START_UTC_EPOCH
            or type(config.get("deadline_utc_epoch")) not in (int, float)
            or config["deadline_utc_epoch"] != DEADLINE_UTC_EPOCH
            or config.get("maximum_debug_nodes") != 2
            or config.get("maximum_nonterminal_debug_nodes") != 2
            or config.get("maximum_submissions") != 6
            or config.get("clustera_models") != ["llava_next"]
            or config.get("qwen_site") != "clusterb_owned_idle"
            or not isinstance(config.get("user"), str)
            or not re.fullmatch(r"[a-z_][a-z0-9_-]*", config["user"])):
        raise ValueError("not the fixed six-hour, two-node recovery-v6 configuration")
    if (resolve_inside(config["state_root"], root) != root / STUDY_ROOT / "coordinator" / role
            or resolve_inside(config["pbs_script"], root) != root / PBS_SCRIPT):
        raise ValueError("recovery operations cannot use another study's state or PBS script")
    plans = config.get("plan_paths", [])
    if not plans or len(plans) != len(set(plans)):
        raise ValueError("declare unique recovery plan paths")
    plans_root = root / STUDY_ROOT / "plans"
    for name in plans:
        path = resolve_inside(name, plans_root)
        if (path.name != "plan.json" or path.parent.parent != plans_root
                or path.parent.name not in PLAN_NAMES
                or role == "clustera" and not path.parent.name.endswith("-llava_next")):
            raise ValueError("unknown recovery queue or non-LLaVA ClusterA queue")
    pins = config.get("input_hashes", {})
    if any(str(resolve_inside(name, root)) not in pins for name in SOURCE_FILES):
        raise ValueError("recovery safety sources are not all pinned")
    for name in pins:
        resolve_inside(name, root)
    return config


def validate_published_plans(config):
    """Fail closed on source drift or a mislabeled incompatible ClusterA task."""
    validate_config(config)
    root = Path(config["cwd"]).resolve()
    for name, expected in config["input_hashes"].items():
        if file_digest(resolve_inside(name, root)) != expected:
            raise ValueError("recovery source changed: " + name)
    for name in config["plan_paths"]:
        path = resolve_inside(name, root)
        if not path.exists():
            continue
        plan = read_plan(path)
        if Path(plan["cwd"]).resolve() != root:
            raise ValueError("recovery plan belongs to another workspace")
        expected_deadline = (TRAINING_CUTOFF_UTC_EPOCH if path.parent.name.startswith("training-")
                             else NEW_WORK_CUTOFF_UTC_EPOCH)
        if plan_deadline(plan) != expected_deadline:
            raise ValueError("recovery plan changed its predeclared stage expiry")
        verify_sources(plan)
        if config["coordinator_role"] == "clustera":
            for task in plan["tasks"]:
                if task.get("resource", "gpu") == "cpu":
                    continue
                argv = task.get("argv", [])
                if (task.get("model_tag") != "llava_next" or not isinstance(argv, list)
                        or any(any(token in str(arg).lower() for token in ("qwen", "phi3", "phi35")) for arg in argv)):
                    raise ValueError("ClusterA recovery tasks require explicit LLaVA-only compatibility")
                if "--adapter" in argv and argv[argv.index("--adapter") + 1:][:1] != ["llava_next"]:
                    raise ValueError("ClusterA recovery task has an incompatible adapter")


def plan_deadline(plan):
    value = plan.get("deadline_utc_epoch")
    if (type(value) not in (int, float) or not math.isfinite(value)
            or not START_UTC_EPOCH < value <= NEW_WORK_CUTOFF_UTC_EPOCH):
        raise ValueError("recovery plan requires a finite bounded stage expiry")
    return value


@contextmanager
def runtime_limits():
    """Enforce stage expiry in the unchanged readiness and task-runner helpers."""
    original_load = sprint48.load_queues
    original_claim = overnight.claim
    original_run = overnight.run_claim

    def validated_queues(current, resource):
        validate_published_plans(current)
        return original_load(current, resource)

    def bounded_claim(queue_list, *, dry_run=False, remaining_seconds=None):
        for queue in queue_list:
            remaining = max(0., plan_deadline(queue[1]) - time.time())
            if remaining_seconds is not None:
                remaining = min(remaining, remaining_seconds)
            if remaining <= 0:
                continue
            selected = original_claim([queue], dry_run=dry_run, remaining_seconds=remaining)
            if selected is not None:
                return selected
        return None

    def bounded_run(selected, deadline, worker_id, stopping):
        try:
            remaining = max(0., plan_deadline(selected[1]) - time.time())
        except BaseException:
            selected[-1].close()
            raise
        effective = min(deadline, time.monotonic() + remaining)
        return original_run(selected, effective, worker_id, stopping)

    try:
        sprint48.load_queues = validated_queues
        overnight.claim = bounded_claim
        overnight.run_claim = bounded_run
        yield
    finally:
        sprint48.load_queues = original_load
        overnight.claim = original_claim
        overnight.run_claim = original_run


def operation_paths(config):
    coordinator = resolve_inside(config["state_root"], config["cwd"]).parent
    return coordinator / "recovery-operations-v6" / config["sha256"][:12], coordinator / "logs"


def operation_identity(config):
    root = Path(config["cwd"]).resolve()
    sources = {str(resolve_inside(path, root)): file_digest(resolve_inside(path, root)) for path in SOURCE_FILES}
    return {"format": "mlsys_recovery_operations_v6", "configuration_sha256": config["sha256"],
            "operations_version": "audited_owned_idle_strict_debug_total_v1", "owner": config["user"],
            "cwd": str(root), "worker": str(resolve_inside(WORKER, root)), "source_sha256": sources,
            "deadline_utc_epoch": DEADLINE_UTC_EPOCH}


@contextmanager
def installed():
    replacements = {"DEFAULT_CONFIG": DEFAULT_CONFIG, "WORKER": WORKER,
                    "SESSION_PREFIX": SESSION_PREFIX, "operation_paths": operation_paths,
                    "operation_identity": operation_identity}
    saved = {name: getattr(previous, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(previous, name, value)
        yield
    finally:
        for name, value in saved.items():
            setattr(previous, name, value)


def verify_identity(config, *, create=False):
    validate_published_plans(config)
    with installed():
        return previous.verify_identity(config, create=create)


def dispatch_ready_only(config, seconds, worker_id):
    validate_published_plans(config)
    if config["coordinator_role"] != "clusterb":
        raise ValueError("owned-idle dispatch is ClusterB-only")
    with installed(), runtime_limits():
        return previous.dispatch_ready_only(config, seconds, worker_id)


def watch(config_path=DEFAULT_CONFIG, *, once=False, dry_run=False, **kwargs):
    validate_published_plans(sprint48.read_config(config_path))
    with installed(), runtime_limits():
        return previous.watch(config_path, once=once, dry_run=dry_run, **kwargs)


def admission(payload, queue_text, user):
    """Count queued, running, held, and transitional nodes across all owned jobs."""
    result = queue_policy.admission(payload, queue_text, user)
    reserved = sum(row["nodes"] for field in ("running_jobs", "queued_jobs", "other_jobs")
                   for row in result[field])
    return {**result, "reserved_debug_nodes_all_workflows": reserved,
            "eligible": reserved == 0,
            "reason": "eligible" if reserved == 0 else "waiting_for_all_owned_debug_nodes",
            "maximum_nonterminal_debug_nodes": 2, "request_nodes": 2}


def observe(config):
    """Read-only fresh server inventory and queue policy; no fallback on error."""
    validate_config(config)
    jobs = legacy.execute(["qstat", "-x", "-f", "-F", "json"], timeout=45)
    if jobs["returncode"] != 0:
        raise ValueError("job inventory query failed")
    payload = json.loads(jobs["stdout"])
    queue = legacy.execute(["qstat", "-Qf", "debug"], timeout=30)
    if queue["returncode"] != 0:
        raise ValueError("debug queue policy query failed")
    return payload, admission(payload, queue["stdout"], config["user"])


def scheduler(config, *, dry_run=False, **kwargs):
    validate_published_plans(config)
    if config["coordinator_role"] != "clustera" or config["user"] != getpass.getuser():
        raise ValueError("ClusterA scheduler requires this owner's ClusterA config")
    clustera.validate_request(config)
    if dry_run:
        return observe(config)[1]
    original = clustera.observe
    try:
        clustera.observe = observe
        with runtime_limits():
            return clustera.scheduler(config, guard=lambda: validate_published_plans(config), **kwargs)
    finally:
        clustera.observe = original


def dispatch(config, seconds, worker_id):
    validate_published_plans(config)
    if (config["coordinator_role"] != "clustera" or config["user"] != getpass.getuser()
            or not 0 < seconds <= 3000):
        raise ValueError("ClusterA dispatch must use a bounded ClusterA configuration")
    with runtime_limits():
        return sprint48.dispatch(config, "gpu", seconds, worker_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--watch", action="store_true")
    mode.add_argument("--once", action="store_true")
    mode.add_argument("--clustera", action="store_true")
    mode.add_argument("--dispatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seconds", type=int, default=3000)
    parser.add_argument("--worker-id")
    args = parser.parse_args(argv)
    path = args.config or (CLUSTERA_CONFIG if args.clustera or args.dispatch else DEFAULT_CONFIG)
    if args.dispatch:
        if args.dry_run or not args.worker_id:
            parser.error("dispatch needs --worker-id and does not accept --dry-run")
        return dispatch(sprint48.read_config(path), args.seconds, args.worker_id)
    if args.clustera:
        result = scheduler(sprint48.read_config(path), dry_run=args.dry_run)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    watch(path, once=args.once, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
