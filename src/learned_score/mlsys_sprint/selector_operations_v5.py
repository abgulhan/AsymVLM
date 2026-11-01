"""Independent selector-study operations, reusing the audited v4 safety layer.

Configuration construction is pure. Watching or dispatching must be requested
explicitly, after the study has frozen its sources and published its queues.
No command in this module submits a ClusterB allocation or cancels another job.
"""

import argparse
from contextlib import contextmanager
import getpass
from pathlib import Path
import re
import time

from . import adaptive_operations_v4 as previous
from . import sprint48
from .state import digest, file_digest, resolve_inside


STUDY_ROOT = Path("reproduction_results/mlsys_selector_20261017_v5")
DEFAULT_CONFIG = STUDY_ROOT / "coordinator/clusterb-config.json"
WORKER = Path("scripts/mlsys_selector_v5_idle_worker.sh")
PBS_SCRIPT = Path("scripts/mlsys_selector_v5_clustera.pbs")
CLUSTERA_WORKER = Path("scripts/mlsys_selector_v5_clustera_worker.sh")
SESSION_PREFIX = "mlsys-selector-v5-clusterb"
SOURCE_FILES = (
    "learned_score/mlsys_sprint/selector_operations_v5.py",
    "learned_score/mlsys_sprint/adaptive_operations_v4.py",
    "learned_score/mlsys_sprint/cache_dedicated_clusterb_v4.py",
    "learned_score/mlsys_sprint/sprint48.py",
    "learned_score/mlsys_sprint/sprint48_scheduler_v3.py",
    "learned_score/mlsys_sprint/overnight.py",
    "learned_score/mlsys_sprint/worker.py",
    "learned_score/mlsys_sprint/state.py",
    str(WORKER), str(PBS_SCRIPT), str(CLUSTERA_WORKER),
)


def build_config(cwd, plan_paths, deadline_utc_epoch, *, role,
                 created_utc_epoch=None, user=None, extra_sources=()):
    """Return a source-pinned config; do not write files or inspect a cluster.

    Plans may be published later, but their exact paths are declared now. The
    caller supplies separate compatible plan lists for ClusterB and ClusterA.
    """
    cwd = Path(cwd).resolve()
    if role not in {"clusterb", "clustera"}:
        raise ValueError("selector coordinator role must be clusterb or clustera")
    user = getpass.getuser() if user is None else user
    if not isinstance(user, str) or not re.fullmatch(r"[a-z_][a-z0-9_-]*", user):
        raise ValueError("invalid coordinator owner")
    created = time.time() if created_utc_epoch is None else created_utc_epoch
    if (isinstance(created, bool) or isinstance(deadline_utc_epoch, bool)
            or not 0 < float(deadline_utc_epoch) - float(created) <= 86400):
        raise ValueError("selector execution window must be positive and at most 24 hours")
    plans_root = cwd / STUDY_ROOT / "plans"
    paths = [resolve_inside(path, plans_root) for path in plan_paths]
    if (not paths or len(set(paths)) != len(paths)
            or any(path.name != "plan.json" or path.parent == plans_root for path in paths)):
        raise ValueError("declare unique named plan.json paths under this study's plans directory")
    sources = sorted({*SOURCE_FILES, *(str(path) for path in extra_sources)})
    hashes = {str(resolve_inside(path, cwd)): file_digest(resolve_inside(path, cwd)) for path in sources}
    config = {
        "format": sprint48.FORMAT, "coordinator_role": role, "cwd": str(cwd),
        "user": user, "created_utc_epoch": created,
        "deadline_utc_epoch": deadline_utc_epoch,
        "state_root": str(cwd / STUDY_ROOT / "coordinator" / role),
        "pbs_script": str(PBS_SCRIPT), "plan_paths": [str(path) for path in paths],
        "maximum_debug_nodes": 2, "maximum_submissions": 24,
        "input_hashes": hashes, "resource_map": {"selector-v5": "gpu"},
        "study_operations": "selector_v5_owned_idle_only",
    }
    config = {**config, "sha256": digest(config)}
    validate_config(config)
    return config


def validate_config(config):
    """Never attach this new worker to a historical or unbounded study."""
    root = Path(config["cwd"]).resolve()
    role = config.get("coordinator_role")
    if (config.get("study_operations") != "selector_v5_owned_idle_only"
            or config.get("format") != sprint48.FORMAT
            or config.get("sha256") != digest({key: value for key, value in config.items() if key != "sha256"})
            or role not in {"clusterb", "clustera"}
            or not 0 < config["deadline_utc_epoch"] - config["created_utc_epoch"] <= 86400
            or config.get("maximum_debug_nodes") != 2
            or config.get("maximum_submissions") != 24):
        raise ValueError("not a bounded, frozen selector-v5 configuration")
    if (resolve_inside(config["state_root"], root) != root / STUDY_ROOT / "coordinator" / role
            or resolve_inside(config["pbs_script"], root) != root / PBS_SCRIPT):
        raise ValueError("selector operations cannot use another study's state or submission script")
    plans = config.get("plan_paths", [])
    if not plans or len(plans) != len(set(plans)):
        raise ValueError("selector plans must be explicitly declared and unique")
    for name in plans:
        path = resolve_inside(name, root / STUDY_ROOT / "plans")
        if path.name != "plan.json" or path.parent == root / STUDY_ROOT / "plans":
            raise ValueError("selector plan needs a named study queue")
    pins = config.get("input_hashes", {})
    if any(str(resolve_inside(name, root)) not in pins for name in SOURCE_FILES):
        raise ValueError("selector operations safety sources are not all pinned")
    return config


def operation_paths(config):
    coordinator = resolve_inside(config["state_root"], config["cwd"]).parent
    return coordinator / "selector-operations-v5" / config["sha256"][:12], coordinator / "logs"


def operation_identity(config):
    root = Path(config["cwd"]).resolve()
    sources = {str(resolve_inside(path, root)): file_digest(resolve_inside(path, root))
               for path in SOURCE_FILES}
    return {"format": "mlsys_selector_operations_v5", "configuration_sha256": config["sha256"],
            "operations_version": "v4_owned_idle_wrapper_v1", "owner": config["user"],
            "cwd": str(root), "worker": str(resolve_inside(WORKER, root)),
            "source_sha256": sources}


@contextmanager
def installed():
    """Change only this process's entry points; leave frozen files untouched."""
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
    validate_config(config)
    with installed():
        return previous.verify_identity(config, create=create)


def dispatch_ready_only(config, seconds, worker_id):
    validate_config(config)
    with installed():
        return previous.dispatch_ready_only(config, seconds, worker_id)


def watch(config_path=DEFAULT_CONFIG, *, once=False, dry_run=False, **kwargs):
    validate_config(sprint48.read_config(config_path))
    with installed():
        return previous.watch(config_path, once=once, dry_run=dry_run, **kwargs)


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
