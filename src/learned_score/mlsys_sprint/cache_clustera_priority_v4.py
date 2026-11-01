"""Run the frozen v4 ClusterA submitter first, then safely restore frozen v3.

This is an allocation-order supervisor only. It never edits either experiment,
changes a deadline, cancels a PBS job, or signals workers. Both schedulers retain
their existing global two-node checks, submission journals and file locks.
"""

import argparse
import fcntl
import json
from pathlib import Path
import signal
import socket
import time

from learned_score.mlsys_eval.common import write_new_json
from . import sprint48
from .state import atomic_json, digest, file_digest, resolve_inside


FORMAT = "mlsys_clustera_priority_handoff_v4"
SAFE_OUTCOMES = {"complete", "deadline"}


def _preferred_runner(path):
    from .cache_coordinator_v4 import clustera
    return clustera(path)


def _deferred_runner(config):
    from .sprint48_scheduler_v3 import scheduler
    return scheduler(config)


def _journal_clear(config):
    path = resolve_inside(config["state_root"], config["cwd"]) / "submission_journal.json"
    if not path.exists():
        return
    journal = json.loads(path.read_text())
    if journal.get("config_sha256") != config["sha256"]:
        raise ValueError("submission journal belongs to another configuration")
    if journal.get("submission_intent"):
        raise ValueError("unresolved submission intent requires audit before scheduler handoff")


def _validate_pair(preferred, deferred):
    if (preferred.get("coordinator_role") != "clustera" or deferred.get("coordinator_role") != "clustera"
            or preferred["cwd"] != deferred["cwd"] or preferred["user"] != deferred["user"]):
        raise ValueError("handoff requires the same ClusterA user and workspace")
    if preferred.get("maximum_debug_nodes") != 2 or deferred.get("maximum_debug_nodes") != 2:
        raise ValueError("both schedulers must retain the two-debug-node bound")
    expected_lock = resolve_inside(deferred["state_root"], deferred["cwd"]) / "scheduler.lock"
    if Path(preferred.get("predecessor_clustera_lock", "")).resolve() != expected_lock:
        raise ValueError("preferred scheduler does not serialize against the deferred scheduler lock")


def run(preferred_path, deferred_path, *, state_root=None, preferred_runner=None,
        deferred_runner=None, now=None, sleep=None):
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("run through the existing authenticated ClusterA login session")
    now, sleep = now or time.time, sleep or time.sleep
    preferred_runner = preferred_runner or _preferred_runner
    deferred_runner = deferred_runner or _deferred_runner
    preferred_path, deferred_path = Path(preferred_path).resolve(), Path(deferred_path).resolve()
    preferred, deferred = sprint48.read_config(preferred_path), sprint48.read_config(deferred_path)
    _validate_pair(preferred, deferred)
    root = resolve_inside(state_root or preferred_path.parent / "priority_handoff", preferred["cwd"])
    if root == Path(preferred["cwd"]).resolve():
        raise ValueError("handoff state must have a dedicated subdirectory")
    root.mkdir(parents=True, exist_ok=True)
    receipt = {"format": FORMAT, "preferred_config": str(preferred_path),
               "preferred_config_file_sha256": file_digest(preferred_path),
               "preferred_configuration_sha256": preferred["sha256"],
               "deferred_config": str(deferred_path), "deferred_config_file_sha256": file_digest(deferred_path),
               "deferred_configuration_sha256": deferred["sha256"],
               "supervisor_source_sha256": file_digest(__file__),
               "no_job_cancellation": True, "no_deadline_extension": True}
    receipt["sha256"] = digest(receipt)
    with (root / "control.lock").open("a+") as control:
        fcntl.flock(control, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write_new_json(root / "receipt.json", receipt)
        terminal_path, status_path = root / "terminal.json", root / "status.json"
        if terminal_path.exists():
            terminal = json.loads(terminal_path.read_text())
            if terminal.get("receipt_sha256") != receipt["sha256"]:
                raise ValueError("terminal record belongs to another handoff")
            return terminal
        state = json.loads(status_path.read_text()) if status_path.exists() else {
            "format": FORMAT, "receipt_sha256": receipt["sha256"], "phase": "preferred",
            "preferred_outcome": None, "deferred_outcome": None}
        if state.get("receipt_sha256") != receipt["sha256"] or state.get("phase") not in {"preferred", "deferred"}:
            raise ValueError("invalid or changed handoff state")

        def persist(status, **extra):
            state.update(status=status, updated_epoch=now(), **extra)
            atomic_json(status_path, state)

        def terminal(status, success, **extra):
            persist(status, **extra)
            result = {**state, "terminal": True, "success": success, "finished_epoch": now()}
            write_new_json(terminal_path, result)
            return result

        def revalidate():
            if (file_digest(preferred_path) != receipt["preferred_config_file_sha256"]
                    or file_digest(deferred_path) != receipt["deferred_config_file_sha256"]
                    or file_digest(__file__) != receipt["supervisor_source_sha256"]):
                raise ValueError("handoff configuration or supervisor source changed")
            return sprint48.read_config(preferred_path), sprint48.read_config(deferred_path)

        try:
            if state["phase"] == "preferred":
                while True:
                    preferred, deferred = revalidate()
                    if now() >= preferred["deadline_utc_epoch"]:
                        outcome = "deadline"
                        break
                    persist("preferred_scheduler_running")
                    try:
                        outcome = preferred_runner(preferred_path)
                        break
                    except BlockingIOError:
                        persist("waiting_for_existing_preferred_lock")
                        sleep(30)
                persist("preferred_scheduler_returned", preferred_outcome=outcome)
                if outcome not in SAFE_OUTCOMES:
                    return terminal("preferred_scheduler_requires_audit", False)
                preferred, deferred = revalidate()
                _journal_clear(preferred)
                _journal_clear(deferred)
                persist("ready_to_restore_deferred", phase="deferred")
            if state.get("preferred_outcome") not in SAFE_OUTCOMES:
                raise ValueError("deferred scheduler cannot bypass a safe preferred outcome")
            while True:
                preferred, deferred = revalidate()
                _journal_clear(preferred)
                _journal_clear(deferred)
                if now() >= deferred["deadline_utc_epoch"]:
                    return terminal("deferred_deadline_already_expired", True,
                                    deferred_outcome="not_started_deadline_expired")
                persist("deferred_scheduler_running")
                try:
                    outcome = deferred_runner(deferred)
                    break
                except BlockingIOError:
                    persist("waiting_for_existing_deferred_lock")
                    sleep(30)
            persist("deferred_scheduler_returned", deferred_outcome=outcome)
            if outcome not in SAFE_OUTCOMES:
                return terminal("deferred_scheduler_requires_audit", False)
            _journal_clear(deferred)
            return terminal("handoff_complete", True)
        except BaseException as exc:
            # Includes operator interrupts: never advance to v3 after an error.
            return terminal("handoff_requires_audit", False, error_type=type(exc).__name__, error=str(exc))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferred", type=Path, required=True)
    parser.add_argument("--deferred", type=Path, required=True)
    parser.add_argument("--state-root", type=Path)
    args = parser.parse_args(argv)
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"supervisor received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run(args.preferred, args.deferred, state_root=args.state_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
