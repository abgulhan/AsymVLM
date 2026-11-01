"""Queue one successor without mistaking queued nodes for occupied GPUs.

Operational overlay only: frozen experiments, PBS scripts, per-run journals,
deadlines and submission allowances are unchanged. This does not provide a
cross-project fairness guarantee or change the independent KV submitter.
"""

import argparse
import fcntl
import json
from pathlib import Path
import re
import signal
import socket
import time

from learned_score.mlsys_eval.common import write_new_json
from . import cache_clustera_priority_v4 as priority
from . import overnight, clustera_queue_policy_v5 as policy, sprint48
from . import sprint48_scheduler_v3 as legacy
from .state import atomic_json, digest, file_digest, resolve_inside
from .worker import _state


FORMAT = "mlsys_clustera_queue_admission_v5"


def validate_request(config):
    """The one-running-job queue limit must cover our exact two-node request."""
    if config.get("maximum_debug_nodes") != 2:
        raise ValueError("expected unchanged two-debug-node bound")
    script = resolve_inside(config["pbs_script"], config["cwd"]).read_text()
    for pattern, expected in (
        (r"^#PBS\s+-q\s+(\S+)\s*$", "debug"),
        (r"^#PBS\s+-l\s+walltime=(\S+)\s*$", "01:00:00"),
    ):
        if re.findall(pattern, script, re.MULTILINE) != [expected]:
            raise ValueError("expected debug queue and one-hour PBS request")
    selections = re.findall(r"^#PBS\s+-l\s+select=(\S+)\s*$", script, re.MULTILINE)
    if len(selections) != 1 or not re.fullmatch(r"2(?::[A-Za-z_][\w]*=[\w.-]+)*", selections[0]):
        raise ValueError("expected exactly two PBS nodes")
    legacy.pbs_name(config)


def observe(config):
    """Fresh server inventory plus queue policy; never return partial inventory."""
    jobs = legacy.execute(["qstat", "-x", "-f", "-F", "json"], timeout=45)
    if jobs["returncode"] != 0:
        raise ValueError("job inventory query failed")
    payload = json.loads(jobs["stdout"])
    queue = legacy.execute(["qstat", "-Qf", "debug"], timeout=30)
    if queue["returncode"] != 0:
        raise ValueError("debug queue policy query failed")
    decision = policy.admission(payload, queue["stdout"], config["user"])
    decision["observed_epoch"] = time.time()
    return payload, decision


def ready_work(config):
    queues, missing = sprint48.load_queues(config, "gpu")
    selected = overnight.claim(queues, dry_run=True, remaining_seconds=3300)
    if selected:
        selected[-1].close()
    return queues, missing, selected is not None


def compact_owned(payload, config):
    # Preserve the legacy conservative guard against any other MLSys allocation.
    return {job: {key: value.get(key) for key in ("Job_Name", "job_state", "queue")}
            for job, value in overnight.owned_jobs(payload, config["user"]).items()}


def wait_reason(config, journal, owned, ready, decision, now):
    if owned:
        return "existing_mlsys_allocation"
    if not ready:
        return "no_ready_gpu_work"
    if len(journal["submissions"]) >= config["maximum_submissions"]:
        return "submission_allowance_exhausted"
    if journal.get("total_rejections_v2", 0) >= legacy.MAX_REJECTIONS:
        return "rejection_allowance_exhausted"
    if now < journal.get("retry_after_epoch", 0):
        return "submission_rejection_backoff"
    if now + 3600 > config["deadline_utc_epoch"]:
        return "insufficient_time_before_deadline"
    return decision["reason"] if not decision["eligible"] else "eligible"


def scheduler(config, *, guard=lambda: None, dry_run=False, now=None, sleep=None):
    """Reuse original journal/lock, refreshing admission immediately before qsub.

    dry_run does not submit or change journals; queue discovery can write catalog
    and lock artifacts, as in the original scheduler. CLI --dry-run uses only
    observe(), so it can safely inspect alongside the old running scheduler.
    """
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("use the existing authenticated ClusterA login session")
    now, sleep = now or time.time, sleep or time.sleep
    validate_request(config)
    state = resolve_inside(config["state_root"], config["cwd"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / "submission_journal.json"
        journal = _state(journal_path) or dict(config_sha256=config["sha256"], submissions=[])
        if journal.get("config_sha256") != config["sha256"] or not isinstance(journal.get("submissions"), list):
            raise ValueError("invalid submission journal or wrong configuration")
        failures = 0

        def persist(status, **extra):
            value = dict(format=FORMAT, status=status, time=now(), configuration_sha256=config["sha256"],
                         submissions=journal["submissions"], **extra)
            atomic_json(state / "scheduler_status_v5.json", value)
            print(json.dumps(value, sort_keys=True), flush=True)

        while now() < config["deadline_utc_epoch"]:
            guard()
            try:
                payload, decision = observe(config)
            except (ValueError, TypeError, KeyError) as exc:
                failures += 1
                persist("waiting_for_verified_queue_inventory", reason=str(exc), failures=failures)
                if failures >= 48 or dry_run:
                    return "qstat_failed"
                sleep(min(300, 30 * 2 ** min(failures - 1, 4)))
                continue
            failures = 0
            if journal.get("submission_intent"):
                matches = legacy.nonce_matches(payload, config, journal["submission_intent"])
                if not matches:
                    persist("uncertain_submission_requires_audit")
                    return "uncertain_submission_requires_audit"
                legacy.append_accepted(journal, matches[0], reconciled=True)
                if not dry_run:
                    atomic_json(journal_path, journal)
            absent = [row["job"] for row in journal["submissions"] if row["job"] not in payload["Jobs"]]
            if absent:
                # Missing from one inventory is not evidence that an accepted
                # allocation finished. Never spend the allowance twice on it.
                persist("accepted_job_missing_requires_audit", missing_jobs=absent)
                return "accepted_job_missing_requires_audit"
            owned = compact_owned(payload, config)
            queues, missing, ready = ready_work(config)
            reason = wait_reason(config, journal, owned, ready, decision, now())
            persist("monitoring", reason=reason, admission=decision, owned_jobs=owned,
                    ready_gpu_work=ready, unpublished_plans=missing,
                    queues=overnight.queue_snapshot(queues))
            if dry_run:
                return "dry_run"
            if len(owned) > 1:
                return "multiple_owned_allocations_requires_audit"
            if not owned and sprint48.all_terminal(queues, missing):
                return "complete"
            if reason == "eligible":
                # Discovery may be slow; do not qsub from the pre-discovery view.
                guard()
                _, _, fresh_ready = ready_work(config)
                try:
                    fresh_payload, fresh = observe(config)
                except (ValueError, TypeError, KeyError) as exc:
                    persist("waiting_for_verified_queue_inventory", reason=str(exc), before_submit=True)
                    sleep(30)
                    continue
                fresh_owned = compact_owned(fresh_payload, config)
                reason = wait_reason(config, journal, fresh_owned, fresh_ready, fresh, now())
                if reason != "eligible":
                    persist("admission_changed_before_submit", reason=reason, admission=fresh,
                            owned_jobs=fresh_owned, ready_gpu_work=fresh_ready)
                else:
                    outcome = legacy.submit(config, journal, journal_path)
                    persist("submission_" + outcome, admission=fresh,
                            job=journal["submissions"][-1]["job"] if outcome == "accepted" else None)
                    if outcome == "uncertain" or journal.get("total_rejections_v2", 0) >= legacy.MAX_REJECTIONS:
                        return outcome
            sleep(30)
    return "deadline"


def run(preferred_path, deferred_path, *, state_root=None):
    """Bind the new operational code while reusing frozen priority machinery."""
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("use the existing authenticated ClusterA login session")
    preferred_path, deferred_path = Path(preferred_path).resolve(), Path(deferred_path).resolve()
    preferred, deferred = sprint48.read_config(preferred_path), sprint48.read_config(deferred_path)
    priority._validate_pair(preferred, deferred)
    validate_request(preferred)
    validate_request(deferred)
    root = resolve_inside(state_root or preferred_path.parent / "queue_handoff_v5", preferred["cwd"])
    if root == Path(preferred["cwd"]).resolve():
        raise ValueError("dedicated operational state directory required")
    sources = [Path(__file__).resolve(), Path(policy.__file__).resolve(),
               Path(priority.__file__).resolve(), Path(legacy.__file__).resolve(),
               preferred_path, deferred_path]
    receipt = dict(format=FORMAT, hashes={str(path): file_digest(path) for path in sources},
                   preferred_configuration_sha256=preferred["sha256"],
                   deferred_configuration_sha256=deferred["sha256"],
                   maximum_running_nodes=2, maximum_queued_jobs=1,
                   other_workflow_unchanged=True, no_job_cancellation=True, no_deadline_extension=True)
    receipt["sha256"] = digest(receipt)
    write_new_json(root / "queue_policy_receipt.json", receipt)

    def guard():
        for path, expected in receipt["hashes"].items():
            if file_digest(path) != expected:
                raise ValueError("operational source/configuration changed: " + path)

    def preferred_runner(path):
        guard()
        current = sprint48.read_config(path)
        predecessor = resolve_inside(current["predecessor_clustera_lock"], current["cwd"])
        predecessor.parent.mkdir(parents=True, exist_ok=True)
        with predecessor.open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return scheduler(current, guard=guard)

    def deferred_runner(config):
        guard()
        return scheduler(config, guard=guard)

    return priority.run(preferred_path, deferred_path, state_root=root,
                        preferred_runner=preferred_runner, deferred_runner=deferred_runner)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preferred", required=True, type=Path)
    parser.add_argument("--deferred", required=True, type=Path)
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="read-only live admission inspection; never submits")
    args = parser.parse_args(argv)
    if args.dry_run:
        preferred = sprint48.read_config(args.preferred)
        deferred = sprint48.read_config(args.deferred)
        priority._validate_pair(preferred, deferred)
        validate_request(preferred)
        validate_request(deferred)
        _, decision = observe(preferred)
        print(json.dumps(dict(format=FORMAT, dry_run=True, admission=decision), sort_keys=True), flush=True)
        return 0

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"queue supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    result = run(args.preferred, args.deferred, state_root=args.state_root)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
