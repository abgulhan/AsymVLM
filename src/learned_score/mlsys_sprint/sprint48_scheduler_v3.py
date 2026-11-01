"""Duration-aware v3 ClusterA submission, preserving frozen v1/v2 sources.

Uses the v1 coordinator lock/journal and validates its original configuration.
Only explicit server rejections can be retried. Lost/ambiguous acknowledgements
remain pending until a nonce identifies the accepted job or an operator supplies
an evidence-hashed recovery receipt. No jobs are cancelled by this module.
"""

import argparse
import fcntl
import json
from pathlib import Path
import re
import socket
import subprocess
import time
import uuid

from . import overnight, sprint48
from .state import atomic_json, digest, file_digest, resolve_inside
from .worker import _state


JOB_ID = re.compile(r"\d+\.clustera-pbs-[A-Za-z0-9.-]+")
NONCE_KEY = "MLSYS_SUBMISSION_NONCE"
MAX_REJECTIONS = 48
TERMINAL = {"F", "X", "C"}
# Do not equate arbitrary nonzero return codes (including 38), transport
# failures, timeouts, or empty output with a definite server rejection.
REJECTIONS = (
    r"qsub: (?:Job violates queue and/or server resource limits|"
    r"Maximum number of jobs already in queue|"
    r"would exceed (?:queue|server|user).*limit|"
    r"Access to queue is denied|Queue is not enabled|Unknown queue)",
)


def pbs_name(config):
    script = resolve_inside(config["pbs_script"], config["cwd"]).read_text()
    names = re.findall(r"^#PBS\s+-N\s+(\S+)\s*$", script, flags=re.MULTILINE)
    if len(names) != 1 or not re.fullmatch(r"mlsys-[A-Za-z0-9_-]+", names[0]):
        raise ValueError("expected one explicitly named MLSys PBS job")
    return names[0]


def classify_result(result):
    stdout, stderr = result.get("stdout", "").strip(), result.get("stderr", "").strip()
    if result.get("returncode") == 0 and JOB_ID.fullmatch(stdout):
        return "accepted"
    if (result.get("returncode") not in {None, 0} and not stdout
            and any(re.match(pattern, stderr, flags=re.IGNORECASE) for pattern in REJECTIONS)):
        return "rejected"
    return "uncertain"


def _text(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else (value or "")


def execute(argv, *, cwd=None, timeout=60):
    started = time.time()
    try:
        result = subprocess.run(argv, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return dict(argv=argv, started=started, finished=time.time(),
                    returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired as exc:
        return dict(argv=argv, started=started, finished=time.time(), returncode=None,
                    stdout=_text(exc.stdout), stderr=_text(exc.stderr), error="timeout")
    except OSError as exc:
        return dict(argv=argv, started=started, finished=time.time(), returncode=None,
                    stdout="", stderr=str(exc), error="process_launch_failed")


def nonce_matches(payload, config, intent):
    if not intent.get("nonce"):
        return []
    matched = []
    for job, value in payload.get("Jobs", {}).items():
        variables = value.get("Variable_List", {})
        if isinstance(variables, str):
            variables = dict(part.split("=", 1) for part in variables.split(",") if "=" in part)
        if not isinstance(variables, dict):
            raise ValueError("malformed PBS variable list")
        if variables.get(NONCE_KEY) == intent["nonce"]:
            if (value.get("Job_Owner", "").split("@", 1)[0] != config["user"]
                    or value.get("Job_Name") != pbs_name(config) or not JOB_ID.fullmatch(job)):
                raise ValueError("submission nonce matches unexpected ownership/name")
            matched.append(job)
    if len(matched) > 1:
        raise ValueError("multiple jobs match one submission nonce")
    return matched


def append_accepted(journal, job, *, reconciled=False):
    if any(row["job"] == job for row in journal["submissions"]):
        raise ValueError("accepted job is already in submission journal")
    intent = journal.pop("submission_intent")
    journal["submissions"].append(dict(job=job, submitted=intent["time"],
                                      reconciled=reconciled, submission_nonce=intent.get("nonce")))
    journal["consecutive_rejections"] = 0


def submit(config, journal, journal_path):
    """Durably record intent before qsub and preserve every returned byte."""
    if journal.get("submission_intent"):
        raise ValueError("unresolved submission prevents another qsub")
    if journal.get("total_rejections_v2", 0) >= MAX_REJECTIONS:
        raise ValueError("bounded server-rejection retry limit reached")
    for source, expected in config["input_hashes"].items():
        if file_digest(Path(config["cwd"]) / source) != expected:
            raise ValueError(f"submission source changed: {source}")
    intent = dict(time=time.time(), script=config["pbs_script"], nonce=uuid.uuid4().hex)
    journal["submission_intent"] = intent
    atomic_json(journal_path, journal)
    result = execute(["qsub", "-v", f"{NONCE_KEY}={intent['nonce']}", config["pbs_script"]],
                     cwd=config["cwd"])
    outcome = classify_result(result)
    journal.setdefault("attempts_v2", []).append(dict(intent=intent, outcome=outcome, result=result))
    if outcome == "accepted":
        append_accepted(journal, result["stdout"].strip())
    elif outcome == "rejected":
        journal.pop("submission_intent")
        journal["consecutive_rejections"] = journal.get("consecutive_rejections", 0) + 1
        journal["total_rejections_v2"] = journal.get("total_rejections_v2", 0) + 1
        delay = min(900, 60 * 2 ** min(journal["consecutive_rejections"] - 1, 4))
        journal["retry_after_epoch"] = time.time() + delay
    atomic_json(journal_path, journal)
    return outcome


def apply_recovery_receipt(config, journal, receipt_path):
    """An explicit manual audit is required for a legacy ambiguous result.

    Evidence may include full qstat history, tracejob/accounting, and the original
    crash log. Absence from one live qstat response is NOT sufficient evidence;
    the caller must establish rejection/non-acceptance before attesting it.
    """
    receipt_path = resolve_inside(receipt_path, config["cwd"])
    receipt = json.loads(receipt_path.read_text())
    intent = journal.get("submission_intent")
    if (not intent or receipt.get("configuration_sha256") != config["sha256"]
            or receipt.get("submission_intent_sha256") != digest(intent)):
        raise ValueError("receipt does not bind current configuration and pending intent")
    if not receipt.get("reason", "").strip() or not receipt.get("evidence"):
        raise ValueError("recovery requires reason and hashed evidence")
    for row in receipt["evidence"]:
        path = resolve_inside(row["path"], config["cwd"])
        if file_digest(path) != row["sha256"]:
            raise ValueError("recovery evidence changed")
    decision = receipt.get("decision")
    if decision == "not_submitted":
        if receipt.get("operator_verified_no_job_was_accepted") is not True:
            raise ValueError("legacy absence needs explicit completed audit attestation")
        journal.pop("submission_intent")
    elif decision == "adopt":
        job = receipt.get("job", "")
        if not JOB_ID.fullmatch(job):
            raise ValueError("recovery job ID invalid")
        append_accepted(journal, job, reconciled=True)
    else:
        raise ValueError("unknown recovery decision")
    journal.setdefault("recovery_receipts_v2", []).append(
        dict(path=str(receipt_path), sha256=file_digest(receipt_path), time=time.time(), intent=intent))


def scheduler(config, *, dry_run=False, receipt_path=None):
    if not socket.gethostname().startswith("clustera-login-"):
        raise ValueError("use the existing authenticated ClusterA login session")
    state = resolve_inside(config["state_root"], config["cwd"])
    state.mkdir(parents=True, exist_ok=True)
    with (state / "scheduler.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / "submission_journal.json"
        journal = _state(journal_path) or dict(config_sha256=config["sha256"], submissions=[])
        if journal["config_sha256"] != config["sha256"]:
            raise ValueError("journal belongs to another configuration")
        if receipt_path:
            apply_recovery_receipt(config, journal, receipt_path)
            if not dry_run:
                atomic_json(journal_path, journal)
        query_failures = 0
        while time.time() < config["deadline_utc_epoch"]:
            # Include completed jobs so an accepted-and-finished submission can
            # still be reconciled after losing its original acknowledgement.
            result = execute(["qstat", "-x", "-f", "-F", "json"], timeout=45)
            try:
                if result["returncode"] != 0:
                    raise ValueError("qstat failed")
                payload = json.loads(result["stdout"])
                if not isinstance(payload.get("Jobs", {}), dict):
                    raise ValueError("malformed PBS Jobs response")
            except (ValueError, TypeError):
                query_failures += 1
                atomic_json(state / "scheduler_status_v2.json", dict(status="qstat_failed", query=result,
                                                                     failures=query_failures, time=time.time()))
                if query_failures >= 48 or dry_run:
                    return "qstat_failed"
                time.sleep(min(300, 30 * 2 ** min(query_failures - 1, 4)))
                continue
            query_failures = 0
            if journal.get("submission_intent"):
                matches = nonce_matches(payload, config, journal["submission_intent"])
                if matches:
                    append_accepted(journal, matches[0], reconciled=True)
                    if not dry_run:
                        atomic_json(journal_path, journal)
                else:
                    atomic_json(state / "scheduler_status_v2.json", dict(
                        status="uncertain_submission_requires_audit", time=time.time(),
                        submission_intent=journal["submission_intent"], query=result))
                    return "uncertain_submission_requires_audit"
            jobs = overnight.owned_jobs(payload, config["user"])
            # Count queued debug requests too, so another workflow's queued
            # nodes cannot race our submission and exceed the two-node limit.
            reserved_nodes = sum(_debug_nodes(value) for value in payload.get("Jobs", {}).values()
                                 if value.get("Job_Owner", "").split("@", 1)[0] == config["user"]
                                 and value.get("queue") == "debug" and value.get("job_state") not in TERMINAL)
            queues, missing = sprint48.load_queues(config, "gpu")
            selected = overnight.claim(queues, dry_run=True, remaining_seconds=3300)
            ready = selected is not None
            if selected:
                selected[-1].close()
            status = dict(status="monitoring", time=time.time(), configuration_sha256=config["sha256"],
                          owned_jobs=jobs, reserved_debug_nodes_all_workflows=reserved_nodes,
                          ready_gpu_work=ready, unpublished_plans=missing,
                          queues=overnight.queue_snapshot(queues), submissions=journal["submissions"])
            atomic_json(state / "scheduler_status_v2.json", status)
            print(json.dumps(status), flush=True)
            if dry_run:
                return "dry_run"
            if len(jobs) > 1:
                return "multiple_owned_allocations_requires_audit"
            if (not jobs and reserved_nodes == 0 and ready
                    and len(journal["submissions"]) < config["maximum_submissions"]
                    and time.time() >= journal.get("retry_after_epoch", 0)
                    and time.time() + 3600 <= config["deadline_utc_epoch"]):
                outcome = submit(config, journal, journal_path)
                print(json.dumps(dict(submission_outcome=outcome, attempt=journal["attempts_v2"][-1])), flush=True)
                if outcome == "uncertain" or journal.get("total_rejections_v2", 0) >= MAX_REJECTIONS:
                    return outcome
            if not jobs and sprint48.all_terminal(queues, missing):
                return "complete"
            time.sleep(30)
    return "deadline"


def _debug_nodes(value):
    resources = value.get("Resource_List", {})
    nodes = resources.get("nodect")
    if nodes is None:
        counts = [chunk.split(":", 1)[0] for chunk in str(resources.get("select", "")).split("+")]
        if not all(item.isdigit() for item in counts):
            raise ValueError("cannot verify reserved debug node count")
        nodes = sum(map(int, counts))
    if int(nodes) < 1:
        raise ValueError("invalid reserved debug node count")
    return int(nodes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--recovery-receipt", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = sprint48.read_config(args.config)
    outcome = scheduler(config, dry_run=args.dry_run, receipt_path=args.recovery_receipt)
    print(json.dumps(dict(scheduler_outcome=outcome)), flush=True)
    return 0 if outcome in {"dry_run", "complete", "deadline"} else 1


if __name__ == "__main__":
    raise SystemExit(main())

