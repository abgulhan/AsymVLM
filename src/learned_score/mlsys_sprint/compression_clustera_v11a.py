"""Bounded ClusterA submission using PBS's verified running-node cap.

This additive submitter does not change the frozen v11 dispatcher or its old
disabled submission policy. The October 21 request authorizes a new submission
path: debug max_run=1/user and resources_max.nodect=2 enforce at most two
RUNNING debug nodes across the account's workflows. This is not a cooperative
global queue lock: another workflow may race to queue a successor. We never
cancel or modify its jobs, and allow only one outstanding job of our own.
Revision c waits for the queued slot: PBS rejected a second queued job on
October 21 even though that additional server limit was absent from -Qf debug.
"""
import argparse
import copy
import fcntl
import getpass
import json
from pathlib import Path
import socket
import time

from . import compression_operations_v11 as frozen
from . import clustera_queue_policy_v5 as queue_policy
from . import sprint48_scheduler_v3 as submissions
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_clustera_server_cap_v11c'
STATE_DIRECTORY = 'coordinator/clustera-v11c'


def normalize_inventory(payload):
    value = copy.deepcopy(payload)
    if not isinstance(value, dict) or not isinstance(value.get('Jobs'), dict):
        raise ValueError('malformed PBS inventory')
    for job in value['Jobs'].values():
        if not isinstance(job, dict):
            raise ValueError('malformed PBS job')
        resources = job.get('Resource_List', {})
        if not isinstance(resources, dict):
            raise ValueError('malformed PBS resources')
        select = resources.get('select')
        if type(select) is int:
            if select < 1:
                raise ValueError('invalid numeric PBS select')
            resources['select'] = str(select)
    return value


def admission(payload, queue_text, user):
    decision = queue_policy.admission(normalize_inventory(payload), queue_text, user)
    return {**decision, 'running_node_cap': 2, 'request_nodes': 2,
        'enforcement': 'PBS debug max_run=1 per user and maximum two nodes per job',
        'joint_submission_lock': False,
        'global_queued_job_limit_guaranteed': False}


def prepare(config_path, *, max_submissions=4):
    if type(max_submissions) is not int or not 1 <= max_submissions <= 4:
        raise ValueError('at most four accepted debug allocations per authorized run')
    path = Path(config_path).resolve()
    config = frozen.read_config(path)
    frozen.validate_published_plans(config)
    if config['coordinator_role'] != 'clustera' or config['user'] != getpass.getuser():
        raise ValueError('this owner\'s ClusterA dispatch configuration required')
    root = Path(config['study_root'])/STATE_DIRECTORY
    output = root/'authorization.json'
    if output.exists():
        auth = json.loads(output.read_text())
        validate(auth)
        if auth['config_path'] != str(path) or auth['maximum_submissions'] != max_submissions:
            raise ValueError('cannot replace an existing submission authorization')
        return auth
    body = {'format': FORMAT, 'config_path': str(path), 'parent_config_sha256': config['sha256'],
        'state_root': str(root), 'user': config['user'], 'created_utc_epoch': time.time(),
        'deadline_utc_epoch': config['deadline_utc_epoch'], 'maximum_submissions': max_submissions,
        'authorization': 'debug execution enabled',
        'maximum_running_debug_nodes_all_workflows': 2, 'maximum_own_nonterminal_jobs': 1,
        'maximum_existing_queued_jobs_at_observation': 0,
        'submission_model': 'server_enforced_running_limit_not_joint_queue_lock',
        'input_hashes': {**config['input_hashes'], str(path): file_digest(path),
            str(Path(__file__).resolve()): file_digest(__file__)}}
    auth = {**body, 'sha256': digest(body)}
    write_immutable(output, auth)
    return auth


def validate(auth):
    if (auth.get('format') != FORMAT or auth.get('sha256') != digest({k:v for k,v in auth.items() if k!='sha256'})
            or type(auth.get('maximum_submissions')) is not int or not 1 <= auth['maximum_submissions'] <= 4
            or auth.get('maximum_running_debug_nodes_all_workflows') != 2
            or auth.get('maximum_own_nonterminal_jobs') != 1
            or auth.get('maximum_existing_queued_jobs_at_observation') != 0
            or auth.get('submission_model') != 'server_enforced_running_limit_not_joint_queue_lock'):
        raise ValueError('invalid server-cap authorization')
    for path, expected in auth['input_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('authorized submission source changed: '+path)
    config = frozen.read_config(auth['config_path'])
    frozen.validate_published_plans(config)
    if (config['sha256'] != auth['parent_config_sha256'] or config['coordinator_role'] != 'clustera'
            or config['user'] != auth['user'] or config['deadline_utc_epoch'] != auth['deadline_utc_epoch']
            or Path(auth['state_root']) != Path(config['study_root'])/STATE_DIRECTORY):
        raise ValueError('authorization differs from its immutable dispatcher')
    required_pins = set(config['input_hashes']) | {str(Path(__file__).resolve()), auth['config_path']}
    if not required_pins <= set(auth['input_hashes']):
        raise ValueError('missing authorized source pins')
    legacy_roots = [Path(config['state_root']),
        Path(config['study_root'])/'coordinator/clustera-v11a',
        Path(config['study_root'])/'coordinator/clustera-v11b']
    for legacy_root in legacy_roots:
        legacy_journal = legacy_root/'submission_journal.json'
        if legacy_journal.exists():
            legacy = json.loads(legacy_journal.read_text())
            if legacy.get('submission_intent') or legacy.get('submissions'):
                raise ValueError('audit legacy submissions before using separate authorization')
    return config


def observe(config, run):
    jobs = run(['qstat', '-x', '-f', '-F', 'json'], timeout=45)
    queue = run(['qstat', '-Qf', 'debug'], timeout=30)
    if jobs['returncode'] != 0 or queue['returncode'] != 0:
        raise ValueError('cannot verify ClusterA jobs and queue policy')
    payload = normalize_inventory(json.loads(jobs['stdout']))
    return payload, admission(payload, queue['stdout'], config['user']), queue['stdout']


def tick(auth, journal, journal_path, *, run=submissions.execute, now=None, dry_run=False):
    config = validate(auth)
    live_clock = now is None
    now = time.time() if now is None else now
    if journal.get('config_sha256') != auth['sha256']:
        raise ValueError('submission journal belongs to another authorization')
    payload, decision, queue_text = observe(config, run)
    snapshot = {'time': now, 'status': 'monitoring', 'admission': decision,
        'configuration_sha256': auth['sha256'], 'submissions': list(journal['submissions'])}
    if journal.get('submission_intent'):
        matches = submissions.nonce_matches(payload, config, journal['submission_intent'])
        if not matches:
            return {**snapshot, 'status': 'uncertain_submission_requires_audit'}
        if dry_run:
            return {**snapshot, 'status': 'pending_submission_reconcilable', 'job': matches[0]}
        submissions.append_accepted(journal, matches[0], reconciled=True)
        atomic_json(journal_path, journal)
    own_jobs = []
    for accepted in journal['submissions']:
        job = payload['Jobs'].get(accepted['job'])
        if job is None:
            return {**snapshot, 'status': 'accepted_job_missing_requires_audit'}
        if (job.get('Job_Owner', '').split('@')[0] != config['user'] or
                job.get('Job_Name') != submissions.pbs_name(config) or job.get('queue') != 'debug'):
            raise ValueError('accepted job ownership, queue, or name changed')
        if job.get('job_state') not in submissions.TERMINAL:
            own_jobs.append(accepted['job'])
    # Also detect a job launched manually or by another process, even if this
    # journal did not accept it. Do not submit a competing same-study job.
    same_name = [jid for jid, job in payload['Jobs'].items()
        if job.get('Job_Owner', '').split('@')[0] == config['user']
        and job.get('Job_Name') == submissions.pbs_name(config)
        and job.get('job_state') not in submissions.TERMINAL]
    if own_jobs or same_name:
        return {**snapshot, 'status': 'own_allocation_outstanding', 'jobs': sorted(set(own_jobs+same_name))}
    ready, terminal, missing = frozen.ready_work(config, seconds=3000)
    snapshot.update(ready_tasks=ready, all_terminal=terminal, unpublished_plans=missing)
    if terminal:
        return {**snapshot, 'status': 'complete'}
    if len(journal['submissions']) >= auth['maximum_submissions']:
        return {**snapshot, 'status': 'submission_budget_exhausted'}
    if now + 3600 >= auth['deadline_utc_epoch']:
        return {**snapshot, 'status': 'insufficient_study_lifetime'}
    if not ready or not decision['eligible']:
        return {**snapshot, 'status': 'waiting_for_ready_work_or_debug_queue'}
    if journal.get('total_rejections_v2', 0) >= 4:
        return {**snapshot, 'status': 'bounded_rejections_exhausted'}
    if now < journal.get('retry_after_epoch', 0):
        return {**snapshot, 'status': 'waiting_after_explicit_rejection'}
    if dry_run:
        return {**snapshot, 'status': 'eligible', 'would_submit': True}
    # An independent submitter may race this second read. PBS atomically limits
    # running jobs; we make no false claim of a globally locked queue.
    latest_payload, latest, latest_queue = observe(config, run)
    if not latest['eligible']:
        return {**snapshot, 'status': 'queue_changed_before_submission', 'admission': latest}
    appeared = [jid for jid, job in latest_payload['Jobs'].items()
        if job.get('Job_Owner', '').split('@')[0] == config['user']
        and job.get('Job_Name') == submissions.pbs_name(config)
        and job.get('job_state') not in submissions.TERMINAL]
    if appeared:
        return {**snapshot, 'status': 'own_allocation_outstanding', 'jobs': sorted(appeared)}
    validate(auth)
    ready, terminal, missing = frozen.ready_work(config, seconds=3000)
    snapshot.update(ready_tasks=ready, all_terminal=terminal, unpublished_plans=missing)
    if terminal or not ready:
        return {**snapshot, 'status': 'complete' if terminal else 'waiting_for_ready_work_or_debug_queue'}
    now = time.time() if live_clock else now
    if now + 3600 >= auth['deadline_utc_epoch']:
        return {**snapshot, 'status': 'insufficient_study_lifetime'}
    request = {**config, 'input_hashes': auth['input_hashes']}
    atomic_json(Path(auth['state_root'])/'last_admission.json', {
        'time': now, 'admission': latest, 'queue_policy_text': latest_queue,
        'authorization_sha256': auth['sha256'], 'work_ready': ready})
    # The existing helper durably writes a nonce before qsub and never retries
    # an ambiguous acknowledgement. Submission is the only external mutation.
    outcome = submissions.submit(request, journal, journal_path)
    return {**snapshot, 'status': 'submitted' if outcome == 'accepted' else outcome,
        'submission_outcome': outcome, 'submissions': list(journal['submissions'])}


def watch(auth, *, once=False, dry_run=False):
    validate(auth)
    if not socket.gethostname().startswith('clustera-login-') or auth['user'] != getpass.getuser():
        raise ValueError('use this owner\'s authenticated ClusterA login session')
    root = Path(auth['state_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root/'scheduler.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = root/'submission_journal.json'
        journal = json.loads(path.read_text()) if path.exists() else {'config_sha256': auth['sha256'], 'submissions': []}
        while time.time() < auth['deadline_utc_epoch']:
            try:
                result = tick(auth, journal, path, dry_run=dry_run)
            except Exception as error:
                atomic_json(root/'error.json', {'time': time.time(), 'type': type(error).__name__, 'error': str(error)})
                raise
            atomic_json(root/'status.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result['status'] in {'complete', 'uncertain', 'uncertain_submission_requires_audit',
                    'accepted_job_missing_requires_audit', 'submission_budget_exhausted',
                    'insufficient_study_lifetime', 'bounded_rejections_exhausted'}:
                return result
            time.sleep(30)
    return {'status': 'deadline'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'watch', 'once', 'inspect'))
    parser.add_argument('--config', type=Path, default=frozen.CLUSTERA_CONFIG)
    args = parser.parse_args(argv)
    auth = prepare(args.config)
    if args.phase == 'prepare':
        result = {'authorization': str(Path(auth['state_root'])/'authorization.json'), 'sha256': auth['sha256']}
    else:
        result = watch(auth, once=args.phase=='once', dry_run=args.phase=='inspect')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
