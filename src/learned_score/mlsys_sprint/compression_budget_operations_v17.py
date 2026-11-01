"""Bounded two-worker ClusterA execution for the v17 budget experiment.

Only additive configuration and operational state are written. Scientific plans
retain the existing immutable catalog, task locks, atomic output checks, and one
unexpected-failure retry. Submission uses the established durable nonce journal;
ambiguous acknowledgements never cause an automatic duplicate qsub.
"""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from . import overnight, sprint48, sprint48_scheduler_v3 as submissions
from . import clustera_queue_policy_v5 as queue_policy
from . import coverage_audit_resume_operations as leases
from . import coverage_audit_resume_clustera_worker as allocation_checks
from .compression_matrix_v8 import checked, seal
from .compression_operations_v16 import CUDNN_DIRECTORY, CUDNN_HASHES, _library_links
from .compression_clustera_v11a import normalize_inventory
from .followup_common import write_immutable
from .state import atomic_json, file_digest
from .worker import verify_sources

ROOT = Path('reproduction_results/mlsys_budget_20261025_v17')
SHARED_LEASE_ROOT = Path('reproduction_results/mlsys_compression_20261025_v16/coverage_audit_resume_20261025_164134')
JOB_NAME = 'mlsys-budget-v17'
PBS_SCRIPT = 'scripts/mlsys_budget_v17_clustera.pbs'
WORKER_SCRIPT = 'scripts/mlsys_budget_v17_worker.sh'
PLAN_NAMES = tuple(f'features-{split}-{model}' for model in ('qwen25', 'llava_next', 'phi35')
                   for split in ('fit', 'screening')) + (
    'features-extra-fit-phi35', 'features-extra-screening-phi35',
    'physical-qwen25', 'physical-llava_next', 'physical-phi35')
TIME_KEYS = ('created_utc_epoch', 'stop_new_work_utc_epoch',
             'physical_deadline_utc_epoch', 'deadline_utc_epoch')


def source_paths(project):
    names = ('learned_score/mlsys_sprint/compression_budget_operations_v17.py',
             'learned_score/mlsys_sprint/coverage_audit_resume_operations.py',
             'learned_score/mlsys_sprint/coverage_audit_resume_clustera_worker.py',
             'learned_score/mlsys_sprint/coverage_audit_operations.py',
             'learned_score/mlsys_sprint/cache_dedicated_clusterb_v4.py',
             'learned_score/mlsys_sprint/adaptive_operations_v4.py',
             'learned_score/mlsys_sprint/compression_operations_v16.py',
             'learned_score/mlsys_sprint/compression_clustera_v11a.py',
             'learned_score/mlsys_sprint/clustera_queue_policy_v5.py',
             'learned_score/mlsys_sprint/sprint48_scheduler_v3.py',
             'learned_score/mlsys_sprint/sprint48.py',
             'learned_score/mlsys_sprint/overnight.py',
             'learned_score/mlsys_sprint/worker.py',
             'learned_score/mlsys_sprint/state.py',
             'learned_score/mlsys_sprint/compression_matrix_v8.py',
             'learned_score/mlsys_sprint/followup_common.py',
             'activate_env.sh', PBS_SCRIPT, WORKER_SCRIPT)
    return [Path(project) / name for name in names] + [
        Path(project) / CUDNN_DIRECTORY / name for name in CUDNN_HASHES]


def validate_protocol(path):
    protocol = checked(path)
    project = Path(protocol['cwd']).resolve()
    root = Path(protocol['output_root']).resolve()
    times = [protocol.get(k) for k in TIME_KEYS]
    if (root != project / ROOT or Path(path).resolve() != root / 'protocol.json'
            or protocol.get('format') != 'mlsys_budget_v17_protocol'
            or protocol.get('maximum_gpu_workers') != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or not times[0] < times[1] < times[2] < times[3]
            or times[-1] - times[0] > 2 * 3600 + 60
            or times[2] - times[1] != 600 or times[3] - times[2] != 300):
        raise ValueError('v17 identity or hard deadline ordering changed')
    expected = {root / 'plans' / name / 'plan.json' for name in PLAN_NAMES}
    paths = [Path(p).resolve() for p in protocol['plan_paths']]
    if len(paths) != len(expected) or set(paths) != expected:
        raise ValueError('v17 requires exactly the eight feature and three physical plan paths')
    return protocol


def build_config(protocol_path):
    protocol = validate_protocol(protocol_path)
    project, root = Path(protocol['cwd']).resolve(), Path(protocol['output_root']).resolve()
    path = root / 'operations/config.json'
    if path.exists():
        return read_config(path)
    _library_links(project)
    pins = {**protocol.get('input_hashes', {}), str(Path(protocol_path).resolve()): file_digest(protocol_path),
            **{str(p.resolve()): file_digest(p) for p in source_paths(project)}}
    config = seal({'format': sprint48.FORMAT, 'study_operations': 'compression_budget_v17',
        'cwd': str(project), 'study_root': str(root), 'user': getpass.getuser(),
        'protocol_path': str(Path(protocol_path).resolve()), 'protocol_sha256': protocol['sha256'],
        'coordinator_role': 'clustera', 'created_utc_epoch': protocol['created_utc_epoch'],
        'deadline_utc_epoch': protocol['physical_deadline_utc_epoch'],
        'report_deadline_utc_epoch': protocol['deadline_utc_epoch'],
        'stop_new_work_utc_epoch': protocol['stop_new_work_utc_epoch'],
        'maximum_submissions': 3, 'maximum_nonterminal_debug_nodes': 2,
        'maximum_owned_clustera_nodes': 1, 'maximum_idle_gpus': 2,
        'shared_lease_root': str(project / SHARED_LEASE_ROOT),
        'state_root': str(root / 'operations/state'), 'pbs_script': PBS_SCRIPT,
        'plan_paths': protocol['plan_paths'], 'input_hashes': pins,
        'clustera_launch': {'nodes': 1, 'mpi_ranks': 2, 'ranks_per_node': 2,
                          'walltime_seconds': 3600, 'cpu_depth': 4, 'threads_per_worker': 2}})
    write_immutable(path, config)
    return read_config(path)


def read_config(path):
    config = sprint48.read_config(path)
    protocol = validate_protocol(config['protocol_path'])
    project, root = Path(protocol['cwd']).resolve(), Path(protocol['output_root']).resolve()
    expected = {'study_operations': 'compression_budget_v17', 'cwd': str(project),
        'study_root': str(root), 'user': getpass.getuser(), 'protocol_sha256': protocol['sha256'],
        'coordinator_role': 'clustera', 'created_utc_epoch': protocol['created_utc_epoch'],
        'deadline_utc_epoch': protocol['physical_deadline_utc_epoch'],
        'report_deadline_utc_epoch': protocol['deadline_utc_epoch'],
        'stop_new_work_utc_epoch': protocol['stop_new_work_utc_epoch'],
        'maximum_submissions': 3, 'maximum_nonterminal_debug_nodes': 2,
        'maximum_owned_clustera_nodes': 1, 'maximum_idle_gpus': 2,
        'shared_lease_root': str(project / SHARED_LEASE_ROOT),
        'state_root': str(root / 'operations/state'), 'pbs_script': PBS_SCRIPT,
        'plan_paths': protocol['plan_paths'],
        'clustera_launch': {'nodes': 1, 'mpi_ranks': 2, 'ranks_per_node': 2,
                          'walltime_seconds': 3600, 'cpu_depth': 4, 'threads_per_worker': 2}}
    if Path(path).resolve() != root / 'operations/config.json' or any(config.get(k) != v for k, v in expected.items()):
        raise ValueError('v17 operational identity, deadlines, or resource limits changed')
    required = {str(p.resolve()) for p in source_paths(project)} | {str(Path(config['protocol_path']).resolve())}
    if not required <= set(config['input_hashes']):
        raise ValueError('v17 operational dependency pins missing')
    if any(config['input_hashes'].get(p) != sha for p, sha in protocol.get('input_hashes', {}).items()):
        raise ValueError('protocol scientific input pins missing or changed')
    _library_links(project)
    for name, sha in CUDNN_HASHES.items():
        if config['input_hashes'][str((project / CUDNN_DIRECTORY / name).resolve())] != sha:
            raise ValueError('numerical library identity changed')
    return config


def terminal_marker(config):
    path = Path(config['study_root']) / 'pipeline-terminal.json'
    if not path.exists():
        return None
    value = checked(path)
    if (value.get('protocol_sha256') != config['protocol_sha256'] or
            value.get('status') not in {'complete', 'no_candidate', 'deadline_reached', 'failed'}):
        raise ValueError('pipeline terminal receipt is not bound to this protocol')
    return value


def admission(payload, queue_text, user):
    result = queue_policy.admission(normalize_inventory(payload), queue_text, user)
    reserved = sum(j['nodes'] for k in ('running_jobs', 'queued_jobs', 'other_jobs') for j in result[k])
    eligible = result['eligible'] and reserved + 1 <= 2
    return {**result, 'eligible': eligible, 'reserved_nodes_all_workflows': reserved,
            'reason': result['reason'] if eligible else 'waiting_for_verified_debug_capacity'}


def observe(config):
    jobs = submissions.execute(['qstat', '-x', '-f', '-F', 'json'], timeout=45)
    queue = submissions.execute(['qstat', '-Qf', 'debug'], timeout=30)
    if jobs['returncode'] != 0 or queue['returncode'] != 0:
        raise ValueError('cannot verify debug ownership, inventory, or queue limits')
    payload = json.loads(jobs['stdout'])
    return payload, admission(payload, queue['stdout'], config['user'])


def readiness(config):
    queues, missing = sprint48.load_queues(config, 'gpu')
    remaining = min(3300, config['deadline_utc_epoch'] - time.time() - 120)
    claim = overnight.claim(queues, dry_run=True, remaining_seconds=max(0, remaining))
    if claim:
        claim[-1].close()
    return bool(claim), sprint48.all_terminal(queues, missing)


def tick(config, journal, journal_path, *, dry_run=False):
    now = time.time()
    if now >= config['stop_new_work_utc_epoch']:
        return {'time': now, 'status': 'admission_closed'}
    if terminal_marker(config):
        return {'time': now, 'status': 'pipeline_terminal'}
    payload, decision = observe(config)
    result = {'time': now, 'admission': decision, 'status': 'waiting_for_ready_work_or_debug_queue'}
    if journal.get('submission_intent'):
        matches = submissions.nonce_matches(payload, config, journal['submission_intent'])
        if not matches:
            return {**result, 'status': 'uncertain_submission_requires_audit'}
        if dry_run:
            return {**result, 'status': 'nonce_reconciliation_required', 'job': matches[0]}
        submissions.append_accepted(journal, matches[0], reconciled=True)
        atomic_json(journal_path, journal)
    for accepted in journal['submissions']:
        job = payload['Jobs'].get(accepted['job'])
        if job is None:
            return {**result, 'status': 'accepted_job_missing_requires_audit'}
        if job.get('Job_Owner', '').split('@')[0] != config['user'] or job.get('Job_Name') != JOB_NAME:
            raise ValueError('accepted job identity changed')
        if job.get('job_state') not in submissions.TERMINAL:
            return {**result, 'status': 'own_allocation_outstanding', 'job': accepted['job']}
    ready, terminal = readiness(config)
    result.update(ready_tasks=ready, all_published_plans_terminal=terminal)
    if terminal:
        return {**result, 'status': 'complete'}
    if len(journal['submissions']) >= 3 or journal.get('total_rejections_v2', 0) >= submissions.MAX_REJECTIONS:
        return {**result, 'status': 'submission_budget_exhausted'}
    if config['deadline_utc_epoch'] - now < 600:
        return {**result, 'status': 'insufficient_study_lifetime'}
    if not ready or not decision['eligible'] or now < journal.get('retry_after_epoch', 0):
        return result
    if dry_run:
        return {**result, 'status': 'eligible_not_submitted'}
    _, fresh = observe(config)
    if not fresh['eligible']:
        return {**result, 'admission': fresh}
    return {**result, 'status': submissions.submit(config, journal, journal_path),
            'submissions': journal['submissions']}


def watch(path, *, once=False, dry_run=False):
    if not socket.gethostname().startswith('clustera-login-'):
        raise ValueError('use the authenticated clustera-mlsys login session')
    config = read_config(path); state = Path(config['state_root']); state.mkdir(parents=True, exist_ok=True)
    with (state / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / 'submission_journal.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            'config_sha256': config['sha256'], 'submissions': []}
        if journal.get('config_sha256') != config['sha256']:
            raise ValueError('submission journal belongs to another protocol')
        while time.time() < config['deadline_utc_epoch']:
            try:
                result = tick(read_config(path), journal, journal_path, dry_run=dry_run)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_state', 'error': repr(exc)}
            atomic_json(state / 'status.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result['status'] in {'complete', 'pipeline_terminal', 'admission_closed',
                    'uncertain', 'uncertain_submission_requires_audit', 'accepted_job_missing_requires_audit',
                    'submission_budget_exhausted', 'insufficient_study_lifetime'}:
                return result
            time.sleep(min(30, max(0, config['deadline_utc_epoch'] - time.time())))
    return {'status': 'physical_deadline_reached'}


@contextmanager
def allocation_binding():
    previous = allocation_checks.JOB_NAME
    allocation_checks.JOB_NAME = JOB_NAME
    try:
        yield
    finally:
        allocation_checks.JOB_NAME = previous


def run_claims(config, allocation, seconds, slot_fd, gpu_fd, stopping):
    end = time.monotonic() + min(seconds, allocation['seconds'],
                                 max(0, config['deadline_utc_epoch'] - time.time() - 20))
    completed = 0
    while not stopping() and end - time.monotonic() >= 180:
        if time.time() >= config['stop_new_work_utc_epoch'] or terminal_marker(config):
            return {'status': 'admission_closed_or_pipeline_terminal', 'claims_run': completed}
        if not leases.gpu_is_empty(allocation['gpu']):
            return {'status': 'yielded_to_occupied_gpu', 'claims_run': completed}
        queues, missing = sprint48.load_queues(config, 'gpu')
        queues = sorted(queues, key=lambda item: 'physical-' not in str(item[0]['plan']))
        selected = overnight.claim(queues, remaining_seconds=end - time.monotonic())
        if selected is None:
            if sprint48.all_terminal(queues, missing):
                return {'status': 'queue_terminal', 'claims_run': completed}
            # The CPU coordinator may still be fitting/publishing physical plans.
            time.sleep(min(15, max(0, end - time.monotonic())))
            continue
        task, plan, state_root, state, lock = selected
        try:
            verify_sources(plan)
        except BaseException:
            lock.close()
            raise
        guarded = {**task, 'argv': [str(Path(config['cwd']) / '.conda/asymvlm/bin/python'),
            '-B', '-u', '-m', 'learned_score.mlsys_sprint.sprint48', 'guard', '--mode',
            'exclusive' if task.get('resource') == 'isolated_gpu' else 'shared', '--', *task['argv']]}
        with leases.inherited_child_leases((slot_fd, gpu_fd)):
            overnight.run_claim((guarded, plan, state_root, state, lock), end,
                                f'budget-v17-{allocation["job_id"]}-{allocation["gpu"]}', stopping)
        completed += 1
    return {'status': 'worker_deadline_or_signal', 'claims_run': completed}


def worker(config, seconds=3300):
    if type(seconds) is not int or not 180 <= seconds <= 3300:
        raise ValueError('bounded worker duration must be 180 to 3300 seconds')
    if time.time() >= config['stop_new_work_utc_epoch']:
        return {'status': 'admission_closed', 'claims_run': 0}
    with allocation_binding():
        allocation = allocation_checks.inspect_allocation(config)
    if allocation['seconds'] < 180:
        return {'status': 'insufficient_allocation_lifetime', 'claims_run': 0}
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with leases.worker_slot(config['shared_lease_root']) as slot_fd, allocation_checks.gpu_lease(
                socket.gethostname().split('.')[0], allocation['gpu']) as gpu_fd:
            return run_claims(config, allocation, seconds, slot_fd, gpu_fd, lambda: stop[0])
    except ValueError as exc:
        if str(exc) != 'two audit workers already active':
            raise
        return {'status': 'two_global_workers_already_active', 'claims_run': 0}
    except BlockingIOError:
        return {'status': 'gpu_lease_occupied', 'claims_run': 0}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('setup', 'watch', 'once', 'inspect', 'worker'))
    parser.add_argument('--protocol', type=Path, default=ROOT / 'protocol.json')
    parser.add_argument('--config', type=Path, default=ROOT / 'operations/config.json')
    parser.add_argument('--seconds', type=int, default=3300)
    args = parser.parse_args(argv)
    if args.phase == 'setup':
        result = build_config(args.protocol)
        result = {'status': 'prepared', 'config_sha256': result['sha256']}
    elif args.phase == 'worker':
        result = worker(read_config(args.config), args.seconds)
    else:
        result = watch(args.config, once=args.phase == 'once', dry_run=args.phase == 'inspect')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
