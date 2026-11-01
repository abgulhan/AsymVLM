"""Four-hour, eight-GPU evidence pilot on owned ClusterA debug allocations.

New immutable plans may be published below the protocol's plans directory.
Every first-seen plan is pinned by the existing locked catalog, and existing
workers retain atomic completion markers and one unexpected-failure retry.
No allocation belonging to another workflow is modified or terminated.
"""
import argparse
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
from .compression_budget_operations_v17a import gpu_idle
from .compression_matrix_v8 import checked, seal
from .compression_operations_v16 import CUDNN_DIRECTORY, CUDNN_HASHES, _library_links
from .compression_clustera_v11a import normalize_inventory
from .cache_dedicated_clusterb_v4 import start_epoch
from .followup_common import write_immutable
from .state import atomic_json, file_digest
from .worker import verify_sources

ROOT = Path('reproduction_results/mlsys_evidence_20261025_v18')
JOB_NAME = 'mlsys-evidence-v18'
PBS_SCRIPT = 'scripts/mlsys_evidence_v18_clustera.pbs'
WORKER_SCRIPT = 'scripts/mlsys_evidence_v18_worker.sh'
TIME_KEYS = ('created_utc_epoch', 'stop_new_work_utc_epoch',
             'physical_deadline_utc_epoch', 'deadline_utc_epoch')


def source_paths(project):
    from .compression_budget_operations_v17 import source_paths as previous_sources
    return list(dict.fromkeys(previous_sources(project) + [
        Path(__file__).resolve(), Path(project) / PBS_SCRIPT, Path(project) / WORKER_SCRIPT,
        Path(project) / 'learned_score/mlsys_sprint/compression_budget_operations_v17a.py']))


def validate_protocol(path):
    value = checked(path)
    project, root = Path(value['cwd']).resolve(), Path(value['output_root']).resolve()
    times = [value.get(k) for k in TIME_KEYS]
    if (root != project / ROOT or Path(path).resolve() != root / 'protocol.json'
            or value.get('format') != 'mlsys_evidence_v18_protocol'
            or value.get('maximum_gpu_workers') != 8
            or any(type(t) not in (int, float) or not math.isfinite(t) for t in times)
            or not times[0] < times[1] < times[2] < times[3]
            or times[-1] - times[0] != 4 * 3600
            or times[1] - times[0] != 3 * 3600
            or times[-1] - times[2] != 600):
        raise ValueError('v18 protocol identity, eight-GPU bound, or four-hour deadlines changed')
    return value


def configuration_fields(protocol):
    project, root = Path(protocol['cwd']).resolve(), Path(protocol['output_root']).resolve()
    return dict(format=sprint48.FORMAT, study_operations='evidence_v18', cwd=str(project),
        study_root=str(root), user=getpass.getuser(), coordinator_role='clustera',
        protocol_path=str(root / 'protocol.json'), protocol_sha256=protocol['sha256'],
        created_utc_epoch=protocol['created_utc_epoch'],
        deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],
        report_deadline_utc_epoch=protocol['deadline_utc_epoch'],
        stop_new_work_utc_epoch=protocol['stop_new_work_utc_epoch'],
        maximum_submissions=12, maximum_nonterminal_debug_nodes=2,
        maximum_owned_clustera_nodes=2, maximum_idle_gpus=8,
        state_root=str(root / 'operations/state'), pbs_script=PBS_SCRIPT,
        plan_paths=[], plan_directory=str(root / 'plans'),
        clustera_launch=dict(nodes=2, mpi_ranks=8, ranks_per_node=4,
            walltime_seconds=3600, cpu_depth=4, threads_per_worker=2))


def build_config(protocol_path):
    protocol = validate_protocol(protocol_path)
    project, root = Path(protocol['cwd']), Path(protocol['output_root'])
    path = root / 'operations/config.json'
    if path.exists():
        return read_config(path)
    _library_links(project)
    pins = {**protocol.get('input_hashes', {}),
            str(Path(protocol_path).resolve()): file_digest(protocol_path),
            **{str(p.resolve()): file_digest(p) for p in source_paths(project)}}
    write_immutable(path, seal({**configuration_fields(protocol), 'input_hashes': pins}))
    return read_config(path)


def read_config(path):
    value = sprint48.read_config(path)
    protocol = validate_protocol(value['protocol_path'])
    if (Path(path).resolve() != Path(protocol['output_root']) / 'operations/config.json'
            or any(value.get(k) != v for k, v in configuration_fields(protocol).items())):
        raise ValueError('v18 configuration identity or safety limits changed')
    required = {str(p.resolve()): file_digest(p) for p in source_paths(value['cwd'])}
    required.update(protocol.get('input_hashes', {}))
    required[str(Path(value['protocol_path']).resolve())] = file_digest(value['protocol_path'])
    if any(value['input_hashes'].get(p) != sha for p, sha in required.items()):
        raise ValueError('v18 required immutable dependencies missing or changed')
    return value


def load_queues(config):
    """Discover only direct plan children; preserve removed-plan detection."""
    directory = Path(config['plan_directory'])
    if directory.is_symlink():
        raise ValueError('plan directory must not be a symbolic link')
    discovered = set()
    for path in directory.glob('*/plan.json'):
        if path.is_symlink() or path.parent.is_symlink() or path.resolve().parent.parent != directory.resolve():
            raise ValueError('plan publication escapes fixed directory')
        discovered.add(str(path.resolve()))
    catalog = Path(config['state_root']) / 'catalog.json'
    if catalog.exists():
        ledger = json.loads(catalog.read_text())
        for path in ledger.get('plans', {}):
            if Path(path).resolve().parent.parent != directory.resolve():
                raise ValueError('catalog path escapes fixed directory')
            discovered.add(path)
    queues, missing = sprint48.load_queues({**config, 'plan_paths': sorted(discovered)}, 'gpu')
    for _, plan, _, _ in queues:
        if plan.get('unexpected_retries', 1) != 1:
            raise ValueError('v18 plans require exactly one unexpected-failure retry')
    # A completed stage is not a completed campaign: the CPU pipeline can
    # publish later stages. Only a bound terminal receipt closes the campaign.
    return queues, missing


def terminal_marker(config):
    path = Path(config['study_root']) / 'pipeline-terminal.json'
    if not path.exists():
        return None
    value = checked(path)
    if (value.get('protocol_sha256') != config['protocol_sha256'] or
            value.get('status') not in {'complete', 'no_candidate', 'deadline_reached', 'failed'}):
        raise ValueError('terminal receipt is not bound to v18 protocol')
    return value


def admission(payload, queue_text, user):
    value = queue_policy.admission(normalize_inventory(payload), queue_text, user)
    reserved = sum(j['nodes'] for k in ('running_jobs', 'queued_jobs', 'other_jobs') for j in value[k])
    eligible = value['eligible'] and reserved + 2 <= 2
    return {**value, 'eligible': eligible, 'reserved_nodes_all_workflows': reserved,
            'reason': value['reason'] if eligible else 'waiting_for_verified_two_node_debug_capacity'}


def observe(config):
    jobs = submissions.execute(['qstat', '-x', '-f', '-F', 'json'], timeout=45)
    queue = submissions.execute(['qstat', '-Qf', 'debug'], timeout=30)
    if jobs['returncode'] != 0 or queue['returncode'] != 0:
        raise ValueError('cannot verify current user debug inventory or queue policy')
    payload = json.loads(jobs['stdout'])
    return payload, admission(payload, queue['stdout'], config['user'])


def readiness(config):
    queues, _ = load_queues(config)
    remaining = min(3300, config['deadline_utc_epoch'] - time.time() - 120)
    claim = overnight.claim(queues, dry_run=True, remaining_seconds=max(0, remaining))
    if claim:
        claim[-1].close()
    return bool(claim)


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
            raise ValueError('accepted job ownership or name changed')
        if job.get('job_state') not in submissions.TERMINAL:
            return {**result, 'status': 'own_allocation_outstanding', 'job': accepted['job']}
    ready = readiness(config)
    result['ready_tasks'] = ready
    if len(journal['submissions']) >= config['maximum_submissions']:
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
        raise ValueError('use authenticated clustera-mlsys session')
    config = read_config(path)
    state = Path(config['state_root']); state.mkdir(parents=True, exist_ok=True)
    with (state / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / 'submission_journal.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            'config_sha256': config['sha256'], 'submissions': []}
        if journal.get('config_sha256') != config['sha256']:
            raise ValueError('submission journal belongs to another configuration')
        while time.time() < config['deadline_utc_epoch']:
            try:
                result = tick(read_config(path), journal, journal_path, dry_run=dry_run)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_state', 'error': repr(exc)}
            atomic_json(state / 'status.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result['status'] in {
                    'pipeline_terminal', 'admission_closed', 'uncertain',
                    'uncertain_submission_requires_audit', 'accepted_job_missing_requires_audit',
                    'submission_budget_exhausted', 'insufficient_study_lifetime'}:
                return result
            time.sleep(min(30, max(0, config['deadline_utc_epoch'] - time.time())))
    return {'status': 'physical_deadline_reached'}


def validate_allocation(config, payload, environment, host, *, now):
    job_id = environment.get('PBS_JOBID', '')
    rank = environment.get('PMI_RANK', '')
    local_rank = str(int(rank) % 4) if rank in tuple(str(i) for i in range(8)) else None
    if (config['user'] != getpass.getuser() or not allocation_checks.JOB_PATTERN.fullmatch(job_id)
            or not allocation_checks.HOST_PATTERN.fullmatch(host)
            or environment.get('PBS_JOBNAME') != JOB_NAME
            or local_rank is None or environment.get('PMI_SIZE') != '8'
            or environment.get('CUDA_VISIBLE_DEVICES') != local_rank):
        raise ValueError('owned two-node ClusterA allocation with eight ranks required')
    jobs = payload.get('Jobs')
    if not isinstance(jobs, dict) or len(jobs) != 1:
        raise ValueError('exactly one explicitly queried PBS job required')
    actual_id, job = next(iter(jobs.items()))
    if (actual_id.split('.')[0] != job_id.split('.')[0]
            or job.get('Job_Owner', '').split('@')[0] != config['user']
            or job.get('Job_Name') != JOB_NAME or job.get('queue', '').split('@')[0] != 'debug'
            or job.get('job_state') != 'R'):
        raise ValueError('PBS allocation ownership, queue, or running identity mismatch')
    hosts = {item.split('/')[0].split('.')[0] for item in job.get('exec_host', '').split('+') if item}
    resources = job.get('Resource_List', {})
    if (len(hosts) != 2 or host not in hosts or str(resources.get('nodect')) != '2'
            or str(resources.get('select', '')).split(':')[0] != '2'):
        raise ValueError('job must contain exactly two execution nodes including this host')
    wall = sprint48.hms_seconds(resources.get('walltime'))
    used = 0 if 'resources_used' not in job else sprint48.hms_seconds(job['resources_used'].get('walltime'))
    started = start_epoch(job.get('stime'))
    if not 0 < wall <= 3600 or not 0 <= used <= wall or not 0 < started <= now + 60:
        raise ValueError('invalid debug allocation lifetime')
    remaining = min(wall - used - 120, started + wall - now - 120,
                    config['deadline_utc_epoch'] - now - 20)
    return {'job_id': job_id, 'gpu': int(local_rank), 'seconds': max(0, int(remaining)), 'hosts': hosts}


def inspect_allocation(config):
    job_id = os.environ.get('PBS_JOBID', '')
    if not allocation_checks.JOB_PATTERN.fullmatch(job_id):
        raise ValueError('explicit PBS identity required')
    result = subprocess.run(['qstat', '-f', '-F', 'json', job_id], check=True,
                            capture_output=True, text=True, timeout=45)
    allocation = validate_allocation(config, json.loads(result.stdout), os.environ,
                                    socket.gethostname().split('.')[0], now=time.time())
    nodefile = os.environ.get('PBS_NODEFILE', '')
    if not nodefile or {line.strip().split('.')[0] for line in Path(nodefile).read_text().splitlines()
                        if line.strip()} != allocation['hosts']:
        raise ValueError('PBS nodefile differs from verified allocated hosts')
    return allocation


def run_claims(config, allocation, seconds, gpu_fd, stopping):
    end = time.monotonic() + min(seconds, allocation['seconds'],
                                max(0, config['deadline_utc_epoch'] - time.time() - 20))
    completed = 0
    while not stopping() and end - time.monotonic() >= 180:
        if time.time() >= config['stop_new_work_utc_epoch'] or terminal_marker(config):
            return {'status': 'admission_closed_or_pipeline_terminal', 'claims_run': completed}
        if not gpu_idle(allocation['gpu'], allow_cooldown=completed > 0):
            return {'status': 'yielded_to_occupied_gpu', 'claims_run': completed}
        queues, _ = load_queues(config)
        selected = overnight.claim(queues, remaining_seconds=end - time.monotonic())
        if selected is None:
            time.sleep(min(15, max(0, end - time.monotonic())))
            continue
        task, plan, state_root, state, lock = selected
        try:
            verify_sources(plan)
        except BaseException:
            lock.close(); raise
        guarded = {**task, 'argv': [str(Path(config['cwd']) / '.conda/asymvlm/bin/python'),
            '-B', '-u', '-m', 'learned_score.mlsys_sprint.sprint48', 'guard', '--mode',
            'exclusive' if task.get('resource') == 'isolated_gpu' else 'shared', '--', *task['argv']]}
        with leases.inherited_child_leases((gpu_fd,)):
            overnight.run_claim((guarded, plan, state_root, state, lock), end,
                f'evidence-v18-{allocation["job_id"]}-{socket.gethostname()}-{allocation["gpu"]}', stopping)
        completed += 1
    return {'status': 'worker_deadline_or_signal', 'claims_run': completed}


def worker(config, seconds=3300):
    if type(seconds) is not int or not 180 <= seconds <= 3300:
        raise ValueError('worker duration must be 180 to 3300 seconds')
    if time.time() >= config['stop_new_work_utc_epoch']:
        return {'status': 'admission_closed', 'claims_run': 0}
    allocation = inspect_allocation(config)
    if allocation['seconds'] < 180:
        return {'status': 'insufficient_allocation_lifetime', 'claims_run': 0}
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with allocation_checks.gpu_lease(socket.gethostname().split('.')[0], allocation['gpu']) as gpu_fd:
            return run_claims(config, allocation, seconds, gpu_fd, lambda: stop[0])
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
        result = {'status': 'prepared', 'config_sha256': build_config(args.protocol)['sha256']}
    elif args.phase == 'worker':
        result = worker(read_config(args.config), args.seconds)
    else:
        result = watch(args.config, once=args.phase == 'once', dry_run=args.phase == 'inspect')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
