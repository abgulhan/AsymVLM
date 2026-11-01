"""Idle-only attachment for the separately authorized eight-hour diagnostic.

No allocation submission, cancellation, or alteration of the completed parent.
Old scientific files stay pinned. A verified completion receipt, not a fabricated
heartbeat, allows work after the parent campaign has actually finished.
"""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import time

from . import coverage_audit_operations as old
from . import sprint48
from .coverage_audit_resume import ROOT, PARENT, checked
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest


WORKER = 'scripts/mlsys_coverage_audit_resume_idle_worker.sh'
PREFIX = old.PREFIX + 'resume-164134-'
_ORIGINAL_PRIORITY = old.main_has_priority
CONFIG_NAME = 'config-v2.json'  # Preserve the unused pre-launch v1 config.


def setup(project):
    project = Path(project).resolve()
    root = project / ROOT
    audit = checked(root / 'audit.json')
    path = root / 'operations' / CONFIG_NAME
    if path.exists():
        return read_config(path)
    parent = project / PARENT
    proof = [parent / 'study.json', parent / 'postprocessing-recovery.json',
             parent / 'paper-tables/report.json',
             *[parent / f'systems-report-accounting/{model}/report.json'
               for model in ('qwen25', 'llava_next', 'phi35')],
             *[parent / f'qwen25/text/reports/sources{n}/report.json' for n in (64, 128)]]
    sources = [Path(__file__).resolve(), project / WORKER,
               project / 'learned_score/mlsys_sprint/coverage_audit_resume_report.py',
               root / 'audit.json', root / 'resume-receipt.json', root / 'cpu-reuse.json']
    pins = {**audit['input_hashes'], **{str(p): file_digest(p) for p in [*proof, *sources]}}
    value = {'format': sprint48.FORMAT, 'cwd': str(project), 'user': getpass.getuser(),
             'created_utc_epoch': audit['created_utc_epoch'],
             'deadline_utc_epoch': audit['physical_deadline_utc_epoch'],
             'maximum_submissions': 1, 'allocation_submission_disabled': True,
             'coordinator_role': 'clusterb', 'maximum_idle_gpus': 2,
             'state_root': str(root / 'operations/state-v2'), 'pbs_script': WORKER,
             'plan_paths': list(audit['plans'].values()), 'input_hashes': pins,
             'audit_manifest': str(root / 'audit.json'),
             'main_status': str(parent / 'coordinator/clusterb/status.json'),
             'parent_completion_receipt': str(parent / 'postprocessing-recovery.json'),
             'parent_study': str(parent / 'study.json'),
             'parent_proof_paths': [str(p) for p in proof],
             'stop_new_physical_utc_epoch': audit['stop_new_physical_utc_epoch']}
    verify_parent_complete(value)
    write_immutable(path, {**value, 'sha256': digest(value)})
    return read_config(path)


def read_config(path):
    config = sprint48.read_config(path)
    audit = checked(config['audit_manifest'])
    if (config.get('allocation_submission_disabled') is not True
            or config.get('maximum_idle_gpus') != 2
            or config.get('user') != getpass.getuser()
            or config['plan_paths'] != list(audit['plans'].values())
            or config['deadline_utc_epoch'] != audit['physical_deadline_utc_epoch']
            or config['stop_new_physical_utc_epoch'] != audit['stop_new_physical_utc_epoch']
            or config['pbs_script'] != WORKER):
        raise ValueError('resume operations identity or limits changed')
    verify_parent_complete(config)
    return config


def verify_parent_complete(config, *, now=None):
    now = time.time() if now is None else now
    paths = config['parent_proof_paths']
    if not paths or config['parent_completion_receipt'] not in paths or config['parent_study'] not in paths:
        raise ValueError('parent completion proof missing')
    for name in paths:
        if config['input_hashes'].get(name) != file_digest(name):
            raise ValueError('parent completion proof changed: ' + name)
    receipt = old.read(config['parent_completion_receipt'])
    study = old.read(config['parent_study'])
    if (receipt.get('complete') is not True or receipt.get('gpu_tasks') !=
            {'complete': 679, 'failed': 0, 'pending': 0, 'running': 0}
            or not isinstance(study.get('deadline_utc_epoch'), (int, float))
            or study['deadline_utc_epoch'] >= now):
        raise ValueError('parent is not verifiably complete and past its deadline')
    return True


def main_has_priority(config, *, now=None):
    now = time.time() if now is None else now
    path = Path(config['main_status'])
    try:
        state = old.read(path) if path.exists() else {}
        stamp = state.get('time')
        if isinstance(stamp, (int, float)) and stamp > now:
            return True, 'main_status_from_future'
        if isinstance(stamp, (int, float)) and 0 <= now - stamp <= 90:
            return _ORIGINAL_PRIORITY(config, now=now)
        verify_parent_complete(config, now=now)
    except (ValueError, OSError, KeyError, TypeError):
        return True, 'parent_completion_not_verified'
    return False, 'parent_completed_receipt_verified'


def gpu_is_empty(gpu):
    apps = old.clusterb.command(['nvidia-smi', '-i', str(gpu),
                              '--query-compute-apps=pid', '--format=csv,noheader,nounits'])
    inventory = old.clusterb.command(['nvidia-smi', '-i', str(gpu),
                                   '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'])
    return not apps.strip() and [s.strip() for s in inventory.strip().split(',')] == ['0', '0']


@contextmanager
def worker_slot(root):
    """Two shared-filesystem leases also cap manually started resume workers."""
    folder = Path(root) / 'operations'
    folder.mkdir(parents=True, exist_ok=True)
    selected = None
    for index in range(2):
        fd = os.open(folder / f'worker-slot-{index}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            os.close(fd)
            raise ValueError('invalid worker slot lease')
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(fd)
            continue
        selected = fd
        os.set_inheritable(fd, True)
        break
    if selected is None:
        raise ValueError('two audit workers already active')
    try:
        yield selected
    finally:
        os.close(selected)


@contextmanager
def inherited_child_leases(fds):
    """Keep GPU/worker slots held if the launcher dies before its child.

    The existing run_claim explicitly passes only its task lock. Add the two
    resource locks locally without editing that frozen implementation. This
    single-threaded launcher binds Popen only for the duration of one claim.
    """
    previous = old.overnight.subprocess.Popen
    for fd in fds:
        os.fstat(fd)
    def launch(*args, **kwargs):
        kwargs['pass_fds'] = tuple(sorted(set(kwargs.get('pass_fds', ())) | set(fds)))
        return previous(*args, **kwargs)
    old.overnight.subprocess.Popen = launch
    try:
        yield
    finally:
        old.overnight.subprocess.Popen = previous


def idle_worker(config, job, gpu, seconds):
    from .coverage_audit_resume_report import report
    host = socket.gethostname().split('.')[0]
    if (config['user'] != getpass.getuser() or type(seconds) is not int or not 600 <= seconds <= 1800
            or not old.clusterb.HOST_PATTERN.fullmatch(host) or type(gpu) is not int or gpu not in range(8)
            or not old.clusterb.JOB_PATTERN.fullmatch(job)):
        raise ValueError('explicit valid ClusterB job/node/GPU required')
    owned = old.clusterb.owned_allocations(old.local_query(config['user']), config['user'],
                                        time.time(), config['deadline_utc_epoch'])
    allocation = next((a for a in owned if a['job_id'] == job and a['host'] == host), None)
    if allocation is None:
        raise ValueError('not an owned live interactive allocation')
    root = Path(config['audit_manifest']).parent
    with worker_slot(root) as slot_fd:
        gpu_fd = os.open(f'/tmp/kv_day3_{host}_gpu{gpu}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            info = os.fstat(gpu_fd)
            if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
                raise ValueError('invalid cross-workflow GPU lease')
            fcntl.flock(gpu_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.set_inheritable(gpu_fd, True)
            if not gpu_is_empty(gpu):
                raise ValueError('GPU is occupied')
            os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PBS_JOBID=job)
            end = time.monotonic() + min(seconds, allocation['seconds'], config['deadline_utc_epoch'] - time.time())
            while end - time.monotonic() >= 600 and time.time() < config['stop_new_physical_utc_epoch']:
                busy, reason = main_has_priority(config)
                if busy or not gpu_is_empty(gpu):
                    print(json.dumps({'event': 'audit_yield', 'reason': reason if busy else 'gpu_became_occupied'}), flush=True)
                    break
                queues, _ = sprint48.load_queues(config, 'gpu')
                found = old.overnight.claim(queues, remaining_seconds=end - time.monotonic())
                if found is None:
                    break
                task, plan, state_root, state, lock = found
                guarded = {**task, 'argv': [str(Path(config['cwd']) / '.conda/asymvlm/bin/python'), '-B', '-u', '-m',
                           'learned_score.mlsys_sprint.sprint48', 'guard', '--mode', 'shared', '--', *task['argv']]}
                with inherited_child_leases((slot_fd, gpu_fd)):
                    old.overnight.run_claim((guarded, plan, state_root, state, lock), end,
                                           f'coverage-audit-resume-{job}-{host}-{gpu}', lambda: False)
                report(root)
        finally:
            os.close(gpu_fd)


def once(config, path):
    if socket.gethostname().split('.')[0] != 'clusterb-login-01':
        raise ValueError('attachment requires authenticated ClusterB login')
    busy, reason = main_has_priority(config)
    status = {'time': time.time(), 'status': 'waiting_for_spare_capacity',
              'reason': reason, 'launched': []}
    if time.time() >= config['stop_new_physical_utc_epoch']:
        return {**status, 'status': 'admission_closed'}
    if busy:
        return status
    sessions = old.clusterb.tmux_sessions()
    available = max(0, 2 - sum(s.startswith(old.PREFIX) for s in sessions))
    if not available:
        return {**status, 'status': 'two_audit_workers_active'}
    queues, missing = sprint48.load_queues(config, 'gpu')
    if sprint48.all_terminal(queues, missing):
        return {**status, 'status': 'audit_queue_terminal'}
    owned = old.clusterb.owned_allocations(old.local_query(config['user']), config['user'],
                                        time.time(), config['deadline_utc_epoch'])
    for allocation in sorted(owned, key=lambda a: a['host']):
        seconds = min(1800, int(allocation['seconds']), int(config['deadline_utc_epoch'] - time.time()))
        if seconds < 600:
            continue
        for gpu in old.direct_idle_gpus(allocation['host']):
            if not available:
                return status
            if old.gpu_session_present(sessions, allocation['host'], gpu):
                continue
            name = PREFIX + allocation['host'] + '-gpu' + str(gpu)
            remote = ['bash', str(Path(config['cwd']) / WORKER),
                      allocation['job_id'], str(gpu), str(seconds), str(path)]
            log = Path(config['audit_manifest']).parent / 'operations/logs' / (name + '-' + str(int(time.time())) + '.log')
            log.parent.mkdir(parents=True, exist_ok=True)
            command = ['ssh', *old.clusterb.SSH_OPTIONS, allocation['host'], shlex.join(remote)]
            old.clusterb.command(['tmux', 'new-session', '-d', '-s', name,
                               'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'])
            sessions.add(name)
            available -= 1
            status['launched'].append({'session': name, 'job': allocation['job_id'], 'gpu': gpu, 'log': str(log)})
    return status


def watch(config, path):
    from .coverage_audit_resume_report import report
    root = Path(config['audit_manifest']).parent
    with (root / 'operations/attach.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < config['deadline_utc_epoch']:
            try:
                config = read_config(path)
                result = once(config, path)
                report(root)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_spare_capacity', 'error': str(exc)}
            atomic_json(root / 'operations/status.json', result)
            print(json.dumps(result), flush=True)
            if result['status'] == 'audit_queue_terminal':
                return result
            time.sleep(min(30, max(0, config['deadline_utc_epoch'] - time.time())))
    result = {'time': time.time(), 'status': 'physical_deadline_reached'}
    atomic_json(root / 'operations/status.json', result)
    report(root)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('setup', 'once', 'watch', 'idle-worker'))
    parser.add_argument('--project', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path)
    parser.add_argument('--job')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    path = args.config or args.project / ROOT / 'operations' / CONFIG_NAME
    config = setup(args.project) if args.phase == 'setup' else read_config(path)
    if args.phase == 'idle-worker':
        return idle_worker(config, args.job, args.gpu, args.seconds)
    if args.phase == 'watch':
        result = watch(config, path)
    elif args.phase == 'once':
        result = once(config, path)
        atomic_json(Path(config['audit_manifest']).parent / 'operations/status.json', result)
    else:
        result = {'status': 'prepared', 'config': str(path)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
