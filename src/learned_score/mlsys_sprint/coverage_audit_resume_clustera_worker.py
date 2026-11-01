"""Two-rank ClusterA execution of the unchanged coverage-audit resume plans.

This worker never submits, cancels, or edits allocations. The scheduler pins
these additive launch sources; the original base configuration still owns the
shared queue catalog, task retries, scientific pins, and campaign deadlines.
"""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import signal
import socket
import stat
import subprocess
import time

from . import coverage_audit_resume_operations as ops
from . import overnight, sprint48
from .cache_dedicated_clusterb_v4 import start_epoch
from .coverage_audit_resume import ROOT


JOB_NAME = 'mlsys-coverage-audit-resume'
JOB_PATTERN = re.compile(r'[0-9]+(?:\.[A-Za-z0-9.-]+)?')
HOST_PATTERN = re.compile(r'x[0-9]+[a-z0-9]+')
PBS_SCRIPT = 'scripts/mlsys_coverage_audit_resume_clustera.pbs'
WORKER_SCRIPT = 'scripts/mlsys_coverage_audit_resume_clustera_worker.sh'


def read_configs(base_path, launch_path):
    """Verify additive launch pins without changing the shared base catalog."""
    base = ops.read_config(base_path)
    launch = sprint48.read_config(launch_path)
    project = Path(base['cwd']).resolve()
    if (Path(base_path).resolve() != project / ROOT / 'operations/config-v2.json'
            or Path(launch_path).resolve() != project / ROOT / 'operations/clustera/config.json'
            or launch.get('base_config_path') != str(Path(base_path).resolve())
            or launch.get('base_config_sha256') != base['sha256']
            or launch.get('coordinator_role') != 'clustera'
            or launch.get('maximum_idle_gpus') != 2 or launch.get('pbs_script') != PBS_SCRIPT
            or any(launch.get(key) != base[key] for key in
                   ('cwd', 'user', 'plan_paths', 'deadline_utc_epoch', 'stop_new_physical_utc_epoch'))):
        raise ValueError('ClusterA launch identity differs from frozen base configuration')
    required = {str(Path(base_path).resolve()), str(Path(__file__).resolve()),
                str(project / PBS_SCRIPT), str(project / WORKER_SCRIPT),
                str(project / 'learned_score/mlsys_sprint/coverage_audit_resume_report_v2.py')}
    if not required <= set(launch['input_hashes']):
        raise ValueError('missing additive ClusterA launch source pins')
    return base


def validate_allocation(config, payload, environment, host, *, now):
    """Require this owner's running one-node debug job and exactly two ranks."""
    job_id = environment.get('PBS_JOBID', '')
    rank = environment.get('PMI_RANK', '')
    if (config['user'] != getpass.getuser() or not JOB_PATTERN.fullmatch(job_id)
            or not HOST_PATTERN.fullmatch(host)
            or environment.get('PBS_JOBNAME') != JOB_NAME
            or rank not in ('0', '1') or environment.get('PMI_SIZE') != '2'
            or environment.get('CUDA_VISIBLE_DEVICES') != rank):
        raise ValueError('owned one-node ClusterA allocation with ranks/GPU 0 and 1 required')
    jobs = payload.get('Jobs')
    if not isinstance(jobs, dict) or len(jobs) != 1:
        raise ValueError('one explicitly queried PBS job is required')
    actual_id, job = next(iter(jobs.items()))
    if (not JOB_PATTERN.fullmatch(actual_id) or actual_id.split('.')[0] != job_id.split('.')[0]
            or job.get('Job_Owner', '').split('@')[0] != config['user']
            or job.get('Job_Name') != JOB_NAME or job.get('queue', '').split('@')[0] != 'debug'
            or job.get('job_state') != 'R'):
        raise ValueError('PBS ownership, name, queue, or running identity mismatch')
    hosts = {item.split('/')[0].split('.')[0] for item in job.get('exec_host', '').split('+') if item}
    resources = job.get('Resource_List', {})
    selection = str(resources.get('select', '')).split(':')[0]
    if hosts != {host} or str(resources.get('nodect')) != '1' or selection != '1':
        raise ValueError('job must contain exactly this one execution node')
    wall = sprint48.hms_seconds(resources.get('walltime'))
    # PBS may omit resources_used immediately after transition to R. Only that
    # whole-field absence permits zero; malformed present accounting fails.
    # The independent stime+wall bound still limits the resulting lifetime.
    used = (0 if 'resources_used' not in job else
            sprint48.hms_seconds(job['resources_used'].get('walltime')))
    started = start_epoch(job.get('stime'))
    if not 0 < wall <= 3600 or not 0 <= used <= wall or not 0 < started <= now + 60:
        raise ValueError('invalid debug-job lifetime')
    remaining = min(wall - used - 120, started + wall - now - 120,
                    config['deadline_utc_epoch'] - now)
    return {'job_id': job_id, 'gpu': int(rank), 'seconds': max(0, int(remaining))}


def inspect_allocation(config):
    job_id = os.environ.get('PBS_JOBID', '')
    if not JOB_PATTERN.fullmatch(job_id):
        raise ValueError('explicit PBS job identity required')
    result = subprocess.run(['qstat', '-f', '-F', 'json', job_id], check=True,
                            capture_output=True, text=True, timeout=45)
    host = socket.gethostname().split('.')[0]
    allocation = validate_allocation(config, json.loads(result.stdout), os.environ,
                                     host, now=time.time())
    nodefile = os.environ.get('PBS_NODEFILE', '')
    if not nodefile or {line.strip().split('.')[0] for line in Path(nodefile).read_text().splitlines()
                        if line.strip()} != {host}:
        raise ValueError('PBS nodefile does not match the owned execution host')
    return allocation


@contextmanager
def gpu_lease(host, gpu):
    path = f'/tmp/kv_day3_{host}_gpu{gpu}.lock'
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(fd)
        if info.st_uid != os.getuid() or not stat.S_ISREG(info.st_mode):
            raise ValueError('invalid cross-workflow GPU lease')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.set_inheritable(fd, True)
        yield fd
    finally:
        os.close(fd)


def run_claims(config, allocation, seconds, slot_fd, gpu_fd, stopping):
    from .coverage_audit_resume_report_v2 import report
    root = Path(config['audit_manifest']).parent
    gpu = allocation['gpu']
    end = time.monotonic() + min(seconds, allocation['seconds'],
                                 max(0, config['deadline_utc_epoch'] - time.time()))
    completed = 0
    while not stopping() and end - time.monotonic() >= 600:
        if time.time() >= config['stop_new_physical_utc_epoch']:
            return {'status': 'admission_closed', 'claims_run': completed}
        busy, reason = ops.main_has_priority(config)
        if busy or not ops.gpu_is_empty(gpu):
            return {'status': 'yielded', 'reason': reason if busy else 'gpu_became_occupied',
                    'claims_run': completed}
        queues, _ = sprint48.load_queues(config, 'gpu')
        selected = overnight.claim(queues, remaining_seconds=end - time.monotonic())
        if selected is None:
            return {'status': 'no_claimable_work', 'claims_run': completed}
        task, plan, state_root, state, lock = selected
        guarded = {**task, 'argv': [str(Path(config['cwd']) / '.conda/asymvlm/bin/python'),
                   '-B', '-u', '-m', 'learned_score.mlsys_sprint.sprint48',
                   'guard', '--mode', 'shared', '--', *task['argv']]}
        with ops.inherited_child_leases((slot_fd, gpu_fd)):
            overnight.run_claim((guarded, plan, state_root, state, lock), end,
                                f'coverage-resume-clustera-{allocation["job_id"]}-{gpu}', stopping)
        completed += 1
        try:
            report(root)
        except (KeyError, TypeError) as exc:
            # Reporting field/schema bugs do not invalidate completed physical
            # rows. ValueError/source-integrity failures deliberately propagate.
            print(json.dumps({'event': 'report_error', 'type': type(exc).__name__,
                              'error': str(exc), 'claims_run': completed}), flush=True)
    return {'status': 'worker_deadline_or_signal', 'claims_run': completed}


def worker(config, seconds):
    if type(seconds) is not int or not 600 <= seconds <= 3300:
        raise ValueError('bounded worker duration must be 600 to 3300 seconds')
    if time.time() >= config['stop_new_physical_utc_epoch']:
        return {'status': 'admission_closed', 'claims_run': 0}
    allocation = inspect_allocation(config)
    if allocation['seconds'] < 600:
        return {'status': 'insufficient_allocation_lifetime', 'claims_run': 0}
    root = Path(config['audit_manifest']).parent
    host = socket.gethostname().split('.')[0]
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with ops.worker_slot(root) as slot_fd, gpu_lease(host, allocation['gpu']) as gpu_fd:
            return run_claims(config, allocation, seconds, slot_fd, gpu_fd, lambda: stop[0])
    except ValueError as exc:
        if str(exc) != 'two audit workers already active':
            raise
        return {'status': 'two_audit_workers_already_active', 'claims_run': 0}
    except BlockingIOError:
        return {'status': 'gpu_lease_occupied', 'claims_run': 0}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--launch-config', type=Path,
                        default=Path(__file__).resolve().parents[2] / ROOT / 'operations/clustera/config.json')
    parser.add_argument('--seconds', type=int, default=3300)
    args = parser.parse_args(argv)
    result = worker(read_configs(args.config, args.launch_config), args.seconds)
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
