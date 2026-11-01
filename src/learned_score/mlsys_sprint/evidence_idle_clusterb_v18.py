"""Idle-only overflow into explicitly verified existing ClusterB job198677.

This additive wrapper does not submit/cancel jobs or alter frozen configurations.
The ClusterA watcher must be stopped and its queued allocation terminal before
launch. A pinned handoff receipt enforces that operator-verified transition.
"""
import argparse
import getpass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from . import evidence_operations_v18 as base
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import file_digest

JOB = '1000000.clusterb-pbs-01.example.org'
HOSTS = {'clusterb-gpu-01', 'clusterb-gpu-02'}
SCRIPT = 'scripts/mlsys_evidence_v18_idle_clusterb.sh'


def validate_allocation(config, payload, host, gpu, now):
    job = payload.get('Jobs', {}).get(JOB)
    if (host not in HOSTS or type(gpu) is not int or gpu not in range(8)
            or not isinstance(job, dict) or job.get('Job_Name') != 'kv-adaptive'
            or job.get('Job_Owner', '').split('@')[0] != getpass.getuser()
            or job.get('job_state') != 'R' or job.get('queue') != 'by-node'
            or str(job.get('interactive', '')).lower() not in {'true', '1'}):
        raise ValueError('existing owned ClusterB allocation identity mismatch')
    hosts = {item.split('/')[0].split('.')[0] for item in job.get('exec_host', '').split('+') if item}
    resource = job.get('Resource_List', {})
    if hosts != HOSTS or str(resource.get('nodect')) != '2' or str(resource.get('ngpus')) != '16':
        raise ValueError('existing ClusterB execution nodes or GPU count changed')
    wall = base.sprint48.hms_seconds(resource.get('walltime'))
    used = base.sprint48.hms_seconds(job.get('resources_used', {}).get('walltime'))
    started = base.start_epoch(job.get('stime'))
    if not 0 < wall <= 86400 or not 0 <= used <= wall or not 0 < started <= now:
        raise ValueError('invalid ClusterB allocation lifetime')
    remaining = min(wall-used-300, started+wall-now-300, config['deadline_utc_epoch']-now-20)
    return dict(job_id=JOB, gpu=gpu, seconds=max(0, int(remaining)))


def prepare(config_path, transition_path):
    config = base.read_config(config_path)
    transition = checked(transition_path)
    if (transition.get('config_sha256') != config['sha256']
            or transition.get('clustera_job') != '2000000.clustera-pbs-01.example.org'
            or transition.get('clustera_job_state') not in base.submissions.TERMINAL
            or transition.get('watcher_stopped') is not True):
        raise ValueError('verified stopped-watcher and terminal-ClusterA handoff required')
    value = seal(dict(format='mlsys_evidence_v18_idle_clusterb', parent_config=str(Path(config_path).resolve()),
        parent_sha256=config['sha256'], transition=str(Path(transition_path).resolve()),
        transition_sha256=file_digest(transition_path), source_sha256=file_digest(__file__),
        launcher_sha256=file_digest(Path(config['cwd'])/SCRIPT),
        job=JOB, allowed_host='clusterb-gpu-01', maximum_workers=8))
    path = Path(config['study_root'])/'operations/clusterb-config.json'
    write_immutable(path, value)
    return value


def worker(path, gpu, seconds):
    launch = checked(path)
    config = base.read_config(launch['parent_config'])
    if (launch.get('format') != 'mlsys_evidence_v18_idle_clusterb'
            or launch.get('parent_sha256') != config['sha256']
            or launch.get('source_sha256') != file_digest(__file__)
            or launch.get('launcher_sha256') != file_digest(Path(config['cwd'])/SCRIPT)
            or launch.get('transition_sha256') != file_digest(launch['transition'])
            or launch.get('job') != JOB or launch.get('maximum_workers') != 8
            or launch.get('allowed_host') != 'clusterb-gpu-01'
            or socket.gethostname().split('.')[0] != launch['allowed_host']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != str(gpu)
            or type(seconds) is not int or not 180 <= seconds <= 12000):
        raise ValueError('idle ClusterB worker identity, source, GPU, or duration mismatch')
    if time.time() >= config['stop_new_work_utc_epoch']:
        return {'status': 'admission_closed', 'claims_run': 0}
    result = subprocess.run(['qstat', '-f', '-F', 'json', JOB], check=True, capture_output=True, text=True, timeout=30)
    allocation = validate_allocation(config, json.loads(result.stdout), launch['allowed_host'], gpu, time.time())
    stop = [False]
    previous = {sig: signal.signal(sig, lambda *_: stop.__setitem__(0, True)) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with base.allocation_checks.gpu_lease(launch['allowed_host'], gpu) as fd:
            return base.run_claims(config, allocation, seconds, fd, lambda: stop[0])
    except BlockingIOError:
        return {'status': 'gpu_lease_occupied', 'claims_run': 0}
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'worker'])
    parser.add_argument('--config', type=Path, default=base.ROOT/'operations/config.json')
    parser.add_argument('--transition', type=Path, default=base.ROOT/'operations/clustera-to-idle-clusterb.json')
    parser.add_argument('--clusterb-config', type=Path, default=base.ROOT/'operations/clusterb-config.json')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--seconds', type=int, default=12000)
    args = parser.parse_args(argv)
    result = prepare(args.config, args.transition) if args.phase == 'prepare' else worker(args.clusterb_config, args.gpu, args.seconds)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
