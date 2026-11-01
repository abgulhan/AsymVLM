"""Restartable frozen-method campaign on idle ClusterB and bounded ClusterA debug.

Reuse the tested v13 task guards and nonce-based PBS admission without modifying
historical files. Process-local bindings change only job identity and priority.
ClusterB queries PBS locally and probes nodes directly, avoiding failed self-SSH.
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
import subprocess
import time

from . import compression_operations_v13 as engine
from .compression_operations_v13a import local_query
from .compression_campaign_v14 import ROOT, verify
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, file_digest

JOB_NAME = 'mlsys-compression-v14'
PBS_SCRIPT = 'scripts/mlsys_compression_v14_clustera.pbs'


def plan_names(role):
    if role not in ('clusterb', 'clustera'):
        raise ValueError('unknown site')
    names = [f'{stage}-{model}' for stage in ('full-features', 'full-primary')
             for model in ('qwen25', 'llava_next')]
    if role == 'clusterb':
        names = ['text-qwen25', 'text-llava_next', *names]
    return names


def build_config(study, role):
    verify(study)
    names = plan_names(role)
    root = Path(study['output_root'])
    value = seal({'format': engine.sprint48.FORMAT, 'study_operations': 'compression_v14',
        'study_path': str(root / 'study.json'), 'study_sha256': study['sha256'],
        'study_root': str(root), 'cwd': study['cwd'], 'user': getpass.getuser(),
        'coordinator_role': role, 'created_utc_epoch': study['created_utc_epoch'],
        'deadline_utc_epoch': study['deadline_utc_epoch'], 'maximum_submissions': 12,
        'maximum_nonterminal_debug_nodes': 2, 'state_root': str(root / 'coordinator' / role),
        'pbs_script': PBS_SCRIPT,
        'plan_paths': [str(root / 'plans' / name / 'plan.json') for name in names],
        'input_hashes': {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json')}})
    write_immutable(root / 'coordinator' / (role + '-config.json'), value)
    return value


def read_config(path):
    config = engine.sprint48.read_config(path)
    study = checked(config['study_path'])
    verify(study)
    expected = [str(Path(study['output_root']) / 'plans' / name / 'plan.json')
                for name in plan_names(config['coordinator_role'])]
    if (config.get('study_operations') != 'compression_v14' or config['study_sha256'] != study['sha256']
            or config['user'] != getpass.getuser() or config['maximum_nonterminal_debug_nodes'] != 2
            or config['deadline_utc_epoch'] != study['deadline_utc_epoch']
            or config['pbs_script'] != PBS_SCRIPT or config['plan_paths'] != expected):
        raise ValueError('campaign operations identity changed')
    return config


def preferred_queues(queues, device, role):
    preferred = 'qwen25' if int(device) < (4 if role == 'clusterb' else 2) else 'llava_next'
    def order(item):
        tasks = item[1]['tasks']
        label = ' '.join(t.get('id', '') for t in tasks)
        stage = 0 if 'text-' in label else 1 if 'features-' in label else 2
        return stage, preferred not in {t.get('model_tag') for t in tasks}
    return sorted(queues, key=order)


@contextmanager
def bindings():
    previous = engine.JOB_NAME, engine.preferred_queues
    engine.JOB_NAME, engine.preferred_queues = JOB_NAME, preferred_queues
    try:
        yield
    finally:
        engine.JOB_NAME, engine.preferred_queues = previous


def dispatch(config, seconds, worker_id):
    with bindings():
        return engine.dispatch(config, seconds, worker_id)


def direct_idle_gpus(host, run=None):
    if socket.gethostname().split('.')[0] != 'clusterb-login-01':
        raise ValueError('direct GPU probes require ClusterB login')
    execute = engine.clusterb.command if run is None else run
    def direct(argv):
        expected = ['ssh', *engine.clusterb.SSH_OPTIONS, '-J', 'clusterb-login-01', host]
        if argv[:-1] != expected:
            raise ValueError('unexpected frozen GPU probe transport')
        return execute(['ssh', *engine.clusterb.SSH_OPTIONS, host, argv[-1]])
    return engine.clusterb.idle_gpus(host, run=direct)


def clusterb_worker(config, job, gpu, seconds):
    host = socket.gethostname().split('.')[0]
    if not engine.clusterb.JOB_PATTERN.fullmatch(job) or gpu not in range(8):
        raise ValueError('invalid ClusterB job or GPU')
    owned = engine.clusterb.owned_allocations(local_query(config['user']), config['user'],
                                            time.time(), config['deadline_utc_epoch'])
    allocation = next((a for a in owned if a['job_id'] == job and a['host'] == host), None)
    if allocation is None:
        raise ValueError('allocation is not owned, interactive, live, or on this host')
    inventory = engine.clusterb.command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'])
    apps = engine.clusterb.command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader,nounits'])
    if gpu not in engine.clusterb.parse_idle_gpus(inventory, apps):
        raise ValueError('GPU is no longer idle')
    # The historical dispatcher checks this exact lease name against the new
    # config hash; it is not permission to enter a different study's lease.
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PBS_JOBID=job,
                      ASYMVLM_V13_IDLE_LEASE=config['sha256'])
    return dispatch(config, min(seconds, allocation['seconds']), f'v14-{job}-{host}-{gpu}')


def clusterb_tick(config, config_path):
    count, terminal = engine.ready(config)
    result = {'time': time.time(), 'ready_tasks': count, 'all_terminal': terminal, 'launched': []}
    if terminal:
        return {**result, 'status': 'complete'}
    if not count:
        return {**result, 'status': 'waiting_for_published_work'}
    owned = engine.clusterb.owned_allocations(local_query(config['user']), config['user'],
                                            time.time(), config['deadline_utc_epoch'])
    # Prefer this campaign's retained node; still permit verified idle GPUs in
    # other owned allocations, as explicitly authorized, without stopping jobs.
    owned.sort(key=lambda a: (a['host'] != 'clusterb-gpu-09', a['host']))
    sessions = engine.clusterb.tmux_sessions()
    logs = Path(config['study_root']) / 'coordinator/logs'
    logs.mkdir(parents=True, exist_ok=True)
    for allocation in owned:
        if not count:
            break
        for gpu in direct_idle_gpus(allocation['host']):
            name = f'mlsys-v14-{allocation["host"]}-gpu{gpu}'
            if name in sessions or not count:
                continue
            remote = 'bash ' + shlex.join([str(Path(config['cwd']) / 'scripts/mlsys_compression_v14_idle_worker.sh'),
                       allocation['job_id'], str(gpu), str(allocation['seconds']), str(config_path)])
            command = ['ssh', *engine.clusterb.SSH_OPTIONS, allocation['host'], remote]
            log = logs / (name + '-' + str(int(time.time())) + '.log')
            engine.clusterb.command(['tmux', 'new-session', '-d', '-s', name,
                                  'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'])
            sessions.add(name)
            result['launched'].append({'session': name, 'job': allocation['job_id'], 'gpu': gpu, 'log': str(log)})
            count -= 1
    return {**result, 'status': 'workers_active_or_waiting_for_idle_gpu'}


def watch(config_path, *, once=False, dry_run=False):
    config = read_config(config_path)
    role = config['coordinator_role']
    if not socket.gethostname().startswith(role + '-login-'):
        raise ValueError('watcher requires authenticated site login')
    root = Path(config['state_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = root / 'submission_journal.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            'config_sha256': config['sha256'], 'submissions': []}
        if journal['config_sha256'] != config['sha256']:
            raise ValueError('wrong submission journal')
        while time.time() < config['deadline_utc_epoch']:
            try:
                config = read_config(config_path)
                if role == 'clustera':
                    with bindings():
                        result = engine.clustera_tick(config, journal, journal_path, dry_run=dry_run)
                elif dry_run:
                    result = {'ready_tasks': engine.ready(config)[0], 'status': 'dry_run_no_launch'}
                else:
                    result = clusterb_tick(config, config_path)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_state',
                          'error': str(exc), 'type': type(exc).__name__}
            atomic_json(root / 'status.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result.get('status') in {'complete', 'uncertain',
                    'uncertain_submission_requires_audit', 'accepted_job_missing_requires_audit',
                    'submission_budget_exhausted', 'insufficient_study_lifetime'}:
                return result
            time.sleep(min(30, max(0, config['deadline_utc_epoch'] - time.time())))
    return {'status': 'deadline'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'watch', 'once', 'inspect', 'dispatch', 'owned-idle-worker'))
    p.add_argument('--study', type=Path, default=ROOT / 'study.json')
    p.add_argument('--config', type=Path)
    p.add_argument('--role', choices=('clusterb', 'clustera'), default='clusterb')
    p.add_argument('--seconds', type=int, default=3000)
    p.add_argument('--worker-id', default='v14-worker')
    p.add_argument('--job-id')
    p.add_argument('--gpu', type=int)
    a = p.parse_args(argv)
    if a.phase == 'prepare':
        result = build_config(checked(a.study), a.role)
    else:
        path = a.config or ROOT / 'coordinator' / (a.role + '-config.json')
        config = read_config(path)
        if a.phase == 'dispatch':
            result = dispatch(config, a.seconds, a.worker_id)
        elif a.phase == 'owned-idle-worker':
            result = clusterb_worker(config, a.job_id, a.gpu, a.seconds)
        else:
            result = watch(path, once=a.phase == 'once', dry_run=a.phase == 'inspect')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
