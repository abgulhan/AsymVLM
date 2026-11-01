"""Bounded multi-model paper-evidence campaign with site-pinned physical work.

Only declared immutable plans are admitted. ClusterB consumes verified idle GPUs
in owned interactive allocations; ClusterA admits at most two debug nodes across
all workflows. Failed work is never reported as successful completion.
"""
import argparse
from contextlib import contextmanager
import fcntl
import getpass
import importlib
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import time

from . import compression_operations_v13 as engine
from .compression_operations_v14 import direct_idle_gpus
from .compression_operations_v13a import local_query
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, file_digest
from .worker import _state

ROOT = Path('reproduction_results/mlsys_compression_20261023_v15')
JOB_NAME = 'mlsys-compression-v15'
PBS_SCRIPT = 'scripts/mlsys_compression_v15_clustera.pbs'
_BASE_CLAIM = engine.overnight.claim


def plan_paths(root, role):
    root = Path(root)
    if role == 'clustera':
        return [str(root / 'text/plans' / f'text-{phase}-{model}' / 'plan.json')
                for phase in ('canary', 'prepare', 'pages') for model in ('qwen25', 'qwen25_7b')]
    if role != 'clusterb':
        raise ValueError('unknown site')
    phi = ['parity-phi35'] + [f'features-{s}-phi35' for s in ('fit', 'screening', 'evaluation')]
    phi += [f'physical-{s}-phi35' for s in ('fit', 'screening', 'evaluation', 'adaptive')]
    return ([str(root / 'third-model/plans' / name / 'plan.json') for name in phi]
        + [str(root / 'visual/plans' / model / 'plan.json') for model in ('qwen25', 'llava_next')]
        + [str(root / 'batch/plans' / model / 'plan.json') for model in ('qwen25', 'llava_next')])


def build_config(project, role, *, hours=48, maximum_submissions=48):
    project = Path(project).resolve()
    root = project / ROOT
    path = root / 'coordinator' / (role + '-config.json')
    if path.exists():
        return read_config(path)
    if not 0 < hours <= 48 or not 1 <= maximum_submissions <= 48:
        raise ValueError('campaign must remain within 48 hours and 48 submissions')
    now = time.time()
    sources = [Path(__file__), Path(engine.__file__),
               project / 'learned_score/mlsys_sprint/compression_operations_v14.py',
               project / 'learned_score/mlsys_sprint/compression_operations_v13a.py',
               project / 'learned_score/mlsys_sprint/sprint48.py',
               project / 'learned_score/mlsys_sprint/overnight.py',
               project / 'learned_score/mlsys_sprint/worker.py',
               project / PBS_SCRIPT,
               project / 'scripts/mlsys_compression_v15_clustera_worker.sh',
               project / 'scripts/mlsys_compression_v15_idle_worker.sh']
    value = seal({'format': engine.sprint48.FORMAT, 'study_operations': 'compression_v15',
        'study_root': str(root), 'cwd': str(project), 'user': getpass.getuser(),
        'coordinator_role': role, 'created_utc_epoch': now, 'deadline_utc_epoch': now + hours * 3600,
        'maximum_submissions': maximum_submissions, 'maximum_nonterminal_debug_nodes': 2,
        'maximum_clusterb_workers': 16, 'state_root': str(root / 'coordinator' / role),
        'pbs_script': PBS_SCRIPT, 'plan_paths': plan_paths(root, role),
        'input_hashes': {str(p.resolve()): file_digest(p) for p in sources}})
    write_immutable(path, value)
    return read_config(path)


def read_config(path):
    config = engine.sprint48.read_config(path)
    root = Path(config['cwd']) / ROOT
    if (config.get('study_operations') != 'compression_v15'
            or config['user'] != getpass.getuser()
            or Path(config['study_root']) != root
            or config['plan_paths'] != plan_paths(root, config['coordinator_role'])
            or config['maximum_nonterminal_debug_nodes'] != 2
            or not 1 <= config['maximum_clusterb_workers'] <= 16
            or config['pbs_script'] != PBS_SCRIPT):
        raise ValueError('v15 operations identity or site-affinity changed')
    return config


def queue_status(config):
    queues, missing = engine.sprint48.load_queues(config, 'gpu')
    engine.overnight.propagate_blocked(queues)
    counts, failures = {}, []
    for _, plan, state, tags in queues:
        for task in plan['tasks']:
            if task['tag'] not in tags:
                continue
            record = _state(state / (task['id'] + '.json'))
            name = record.get('status', 'pending')
            if name not in engine.overnight.TERMINAL and time.time() >= plan['deadline_utc_epoch']:
                name = 'expired'
            counts[name] = counts.get(name, 0) + 1
            if name in ('failed', 'blocked', 'expired'):
                failures.append({'task': task['id'], 'status': name, 'reason': record.get('reason'),
                                 'state': str(state / (task['id'] + '.json'))})
    terminal = not missing and not (set(counts) - {'complete', 'failed', 'blocked', 'expired'})
    return {'counts': counts, 'unpublished_plans': missing, 'failures': failures,
            'all_terminal': terminal, 'complete': terminal and not failures}


def preferred_queues(queues, device, role):
    queues = [q for q in queues if time.time() < q[1]['deadline_utc_epoch']]
    preferred = ('qwen25' if int(device) < 2 else 'qwen25_7b') if role == 'clustera' else (
        'phi35' if int(device) < 2 else 'qwen25' if int(device) < 5 else 'llava_next')
    def order(item):
        tasks = item[1]['tasks']
        label = ' '.join(t['id'] for t in tasks)
        stage = (0 if 'canary' in label or 'parity' in label else
                 1 if 'prepare' in label or 'features' in label else
                 2 if 'fit' in label else 3)
        return stage, preferred not in {t.get('model_tag') for t in tasks}
    return sorted(queues, key=order)


def claim_with_deadline(queues, *, dry_run=False, remaining_seconds=None):
    for queue in queues:
        remaining = queue[1]['deadline_utc_epoch'] - time.time()
        if remaining <= 0:
            continue
        if remaining_seconds is not None:
            remaining = min(remaining, remaining_seconds)
        found = _BASE_CLAIM([queue], dry_run=dry_run, remaining_seconds=remaining)
        if found is not None:
            return found
    return None


def ready(config, maximum=16):
    queues, _ = engine.sprint48.load_queues(config, 'gpu')
    queues = [q for q in queues if time.time() < q[1]['deadline_utc_epoch']]
    locks = []
    try:
        for _ in range(maximum):
            found = claim_with_deadline(queues, remaining_seconds=3300)
            if found is None:
                break
            locks.append(found[-1])
        return len(locks), queue_status(config)['all_terminal']
    finally:
        for lock in locks:
            lock.close()


@contextmanager
def bindings():
    prior = engine.JOB_NAME, engine.preferred_queues, engine.ready, engine.overnight.run_claim, engine.overnight.claim
    def bounded_run(selected, deadline, worker_id, stopping):
        remaining = max(0, selected[1]['deadline_utc_epoch'] - time.time())
        return prior[3](selected, min(deadline, time.monotonic() + remaining), worker_id, stopping)
    engine.JOB_NAME, engine.preferred_queues, engine.ready = JOB_NAME, preferred_queues, ready
    engine.overnight.run_claim = bounded_run
    engine.overnight.claim = claim_with_deadline
    try:
        yield
    finally:
        engine.JOB_NAME, engine.preferred_queues, engine.ready, engine.overnight.run_claim, engine.overnight.claim = prior


def dispatch(config, seconds, worker_id):
    with bindings():
        return engine.dispatch(config, min(seconds, 10800), worker_id)


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
    apps = engine.clusterb.command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if gpu not in engine.clusterb.parse_idle_gpus(inventory, apps):
        raise ValueError('GPU is no longer idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PBS_JOBID=job,
                      ASYMVLM_V13_IDLE_LEASE=config['sha256'])
    return dispatch(config, min(seconds, allocation['seconds'], 10800), f'v15-{job}-{host}-{gpu}')


def clusterb_tick(config, config_path):
    inventory = queue_status(config)
    count, _ = ready(config)
    result = {'time': time.time(), 'ready_tasks': count, 'queue': inventory, 'launched': []}
    if inventory['all_terminal']:
        return {**result, 'status': 'complete' if inventory['complete'] else 'finished_with_failures'}
    if not count:
        return {**result, 'status': 'waiting_for_published_work_or_dependencies'}
    owned = engine.clusterb.owned_allocations(local_query(config['user']), config['user'],
                                            time.time(), config['deadline_utc_epoch'])
    sessions = engine.clusterb.tmux_sessions()
    live = sum(s.startswith('mlsys-v15-clusterb-gpu-') for s in sessions)
    count = min(count, max(0, config['maximum_clusterb_workers'] - live))
    logs = Path(config['study_root']) / 'coordinator/logs'
    logs.mkdir(parents=True, exist_ok=True)
    for allocation in sorted(owned, key=lambda a: a['host']):
        if not count:
            break
        for gpu in direct_idle_gpus(allocation['host']):
            name = f'mlsys-v15-{allocation["host"]}-gpu{gpu}'
            if name in sessions or not count:
                continue
            remote = 'bash ' + shlex.join([str(Path(config['cwd']) / 'scripts/mlsys_compression_v15_idle_worker.sh'),
                allocation['job_id'], str(gpu), str(min(10800, allocation['seconds'])), str(config_path)])
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
                    status = queue_status(config)
                    result['queue'] = status
                    if result.get('status') == 'complete' and not status['complete']:
                        result['status'] = 'finished_with_failures'
                elif dry_run:
                    result = {'ready_tasks': ready(config)[0], 'queue': queue_status(config),
                              'status': 'dry_run_no_launch'}
                else:
                    result = clusterb_tick(config, config_path)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_state',
                          'error': str(exc), 'type': type(exc).__name__}
            atomic_json(root / 'status.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result.get('status') in {'complete', 'finished_with_failures',
                    'uncertain', 'uncertain_submission_requires_audit', 'accepted_job_missing_requires_audit',
                    'submission_budget_exhausted', 'insufficient_study_lifetime'}:
                return result
            time.sleep(min(30, max(0, config['deadline_utc_epoch'] - time.time())))
    return {'status': 'deadline'}


def advance_cpu(project):
    root = Path(project).resolve() / ROOT
    branches = [('visual', 'compression_frontier_v15', 'study.json'),
                ('third-model', 'compression_third_model_v15', 'study.json'),
                ('text', 'compression_text_v15', 'config.json'),
                ('batch', 'compression_batch_v15', 'study.json')]
    result = {'time': time.time(), 'branches': {}, 'errors': {}}
    for branch, module, filename in branches:
        path = root / branch / filename
        if not path.exists():
            result['branches'][branch] = {'status': 'waiting_for_protocol'}
            continue
        try:
            api = importlib.import_module('.' + module, __package__)
            value = checked(path)
            result['branches'][branch] = api.advance(path if branch == 'text' else value)
        except Exception as exc:
            result['errors'][branch] = {'type': type(exc).__name__, 'error': str(exc)}
    result['complete'] = not result['errors'] and all(
        r.get('complete') is True or r.get('status') == 'complete' for r in result['branches'].values())
    atomic_json(root / 'status.json', result)
    return result


def cpu_watch(project, seconds=172800):
    if not 0 < seconds <= 172800:
        raise ValueError('CPU coordinator is limited to 48 hours')
    root = Path(project).resolve() / ROOT
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'cpu-watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            result = advance_cpu(project)
            print(json.dumps(result), flush=True)
            if result['complete']:
                return result
            time.sleep(min(30, max(0, end - time.monotonic())))
    return {'complete': False, 'status': 'deadline'}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('prepare', 'watch', 'once', 'inspect', 'dispatch',
                                    'owned-idle-worker', 'advance', 'cpu-watch'))
    p.add_argument('--role', choices=('clusterb', 'clustera'), default='clusterb')
    p.add_argument('--config', type=Path)
    p.add_argument('--seconds', type=int, default=3000)
    p.add_argument('--worker-id', default='v15-worker')
    p.add_argument('--job-id')
    p.add_argument('--gpu', type=int)
    a = p.parse_args(argv)
    if a.phase == 'prepare':
        result = build_config(Path.cwd(), a.role)
    elif a.phase in ('advance', 'cpu-watch'):
        result = advance_cpu(Path.cwd()) if a.phase == 'advance' else cpu_watch(Path.cwd(), a.seconds)
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
