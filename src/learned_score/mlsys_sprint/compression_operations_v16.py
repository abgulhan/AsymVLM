"""Eight-hour v16 queues on owned idle ClusterB GPUs and bounded ClusterA debug.

Reuse the frozen task locks, node guards, catalog and nonce-aware PBS admission.
Only process-local bindings change job identity and queue preference.  Scientific
plans verify their own sources; this coordinator does not pin a changing campaign
implementation.  No existing jobs, plans, states or results are rewritten.
"""
import argparse
from contextlib import contextmanager
import getpass
import json
import math
import os
from pathlib import Path
import shlex
import socket
import time

from . import compression_operations_v15 as base
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261025_v16')
JOB_NAME = 'mlsys-compression-v16'
PBS_SCRIPT = 'scripts/mlsys_compression_v16_clustera.pbs'
CLUSTERA_WORKER = 'scripts/mlsys_compression_v16_clustera_worker.sh'
IDLE_WORKER = 'scripts/mlsys_compression_v16_idle_worker.sh'
SESSION_PREFIX = 'mlsys-v16-clusterb-gpu-'
MODELS = ('qwen25', 'llava_next', 'phi35')
STAGES = ('canary', 'features-fit', 'features-screening', 'features-evaluation',
          'labels-fit', 'labels-screening', 'evaluation-primary',
          'evaluation-conservative', 'evaluation-aggressive', 'dev-controls', 'systems', 'text',
          'features-expanded', 'evaluation-expanded-primary',
          'evaluation-expanded-conservative', 'evaluation-expanded-aggressive')
PHI_EXTRA_STAGES = ('features-extra-fit', 'features-extra-screening', 'labels-extra-fit', 'labels-extra-screening')
CUDNN_DIRECTORY = '.data/cudnn-v15g-diagnostic'
CUDNN_HASHES = {
    'libcudnn_engines_precompiled.so.9.13.0': '436044d9469d4c1e06cf39662838e44b55c71301fa464637e019bdc58afe57a6',
    'libcudnn_engines_runtime_compiled.so.9.13.0': '27863fcc2ce622bcdc4a0dc647dbbb16bc293d233af87641ca54ef54e538bd27',
    'libcudnn_graph.so.9.13.0': 'f1c05da0b12be56661ad749e025fc55d94bf1b9cda6ec7016b7b7a6e4a4ee9f5',
    'libcudnn_heuristic.so.9.13.0': 'd0bbf926390962a40358bdfd4daa1ff5caf9215528ab1cbd71a1cb9f804131a6',
}


def plan_paths(root, role):
    if role not in ('clusterb', 'clustera'):
        raise ValueError('unknown site')
    return ([str(Path(root) / 'plans' / f'{stage}-{model}' / 'plan.json')
             for stage in STAGES for model in MODELS]
            + [str(Path(root) / 'plans' / f'{stage}-phi35' / 'plan.json') for stage in PHI_EXTRA_STAGES])


def _study(path):
    path = Path(path)
    study = checked(path)
    project, root = Path(study['cwd']), Path(study['output_root'])
    if (study.get('format') != 'mlsys_compression_campaign_v16'
            or root.resolve() != (project / ROOT).resolve()
            or path.resolve() != (root / 'study.json').resolve()):
        raise ValueError('the sealed v16 study and dedicated root are required')
    times = [study.get(k) for k in ('created_utc_epoch', 'method_freeze_utc_epoch', 'deadline_utc_epoch')]
    if (any(type(v) not in (int, float) or not math.isfinite(v) for v in times)
            or not times[0] < times[1] < times[2]
            or times[2] - times[0] > 8 * 3600):
        raise ValueError('study must preserve an at-most-eight-hour window and earlier method freeze')
    return study


def source_paths(project):
    project = Path(project)
    relative = (
        'learned_score/mlsys_sprint/compression_operations_v16.py',
        'learned_score/mlsys_sprint/compression_operations_v15.py',
        'learned_score/mlsys_sprint/compression_operations_v14.py',
        'learned_score/mlsys_sprint/compression_operations_v13.py',
        'learned_score/mlsys_sprint/compression_operations_v13a.py',
        'learned_score/mlsys_sprint/adaptive_operations_v4.py',
        'learned_score/mlsys_sprint/cache_dedicated_clusterb_v4.py',
        'learned_score/mlsys_sprint/sprint48.py',
        'learned_score/mlsys_sprint/sprint48_scheduler_v3.py',
        'learned_score/mlsys_sprint/clustera_queue_policy_v5.py',
        'learned_score/mlsys_sprint/compression_clustera_v11a.py',
        'learned_score/mlsys_sprint/overnight.py',
        'learned_score/mlsys_sprint/worker.py',
        'learned_score/mlsys_sprint/state.py',
        'learned_score/mlsys_sprint/compression_matrix_v8.py',
        'learned_score/mlsys_sprint/followup_common.py',
        'activate_env.sh', PBS_SCRIPT, CLUSTERA_WORKER, IDLE_WORKER)
    return [project / name for name in relative] + [project / CUDNN_DIRECTORY / name for name in CUDNN_HASHES]


def _library_links(project):
    root = Path(project) / CUDNN_DIRECTORY
    for name in CUDNN_HASHES:
        target = root / name
        alias = root / name.removesuffix('.13.0')
        if not target.is_file() or alias.resolve() != target.resolve():
            raise ValueError('shared cuDNN alias is missing or points outside its pinned version: ' + name)


def build_config(study_path, role):
    study = _study(study_path)
    root, project = Path(study['output_root']), Path(study['cwd'])
    path = root / 'coordinator' / (role + '-config.json')
    paths = plan_paths(root, role)
    if path.exists():
        return read_config(path)
    _library_links(project)
    sources = [Path(study_path), *source_paths(project)]
    value = seal({
        'format': base.engine.sprint48.FORMAT, 'study_operations': 'compression_v16',
        'study_path': str(root / 'study.json'), 'study_sha256': study['sha256'],
        'study_root': str(root), 'cwd': study['cwd'], 'user': getpass.getuser(),
        'coordinator_role': role, 'created_utc_epoch': study['created_utc_epoch'],
        'method_freeze_utc_epoch': study['method_freeze_utc_epoch'],
        'deadline_utc_epoch': study['deadline_utc_epoch'],
        'maximum_submissions': 10, 'maximum_nonterminal_debug_nodes': 2,
        'maximum_clusterb_workers': 16, 'state_root': str(root / 'coordinator' / role),
        'pbs_script': PBS_SCRIPT, 'plan_paths': paths,
        'shared_plan_task_locks_across_sites': True,
        'clustera_launch': {'mpi_ranks': 8, 'ranks_per_node': 4, 'nodes': 2, 'walltime_seconds': 3600},
        'clusterb_initial_affinity': {'qwen25': [0, 1], 'llava_next': [2, 3], 'phi35': [4, 5, 6, 7]},
        'cudnn_directory': CUDNN_DIRECTORY, 'cudnn_versioned_libraries': CUDNN_HASHES,
        'input_hashes': {str(p.resolve()): file_digest(p) for p in sources},
    })
    write_immutable(path, value)
    return read_config(path)


def read_config(path):
    config = base.engine.sprint48.read_config(path)
    study = _study(config['study_path'])
    root = Path(study['output_root'])
    role = config['coordinator_role']
    if (config.get('study_operations') != 'compression_v16'
            or config.get('study_sha256') != study['sha256']
            or config['user'] != getpass.getuser()
            or Path(config['cwd']).resolve() != Path(study['cwd']).resolve()
            or Path(config['study_root']).resolve() != root.resolve()
            or Path(config['state_root']).resolve() != (root / 'coordinator' / role).resolve()
            or Path(path).resolve() != (root / 'coordinator' / (role + '-config.json')).resolve()
            or config['plan_paths'] != plan_paths(root, role)
            or any(config[k] != study[k] for k in ('created_utc_epoch', 'method_freeze_utc_epoch', 'deadline_utc_epoch'))
            or config['maximum_submissions'] != 10
            or config['maximum_nonterminal_debug_nodes'] != 2
            or config['maximum_clusterb_workers'] != 16
            or config['pbs_script'] != PBS_SCRIPT
            or config.get('shared_plan_task_locks_across_sites') is not True
            or config.get('clustera_launch') != {'mpi_ranks': 8, 'ranks_per_node': 4, 'nodes': 2, 'walltime_seconds': 3600}
            or config.get('clusterb_initial_affinity') != {'qwen25': [0, 1], 'llava_next': [2, 3], 'phi35': [4, 5, 6, 7]}):
        raise ValueError('v16 operations identity, catalog, limits or deadline changed')
    if config.get('cudnn_directory') != CUDNN_DIRECTORY or config.get('cudnn_versioned_libraries') != CUDNN_HASHES:
        raise ValueError('shared numerical-library identity changed')
    _library_links(config['cwd'])
    for name, expected in CUDNN_HASHES.items():
        if config['input_hashes'].get(str((Path(config['cwd']) / CUDNN_DIRECTORY / name).resolve())) != expected:
            raise ValueError('shared numerical-library hash differs: ' + name)
    required = {str(p.resolve()) for p in [Path(config['study_path']), *source_paths(config['cwd'])]}
    if set(config['input_hashes']) != required:
        raise ValueError('v16 operations source pins differ from the fixed dependency inventory')
    return config


def preferred_queues(queues, device, role):
    if role not in ('clusterb', 'clustera') or not str(device).isdigit():
        raise ValueError('explicit valid site and GPU index required')
    index = int(device)
    if index not in range(8 if role == 'clusterb' else 4):
        raise ValueError('GPU index outside site allocation')
    preferred = ('qwen25' if index < 2 else 'llava_next' if index < 4 else 'phi35') if role == 'clusterb' else (
        'qwen25' if index == 0 else 'llava_next' if index == 1 else 'phi35')
    priorities = {'canary': 0, 'features-fit': 1, 'labels-fit': 2,
                  'features-extra-fit': 1, 'labels-extra-fit': 2,
                  'features-screening': 3, 'labels-screening': 4, 'dev-controls': 5,
                  'features-extra-screening': 3, 'labels-extra-screening': 4,
                  'features-evaluation': 6, 'evaluation-primary': 7,
                  'evaluation-conservative': 8, 'evaluation-aggressive': 9,
                  'features-expanded': 10, 'evaluation-expanded-primary': 11,
                  'evaluation-expanded-conservative': 12, 'evaluation-expanded-aggressive': 13,
                  'systems': 14, 'text': 15}
    def order(queue):
        item, plan, _, _ = queue
        label = Path(item.get('plan', '')).parent.name + ' ' + ' '.join(t['id'] for t in plan['tasks'])
        stage = min((priority for name, priority in priorities.items() if name in label), default=99)
        return stage, preferred not in {task.get('model_tag') for task in plan['tasks']}
    return sorted((q for q in queues if time.time() < q[1]['deadline_utc_epoch']), key=order)


def clusterb_tick(config, config_path):
    inventory = base.queue_status(config)
    count, _ = base.ready(config)
    result = {'time': time.time(), 'ready_tasks': count, 'queue': inventory, 'launched': []}
    if inventory['all_terminal']:
        return {**result, 'status': 'complete' if inventory['complete'] else 'finished_with_failures'}
    if not count:
        return {**result, 'status': 'waiting_for_published_work_or_dependencies'}
    owned = base.engine.clusterb.owned_allocations(base.local_query(config['user']), config['user'],
                                                time.time(), config['deadline_utc_epoch'])
    sessions = base.engine.clusterb.tmux_sessions()
    count = min(count, max(0, config['maximum_clusterb_workers'] - sum(s.startswith(SESSION_PREFIX) for s in sessions)))
    logs = Path(config['study_root']) / 'coordinator/logs'
    logs.mkdir(parents=True, exist_ok=True)
    for allocation in sorted(owned, key=lambda a: a['host']):
        if not count:
            break
        for gpu in base.direct_idle_gpus(allocation['host']):
            name = f'mlsys-v16-{allocation["host"]}-gpu{gpu}'
            if name in sessions or not count:
                continue
            seconds = min(10800, allocation['seconds'], int(config['deadline_utc_epoch'] - time.time()))
            if seconds < 600:
                continue
            remote = 'bash ' + shlex.join([str(Path(config['cwd']) / IDLE_WORKER),
                allocation['job_id'], str(gpu), str(seconds), str(config_path)])
            command = ['ssh', *base.engine.clusterb.SSH_OPTIONS, allocation['host'], remote]
            log = logs / (name + '-' + str(int(time.time())) + '.log')
            base.engine.clusterb.command(['tmux', 'new-session', '-d', '-s', name,
                'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'])
            sessions.add(name)
            result['launched'].append({'session': name, 'job': allocation['job_id'], 'gpu': gpu, 'log': str(log)})
            count -= 1
    return {**result, 'status': 'workers_active_or_waiting_for_idle_gpu'}


@contextmanager
def bindings():
    previous = base.read_config, base.clusterb_tick, base.JOB_NAME, base.preferred_queues
    base.read_config, base.clusterb_tick = read_config, clusterb_tick
    base.JOB_NAME, base.preferred_queues = JOB_NAME, preferred_queues
    try:
        yield
    finally:
        base.read_config, base.clusterb_tick, base.JOB_NAME, base.preferred_queues = previous


def watch(path, *, once=False, dry_run=False):
    read_config(path)
    with bindings():
        return base.watch(path, once=once, dry_run=dry_run)


def dispatch(config, seconds, worker_id):
    if type(seconds) is not int or not 0 < seconds <= 10800:
        raise ValueError('worker duration must be an integer between 1 and 10800 seconds')
    with bindings():
        return base.dispatch(config, seconds, worker_id)


def clusterb_worker(config, job, gpu, seconds):
    host = socket.gethostname().split('.')[0]
    if (config['coordinator_role'] != 'clusterb' or not isinstance(job, str)
            or not base.engine.clusterb.JOB_PATTERN.fullmatch(job)
            or type(gpu) is not int or gpu not in range(8)
            or type(seconds) is not int or not 0 < seconds <= 10800):
        raise ValueError('invalid ClusterB role, job, GPU or worker duration')
    owned = base.engine.clusterb.owned_allocations(base.local_query(config['user']), config['user'],
                                                time.time(), config['deadline_utc_epoch'])
    allocation = next((a for a in owned if a['job_id'] == job and a['host'] == host), None)
    if allocation is None:
        raise ValueError('allocation is not owned, interactive, live, or on this host')
    inventory = base.engine.clusterb.command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                           '--format=csv,noheader,nounits'])
    apps = base.engine.clusterb.command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if gpu not in base.engine.clusterb.parse_idle_gpus(inventory, apps):
        raise ValueError('GPU is no longer idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PBS_JOBID=job, ASYMVLM_V13_IDLE_LEASE=config['sha256'])
    return dispatch(config, min(seconds, allocation['seconds']), f'v16-{job}-{host}-{gpu}')


def allocation(config):
    """Read-only allocation/admission snapshot; never submit or terminate jobs."""
    if config['coordinator_role'] == 'clusterb':
        return {'owned_allocations': base.engine.clusterb.owned_allocations(
            base.local_query(config['user']), config['user'], time.time(), config['deadline_utc_epoch'])}
    if not socket.gethostname().startswith('clustera-login-'):
        raise ValueError('ClusterA allocation inspection requires its authenticated login')
    _, admission = base.engine.observe_clustera(config)
    return {'admission': admission}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'watch', 'once', 'inspect', 'dispatch', 'owned-idle-worker', 'allocation'))
    parser.add_argument('--study', type=Path, default=ROOT / 'study.json')
    parser.add_argument('--role', choices=('clusterb', 'clustera'), default='clusterb')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--seconds', type=int, default=3300)
    parser.add_argument('--worker-id', default='v16-worker')
    parser.add_argument('--job-id')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        result = build_config(args.study, args.role)
    else:
        path = args.config or ROOT / 'coordinator' / (args.role + '-config.json')
        config = read_config(path)
        if args.phase == 'dispatch':
            result = dispatch(config, args.seconds, args.worker_id)
        elif args.phase == 'owned-idle-worker':
            result = clusterb_worker(config, args.job_id, args.gpu, args.seconds)
        elif args.phase == 'allocation':
            result = allocation(config)
        else:
            result = watch(path, once=args.phase == 'once', dry_run=args.phase == 'inspect')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
