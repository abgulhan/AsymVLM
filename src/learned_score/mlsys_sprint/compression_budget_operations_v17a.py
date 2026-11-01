"""Bounded post-child GPU-accounting cooldown; frozen v17 remains unchanged.

The original allocation and nonce journal are adopted, never reset. An already
queued v17 job runs unchanged; only a remaining authorized successor uses this
worker. No new submission allowance, task, scientific source, or deadline.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import socket
import subprocess
import time

from . import compression_budget_operations_v17 as base
from .compression_matrix_v8 import seal
from .followup_common import write_immutable
from .state import atomic_json, file_digest

PBS_SCRIPT = 'scripts/mlsys_budget_v17a_clustera.pbs'
WORKER_SCRIPT = 'scripts/mlsys_budget_v17a_worker.sh'
CONFIG = base.ROOT / 'operations/config-v17a.json'


def sources(project):
    return [Path(__file__).resolve(), Path(project) / PBS_SCRIPT, Path(project) / WORKER_SCRIPT]


def prepare(parent_path):
    parent = base.read_config(parent_path)
    root = Path(parent['study_root']); path = root / 'operations/config-v17a.json'
    if path.exists():
        return read_config(path)[1]
    pins = {**parent['input_hashes'], str(Path(parent_path).resolve()): file_digest(parent_path),
            **{str(p.resolve()): file_digest(p) for p in sources(parent['cwd'])}}
    value = seal({**{k:v for k,v in parent.items() if k != 'sha256'},
        'study_operations': 'compression_budget_v17a', 'pbs_script': PBS_SCRIPT,
        'parent_config_path': str(Path(parent_path).resolve()), 'parent_config_sha256': parent['sha256'],
        'journal_config_sha256': parent['sha256'], 'input_hashes': pins,
        'repair': 'at most15seconds idle-accounting cooldown only after our child exits and while no compute apps exist',
        'maximum_post_child_cooldown_seconds': 15,
        'original_history_preserved': True, 'scientific_protocol_changed': False})
    write_immutable(path, value)
    return read_config(path)[1]


def read_config(path):
    value = base.sprint48.read_config(path)
    parent = base.read_config(value['parent_config_path'])
    root = Path(parent['study_root']); project = Path(parent['cwd'])
    mutable = {'sha256', 'input_hashes', 'pbs_script', 'study_operations'}
    if (Path(path).resolve() != root / 'operations/config-v17a.json'
            or Path(value['parent_config_path']).resolve() != root / 'operations/config.json'
            or value.get('parent_config_sha256') != parent['sha256']
            or value.get('journal_config_sha256') != parent['sha256']
            or value.get('study_operations') != 'compression_budget_v17a'
            or value.get('pbs_script') != PBS_SCRIPT
            or value.get('maximum_post_child_cooldown_seconds') != 15
            or value.get('original_history_preserved') is not True
            or value.get('scientific_protocol_changed') is not False
            or any(value.get(k) != v for k,v in parent.items() if k not in mutable)):
        raise ValueError('v17a may change only the bounded post-child idle probe and launcher')
    required = {str(p.resolve()): file_digest(p) for p in sources(project)}
    required[str(Path(value['parent_config_path']).resolve())] = file_digest(value['parent_config_path'])
    required.update(parent['input_hashes'])
    if any(value['input_hashes'].get(k) != v for k,v in required.items()):
        raise ValueError('v17a dependency or parent pin missing')
    return parent, value


def probe_gpu(gpu, timeout):
    prefix = ['nvidia-smi', '-i', str(gpu)]
    apps = subprocess.check_output([*prefix, '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                   text=True, timeout=timeout).strip()
    if apps:
        return {'empty': False, 'apps': apps, 'memory_mib': None, 'utilization_percent': None}
    raw = subprocess.check_output([*prefix, '--query-gpu=memory.used,utilization.gpu',
                                   '--format=csv,noheader,nounits'], text=True, timeout=timeout)
    parts = [x.strip() for x in raw.strip().split(',')]
    if len(parts) != 2 or any(not p.isdigit() for p in parts):
        raise ValueError('unverifiable GPU accounting: ' + raw)
    memory, utilization = map(int, parts)
    return {'empty': memory == 0 and utilization == 0, 'apps': '',
            'memory_mib': memory, 'utilization_percent': utilization}


def gpu_idle(gpu, *, allow_cooldown):
    start = time.monotonic(); deadline = start + (15 if allow_cooldown else 6)
    first = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        observation = probe_gpu(gpu, timeout=min(3, remaining / 2))
        first = observation if first is None else first
        if observation['empty'] or observation['apps'] or not allow_cooldown:
            if first != observation or not observation['empty']:
                print(json.dumps({'event': 'v17a_gpu_idle_probe', 'gpu': gpu,
                    'after_owned_child': allow_cooldown, 'first': first, 'last': observation,
                    'elapsed_seconds': time.monotonic()-start}), flush=True)
            return observation['empty']
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(1, remaining))


@contextmanager
def worker_binding():
    old_idle, old_run = base.leases.gpu_is_empty, base.overnight.run_claim
    state = {'owned_child_just_finished': False}
    def run(*args, **kwargs):
        result = old_run(*args, **kwargs)
        state['owned_child_just_finished'] = True
        return result
    def idle(gpu):
        allow = state['owned_child_just_finished']
        state['owned_child_just_finished'] = False
        return gpu_idle(gpu, allow_cooldown=allow)
    base.leases.gpu_is_empty, base.overnight.run_claim = idle, run
    try:
        yield
    finally:
        base.leases.gpu_is_empty, base.overnight.run_claim = old_idle, old_run


@contextmanager
def catalog_binding(parent):
    previous = base.readiness
    base.readiness = lambda _config: previous(parent)
    try:
        yield
    finally:
        base.readiness = previous


def watch(path, *, once=False, dry_run=False):
    if not socket.gethostname().startswith('clustera-login-'):
        raise ValueError('use the authenticated clustera-mlsys login session')
    parent, launch = read_config(path)
    state = Path(parent['state_root']); state.mkdir(parents=True, exist_ok=True)
    # The old watcher must first release this same lock. Catalog/task state and
    # journal config identity stay exactly the original values.
    with (state / 'watch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        journal_path = state / 'submission_journal.json'
        journal = json.loads(journal_path.read_text()) if journal_path.exists() else {
            'config_sha256': parent['sha256'], 'submissions': []}
        if journal.get('config_sha256') != parent['sha256']:
            raise ValueError('existing accepted-job history belongs to another protocol')
        while time.time() < parent['deadline_utc_epoch']:
            try:
                parent, launch = read_config(path)
                with catalog_binding(parent):
                    result = base.tick(launch, journal, journal_path, dry_run=dry_run)
            except (ValueError, OSError, subprocess.SubprocessError) as exc:
                result = {'time': time.time(), 'status': 'waiting_for_verified_state', 'error': repr(exc)}
            result['operational_repair_config_sha256'] = launch['sha256']
            atomic_json(state / 'status-v17a.json', result)
            print(json.dumps(result), flush=True)
            if once or dry_run or result['status'] in {'complete', 'pipeline_terminal', 'admission_closed',
                    'uncertain', 'uncertain_submission_requires_audit', 'accepted_job_missing_requires_audit',
                    'submission_budget_exhausted', 'insufficient_study_lifetime'}:
                return result
            time.sleep(min(30, max(0, parent['deadline_utc_epoch']-time.time())))
    return {'status': 'physical_deadline_reached'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'watch', 'once', 'inspect', 'worker'))
    parser.add_argument('--parent-config', type=Path, default=base.ROOT/'operations/config.json')
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--seconds', type=int, default=3300)
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        value = prepare(args.parent_config); result = {'status': 'prepared', 'sha256': value['sha256']}
    elif args.phase == 'worker':
        parent, _ = read_config(args.config)
        with worker_binding():
            result = base.worker(parent, args.seconds)
    else:
        result = watch(args.config, once=args.phase == 'once', dry_run=args.phase == 'inspect')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    main()
