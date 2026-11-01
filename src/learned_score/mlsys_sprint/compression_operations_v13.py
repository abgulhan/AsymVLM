"""Bounded v13 dispatch: verified idle ClusterB GPUs and at most two debug nodes.

No imported historical coordinator is patched. PBS must itself enforce one
running two-node debug allocation per user; unknown inventories fail closed.
"""
import argparse
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import time

from . import adaptive_operations_v4 as clusterb
from . import overnight, clustera_queue_policy_v5, sprint48
from . import sprint48_scheduler_v3 as submissions
from .compression_matrix_v8 import checked, seal
from .compression_finalize_v13 import ROOT, verify
from .followup_common import write_immutable
from .state import atomic_json, file_digest
from .worker import verify_sources

PBS_SCRIPT = 'scripts/mlsys_compression_v13_clustera.pbs'
JOB_NAME = 'mlsys-compression-v13'


def build_config(study, role):
    verify(study)
    if role not in ('clusterb', 'clustera'):
        raise ValueError('unsupported site')
    root = Path(study['output_root'])
    models = ('qwen25', 'llava_next') if role == 'clusterb' else ('llava_next',)
    stages = tuple(split + '-' + phase for phase in ('learned-controls', 'div-deploy', 'div-matched')
                   for split in ('fit', 'screening')) + ('systems',)
    names = [f'{stage}-{model}' for stage in stages for model in models]
    sources = [PBS_SCRIPT, 'scripts/mlsys_compression_v13_clustera_worker.sh',
               'scripts/mlsys_compression_v13_idle_worker.sh',
               'learned_score/mlsys_sprint/compression_operations_v13.py']
    value = seal({'format': sprint48.FORMAT, 'study_operations': 'compression_v13',
        'study_path': str(root / 'study.json'), 'study_sha256': study['sha256'],
        'study_root': str(root), 'cwd': study['cwd'], 'user': getpass.getuser(), 'coordinator_role': role,
        'created_utc_epoch': study['created_utc_epoch'], 'deadline_utc_epoch': study['deadline_utc_epoch'],
        'maximum_submissions': 4, 'maximum_nonterminal_debug_nodes': 2,
        'state_root': str(root / 'coordinator' / role), 'pbs_script': PBS_SCRIPT,
        'plan_paths': [str(root / 'plans' / name / 'plan.json') for name in names],
        'input_hashes': {**study['input_hashes'], **{str(Path(study['cwd']) / name):
            file_digest(Path(study['cwd']) / name) for name in sources}}})
    path = root / 'coordinator' / (role + '-config.json')
    write_immutable(path, value)
    return value


def read_config(path):
    config = sprint48.read_config(path)
    study = checked(config['study_path'])
    verify(study)
    if (config.get('study_operations') != 'compression_v13' or config['study_sha256'] != study['sha256']
            or config['user'] != getpass.getuser() or config['maximum_nonterminal_debug_nodes'] != 2
            or config['deadline_utc_epoch'] != study['deadline_utc_epoch']):
        raise ValueError('v13 operations identity changed')
    allowed = Path(study['output_root']) / 'plans'
    for name in config['plan_paths']:
        path = Path(name).resolve()
        if path.parent.parent != allowed or path.name != 'plan.json':
            raise ValueError('plan escaped v13 namespace')
    return config


def ready(config, maximum=16):
    queues, missing = sprint48.load_queues(config, 'gpu')
    held = []
    try:
        for _ in range(maximum):
            claim = overnight.claim(queues, remaining_seconds=3000)
            if claim is None:
                break
            held.append(claim[-1])
        return len(held), sprint48.all_terminal(queues, missing)
    finally:
        for lock in held:
            lock.close()


def runnable_queues(queues, node_busy):
    """Never tie an idle GPU up waiting for unrelated node-wide timing work."""
    if not node_busy:
        return queues
    result = []
    for item, plan, state_root, tags in queues:
        isolated_tags = {task['tag'] for task in plan['tasks'] if task.get('resource') == 'isolated_gpu'}
        result.append((item, plan, state_root, [tag for tag in tags if tag not in isolated_tags]))
    return result


def preferred_queues(queues, device, role):
    """Four ClusterB GPUs/model initially, with immediate spillover and stage priority."""
    if role != 'clusterb':
        return queues
    preferred = 'qwen25' if int(device) < 4 else 'llava_next'
    priority = {'learned-controls': 0, 'div-deploy': 1, 'div-matched': 2, 'systems': 3}
    def order(item):
        tasks = item[1]['tasks']
        models = {task.get('model_tag') for task in tasks}
        label = ' '.join(task.get('id', '') for task in tasks)
        stage = next((n for name, n in priority.items() if name in label), 4)
        return stage, preferred not in models
    return sorted(queues, key=order)


def dispatch(config, seconds, worker_id):
    if not 0 < seconds <= 10800:
        raise ValueError('bounded worker duration required')
    host, device = socket.gethostname().split('.')[0], os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if config['coordinator_role'] == 'clustera':
        if (not re.fullmatch(r'x[0-9]+[a-z0-9]+', host) or device not in ('0', '1', '2', '3')
                or not submissions.JOB_ID.fullmatch(os.environ.get('PBS_JOBID', ''))
                or os.environ.get('PBS_JOBNAME') != JOB_NAME):
            raise ValueError('dispatch requires the owned v13 ClusterA compute allocation')
    elif (not clusterb.HOST_PATTERN.fullmatch(host) or device not in tuple(map(str, range(8)))
          or os.environ.get('ASYMVLM_V13_IDLE_LEASE') != config['sha256']):
        raise ValueError('dispatch requires a validated ClusterB GPU lease')
    end = time.monotonic() + min(seconds, max(0, config['deadline_utc_epoch'] - time.time()))
    stopping = [False]
    previous = {sig: signal.signal(sig, lambda *_: stopping.__setitem__(0, True))
                for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        while not stopping[0] and end - time.monotonic() >= 10:
            queues, _ = sprint48.load_queues(config, 'gpu')
            node_busy = bool(subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                '--format=csv,noheader,nounits'], text=True, timeout=20).strip())
            queues = runnable_queues(queues, node_busy)
            queues = preferred_queues(queues, device, config['coordinator_role'])
            selected = overnight.claim(queues, remaining_seconds=end - time.monotonic())
            if selected is None:
                print(json.dumps({'event': 'idle_gpu_released', 'worker': worker_id}), flush=True)
                return
            task, plan, state_root, state, lock = selected
            verify_sources(plan)
            guarded = {**task, 'argv': [str(Path(config['cwd']) / '.conda/asymvlm/bin/python'),
                '-B', '-u', '-m', 'learned_score.mlsys_sprint.sprint48', 'guard', '--mode',
                'exclusive' if task.get('resource') == 'isolated_gpu' else 'shared', '--', *task['argv']]}
            overnight.run_claim((guarded, plan, state_root, state, lock), end, worker_id, lambda: stopping[0])
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def clusterb_worker(config, job, gpu, seconds):
    host = socket.gethostname().split('.')[0]
    if not clusterb.JOB_PATTERN.fullmatch(job) or gpu not in range(8):
        raise ValueError('invalid ClusterB job/GPU')
    payload = clusterb.query_allocations(config['user'])
    owned = clusterb.owned_allocations(payload, config['user'], time.time(), config['deadline_utc_epoch'])
    allocation = next((a for a in owned if a['job_id'] == job and a['host'] == host), None)
    if allocation is None:
        raise ValueError('execution host not owned or allocation expired')
    gpu_text = clusterb.command(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                               '--format=csv,noheader,nounits'])
    apps = clusterb.command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'])
    if gpu not in clusterb.parse_idle_gpus(gpu_text, apps):
        raise ValueError('requested GPU is no longer idle')
    os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), PBS_JOBID=job, ASYMVLM_V13_IDLE_LEASE=config['sha256'])
    dispatch(config, min(seconds, allocation['seconds']), f'v13-{job}-{host}-{gpu}')


def clusterb_tick(config, config_path):
    count, terminal = ready(config)
    record = {'time': time.time(), 'ready_tasks': count, 'all_terminal': terminal, 'launched': []}
    if not count:
        return record
    owned = clusterb.owned_allocations(clusterb.query_allocations(config['user']), config['user'],
                                     time.time(), config['deadline_utc_epoch'])
    sessions = clusterb.tmux_sessions()
    logs = Path(config['study_root']) / 'coordinator/logs'
    logs.mkdir(parents=True, exist_ok=True)
    for allocation in owned:
        if not count:
            break
        for gpu in clusterb.idle_gpus(allocation['host']):
            name = f'mlsys-v13-{allocation["host"]}-gpu{gpu}'
            if name in sessions or not count:
                continue
            remote = 'bash ' + shlex.join([str(Path(config['cwd']) / 'scripts/mlsys_compression_v13_idle_worker.sh'),
                allocation['job_id'], str(gpu), str(allocation['seconds']), str(config_path)])
            command = ['ssh', *clusterb.SSH_OPTIONS, allocation['host'], remote]
            log = logs / (name + '-' + str(int(time.time())) + '.log')
            clusterb.command(['tmux', 'new-session', '-d', '-s', name,
                'exec ' + shlex.join(command) + ' >> ' + shlex.quote(str(log)) + ' 2>&1'])
            record['launched'].append({'session': name, 'job': allocation['job_id'], 'gpu': gpu, 'log': str(log)})
            count -= 1
    return record


def admission(payload, queue_text, user):
    from .compression_clustera_v11a import normalize_inventory
    result = clustera_queue_policy_v5.admission(normalize_inventory(payload), queue_text, user)
    nodes = sum(j['nodes'] for key in ('running_jobs', 'queued_jobs', 'other_jobs') for j in result[key])
    return {**result, 'eligible': nodes == 0, 'reserved_nodes_all_workflows': nodes,
            'reason': 'eligible' if nodes == 0 else 'waiting_for_other_debug_workflow'}


def observe_clustera(config):
    jobs = submissions.execute(['qstat', '-x', '-f', '-F', 'json'], timeout=45)
    queue = submissions.execute(['qstat', '-Qf', 'debug'], timeout=30)
    if jobs['returncode'] != 0 or queue['returncode'] != 0:
        raise ValueError('cannot verify debug inventory and enforced node limits')
    payload = json.loads(jobs['stdout'])
    return payload, admission(payload, queue['stdout'], config['user'])


def clustera_tick(config, journal, journal_path, *, dry_run=False):
    payload, decision = observe_clustera(config)
    now = time.time()
    result = {'time': now, 'admission': decision, 'status': 'waiting_for_ready_work_or_debug_queue'}
    if journal.get('submission_intent'):
        matches = submissions.nonce_matches(payload, config, journal['submission_intent'])
        if not matches:
            return {**result, 'status': 'uncertain_submission_requires_audit'}
        if not dry_run:
            submissions.append_accepted(journal, matches[0], reconciled=True)
            atomic_json(journal_path, journal)
    for accepted in journal['submissions']:
        job = payload['Jobs'].get(accepted['job'])
        if job is None:
            return {**result, 'status': 'accepted_job_missing_requires_audit'}
        if (job.get('Job_Owner', '').split('@')[0] != config['user']
                or job.get('Job_Name') != JOB_NAME):
            raise ValueError('accepted job ownership changed')
        if job.get('job_state') not in submissions.TERMINAL:
            return {**result, 'status': 'own_allocation_outstanding', 'job': accepted['job']}
    count, terminal = ready(config)
    result.update(ready_tasks=count, all_terminal=terminal)
    if terminal:
        return {**result, 'status': 'complete'}
    if (len(journal['submissions']) >= config['maximum_submissions']
            or journal.get('total_rejections_v2', 0) >= submissions.MAX_REJECTIONS):
        return {**result, 'status': 'submission_budget_exhausted'}
    if now + 3600 >= config['deadline_utc_epoch']:
        return {**result, 'status': 'insufficient_study_lifetime'}
    if not count or not decision['eligible'] or now < journal.get('retry_after_epoch', 0):
        return result
    if dry_run:
        return {**result, 'status': 'eligible_not_submitted'}
    # A second complete inventory check immediately precedes submission.
    _, fresh = observe_clustera(config)
    if not fresh['eligible']:
        return {**result, 'admission': fresh}
    outcome = submissions.submit(config, journal, journal_path)
    return {**result, 'status': outcome, 'submissions': journal['submissions']}


def watch(config_path, *, once=False, dry_run=False):
    config = read_config(config_path)
    role = config['coordinator_role']
    if not socket.gethostname().startswith(role + '-login-'):
        raise ValueError('watcher must run in the authenticated site login session')
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
                    result = clustera_tick(config, journal, journal_path, dry_run=dry_run)
                elif dry_run:
                    result = {'ready_tasks': ready(config)[0], 'status': 'dry_run_no_launch'}
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
    p.add_argument('--worker-id', default='v13-worker')
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

