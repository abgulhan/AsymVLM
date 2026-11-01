"""Frozen, restartable manuscript campaign, independent of research checkpoints.

No absolute submission-time inference deadline: every PBS allocation starts its
own walltime clock. Production admission requires a real two-node debug receipt.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from .manuscript_data_v35 import ROOT, family
from .state import atomic_json, digest, file_digest
from .followup_common import write_immutable

MODELS = ('qwen25', 'llava_next', 'phi35')
METHODS = ('all_tokens', 'divprune', 'cdpruner', 'asymvlm_uniform')
JOB_NAME = 'mlsys-manuscript-v35'
PACKAGES = ('torch', 'transformers', 'accelerate', 'numpy', 'safetensors', 'Pillow', 'pyarrow')


def read(path):
    return json.loads(Path(path).read_text())


def frozen_copy(source, destination):
    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = file_digest(source)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix+'.tmp')
        shutil.copyfile(source, temporary)
        if file_digest(temporary) != expected:
            raise ValueError('source changed during snapshot: '+str(source))
        temporary.replace(destination)
    if file_digest(destination) != expected:
        raise ValueError('existing frozen bundle differs; preserve and version it')
    return expected


def setup(project):
    project = Path(project).resolve()
    root = project/ROOT
    config_path = root/'campaign.json'
    if config_path.exists():
        return read(config_path)
    manifest = read(root/'data/manifest.json')
    prior = read(project/'reproduction_results/mlsys_conservative_budget_v33_20261028/protocol.json')
    bundle = root/'bundle'
    sources = {}
    paths = list((project/'learned_score').rglob('*.py')) + list(project.glob('*.py'))
    # Only source files: no datasets, model weights, caches, or previous artifacts.
    for path in sorted(paths):
        relative = path.relative_to(project)
        sources[str(relative)] = frozen_copy(path, bundle/relative)
    for name in ('mlsys_manuscript_v35_worker.sh', 'mlsys_manuscript_v35_debug.pbs',
                 'mlsys_manuscript_v35_prod.pbs'):
        sources['scripts/'+name] = frozen_copy(project/'scripts'/name, bundle/'scripts'/name)
    vlm = (project/'../incoming/asymvlm-full-drive/StreamingVLM/VLMEvalKit/vlmeval').resolve()
    if not vlm.is_dir():
        raise FileNotFoundError('validated VLMEvalKit package unavailable')
    for path in sorted(vlm.rglob('*.py')):
        relative = Path('vlmeval')/path.relative_to(vlm)
        sources[str(relative)] = frozen_copy(path, bundle/relative)
    models = json.loads(json.dumps(prior['models']))
    pins = {}
    for model, spec in models.items():
        for key in ('checkpoint', 'reference_checkpoint'):
            if not spec.get(key):
                continue
            old = Path(spec[key]); target = root/'checkpoints'/model/key/old.name
            pins[str(target)] = frozen_copy(old, target)
            complete = old.parent/'complete.json'
            if complete.exists():
                pins[str(target.parent/'complete.json')] = frozen_copy(complete, target.parent/'complete.json')
            spec[key] = str(target)
        if file_digest(spec['checkpoint']) != spec['checkpoint_sha256']:
            raise ValueError('scorer changed before freeze')
        model_dir = Path(spec['model_path'])
        weights = sorted(model_dir.glob('*.safetensors')) + sorted(model_dir.glob('pytorch_model*.bin'))
        if not weights:
            raise FileNotFoundError('offline model weights missing: '+model)
        spec['weight_inventory'] = {p.name: dict(bytes=p.stat().st_size, blob=p.resolve().name) for p in weights}
        for name in ('config.json', 'generation_config.json', 'preprocessor_config.json', 'tokenizer_config.json'):
            if (model_dir/name).exists():
                pins[str(model_dir/name)] = file_digest(model_dir/name)
    pins[str(root/'data/manifest.json')] = file_digest(root/'data/manifest.json')
    training = str(prior['scorer_training_manifest'])
    pins[training] = file_digest(training)
    # This hash excludes scorer/predictor checkpoints. Baselines remain reusable.
    baseline_files = {p:h for p,h in sources.items() if p.startswith((
        'learned_score/mlsys_runtime/', 'learned_score/multimodel/adapters/', 'vlmeval/')) or p in (
        'learned_score/mlsys_sprint/manuscript_execute_v35.py',
        'learned_score/mlsys_sprint/stable_divprune_v15c.py',
        'learned_score/mlsys_sprint/stable_cdpruner.py',
        'learned_score/mlsys_sprint/phi_adaptive_v3.py',
        'learned_score/mlsys_sprint/phi_adaptive_v3_compat.py',
        'learned_score/mlsys_sprint/phi_adaptive_v4.py')}
    now = time.time()
    # The reporting cutoff is a fixed campaign setting, not a
    # moving window reset by setup/restart. PBS compute windows are separate.
    campaign_start = datetime(2026,10,28,20,18,tzinfo=timezone.utc).timestamp()
    config = dict(format='mlsys_manuscript_v35', cwd=str(project), output_root=str(root),
        bundle=str(bundle), bundle_hash=digest(sources), source_hashes=sources,
        models=models, manifest=str(root/'data/manifest.json'), scorer_training_manifest=training,
        pins=pins, baseline_implementation_hash=digest(baseline_files),
        packages={p:importlib.metadata.version(p) for p in PACKAGES},
        created=now, initial_report_cutoff=campaign_start+4*3600, research_cutoff=campaign_start+90*60,
        maximum_production_submissions=2, production_nodes=10, production_walltime=10800,
        debug_nodes=2, debug_walltime=3600, user=os.environ.get('ASYMVLM_PBS_USER', 'user'), project=os.environ.get('ASYMVLM_PBS_PROJECT', 'project'),
        generation=dict(max_new_tokens=128, generated_budget=512, greedy=True,
            position_policy='original', text_eviction=False, precision='bfloat16'),
        methods=list(METHODS), source_coverage=manifest['metadata']['coverage'],
        baseline_fidelity=dict(divprune='native projected-feature adaptation; not upstream end-to-end',
            cdpruner='native projected query relevance and stable FP64 DPP adaptation; not upstream CLIP pipeline',
            visionzip='unsupported_pending_native_runtime_port',
            fastv='not_admitted', sparsevlm='not_admitted', pyramiddrop='legacy_LLaVA_only_not_admitted',
            atp_llava='trained_policy_not_available_not_admitted'),
        prior_results_untouched=True, baseline_keys_exclude_asymvlm_checkpoints=True)
    config['sha256'] = digest(config)
    tasks = []
    for model in MODELS:
        for dataset in sorted({family(r['dataset']) for r in manifest['splits']['evaluation']}):
            entries = sorted((r for r in manifest['splits']['evaluation'] if family(r['dataset'])==dataset),
                             key=lambda r:r['panel_order'])
            for method in METHODS:
                for percent in ((100,) if method=='all_tokens' else (70,50,30,90)):
                    for start in range(0,len(entries),32):
                        stage = 0 if start<256 else 1 if start<512 else 3 if start<1024 else 4
                        priority = stage + (0 if percent in (70,100) else 2 if percent==50 else 10)
                        task = dict(model=model,dataset=dataset,method=method,percent=percent,
                            request_ids=[r['row_hash'] for r in entries[start:start+32]],
                            panel_start=start,priority=priority)
                        task['id'] = digest(task)
                        tasks.append(task)
    tasks.sort(key=lambda t:(t['priority'],t['panel_start'],t['dataset'],t['model'],t['method']))
    write_immutable(root/'tasks.json', dict(campaign_sha256=config['sha256'],tasks=tasks))
    write_immutable(config_path, config)
    for directory in ('states','locks','workers','logs','raw','records','preflight','submissions','reports'):
        (root/directory).mkdir(parents=True,exist_ok=True)
    print(dict(event='prepared',root=str(root),tasks=len(tasks),requests=len(manifest['splits']['evaluation']),
               initial_report_utc=datetime.fromtimestamp(config['initial_report_cutoff'],timezone.utc).isoformat()),flush=True)
    return config


def verify(config):
    value = {k:v for k,v in config.items() if k!='sha256'}
    if digest(value)!=config['sha256']:
        raise ValueError('campaign configuration was modified')
    bundle = Path(config['bundle'])
    for name, expected in config['source_hashes'].items():
        if file_digest(bundle/name)!=expected:
            raise ValueError('frozen source changed: '+name)
    for name, expected in config['pins'].items():
        if file_digest(name)!=expected:
            raise ValueError('frozen dependency changed: '+name)
    if {p:importlib.metadata.version(p) for p in PACKAGES} != config['packages']:
        raise ValueError('Python environment changed after freeze')
    for spec in config['models'].values():
        for name, expected in spec['weight_inventory'].items():
            p=Path(spec['model_path'])/name
            if dict(bytes=p.stat().st_size,blob=p.resolve().name)!=expected:
                raise ValueError('offline model inventory changed')
    if Path(__file__).resolve().is_relative_to(bundle) is False:
        raise ValueError('workers must import the frozen bundle, not mutable research sources')


def rank_layout(node_count, rank, size, local_rank):
    if (type(node_count) is not int or node_count not in (1,2,10)
        or size != node_count*4 or not 0<=rank<size or not 0<=local_rank<4):
        raise ValueError('allocation/rank dimensions do not match four GPUs per node')
    return 3-local_rank


def allocation_deadline(start, walltime, now):
    if not 300<=walltime<=10800 or not 0<=now-start<walltime-120:
        raise ValueError('allocation has no usable remaining walltime')
    return start+walltime-120


def hms(value):
    parts = list(map(int,str(value).split(':')))
    if len(parts)!=3 or min(parts)<0:
        raise ValueError('invalid scheduler walltime')
    return parts[0]*3600+parts[1]*60+parts[2]


def allocation(config, cluster, gpu=None, job_id=None):
    host=socket.gethostname().split('.')[0]
    job_id=job_id or os.environ.get('PBS_JOBID','')
    expected = r'\d+\.clusterb-pbs-\d+\.lab\.hpc\.anl\.gov' if cluster=='clusterb' else r'\d+\.clustera-pbs-\d+\.hsn\.cm\.clustera\.hpc\.anl\.gov'
    if not re.fullmatch(expected,job_id):
        raise ValueError('explicit scheduler job identity is required')
    payload=json.loads(subprocess.check_output(['qstat','-f','-F','json',job_id],text=True,timeout=30))
    job=payload['Jobs'][job_id];resources=job['Resource_List']
    hosts={p.split('/')[0].split('.')[0] for p in job.get('exec_host','').split('+') if p}
    if (job['Job_Owner'].split('@')[0]!=config['user'] or job['job_state']!='R'
        or host not in hosts or job.get('Account_Name')!=config['project']):
        raise ValueError('job ownership, account, running state, or node mismatch')
    wall=hms(resources['walltime']);used=hms(job['resources_used']['walltime'])
    now=time.time();remaining=wall-used-180
    if cluster=='clusterb':
        if (not re.fullmatch(r'clusterb-gpu-\d+',host) or gpu not in range(8)
            or str(job.get('interactive')).lower() not in ('true','1') or len(hosts)!=1):
            raise ValueError('ClusterB requires a named user-owned interactive node')
        deadline=min(now+remaining,config['initial_report_cutoff']-120)
        rank=gpu;size=8
    else:
        if job['Job_Name']!=JOB_NAME or len(hosts) not in (2,10):
            raise ValueError('unexpected ClusterA allocation')
        rank=int(os.environ['PMI_RANK']);size=int(os.environ['PMI_SIZE'])
        local=int(os.environ['PMI_LOCAL_RANK'])
        gpu=rank_layout(len(hosts),rank,size,local)
        maximum=config['debug_walltime'] if len(hosts)==2 else config['production_walltime']
        if wall>maximum:
            raise ValueError('walltime exceeds approved allocation')
        deadline=allocation_deadline(now-used,wall,now)-60
    if deadline-now<180:
        raise ValueError('allocation too close to expiry')
    return dict(host=host,gpu=gpu,job_id=job_id,rank=rank,size=size,deadline=deadline,
                cluster=cluster,node_count=len(hosts),walltime=wall)


@contextmanager
def claim(config, model, preference='any'):
    root=Path(config['output_root']);tasks=read(root/'tasks.json')['tasks']
    candidates=[t for t in tasks if t['model']==model]
    candidates.sort(key=lambda t:(t['priority'],
        0 if preference=='any' or (t['method']=='asymvlm_uniform')==(preference=='asymvlm') else 1,
        t['panel_start'],t['dataset']))
    held=None
    try:
        for task in candidates:
            state_path=root/'states'/(task['id']+'.json')
            if state_path.exists() and read(state_path).get('status') in ('complete','failed'):
                continue
            handle=(root/'locks'/(task['id']+'.lock')).open('a+')
            try:
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                handle.close();continue
            state=read(state_path) if state_path.exists() else {}
            if state.get('status') in ('complete','failed'):
                handle.close();continue
            attempts=state.get('unexpected_failures',0)+(state.get('status')=='running')
            if attempts>=2:
                atomic_json(state_path,{**state,'status':'failed','unexpected_failures':attempts,
                    'error':'worker disappeared on both attempts','updated':time.time()})
                handle.close();continue
            state=dict(status='running',unexpected_failures=attempts,updated=time.time(),
                host=socket.gethostname(),pid=os.getpid(),task=task)
            atomic_json(state_path,state);held=handle
            yield task,state_path,state
            return
        yield None
    finally:
        if held is not None:
            held.close()


def debug_ready(config):
    root=Path(config['output_root']);receipts=[]
    for p in (root/'preflight').glob('clustera-*.json'):
        r=read(p)
        if r.get('bundle_hash')==config['bundle_hash'] and r.get('passed') and r.get('node_count')==2:
            receipts.append(r)
    by_job={}
    for r in receipts:
        by_job.setdefault(r['job_id'],[]).append(r)
    for job, rows in by_job.items():
        if ({r['rank'] for r in rows}==set(range(8)) and len({(r['host'],r['gpu']) for r in rows})==8
            and len({r['host'] for r in rows})==2 and {r['model'] for r in rows}==set(MODELS)
            and all(set(r['methods'])==set(METHODS) for r in rows)):
            return dict(passed=True,job_id=job,workers=8,hosts=sorted({r['host'] for r in rows}),
                        bundle_hash=config['bundle_hash'],physical_canaries_verified=True)
    return None


def submit(config, queue):
    if not socket.gethostname().startswith('clustera-login-'):
        raise ValueError('submit from the dedicated ClusterA login session')
    root=Path(config['output_root'])
    with (root/'submissions'/'.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        payload=json.loads(subprocess.check_output(['qstat','-f','-F','json'],text=True,timeout=45))
        jobs=payload.get('Jobs',{})
        owned={k:v for k,v in jobs.items() if v.get('Job_Owner','').split('@')[0]==config['user']
               and v.get('job_state') not in ('F','X')}
        if any(v.get('Job_Name')==JOB_NAME and (v.get('queue')=='debug')==(queue=='debug') for v in owned.values()):
            return dict(status='already_queued_or_running')
        if queue=='debug':
            if any(v.get('queue')=='debug' and v.get('job_state') not in ('F','X')
                   and (v.get('Account_Name')==config['project'] or v.get('Job_Owner','').split('@')[0]==config['user'])
                   for v in jobs.values()):
                return dict(status='waiting_for_shared_debug_capacity')
        else:
            ready=debug_ready(config)
            if not ready:
                return dict(status='waiting_for_physical_two_node_debug_validation')
            atomic_json(root/'preflight'/'production-admission.json',ready)
        receipts=list((root/'submissions').glob(queue+'-*.json'))
        limit=4 if queue=='debug' else config['maximum_production_submissions']
        if len(receipts)>=limit:
            return dict(status='approved_submission_bound_reached')
        if queue=='prod' and receipts:
            prior=read(sorted(receipts)[-1]);previous=prior.get('job_id')
            states=[read(p) for p in (root/'workers').glob('*.json') if read(p).get('job_id')==previous]
            if not states or not any(r.get('completed_tasks',0)>0 for r in states):
                return dict(status='previous_production_did_not_progress_manual_repair_required')
        for q in (('debug',) if queue=='debug' else ('prod','small')):
            audit=subprocess.check_output(['qstat','-Qf',q],text=True,timeout=30)
            if not re.search(r'enabled\s*=\s*True',audit) or not re.search(r'started\s*=\s*True',audit):
                return dict(status='queue_not_enabled',queue=q)
            atomic_json(root/'submissions'/('queue-'+q+'.json'),dict(time=time.time(),text=audit))
        receipt=root/'submissions'/f'{queue}-{len(receipts)+1:02d}.json'
        atomic_json(receipt,dict(status='submitting',time=time.time(),bundle_hash=config['bundle_hash']))
        script=Path(config['bundle'])/'scripts'/f'mlsys_manuscript_v35_{queue}.pbs'
        result=subprocess.run(['qsub','-v','ASYMVLM_V35_CONFIG='+str(root/'campaign.json'),str(script)],
            capture_output=True,text=True,timeout=60)
        value=dict(status='submitted' if result.returncode==0 else 'submission_failed',time=time.time(),
            job_id=result.stdout.strip(),stderr=result.stderr,returncode=result.returncode,bundle_hash=config['bundle_hash'])
        atomic_json(receipt,value)
        return value


def watch(config):
    """Bounded scheduler-side watcher; no LLM/model polling and no extra jobs."""
    root=Path(config['output_root']);deadline=time.time()+24*3600
    with (root/'submissions'/'watch.lock').open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.time()<deadline:
            tasks=read(root/'tasks.json')['tasks'];states=[read(p) for p in (root/'states').glob('*.json')]
            terminal=sum(s.get('status') in ('complete','failed') for s in states)
            if terminal==len(tasks):
                atomic_json(root/'submissions'/'watch-status.json',dict(status='queue_terminal',time=time.time()))
                return
            queue='prod' if debug_ready(config) else 'debug'
            if queue=='debug' and time.time()>config['initial_report_cutoff']:
                atomic_json(root/'submissions'/'watch-status.json',dict(status='debug_validation_not_ready_before_cutoff',time=time.time()))
                return
            status=submit(config,queue)
            atomic_json(root/'submissions'/'watch-status.json',dict(**status,time_checked=time.time(),queue=queue))
            print(status,flush=True)
            if status['status'] in ('approved_submission_bound_reached','previous_production_did_not_progress_manual_repair_required','submission_failed'):
                return
            time.sleep(45)


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('setup','submit-debug','submit-prod','watch','verify'))
    p.add_argument('--config',default=str(ROOT/'campaign.json'));a=p.parse_args()
    if a.phase=='setup':
        setup(Path.cwd());return
    config=read(a.config)
    if a.phase=='verify':verify(config)
    elif a.phase=='watch':watch(config)
    else:print(submit(config,a.phase.split('-')[1]),flush=True)


if __name__=='__main__':
    main()
