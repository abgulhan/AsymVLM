"""Physical manuscript workers using the verified native three-model runtime."""
import argparse
from contextlib import contextmanager
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import threading
import time
import traceback

from . import manuscript_campaign_v35 as campaign
from .manuscript_data_v35 import ROOT, family
from .state import atomic_json, digest, file_digest


def cache_key(config, model, entry, method, count):
    spec=config['models'][model]
    backbone={k:v for k,v in spec.items() if k not in
              ('checkpoint','reference_checkpoint','checkpoint_sha256','name')}
    value=dict(model=backbone,implementation=config['baseline_implementation_hash'],
        input=dict(image=entry['image_sha256'],question=entry['question_sha256'],
                   prompt='yes_no' if entry['metric']=='mme' else 'single_word_or_phrase'),
        generation=config['generation'],selector=method,selected_count=count)
    if method=='asymvlm_uniform':
        value['scorer_sha256']=spec['checkpoint_sha256']
    return digest(value)


def selector(method, count, scorer, observed):
    import torch
    from torch.nn import functional as F
    from .stable_divprune_v15c import stable_divprune_order
    from .stable_cdpruner import stable_cdpruner_order
    from .selector_experiment_v5 import validate_indices
    @torch.no_grad()
    def choose(prepared):
        if prepared.answer_positions.numel():
            raise ValueError('answers must not enter patch selection')
        visual=prepared.inputs_embeds[0,prepared.visual_positions]
        query=prepared.inputs_embeds[0,prepared.semantic_text_positions]
        n=len(visual)
        if not 1<=count<=n:
            raise ValueError('invalid physical retained count')
        if count==n:
            indices=torch.arange(n,device=visual.device,dtype=torch.long)
        elif method=='asymvlm_uniform':
            indices=torch.argsort(scorer(visual,query).float(),descending=True,stable=True)[:count].sort().values
        elif method=='divprune':
            indices=stable_divprune_order(visual,count).sort().values
        elif method=='cdpruner':
            if not len(query):
                raise ValueError('query-conditioned baseline requires question tokens')
            relevance=(F.normalize(visual.float(),dim=-1)@F.normalize(query.float(),dim=-1).T).max(-1).values
            indices=stable_cdpruner_order(visual,relevance,count).sort().values
        else:
            raise ValueError('unsupported method, never silently substitute')
        validate_indices(indices,count,n)
        observed.update(input_visual_tokens=n,input_text_tokens=int(prepared.inputs_embeds.shape[1]-n),
                        retained_visual_tokens=count,selected_indices_sha256=digest(indices.cpu().tolist()))
        return indices
    return choose


def score(prediction, references, metric):
    from .adaptive_experiment_v3 import score_prediction
    if metric=='exact':
        # OCRVQA's exact-answer metric, without substring matching or an LLM judge.
        return float(any(prediction.strip().lower()==r.strip().lower() for r in references))
    return score_prediction(prediction,references,metric)


class Runtime:
    def __init__(self,config,model):
        from .budget_execute_v31 import load_runtime
        from .adaptive_data_v3 import ManifestDatasetV3
        from learned_score.mlsys_scorer.backbone import frozen_versions
        self.config,self.model=config,model
        self.adapter,_,self.inputs,self.backbone,self.scorer,self.manifest=load_runtime(config,model,'evaluation')
        # Explicit live-data root is essential when importing a frozen source bundle.
        self.labels=ManifestDatasetV3(config['manifest'],allowed_splits=('evaluation',),project_root=config['cwd'])
        self.entries={r['row_hash']:r for r in self.manifest.splits['evaluation']}
        self.versions=frozen_versions(self.adapter.model)
        self.scorer_versions=frozen_versions(self.scorer)
        from .adaptive_experiment_v3 import contract_from_config
        self.contract=contract_from_config(self.adapter.model.config)

    def geometry(self,entry,save_feature=False):
        import torch
        from .budget_predictor_v31 import feature_inputs
        self.adapter.dataset=self.inputs
        raw=self.inputs.row_for_entry(entry)
        with torch.no_grad():
            prepared=self.adapter.prepare_sample(raw,include_answer=False)
            if prepared.answer_positions.numel():
                raise ValueError('answer leakage in native preparation')
            n=len(prepared.visual_positions);p=int(prepared.inputs_embeds.shape[1]-n)
            feature=dict(request_id=entry['row_hash'],group_id=entry['group_id'],dataset=entry['dataset'],
                input_visual_tokens=n,input_text_tokens=p,cache_contract=self.contract)
            if save_feature:
                v=prepared.inputs_embeds[0,prepared.visual_positions]
                q=prepared.inputs_embeds[0,prepared.semantic_text_positions]
                feature.update(feature_inputs(self.scorer,v,q,p))
                feature['checkpoint_sha256']=self.config['models'][self.model]['checkpoint_sha256']
                feature['image_sha256']=entry['image_sha256']
                path=Path(self.config['output_root'])/'features'/self.model/(entry['row_hash']+'.json')
                atomic_json(path,feature)
        return raw,prepared,feature

    def run(self,entry,method,percent,*,preflight=False):
        import torch
        from . import adaptive_experiment_v3 as old
        from . import adaptive_experiment_v4 as physical
        from learned_score.mlsys_scorer.backbone import assert_frozen
        root=Path(self.config['output_root'])
        raw,prepared,feature=self.geometry(entry,save_feature=method=='asymvlm_uniform')
        n=feature['input_visual_tokens'];count=n if method=='all_tokens' else (n*percent+99)//100
        key=cache_key(self.config,self.model,entry,method,count)
        target=root/'raw'/key[:2]/(key+'.json')
        del prepared
        expected=physical.expected_cache_bytes(feature,count,self.contract)
        if target.exists() and not preflight:
            saved=campaign.read(target)
            if saved.get('cache_key')!=key or saved.get('selected_count')!=count or saved.get('status')!='ok':
                raise ValueError('invalid raw generation cache')
        else:
            observed={}
            choose=selector(method,count,self.scorer,observed)
            self.adapter.dataset=self.inputs
            cell=old.Cell('scorer' if method=='asymvlm_uniform' else method,
                          'full' if method=='all_tokens' else 'uniform',percent)
            torch.cuda.reset_peak_memory_stats()
            prediction,ids,timing,metrics=old.measured(self.adapter,raw,choose,cell,128)
            metrics=physical.generation_trace(self.adapter,ids,metrics)
            old.check_cache(metrics,expected)
            if observed.get('retained_visual_tokens')!=count:
                raise ValueError('physical generation did not use the requested count')
            saved=dict(format='manuscript_raw_v35',cache_key=key,status='ok',
                model=self.model,method=method,selected_count=count,**observed,
                prediction=prediction,generated_token_ids=ids[0].tolist(),generated_tokens=int(ids.shape[-1]),
                timing=timing,metrics=metrics,expected_cache_bytes=expected,
                full_cache_bytes=physical.expected_cache_bytes(feature,n,self.contract),
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                hardware=torch.cuda.get_device_name(),physical_measurement_performed=True,
                timing_isolated=False,systems_evidence='concurrent_quality_run_not_headline_latency',
                baseline_fidelity=self.config['baseline_fidelity'].get(method,'frozen AsymVLM scorer or all tokens'),
                time=time.time(),host=campaign.socket.gethostname(),job_id=os.environ.get('PBS_JOBID'),
                allocation_job_id=os.environ.get('ASYMVLM_V35_JOB'),bundle_hash=self.config['bundle_hash'])
            if not preflight:
                atomic_json(target,saved)
        # Labels are read only after answer-free generation/selection.
        labeled=self.labels.row_for_entry(entry)
        if labeled['question']!=raw['question']:
            raise ValueError('answer-free and labeled input questions disagree')
        refs=old.benchmark.parse_references(labeled['answer'])
        record=dict(request_id=entry['row_hash'],dataset=family(entry['dataset']),group_id=entry['group_id'],
            category=entry.get('category'),model=self.model,method=method,percent=percent,
            panel_order=entry['panel_order'],cache_key=key,raw_path=str(target),
            quality_score=score(saved['prediction'],refs,entry['metric']),metric=entry['metric'],
            evaluator_sha256=file_digest(__file__),references_sha256=digest(refs),
            retained_visual_tokens=count,input_visual_tokens=n,kv_bytes=saved['expected_cache_bytes'],
            full_kv_bytes=saved['full_cache_bytes'],status='ok',timing_isolated=False)
        assert_frozen(self.adapter.model,self.versions);assert_frozen(self.scorer,self.scorer_versions)
        if not preflight:
            atomic_json(root/'records'/self.model/record['dataset']/method/str(percent)/(entry['row_hash']+'.json'),record)
        return record

    def canary(self,allocation):
        from .compression_execution_v16 import _canary
        candidates=sorted(self.entries.values(),key=lambda r:(family(r['dataset'])!='TextVQA',r['panel_order']))
        # Distinct physical calls on every worker, no cache substitution.
        entry=candidates[allocation['rank']%min(len(candidates),32)]
        raw,prepared,feature=self.geometry(entry)
        proofs=_canary(self.adapter,self.scorer,prepared,feature)
        del prepared,raw
        methods={m:self.run(entry,m,100 if m=='all_tokens' else 70,preflight=True) for m in campaign.METHODS}
        return dict(**allocation,model=self.model,passed=True,methods=list(methods),outcomes=methods,
                    parity=proofs,bundle_hash=self.config['bundle_hash'],time=time.time())


def worker(config,cluster,gpu=None,job=None,canary_only=False):
    campaign.verify(config)
    alloc=campaign.allocation(config,cluster,gpu,job)
    os.environ['CUDA_VISIBLE_DEVICES']=str(alloc['gpu'])
    os.environ['ASYMVLM_V35_JOB']=alloc['job_id']
    from .coverage_audit_resume_clustera_worker import gpu_lease
    from .attention_idle_v19b import gpu_idle
    root=Path(config['output_root'])
    worker_id=f"{cluster}-{alloc['job_id'].split('.')[0]}-{alloc['host']}-{alloc['gpu']}"
    status_path=root/'workers'/(worker_id+'.json')
    state=dict(**alloc,status='starting',completed_tasks=0,completed_examples=0,
               last_progress=time.time(),updated=time.time(),bundle_hash=config['bundle_hash'])
    atomic_json(status_path,state)
    active={};stopping=threading.Event();guard=threading.Lock()
    def heartbeat():
        while not stopping.wait(15):
            with guard:
                state['updated']=time.time();atomic_json(status_path,state)
                expired=time.time()>=alloc['deadline']
                stalled=time.time()-state['last_progress']>900
                if expired or stalled:
                    state['status']='allocation_drained' if expired else 'stalled_worker'
                    atomic_json(status_path,state)
                    if active.get('path') and expired:
                        atomic_json(active['path'],{**active['state'],'status':'paused',
                            'reason':'allocation_walltime','updated':time.time()})
                    os._exit(0 if expired else 75)
    signal.signal(signal.SIGTERM,lambda *_:stopping.set())
    signal.signal(signal.SIGINT,lambda *_:stopping.set())
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        with gpu_lease(alloc['host'],alloc['gpu']):
            if not gpu_idle(alloc['gpu'],allow_cooldown=False):
                state['status']='yielded_occupied_gpu';return
            preferred=campaign.MODELS[alloc['rank']%3]
            runtime=Runtime(config,preferred)
            receipt=runtime.canary(alloc)
            atomic_json(root/'preflight'/(worker_id+'.json'),receipt)
            state.update(status='running',model=preferred,last_progress=time.time())
            if canary_only:
                state['status']='canary_complete';return
            failures=0
            while not stopping.is_set() and time.time()<alloc['deadline']-90:
                did_work=False
                choices=(preferred,)+tuple(m for m in campaign.MODELS if m!=preferred)
                for model in choices:
                    preference='asymvlm' if alloc['rank']%5==4 else 'baseline'
                    with campaign.claim(config,model,preference) as claimed:
                        if claimed is None:
                            continue
                        task,path,task_state=claimed;did_work=True
                        with guard:
                            active.update(path=path,state=task_state)
                        try:
                            if runtime.model!=model:
                                del runtime;gc.collect()
                                import torch
                                torch.cuda.empty_cache()
                                runtime=Runtime(config,model)
                            state.update(model=model,task_id=task['id'])
                            interrupted=False
                            for rid in task['request_ids']:
                                if stopping.is_set() or time.time()>=alloc['deadline']-60:
                                    interrupted=True;break
                                record_path=root/'records'/model/task['dataset']/task['method']/str(task['percent'])/(rid+'.json')
                                if not record_path.exists():
                                    runtime.run(runtime.entries[rid],task['method'],task['percent'])
                                    state['completed_examples']+=1
                                state['last_progress']=time.time()
                            if interrupted:
                                atomic_json(path,{**task_state,'status':'paused','updated':time.time()})
                            else:
                                # Completion is published only after every exact request record exists.
                                records=[root/'records'/model/task['dataset']/task['method']/str(task['percent'])/(r+'.json') for r in task['request_ids']]
                                atomic_json(path,{**task_state,'status':'complete','updated':time.time(),
                                    'record_hashes':{str(p):file_digest(p) for p in records}})
                                state['completed_tasks']+=1;failures=0
                        except Exception as exc:
                            failures+=1;attempts=task_state['unexpected_failures']+1
                            atomic_json(path,{**task_state,'status':'failed' if attempts>=2 else 'retry',
                                'unexpected_failures':attempts,'error':str(exc),'traceback':traceback.format_exc(),
                                'updated':time.time()})
                            print(traceback.format_exc(),flush=True)
                            if failures>=2:
                                state['status']='stopped_repeated_worker_error';return
                        finally:
                            with guard:
                                active.clear();atomic_json(status_path,state)
                    preferred=model
                    break
                if not did_work:
                    # No runnable claims means completed/failed/claimed by others; release this GPU.
                    state['status']='no_runnable_work';return
            state['status']='allocation_drained'
    except Exception as exc:
        state.update(status='startup_failed',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        stopping.set();state['updated']=time.time();atomic_json(status_path,state)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True)
    p.add_argument('--cluster',choices=('clusterb','clustera'),required=True)
    p.add_argument('--gpu',type=int);p.add_argument('--job');p.add_argument('--canary-only',action='store_true')
    a=p.parse_args();config=campaign.read(a.config)
    os.chdir(config['cwd'])
    worker(config,a.cluster,a.gpu,a.job,a.canary_only)


if __name__=='__main__':
    main()
