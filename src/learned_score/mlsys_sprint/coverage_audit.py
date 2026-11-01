"""Additive development-only coverage audit, separate from the live v16 study.

Prepare immutable panels; run CPU diagnostics or exact physical probes; report
only observed outcomes. This module never submits allocations or tunes on eval.
"""
import argparse
from collections import defaultdict
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import socket
import statistics
import time

from . import compression_execution_v16 as execution
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261025_v16/coverage_audit_v2')
STUDY = Path('reproduction_results/mlsys_compression_20261025_v16/study.json')
MODELS = ('qwen25','llava_next','phi35')
DOMAINS = ('DocVQA','TextVQA','ChartQA_TEST','MME')
METHODS = ('all_tokens','importance_050','coverage_greedy_050','importance_070',
           'coverage_greedy_070','importance_legacy_tau','importance_fine_tau','coverage_greedy_tau')
FORMAT = 'mlsys_coverage_audit_20261025'


def read(path):
    return json.loads(Path(path).read_text())


def family(name):
    low = name.lower()
    return next((domain for key,domain in [('docvqa','DocVQA'),('textvqa','TextVQA'),
        ('chartqa','ChartQA_TEST'),('mme','MME')] if key in low), name)


def panel(entries, groups_per_domain=8):
    groups=defaultdict(list)
    for entry in entries:
        if family(entry['dataset']) in DOMAINS:
            groups[(family(entry['dataset']),entry['group_id'])].append(entry)
    selected,counts=[],{}
    for domain in DOMAINS:
        keys=sorted((key for key in groups if key[0]==domain),key=lambda key:digest(['coverage-audit-panel-v1',key]))
        admitted=0
        for key in keys:
            rows=sorted(groups[key],key=lambda row:row['row_hash'])
            if domain=='MME' and len(rows)!=2:
                continue
            selected.extend(rows if domain=='MME' else rows[:1])
            admitted+=1
            if admitted==groups_per_domain:
                break
        counts[domain]={'source_groups':admitted,'missing_groups':groups_per_domain-admitted}
    return selected,counts


def prepare(project):
    from .compression_selection_v16 import incumbent_rules
    from .adaptive_workflow_v4 import environment
    project=Path(project).resolve(); root=project/ROOT
    manifest=root/'audit.json'
    if manifest.exists():
        return read(manifest)
    study=read(project/STUDY)
    rules,pins=incumbent_rules(study)
    pins[str((project/STUDY).resolve())]=file_digest(project/STUDY)
    for name in ('coverage_audit.py','coverage_audit_math.py','coverage_audit_operations.py','coverage_audit_report.py'):
        path=Path(__file__).with_name(name).resolve()
        pins[str(path)]=file_digest(path)
    script=project/'scripts/mlsys_coverage_audit_idle_worker.sh'
    pins[str(script)]=file_digest(script)
    plans,panels,tasks={},{},{}
    for model in MODELS:
        sources=[study]
        if model=='phi35':
            extra=Path(study['output_root'])/'supplemental-phi-study.json'
            if extra.exists():
                sources.append(read(extra))
        panels[model]=[]
        for source_index,source in enumerate(sources):
            entries=read(source['development_manifests'][model])['splits']['screening']
            chosen,counts=panel(entries)
            if not chosen:
                continue
            label=model+('-supplemental' if source_index else '')
            output=root/'physical'/label
            task=execution.build_task(source,'physical',model,'screening',output_root=output,
                request_ids=[e['row_hash'] for e in chosen],retentions=(50,70,100),warmup=0)
            task.pop('sha256')
            task.update(audit_format=FORMAT, audit_methods=list(METHODS), audit_rule=rules[model],
                audit_panel_counts=counts, audit_geometry_ids=[e['row_hash'] for e in chosen[:8]],
                audit_label=label, deadline_utc_epoch=min(study['deadline_utc_epoch'],1792925100.))
            task['input_hashes'].update(pins)
            task=execution.sealed(task)
            task_path=root/'tasks'/(label+'.json'); write_immutable(task_path,task)
            task_records=[]
            # Small source-group shards; never split MME pairs between workers.
            groups=sorted({e['group_id'] for e in chosen})
            shard_count=min(8,len(groups))
            for rank in range(shard_count):
                task_records.append({'id':f'coverage-audit-{label}-{rank}','tag':'coverage-audit',
                    'resource':'gpu','model_tag':model,'depends_on':[],'minimum_start_seconds':600,
                    'argv':[str(project/'.conda/asymvlm/bin/python'),'-B','-u','-m',
                        'learned_score.mlsys_sprint.coverage_audit','physical','--task',str(task_path),
                        '--rank',str(rank),'--world-size',str(shard_count)],
                    'env':{**environment(project),'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2',
                           'OPENBLAS_NUM_THREADS':'1'},
                    'artifacts':[{'path':str(output/f'rank{rank}.complete.json'),'passed':True}]})
            plan=execution.sealed({'format':FORMAT+'_plan','cwd':str(project),'tasks':task_records,
                'input_hashes':{**task['input_hashes'],str(task_path):file_digest(task_path)},
                'unexpected_retries':1,'created_utc_epoch':time.time(),
                'deadline_utc_epoch':task['deadline_utc_epoch']})
            plan_path=root/'plans'/label/'plan.json';write_immutable(plan_path,plan)
            plans[label]=str(plan_path);tasks[label]=str(task_path)
            panels[model].append({'label':label,'domains':counts,'requests':len(chosen),
                'group_ids':groups,'request_ids':[e['row_hash'] for e in chosen]})
    value=execution.sealed({'format':FORMAT,'study':str(project/STUDY),'study_sha256':study['sha256'],
        'created_utc_epoch':time.time(),'deadline_utc_epoch':study['deadline_utc_epoch'],
        'stop_new_physical_utc_epoch':1792921500.,'physical_deadline_utc_epoch':1792925100.,
        'panels':panels,'plans':plans,'tasks':tasks,'rules':rules,'input_hashes':pins,
        'independent_confirmation':False,'protect_main_run':True,'maximum_idle_gpus':2,
        'scorer_training':False,'candidate_promotion':False})
    write_immutable(manifest,value)
    return value


def geometry_probe(vectors,scores,rule,*,seconds=60):
    import torch
    from .coverage_audit_math import Coverage,count_menu,first_feasible
    from .compression_stopping_v11 import learned_score_weights
    obj=Coverage(vectors,learned_score_weights(scores),dtype=torch.float32)
    order=torch.argsort(scores,descending=True,stable=True).cpu().tolist()
    legacy=sorted({(obj.n*p+99)//100 for p in rule['legacy_retention_percentages']})
    fine=obj.curve(order,count_menu(obj.n,1))
    tau=rule['threshold']
    support=rule['base_rule']['support_visual_tokens']
    fallback='visual_count_out_of_support' if not support[0]<=obj.n<=support[1] else ('full_only_threshold' if tau<0 else None)
    legacy_curve=[r for r in fine if r['count'] in legacy]
    coarse_count=obj.n if fallback else first_feasible(legacy_curve,tau)
    fine_count=obj.n if fallback else first_feasible(fine,tau)
    result={'input_visual_tokens':obj.n,'threshold':tau,'fallback_reason':fallback,
        'importance_legacy_count':coarse_count,'importance_fine_count':fine_count,
        'saved_retention_percentage_points':100*(coarse_count-fine_count)/obj.n,
        'legacy_percentages':rule['legacy_retention_percentages'],'fine_curve':fine,
        'curvature':obj.curvature(),'fixed_counts':{}}
    greedy=obj.greedy((obj.n*90+99)//100,seconds=seconds)
    for p in (50,70):
        k=(obj.n*p+99)//100
        base=order[:k];selected=greedy['order'][:k]
        result['fixed_counts'][str(p)]={'count':k,'importance_coverage':obj.value(base),
            'greedy_coverage':obj.value(selected),'importance_distortion':obj.distortion(base),
            'greedy_distortion':obj.distortion(selected),'importance_bound':obj.upper_bound(base,k),
            'greedy_bound':obj.upper_bound(selected,k),
            'selected_set_overlap':len(set(base)&set(selected))/k}
    result.update(greedy_selection_seconds=greedy['seconds'],gain_recomputations=greedy['gain_recomputations'])
    return result


def cpu(audit,root,limit=8):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from learned_score.mlsys_scorer.backbone import frozen_versions,assert_frozen
    from .compression_probe_v16 import TRAINING_CACHE
    from .coverage_audit_math import Coverage,exhaustive_frontier,certification_feasibility
    torch.set_num_threads(1)
    study=read(audit['study']);project=Path(study['cwd'])
    gen=torch.Generator().manual_seed(173)
    obj=Coverage(torch.randn(8,5,generator=gen),torch.ones(8)/8)
    synthetic={'format':FORMAT+'_synthetic','optimal_frontier':exhaustive_frontier(obj),
        'greedy':obj.greedy(8),'curvature':obj.curvature(),'evidence':'synthetic_not_VLM_accuracy'}
    # Runtime values are diagnostic, so preserve the first completed run.
    if not (root/'cpu/synthetic.json').exists(): write_immutable(root/'cpu/synthetic.json',synthetic)
    evaluation=read(study['evaluation_manifest'])
    feasibility=certification_feasibility()
    feasibility.update(evaluation_manifest=study['evaluation_manifest'],
        existing_evaluation_independent=evaluation['metadata'].get('independent_confirmation',False),
        untouched_calibration_verified=False,
        verdict='not_certifiable_from_current_examined_evaluation; independent calibration required')
    write_immutable(root/'cpu/certification.json',feasibility)
    for model in MODELS:
        scorer,trained=load_scorer(study['models'][model]['checkpoint'],device='cpu')
        scorer.eval().requires_grad_(False);versions=frozen_versions(scorer)
        paths=sorted((project/TRAINING_CACHE[model]).glob('*.pt'),key=lambda p:p.stem)[:limit]
        for path in paths:
            dest=root/'cpu'/model/(path.stem+'.json')
            if dest.exists():continue
            cached=torch.load(path,map_location='cpu',weights_only=True)
            if (cached.get('format')!='asymvlm_mlsys_saliency_v1' or cached.get('row_hash')!=path.stem
                    or cached.get('backbone')!=trained['identity']['backbone']
                    or cached.get('manifest_sha256')!=trained['identity']['manifest_sha256']):
                raise ValueError('training-cache provenance differs from scorer')
            value={'format':FORMAT+'_training_geometry','model':model,'request_id':path.stem,
                'dataset':cached['dataset'],'training_cache_not_evaluation':True,
                'answer_quality_measured':False,'scorer_checkpoint_sha256':file_digest(study['models'][model]['checkpoint']),
                'cache_sha256':file_digest(path),'source_sha256':file_digest(Path(__file__))}
            try:
                with torch.no_grad():
                    scores=scorer(cached['vision'],cached['question']).float()
                    projection=F.linear(F.normalize(cached['vision'].float(),dim=-1),scorer.visual.weight.float())
                    value.update(status='ok',**geometry_probe(projection,scores,audit['rules'][model]))
            except TimeoutError as exc:
                value.update(status='selection_timeout',error=str(exc))
            assert_frozen(scorer,versions)
            write_immutable(dest,value)
            print(json.dumps({'event':'coverage_cpu_probe','model':model,'request':path.stem,'status':value['status']}),flush=True)
        del scorer
    return report(root)


def make_selector(method,scorer,rule,observed):
    import torch
    from torch.nn import functional as F
    from .coverage_audit_math import Coverage,count_menu,first_feasible
    from .compression_stopping_v11 import learned_score_weights
    from .selector_experiment_v5 import validate_indices
    if method not in METHODS:raise ValueError('undeclared audit method')
    @torch.no_grad()
    def select(prepared):
        if prepared.inputs_embeds.is_cuda:torch.cuda.synchronize()
        start=time.monotonic()
        visual=prepared.inputs_embeds[0,prepared.visual_positions]
        query=prepared.inputs_embeds[0,prepared.semantic_text_positions]
        n=len(visual);fallback=None;risk=None
        if method=='all_tokens':
            selected=list(range(n))
        else:
            scores=scorer(visual,query).float()
            if not bool(torch.isfinite(scores).all()):
                if not method.endswith('_tau'):
                    raise ValueError('invalid scorer cannot produce a matched-count comparison')
                indices=torch.arange(n,device=visual.device)
                observed.update(selected_count=n,input_visual_tokens=n,
                    input_text_tokens=int(prepared.inputs_embeds.shape[1]-n),
                    selected_patch_indices=indices.cpu().tolist(),selected_patch_indices_sha256=digest(indices.cpu().tolist()),
                    visual_retention=1.,coverage_deficit=None,fallback_reason='invalid_scorer',
                    selector_seconds=time.monotonic()-start,allocation_latency_included=True)
                return indices
            order=torch.argsort(scores,descending=True,stable=True).cpu().tolist()
            fixed=method.endswith('_050') or method.endswith('_070')
            count=(n*int(method[-3:])+99)//100 if fixed else None
            if method.startswith('importance_') and fixed:
                selected=order[:count]
            else:
                tau=rule['threshold'];support=rule['base_rule']['support_visual_tokens']
                if not fixed and (not support[0]<=n<=support[1] or tau<0):
                    fallback='visual_count_out_of_support' if tau>=0 else 'full_only_threshold'
                    selected=list(range(n))
                else:
                    projected=F.linear(F.normalize(visual.float(),dim=-1),scorer.visual.weight.float())
                    try:obj=Coverage(projected,learned_score_weights(scores),dtype=torch.float32)
                    except ValueError:
                        if fixed:raise
                        indices=torch.arange(n,device=visual.device)
                        observed.update(selected_count=n,input_visual_tokens=n,
                            input_text_tokens=int(prepared.inputs_embeds.shape[1]-n),
                            selected_patch_indices=indices.cpu().tolist(),selected_patch_indices_sha256=digest(indices.cpu().tolist()),
                            visual_retention=1.,coverage_deficit=None,fallback_reason='invalid_geometry',
                            selector_seconds=time.monotonic()-start,allocation_latency_included=True)
                        return indices
                    menu=(count_menu(n,1) if method=='importance_fine_tau' else
                          sorted({(n*p+99)//100 for p in rule['legacy_retention_percentages']}))
                    if method.startswith('coverage_greedy'):
                        maximum=count if fixed else max((k for k in menu if k<n),default=1)
                        greedy=obj.greedy(maximum,seconds=60.)
                        if not fixed:
                            curve=[{'count':k,'risk':0. if k==n else 1-greedy['coverage'][k-1]} for k in menu]
                            count=first_feasible(curve,tau)
                            risk=next(v['risk'] for v in curve if v['count']==count)
                        selected=greedy['order'][:count] if count<n else list(range(n))
                    else:
                        curve=obj.curve(order,menu,tau=tau)
                        count=curve[-1]['count'];risk=curve[-1]['risk']
                        selected=order[:count]
        indices=torch.tensor(sorted(selected),device=visual.device,dtype=torch.long)
        validate_indices(indices,len(selected),n)
        if visual.is_cuda:torch.cuda.synchronize()
        observed.update(selected_count=len(selected),input_visual_tokens=n,
            input_text_tokens=int(prepared.inputs_embeds.shape[1]-n),
            selected_patch_indices=indices.cpu().tolist(),selected_patch_indices_sha256=digest(indices.cpu().tolist()),
            visual_retention=len(selected)/n,coverage_deficit=risk,fallback_reason=fallback,
            selector_seconds=time.monotonic()-start,allocation_latency_included=True)
        return indices
    return select


def physical(task,rank,world_size):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions,assert_frozen
    from .adaptive_data_v3 import load_manifest
    from . import adaptive_experiment_v3 as prior
    from . import adaptive_experiment_v4 as runtime
    from . import phi_adaptive_v4 as phi
    execution.validate_task(task,'physical')
    if task.get('audit_format')!=FORMAT or task['split']!='screening' or task['audit_methods']!=list(METHODS):
        raise ValueError('fixed development-only audit task required')
    if not 0<=rank<world_size:raise ValueError('invalid shard')
    torch.set_num_threads(2)
    manifest=load_manifest(task['manifest'])
    entries=[e for e in manifest.splits['screening'] if e['row_hash'] in task['request_ids']]
    groups=sorted({e['group_id'] for e in entries})[rank::world_size]
    entries=sorted((e for e in entries if e['group_id'] in groups),key=lambda e:e['row_hash'])
    root=Path(task['output_root']);done=root/f'rank{rank}.complete.json'
    if done.exists():
        existing=read(done)
        if existing['task_sha256']!=task['sha256']:raise ValueError('changed completed task')
        for p,sha in existing['input_hashes'].items():
            if file_digest(p)!=sha:raise ValueError('changed completed physical result')
        return existing
    if time.time()>=task['deadline_utc_epoch']:raise TimeoutError('physical deadline reached')
    args=execution.runtime_args(task)
    with phi.installed() if args.adapter=='phi3_vision' else nullcontext():
        adapter,dataset,backbone,scorer=prior.load_runtime(args,manifest)
    scorer.eval().requires_grad_(False)
    versions=frozen_versions(adapter.model);scorer_versions=frozen_versions(scorer)
    contract=prior.contract_from_config(adapter.model.config)
    # Reuse the existing position/gather/cache/full-token correctness canary.
    canary_path=root/f'rank{rank}.canary.json'
    if entries and not canary_path.exists():
        canary_row=dataset.row_for_entry(entries[0])
        with torch.no_grad():
            prepared=adapter.prepare_sample(canary_row,include_answer=False)
            feature={'input_visual_tokens':len(prepared.visual_positions),
                     'input_text_tokens':int(prepared.inputs_embeds.shape[1]-len(prepared.visual_positions))}
            checked=execution._canary(adapter,scorer,prepared,feature)
            del prepared
        write_immutable(canary_path,{'task_sha256':task['sha256'],'passed':True,'checks':checked})
    elif canary_path.exists() and read(canary_path).get('task_sha256')!=task['sha256']:
        raise ValueError('canary identity changed')
    pins={};failures=0
    if canary_path.exists():pins[str(canary_path)]=file_digest(canary_path)
    for entry in entries:
        row=dataset.row_for_entry(entry)
        references=prior.benchmark.parse_references(row['answer'])
        metric=entry.get('quality_metric',entry.get('metric'))
        methods=list(METHODS);offset=int(entry['row_hash'][:8],16)%len(methods)
        for method in methods[offset:]+methods[:offset]:
            if time.time()>=task['deadline_utc_epoch']:raise TimeoutError('physical deadline between cells')
            dest=root/'rows'/f"{entry['row_hash']}.{method}.json"
            if dest.exists():
                value=read(dest)
                if value['task_sha256']!=task['sha256']:raise ValueError('physical row identity changed')
                failures+=value['status']!='ok';pins[str(dest)]=file_digest(dest);continue
            observed={};select=make_selector(method,scorer,task['audit_rule'],observed)
            internal=prior.Cell('scorer','full' if method=='all_tokens' else 'uniform',0)
            value={'format':FORMAT+'_physical','task_sha256':task['sha256'],'model':task['model_tag'],
                'dataset':family(entry['dataset']),'request_id':entry['row_hash'],'group_id':entry['group_id'],
                'image_sha256':entry['image_sha256'],'category':entry.get('category'),'method':method,'quality_metric':metric,
                'split':'screening','independent_confirmation':False,'timing_isolated':False,
                'physical_measurement_performed':True,'allocation_latency_included':True,
                'cache_contract':contract,'host':socket.gethostname(),'job_id':os.getenv('PBS_JOBID')}
            try:
                prediction,ids,timing,metrics=prior.measured(adapter,row,select,internal,execution.MAX_NEW_TOKENS)
                metrics=runtime.generation_trace(adapter,ids,metrics)
                feature={k:observed[k] for k in ('input_visual_tokens','input_text_tokens')}
                expected=runtime.expected_cache_bytes(feature,observed['selected_count'],contract)
                prior.check_cache(metrics,expected)
                value.update(status='ok',prediction=prediction,
                    quality_score=prior.score_prediction(prediction,references,metric),
                    generated_tokens=int(ids.shape[-1]),expected_cache_bytes=expected,
                    metrics=metrics,**timing,**observed)
                del ids
            except TimeoutError as exc:
                value.update(status='selection_timeout',error=str(exc),quality_score=None,**observed)
                failures+=1
            except Exception as exc:
                atomic_json(root/'errors'/f"{entry['row_hash']}.{method}.{time.time_ns()}.json",
                    {**value,'status':'failed','error_type':type(exc).__name__,'error':str(exc)})
                raise
            assert_frozen(adapter.model,versions);assert_frozen(scorer,scorer_versions)
            write_immutable(dest,value);pins[str(dest)]=file_digest(dest)
            print(json.dumps({'event':'coverage_physical','model':task['model_tag'],'request':entry['row_hash'],
                              'method':method,'status':value['status']}),flush=True)
    marker=execution.sealed({'format':FORMAT+'_complete','task_sha256':task['sha256'],
        'passed':True,'all_measurements_succeeded':failures==0,'failed_measurements':failures,
        'requests':len(entries),'input_hashes':pins,'meaning':'diagnostic completed; not a quality promotion gate'})
    write_immutable(done,marker)
    return marker


def report(root):
    root=Path(root);summary={'format':FORMAT+'_report','updated_utc_epoch':time.time(),
        'training_geometry':{},'fixed_budget':[],'adaptive_compression':[],
        'physical_failures':[],'independent_confirmation':False,'headline_improvement_claim':False}
    for model in MODELS:
        rows=[read(p) for p in (root/'cpu'/model).glob('*.json')]
        valid=[r for r in rows if r['status']=='ok']
        summary['training_geometry'][model]={'examples':len(rows),'successful':len(valid),
            'timeouts':sum(r['status']=='selection_timeout' for r in rows),
            'mean_curvature':statistics.mean(r['curvature']['curvature'] for r in valid) if valid else None,
            'mean_finer_menu_retention_saving_pp':statistics.mean(r['saved_retention_percentage_points'] for r in valid) if valid else None,
            'coverage_improvement_at_50_percent':statistics.mean(r['fixed_counts']['50']['greedy_coverage']-
                r['fixed_counts']['50']['importance_coverage'] for r in valid) if valid else None,
            'not_answer_accuracy':True}
    records=[read(p) for p in (root/'physical').glob('*/rows/*.json')]
    summary['physical_rows']=len(records)
    summary['physical_failures']=[{k:r.get(k) for k in ('model','request_id','method','status','error')}
        for r in records if r['status']!='ok']
    summary['physical_status']='not_started' if not records else 'partial_or_complete_see_shard_markers'
    ok=[r for r in records if r['status']=='ok']
    bykey={(r['model'],r['request_id'],r['method']):r for r in ok}
    if len(bykey)!=len(ok):raise ValueError('duplicate model/request/method physical records')
    from .coverage_audit_report import comparison_tables
    summary['fixed_budget'],summary['adaptive_compression']=comparison_tables(bykey)
    summary['certification']=read(root/'cpu/certification.json') if (root/'cpu/certification.json').exists() else None
    reuse={}
    parent=root.parent
    for category in ('diagnostics-initial','diagnostics-shuffled','diagnostics-oracle'):
        for model in MODELS:
            path=parent/category/(model+'.json')
            if path.exists():reuse[f'{category}/{model}']={'path':str(path),'sha256':file_digest(path)}
    summary['existing_main_diagnostics_reused_as_references']=reuse
    atomic_json(root/'report.json',summary)
    return summary


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','cpu','physical','report'))
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--task',type=Path)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    parser.add_argument('--limit',type=int,default=8)
    args=parser.parse_args(argv)
    root=args.project/ROOT
    if args.phase=='prepare':result=prepare(args.project)
    elif args.phase=='cpu':result=cpu(read(root/'audit.json'),root,args.limit)
    elif args.phase=='physical':result=physical(read(args.task),args.rank,args.world_size)
    else:result=report(root)
    print(json.dumps(result if args.phase=='report' else {'phase':args.phase,'status':'complete'},sort_keys=True),flush=True)


if __name__=='__main__':main()
