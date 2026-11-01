"""Native exact-count, multi-scorer evaluation for the v18 evidence pilot.

Only new experimental inputs/outputs are accepted. The v16 runtime and original
native adapters are reused without edits. All scorers see image/question
embeddings only; verified answers enter quality scoring or explicit offline CE
diagnostics, never an inference selector.
"""

import argparse
import json
import math
import os
from pathlib import Path
import socket
import time

from . import compression_execution_v16 as execution
from . import evidence_learning_v18 as learning
from .state import atomic_json, digest, file_digest
from .followup_common import write_immutable

FORMAT='mlsys_evidence_evaluate_v18'
TASK_FORMAT=FORMAT+'_task'
DIAGNOSTIC_SELECTORS=('privileged_evidence','shuffled_evidence','random','explicit_indices')


def read(path):
    return json.loads(Path(path).read_text())


def checkpoint_paths(task):
    return sorted(set(str(Path(c['checkpoint']).resolve()) for c in task.get('cells',[])
        if c.get('selector')=='scorer') | set(str(Path(p).resolve()) for p in task.get('feature_checkpoints',[])))


def validate_task(task, *, verify_files=True):
    if (task.get('format') != TASK_FORMAT or
            task.get('sha256') != digest({k:v for k,v in task.items() if k!='sha256'})):
        raise ValueError('invalid v18 evaluation task seal')
    approved=Path(task['cwd']).resolve()/learning.NAMESPACE
    output=Path(task['output_root']).resolve()
    if output==approved or not output.is_relative_to(approved):
        raise ValueError('v18 evaluation output must be a distinct new subdirectory')
    if type(task.get('deadline_utc_epoch')) not in (int,float) or not math.isfinite(task['deadline_utc_epoch']):
        raise ValueError('finite evaluation deadline required')
    ids=task.get('request_ids')
    if not isinstance(ids,list) or not ids or len(set(ids))!=len(ids):
        raise ValueError('explicit unique request panel required')
    keys=set()
    for cell in task.get('cells',[]):
        key=cell.get('key')
        if not isinstance(key,str) or not key or key in keys or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in key):
            raise ValueError('cell keys must be unique safe identifiers')
        keys.add(key)
        if cell.get('selector') not in ('full','scorer','divprune',*DIAGNOSTIC_SELECTORS):
            raise ValueError('unsupported exact-count selector')
        if cell['selector'] in DIAGNOSTIC_SELECTORS:
            if task.get('diagnostic_only') is not True or task.get('split') not in ('scorer_train','scorer_screening'):
                raise ValueError('diagnostic selectors require explicitly marked training/screening task')
        if cell['selector'] in ('privileged_evidence','shuffled_evidence'):
            if task.get('split')!='scorer_train' or not task.get('cache_index'):
                raise ValueError('verified-answer annotations are training-only, never deployable inference')
        if cell['selector']=='explicit_indices':
            indices=cell.get('indices_by_request',{})
            if set(indices)!=set(ids) or any(not isinstance(v,list) or not v or
                    len(set(v))!=len(v) or any(type(i) is not int or i<0 for i in v) for v in indices.values()):
                raise ValueError('explicit diagnostic indices must be unique nonnegative per-request lists')
        if cell['selector']=='scorer' and not cell.get('checkpoint'):
            raise ValueError('each learned selector must name its own checkpoint')
        if cell['selector']=='full':
            if 'counts' in cell or 'retention_percent' in cell:
                raise ValueError('all-token cell cannot override count')
        elif ('counts' in cell)==('retention_percent' in cell):
            raise ValueError('choose exactly one integer retention or per-request count map')
        elif 'counts' in cell:
            if set(cell['counts'])!=set(ids) or any(type(v) is not int or v<1 for v in cell['counts'].values()):
                raise ValueError('exact-count map must cover this request panel')
        elif type(cell['retention_percent']) is not int or not 1<=cell['retention_percent']<=100:
            raise ValueError('retention must be an integer percentage')
        if cell['selector']=='explicit_indices' and ('counts' not in cell or any(
                len(cell['indices_by_request'][key])!=count for key,count in cell['counts'].items())):
            raise ValueError('explicit diagnostic set sizes must match assigned exact counts')
    if not task.get('cells') and not task.get('feature_checkpoints'):
        raise ValueError('task needs physical cells or answer-free feature checkpoints')
    if type(task.get('warmup',1)) is not int or task.get('warmup',1)<0:
        raise ValueError('warmup count invalid')
    if verify_files:
        source=execution.read(task['source_task'])
        execution.validate_task(source,'dev-features')
        if source['model_tag']!='qwen25' or Path(source['cwd']).resolve()!=Path(task['cwd']).resolve():
            raise ValueError('pilot requires original Qwen2.5 runtime from this project')
        pins=task.get('input_hashes',{})
        required=[task['source_task'],task['manifest'],__file__,learning.__file__,*checkpoint_paths(task)]
        if task.get('cache_index'):
            required.append(task['cache_index'])
        required.extend(str(Path(p).parent/'complete.json') for p in checkpoint_paths(task))
        for path in required:
            if pins.get(str(Path(path).resolve()))!=file_digest(path):
                raise ValueError('missing or changed v18 evaluator input pin: '+str(path))
        for path,sha in pins.items():
            if file_digest(path)!=sha:
                raise ValueError('changed evaluator input: '+path)
    return task


def resolve_count(cell,n,row_hash):
    if type(n) is not int or n<1:
        raise ValueError('positive native visual count required')
    count=(n if cell['selector']=='full' else cell['counts'][row_hash]
           if 'counts' in cell else (n*cell['retention_percent']+99)//100)
    if type(count) is not int or not 1<=count<=n:
        raise ValueError('assigned count outside actual native image-token support')
    return count


def risk_curve(scores):
    import torch
    from .compression_stopping_v11 import learned_score_weights
    weights=learned_score_weights(scores)
    order=torch.argsort(scores.float(),descending=True,stable=True)
    # Explicit endpoint and reverse cumulative sum avoid subtraction cancellation.
    removed=torch.cat((weights.index_select(0,order).flip(0).cumsum(0).flip(0),weights.new_zeros(1)))
    values=removed.cpu().tolist()
    values[0],values[-1]=1.,0.
    return dict(removed_weight_by_count=values,weights_sha256=digest(weights.cpu().tolist()),
        ranking_sha256=digest(order.cpu().tolist()),candidate_counts=sorted(set((len(scores)*p+99)//100 for p in (50,60,70,80,90,100))))


def make_selector(cell,scorer,n,p,count,observed,*,diagnostic_weights=None,diagnostic_seed=42):
    import torch
    from .adaptive_benchmark_v1 import select_exact_count
    from .stable_divprune_v15c import stable_divprune_order
    from .selector_experiment_v5 import validate_indices
    from . import adaptive_experiment_v3 as prior
    internal=prior.Cell(cell['selector'],'full' if cell['selector']=='full' else 'uniform',0)
    @torch.no_grad()
    def select(prepared):
        visual=prepared.inputs_embeds[0,prepared.visual_positions]
        query=prepared.inputs_embeds[0,prepared.semantic_text_positions]
        if (len(visual),prepared.inputs_embeds.shape[1]-len(visual))!=(n,p):
            raise ValueError('live image/prompt geometry changed from measured count')
        if cell['selector']=='explicit_indices':
            selected=torch.as_tensor(cell['selected_indices'],device=visual.device,dtype=torch.long).sort().values
        elif count==n or cell['selector']=='full':
            selected=torch.arange(n,device=visual.device,dtype=torch.long)
        elif cell['selector']=='scorer':
            selected=select_exact_count(scorer(visual,query),visual,count)
        elif cell['selector']=='divprune':
            selected=stable_divprune_order(visual,count).sort().values
        elif cell['selector'] in ('privileged_evidence','shuffled_evidence'):
            if diagnostic_weights is None or diagnostic_weights.shape!=(n,):
                raise ValueError('diagnostic annotation weights differ from native patch geometry')
            weights=diagnostic_weights.to(device=visual.device,dtype=torch.float32)
            if cell['selector']=='shuffled_evidence':
                permutation=torch.randperm(n,generator=torch.Generator().manual_seed(diagnostic_seed))
                weights=weights.index_select(0,permutation.to(weights.device))
            selected=select_exact_count(weights,visual,count)
        elif cell['selector']=='random':
            selected=torch.randperm(n,generator=torch.Generator().manual_seed(diagnostic_seed))[:count].sort().values.to(visual.device)
        else:
            raise ValueError('unsupported selector')
        validate_indices(selected,count,n)
        indices=selected.cpu().tolist()
        observed.update(input_visual_tokens=n,input_text_tokens=p,retained_visual_tokens=count,
            visual_retention=count/n,selected_patch_indices=indices,selected_patch_indices_sha256=digest(indices))
        return selected
    return select,internal


def diagnostic_weights(task,entry,n):
    import torch
    index=read(task['cache_index'])
    matches=[r for r in learning.training_entries(index) if r['row_hash']==entry['row_hash']]
    if len(matches)!=1:
        raise ValueError('training diagnostic annotation missing from accepted cache index')
    item=matches[0]
    record=torch.load(item['path'],map_location='cpu',weights_only=True)
    learning.validate_cache(record,item)
    if record['image_sha256']!=entry['image_sha256'] or len(record['evidence_targets'])!=n:
        raise ValueError('privileged annotation image/token mapping mismatch')
    return record['evidence_targets']


def load_candidate(path,reference_sha,reference_payload,device):
    from learned_score.mlsys_scorer.checkpoints import load_scorer as old_load
    from .benchmark import require_completed_checkpoint
    if file_digest(path)==reference_sha:
        scorer,payload=old_load(path,device)
        require_completed_checkpoint(path,payload)
    else:
        scorer,payload=learning.load_scorer(path,device)
        identity=payload['identity']
        if (identity['reference_checkpoint_sha256']!=reference_sha or
                identity['reference_identity']!=reference_payload['identity']):
            raise ValueError('candidate scorer was trained from another backbone/reference')
    return scorer.eval().requires_grad_(False),payload


def answer_metrics(adapter,row,indices):
    """Offline diagnostic only; no answer tensors enter the deployed selector."""
    import torch
    from learned_score.multimodel.controller_v3_supervision import physical_answer_metrics
    if hasattr(row.get('image'),'seek'):
        row['image'].seek(0)
    with torch.no_grad():
        prepared=adapter.prepare_sample(row,include_answer=True)
        teacher=adapter.teacher_forward(prepared)
        selected=torch.as_tensor(indices,dtype=torch.long,device=prepared.inputs_embeds.device)
        metrics=physical_answer_metrics(adapter,prepared,selected,teacher)
    return {**metrics,'answer_metric_scope':'offline_verified_answer_diagnostic_not_selector_input'}


def run(task,rank=0,world_size=1):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions,assert_frozen
    from learned_score.mlsys_scorer.checkpoints import load_checkpoint
    from .adaptive_data_v3 import load_manifest
    from . import adaptive_experiment_v3 as prior
    from . import adaptive_experiment_v4 as physical
    validate_task(task)
    source=execution.read(task['source_task'])
    manifest=load_manifest(task['manifest'])
    entries=execution.shard_entries(manifest.splits[task['split']],task,rank,world_size)
    if time.time()>=task['deadline_utc_epoch']:
        raise TimeoutError('v18 evaluation deadline reached before model load')
    args=execution.runtime_args(source)
    args.manifest,args.split=Path(task['manifest']),task['split']
    adapter,dataset,backbone,reference=learning.load_pilot_runtime(args,manifest)
    reference_sha=file_digest(source['model']['checkpoint'])
    reference_payload=load_checkpoint(source['model']['checkpoint'])
    scorers={}
    for path in checkpoint_paths(task):
        scorer,_=load_candidate(path,reference_sha,reference_payload,args.device)
        if scorer.hidden_size!=adapter.hidden_size:
            raise ValueError('candidate and target hidden sizes differ')
        scorers[path]=scorer
    adapter.dataset=dataset
    contract=prior.contract_from_config(adapter.model.config)
    versions=frozen_versions(adapter.model)
    scorer_versions={p:frozen_versions(s) for p,s in scorers.items()}
    output=Path(task['output_root'])
    metadata=dict(format=FORMAT,task_sha256=task['sha256'],manifest_sha256=manifest.file_sha256,
        source_task_sha256=source['sha256'],backbone=backbone,cache_contract=contract,
        checkpoints={p:file_digest(p) for p in scorers},split=task['split'],world_size=world_size,
        hardware=torch.cuda.get_device_name(adapter.device),max_new_tokens=128,generated_budget=512,
        torch_version=str(torch.__version__),source_hashes=execution.source_pins(args.adapter),
        answer_inputs_to_scorer=False,allocation_count_supplied_offline=True,allocation_latency_included=False,
        baseline_scope='corrected_projected_feature_DivPrune_selector_not_official_pipeline',
        evidence_scope='disjoint_development_panel_not_sealed_confirmation',timing_isolated=False,
        diagnostic_only=task.get('diagnostic_only',False),
        verified_answer_annotation_used_for_selection=any(c['selector'] in ('privileged_evidence','shuffled_evidence') for c in task.get('cells',[])))
    signature=digest(metadata)
    write_immutable(output/f'rank{rank}.metadata.json',metadata)
    complete=output/f'rank{rank}.complete.json'
    if complete.exists():
        marker=read(complete)
        if marker.get('identity_sha256')!=signature:
            raise ValueError('completed evaluation identity differs')
        for path,sha in marker['input_hashes'].items():
            if file_digest(path)!=sha:
                raise ValueError('completed physical result changed')
        return marker
    pins,measurements={},0
    started=time.monotonic()
    for row_index,entry in enumerate(entries):
        if time.time()>=task['deadline_utc_epoch']:
            raise TimeoutError('v18 evaluation deadline reached; completed cells retained')
        row_hash=entry['row_hash']
        row=dataset.row_for_entry(entry)
        if hasattr(row.get('image'),'seek'):
            row['image'].seek(0)
        with torch.no_grad():
            prepared=adapter.prepare_sample(row,include_answer=False)
            visual=prepared.inputs_embeds[0,prepared.visual_positions]
            query=prepared.inputs_embeds[0,prepared.semantic_text_positions]
            n,p=len(visual),int(prepared.inputs_embeds.shape[1]-len(visual))
            feature=dict(format=FORMAT+'_feature',identity_sha256=signature,request_id=row_hash,
                group_id=entry['group_id'],dataset=entry['dataset'],image_sha256=entry['image_sha256'],
                input_visual_tokens=n,input_text_tokens=p,cache_contract=contract,
                answer_inputs=False,curves={file_digest(path):risk_curve(scorer(visual,query)) for path,scorer in scorers.items()})
            if row_index==0 and task.get('canary',True):
                canary_path=output/f'rank{rank}.canary.json'
                if canary_path.exists():
                    saved_canary=read(canary_path)
                    if saved_canary.get('identity_sha256')!=signature or saved_canary.get('passed') is not True:
                        raise ValueError('resumed native parity canary differs')
                else:
                    canary=execution._canary(adapter,reference,prepared,feature)
                    write_immutable(canary_path,dict(identity_sha256=signature,passed=True,outcomes=canary))
                pins[str(canary_path)]=file_digest(canary_path)
        del prepared,visual,query
        feature_path=output/'features'/(row_hash+'.json')
        write_immutable(feature_path,feature)
        pins[str(feature_path)]=file_digest(feature_path)
        references=prior.benchmark.parse_references(row['answer']) if task.get('cells') else []
        metric=entry.get('quality_metric',entry.get('metric')) or learning.quality_metric(entry['dataset'])
        cells=task.get('cells',[])
        offset=int(row_hash[:8],16)%len(cells) if cells else 0
        reused={}
        evidence_weights=(diagnostic_weights(task,entry,n) if any(c['selector'] in
            ('privileged_evidence','shuffled_evidence') for c in cells) else None)
        request_seed=int(digest([42,row_hash])[:15],16)
        for cell in cells[offset:]+cells[:offset]:
            if time.time()>=task['deadline_utc_epoch']:
                raise TimeoutError('v18 evaluation deadline reached between cells')
            count=resolve_count(cell,n,row_hash)
            checkpoint=str(Path(cell['checkpoint']).resolve()) if cell['selector']=='scorer' else None
            checkpoint_sha=file_digest(checkpoint) if checkpoint else None
            selection_identity=(digest(cell['indices_by_request'][row_hash]) if cell['selector']=='explicit_indices'
                                else checkpoint_sha)
            physical_key=(cell['selector'],None if count==n else selection_identity,count)
            path=output/'rows'/f"{row_hash}.{cell['key']}.json"
            if path.exists():
                record=read(path)
                if record.get('identity_sha256')!=signature or record.get('status')!='ok':
                    raise ValueError('resumed evaluation result differs')
                if record.get('physical_measurement_performed'):
                    reused.setdefault(physical_key,record)
                pins[str(path)]=file_digest(path)
                continue
            record=dict(format=FORMAT,identity_sha256=signature,request_id=row_hash,group_id=entry['group_id'],
                image_sha256=entry['image_sha256'],dataset=entry['dataset'],model_tag='qwen25',model=backbone['model'],
                split=task['split'],quality_metric=metric,method=cell['key'],selector=cell['selector'],
                selected_count=count,checkpoint_sha256=checkpoint_sha,manifest_sha256=manifest.file_sha256,
                host=socket.gethostname(),job_id=os.getenv('PBS_JOBID'),hardware=metadata['hardware'],
                timing_isolated=False,max_new_tokens=128,generated_budget=512,cache_contract=contract,
                diagnostic_only=task.get('diagnostic_only',False),
                privileged_evidence=cell['selector'] in ('privileged_evidence','shuffled_evidence'),
                allocation_count_supplied_offline=True,allocation_latency_included=False,
                removed_weight=feature['curves'][checkpoint_sha]['removed_weight_by_count'][count] if checkpoint_sha else None)
            if physical_key in reused:
                record=physical.reuse_physical_measurement(reused[physical_key],record)
            else:
                observed={}
                local_cell={**cell,'selected_indices':cell['indices_by_request'][row_hash]} if cell['selector']=='explicit_indices' else cell
                select,internal=make_selector(local_cell,scorers.get(checkpoint),n,p,count,observed,
                    diagnostic_weights=evidence_weights,diagnostic_seed=request_seed)
                try:
                    if row_index==0:
                        for _ in range(task.get('warmup',1)):
                            _,ids,_,metrics=prior.measured(adapter,row,select,internal,8)
                            physical.generation_trace(adapter,ids,metrics,8)
                            del ids
                    prediction,ids,timing,metrics=prior.measured(adapter,row,select,internal,128)
                    metrics=physical.generation_trace(adapter,ids,metrics)
                    expected=physical.expected_cache_bytes(feature,count,contract)
                    prior.check_cache(metrics,expected)
                    record.update(status='ok',prediction=prediction,
                        quality_score=prior.score_prediction(prediction,references,metric),
                        generated_tokens=int(ids.shape[-1]),generated_token_ids=metrics['generated_token_ids'],
                        stop_reason=metrics['stop_reason'],expected_cache_bytes=expected,metrics=metrics,
                        physical_measurement_performed=True,physical_reuse_of_cell=None,
                        timing_reused_not_an_independent_observation=False,**timing,**observed)
                    del ids
                    if task.get('answer_metrics',False):
                        record.update(answer_metrics(adapter,row,record['selected_patch_indices']))
                except Exception as exc:
                    atomic_json(output/'errors'/f'{row_hash}.{cell["key"]}.{time.time_ns()}.json',
                        dict(record,status='failed',error_type=type(exc).__name__,error=str(exc)))
                    raise
                reused[physical_key]=record
                measurements+=1
            assert_frozen(adapter.model,versions)
            for scorer_path,scorer in scorers.items():
                assert_frozen(scorer,scorer_versions[scorer_path])
            write_immutable(path,record)
            pins[str(path)]=file_digest(path)
        print(json.dumps(dict(event='v18_evaluation_progress',rank=rank,requests=row_index+1,
            physical_measurements=measurements,elapsed_seconds=time.monotonic()-started)),flush=True)
    marker=learning.sealed(dict(format=FORMAT+'_complete',identity_sha256=signature,task_sha256=task['sha256'],
        requests=len(entries),physical_measurements=measurements,input_hashes=pins,passed=True,
        elapsed_seconds=time.monotonic()-started))
    write_immutable(complete,marker)
    return marker


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',required=True,type=Path)
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world-size',type=int,default=1)
    args=parser.parse_args(argv)
    print(json.dumps(run(read(args.task),args.rank,args.world_size)),flush=True)


if __name__=='__main__':
    main()
