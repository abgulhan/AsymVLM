"""Bounded attention-supervision pilot, reusing v18 physical workers and reports.

No box annotations are labels. All embeddings and gradient/attention labels
are freshly generated after a failed historical-cache parity canary. The pipeline
waits in ordinary code; no language-model polling is needed.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time

from . import attention_labels_v19 as labels
from . import evidence_campaign_v18 as campaign
from . import evidence_learning_v18 as learning
from . import evidence_evaluate_v18 as evaluation
from . import evidence_allocation_v18 as allocation
from . import evidence_report_v18 as reporting
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path(learning.NAMESPACE) / 'attention_v19'
START = datetime(2026,10,25,23,41,43,tzinfo=timezone.utc).timestamp()
ARMS = ('gradient','raw','corrected')
MODULE = 'learned_score.mlsys_sprint.attention_pilot_v19'


def read(path):
    return json.loads(Path(path).read_text())


def source_pins():
    return {str(Path(p).resolve()):file_digest(p) for p in
            (__file__,labels.__file__,learning.__file__,evaluation.__file__,allocation.__file__,reporting.__file__)}


def prepare(project):
    project = Path(project).resolve()
    root = project/ROOT
    if (root/'protocol.json').exists():
        return checked(root/'protocol.json')
    old = checked(project/learning.NAMESPACE/'protocol.json')
    manifest = read(old['manifest'])
    train = manifest['splits']['scorer_train']
    cache_root = project/learning.NAMESPACE/'train-native-final/annotation/qwen/cache'
    cache = {e['row_hash']:str(cache_root/(e['row_hash']+'.pt')) for e in train}
    diagnostic = []
    for domain in ('DocVQA','TextVQA'):
        rows = [e for e in train if learning.dataset_family(e['dataset'])==domain]
        diagnostic += [e['row_hash'] for e in sorted(rows,key=lambda e:digest(['attention-v19-audit',e['row_hash']]))[:10]]
    pins = {**old['input_hashes'],**source_pins(),**{p:file_digest(p) for p in cache.values()}}
    protocol = seal(dict(format='mlsys_evidence_v18_protocol',experiment='attention_v19',
        cwd=str(project),output_root=str(root),created_utc_epoch=START,
        stop_new_work_utc_epoch=START+3*3600,physical_deadline_utc_epoch=START+3.5*3600,
        deadline_utc_epoch=START+4*3600,maximum_gpu_workers=8,model=old['model'],
        manifest=old['manifest'],source_task=old['source_task'],training_manifest=old['training_manifest'],
        stage_sizes=old['stage_sizes'],cache_paths=cache,diagnostic_ids=diagnostic,
        historical_cache_used_for_training=False,training_inputs='fresh_native_answer_free',
        input_hashes=pins,training=dict(arms=list(ARMS),epochs=3,seed=42,lr=1e-4,weight_decay=.01),
        correction='8x8_equal_image_relative_attention_leave_image_out_divide_max1',
        layers='middle_half',retentions=[50,70],calibration_retentions=[50,60,70,80,90,100],
        evidence_scope='reused_development_not_independent_confirmation'))
    write_immutable(root/'protocol.json',protocol)
    publish_extract(protocol)
    return protocol


def publish_extract(protocol):
    root,project = Path(protocol['output_root']),Path(protocol['cwd'])
    protocol_path = root/'protocol.json'
    tasks = []
    for rank in range(8):
        tasks.append(dict(id=f'attention-v19-extract-{rank}',resource='gpu',tag='attention-v19',model_tag='qwen25',
            argv=[str(project/'.conda/asymvlm/bin/python'),'-B','-u','-m',MODULE,'extract',
                  '--protocol',str(protocol_path),'--rank',str(rank),'--world-size','8'],
            env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','OMP_NUM_THREADS':'2',
                 'OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONDONTWRITEBYTECODE':'1'},
            depends_on=[],minimum_start_seconds=180,
            artifacts=[dict(path=str(root/'labels'/f'rank{rank}.complete.json'),passed=True)]))
    plan=seal(dict(format='mlsys_attention_v19_worker_plan',cwd=str(project),tasks=tasks,
        unexpected_retries=1,deadline_utc_epoch=protocol['physical_deadline_utc_epoch'],
        input_hashes={**protocol['input_hashes'],str(protocol_path):file_digest(protocol_path)}))
    write_immutable(root/'plans/01-extract/plan.json',plan)


def validate_cache(record, entry, reference_sha):
    import torch
    if (record.get('row_hash')!=entry['row_hash'] or record.get('image_sha256')!=entry['image_sha256']
            or record.get('target_model_checkpoint_sha256')!=reference_sha
            or record.get('answer_free_embeddings') is not True
            or record.get('answers_in_scorer_inputs') is not False
            or record.get('features_inputs_include_answer') is not False):
        raise ValueError('cached scorer inputs have wrong identity or answer scope')
    if {'answer','answers','answer_ids','answer_tokens','answer_positions','input_ids'} & record.keys():
        raise ValueError('answers may not be stored in scorer input cache')
    v,q,s=record['vision'],record['question'],record['saliency']
    if v.ndim!=2 or q.ndim!=2 or not len(v) or not len(q) or v.shape[1]!=q.shape[1] or s.shape!=(len(v),):
        raise ValueError('cached feature geometry changed')
    if not all(torch.isfinite(x).all() for x in (v,q,s)):
        raise ValueError('nonfinite scorer cache')
    labels.cells(record['patch_rectangles'])
    return record


def attention_map(adapter, prepared, query_positions):
    import torch
    chosen = labels.middle_layers(adapter.model.config.num_hidden_layers)
    with torch.no_grad():
        output=adapter.model.model(inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,position_ids=prepared.metadata['position_ids'],
            use_cache=False,return_dict=True,output_attentions=True)
        if output.attentions is None or len(output.attentions)!=adapter.model.config.num_hidden_layers:
            raise ValueError('native decoder did not return attention')
        values=[]
        for index in chosen:
            attn=output.attentions[index]
            if attn is None or attn.ndim!=4 or attn.shape[0]!=1:
                raise ValueError('native attention shape unsupported')
            values.append(attn[0].index_select(1,query_positions).index_select(2,prepared.visual_positions).float().mean((0,1)).cpu())
        raw=torch.stack(values).mean(0).double()
    labels.probability(raw)
    return raw


def fresh_gradient(adapter,prepared):
    from torch.nn.attention import sdpa_kernel,SDPBackend
    from learned_score.multimodel.saliency import compute_answer_gradient_saliency
    # Established safe backward path; decoder/vision inference backend is unchanged.
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        return compute_answer_gradient_saliency(adapter,prepared).detach().float().cpu()


def smoke(project):
    """One leased, idle-GPU canary before freezing/publishing experiment inputs."""
    import socket
    import subprocess
    import torch
    from . import evidence_operations_v18 as operations
    from . import evidence_idle_clusterb_v18 as idle
    from . import compression_execution_v16 as execution
    from .adaptive_data_v3 import load_manifest
    from .evidence_annotation_v18 import qwen_patch_rectangles
    from learned_score.mlsys_scorer.backbone import prepared_inputs,frozen_versions,assert_frozen
    project=Path(project).resolve();old=checked(project/learning.NAMESPACE/'protocol.json')
    host=socket.gethostname().split('.')[0]
    job=json.loads(subprocess.check_output(['qstat','-f','-F','json',idle.JOB],text=True,timeout=30))
    idle.validate_allocation({'deadline_utc_epoch':time.time()+600},job,host,0,time.time())
    with operations.allocation_checks.gpu_lease(host,0):
        if not operations.gpu_idle(0,allow_cooldown=False): raise ValueError('GPU0 is no longer idle')
        args=execution.runtime_args(read(old['source_task']))
        args.manifest,args.split=Path(old['manifest']),'scorer_train'
        manifest=load_manifest(old['manifest'])
        adapter,dataset,_,_=learning.load_pilot_runtime(args,manifest);adapter.dataset=dataset
        versions=frozen_versions(adapter.model)
        entry=manifest.splits['scorer_train'][0];row=dataset.row_for_entry(entry)
        cached_path=project/learning.NAMESPACE/'train-native-final/annotation/qwen/cache'/(entry['row_hash']+'.pt')
        cached=validate_cache(torch.load(cached_path,map_location='cpu',weights_only=True),entry,file_digest(old['model']['checkpoint']))
        prepared,v,q=prepared_inputs(adapter,row,False)
        supervised,sv,sq=prepared_inputs(adapter,row,True)
        assert torch.equal(v,sv) and torch.equal(q,sq)
        comparison=dict(vision_shape=list(v.shape),cached_shape=list(cached['vision'].shape),
            vision_max_absolute_difference=float((v.cpu().float()-cached['vision'].float()).abs().max()),
            question_max_absolute_difference=float((q.cpu().float()-cached['question'].float()).abs().max()),
            cached_backbone=cached.get('backbone'))
        print(json.dumps({'event':'cached_input_comparison',**comparison}),flush=True)
        assert torch.equal(q.cpu().to(torch.bfloat16),cached['question'])
        assert torch.equal(qwen_patch_rectangles(prepared).cpu(),cached['patch_rectangles'])
        raw=attention_map(adapter,supervised,supervised.answer_positions-1)
        gradient=fresh_gradient(adapter,supervised)
        assert gradient.shape==(len(v),) and torch.isfinite(gradient).all()
        neutral,_,_=prepared_inputs(adapter,{**row,'question':'Describe the image briefly.'},False)
        neutral_attention=attention_map(adapter,neutral,torch.tensor([neutral.inputs_embeds.shape[1]-1],device=adapter.device))
        assert len(raw)==len(neutral_attention)==len(v)
        assert prepared.metadata['question']!=neutral.metadata['question']
        assert_frozen(adapter.model,versions)
        return dict(passed=True,tokens=len(v),layers=labels.middle_layers(adapter.model.config.num_hidden_layers),
            visual_attention_mass=float(raw.sum()),fresh_gradient_finite=True,answer_free_live_parity=True,neutral_prompt_changed=True,
            cached_input_comparison=comparison)


def extract(protocol,rank,world_size):
    import torch
    from learned_score.mlsys_scorer.backbone import prepared_inputs,frozen_versions,assert_frozen
    from learned_score.mlsys_scorer.checkpoints import atomic_torch_save
    from . import compression_execution_v16 as execution
    from .adaptive_data_v3 import load_manifest
    from .evidence_annotation_v18 import qwen_patch_rectangles
    if world_size!=8 or not 0<=rank<world_size:
        raise ValueError('eight deterministic extraction shards required')
    root=Path(protocol['output_root']); out=root/'labels'; marker=out/f'rank{rank}.complete.json'
    if marker.exists():
        value=checked(marker)
        if value['protocol_sha256']!=protocol['sha256']: raise ValueError('wrong completed extraction')
        for p,sha in value['input_hashes'].items():
            if file_digest(p)!=sha: raise ValueError('completed extraction changed')
        return value
    if time.time()>=protocol['physical_deadline_utc_epoch']: raise TimeoutError('extraction deadline')
    args=execution.runtime_args(read(protocol['source_task']))
    args.manifest,args.split=Path(protocol['manifest']),'scorer_train'
    manifest=load_manifest(protocol['manifest'])
    adapter,dataset,backbone,scorer=learning.load_pilot_runtime(args,manifest)
    adapter.dataset=dataset
    versions=frozen_versions(adapter.model)
    entries=manifest.splits['scorer_train'][rank::world_size]
    pins={}
    for entry in entries:
        if time.time()>=protocol['physical_deadline_utc_epoch']: raise TimeoutError('extraction deadline')
        rid=entry['row_hash']; path=out/(rid+'.pt')
        if path.exists():
            saved=torch.load(path,map_location='cpu',weights_only=True)
            if saved['protocol_sha256']!=protocol['sha256'] or saved['row_hash']!=rid:
                raise ValueError('resumed extraction identity changed')
        else:
            cached_path=protocol['cache_paths'][rid]
            if file_digest(cached_path)!=protocol['input_hashes'][cached_path]: raise ValueError('cache changed')
            cached=validate_cache(torch.load(cached_path,map_location='cpu',weights_only=True),entry,file_digest(protocol['model']['checkpoint']))
            row=dataset.row_for_entry(entry)
            with torch.no_grad():
                prepared,visual,question=prepared_inputs(adapter,row,False)
                if prepared.answer_positions.numel(): raise ValueError('answer in inference features')
                if not torch.equal(question.cpu().to(torch.bfloat16),cached['question']):
                    raise ValueError('live question differs from historical question')
                cache_difference=float((visual.cpu().float()-cached['vision'].float()).abs().max())
                rectangles=qwen_patch_rectangles(prepared).cpu()
                if not torch.equal(rectangles,cached['patch_rectangles']): raise ValueError('patch geometry mismatch')
                prompt_position=prepared.inputs_embeds.shape[1]-1
            supervised,sv,sq=prepared_inputs(adapter,row,True)
            if not torch.equal(visual,sv) or not torch.equal(question,sq): raise ValueError('answer changed scorer inputs')
            raw=attention_map(adapter,supervised,supervised.answer_positions-1)
            gradient=fresh_gradient(adapter,supervised)
            saved=dict(format='mlsys_attention_labels_v19',protocol_sha256=protocol['sha256'],row_hash=rid,
                image_sha256=entry['image_sha256'],dataset=entry['dataset'],cache_path=cached_path,
                cache_sha256=protocol['input_hashes'][cached_path],raw_attention=raw,
                visual_attention_mass=float(raw.sum()),patch_rectangles=rectangles,
                layers=labels.middle_layers(adapter.model.config.num_hidden_layers),
                label_answer_aware=True,scorer_inputs_answer_free=True,
                vision=visual.detach().cpu().to(torch.bfloat16),question=question.detach().cpu().to(torch.bfloat16),
                saliency=gradient,answer_free_embeddings=True,answers_in_scorer_inputs=False,
                features_inputs_include_answer=False,target_model_checkpoint_sha256=file_digest(protocol['model']['checkpoint']),
                historical_vision_max_absolute_difference=cache_difference,historical_cache_used_for_training=False)
            if rid in protocol['diagnostic_ids']:
                real=attention_map(adapter,prepared,torch.tensor([prompt_position],device=adapter.device))
                neutral_row={**row,'question':'Describe the image briefly.'}
                neutral,_,_=prepared_inputs(adapter,neutral_row,False)
                if not torch.equal(qwen_patch_rectangles(neutral).cpu(),rectangles): raise ValueError('neutral prompt changed image')
                neutral_map=attention_map(adapter,neutral,torch.tensor([neutral.inputs_embeds.shape[1]-1],device=adapter.device))
                saved.update(real_prompt_attention=real,neutral_prompt_attention=neutral_map,
                             diagnostic_scope='last_prompt_position_same_image_neutral_question')
                del neutral
            assert_frozen(adapter.model,versions)
            atomic_torch_save(path,saved)
            del prepared,supervised,visual,question,sv,sq,cached
        pins[str(path)]=file_digest(path)
    result=seal(dict(protocol_sha256=protocol['sha256'],rank=rank,rows=len(entries),passed=True,input_hashes=pins))
    write_immutable(marker,result)
    return result


def load_records(protocol):
    import torch
    root=Path(protocol['output_root'])
    manifest=read(protocol['manifest'])
    expected=manifest['splits']['scorer_train']
    for rank in range(8):
        marker=checked(root/'labels'/f'rank{rank}.complete.json')
        if marker['protocol_sha256']!=protocol['sha256'] or marker.get('passed') is not True:
            raise ValueError('missing valid extraction completion')
        for path,sha in marker['input_hashes'].items():
            if file_digest(path)!=sha: raise ValueError('attention labels changed')
    records=[]
    for entry in expected:
        record=torch.load(root/'labels'/(entry['row_hash']+'.pt'),map_location='cpu',weights_only=True)
        if record['protocol_sha256']!=protocol['sha256'] or record['image_sha256']!=entry['image_sha256']:
            raise ValueError('attention labels outside frozen training split')
        if file_digest(record['cache_path'])!=record['cache_sha256']: raise ValueError('training cache changed')
        validate_cache(record,entry,file_digest(protocol['model']['checkpoint']))
        if record.get('historical_cache_used_for_training') is not False:
            raise ValueError('all arms require fresh matched label-generation inputs')
        records.append(record)
    return records


def fit(protocol,records):
    import torch
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from learned_score.mlsys_scorer.models import optimizer_step
    torch.set_num_threads(2)
    root=Path(protocol['output_root']); result={}
    for record in records:
        record['targets']={'gradient':labels.gradient_target(record['saliency']),
            'raw':labels.probability(record['raw_attention']),
            'corrected':labels.corrected_attention(record,labels.position_bias(records,record['image_sha256']))}
    for arm in ARMS:
        destination=root/'training'/arm; path=destination/'latest.pt'
        if (destination/'complete.json').exists() and (destination/'temperature.json').exists():
            learning.load_scorer(path); result[arm]=str(path); continue
        torch.manual_seed(42)
        scorer,payload=load_scorer(protocol['model']['checkpoint'],'cpu')
        scorer.train().requires_grad_(True)
        optimizer=torch.optim.AdamW(scorer.parameters(),lr=1e-4,weight_decay=.01)
        identity=dict(format='mlsys_attention_training_v19',arm=arm,protocol_sha256=protocol['sha256'],
            recipe=protocol['training'],reference_checkpoint_sha256=file_digest(protocol['model']['checkpoint']),
            reference_identity=payload['identity'],rows=[r['row_hash'] for r in records],
            answer_inputs=False,source_hashes=source_pins())
        epoch0,next_index,steps=0,0,0
        if path.exists():
            _,prior=learning.load_scorer(path,require_complete=False)
            if prior['identity']!=identity: raise ValueError('fit resume identity changed')
            scorer.load_state_dict(prior['state_dict']);optimizer.load_state_dict(prior['optimizer'])
            epoch0,next_index,steps=prior['epoch'],prior['next_index'],prior['steps']
        for epoch in range(epoch0,3):
            order=list(records); random.Random(42+epoch).shuffle(order)
            for index in range(next_index if epoch==epoch0 else 0,len(order)):
                if time.time()>=protocol['stop_new_work_utc_epoch']: raise TimeoutError('fit admission deadline')
                record=order[index];optimizer.zero_grad(set_to_none=True)
                scores=scorer(record['vision'],record['question'])
                optimizer_step(learning.evidence_loss(scores,record['targets'][arm].float()),scorer,optimizer,1.)
                steps+=1
                if (index+1)%25==0:
                    learning.save_checkpoint(path,scorer,optimizer,identity,epoch,index+1,steps)
            learning.save_checkpoint(path,scorer,optimizer,identity,epoch+1,0,steps)
        write_immutable(destination/'complete.json',dict(identity_sha256=digest(identity),
            checkpoint_sha256=file_digest(path),steps=steps,passed=True))
        with torch.no_grad():
            scales=[float(scorer(r['vision'],r['question']).double().std(unbiased=False)) for r in records]
        import statistics
        write_immutable(destination/'temperature.json',dict(temperature=max(1e-8,statistics.median(scales)),
            source='training_only_median_score_standard_deviation',training_rows=len(records)))
        result[arm]=str(path)
    return result


def task_path(protocol,name):
    return Path(protocol['output_root'])/'plans'/name/'task.json'


def stage_done(protocol,name):
    path=task_path(protocol,name)
    if not path.exists(): return False
    root=Path(read(path)['output_root'])
    return all((root/f'rank{r}.complete.json').exists() for r in range(8))


def wait_stage(protocol,name):
    root=Path(protocol['output_root'])
    while not stage_done(protocol,name):
        if time.time()>=protocol['physical_deadline_utc_epoch']: raise TimeoutError('physical deadline: '+name)
        states=list((root/'plans'/name/'state').glob('*.json'))
        for path in states:
            s=read(path)
            if s.get('status') in ('failed','blocked'):
                raise RuntimeError('worker exhausted retry: '+str(path))
        time.sleep(30)
    stage=reporting.collect_task(task_path(protocol,name))
    if not stage['status']['complete']: raise ValueError('completion markers do not cover physical stage')
    return stage


def evaluate(protocol,name,split,cells,**kwargs):
    path=task_path(protocol,name)
    if not path.exists():
        if time.time()>=protocol['stop_new_work_utc_epoch']: raise TimeoutError('stage admission closed: '+name)
        campaign.evaluation_plan(protocol,name,split,cells,**kwargs)
    return path


def diagnostic_plan(protocol,records):
    import torch
    selected=[r for r in records if r['row_hash'] in protocol['diagnostic_ids']]
    groups={key:{} for key in ('suspect','other_attention','random')};counts={};observations=[]
    for r in selected:
        rid=r['row_hash'];raw=labels.probability(r['raw_attention']);n=len(raw);g=max(1,(n*5+99)//100)
        bias=labels.position_bias(records,r['image_sha256'])[labels.cells(r['patch_rectangles'])]
        suspect=torch.argsort(bias*raw,descending=True,stable=True)[:g].tolist()
        others=[i for i in torch.argsort(raw,descending=True,stable=True).tolist() if i not in set(suspect)][:g]
        random_group=torch.randperm(n,generator=torch.Generator().manual_seed(int(digest([42,rid])[:12],16)))[:g].tolist()
        counts[rid]=n-g
        for key,removed in (('suspect',suspect),('other_attention',others),('random',random_group)):
            groups[key][rid]=[i for i in range(n) if i not in set(removed)]
        real=labels.probability(r['real_prompt_attention']);neutral=labels.probability(r['neutral_prompt_attention'])
        real_top=set(torch.argsort(real,descending=True)[:g].tolist());neutral_top=set(torch.argsort(neutral,descending=True)[:g].tolist())
        observations.append(dict(request_id=rid,dataset=r['dataset'],removed_count=g,
            top5percent_question_neutral_overlap=len(real_top&neutral_top)/g,
            suspect_attention_mass=float(raw[suspect].sum()),suspect_indices=suspect))
    root=Path(protocol['output_root'])
    write_immutable(root/'sink-diagnostic-design.json',seal(dict(protocol_sha256=protocol['sha256'],rows=observations,
        interpretation='suspected_positional_sinks_not_known_irrelevance; removals are diagnostic only')))
    cells=[dict(key='full',selector='full')]+[dict(key=key,selector='explicit_indices',counts=counts,indices_by_request=sets) for key,sets in groups.items()]
    return evaluate(protocol,'02-sink-diagnostic','scorer_train',cells,
        request_ids=[r['row_hash'] for r in selected],extra=dict(diagnostic_only=True,answer_metrics=True,warmup=0))


def screening_plan(protocol,checkpoints):
    cells=[dict(key='full',selector='full')]
    for arm,path in {'current':protocol['model']['checkpoint'],**checkpoints}.items():
        cells += [dict(key=f'{arm}_{p:03d}',selector='scorer',checkpoint=path,retention_percent=p) for p in (50,70)]
    return evaluate(protocol,'03-screen','scorer_screening',cells)


def write_report(protocol,screen,final=None,status='screening_complete'):
    from .adaptive_report_v1 import _atomic_text
    root=Path(protocol['output_root'])
    report=reporting.report(selection_tasks=[screen] if screen else [],allocation_tasks=[final] if final else [])
    report['pilot_status']=status
    report['protocol_sha256']=protocol['sha256']
    report['input_hashes'].update({str(root/'protocol.json'):file_digest(root/'protocol.json'),**source_pins()})
    if final:
        rows=reporting.collect_task(final)['rows']
        grouped=reporting._by_domain_method(rows)
        report['matched_pruner_table']=[]
        for (_,domain,method),values in grouped.items():
            if method=='full': continue
            ref=grouped[('qwen25',domain,'full')];adapt=grouped[('qwen25',domain,'adaptive')]
            summary=reporting._summary(values);a=reporting._summary(adapt)
            if summary['total_kv_bytes']!=a['total_kv_bytes'] or summary['total_retained_tokens']!=a['total_retained_tokens']:
                raise ValueError('final competing-pruner budget not matched')
            report['matched_pruner_table'].append(dict(dataset=domain,method=method,**summary,
                kv_reduction_percent=100*(1-summary['total_kv_bytes']/reporting._summary(ref)['total_kv_bytes']),
                difference_from_full=reporting._paired(values,ref),difference_from_adaptive=reporting._paired(values,adapt)))
    atomic_json(root/'report.json',report)
    table={(r['dataset'],r['method']):r for r in report['selection_table']}
    text='Qwen2.5-VL-3B | Physical development pilot, not independent confirmation\n'
    text+='Table 1. Fixed-count comparison; native accuracy scores in percent\n'
    text+='Dataset | Kept % | Pruned KV MiB | All tokens | Unchanged scorer | Gradient-trained | Raw attention | Corrected attention\n'
    for domain in sorted({d for d,m in table}):
        for p in (50,70):
            reference=table[domain,f'current_{p:03d}']
            scores=[table[domain,f'{a}_{p:03d}']['score_percent'] for a in ('current','gradient','raw','corrected')]
            text+=f"{domain} | {p} | {reference['mean_kv_mib']:.2f} | {table[domain,'full']['score_percent']:.2f} | "+' | '.join(f'{s:.2f}' for s in scores)+'\n'
    text+='\nTable 2. Final adaptive/uniform/DivPrune comparison at identical aggregate KV budgets\n'
    if final:
        text+='Dataset | Method | Score % | Mean KV MiB | KV saved % | Requests compressed % | Change vs all tokens [95% CI], pp\n'
        names={'adaptive':'AsymVLM adaptive','uniform':'AsymVLM uniform','divprune':'DivPrune projected-feature adaptation'}
        for r in report['matched_pruner_table']:
            delta=reporting._delta(r['difference_from_full'])
            text+=f"{r['dataset']} | {names[r['method']]} | {r['score_percent']:.2f} | {r['mean_kv_mib']:.2f} | {r['kv_reduction_percent']:.2f} | {r['requests_compressed_percent']:.2f} | {delta}\n"
        for r in report['allocation_table']:
            text+=f"{r['dataset']} adaptive minus uniform: {reporting._delta(r['difference_adaptive_minus_uniform'])}\n"
    else:
        text+=('Not run: neither attention-trained scorer passed the predeclared screening gate. Final labels remain unopened.\n'
               if status=='no_candidate' else 'Not run yet; final labels remain unopened.\n')
    text+='\nKV is allocated decoder-cache capacity, not total GPU memory. No controlled latency claim.\n'
    _atomic_text(root/'TABLES.txt',text)
    return report


def feature_stage(protocol,name):
    task=read(task_path(protocol,name));root=Path(task['output_root'])
    values=[read(root/'features'/(rid+'.json')) for rid in task['request_ids']]
    return values


def converted_features(features,sha,temperature):
    """Recover normalized logits from standard weights; std is supplied separately.

    Fixed-temperature features are emitted by the answer-free feature wrapper,
    not inferred from already-normalized weights.
    """
    result=[]
    for f in features:
        if temperature is None: result.append(f);continue
        curve=f['curves'][sha]
        if 'fixed_temperature_curve' not in curve: raise ValueError('missing fixed-temperature scores')
        result.append({**f,'curves':{**f['curves'],sha:curve['fixed_temperature_curve']}})
    return result


def temperature_evaluation(task,rank,world_size):
    """Add a second answer-free curve, retaining native physical evaluation."""
    original=evaluation.risk_curve
    temperature=task['training_temperature']
    def curves(scores):
        return {**labels.risk_curve(scores),
                'fixed_temperature_curve':labels.risk_curve(scores,temperature)}
    evaluation.risk_curve=curves
    try: return evaluation.run(task,rank,world_size)
    finally: evaluation.risk_curve=original


def temperature_plan(protocol,name,split,cells,checkpoint,temperature):
    project,root=Path(protocol['cwd']),Path(protocol['output_root'])
    path=task_path(protocol,name)
    if path.exists(): return path
    # Publish once: create in-memory plan through the existing builder before a worker can see it.
    original=campaign.write_immutable
    pending={}
    try:
        # The builder hashes its task after writing; write only the task, defer plan visibility.
        def defer_plan(path,value):
            if Path(path).name=='plan.json': pending[Path(path)]=value
            else: original(path,value)
        campaign.write_immutable=defer_plan
        campaign.evaluation_plan(protocol,name,split,cells,feature_checkpoints=[checkpoint],
            extra=dict(training_temperature=temperature))
    finally: campaign.write_immutable=original
    plan_path=root/'plans'/name/'plan.json';plan=pending[plan_path]
    for task in plan['tasks']:
        argv=task['argv'];i=argv.index('learned_score.mlsys_sprint.evidence_evaluate_v18')
        argv[i:i+1]=[MODULE,'evaluate']
    plan['input_hashes'].update(source_pins())
    write_immutable(plan_path,seal({k:v for k,v in plan.items() if k!='sha256'}))
    return path


def pipeline(protocol):
    import fcntl
    root=Path(protocol['output_root']);root.mkdir(parents=True,exist_ok=True)
    with (root/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'pipeline-terminal.json').exists(): return checked(root/'pipeline-terminal.json')
        screen=None
        try:
            atomic_json(root/'pipeline-status.json',dict(stage='extracting_training_attention',time=time.time()))
            while not all((root/'labels'/f'rank{r}.complete.json').exists() for r in range(8)):
                if time.time()>=protocol['stop_new_work_utc_epoch']: raise TimeoutError('attention extraction admission deadline')
                for path in (root/'plans/01-extract/state').glob('*.json'):
                    if read(path).get('status') in ('failed','blocked'):
                        raise RuntimeError('extraction worker exhausted retry: '+str(path))
                time.sleep(30)
            records=load_records(protocol)
            diagnostic=diagnostic_plan(protocol,records)
            atomic_json(root/'pipeline-status.json',dict(stage='fitting_three_scorers',time=time.time()))
            checkpoints=fit(protocol,records)
            screen=screening_plan(protocol,checkpoints)
            atomic_json(root/'pipeline-status.json',dict(stage='physical_screening',time=time.time()))
            stage=wait_stage(protocol,'03-screen')
            gate=labels.screen_gate(stage['rows'],read(screen)['request_ids'])
            write_immutable(root/'screening-gate.json',seal(gate))
            write_report(protocol,screen,status='screening_complete')
            diagnostic_stage=wait_stage(protocol,'02-sink-diagnostic')
            write_immutable(root/'sink-diagnostic-results.json',dict(rows=diagnostic_stage['rows'],
                interpretation='small_training_panel; no safe-deletion claim; not used to choose a scorer'))
            if gate['winner'] is None:
                status='no_candidate'
                write_report(protocol,screen,status=status)
            else:
                winner=gate['winner'];checkpoint=checkpoints[winner];sha=file_digest(checkpoint)
                temperature=read(Path(checkpoint).parent/'temperature.json')['temperature']
                cells=[dict(key='full',selector='full')]+[dict(key=f'winner_{p:03d}',selector='scorer',checkpoint=checkpoint,retention_percent=p) for p in (50,60,70,80,90)]
                temperature_plan(protocol,'04-calibration','controller_calibration',cells,checkpoint,temperature)
                atomic_json(root/'pipeline-status.json',dict(stage='physical_budget_calibration',winner=winner,time=time.time()))
                calibration=wait_stage(protocol,'04-calibration');features=feature_stage(protocol,'04-calibration')
                choices={}
                for conversion,temp in (('current',None),('fixed',temperature)):
                    choices[conversion]=allocation.calibrate(converted_features(features,sha,temp),calibration['rows'],sha)
                conversion=min(choices,key=lambda key:(choices[key]['selected']['aggregate_kv_bytes'],key!='current'))
                chosen=dict(checkpoint=checkpoint,checkpoint_sha256=sha,weight_conversion=conversion,
                    temperature=temperature if conversion=='fixed' else None,tau=choices[conversion]['selected']['tau'],
                    calibration=choices,final_labels_opened=False)
                write_immutable(root/'frozen-method.json',seal(chosen))
                temperature_plan(protocol,'05-final-features','scorer_selection',[],checkpoint,temperature)
                wait_stage(protocol,'05-final-features')
                features=converted_features(feature_stage(protocol,'05-final-features'),sha,chosen['temperature'])
                adaptive,uniform=allocation.matched_assignments(features,sha,chosen['tau'])
                cells=[dict(key='full',selector='full'),dict(key='adaptive',selector='scorer',checkpoint=checkpoint,counts=adaptive),
                    dict(key='uniform',selector='scorer',checkpoint=checkpoint,counts=uniform),dict(key='divprune',selector='divprune',counts=uniform)]
                final=evaluate(protocol,'06-final','scorer_selection',cells)
                atomic_json(root/'pipeline-status.json',dict(stage='final_physical_comparison',time=time.time()))
                wait_stage(protocol,'06-final');write_report(protocol,screen,final,status='complete')
                status='complete'
            result=seal(dict(protocol_sha256=protocol['sha256'],status=status,finished_utc_epoch=time.time(),
                screening_gate=gate,final_evaluation_run=gate['winner'] is not None))
        except Exception as exc:
            result=seal(dict(protocol_sha256=protocol['sha256'],status='deadline_reached' if isinstance(exc,TimeoutError) else 'failed',
                error_type=type(exc).__name__,error=str(exc),finished_utc_epoch=time.time()))
            import traceback
            traceback.print_exc()
        write_immutable(root/'pipeline-terminal.json',result)
        atomic_json(root/'pipeline-status.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='screening_gate'}),flush=True)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('prepare','extract','pipeline','evaluate','smoke'))
    parser.add_argument('--project',type=Path,default=Path.cwd())
    parser.add_argument('--protocol',type=Path,default=ROOT/'protocol.json')
    parser.add_argument('--task',type=Path)
    parser.add_argument('--rank',type=int,default=0);parser.add_argument('--world-size',type=int,default=8)
    args=parser.parse_args(argv)
    if args.phase=='prepare': result=prepare(args.project)
    elif args.phase=='smoke': result=smoke(args.project)
    elif args.phase=='evaluate': result=temperature_evaluation(read(args.task),args.rank,args.world_size)
    else:
        protocol=checked(args.protocol)
        result=extract(protocol,args.rank,args.world_size) if args.phase=='extract' else pipeline(protocol)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_hashes','cache_paths','screening_gate')}),flush=True)


if __name__=='__main__': main()
