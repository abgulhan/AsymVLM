"""Native three-model physical execution for the additive V31 campaign."""
import argparse
import fcntl
from contextlib import nullcontext
import json
import os
from pathlib import Path
import re
import socket
import time

from . import budget_predictor_v31 as budget
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest


def feature_path(root, model, request):
    return Path(root)/'cache'/model/'features'/(request+'.json')


def row_path(root, model, request, selector, count):
    return Path(root)/'cache'/model/'rows'/f'{request}.{selector}.{count}.json'


def spec_key(protocol, model):
    spec = protocol['models'][model]
    return digest(dict(model=spec, generation=dict(max_new_tokens=128, generated_budget=512,
        greedy=True, position_policy='original', text_eviction=False),
        source_hashes=protocol['physical_source_hashes']))


def locked_entries(entries, root, model):
    """Only one physical producer per request across grid and matched tasks."""
    directory=Path(root)/'cache'/model/'locks';directory.mkdir(parents=True,exist_ok=True)
    for entry in entries:
        with (directory/(entry['row_hash']+'.lock')).open('a+') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            yield entry


def load_runtime(protocol, model, split):
    from . import compression_execution_v16 as native
    from . import adaptive_experiment_v3 as old
    from . import phi_adaptive_v4 as phi
    from .adaptive_data_v3 import load_manifest
    from .query_selector_v24 import load as load_query_scorer
    manifest = load_manifest(protocol['manifest'])
    spec = dict(protocol['models'][model])
    spec['checkpoint'] = spec.get('reference_checkpoint', spec['checkpoint'])
    args = native.runtime_args(dict(model=spec, cwd=protocol['cwd'], manifest=protocol['manifest'],
        split=split, training_manifest=protocol['scorer_training_manifest']))
    with phi.installed() if args.adapter == 'phi3_vision' else nullcontext():
        adapter, labels, backbone, scorer = old.load_runtime(args, manifest)
    selected = protocol['models'][model]
    if selected['checkpoint'] != spec['checkpoint']:
        scorer, payload = load_query_scorer(selected['checkpoint'], 'cuda:0')
        if scorer.hidden_size != adapter.hidden_size or payload.get('mode') != 'retrained':
            raise ValueError('query-aware selected scorer/backbone mismatch')
    scorer.eval().requires_grad_(False)
    task = dict(cwd=protocol['cwd'], split=split)
    inputs = native.answer_free_dataset(task, manifest)
    return adapter, labels, inputs, backbone, scorer, manifest


def validate_cached(record, protocol, model, entry):
    if (record.get('execution_key') != spec_key(protocol,model)
            or record.get('request_id') != entry['row_hash']
            or record.get('image_sha256') != entry['image_sha256']
            or record.get('checkpoint_sha256') != protocol['models'][model]['checkpoint_sha256']):
        raise ValueError('physical cache input/selector identity mismatch')


def require_systems_isolation(phase):
    """Systems claims require the existing dispatcher's exclusive node guard."""
    if phase != 'systems':
        return None
    lease = os.environ.get('ASYMVLM_ISOLATION_LEASE', '')
    if (os.environ.get('ASYMVLM_ISOLATED_NODE') != '1'
            or re.fullmatch(r'[0-9a-f]{64}', lease) is None):
        raise ValueError('systems measurements require an isolated_gpu task and verified exclusive node guard')
    return lease


def check_feature_replay(saved, live):
    """Never silently use a different numeric input to the frozen predictor."""
    import numpy as np
    differences = {}
    for key in ('input_visual_tokens', 'input_text_tokens', 'input_question_tokens',
                'ranking_sha256', 'answer_inputs'):
        if saved.get(key) != live.get(key):
            differences[key] = dict(saved=saved.get(key), live=live.get(key))
    if 'cache_contract' in live and saved.get('cache_contract') != live['cache_contract']:
        differences['cache_contract'] = 'changed'
    for key in ('scores_sorted', 'pooled96'):
        a, b = np.asarray(saved.get(key, []), dtype=float), np.asarray(live.get(key, []), dtype=float)
        if a.shape != b.shape:
            differences[key] = dict(saved_shape=list(a.shape), live_shape=list(b.shape))
        elif not np.isfinite(a).all() or not np.isfinite(b).all():
            differences[key] = dict(nonfinite=True)
        elif not np.array_equal(a, b):
            differences[key] = dict(max_absolute_difference=float(np.max(np.abs(a-b))))
    if differences:
        raise ValueError('cached/live predictor input drift: ' + json.dumps(differences, sort_keys=True))


def select_callback(cell, scorer, feature, observed, live_policy=None):
    import torch
    from .stable_divprune_v15c import stable_divprune_order
    from .selector_experiment_v5 import validate_indices
    count = cell['selected_count']; n = feature['input_visual_tokens']
    if type(count) is not int or not 1 <= count <= n:
        raise ValueError('invalid exact selected count')

    @torch.no_grad()
    def select(prepared):
        visual = prepared.inputs_embeds[0,prepared.visual_positions]
        query = prepared.inputs_embeds[0,prepared.semantic_text_positions]
        p = int(prepared.inputs_embeds.shape[1]-len(visual))
        if (len(visual),p) != (n,feature['input_text_tokens']):
            raise ValueError('native live geometry changed')
        ranking = scores = None
        if live_policy is not None or (cell['selector'] == 'scorer' and count < n):
            scores = scorer(visual,query).float()
            ranking = torch.argsort(scores,descending=True,stable=True)
            if digest(ranking.cpu().tolist()) != feature['ranking_sha256']:
                raise ValueError('live patch ranking changed from frozen input features')
        if live_policy is not None:
            live = {**feature, **budget.feature_inputs(scorer,visual,query,p,scores=scores)}
            check_feature_replay(feature,live)
            decision = budget.assign(live_policy['model'],[live],live_policy['threshold'])[feature['request_id']]
            if decision != count:
                raise ValueError('live predictor count differs from frozen assignment')
        if count == n:
            indices = torch.arange(n,device=visual.device,dtype=torch.long)
        elif cell['selector'] == 'scorer':
            indices = ranking[:count].sort().values
        elif cell['selector'] == 'divprune':
            indices = stable_divprune_order(visual,count).sort().values
        else:
            raise ValueError('unknown selector')
        validate_indices(indices,count,n)
        values = indices.cpu().tolist()
        observed.update(input_visual_tokens=n,input_text_tokens=p,retained_visual_tokens=count,
            selected_patch_indices=values,selected_patch_indices_sha256=digest(values),visual_retention=count/n)
        return indices
    return select


def execute(task_path):
    import torch
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    from . import adaptive_experiment_v3 as old
    from . import adaptive_experiment_v4 as physical
    from . import compression_execution_v16 as native
    task = checked(task_path); protocol = checked(task['protocol'])
    for path, expected in task['input_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('changed task dependency: '+path)
    root, model, split, phase = Path(protocol['output_root']),task['model'],task['split'],task['phase']
    isolation_lease = require_systems_isolation(phase)
    marker = Path(task['completion'])
    if marker.exists():
        record = checked(marker)
        if record['task_sha256'] != task['sha256']:
            raise ValueError('completion belongs to another task')
        for p,h in record['input_hashes'].items():
            if file_digest(p) != h:
                raise ValueError('completed result changed')
        return record
    if time.time() >= protocol['physical_deadline_utc_epoch']:
        raise TimeoutError('physical deadline reached before model load')
    adapter, labels, inputs, backbone, scorer, manifest = load_runtime(protocol,model,split)
    entries = {e['row_hash']:e for e in manifest.splits[split]}
    selected = [entries[r] for r in task['request_ids']]
    selected.sort(key=lambda e:(e['source_file'],e['source_row_index']))
    contract = old.contract_from_config(adapter.model.config)
    versions, sv = frozen_versions(adapter.model), frozen_versions(scorer)
    hardware = torch.cuda.get_device_name(adapter.device)
    pins, measured, started = {},0,time.monotonic()
    reuse = checked(protocol['reuse_index']) if model=='qwen25' and protocol.get('reuse_index') else None
    for entry_index, entry in enumerate(locked_entries(selected,root,model)):
        if time.time() >= protocol['physical_deadline_utc_epoch']:
            raise TimeoutError('physical deadline; completed atomic rows preserved')
        rid=entry['row_hash']; fp=feature_path(root,model,rid)
        adapter.dataset=inputs
        raw=inputs.row_for_entry(entry)
        with torch.no_grad():
            prepared=adapter.prepare_sample(raw,include_answer=False)
            if len(prepared.answer_positions):
                raise ValueError('answer span entered the predictor')
            visual=prepared.inputs_embeds[0,prepared.visual_positions]
            query=prepared.inputs_embeds[0,prepared.semantic_text_positions]
            feature=dict(request_id=rid,group_id=entry['group_id'],dataset=entry['dataset'],
                image_sha256=entry['image_sha256'],checkpoint_sha256=protocol['models'][model]['checkpoint_sha256'],
                execution_key=spec_key(protocol,model),cache_contract=contract,
                **budget.feature_inputs(scorer,visual,query,prepared.inputs_embeds.shape[1]-len(visual)))
        if fp.exists():
            saved=json.loads(fp.read_text());validate_cached(saved,protocol,model,entry)
            check_feature_replay(saved,feature)
            feature=saved
        else:
            write_immutable(fp,feature)
        pins[str(fp)]=file_digest(fp)
        if phase=='canary':
            outcomes=native._canary(adapter,scorer,prepared,feature)
            cp=root/'canaries'/model/(rid+'.json')
            write_immutable(cp,dict(passed=True,model=model,execution_key=spec_key(protocol,model),outcomes=outcomes))
            pins[str(cp)]=file_digest(cp)
        del prepared,visual,query
        if phase in ('features','canary'):
            continue
        adapter.dataset=labels
        row=labels.row_for_entry(entry)
        references=old.benchmark.parse_references(row['answer'])
        metric=entry.get('quality_metric',entry['metric'])
        cells=([dict(selector='scorer',selected_count=int(k),name=f'uniform_{p:03d}')
                for p,k in zip(budget.GRID,budget.counts(feature['input_visual_tokens']))]
               if phase=='grid' else task['cells'][rid])
        offset=int(rid[:8],16)%len(cells)
        for cell in cells[offset:]+cells[:offset]:
            if time.time()>=protocol['physical_deadline_utc_epoch']:
                raise TimeoutError('physical deadline between measured cells')
            count=cell['selected_count'];selector=cell['selector']
            path=row_path(root,model,rid,selector,count)
            systems=phase=='systems'
            if systems:
                path=root/'systems'/model/f"{rid}.{cell['name']}.{cell['repeat']}.json"
            if path.exists():
                saved=json.loads(path.read_text());validate_cached(saved,protocol,model,entry)
                if saved.get('status')!='ok' or saved['selected_count']!=count or saved['selector']!=selector:
                    raise ValueError('invalid completed physical cell')
                pins[str(path)]=file_digest(path);continue
            if reuse is not None and selector=='scorer' and not systems:
                from .budget_reuse_v31 import get
                reused = get(reuse,feature,count,spec_key(protocol,model))
                if reused is not None:
                    write_immutable(path,reused);pins[str(path)]=file_digest(path);continue
            live_policy=None
            if systems and cell.get('predictor'):
                live_policy=dict(model=checked(cell['predictor']),threshold=cell['threshold'])
            observed={}; select=select_callback(cell,scorer,feature,observed,live_policy)
            internal=old.Cell(selector,'full' if count==feature['input_visual_tokens'] else 'uniform',0)
            if systems and cell['repeat']==0 and entry_index==0:
                _,ids,_,_=old.measured(adapter,row,select,internal,128);del ids
            prediction,ids,timing,metrics=old.measured(adapter,row,select,internal,128)
            metrics=physical.generation_trace(adapter,ids,metrics)
            expected=physical.expected_cache_bytes(feature,count,contract)
            old.check_cache(metrics,expected)
            record=dict(format='mlsys_budget_physical_v31',execution_key=spec_key(protocol,model),
                request_id=rid,group_id=entry['group_id'],image_sha256=entry['image_sha256'],
                dataset=entry['dataset'],category=entry.get('category'),model_tag=model,model=backbone['model'],
                selector=selector,selected_count=count,checkpoint_sha256=protocol['models'][model]['checkpoint_sha256'],
                quality_metric=metric,quality_score=old.score_prediction(prediction,references,metric),
                prediction=prediction,status='ok',expected_cache_bytes=expected,cache_contract=contract,
                generated_tokens=int(ids.shape[-1]),max_new_tokens=128,generated_budget=512,
                metrics=metrics,host=socket.gethostname(),hardware=hardware,job_id=os.environ.get('PBS_JOBID'),
                physical_measurement_performed=True,allocation_count_supplied_offline=not bool(live_policy),
                allocation_latency_included=bool(live_policy),timing_isolated=systems,
                timing_isolation_scope='whole_node_exclusive_guard' if systems else None,
                isolation_lease=isolation_lease,
                system_cell=cell if systems else None,**timing,**observed)
            del ids
            assert_frozen(adapter.model,versions);assert_frozen(scorer,sv)
            write_immutable(path,record);pins[str(path)]=file_digest(path);measured+=1
        print(json.dumps(dict(event='v31_progress',model=model,phase=phase,requests=entry_index+1,
            measured=measured,elapsed=time.monotonic()-started)),flush=True)
    assert_frozen(adapter.model,versions);assert_frozen(scorer,sv)
    receipt=seal(dict(passed=True,task_sha256=task['sha256'],model=model,phase=phase,requests=len(selected),
        physical_measurements=measured,input_hashes=pins,elapsed_seconds=time.monotonic()-started))
    write_immutable(marker,receipt)
    return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--task',type=Path,required=True);a=p.parse_args()
    try:
        print(json.dumps(execute(a.task)),flush=True)
    except Exception as exc:
        import traceback
        task=json.loads(a.task.read_text())
        atomic_json(Path(task['completion']).with_suffix('.error.json'),dict(error_type=type(exc).__name__,
            error=str(exc),traceback=traceback.format_exc(),time=time.time(),task=str(a.task)))
        raise


if __name__=='__main__':
    main()
