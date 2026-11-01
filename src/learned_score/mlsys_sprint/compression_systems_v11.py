"""Exact-matched integrated timing for v11 selector-native stopping rules.

Reuse the frozen v9b capture/residency repairs without editing historical code.
The process-local binding below replaces the live policy calculation and
versioned publication checks. Each method receives one warmup and three fresh
measurements. Dense curve construction is timed; no early-stop speedup is assumed.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from . import compression_execution_v11 as execution
from . import compression_features_v11 as features_api
from . import compression_policy_v11 as policy_api
from . import compression_policy_v8 as geometry
from . import compression_systems_v9 as base_timing
from . import compression_systems_v9b as repaired
from . import compression_systems_run_v9 as coordinator
from .followup_common import write_immutable
from .state import digest, file_digest
from .worker import read_plan, verify_sources

FORMAT = 'mlsys_compression_systems_v11'
REPETITIONS = repaired.REPETITIONS
TIMING_FIELDS = repaired.TIMING_FIELDS
RUNTIME_TIMINGS = repaired.RUNTIME_TIMINGS
POST_PREFILL_FIELDS = repaired.POST_PREFILL_FIELDS
read, checked, seal = execution.read, execution.checked, execution.seal


def source_hashes(adapter):
    from . import compression_stopping_v11 as stopping
    return {**repaired.source_hashes(adapter),
            'mlsys_sprint/compression_stopping_v11.py':file_digest(stopping.__file__),
            'mlsys_sprint/compression_systems_v11.py': file_digest(__file__),
            'mlsys_sprint/compression_policy_v11.py': file_digest(policy_api.__file__),
            'mlsys_sprint/compression_features_v11.py': file_digest(features_api.__file__),
            'mlsys_sprint/compression_execution_v11.py': file_digest(execution.__file__)}


def export_rule(rule):
    """Exclude training scores from the deployable policy artifact."""
    if rule.get('sha256') != digest({k:v for k,v in rule.items() if k!='sha256'}):
        raise ValueError('nested policy identity changed')
    if rule['selector'] not in ('scorer','divprune'):
        raise ValueError('live CDPruner allocation is not integrated')
    keys=('format','selector','kind','target_fraction','threshold','coefficients',
          'support_visual_tokens','fit_group_ids','fit_row_hashes','dataset_is_policy_input')
    return seal({**{k:copy.deepcopy(rule[k]) for k in keys},'trained_rule_sha256':rule['sha256'],
        'evidence':'development excluded-fold rule; no independent confirmation'})


def live_curve(prepared,scorer,rule):
    """Dense measured-menu implementation, including all actual computation."""
    import torch
    from torch.nn import functional as F
    from . import compression_stopping_v11 as stopping
    from learned_score.llava_selectors import divprune_order
    vision=prepared.inputs_embeds[0,prepared.visual_positions]
    question=prepared.inputs_embeds[0,prepared.semantic_text_positions]
    n=len(vision)
    counts=sorted({(n*p+99)//100 for p in features_api.RETENTIONS})
    nonfull=[k for k in counts if k<n]
    scores=None
    try:
        if rule['selector']=='scorer':
            scores=scorer(vision,question).float()
            geometry=F.linear(F.normalize(vision.float(),dim=-1),scorer.visual.weight.float())
            order=torch.argsort(scores,descending=True,stable=True)[:max(nonfull,default=0)]
        else:
            geometry=vision
            order=(divprune_order(vision,max(nonfull)) if nonfull
                   else torch.empty(0,dtype=torch.long,device=vision.device))
        curve=stopping.stopping_curve(visual=geometry,order=order,counts=counts,
            scores=scores,weight_mode='learned_scores' if rule['selector']=='scorer' else 'uniform',
            selector='asymvlm' if rule['selector']=='scorer' else 'divprune',
            geometry=features_api.CONTRACT[rule['selector']+'_geometry'])
        curve['selector']=rule['selector']
    except (ValueError,RuntimeError) as error:
        if isinstance(error,RuntimeError) and 'out of memory' in str(error).lower():
            raise
        curve=features_api._fallback(n,rule['selector'],f'{type(error).__name__}: {error}')
        order=torch.empty(0,dtype=torch.long,device=vision.device)
    return curve,order,scores


def live_assignment(rule,feature,cell,curve):
    row={'row_hash':feature['request_id'],'group_id':feature['group_id'],
         'input_visual_tokens':feature['input_visual_tokens'],'curves':{rule['selector']:curve}}
    result=policy_api.predict(rule,[row])
    count=result['counts'][feature['request_id']]
    if count!=cell['selected_count']:
        raise ValueError('live v11 stopping rule differs from excluded-fold physical schedule')
    return {'selected_count':count,'fallbacks':result['fallbacks']}


def timed_selector(base,scorer,feature,cell,observed):
    rule=cell.get('budget_policy')
    if rule is None:
        return base
    if rule['selector'] not in ('scorer','divprune'):
        raise ValueError('live CDPruner allocation is not integrated')

    def select(prepared):
        import torch
        from .compression_stopping_v11 import selected_positions
        vision=prepared.inputs_embeds[0,prepared.visual_positions]
        question=prepared.inputs_embeds[0,prepared.semantic_text_positions]
        if (len(vision)!=feature['input_visual_tokens'] or
                prepared.inputs_embeds.shape[1]-len(vision)!=feature['input_text_tokens']):
            raise ValueError('live timing geometry differs from frozen native features')
        with torch.no_grad():
            if vision.is_cuda:torch.cuda.synchronize(vision.device)
            started=time.perf_counter()
            curve,order,scores=live_curve(prepared,scorer,rule)
            assignment=live_assignment(rule,feature,cell,curve)
            if vision.is_cuda:torch.cuda.synchronize(vision.device)
            elapsed=time.perf_counter()-started
            if cell['selector'] in (rule['selector'],'full'):
                indices=selected_positions(order,assignment['selected_count'],input_visual_tokens=len(vision))
                # Reuse the exact frozen selector order computed for coverage.
                # No redundant second scorer or diversity forward is charged.
                if observed is not None:
                    values=indices.cpu().tolist()
                    observed.update(selected_patch_indices=values,selected_patch_indices_sha256=digest(values),
                        candidate='current_rank32',query_control='correct',input_visual_tokens=len(vision),
                        input_text_tokens=feature['input_text_tokens'],retained_visual_tokens=len(indices),
                        visual_retention=len(indices)/len(vision),selector_seconds=elapsed,
                        diagnostics_in_end_to_end_timing=False)
            else:
                indices=base(prepared)
            if observed is not None:
                observed.update(budget_statistic=rule['kind'],budget_policy_sha256=rule['sha256'],
                    budget_in_end_to_end_timing=True,score_and_budget_seconds=elapsed,
                    budget_selected_count=assignment['selected_count'],budget_fallback_reasons=assignment['fallbacks'],
                    budget_dataset_is_prediction_input=False,budget_policy_kind=rule['kind'],
                    stopping_implementation='dense measured-menu curve; no early-stop optimization',
                    budget_scorer_executed=rule['selector']=='scorer')
            return indices
    return select

@contextmanager
def repeated():
    original = base_timing.timed_selector
    base_timing.timed_selector = timed_selector
    try:
        with repaired.repeated():
            previous_sources = repaired.physical.source_hashes
            repaired.physical.source_hashes = source_hashes
            try:
                yield
            finally:
                repaired.physical.source_hashes = previous_sources
    finally:
        base_timing.timed_selector = original


def physical_main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = repaired.runtime.parse_args(base_timing.previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('systems methods require one fresh GPU process, no partial timing resume')
    with repeated():
        repaired.physical.main(rest)


@contextmanager
def _coordinator():
    """Use legacy audits with v11 bindings, restoring all module state on exit."""
    names = ('FORMAT', 'verify', 'repeated', 'expected_sources','measurement_argv')
    originals = {name: getattr(coordinator, name) for name in names}
    coordinator.FORMAT = FORMAT
    coordinator.verify = execution.verify
    coordinator.expected_sources = source_hashes
    coordinator.repeated = SimpleNamespace(main=physical_main, source_hashes=source_hashes,
        REPETITIONS=REPETITIONS, TIMING_FIELDS=TIMING_FIELDS, RUNTIME_TIMINGS=RUNTIME_TIMINGS,
        POST_PREFILL_FIELDS=POST_PREFILL_FIELDS)
    def measurement_argv(study,model,descriptor,directory):
        argv=originals['measurement_argv'](study,model,descriptor,directory)
        schedule=checked(Path(study['output_root'])/model/'schedules/systems.json')
        needs_clip=any(c['selector']=='cdpruner' for row in schedule['rows'].values() for c in row['cells'])
        if not needs_clip and '--reference-clip' in argv:
            argv.remove('--reference-clip')
        return argv
    coordinator.measurement_argv=measurement_argv
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(coordinator, name, value)


def make_schedule(study,model,source,manifest,nested,alternative):
    """Eight intact identity-selected groups per domain, with exact subset match."""
    if nested['selector'] not in ('scorer','divprune') or alternative not in ('divprune','cdpruner'):
        raise ValueError('unsupported live allocation or alternative selector')
    value=copy.deepcopy({k:v for k,v in source.items() if k!='sha256'})
    rules={}
    for fold in nested['folds']:
        rule=export_rule(fold['rule'])
        for request,count in fold['validation_assignments']['counts'].items():
            if request in rules or request in rule['fit_row_hashes']:
                raise ValueError('systems allocation is not excluded-fold')
            rules[request]=rule
    adaptive=('scorer_adaptive',alternative+'_adaptive')
    uniform=('scorer_uniform',alternative+'_uniform')
    for request,row in value['rows'].items():
        if request not in rules or row['group_id'] in rules[request]['fit_group_ids']:
            raise ValueError('missing excluded-fold rule or training-image overlap')
        cells={c['key']:c for c in row['cells']}
        if set(cells)!=set((*adaptive,*uniform)):
            raise ValueError('systems source is not the complete two-by-two factorial')
        for method in adaptive:
            if cells[method]['selected_count']!=nested['counts'][request]:
                raise ValueError('nested assignment differs from physical factorial')
            cells[method]['budget_policy']=rules[request]
        row['cells']=[{'key':'full','selector':'full','selected_count':row['input_visual_tokens'],
            'mode':'exact','policy':'all_image_tokens'},*cells.values()]
    with _coordinator():
        selected=coordinator.make_schedule(study,model,seal(value),manifest)
    value=copy.deepcopy({k:v for k,v in selected.items() if k!='sha256'})
    domains=defaultdict(list)
    for request,row in value['rows'].items():
        domains[row['dataset']].append((request,row,{c['key']:c for c in row['cells']}))
    totals={}
    for domain,members in domains.items():
        goal=sum(cells[adaptive[0]]['selected_count'] for _,_,cells in members)
        rows=[{'row_hash':request,**{k:row[k] for k in
            ('group_id','dataset','input_visual_tokens','input_text_tokens')}} for request,row,_ in members]
        assignment=geometry.uniform_at_visual_total(rows,goal)
        if not assignment['feasible']:raise ValueError('systems subset cannot exactly match tokens')
        for request,row,cells in members:
            for method in uniform:
                count=assignment['counts'][request]
                selector='scorer' if method.startswith('scorer_') else alternative
                cells[method].update(selected_count=count,selector='full' if count==row['input_visual_tokens'] else selector,
                    timing_subset_rebalanced=True)
                cells[method].pop('keep_percent',None)
        totals[domain]={method:sum(cells[method]['selected_count'] for _,_,cells in members)
                       for method in ('full',*adaptive,*uniform)}
        if any(total!=goal for method,total in totals[domain].items() if method!='full'):
            raise ValueError('systems subset methods do not match aggregate image tokens')
    value.update(matched_visual_totals=totals,source_schedule_sha256=source['sha256'],
        nested_policy_sha256=nested['sha256'],allocation_feature_selector=nested['selector'],
        stopping_implementation='dense measured-menu curve; no early-stop optimization',
        reference_clip_resident=any(c['selector']=='cdpruner' for row in value['rows'].values() for c in row['cells']),
        matched_control_note='All four compressed methods match image-token totals on this exact timing subset.')
    return seal(value)


def publish_systems(study,model,*,target=70):
    """Publish integrated systems work once the chosen physical schedule exists."""
    execution.verify(study)
    if target!=70:raise ValueError('the predeclared systems target is 70 percent')
    root,spec=Path(study['output_root']),study['models'][model]
    decision_path=root/model/'development/selected.json'
    if not decision_path.exists():return None
    decision=checked(decision_path)
    selector=decision['allocation_feature_selector']
    if selector=='cdpruner':
        return {'status':'not_ready','reason':'live reference-CLIP CDPruner allocation is not integrated'}
    nested_path=root/model/f'development/{selector}-p{target:03d}-nested.json'
    nested=checked(nested_path)['result']
    source_path=root/model/f'schedules/screening-matched-p{target:03d}-master-v11.json'
    if not source_path.exists():return None
    plan_path=root/f'plans/systems-{model}/plan.json'
    if not plan_path.exists() and time.time()>=study['deadline_utc_epoch']:
        raise ValueError('cannot publish systems work after deadline')
    source=checked(source_path)
    schedule=make_schedule(study,model,source,read(study['manifest']),nested,decision['alternative_selector'])
    output=root/model/'schedules/systems.json'
    policy_path=root/model/'systems/policy.json'
    policies={c['budget_policy']['sha256']:c['budget_policy'] for row in schedule['rows'].values()
              for c in row['cells'] if c.get('budget_policy') is not None}
    write_immutable(output,schedule)
    write_immutable(policy_path,seal({'rules':policies,'nested_sha256':nested['sha256']}))
    _,feature_root,_,_=execution.development_features(study,model,source['split'])
    pins={**execution.source_pins(study,model),
        **{str(Path(study['cwd'])/'learned_score'/name):sha for name,sha in source_hashes(spec['adapter']).items()},
        **{str(Path(module.__file__).resolve()):file_digest(module.__file__)
           for module in (coordinator,coordinator.audit,coordinator.reporting)}}
    for path in (source_path,nested_path,decision_path,Path(study['manifest']),output,policy_path):
        pins[str(path.resolve())]=file_digest(path)
    for request,row in schedule['rows'].items():
        path=feature_root/'features'/(request+'.json')
        feature=read(path)
        if (feature.get('checkpoint_sha256')!=schedule['checkpoint_sha256']
                or feature.get('request_id')!=request
                or any(feature.get(k)!=row[k] for k in ('group_id','dataset','input_visual_tokens','input_text_tokens'))):
            raise ValueError('systems feature identity differs from native schedule')
        pins[str(path.resolve())]=file_digest(path)
    descriptor=seal({'format':FORMAT+'_input','study_sha256':study['sha256'],'model':model,
        'manifest_path':study['manifest'],'feature_root':str(feature_root.resolve()),
        'policy_path':str(policy_path),'input_hashes':pins,'evidence':'development timing only'})
    descriptor_path=root/model/'systems/input.json'
    write_immutable(descriptor_path,descriptor)
    pins[str(descriptor_path)]=file_digest(descriptor_path)
    argv=execution.matrix.previous.python(study,'compression_systems_v11')+[
        '--phase','measure','--study',str(root/'study.json'),'--model-tag',model,'--adapter',spec['adapter']]
    task=execution.matrix.previous.task(study,'systems-'+model,argv,[root/model/'systems/complete.json'])
    task.update(id='compression-v11-systems-'+model,tag='compression-v11',model_tag=model,minimum_start_seconds=900)
    plan=seal({'format':FORMAT+'_plan','cwd':study['cwd'],'tasks':[task],'input_hashes':pins,
        'unexpected_retries':1,'created_utc_epoch':study['created_utc_epoch'],
        'deadline_utc_epoch':study['deadline_utc_epoch'],'world_size':1})
    write_immutable(plan_path,plan)
    verify_sources(read_plan(plan_path))
    return {'status':'prepared_not_launched','plan':str(plan_path),'questions':len(schedule['rows'])}

def measure(study, model, adapter):
    with _coordinator():
        return coordinator.measure(study, model, adapter)


def collect(study, model):
    execution.verify(study)
    path = Path(study['output_root'])/f'plans/systems-{model}/plan.json'
    if not path.exists():
        return None
    verify_sources(read_plan(path))
    with _coordinator():
        return coordinator.collect(study, model)


def summarize(rows):
    with _coordinator():
        result = coordinator.summarize(rows)
    result.update(scope='sequential batch-one inference; live dense stopping-curve computation and budget work included',
        matched_control_note='All compressed methods match aggregate visual-token counts on the measured subset.')
    for panel in result['panels']:
        values = [r for r in rows if (r['model_tag'], r['dataset'], r['method']) ==
                  (panel['model'], panel['dataset'], panel['method'])]
        overhead = [r.get('score_and_budget_seconds') for r in values]
        panel['mean_score_and_budget_seconds'] = (statistics.mean(overhead)
            if all(isinstance(v, (int, float)) for v in overhead) else None)
        panel['mean_score_and_budget_note'] = 'Last repetition per input; end-to-end timing averages all three repetitions.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure',), required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-tag', choices=execution.MODELS, required=True)
    parser.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(measure(checked(args.study), args.model_tag, args.adapter), sort_keys=True))


if __name__ == '__main__':
    main()

