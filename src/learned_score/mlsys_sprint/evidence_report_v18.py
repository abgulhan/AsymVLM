"""Two compact, paired physical tables for the v18 evidence pilot.

No policy fitting, generation, interpolation, or partial-cohort quality claims.
Each requested stage must be complete; optional incomplete mode emits status
only. Ordinary selector results and privileged diagnostics are never pooled.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from . import evidence_evaluate_v18 as evaluation
from .adaptive_data_v3 import dataset_family
from .adaptive_report_v3 import paired_ci
from .adaptive_report_v1 import _atomic_text
from .state import atomic_json, digest, file_digest

FORMAT='mlsys_evidence_report_v18'


def read(path):
    return json.loads(Path(path).read_text())


def _same(a,b,fields,what):
    if any(a.get(k)!=b.get(k) for k in fields):
        raise ValueError(what+' differs')


def collect_task(path):
    path=Path(path).resolve()
    task=read(path)
    evaluation.validate_task(task)
    directory=Path(task['output_root'])
    expected={(key,cell['key']) for key in task['request_ids'] for cell in task['cells']}
    cells={cell['key']:cell for cell in task['cells']}
    metadata={}
    marker_pins={}
    complete_ranks=[]
    world_size=None
    for meta_path in sorted(directory.glob('rank*.metadata.json')):
        meta=read(meta_path)
        rank=int(meta_path.name.split('.')[0][4:])
        if meta.get('task_sha256')!=task['sha256'] or meta.get('format')!=evaluation.FORMAT:
            raise ValueError('metadata does not bind the requested physical task')
        if world_size is not None and meta['world_size']!=world_size:
            raise ValueError('physical shard counts disagree')
        world_size=meta['world_size']
        if rank<0 or rank>=world_size or rank in metadata:
            raise ValueError('invalid duplicate physical shard metadata')
        metadata[rank]=meta
        marker_pins[str(meta_path)]=file_digest(meta_path)
        marker_path=directory/f'rank{rank}.complete.json'
        if marker_path.exists():
            marker=read(marker_path)
            if (marker.get('identity_sha256')!=digest(meta) or marker.get('task_sha256')!=task['sha256'] or
                    marker.get('sha256')!=digest({k:v for k,v in marker.items() if k!='sha256'}) or
                    marker.get('passed') is not True):
                raise ValueError('invalid physical completion marker')
            for filename,sha in marker['input_hashes'].items():
                if file_digest(filename)!=sha:
                    raise ValueError('completed physical artifact changed: '+filename)
                marker_pins[filename]=sha
            marker_pins[str(marker_path)]=file_digest(marker_path)
            complete_ranks.append(rank)
    signatures={digest(meta):meta for meta in metadata.values()}
    rows={}
    for row_path in sorted((directory/'rows').glob('*.json')):
        row=read(row_path)
        key=(row.get('request_id'),row.get('method'))
        if key not in expected or key in rows:
            raise ValueError('physical result outside task or duplicated')
        meta=signatures.get(row.get('identity_sha256'))
        quality=row.get('quality_score')
        if (meta is None or row.get('status')!='ok' or type(quality) not in (int,float) or
                not math.isfinite(quality) or not 0<=quality<=1):
            raise ValueError('physical row lacks valid identity/quality')
        cell=cells[row['method']]
        n,p,k=row['input_visual_tokens'],row['input_text_tokens'],row['selected_count']
        if any(type(x) is not int for x in (n,p,k)) or n<1 or p<0:
            raise ValueError('invalid measured token geometry')
        if evaluation.resolve_count(cell,n,row['request_id'])!=k:
            raise ValueError('measured count differs from task assignment')
        checkpoint_sha=file_digest(cell['checkpoint']) if cell['selector']=='scorer' else None
        if row.get('checkpoint_sha256')!=checkpoint_sha or row['selector']!=cell['selector']:
            raise ValueError('measured selector/checkpoint differs from scheduled cell')
        if (row.get('max_new_tokens')!=128 or row.get('generated_budget')!=512 or
                row.get('cache_contract')!=meta['cache_contract']):
            raise ValueError('decoding/cache contract changed')
        from .adaptive_experiment_v4 import expected_cache_bytes
        from .adaptive_experiment_v3 import check_cache
        expected_bytes=expected_cache_bytes(dict(input_visual_tokens=n,input_text_tokens=p),k,row['cache_contract'])
        if row.get('expected_cache_bytes')!=expected_bytes:
            raise ValueError('reported memory differs from native count contract')
        check_cache(row['metrics'],expected_bytes)
        if row.get('diagnostic_only',False)!=task.get('diagnostic_only',False):
            raise ValueError('diagnostic result mislabelled as deployment evidence')
        rows[key]=row
        marker_pins[str(row_path)]=file_digest(row_path)
    for row in rows.values():
        if row.get('physical_measurement_performed') is True:
            if row.get('physical_reuse_of_cell') is not None:
                raise ValueError('direct physical result claims an alias source')
        elif row.get('physical_measurement_performed') is False:
            source=rows.get((row['request_id'],row.get('physical_reuse_of_cell')))
            if source is None or source.get('physical_measurement_performed') is not True or row.get('physical_source_row_sha256')!=digest(source):
                raise ValueError('alias lacks its directly measured source')
            _same(row,source,('selected_count','selector','quality_score','prediction','generated_token_ids',
                              'expected_cache_bytes','input_visual_tokens','input_text_tokens',
                              'selected_patch_indices','selected_patch_indices_sha256'),'exact-count alias')
        else:
            raise ValueError('row must distinguish physical inference from exact-count alias')
    complete=(set(rows)==expected and world_size is not None and sorted(complete_ranks)==list(range(world_size)))
    status=dict(task=str(path),complete=complete,expected_cells=len(expected),available_cells=len(rows),
        world_size=world_size,complete_ranks=complete_ranks,split=task['split'],
        diagnostic_only=task.get('diagnostic_only',False),missing_cells=len(expected-set(rows)))
    return dict(task=task,status=status,rows=list(rows.values()),pins={str(path):file_digest(path),**marker_pins})


def merge_records(stages):
    """Audit repeated controls; never count reruns as independent questions."""
    result={}
    repeats=0
    for stage in stages:
        for row in stage['rows']:
            key=row['request_id'],row['method']
            if key in result:
                _same(row,result[key],('dataset','group_id','model_tag','checkpoint_sha256','selected_count',
                    'input_visual_tokens','input_text_tokens','quality_score','prediction','generated_token_ids',
                    'expected_cache_bytes','diagnostic_only','privileged_evidence'),'repeated control')
                repeats+=1
            else:
                result[key]=row
    return list(result.values()),repeats


def _by_domain_method(rows):
    grouped=defaultdict(list)
    for row in rows:
        grouped[(row['model_tag'],dataset_family(row['dataset']),row['method'])].append(row)
    return grouped


def _summary(rows):
    if not rows:
        raise ValueError('empty physical cohort')
    return dict(requests=len(rows),retained_tokens_mean=statistics.mean(r['selected_count'] for r in rows),
        input_visual_tokens_mean=statistics.mean(r['input_visual_tokens'] for r in rows),
        retained_fraction=sum(r['selected_count'] for r in rows)/sum(r['input_visual_tokens'] for r in rows),
        mean_kv_mib=statistics.mean(r['expected_cache_bytes'] for r in rows)/2**20,
        score_percent=100*statistics.mean(r['quality_score'] for r in rows),
        requests_compressed_percent=100*statistics.mean(r['selected_count']<r['input_visual_tokens'] for r in rows),
        total_retained_tokens=sum(r['selected_count'] for r in rows),
        total_kv_bytes=sum(r['expected_cache_bytes'] for r in rows),
        privileged_diagnostic=any(r.get('privileged_evidence',False) for r in rows))


def _paired(candidate,reference):
    a={r['request_id']:r for r in candidate}
    b={r['request_id']:r for r in reference}
    if not a or a.keys()!=b.keys() or len(a)!=len(candidate) or len(b)!=len(reference):
        raise ValueError('comparison requires the complete identical request cohort')
    for key,row in a.items():
        _same(row,b[key],('dataset','group_id','model_tag','input_visual_tokens','input_text_tokens','cache_contract'),
              'paired request image/prompt/cache geometry')
    metrics={r['quality_metric'] for r in candidate+reference}
    if len(metrics)!=1:
        raise ValueError('paired quality metrics differ')
    return paired_ci(candidate,reference,next(iter(metrics)),repetitions=10000,seed=42)


def selection_table(rows,stages,full_method='full'):
    grouped=_by_domain_method(rows)
    nominal={}
    for stage in stages:
        for cell in stage['task']['cells']:
            if 'retention_percent' in cell:
                old=nominal.setdefault(cell['key'],cell['retention_percent'])
                if old!=cell['retention_percent']:
                    raise ValueError('same method key names differing nominal budgets')
    output=[]
    for (model,domain,method),values in sorted(grouped.items()):
        full=grouped.get((model,domain,full_method))
        if full is None:
            raise ValueError('selection comparison requires all-token baseline for every domain')
        full_delta=_paired(values,full)
        current_method=f"current_{nominal[method]:03d}" if method in nominal else None
        current=grouped.get((model,domain,current_method)) if current_method else None
        current_delta=None
        if current is not None:
            a={r['request_id']:r for r in values}
            b={r['request_id']:r for r in current}
            if a.keys()!=b.keys() or any(a[k]['selected_count']!=b[k]['selected_count'] for k in a):
                raise ValueError('current-scorer comparison is not at identical per-request counts')
            current_delta=_paired(values,current)
        output.append(dict(model=model,dataset=domain,method=method,metric=values[0]['quality_metric'],
            **_summary(values),difference_from_all_tokens=full_delta,difference_from_current=current_delta,
            current_reference=current_method if current is not None else None))
    return output


def allocation_table(rows,adaptive_method='adaptive',uniform_method='uniform'):
    grouped=_by_domain_method(rows)
    domains=sorted({(model,domain) for model,domain,_ in grouped})
    output=[]
    for model,domain in domains:
        adaptive=grouped.get((model,domain,adaptive_method))
        uniform=grouped.get((model,domain,uniform_method))
        if adaptive is None or uniform is None:
            raise ValueError('allocation comparison requires both adaptive and uniform in every domain')
        delta=_paired(adaptive,uniform)
        checkpoint_a={r.get('checkpoint_sha256') for r in adaptive}
        checkpoint_b={r.get('checkpoint_sha256') for r in uniform}
        if len(checkpoint_a)!=1 or None in checkpoint_a or checkpoint_a!=checkpoint_b:
            raise ValueError('adaptive/uniform must use the exact same scorer checkpoint')
        a,b=_summary(adaptive),_summary(uniform)
        if a['total_retained_tokens']!=b['total_retained_tokens'] or a['total_kv_bytes']!=b['total_kv_bytes']:
            raise ValueError('adaptive/uniform aggregate image-token and KV budgets must match exactly')
        if a['privileged_diagnostic'] or b['privileged_diagnostic']:
            raise ValueError('privileged annotations cannot support a deployable adaptive claim')
        output.append(dict(model=model,dataset=domain,adaptive_method=adaptive_method,uniform_method=uniform_method,
            adaptive=a,uniform=b,difference_adaptive_minus_uniform=delta,exact_memory_match=True,
            checkpoint_sha256=next(iter(checkpoint_a))))
        full=grouped.get((model,domain,'full'))
        if full is not None:
            output[-1].update(all_tokens=_summary(full),difference_adaptive_minus_all_tokens=_paired(adaptive,full))
    return output


def _delta(value):
    if value is None:
        return '—'
    lo,hi=value['ci_lower_pp'],value['ci_upper_pp']
    interval='CI unavailable' if lo is None else f'[{lo:+.2f}, {hi:+.2f}]'
    return f"{value['delta_pp']:+.2f} {interval}"


def _method_name(method):
    if method=='full':
        return 'All image tokens'
    labels={'current':'AsymVLM original scorer','control':'AsymVLM matched extra training',
            'evidence':'AsymVLM evidence-supervised scorer','divprune':'DivPrune adaptation',
            'privileged_evidence':'Teacher evidence','shuffled_evidence':'Shuffled teacher targets',
            'random':'Random patches'}
    prefix,separator,suffix=method.rpartition('_')
    if separator and suffix.isdigit() and prefix in labels:
        return f'{labels[prefix]} ({int(suffix)}% kept)'
    return labels.get(method,method)


def render(report):
    lines=['Qwen2.5-VL-3B: evidence-supervision development pilot','',
        'Quality is native ANLS or VQA score ×100. Differences are percentage points.',
        'KV is allocated cache capacity, not total GPU peak memory. Keep% is total kept / total original image tokens.',
        'Paired 95% intervals resample image groups 10,000 times with seed42. No unlabeled pooled average.',
        'Previously examined development sources; not independent confirmation. Timings were not isolated.',
        'DivPrune is the existing native-token selector adaptation, not the official end-to-end pipeline.','']
    for kind,stages in report['stage_status'].items():
        for stage in stages:
            lines.append(f"{kind}: {Path(stage['task']).parent.name}: {stage['available_cells']}/{stage['expected_cells']} cells; complete={stage['complete']}")
    if not report['complete']:
        return '\n'.join(lines+['','INCOMPLETE: no partial-cohort quality or matched-memory headline tables published.'])+'\n'
    lines+=['','Table 1 — Patch selection at physically evaluated counts','',
        '| Dataset | Method | Requests | Mean kept tokens | Keep% | Mean KV MiB | Score | Difference from all tokens [95% CI] | Difference from current scorer [95% CI] |',
        '|---|---|---:|---:|---:|---:|---:|---|---|']
    for row in report['selection_table']:
        name=_method_name(row['method'])+(' (PRIVILEGED diagnostic)' if row['privileged_diagnostic'] else '')
        lines.append(f"| {row['dataset']} | {name} | {row['requests']} | {row['retained_tokens_mean']:.1f} | {100*row['retained_fraction']:.2f} | {row['mean_kv_mib']:.2f} | {row['score_percent']:.2f} | {_delta(row['difference_from_all_tokens'])} | {_delta(row['difference_from_current'])} |")
    if not report['selection_table']:
        lines.append('Selection stage not supplied.')
    lines+=['','Table 2 — Adaptive versus uniform with the same scorer and exactly matched aggregate KV','',
        '| Dataset | Shared mean KV MiB | All-token score | Adaptive score | Uniform score | Adaptive minus uniform [95% CI] | Adaptive requests compressed | Uniform requests compressed |',
        '|---|---:|---:|---:|---:|---|---:|---:|']
    for row in report['allocation_table']:
        a,b=row['adaptive'],row['uniform']
        full_score=f"{row['all_tokens']['score_percent']:.2f}" if row.get('all_tokens') else '—'
        lines.append(f"| {row['dataset']} | {a['mean_kv_mib']:.2f} | {full_score} | {a['score_percent']:.2f} | {b['score_percent']:.2f} | {_delta(row['difference_adaptive_minus_uniform'])} | {a['requests_compressed_percent']:.2f}% | {b['requests_compressed_percent']:.2f}% |")
    if not report['allocation_table']:
        lines.append('Adaptive stage not supplied / not advanced; no adaptive benefit claimed.')
    lines+=['','Exact-count aliases are identified and are not independent repetitions.',
        'A zero empirical interval on a small unchanged panel does not establish population equivalence.']
    return '\n'.join(lines)+'\n'


def report(selection_tasks=(),allocation_tasks=(),*,allow_incomplete=False,adaptive_method='adaptive',uniform_method='uniform'):
    stages={kind:[collect_task(path) for path in paths] for kind,paths in
            (('selection',selection_tasks),('allocation',allocation_tasks))}
    if not any(stages.values()):
        raise ValueError('at least one physical task required')
    complete=all(stage['status']['complete'] for values in stages.values() for stage in values)
    if not complete and not allow_incomplete:
        raise ValueError('physical stages incomplete; use --allow-incomplete for status only')
    result=dict(format=FORMAT,complete=complete,selection_table=[],allocation_table=[],
        stage_status={kind:[stage['status'] for stage in values] for kind,values in stages.items()},
        input_hashes={key:value for values in stages.values() for stage in values for key,value in stage['pins'].items()},
        evidence_scope='physical_development_not_independent_confirmation',bootstrap_seed=42,bootstrap_repetitions=10000,
        timing_claim='uncontrolled_no_production_speedup_claim',score_averaging='within_dataset_native_score_mean_no_pooled_macro')
    if complete:
        for kind,values in stages.items():
            rows,repeats=merge_records(values)
            result[kind+'_repeated_control_cells_audited']=repeats
            if rows:
                if kind=='selection':
                    scopes={r.get('diagnostic_only',False) for r in rows}
                    if len(scopes)!=1:
                        raise ValueError('ordinary selection and privileged diagnostics require separate reports')
                    result['selection_table']=selection_table(rows,values)
                else:
                    result['allocation_table']=allocation_table(rows,adaptive_method,uniform_method)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection-tasks',nargs='*',type=Path,default=[])
    parser.add_argument('--allocation-tasks',nargs='*',type=Path,default=[])
    parser.add_argument('--adaptive-method',default='adaptive')
    parser.add_argument('--uniform-method',default='uniform')
    parser.add_argument('--allow-incomplete',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    result=report(args.selection_tasks,args.allocation_tasks,allow_incomplete=args.allow_incomplete,
        adaptive_method=args.adaptive_method,uniform_method=args.uniform_method)
    first_task=read((args.selection_tasks+args.allocation_tasks)[0])
    approved=Path(first_task['cwd']).resolve()/evaluation.learning.NAMESPACE/'report'
    if args.output.resolve()!=approved and not args.output.resolve().is_relative_to(approved):
        raise ValueError('report output must stay in the new campaign report directory')
    atomic_json(args.output/'report.json',result)
    _atomic_text(args.output/'humanreadable.txt',render(result))
    print(json.dumps(dict(complete=result['complete'],selection_rows=len(result['selection_table']),
        allocation_rows=len(result['allocation_table']),output=str(args.output))),flush=True)


if __name__=='__main__':
    main()
