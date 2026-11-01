"""Strict, completed-cohort reporting for the frozen coverage diagnostic.

No inference, method selection, or policy fitting. Original question-level MME
estimand is retained and labeled; this is not the official full MME score.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics

from . import coverage_audit as original
from .coverage_audit_report import comparison_tables, group_ci
from .compression_matrix_v8 import checked
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_coverage_audit_resume_strict_report_v1'


class Pins:
    def __init__(self):
        self.values = {}

    def add(self, path, expected=None):
        path = str(Path(path).resolve())
        actual = self.values.get(path)
        if actual is None:
            actual = file_digest(path)
            self.values[path] = actual
        if expected is not None and actual != expected:
            raise ValueError('changed pinned coverage input: '+path)
        return actual

    def verify(self, values):
        if not isinstance(values, dict) or not values:
            raise ValueError('nonempty coverage source pins required')
        for path, sha in values.items():
            self.add(path, sha)


def finite(value, name, *, positive=False):
    if (type(value) not in (int,float) or not math.isfinite(value)
            or value < 0 or (positive and value == 0)):
        raise ValueError('invalid coverage measurement: '+name)
    return value


def cpu_reuse(audit, root, pins):
    path = Path(audit['cpu_reuse_receipt'])
    if path.resolve() != root/'cpu-reuse.json':
        raise ValueError('CPU reuse receipt outside resumed root')
    receipt = checked(path); pins.add(path)
    source = Path(audit['source_audit']).resolve().parent
    if (receipt.get('format') != 'mlsys_coverage_audit_resume_v1_cpu_reuse'
            or Path(receipt['source_root']).resolve() != source
            or Path(receipt['output_root']).resolve() != root
            or receipt.get('source_audit_sha256') != audit['source_audit_sha256']
            or receipt.get('probe_count') != 24 or receipt.get('probes_per_model') != 8
            or receipt.get('cpu_inference_rerun') is not False
            or receipt.get('new_physical_evidence') is not False):
        raise ValueError('CPU reuse provenance differs from frozen source')
    pins.verify(receipt['input_hashes'])
    expected = {Path('cpu')/name for name in ('synthetic.json','certification.json')}
    observed = set(); values = defaultdict(list)
    for item in receipt['copies']:
        old, new = Path(item['source']).resolve(), Path(item['destination']).resolve()
        if not old.is_relative_to(source/'cpu') or new != root/old.relative_to(source):
            raise ValueError('CPU reused copy escapes source/destination tree')
        relative = old.relative_to(source)
        if relative in observed or item.get('byte_identical') is not True:
            raise ValueError('duplicate or nonidentical CPU reuse')
        observed.add(relative); pins.add(old,item['sha256']); pins.add(new,item['sha256'])
        value = original.read(new)
        if len(relative.parts)==3:
            model = relative.parts[1]
            if (model not in original.MODELS or value.get('model') != model
                    or value.get('request_id') != new.stem
                    or value.get('training_cache_not_evaluation') is not True
                    or value.get('answer_quality_measured') is not False):
                raise ValueError('CPU probe is not the declared training-only geometry')
            values[model].append(value); expected.add(relative)
    if observed != expected or len(observed)!=26 or any(len(values[m])!=8 for m in original.MODELS):
        raise ValueError('incomplete frozen 24-probe CPU reuse')
    summary = {}
    for model in original.MODELS:
        valid = [v for v in values[model] if v['status']=='ok']
        summary[model] = {'examples':8,'successful':len(valid),
            'timeouts':sum(v['status']=='selection_timeout' for v in values[model]),
            'mean_curvature':statistics.mean(v['curvature']['curvature'] for v in valid) if valid else None,
            'mean_finer_menu_retention_saving_pp':statistics.mean(v['saved_retention_percentage_points'] for v in valid) if valid else None,
            'not_answer_accuracy':True,'copied_byte_identically_not_rerun':True}
    return {'receipt':str(path),'receipt_sha256':pins.add(path),'verified':True,'training_geometry':summary}


def validate_row(row, task, entry, method):
    if (row.get('format')!=original.FORMAT+'_physical' or row.get('task_sha256')!=task['sha256']
            or row.get('model')!=task['model_tag'] or row.get('request_id')!=entry['row_hash']
            or row.get('group_id')!=entry['group_id'] or row.get('image_sha256')!=entry['image_sha256']
            or row.get('dataset')!=original.family(entry['dataset']) or row.get('category')!=entry.get('category')
            or row.get('quality_metric')!=entry.get('quality_metric',entry.get('metric'))
            or row.get('method')!=method or row.get('split')!='screening'
            or row.get('independent_confirmation') is not False
            or row.get('physical_measurement_performed') is not True
            or row.get('allocation_latency_included') is not True or row.get('timing_isolated') is not False):
        raise ValueError('physical coverage row differs from frozen identity')
    if row.get('status')=='selection_timeout':
        if row.get('quality_score') is not None or not row.get('error'):
            raise ValueError('timeout cannot carry a successful quality score')
        return
    if row.get('status')!='ok':
        raise ValueError('unknown physical coverage row status')
    n,k,p = row['input_visual_tokens'],row['selected_count'],row['input_text_tokens']
    ids = row['selected_patch_indices']
    if (any(type(v) is not int for v in (n,k,p)) or not 1<=k<=n or p<0
            or ids!=sorted(set(ids)) or len(ids)!=k or any(type(v) is not int or not 0<=v<n for v in ids)
            or row['selected_patch_indices_sha256']!=digest(ids)
            or row['visual_retention']!=k/n):
        raise ValueError('selected patches violate exact count/original order')
    if method=='all_tokens' and ids!=list(range(n)):
        raise ValueError('full baseline is not all image patches')
    if method.endswith(('_050','_070')) and k!=(n*int(method[-3:])+99)//100:
        raise ValueError('fixed-count diagnostic changed its integer budget')
    if row.get('fallback_reason') is not None and (not method.endswith('_tau') or k!=n):
        raise ValueError('invalid or hidden fallback')
    q=finite(row['quality_score'],'quality')
    if q>1:raise ValueError('quality must lie in [0,1]')
    for field in ('selector_seconds','end_to_end_seconds','ttft_seconds'):
        finite(row[field],field)
    contract,cache=row['cache_contract'],row['metrics']['cache']
    slots=p+k+contract['reserved_generated_slots']
    if (cache['kv_capacity_bytes']!=slots*contract['kv_bytes_per_slot']
            or cache['total_owned_bytes']!=slots*contract['total_owned_bytes_per_slot']
            or row['expected_cache_bytes']!=cache['total_owned_bytes']
            or row['metrics']['peak_cache_bytes']!=cache['total_owned_bytes']
            or cache['scratch_bytes']!=0 or cache['compaction_events']!=0):
        raise ValueError('physical cache differs from exact selected-token geometry')
    tokens=row['metrics']['generated_token_ids']; eos=row['metrics']['effective_eos_token_ids']
    if (not tokens or len(tokens)!=row['generated_tokens'] or len(tokens)>128
            or any(type(t) is not int or t<0 for t in tokens)
            or any(t in eos for t in tokens[:-1])
            or (tokens[-1] not in eos and len(tokens)!=128)
            or row['metrics'].get('stop_on_eos') is not True):
        raise ValueError('invalid natural-stopping generation trace')


def collect(root):
    root=Path(root).resolve();audit=checked(root/'audit.json');pins=Pins();pins.add(root/'audit.json')
    if (audit.get('format')!=original.FORMAT or Path(audit['output_root']).resolve()!=root
            or audit.get('scientific_protocol_unchanged') is not True
            or audit.get('independent_confirmation') is not False):
        raise ValueError('strict report requires a frozen resumed development audit')
    pins.verify(audit['input_hashes'])
    resume_path=root/'resume-receipt.json';resume=checked(resume_path);pins.add(resume_path)
    if (resume.get('audit_sha256')!=audit['sha256'] or resume.get('source_audit_sha256')!=audit['source_audit_sha256']
            or resume.get('scientific_protocol_unchanged') is not True or resume.get('physical_outcomes_read') is not False):
        raise ValueError('resume receipt does not bind the frozen audit')
    pins.verify(resume['input_hashes']);pins.verify(resume['output_hashes'])
    parent=checked(audit['source_audit']);pins.add(audit['source_audit'])
    if (parent['sha256']!=audit['source_audit_sha256'] or parent['panels']!=audit['panels']
            or parent['rules']!=audit['rules'] or set(parent['tasks'])!=set(audit['tasks'])
            or set(audit['tasks'])!=set(audit['plans'])):
        raise ValueError('resumed methods, panels, or policies differ from source audit')
    cpu=cpu_reuse(audit,root,pins)
    expected={};rows={};pending=[];failures=[];shards=[]
    for label in sorted(audit['tasks']):
        task_path,plan_path=Path(audit['tasks'][label]).resolve(),Path(audit['plans'][label]).resolve()
        if task_path!=root/'tasks'/(label+'.json') or plan_path!=root/'plans'/label/'plan.json':
            raise ValueError('task or plan outside declared resumed root')
        task,plan=checked(task_path),checked(plan_path);pins.add(task_path);pins.add(plan_path)
        original.execution.validate_task(task,'physical',verify_files=False)
        pins.verify(task['input_hashes']);pins.verify(plan['input_hashes'])
        source_task_path=Path(parent['tasks'][label]).resolve()
        source_plan_path=Path(parent['plans'][label]).resolve()
        if any(str(p) not in plan['input_hashes'] for p in (source_task_path,)):
            raise ValueError('resumed plan omits original task provenance')
        source_task=checked(source_task_path);source_plan=checked(source_plan_path)
        pins.add(source_task_path);pins.add(source_plan_path)
        changed={'sha256','output_root','deadline_utc_epoch','input_hashes'}
        if any(task.get(k)!=v for k,v in source_task.items() if k not in changed):
            raise ValueError('resumed task changed a frozen scientific input')
        if len(plan['tasks'])!=len(source_plan['tasks']):
            raise ValueError('resumed plan changed source-group shard count')
        model=task['model_tag'];directory=root/'physical'/label
        panel=next((p for p in audit['panels'][model] if p['label']==label),None)
        if (task.get('audit_format')!=original.FORMAT or task.get('audit_methods')!=list(original.METHODS)
                or task['split']!='screening' or task['audit_rule']!=audit['rules'][model]
                or Path(task['output_root']).resolve()!=directory or panel is None
                or task['request_ids']!=panel['request_ids']):
            raise ValueError('task differs from frozen coverage panel/rule/methods')
        manifest=original.read(task['manifest'])
        entries={e['row_hash']:e for e in manifest['splits']['screening'] if e['row_hash'] in task['request_ids']}
        if set(entries)!=set(task['request_ids']) or sorted({e['group_id'] for e in entries.values()})!=panel['group_ids']:
            raise ValueError('panel identities or source groups differ from manifest')
        for key,entry in entries.items():
            if (model,key) in expected:raise ValueError('duplicate request across resumed tasks')
            expected[model,key]=entry
        groups=panel['group_ids'];world=len(plan['tasks'])
        if not world:raise ValueError('empty physical plan')
        for rank,record in enumerate(plan['tasks']):
            marker_path=directory/f'rank{rank}.complete.json'
            artifact=[a for a in record['artifacts'] if Path(a['path']).resolve()==marker_path]
            if len(artifact)!=1 or artifact[0].get('passed') is not True:
                raise ValueError('plan does not bind its expected shard marker')
            shard=[e for e in entries.values() if e['group_id'] in set(groups[rank::world])]
            wanted={directory/'rows'/f'{e["row_hash"]}.{m}.json':(e,m) for e in shard for m in original.METHODS}
            if not marker_path.exists():
                pending.append(str(marker_path));continue
            marker=checked(marker_path);pins.add(marker_path)
            canary_path=directory/f'rank{rank}.canary.json'
            if (marker.get('format')!=original.FORMAT+'_complete' or marker.get('task_sha256')!=task['sha256']
                    or marker.get('passed') is not True or marker.get('requests')!=len(shard)
                    or {Path(p).resolve() for p in marker['input_hashes']}!=set(wanted)|{canary_path}):
                raise ValueError('completion marker does not bind exact prespecified shard')
            pins.verify(marker['input_hashes']);canary=original.read(canary_path)
            checks=canary.get('checks',[])
            if (canary.get('task_sha256')!=task['sha256'] or canary.get('passed') is not True or len(checks)!=2
                    or not all(c.get('native_dynamic_ids_equal') is True and c.get('embedding_gather_exact') is True
                        and c.get('protected_prefix_indices_preserved') is True and c.get('original_position_policy')=='original' for c in checks)
                    or not any(c.get('native_multimodal_ids_equal') is True for c in checks)):
                raise ValueError('full/partial original-position canary is absent or failed')
            failed=0
            for path,(entry,method) in wanted.items():
                row=original.read(path);validate_row(row,task,entry,method)
                key=(model,entry['row_hash'],method)
                if key in rows:raise ValueError('duplicate physical cell')
                rows[key]=row
                if row['status']!='ok':
                    failed+=1;failures.append({k:row.get(k) for k in ('model','dataset','request_id','method','status','error')})
            if marker.get('failed_measurements')!=failed or marker.get('all_measurements_succeeded')!=(failed==0):
                raise ValueError('marker conceals failed or timed-out cells')
            shards.append({'label':label,'rank':rank,'requests':len(shard),'failed_cells':failed})
        for path in sorted((directory/'errors').glob('*.json')):
            error=original.read(path)
            if (error.get('task_sha256')!=task['sha256'] or error.get('request_id') not in entries
                    or error.get('method') not in original.METHODS or error.get('status')!='failed'):
                raise ValueError('worker error log belongs to an undeclared cell')
            pins.add(path)
            failures.append({k:error.get(k) for k in ('model','dataset','request_id','method','status','error')}
                            | {'source_path':str(path),'diagnostic_error_log_not_successful_measurement':True})
    for model,key in expected:
        members=[r for (m,k,_),r in rows.items() if (m,k)==(model,key) and r['status']=='ok']
        if len({(r['input_visual_tokens'],r['input_text_tokens'],digest(r['cache_contract'])) for r in members})>1:
            raise ValueError('paired methods disagree on physical input/cache geometry')
    return audit,expected,rows,pending,failures,shards,cpu,pins


def tables(expected,rows):
    groups=defaultdict(list)
    for (model,key),entry in expected.items():groups[model,original.family(entry['dataset'])].append(key)
    fixed,adaptive,status=[],[],[]
    for (model,dataset),keys in sorted(groups.items()):
        if dataset=='MME':
            pairs=defaultdict(list)
            for key in keys:pairs[expected[model,key]['group_id']].append(key)
            if any(len(v)!=2 for v in pairs.values()):raise ValueError('frozen MME cohort lacks complete pairs')
        for kind,methods in [('fixed50',('all_tokens','importance_050','coverage_greedy_050')),
                ('fixed70',('all_tokens','importance_070','coverage_greedy_070')),
                ('adaptive',('all_tokens','importance_legacy_tau','importance_fine_tau','coverage_greedy_tau'))]:
            cells=[(model,key,m) for key in keys for m in methods]
            missing=[c for c in cells if c not in rows];bad=[c for c in cells if c in rows and rows[c]['status']!='ok']
            state='pending' if missing else 'finished_with_failures' if bad else 'complete'
            status.append({'model':model,'dataset':dataset,'comparison':kind,'status':state,
                'expected_requests':len(keys),'missing_cells':len(missing),'failed_cells':len(bad)})
            if state!='complete':continue
            subset={c:rows[c] for c in cells};f,a=comparison_tables(subset)
            if kind=='adaptive':adaptive.extend(a);continue
            for item in f:
                base=[rows[model,key,'all_tokens'] for key in keys]
                imp=[rows[model,key,item['reference']] for key in keys]
                greedy=[rows[model,key,item['method']] for key in keys]
                item.update(all_tokens_accuracy_percent=100*statistics.mean(r['quality_score'] for r in base),
                    all_tokens_mean_kv_capacity_mib=statistics.mean(r['metrics']['cache']['kv_capacity_bytes'] for r in base)/2**20,
                    importance_minus_all_tokens_pp=100*statistics.mean(a['quality_score']-b['quality_score'] for a,b in zip(imp,base)),
                    coverage_greedy_minus_all_tokens_pp=100*statistics.mean(a['quality_score']-b['quality_score'] for a,b in zip(greedy,base)),
                    importance_minus_all_tokens_ci95_pp=group_ci(list(zip(imp,base))),
                    coverage_greedy_minus_all_tokens_ci95_pp=group_ci(list(zip(greedy,base))),
                    complete_prespecified_cohort=True)
                fixed.append(item)
    return fixed,adaptive,status


def report(root):
    audit,expected,rows,pending,failures,shards,cpu,pins=collect(root)
    fixed,adaptive,cohorts=tables(expected,rows)
    result={'format':FORMAT,'audit_sha256':audit['sha256'],'complete':not pending,
        'status':'pending' if pending else 'finished_with_failures' if failures else 'complete',
        'scientific_success_claimed':False,'independent_confirmation':False,
        'expected_physical_cells':len(expected)*len(original.METHODS),'validated_physical_cells':len(rows),
        'pending_shards':pending,'completed_shards':shards,'physical_failures':failures,
        'cohorts':cohorts,'fixed_budget':fixed,'adaptive_compression':adaptive,'cpu_reuse':cpu,
        'mme_metric':'complete-pair question accuracy diagnostic; not official MME category macro/total',
        'latency_claim':'live selector included; shared-node development diagnostic, no isolated speedup claim',
        'completed_cohorts_only':True,'input_hashes':pins.values,'reporter_sha256':file_digest(__file__)}
    atomic_json(Path(root)/'report.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args(argv);value=report(args.root)
    print(json.dumps({k:value[k] for k in ('status','complete','expected_physical_cells','validated_physical_cells')}))


if __name__=='__main__':main()
