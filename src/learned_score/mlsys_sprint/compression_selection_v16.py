"""Collect completed development shards and freeze one v16 family on time.

This CPU driver does not submit jobs, read evaluation outcomes, or fit after the
declared method-freeze deadline. A missing/failed development prerequisite
preserves the exact prior incumbent, explicitly without claiming improvement.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import fcntl
import json
from pathlib import Path
import time

from . import compression_policy_v16 as policy
from . import compression_diagnostics_v16 as diagnostics
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT='mlsys_compression_selection_v16'
MODELS=policy.MODELS


def required_stages(model):
    stages=[('features-'+split,'dev-features',split) for split in ('fit','screening')]
    if model in ('qwen25','phi35'):
        stages += [('labels-'+split,'physical',split) for split in ('fit','screening')]
    if model=='phi35':
        stages += [(prefix+split,phase,split) for split in ('fit','screening')
                   for prefix,phase in (('features-extra-','dev-features'),('labels-extra-','physical'))]
    return stages


def verify_study(study):
    if (study.get('sha256')!=digest({k:v for k,v in study.items() if k!='sha256'})
            or set(study.get('models',{}))!=set(MODELS)
            or study.get('final_selection_uses_evaluation_labels') is not False
            or study.get('independent_confirmation') is not False):
        raise ValueError('invalid v16 selection study or evidence scope')
    for path,sha in study['input_hashes'].items():
        if file_digest(path)!=sha:
            raise ValueError('frozen campaign input changed: '+path)
    return study


def pending_stages(study):
    from .compression_campaign_v16 import complete_plan
    root=Path(study['output_root'])
    pending=[]
    for model in MODELS:
        for stage in ['canary',*(item[0] for item in required_stages(model))]:
            path=root/'plans'/(stage+'-'+model)/'plan.json'
            if not complete_plan(path):
                pending.append({'model':model,'stage':stage,'plan':str(path),
                                'reason':'plan_not_published' if not path.exists() else 'shards_not_complete'})
    return pending


def collect_stage(study,model,stage,phase,split):
    """Both cooperative worker states AND every hashed marker must complete."""
    from .compression_campaign_v16 import complete_plan
    from .worker import read_plan,verify_sources
    root=Path(study['output_root'])
    plan_path=root/'plans'/(stage+'-'+model)/'plan.json'
    if not complete_plan(plan_path):
        return None
    plan=read_plan(plan_path)
    verify_sources(plan)
    task_path=plan_path.parent/'task.json'
    task=checked(task_path)
    expected_study=study
    if '-extra-' in stage:
        expected_study=checked(root/'supplemental-phi-study.json')
        if (model!='phi35' or expected_study.get('parent_study_sha256')!=study['sha256']
                or expected_study.get('supplemental_development_only') is not True):
            raise ValueError('supplemental Phi development provenance mismatch')
    if (task['study_sha256']!=expected_study['sha256'] or task['model_tag']!=model
            or task['split']!=split or split not in ('fit','screening')
            or Path(task['manifest']).resolve()!=Path(expected_study['development_manifests'][model]).resolve()):
        raise ValueError('development task points to wrong model, split, or manifest')
    output=Path(task['output_root']).resolve()
    if output!= (root/model/stage).resolve():
        raise ValueError('task output directory differs from declared stage')
    manifest=json.loads(Path(task['manifest']).read_text())
    entries=manifest['splits'][split]
    if task.get('request_ids') is not None:
        allowed=set(task['request_ids'])
        entries=[e for e in entries if e['row_hash'] in allowed]
        if {e['row_hash'] for e in entries}!=allowed:
            raise ValueError('task request IDs are outside declared development split')
    expected={e['row_hash']:e for e in entries}
    records,receipts,contracts={},{},set()
    for rank in range(len(plan['tasks'])):
        marker_path=output/f'rank{rank}.complete.json'
        metadata_path=output/f'rank{rank}.metadata.json'
        marker=checked(marker_path)
        metadata=json.loads(metadata_path.read_text())
        if (marker.get('passed') is not True or marker['task_sha256']!=task['sha256']
                or marker['identity_sha256']!=digest(metadata) or metadata['task_sha256']!=task['sha256']
                or metadata['phase']!=phase or metadata['split']!=split or metadata['model_tag']!=model
                or metadata['checkpoint_sha256']!=file_digest(study['models'][model]['checkpoint'])
                or metadata['manifest_sha256']!=file_digest(task['manifest'])):
            raise ValueError('completed shard metadata differs from frozen development task')
        if metadata.get('answers_loaded')!=(phase=='physical'):
            raise ValueError('feature extraction must be answer-free; labels only in physical phase')
        contracts.add(digest(metadata['cache_contract']))
        receipts[str(marker_path)]=file_digest(marker_path)
        receipts[str(metadata_path)]=file_digest(metadata_path)
        for filename,sha in marker['input_hashes'].items():
            path=Path(filename).resolve()
            if file_digest(path)!=sha:
                raise ValueError('completed development artifact modified: '+filename)
            receipts[str(path)]=sha
            directory=output/('canaries' if phase=='canary' else 'features' if phase=='dev-features' else 'rows')
            if path.parent!=directory:
                continue  # e.g. frozen feature inputs pinned by physical worker
            record=json.loads(path.read_text())
            key=record.get('request_id')
            if (key not in expected or record.get('identity_sha256')!=marker['identity_sha256']
                    or record.get('group_id')!=expected[key]['group_id']
                    or record.get('image_sha256')!=expected[key]['image_sha256']
                    or record.get('split')!=split or record.get('model_tag')!=model):
                raise ValueError('development record identity differs from its manifest/shard')
            record_key=key if phase!='physical' else (key,record['method'])
            if record_key in records:
                raise ValueError('duplicate completed development artifact')
            if phase!='physical':
                if record.get('answers_loaded') is not False or record.get('answers_consumed_by_feature_extractor') is not False:
                    raise ValueError('feature worker consumed labels')
                if phase=='canary':
                    n=record['input_visual_tokens']
                    outcomes=record.get('outcomes',[])
                    if record.get('passed') is not True or {o.get('retained_visual_tokens') for o in outcomes}!={n,(n+1)//2}:
                        raise ValueError('native canary did not cover full and partial counts')
                    for proof in outcomes:
                        if (proof.get('native_dynamic_ids_equal') is not True
                                or proof.get('protected_prefix_indices_preserved') is not True
                                or proof.get('embedding_gather_exact') is not True
                                or proof['cache']['compaction_events'] or proof['cache']['evicted_tokens']
                                or (proof['retained_visual_tokens']==n and proof.get('native_multimodal_ids_equal') is not True)):
                            raise ValueError('native canary generation/positions/prefix proof failed')
            else:
                if record.get('status')!='ok':
                    raise ValueError('unsuccessful physical measurement')
                contract=record['cache_contract']
                expected_bytes=(record['input_text_tokens']+record['selected_count']+contract['reserved_generated_slots'])*contract['total_owned_bytes_per_slot']
                if record.get('expected_cache_bytes')!=expected_bytes or record['metrics']['cache']['total_owned_bytes']!=expected_bytes:
                    raise ValueError('physical KV capacity differs from exact count contract')
            records[record_key]=record
    if len(contracts)!=1:
        raise ValueError('development shards disagree on cache contract')
    if phase!='physical':
        if set(records)!=set(expected):
            raise ValueError('completed feature shards do not cover exact development cohort')
    else:
        from .compression_execution_v16 import knot_cells
        observed={key for key,_ in records}
        if observed!=set(expected):
            raise ValueError('physical shards do not cover every admitted development request')
        for key in expected:
            example=next(v for (request,_),v in records.items() if request==key)
            wanted={(key,cell['key']) for cell in knot_cells(example['input_visual_tokens'],
                retentions=task['retentions'],selectors=task.get('selectors',['scorer']))}
            if {cell for cell in records if cell[0]==key}!=wanted:
                raise ValueError('completed physical shard is missing declared retention knots')
    receipts[str(plan_path)]=file_digest(plan_path)
    receipts[str(task_path)]=file_digest(task_path)
    return {'records':records,'receipts':receipts,'cache_contract':metadata['cache_contract']}


def collect_development(study,model):
    if model not in MODELS:
        raise ValueError('unknown model')
    root=Path(study['output_root'])
    canary=collect_stage(study,model,'canary','canary','fit')
    if canary is None:
        return None
    stages={}
    for stage,phase,split in required_stages(model):
        result=collect_stage(study,model,stage,phase,split)
        if result is None:
            return None
        stages[stage]=result
    rows,provenance,config=policy.load_development(study['cwd'],model)
    features={}
    physical=[]
    receipts=dict(canary['receipts'])
    for stage,result in stages.items():
        receipts.update(result['receipts'])
        if stage.startswith('features'):
            if set(features)&set(result['records']):
                raise ValueError('source request overlaps development feature stages')
            features.update(result['records'])
        else:
            physical.extend(result['records'].values())
    if model=='qwen25':
        # Qwen's current runtime changed learned prefixes relative to the old
        # cache. The WHOLE cohort therefore receives fresh physical labels,
        # including full, instead of selectively reusing favorable outcomes.
        # Passing old rows as identity-only entries discards every old label.
        rows=policy.extend_physical_development([],features,physical,additional_entries=rows)
        provenance={**provenance,'historical_quality_labels_ignored':True,
                    'reason':'current_runtime_learned_prefixes_differ_from_historical_cache',
                    'quality_source':'complete_v16_physical_retention_matrix_including_full'}
    elif model=='phi35':
        supplemental=checked(root/'supplemental-phi-study.json')
        manifest_path=Path(supplemental['development_manifests'][model])
        manifest=json.loads(manifest_path.read_text())
        if (manifest['metadata'].get('development_only') is not True
                or manifest['metadata'].get('identity_only_admission') is not True
                or manifest['metadata'].get('selection_uses_labels') is not False):
            raise ValueError('supplemental development admission is not identity-only')
        extras=[r for split in ('fit','screening') for r in manifest['splits'][split]]
        rows=policy.extend_physical_development(rows,features,physical,additional_entries=extras)
        receipts[str(manifest_path)]=file_digest(manifest_path)
        receipts[str(root/'supplemental-phi-study.json')]=file_digest(root/'supplemental-phi-study.json')
    else:
        rows=policy.merge_features(rows,features)
    for stage,result in stages.items():
        contract=result['cache_contract']
        if (contract['kv_bytes_per_slot']!=config['bytes_per_token']
                or contract['reserved_generated_slots']!=config['generation_reserve']):
            raise ValueError('historical and new development memory contracts differ')
    if set(r['dataset'] for r in rows)!=set(policy.base.DATASETS) or set(policy.available_kinds(rows))!=set(policy.KINDS):
        raise ValueError('all six candidates and all four development domains are required')
    return {'rows':rows,'configuration':config,'provenance':provenance,'input_hashes':receipts,
            'domain_requests':dict(Counter(r['dataset'] for r in rows)),
            'domain_groups':{d:len({r['group_id'] for r in rows if r['dataset']==d}) for d in policy.base.DATASETS}}


def _wrap_incumbent(rule,model,source_path):
    if (rule.get('format')!=policy.base.FORMAT or rule.get('kind')!='rank32_coverage'
            or rule.get('sha256')!=digest({k:v for k,v in rule.items() if k!='sha256'})):
        raise ValueError('incumbent must be intact frozen rank32 coverage')
    return seal({'format':policy.FORMAT,'kind':'rank32_coverage',
        'definition':copy.deepcopy(policy.CANDIDATES['rank32_coverage']),
        'model':model,'profile':'legacy_incumbent','base_rule':rule,'threshold':rule['threshold'],
        'bytes_per_token':rule['bytes_per_token'],'generation_reserve':rule['generation_reserve'],
        'fit_summary':rule['fit_summary'],'physical_labels_only':True,'scorer_changed':False,
        'source_rows_sha256':rule['fit_data_sha256'],'source_incumbent':str(source_path),
        'legacy_retention_percentages':[30,50,70,90,100] if model=='phi35' else list(range(10,101,10)),
        'source_incumbent_sha256':file_digest(source_path),'independent_confirmation':False})


def incumbent_rules(study):
    visual_path=Path(study['source_studies']['incumbent']).parent/'decision.json'
    phi_path=Path(study['source_studies']['phi35']).parent/'calibration.json'
    visual,phi=checked(visual_path),checked(phi_path)
    rules={model:_wrap_incumbent(visual['model_rules'][model],model,visual_path) for model in ('qwen25','llava_next')}
    rules['phi35']=_wrap_incumbent(phi['rules']['original'],'phi35',phi_path)
    return rules,{str(visual_path):file_digest(visual_path),str(phi_path):file_digest(phi_path)}


def _freeze_incumbent(study,reason,remaining,clock):
    root=Path(study['output_root'])
    rules,pins=incumbent_rules(study)
    decision=seal({'format':FORMAT+'_decision','study_sha256':study['sha256'],
        'status':'frozen_incumbent_'+reason,'reason':reason,'selected_family':'rank32_coverage',
        'rules':{'primary':rules},'incumbent_rules':rules,'remaining':remaining,
        'primary_quality_gate_passed':False,'improvement_claim_supported':False,
        'freeze_utc_epoch':clock(),'method_freeze_utc_epoch':study['method_freeze_utc_epoch'],
        'input_hashes':pins,'final_selection_uses_evaluation_labels':False,
        'independent_confirmation':False,'nonprimary_profiles_not_fabricated':True})
    write_immutable(root/'decision.json',decision)
    return decision


def advance(study,*,now=None):
    """Safe for a once-per-minute existing CPU watch; never launch GPU work."""
    verify_study(study)
    root=Path(study['output_root'])
    root.mkdir(parents=True,exist_ok=True)
    clock=time.time if now is None else lambda:now
    with (root/'selection.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        decision_path=root/'decision.json'
        if decision_path.exists():
            decision=checked(decision_path)
            if decision['study_sha256']!=study['sha256']:
                raise ValueError('frozen selection belongs to another study')
            for path,sha in decision['input_hashes'].items():
                if file_digest(path)!=sha:
                    raise ValueError('frozen selection input changed: '+path)
            return decision
        remaining=pending_stages(study)
        if clock()>=study['method_freeze_utc_epoch']:
            return _freeze_incumbent(study,'development_incomplete_at_deadline',remaining,clock)
        if remaining:
            status={'format':FORMAT+'_status','status':'waiting_for_complete_development',
                    'remaining':remaining,'time':clock(),'freeze_utc_epoch':study['method_freeze_utc_epoch'],
                    'evaluation_labels_opened':False}
            atomic_json(root/'selection-status.json',status)
            return status
        reports,receipts={},{}
        for model in MODELS:
            data=collect_development(study,model)
            if data is None:
                raise ValueError('completed plan lost artifacts during collection')
            receipts.update(data['input_hashes'])
            destination=root/'development-final'/model
            rows_path=destination/'rows.json'
            write_immutable(rows_path,seal(data))
            receipts[str(rows_path)]=file_digest(rows_path)
            reports[model]={}
            for profile in policy.PROFILES:
                if clock()>=study['method_freeze_utc_epoch']:
                    return _freeze_incumbent(study,'calibration_not_completed_before_deadline',[],clock)
                path=destination/(profile+'.json')
                if path.exists():
                    report=checked(path)
                    if report['input_rows_sha256']!=digest(data['rows']):
                        raise ValueError('cached final-development report belongs to changed rows')
                else:
                    report=policy.development_report(data['rows'],model=model,profile=profile,**data['configuration'])
                    write_immutable(path,report)
                reports[model][profile]=report
                receipts[str(path)]=file_digest(path)
            # Run/report diagnostics concurrently AFTER the freeze in the root
            # CPU driver; they must not delay the strict selection deadline.
        operating={profile:{model:policy.select_operating_points(reports[model],target_profile=profile)
                            for model in MODELS} for profile in policy.PROFILES}
        for profile,values in operating.items():
            for model,report in values.items():
                path=root/'development-final'/model/('operating-point-'+profile+'.json')
                write_immutable(path,report)
                receipts[str(path)]=file_digest(path)
        primary=policy.select_shared(operating['primary'])
        shared_path=root/'development-final'/'shared-primary.json'
        write_immutable(shared_path,primary)
        receipts[str(shared_path)]=file_digest(shared_path)
        if clock()>=study['method_freeze_utc_epoch']:
            return _freeze_incumbent(study,'calibration_not_completed_before_deadline',[],clock)
        if primary['rule_kind'] is None:
            return _freeze_incumbent(study,'no_qualifying_family',[],clock)
        kind=primary['rule_kind']
        incumbents,pins=incumbent_rules(study)
        receipts.update(pins)
        for module in (Path(__file__),Path(policy.__file__),Path(policy.base.__file__),Path(policy.base.previous.__file__)):
            receipts[str(module.resolve())]=file_digest(module)
        decision=seal({'format':FORMAT+'_decision','study_sha256':study['sha256'],
            'status':'frozen_candidate','selected_family':kind,'primary_quality_gate_passed':True,
            'rules':{p:{m:operating[p][m]['final_rules'][kind] for m in MODELS} for p in policy.PROFILES},
            'profile_gates':{p:{m:operating[p][m]['candidates'][kind]['eligible'] for m in MODELS} for p in policy.PROFILES},
            'calibration_profiles':{p:{m:operating[p][m]['choices'][kind]['calibration_profile'] for m in MODELS} for p in policy.PROFILES},
            'incumbent_rules':incumbents,'primary_selection':primary,
            'freeze_utc_epoch':clock(),'method_freeze_utc_epoch':study['method_freeze_utc_epoch'],
            'input_hashes':receipts,'final_selection_uses_evaluation_labels':False,
            'independent_confirmation':False,'physical_matched_benchmarks_required':True,
            'nonprimary_failed_gates_are_exploratory_tradeoff_points':True})
        write_immutable(decision_path,decision)
        return decision


def final_diagnostics(study):
    """Optional CPU follow-up, never changes a frozen rule or decision."""
    root=Path(study['output_root'])
    decision=checked(root/'decision.json')
    if decision['study_sha256']!=study['sha256']:
        raise ValueError('diagnostic study differs from frozen decision')
    produced=[]
    for model in MODELS:
        source=root/'development-final'/model/'rows.json'
        if not source.exists():
            continue
        output=root/'diagnostics-final'/(model+'.json')
        if output.exists():
            checked(output)
            produced.append(str(output))
            continue
        data=checked(source)
        value=diagnostics.report(data['rows'],model=model,**data['configuration'])
        write_immutable(output,value)
        produced.append(str(output))
    return produced


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--once',action='store_true')
    parser.add_argument('--diagnostics',action='store_true')
    args=parser.parse_args(argv)
    study=checked(args.study)
    while True:
        try:
            result=advance(study)
        except (ValueError,OSError,KeyError) as exc:
            atomic_json(Path(study['output_root'])/'selection-error.json',
                        {'time':time.time(),'error_type':type(exc).__name__,'error':str(exc),
                         'scientific_gate_not_bypassed':True})
            raise
        print(json.dumps({'status':result['status'],'remaining':len(result.get('remaining',[])),
                          'selected_family':result.get('selected_family')}),flush=True)
        if result['status'].startswith('frozen'):
            if args.diagnostics:
                print(json.dumps({'diagnostic_paths':final_diagnostics(study)}),flush=True)
            return
        if args.once:
            return
        time.sleep(30)


if __name__=='__main__':
    main()
