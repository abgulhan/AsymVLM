"""Conditionally admitted, frozen-policy full evaluation through the v5 runtime.

Feature extraction is answer-free. All quality runs use physically selected
integer token counts and exactly matched within-dataset aggregate KV budgets.
No development runner gate is bypassed: this module owns a separate final gate,
and delegates only the unchanged model inference implementation.
"""
import argparse
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

from . import compression_features_v12 as features
from . import compression_policy_v12 as policies
from . import compression_policy_v11 as budget_policy
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_full_suite_v12'


def admission_reasons(study, selection, admission):
    reasons = []
    if not admission:
        reasons.append('identity_audit_pending')
    elif (admission.get('study_sha256') != study['sha256'] or admission.get('labels_loaded') is not False):
        raise ValueError('full evaluation admission identity changed')
    elif not admission.get('manifests'):
        reasons.append('identity_admission_blocked')
    if not selection:
        reasons.append('method_not_frozen')
    elif selection.get('study_sha256') != study['sha256']:
        raise ValueError('frozen selection belongs to another study')
    elif selection.get('frozen') is not True:
        reasons.append('deployment_not_frozen')
    elif selection.get('rule_kind') is None:
        reasons.append('no_supported_shared_rule')
    elif selection.get('provisional_pending_overhead'):
        reasons.append('native_coverage_overhead_gate_pending')
    return reasons


def _pins(gate):
    for path, expected in gate['input_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('full-suite frozen input changed: '+path)


def freeze_gate(study, selection, admission):
    from learned_score.mlsys_eval.common import load_manifest, seal_manifest
    from .compression_study_v12 import verify
    from .compression_data_v9 import family, identity_only
    verify(study)
    if admission_reasons(study, selection, admission):
        raise ValueError('full-suite admission remains closed')
    root = Path(study['output_root'])
    destination = root/'full-suite/gate.json'
    if destination.exists():
        gate = checked(destination)
        if gate['selection_sha256'] != selection['sha256'] or gate['admission_sha256'] != admission['sha256']:
            raise ValueError('cannot replace the pre-label full-suite gate')
        _pins(gate)
        return gate
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('full-suite publication deadline expired')
    if checked(root/'deployment.json') != selection or checked(root/'admission/status.json') != admission:
        raise ValueError('full-suite inputs differ from immutable publications')
    entries, pins = [], {**study['input_hashes'], **admission['input_hashes']}
    for domain, filename in sorted(admission['manifests'].items()):
        record, path = admission['domains'][domain], Path(filename)
        if (not record.get('eligible') or record.get('status') != 'admitted'
                or record.get('image_exclusions_verified') is not True
                or record.get('manifest_sha256') != file_digest(path) or record.get('blocking_reasons')):
            raise ValueError('domain is not source/image verified: '+domain)
        manifest = load_manifest(path)
        if (manifest['metadata'].get('fresh_identity_audit_passed') is not True
                or manifest['metadata'].get('complete_available_eligible_evaluation_split') is not True
                or any(manifest['splits'].get(k) for k in ('fit', 'screening', 'phi_train'))):
            raise ValueError('final manifest must include complete eligible evaluation rows only')
        rows = identity_only(manifest['splits']['evaluation'])
        if not rows or any(family(r['dataset']) != domain or not r.get('image_sha256') for r in rows):
            raise ValueError('final manifest missing domain/image provenance')
        entries.extend(rows)
        pins[str(path)] = file_digest(path)
    if len({r['row_hash'] for r in entries}) != len(entries):
        raise ValueError('duplicate full-suite request identity')
    models = {}
    for model, selected in selection['models'].items():
        rule = selected['rule']
        if any(r['group_id'] in rule['fit_group_ids'] or r['row_hash'] in rule['fit_row_hashes'] for r in entries):
            raise ValueError('final identities overlap policy fitting')
        development_path = root/model/'development/rows.json'
        development = checked(development_path)['rows']
        selector = 'native' if rule['kind'] == 'native_coverage' else 'scorer'
        kind = 'uniform' if rule['kind'] == 'static' else 'missing_mass' if rule['kind'] == 'missing_mass' else 'uncovered_mass'
        models[model] = {'primary': rule, 'sweeps': {
            str(target): budget_policy.fit_rule(development, selector, kind, target/100)
            for target in (50, 70, 90)}}
        pins[str(development_path)] = file_digest(development_path)
    combined = seal_manifest({'format': 'mlsys_adaptive_data_v3', 'dataset': 'ASYMVLM_FULL_SUITE_V12',
        'metadata': {'study_sha256': study['sha256'], 'fresh_identity_audit_passed': True,
                     'complete_available_eligible_evaluation_split': True, 'selection_uses_labels': False},
        'splits': {'fit': [], 'screening': [], 'evaluation': sorted(entries, key=lambda r:(r['dataset'],r['row_hash'])), 'phi_train': []}})
    manifest_path = root/'full-suite/manifest.json'
    write_immutable(manifest_path, combined)
    for path in (manifest_path, root/'study.json', root/'deployment.json', root/'admission/status.json', Path(__file__)):
        pins[str(path.resolve())] = file_digest(path)
    gate = seal({'format': FORMAT+'_gate', 'study_sha256': study['sha256'],
        'selection_sha256': selection['sha256'], 'admission_sha256': admission['sha256'],
        'manifest': str(manifest_path), 'manifest_sha256': file_digest(manifest_path),
        'models': models, 'input_hashes': pins, 'created_utc_epoch': time.time(),
        'labels_opened': False, 'policies_frozen_before_labels': True,
        'request_ids': [r['row_hash'] for r in combined['splits']['evaluation']]})
    _pins(gate)
    write_immutable(destination, gate)
    return gate


class FinalAnswerFreeDataset(features.previous.AnswerFreeDataset):
    def __init__(self, manifest_path, *, project_root):
        from .adaptive_data_v3 import load_manifest
        self.manifest = load_manifest(manifest_path)
        self.root = Path(project_root).resolve()
        self.allowed = {row['row_hash']: row for row in self.manifest.splits['evaluation']}
        self._verified, self._tables = set(), {}


def feature_worker(study, model, rank):
    import torch
    from . import adaptive_experiment_v3 as runtime
    from .adaptive_data_v3 import load_manifest
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    root = Path(study['output_root'])
    gate = checked(root/'full-suite/gate.json')
    _pins(gate)
    if gate['study_sha256'] != study['sha256'] or model not in gate['models'] or not 0 <= rank < study['world_size']:
        raise ValueError('invalid full-suite feature task')
    spec, manifest = study['models'][model], load_manifest(gate['manifest'])
    args = SimpleNamespace(adapter=spec['adapter'], model=spec['model_path'], model_revision=spec['revision'],
        checkpoint=Path(spec['checkpoint']), manifest=Path(gate['manifest']), split='evaluation',
        training_manifest=Path(study['cwd'])/'reproduction_results/mlsys_20261012/manifest.json',
        device='cuda:0', max_visual_patches=spec.get('max_visual_patches',1280),
        attention_implementation='sdpa', backbone_dtype='bfloat16', dataset_splits=('scorer_dev',))
    adapter, _, backbone, scorer = runtime.load_runtime(args, manifest)
    dataset = FinalAnswerFreeDataset(gate['manifest'], project_root=study['cwd'])
    adapter.dataset = dataset
    contract = runtime.contract_from_config(adapter.model.config)
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    directory = root/model/'full-features'
    metadata = {'format': FORMAT+'_features', 'gate_sha256': gate['sha256'], 'model': model,
        'checkpoint_sha256': file_digest(spec['checkpoint']), 'world_size': study['world_size'],
        'cache_contract': contract, 'backbone': backbone, 'answers_loaded': False,
        'source_hashes': features.source_pins(spec['adapter'])}
    write_immutable(directory/f'rank{rank}.metadata.json', metadata)
    pins = {}
    entries = sorted(manifest.splits['evaluation'], key=lambda r:(r['dataset'],r['row_hash']))[rank::study['world_size']]
    for entry in entries:
        if time.time() >= study['deadline_utc_epoch']:
            raise ValueError('full-suite feature deadline expired')
        path = directory/'curves'/(entry['row_hash']+'.json')
        legacy_path = directory/'features'/(entry['row_hash']+'.json')
        if path.exists() and legacy_path.exists():
            value, legacy = checked(path), json.loads(legacy_path.read_text())
            if value['gate_sha256'] != gate['sha256'] or value['legacy_sha256'] != file_digest(legacy_path):
                raise ValueError('full feature resume provenance changed')
        else:
            with torch.no_grad():
                prepared = adapter.prepare_sample(dataset.row_for_entry(entry), include_answer=False)
                curves, old = features.summarize(prepared, scorer), runtime.feature_summary(prepared, scorer)
                del prepared
            legacy = {'format': 'mlsys_answer_free_features_v3', 'measurement_version': 'v4',
                'request_id': entry['row_hash'], 'group_id': entry['group_id'], 'image_sha256': entry['image_sha256'],
                'dataset': entry['dataset'], 'split': 'evaluation', 'identity_sha256': digest(metadata),
                'checkpoint_sha256': metadata['checkpoint_sha256'], 'cache_contract': contract,
                'answers_consumed_by_feature_extractor': False, **old}
            write_immutable(legacy_path, legacy)
            value = seal({'gate_sha256': gate['sha256'], 'legacy_sha256': file_digest(legacy_path),
                'request_id': entry['row_hash'], 'group_id': entry['group_id'], 'dataset': entry['dataset'],
                'input_visual_tokens': old['input_visual_tokens'], 'input_text_tokens': old['input_text_tokens'],
                'curves': curves['curves'], 'answers_loaded': False})
            write_immutable(path, value)
        assert_frozen(adapter.model, versions)
        assert_frozen(scorer, scorer_versions)
        pins[entry['row_hash']] = {'curve': file_digest(path), 'legacy': file_digest(legacy_path)}
    write_immutable(directory/f'rank{rank}.complete.json',
                    {'identity_sha256': digest(metadata), 'requests': len(pins), 'rows': pins})


def collect(study, model, gate):
    directory = Path(study['output_root'])/model/'full-features'
    if not all((directory/f'rank{r}.complete.json').exists() for r in range(study['world_size'])):
        return None
    result, contracts = {}, set()
    for rank in range(study['world_size']):
        meta = json.loads((directory/f'rank{rank}.metadata.json').read_text())
        marker = json.loads((directory/f'rank{rank}.complete.json').read_text())
        if (meta['gate_sha256'] != gate['sha256'] or marker['identity_sha256'] != digest(meta)
                or meta.get('model') != model or meta.get('world_size') != study['world_size']
                or meta.get('answers_loaded') is not False
                or marker.get('requests') != len(marker['rows'])
                or meta.get('checkpoint_sha256') != file_digest(study['models'][model]['checkpoint'])
                or meta.get('source_hashes') != features.source_pins(study['models'][model]['adapter'])):
            raise ValueError('mixed full-suite feature identities')
        contracts.add(digest(meta['cache_contract']))
        for request, pins in marker['rows'].items():
            path, legacy = directory/'curves'/(request+'.json'), directory/'features'/(request+'.json')
            if file_digest(path) != pins['curve'] or file_digest(legacy) != pins['legacy'] or request in result:
                raise ValueError('full-suite atomic feature shard changed or duplicated')
            value = checked(path)
            if value['gate_sha256'] != gate['sha256'] or value['legacy_sha256'] != pins['legacy']:
                raise ValueError('full-suite curve identity differs')
            result[request] = {**value, 'row_hash': request}
    if set(result) != set(gate['request_ids']) or len(contracts) != 1:
        raise ValueError('full-suite incomplete cohort or mixed cache contracts')
    return list(result.values()), meta['cache_contract']


def make_schedule(study, model, gate, rows, contract, target='primary'):
    from .compression_data_v9 import family
    rule = gate['models'][model]['primary'] if target == 'primary' else gate['models'][model]['sweeps'][str(target)]
    free = policies.answer_free(rows)
    output = policies.infer(rule, free) if target == 'primary' else budget_policy.predict(rule, free)
    counts = output['counts']
    rows = [{**r, 'dataset': family(r['dataset'])} for r in rows]
    factorial = budget_policy.factorial_assignments(rows, counts, 'divprune')
    cells = {k: v for k,v in factorial['cells'].items() if k != 'divprune_adaptive'}
    if model == 'llava_next':
        cells['cdpruner_uniform'] = {'selector': 'cdpruner', 'counts': factorial['counts']['uniform']}
    if target == 'primary':
        cells['all_tokens'] = {'selector': 'full', 'counts': {r['row_hash']:r['input_visual_tokens'] for r in rows}}
    scheduled = {}
    for row in rows:
        request, n = row['row_hash'], row['input_visual_tokens']
        scheduled[request] = {k: row[k] for k in ('row_hash','group_id','dataset','input_visual_tokens','input_text_tokens')}
        scheduled[request]['cells'] = [{'key':key, 'selector':'full' if c['counts'][request] == n else c['selector'],
            'selected_count':c['counts'][request], 'mode':'exact', 'policy':'frozen_full_suite_v12'} for key,c in cells.items()]
    return seal({'format': FORMAT+'_schedule', 'study_sha256': study['sha256'], 'gate_sha256': gate['sha256'],
        'manifest_sha256': gate['manifest_sha256'], 'checkpoint_sha256': file_digest(study['models'][model]['checkpoint']),
        'split':'evaluation', 'cache_contract':contract, 'model':model, 'rows':scheduled,
        'answers_used_to_choose_counts':False, 'exact_aggregate_kv_matched_within_dataset':True,
        'policy_sha256':rule['sha256'], 'feature_rows_sha256':digest(rows), 'quality_requires_physical_inference':True})


def advance(study, selection=None):
    from .compression_execution_v12 import publish, make_task, python
    from .compression_study_v12 import verify
    from .selector_experiment_v5 import source_hashes
    verify(study)
    root = Path(study['output_root'])
    admission = checked(root/'admission/status.json') if (root/'admission/status.json').exists() else None
    selection = selection or (checked(root/'deployment.json') if (root/'deployment.json').exists() else None)
    reasons = admission_reasons(study, selection, admission)
    if reasons:
        status = {'status':'blocked', 'blocking_reasons':reasons, 'plans_published':False, 'labels_opened':False}
        atomic_json(root/'full-suite/status.json', status)
        return status
    gate = freeze_gate(study, selection, admission)
    paths = []
    for model, spec in study['models'].items():
        directory = root/model/'full-features'
        commands = []
        for rank in range(study['world_size']):
            argv = python(study, 'compression_full_suite_v12')+['features','--study',str(root/'study.json'),
                '--model',model,'--rank',str(rank)]
            commands.append(make_task(study,f'full-features-{model}-{rank}',argv,
                                      [directory/f'rank{rank}.complete.json'],model=model))
        extra = {**gate['input_hashes'],str(root/'full-suite/gate.json'):file_digest(root/'full-suite/gate.json')}
        paths.append(str(publish(study,'full-features-'+model,commands,extra)))
        collected = collect(study,model,gate)
        if collected is None:
            continue
        rows, contract = collected
        for target in ('primary',50,70,90):
            stage = 'full-primary' if target == 'primary' else f'full-p{target:03d}'
            schedule = make_schedule(study,model,gate,rows,contract,target)
            schedule_path = root/model/'schedules'/(stage+'-execution.json')
            write_immutable(schedule_path,schedule)
            pins = {**extra,str(schedule_path):file_digest(schedule_path),
                **{str(Path(study['cwd'])/'learned_score'/name):sha for name,sha in source_hashes(spec['adapter']).items()}}
            for row in rows:
                for subdirectory in ('curves','features'):
                    path = directory/subdirectory/(row['row_hash']+'.json')
                    pins[str(path)] = file_digest(path)
            tasks = []
            for rank in range(study['world_size']):
                argv = python(study,'selector_experiment_v5')+['--phase','physical','--candidate','current_rank32',
                    '--warmstart',spec['checkpoint'],'--checkpoint',spec['checkpoint'],'--split','evaluation',
                    '--manifest',gate['manifest'],'--schedule',str(schedule_path),'--adapter',spec['adapter'],
                    '--model',spec['model_path'],'--model-revision',spec['revision'],'--output',str(root/model/stage),
                    '--rank',str(rank),'--world-size',str(study['world_size']),'--query-control','correct',
                    '--feature-root',str(directory),'--resume']
                if model == 'llava_next':
                    argv.append('--reference-clip')
                tasks.append(make_task(study,f'{stage}-{model}-{rank}',argv,
                    [root/model/stage/f'rank{rank}.complete.json'],model=model))
            paths.append(str(publish(study,stage+'-'+model,tasks,pins)))
    status = {'status':'published_admitted_full_suite','plans':paths,'plans_published':True,
              'gate_sha256':gate['sha256'],'independent_evaluation':True,'quality_results_complete':False}
    atomic_json(root/'full-suite/status.json',status)
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('features','advance'))
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--model',choices=('qwen25','llava_next'))
    parser.add_argument('--rank',type=int,default=0)
    args = parser.parse_args(argv)
    study = checked(args.study)
    if args.phase == 'features':
        feature_worker(study,args.model,args.rank)
    else:
        print(json.dumps(advance(study)),flush=True)


if __name__ == '__main__':
    main()
