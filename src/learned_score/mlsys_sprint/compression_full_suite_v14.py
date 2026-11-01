"""Frozen v13 policy on complete available, training-excluded official splits.

This is post-selection development/transfer evaluation, NOT independent
confirmation. Manifest selection accesses identities/images, never answers.
No threshold is fitted here. Every reported count is physically inferred.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest
from . import compression_data_v9 as data
from . import compression_full_suite_v12 as previous
from . import compression_policy_v12 as policy

FORMAT = 'mlsys_full_suite_v14'
EVIDENCE = 'post_selection_development_transfer_not_independent_confirmation'
METHODS = ('all_tokens', 'scorer_adaptive', 'scorer_uniform',
           'divprune_adaptive', 'divprune_uniform')
LABELS = {'all_tokens': 'All image tokens', 'scorer_adaptive': 'AsymVLM adaptive',
          'scorer_uniform': 'AsymVLM uniform, same average KV capacity',
          'divprune_adaptive': 'DivPrune, same adaptive counts',
          'divprune_uniform': 'DivPrune uniform, same average KV capacity'}
EXCLUSION_MANIFESTS = (
    'reproduction_results/native_protocol/manifest.json',
    'reproduction_results/mlsys_20261012/manifest.json',
    'reproduction_results/mlsys_adaptive_20261016_v3/data/manifest.json',
    'reproduction_results/mlsys_adaptive_20261016_v4/data/manifest.json')
TRAINING_ROLES = frozenset(('scorer_train', 'scorer_dev', 'scorer_screening',
    'scorer_selection', 'controller_calibration', 'fit', 'screening', 'phi_train'))
EVALUATION_ROLES = frozenset(('evaluation', 'final_test', 'confirmation', 'reserve'))
RESERVED_ROLES = frozenset(('final_test', 'confirmation', 'reserve'))


def _pins(value):
    for filename, expected in value['input_hashes'].items():
        if file_digest(filename) != expected:
            raise ValueError('full-suite frozen input changed: ' + filename)


def source_selection(study):
    source = checked(study['source_study'])
    path = Path(study.get('decision_path', Path(source['output_root']) / 'decision.json'))
    selected = checked(path)
    if (source['format'] != 'mlsys_compression_finalize_v13'
            or selected['study_sha256'] != source['sha256']
            or selected.get('selected_method') != 'scorer_adaptive'
            or selected.get('selector') != 'scorer'
            or selected.get('rule_kind') != 'rank32_coverage'
            or set(selected['model_rules']) != set(study['models'])):
        raise ValueError('campaign requires the actual frozen v13 learned coverage rule')
    for model, rule in selected['model_rules'].items():
        if (rule['model'] != model or rule['bytes_per_token'] != study['bytes_per_token'][model]
                or rule['generation_reserve'] != study['generation_reserve']
                or study['models'][model] != source['models'][model]):
            raise ValueError('frozen model/scorer/cache identity differs')
    return source, path, selected


def exclusion_records(project_root):
    """Explicit training/calibration roles; older evaluation is not called train."""
    from learned_score.mlsys_eval.common import load_manifest
    rows, reserved, sources, pins = [], [], [], {}
    for filename in EXCLUSION_MANIFESTS:
        path = Path(project_root) / filename
        manifest = load_manifest(path)
        unknown = set(manifest['splits']) - TRAINING_ROLES - EVALUATION_ROLES
        if unknown:
            raise ValueError('unclassified historical manifest roles: ' + repr(sorted(unknown)))
        for role, members in manifest['splits'].items():
            if role in TRAINING_ROLES:
                rows.extend(members)
            if role in RESERVED_ROLES:
                reserved.extend(members)
            sources.append({'manifest': str(path), 'split': role, 'rows': len(members),
                            'excluded_as_training_or_calibration': role in TRAINING_ROLES,
                            'protected_unopened_reservation': role in RESERVED_ROLES})
        pins[str(path)] = file_digest(path)
    data.identity_only(rows)
    data.identity_only(reserved)
    return rows, reserved, sources, pins


def select_partition(rows, exclusions, rules, reservations=()):
    """Whole-document/image closure, including byte aliases and frozen fit groups."""
    data.identity_only(rows)
    keys = set()
    for row in data.identity_only(exclusions):
        keys.update(data.qualify_keys(row))
    for rule in rules.values():
        keys.update('row_hash:' + value for value in rule['fit_row_hashes'])
        keys.update('group_id:' + value for value in rule['fit_group_ids'])
    training_free, excluded_groups = data.group_closure(rows, keys)
    reservation_keys = set()
    for row in data.identity_only(list(reservations)):
        reservation_keys.update(data.qualify_keys(row))
    selected, reserved_groups = data.group_closure(training_free, reservation_keys)
    if len({r['row_hash'] for r in rows}) != len(rows):
        raise ValueError('duplicate official evaluation request')
    counts = {}
    for domain in data.FAMILIES:
        available = [r for r in rows if data.family(r['dataset']) == domain]
        accepted = [r for r in selected if data.family(r['dataset']) == domain]
        if not available or not accepted:
            raise ValueError('no complete nontraining evaluation partition for ' + domain)
        if domain == 'MME':
            pairs = Counter(r['group_id'] for r in accepted)
            if any(count != 2 for count in pairs.values()):
                raise ValueError('training exclusion split an MME image pair')
        train_free = [r for r in training_free if data.family(r['dataset']) == domain]
        counts[domain] = {'available_official_rows': len(available), 'evaluated_rows': len(accepted),
            'excluded_training_calibration_rows': len(available) - len(train_free),
            'protected_reservation_rows': len(train_free) - len(accepted),
            'evaluated_groups': len({r['group_id'] for r in accepted}),
            'complete_training_and_reservation_excluded_available_partition': True}
    selected = sorted(selected, key=lambda r: (r['dataset'], r['row_hash']))
    return selected, {'domains': counts, 'excluded_groups': sorted(excluded_groups),
        'protected_reserved_groups': sorted(reserved_groups),
        'reservation_identity_keys_sha256': digest(sorted(reservation_keys)),
        'exclusion_identity_keys_sha256': digest(sorted(keys)),
        'excluded_row_hashes': sorted(set(r['row_hash'] for r in rows) - set(r['row_hash'] for r in selected))}


def prepare(study):
    """Freeze the entire identity-only partition and original deployment rules."""
    from .compression_campaign_v14 import verify
    from . import compression_admission_v12 as admission
    from learned_score.mlsys_eval.common import seal_manifest, load_manifest
    verify(study)
    root, project = Path(study['output_root']), Path(study['cwd'])
    gate_path = root / 'full-suite/gate.json'
    if gate_path.exists():
        gate = checked(gate_path)
        if gate['study_sha256'] != study['sha256']:
            raise ValueError('full-suite gate belongs to another study')
        _pins(gate)
        return gate
    source, decision_path, selection = source_selection(study)
    source_v12 = checked(source['source_study'])
    audit_path = Path(source_v12['output_root']) / 'admission/status.json'
    audit = checked(audit_path)
    rows, sources, blockers = admission.evaluation_inventory(project, audit,
        deadline=study['deadline_utc_epoch'])
    if any(blockers.values()):
        raise ValueError('official source unavailable or changed: ' + repr(blockers))
    exclusions, reservations, exclusion_sources, exclusion_pins = exclusion_records(project)
    reservation_path = Path(study['reservation_manifest'])
    reservation = load_manifest(reservation_path)
    if (reservation['metadata'].get('labels_loaded') is not False
            or reservation['metadata'].get('reservation_is_not_admission') is not True):
        raise ValueError('invalid unopened identity reservation')
    reservations.extend(reservation['splits']['reservation'])
    exclusion_pins[str(reservation_path)] = file_digest(reservation_path)
    # Hash before partitioning so the same bytes under another image ID cannot
    # silently cross the training/evaluation boundary.
    rows = data.enrich_images(rows, project, root / 'full-suite')
    selected, partition = select_partition(rows, exclusions, selection['model_rules'], reservations)
    for row in selected:
        domain = data.family(row['dataset'])
        row.update(metric=data.data.METRICS[domain], quality_metric=data.data.METRICS[domain],
            evaluation_stage=EVIDENCE, known_training_calibration_overlap=False,
            historical_evaluation_exposure='previously_examined_or_unresolved_not_fresh')
    manifest = seal_manifest({'format': 'mlsys_adaptive_data_v3', 'dataset': 'ASYMVLM_FULL_SUITE_V14',
        'metadata': {'study_sha256': study['sha256'], 'selection_uses_labels': False,
            'independent_confirmation': False, 'fresh_identity_audit_passed': False,
            'evidence': EVIDENCE, 'complete_training_and_reservation_excluded_available_partition': True,
            'known_training_calibration_excluded': True, 'partition': partition},
        'splits': {'fit': [], 'screening': [], 'evaluation': selected, 'phi_train': []}})
    manifest_path = root / 'full-suite/manifest.json'
    write_immutable(manifest_path, manifest)
    pins = {**exclusion_pins, **{item['path']: item['sha256'] for item in sources}}
    pins.update({str(path): file_digest(path) for path in
                 (audit_path, decision_path, Path(study['source_study']), manifest_path, Path(__file__),
                  Path(previous.__file__), Path(policy.__file__), Path(data.__file__), Path(admission.__file__))})
    gate = seal({'format': FORMAT + '_gate', 'study_sha256': study['sha256'],
        'selection_sha256': selection['sha256'], 'manifest': str(manifest_path),
        'manifest_sha256': file_digest(manifest_path), 'models': {
            model: {'primary': copy.deepcopy(rule)} for model, rule in selection['model_rules'].items()},
        'request_ids': [r['row_hash'] for r in selected], 'input_hashes': pins,
        'partition': partition, 'exclusion_sources': exclusion_sources, 'source_files': sources,
        'evidence': EVIDENCE, 'independent_confirmation': False,
        'labels_opened_for_manifest_selection': False, 'policies_frozen_before_new_labels': True,
        'scorer_training': False, 'threshold_refitting': False, 'created_utc_epoch': time.time()})
    _pins(gate)
    write_immutable(gate_path, gate)
    return gate


@contextmanager
def _feature_bindings():
    """Reuse only the old answer-free worker, not its independent-admission gate."""
    saved = previous.FORMAT
    previous.FORMAT = FORMAT
    try:
        yield
    finally:
        previous.FORMAT = saved


def feature_worker(study, model, rank):
    from .compression_campaign_v14 import verify
    verify(study)
    gate = checked(Path(study['output_root']) / 'full-suite/gate.json')
    if gate['format'] != FORMAT + '_gate' or gate['independent_confirmation'] is not False:
        raise ValueError('wrong feature admission contract')
    with _feature_bindings():
        previous.feature_worker(study, model, rank)


def collect_features(study, model, gate):
    _pins(gate)
    directory = Path(study['output_root']) / model / 'full-features'
    for rank in range(study['world_size']):
        path = directory / f'rank{rank}.metadata.json'
        if path.exists() and json.loads(path.read_text()).get('format') != FORMAT + '_features':
            raise ValueError('full-suite feature metadata has another version')
    return previous.collect(study, model, gate)


def make_schedule(study, model, gate, rows, contract):
    from .compression_policy_v11 import factorial_assignments
    from .adaptive_experiment_v4 import validate_schedule
    if (contract.get('reserved_generated_slots') != study['generation_reserve']
            or contract.get('total_owned_bytes_per_slot') != study['bytes_per_token'][model]):
        raise ValueError('full-suite cache contract differs from frozen accounting')
    rule = gate['models'][model]['primary']
    output = policy.infer(rule, policy.answer_free(rows))
    normalized = [{**r, 'dataset': data.family(r['dataset'])} for r in rows]
    factorial = factorial_assignments(normalized, output['counts'], 'divprune')
    cells = {**factorial['cells'], 'all_tokens': {'selector': 'full',
             'counts': {r['row_hash']: r['input_visual_tokens'] for r in rows}}}
    scheduled = {}
    for row in rows:
        request, n = row['row_hash'], row['input_visual_tokens']
        scheduled[request] = {k: row[k] for k in
            ('row_hash', 'group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')}
        scheduled[request]['cells'] = [{'key': key,
            'selector': 'full' if cells[key]['counts'][request] == n else cells[key]['selector'],
            'selected_count': cells[key]['counts'][request], 'mode': 'exact',
            'policy': 'frozen_v13_full_partition_v14'} for key in METHODS]
    result = seal({'format': FORMAT + '_schedule', 'study_sha256': study['sha256'],
        'gate_sha256': gate['sha256'], 'manifest_sha256': gate['manifest_sha256'],
        'checkpoint_sha256': file_digest(study['models'][model]['checkpoint']), 'split': 'evaluation',
        'cache_contract': contract, 'model': model, 'rows': scheduled, 'policy_sha256': rule['sha256'],
        'feature_rows_sha256': digest(rows), 'answers_used_to_choose_counts': False,
        'exact_aggregate_kv_matched_within_dataset': True, 'quality_requires_physical_inference': True,
        'fallbacks': output['fallbacks'], 'independent_confirmation': False, 'evidence': EVIDENCE})
    validate_schedule(result, manifest_sha256=gate['manifest_sha256'],
        checkpoint_sha256=result['checkpoint_sha256'], split='evaluation', allowed_ids=gate['request_ids'])
    return result


def advance(study):
    from .compression_campaign_v14 import verify, python, make_task, publish_plan
    from .selector_experiment_v5 import source_hashes
    verify(study)
    gate, root = prepare(study), Path(study['output_root'])
    paths, states = [], {}
    for model, spec in study['models'].items():
        directory = root / model / 'full-features'
        extra = {**gate['input_hashes'], str(root / 'full-suite/gate.json'): file_digest(root / 'full-suite/gate.json')}
        tasks = []
        for rank in range(study['world_size']):
            argv = python(study, 'compression_full_suite_v14') + ['features', '--study', str(root / 'study.json'),
                '--model', model, '--rank', str(rank)]
            tasks.append(make_task(study, f'full-features-{model}-{rank}', argv,
                [directory / f'rank{rank}.complete.json'], model=model))
        paths.append(str(publish_plan(study, 'full-features-' + model, tasks, extra)))
        collected = collect_features(study, model, gate)
        if collected is None:
            states[model] = 'features_incomplete'
            continue
        rows, contract = collected
        stage = 'full-primary'
        schedule_path = root / model / 'schedules' / (stage + '-execution.json')
        schedule = make_schedule(study, model, gate, rows, contract)
        write_immutable(schedule_path, schedule)
        pins = {**extra, str(schedule_path): file_digest(schedule_path),
            **{str(Path(study['cwd']) / 'learned_score' / name): sha for name, sha in source_hashes(spec['adapter']).items()}}
        for row in rows:
            for subdirectory in ('curves', 'features'):
                path = directory / subdirectory / (row['row_hash'] + '.json')
                pins[str(path)] = file_digest(path)
        tasks = []
        for rank in range(study['world_size']):
            argv = python(study, 'selector_experiment_v5') + ['--phase', 'physical', '--candidate', 'current_rank32',
                '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'], '--split', 'evaluation',
                '--manifest', gate['manifest'], '--schedule', str(schedule_path), '--adapter', spec['adapter'],
                '--model', spec['model_path'], '--model-revision', spec['revision'], '--output', str(root / model / stage),
                '--rank', str(rank), '--world-size', str(study['world_size']), '--query-control', 'correct',
                '--feature-root', str(directory)]
            if model == 'llava_next':
                argv.append('--reference-clip')
            task = make_task(study, f'{stage}-{model}-{rank}', argv,
                [root / model / stage / f'rank{rank}.complete.json'], model=model)
            task.update(resume_checkpoint=str(root / model / stage / f'rank{rank}.metadata.json'), resume_args=['--resume'])
            tasks.append(task)
        paths.append(str(publish_plan(study, stage + '-' + model, tasks, pins)))
        report = collect_model(study, model)
        states[model] = 'complete' if report else 'physical_incomplete'
    status = {'status': 'complete' if all(v == 'complete' for v in states.values()) else 'running',
        'plans': paths, 'models': states, 'partition': gate['partition']['domains'],
        'independent_confirmation': False, 'evidence': EVIDENCE}
    atomic_json(root / 'full-suite/status.json', status)
    return status


def report_rows(study, model, schedule, measured, *, repetitions=2000):
    from .adaptive_report_v3 import paired_ci, quality_summary
    from .compression_report_v8 import memory_summary
    ids = set(schedule['rows'])
    index = {}
    for row in measured:
        key = (row['request_id'], row['method'])
        if key in index or row['request_id'] not in ids or row['method'] not in METHODS:
            raise ValueError('duplicated or unknown physical full-suite result')
        planned = next(cell for cell in schedule['rows'][key[0]]['cells'] if cell['key'] == key[1])
        expected = (row['input_text_tokens'] + planned['selected_count'] + study['generation_reserve']) * study['bytes_per_token'][model]
        if (row['selected_count'] != planned['selected_count'] or row['retained_visual_tokens'] != planned['selected_count']
                or row['metrics']['cache']['kv_capacity_bytes'] != expected):
            raise ValueError('physical count or allocated KV differs from frozen schedule')
        index[key] = row
    if set(index) != {(request, method) for request in ids for method in METHODS}:
        raise ValueError('full-suite physical cohort incomplete')
    output, audits, contrasts = [], {}, []
    for domain in data.FAMILIES:
        requests = {request for request in ids if data.family(schedule['rows'][request]['dataset']) == domain}
        if not requests:
            raise ValueError('missing declared benchmark')
        methods = {method: [index[(request, method)] for request in sorted(requests)] for method in METHODS}
        reference = methods['all_tokens']
        totals = []
        for method, values in methods.items():
            memory = memory_summary(values, reference)
            quality = quality_summary(values, reference[0]['quality_metric'])
            if method != 'all_tokens':
                totals.append(memory['aggregate_allocated_kv_bytes'])
            output.append({'model': model, 'dataset': domain, 'method': method, 'label': LABELS[method],
                'requests': len(values), 'groups': len({r['group_id'] for r in values}),
                'score_percent': quality['score_percent'], 'quality': quality,
                'mean_allocated_kv_mib': memory['mean_allocated_kv_bytes'] / 2**20,
                'kv_reduction_percent': memory['aggregate_allocated_kv_reduction_percent'],
                'percent_requests_with_memory_reduction': 100 * memory['fraction_compressed'],
                'versus_all_tokens': paired_ci(values, reference, reference[0]['quality_metric'], repetitions=repetitions),
                'memory': memory, 'quality_is_physical': True, 'independent_confirmation': False})
        if len(set(totals)) != 1:
            raise ValueError('compressed methods have unequal aggregate allocated KV')
        audits[domain] = {'requests': len(requests), 'matched_aggregate_kv_bytes': totals[0]}
        for a, b in (('scorer_adaptive', 'scorer_uniform'), ('scorer_adaptive', 'divprune_uniform'),
                     ('scorer_adaptive', 'divprune_adaptive')):
            contrasts.append({'model': model, 'dataset': domain, 'candidate': a, 'reference': b,
                **paired_ci(methods[a], methods[b], reference[0]['quality_metric'], repetitions=repetitions)})
    return {'rows': output, 'paired_equal_budget_contrasts': contrasts,
        'matched_capacity_audit': audits, 'all_declared_cohorts_complete': True,
        'independent_confirmation': False, 'evidence': EVIDENCE,
        'text_policy': 'protected_prefix_generated_recency_512_max_new_tokens_128_no_eviction',
        'memory_definition': 'mean_allocated_KV_capacity_not_total_GPU_or_mean_occupied_KV',
        'threshold_refitting': False, 'interpolation_used': False}


def collect_model(study, model):
    from .compression_matrix_v8 import scan_execution
    root = Path(study['output_root'])
    path = root / model / 'schedules/full-primary-execution.json'
    if not path.exists():
        return None
    measured = scan_execution(study, model, 'full-primary')
    if measured is None:
        return None
    gate = checked(root / 'full-suite/gate.json')
    _pins(gate)
    schedule = checked(path)
    if schedule['study_sha256'] != study['sha256'] or schedule['gate_sha256'] != gate['sha256']:
        raise ValueError('physical campaign belongs to another gate')
    destination = root / 'full-suite' / (model + '-report.json')
    source_digest = digest([(r['physical_source_path'], r['physical_source_sha256']) for r in measured])
    if destination.exists():
        report = checked(destination)
        if report['physical_sources_sha256'] != source_digest or report['schedule_sha256'] != schedule['sha256']:
            raise ValueError('completed full-suite evidence changed')
        return report
    result = report_rows(study, model, schedule, measured)
    report = seal({'format': FORMAT + '_report', 'study_sha256': study['sha256'], 'model': model,
        'gate_sha256': gate['sha256'], 'schedule_sha256': schedule['sha256'],
        'physical_sources_sha256': source_digest, 'report_source_sha256': file_digest(__file__),
        'partition': gate['partition'], **result})
    write_immutable(destination, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'features', 'advance', 'collect'))
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model', choices=('qwen25', 'llava_next'))
    parser.add_argument('--rank', type=int, default=0)
    args = parser.parse_args(argv)
    study = checked(args.study)
    if args.phase == 'features':
        feature_worker(study, args.model, args.rank)
    elif args.phase == 'prepare':
        gate = prepare(study)
        print(json.dumps({'gate_sha256': gate['sha256'], 'partition': gate['partition']['domains']}), flush=True)
    elif args.phase == 'collect':
        print(json.dumps({'complete': collect_model(study, args.model) is not None}), flush=True)
    else:
        print(json.dumps(advance(study)), flush=True)


if __name__ == '__main__':
    main()
