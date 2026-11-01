"""Development-first orchestration of the accuracy-preserving v8 study.

GPU measurement plans are independent of this CPU policy/report pipeline.
An inspected development panel is never silently promoted to confirmation.
"""

import argparse
from collections import defaultdict
import copy
import fcntl
import json
from pathlib import Path
import time

from . import compression_matrix_v8 as matrix
from . import compression_policy_v8 as policies
from . import compression_report_v8 as reporting
from . import compression_confirmation_v8 as freshness
from . import compression_confirmation_run_v8 as confirmation
from . import compression_systems_run_v8 as systems
from .adaptive_report_v1 import _atomic_text
from .adaptive_report_v3 import paired_ci
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_study_v8'
read, checked, seal = matrix.read, matrix.checked, matrix.seal


def sources():
    helpers = [Path(__file__).with_name(name) for name in
               ('adaptive_report_v3.py', 'adaptive_report_v1.py', 'followup_common.py', 'state.py')]
    return {**policies.sources(), **{str(path.resolve()): file_digest(path) for path in helpers},
            **{str(Path(module.__file__).resolve()): file_digest(module.__file__)
            for module in (matrix, policies, reporting, freshness, confirmation, systems, systems.repeated)
            }, str(Path(__file__).resolve()): file_digest(__file__)}


def freeze_pipeline(study):
    path = Path(study['output_root']) / 'pipeline.json'
    payload = seal({'format': FORMAT, 'study_sha256': study['sha256'], 'source_hashes': sources(),
        'candidate_families': study['candidate_families'], 'fit_allowances_pp': study['fit_allowances_pp'],
        'selection': ['most_promising_domains', 'mean_domain_balanced_patch_reduction',
                      'lowest_worst_domain_loss', 'single_statistic_before_combined', 'lexical_name'],
        'screening_rule': 'point loss <=1pp; development selection, not confidence certification',
        'confirmation_rule': 'point loss <=1pp and one-sided95% upper loss <1pp',
        'created_utc_epoch': time.time()})
    if path.exists():
        old = checked(path)
        if old['source_hashes'] != sources() or old['study_sha256'] != study['sha256']:
            raise ValueError('frozen v8 pipeline or source changed')
        return old
    write_immutable(path, payload)
    return payload


def normalized_features(study, model, split):
    manifest, _, _, _, features = matrix.features(study, model, split)
    entries = {r['row_hash']: r for r in manifest.splits[split]}
    return [{**matrix.planning._geometry(feature),
             'benefit_features': policies.features_from_cached(feature),
             'quality_metric': entries[request].get('quality_metric', entries[request].get('metric')),
             'category': entries[request].get('category') or None}
            for request, feature in sorted(features.items())]


def physical_index(rows):
    result = {}
    for row in rows:
        key = row['request_id'], row['selector'], row['selected_count']
        if key in result and any(row.get(k) != result[key].get(k) for k in
                                 ('quality_score', 'prediction', 'generated_token_ids', 'selected_patch_indices',
                                  'group_id', 'dataset', 'quality_metric', 'category', 'input_visual_tokens',
                                  'input_text_tokens', 'runtime_contract_sha256', 'model_tag')):
            raise ValueError('same physical selector/count has inconsistent answers or indices')
        result[key] = row
    return result


def fit_rows(feature_rows, physical_rows):
    index = physical_index(physical_rows)
    rows = []
    for feature in feature_rows:
        request, n = feature['row_hash'], feature['input_visual_tokens']
        quality = {}
        for count in sorted(set(policies.physical_count_menu(n).values())):
            selector = 'full' if count == n else 'scorer'
            row = index.get((request, selector, count))
            if row is None:
                raise ValueError('fit matrix lacks a physically measured count/full reference')
            quality[str(count)] = row['quality_score']
        rows.append({**feature, 'quality_by_count': quality})
    return rows


def fit_model(study, model, physical_rows):
    root = Path(study['output_root'])
    path = root / model / 'policies/index.json'
    pipeline = freeze_pipeline(study)
    if path.exists():
        value = checked(path)
        if (value.get('study_sha256'), value.get('pipeline_sha256'), value.get('model')) != (
                study['sha256'], pipeline['sha256'], model):
            raise ValueError('frozen policy index belongs to another study or model')
        for artifact in value['bundles']:
            if file_digest(artifact['path']) != artifact['sha256']:
                raise ValueError('frozen policy bundle changed')
        return value
    feature_rows = normalized_features(study, model, 'fit')
    grouped = defaultdict(list)
    for row in fit_rows(feature_rows, physical_rows):
        grouped[row['dataset']].append(row)
    artifacts = []
    for dataset, group in sorted(grouped.items()):
        identity = {'split': 'fit', 'model': model, 'dataset': dataset, 'study_sha256': study['sha256'],
                    'pipeline_sha256': pipeline['sha256'], 'fit_data_sha256': digest(group)}
        for family in (*study['candidate_families'], 'static'):
            output = root / model / 'policies' / f'{digest(dataset)[:12]}-{family}.json'
            if output.exists():
                bundle = checked(output)
                if bundle['identity'] != identity:
                    raise ValueError('resumed policy fitting has different data')
            else:
                bundle = policies.fit_policy_family(group, statistic=family, identity=identity,
                    allowances_pp=tuple(study['fit_allowances_pp']), min_groups=study['minimum_bin_groups'], diagnostics=True)
                write_immutable(output, bundle)
            artifacts.append({'dataset': dataset, 'family': family, 'path': str(output), 'sha256': file_digest(output)})
    value = seal({'format': FORMAT + '_fit_index', 'study_sha256': study['sha256'],
                  'pipeline_sha256': pipeline['sha256'], 'model': model, 'bundles': artifacts,
                  'screening_labels_used_for_fitting': False})
    write_immutable(path, value)
    return value


def projected_rows(features, index, counts, method, *, selector='scorer', assignments=None):
    """View an independently measured count as a policy outcome, never interpolate."""
    result = []
    if set(counts) != {row['row_hash'] for row in features}:
        raise ValueError('count assignment must cover the exact feature cohort')
    for feature in features:
        request, n = feature['row_hash'], feature['input_visual_tokens']
        count = counts[request]
        if type(count) is not int or not 1 <= count <= n:
            raise ValueError('invalid exact assigned visual count')
        source_selector = 'full' if count == n else selector
        source = index.get((request, source_selector, count))
        if source is None:
            raise ValueError('policy assigned an unmeasured exact count')
        if any(source.get(key) != feature[key] for key in
               ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')):
            raise ValueError('physical row has different feature geometry or identity')
        assigned = (assignments or {}).get(request, {})
        fallback = bool(assigned.get('fallback', False))
        if fallback and count != n:
            raise ValueError('full fallback must retain every image token')
        result.append({**source, 'method': method, 'full_fallback': fallback,
            'policy_assignment': assigned, 'physical_reuse_view': True,
            'timing_reused_not_an_independent_observation': True})
    return result


def evaluate_model(study, model, index_file, physical_rows):
    features = normalized_features(study, model, 'screening')
    grouped = defaultdict(list)
    for row in features:
        grouped[row['dataset']].append(row)
    index, panels, row_views = physical_index(physical_rows), [], {}
    artifacts = index_file['bundles']
    for dataset, group in sorted(grouped.items()):
        expected = [r['row_hash'] for r in group]
        full_counts = {r['row_hash']: r['input_visual_tokens'] for r in group}
        full = projected_rows(group, index, full_counts, 'All image patches / full KV cache')
        candidates = []
        for artifact in artifacts:
            if artifact['dataset'] != dataset:
                continue
            bundle = checked(artifact['path'])
            for policy in bundle['policies']:
                assignment = policies.predict_counts(policy, group)
                name = artifact['family'] + ':' + policy['sha256'][:12]
                rows = projected_rows(group, index, assignment['counts'], name, assignments=assignment['rows'])
                comparison = reporting.compare_to_full(rows, full, expected_request_ids=expected)
                candidates.append({'family': artifact['family'], 'policy': policy, 'policy_file': artifact,
                    'assignment': assignment, 'comparison': comparison, 'method': name})
                row_views[(dataset, name)] = rows
        baselines = []
        for selector in ('scorer', 'divprune') + (('cdpruner',) if model == 'llava_next' else ()):
            for keep in matrix.KEEP:
                counts = {r['row_hash']: policies.physical_count_menu(r['input_visual_tokens'])[keep] for r in group}
                method = f'{selector} uniform {keep}%'
                rows = projected_rows(group, index, counts, method, selector=selector)
                comparison = reporting.compare_to_full(rows, full, expected_request_ids=expected)
                baselines.append({'selector': selector, 'keep_percent': keep, 'comparison': comparison, 'method': method})
                row_views[(dataset, method)] = rows
        panels.append({'model': model, 'dataset': dataset, 'expected_requests': expected,
                       'candidates': candidates, 'baselines': baselines})
        row_views[(dataset, 'full')] = full
    return panels, row_views


def promising(comparison):
    return comparison.get('cohort_complete') is True and comparison.get('mean_loss_pp') is not None and comparison['mean_loss_pp'] <= 1.


def compression_key(record):
    comparison = record['comparison']
    return (comparison['memory']['aggregate_retained_visual_tokens'], comparison['mean_loss_pp'], record['method'])


def choose_shared_family(panels, families):
    if not panels or not families or len(set(families)) != len(families):
        raise ValueError('nonempty panels and unique candidate families required')
    keys = [(panel['model'], panel['dataset']) for panel in panels]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate model/dataset selection panel')
    summaries = []
    for family in families:
        selected, reductions, losses, passed = {}, [], [], 0
        for panel in panels:
            eligible = [c for c in panel['candidates'] if c['family'] == family and promising(c['comparison'])]
            best = min(eligible, key=compression_key) if eligible else None
            key = panel['model'] + '|' + panel['dataset']
            selected[key] = best
            reduction = best['comparison']['memory']['aggregate_patch_reduction_percent'] if best else 0.
            reductions.append(reduction)
            losses.append(best['comparison']['mean_loss_pp'] if best else 0.)
            passed += reduction > 0
        summaries.append({'family': family, 'promising_compressed_domains': passed,
                          'mean_patch_reduction_percent': sum(reductions) / len(reductions),
                          'worst_loss_pp': max(losses), 'selected': selected})
    selected = min(summaries, key=lambda r: (-r['promising_compressed_domains'], -r['mean_patch_reduction_percent'],
                                           r['worst_loss_pp'], r['family'] == 'combined_ridge', r['family']))
    return {'selected_family': selected['family'], 'domains': selected['selected'],
            'family_summary': [{k: v for k, v in r.items() if k != 'selected'} for r in summaries],
            'evidence': 'development selection, not confirmation'}


def publish_matched(study, model, features, physical_rows, selected_domains):
    root = Path(study['output_root']) / model
    path = root / 'schedules/matched.json'
    paths = {'master': path, 'execution': root / 'schedules/matched-execution.json', 'reuse': root / 'schedules/matched-reuse.json'}
    if all(value.exists() for value in paths.values()):
        master = checked(path)
        if (master['study_sha256'] != study['sha256'] or master['selection_sha256'] !=
                file_digest(Path(study['output_root']) / 'selection.json')):
            raise ValueError('matched schedule belongs to a different selection')
        matrix.publish_execution(study, model, 'matched', 'screening', paths, checked(paths['execution']))
        return path
    _, _, _, metadata, raw_features = matrix.features(study, model, 'screening')
    index = physical_index(physical_rows)
    rows, inventory = {}, {}
    grouped = defaultdict(list)
    for feature in features:
        grouped[feature['dataset']].append(feature)
    for dataset, group in grouped.items():
        choice = selected_domains[model + '|' + dataset]
        total = choice['assignment']['total_visual_tokens'] if choice else sum(r['input_visual_tokens'] for r in group)
        assigned = policies.uniform_at_visual_total(group, total)
        if not assigned['feasible'] or assigned['token_error'] != 0:
            raise ValueError('matched uniform count assignment is infeasible')
        for feature in group:
            request, count = feature['row_hash'], assigned['counts'][feature['row_hash']]
            selector = 'full' if count == feature['input_visual_tokens'] else 'scorer'
            rows[request] = {**matrix.planning._geometry(raw_features[request]), 'cells': [{
                'key': 'matched_uniform', 'selector': selector, 'selected_count': count,
                'mode': 'exact_aggregate_matched_uniform', 'policy': 'uniform_at_selected_adaptive_total'}]}
    for key, value in index.items():
        path0 = Path(value['physical_source_path'])
        meta_path = path0.parent.parent / 'rank0.metadata.json'
        inventory[key] = {'row': value, 'path': str(path0), 'sha256': file_digest(path0),
            'metadata_path': str(meta_path), 'metadata_sha256': file_digest(meta_path),
            'runtime_contract_sha256': value['runtime_contract_sha256']}
    master = seal({'format': FORMAT + '_matched_schedule', 'study_sha256': study['sha256'],
        'model': model, 'split': 'screening', 'candidate': 'current_rank32',
        'manifest_sha256': study['manifest_sha256'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
        'cache_contract': metadata['cache_contract'], 'rows': rows,
        'selection_sha256': file_digest(Path(study['output_root']) / 'selection.json')})
    execution, reuse = matrix.split_reuse(master, inventory)
    for name, value in (('master', master), ('execution', execution), ('reuse', reuse)):
        write_immutable(paths[name], value)
    matrix.publish_execution(study, model, 'matched', 'screening', paths, execution)
    return path


def develop(study, model_rows, indexes):
    root = Path(study['output_root'])
    path = root / 'selection.json'
    previous = checked(path) if path.exists() else None
    if previous and all((root / relative).exists() for relative in
                        ('reports/development_candidates.json', 'reports/development.json', 'reports/TABLES.md')):
        return previous
    panels, views = [], {}
    for model in matrix.MODELS:
        current, mapped = evaluate_model(study, model, indexes[model], model_rows[model])
        panels.extend(current)
        views[model] = mapped
    chosen = choose_shared_family(panels, study['candidate_families'])
    selected_baselines, report_rows, expected = {}, [], {}
    for panel in panels:
        model, dataset = panel['model'], panel['dataset']
        expected[(model, dataset)] = panel['expected_requests']
        full = [{**r, 'method': 'full'} for r in views[model][(dataset, 'full')]]
        report_rows.extend(full)
        choice = chosen['domains'][model + '|' + dataset]
        adaptive = views[model][(dataset, choice['method'])] if choice else full
        report_rows.extend({**r, 'method': 'AsymVLM adaptive (' + chosen['selected_family'] + ')',
                            'full_fallback': r.get('full_fallback', False) if choice else True} for r in adaptive)
        for selector, label in (('scorer', 'AsymVLM uniform'), ('divprune', 'DivPrune adaptation'),
                                ('cdpruner', 'CDPruner reference-CLIP adaptation')):
            options = [r for r in panel['baselines'] if r['selector'] == selector and promising(r['comparison'])]
            if not options:
                continue
            best = min(options, key=compression_key)
            selected_baselines[model + '|' + dataset + '|' + selector] = best
            report_rows.extend({**r, 'method': label} for r in views[model][(dataset, best['method'])])
    report = reporting.summarize(report_rows, expected_cohorts=expected)
    payload = seal({'format': FORMAT + '_selection', 'study_sha256': study['sha256'],
        'pipeline_sha256': checked(root / 'pipeline.json')['sha256'], **chosen,
        'baselines': selected_baselines, 'created_utc_epoch': previous['created_utc_epoch'] if previous else time.time(),
        'confirmation_labels_consumed': False, 'fit_index_hashes': {
            model: file_digest(root / model / 'policies/index.json') for model in matrix.MODELS}})
    write_immutable(path, payload)
    write_immutable(root / 'reports/development_candidates.json', {'panels': panels})
    write_immutable(root / 'reports/development.json', report)
    _atomic_text(root / 'reports/TABLES.md', reporting.render_tables(report))
    return payload


def compare_matched(adaptive_rows, uniform_rows, *, expected_request_ids):
    """Development-only allocation contrast; uniform is NOT a full reference."""
    adaptive, uniform = reporting._index(adaptive_rows), reporting._index(uniform_rows)
    expected = list(expected_request_ids)
    if not expected or len(set(expected)) != len(expected):
        raise ValueError('declare a unique nonempty matched-control cohort')
    if set(adaptive) != set(expected) or set(uniform) != set(expected):
        raise ValueError('matched control lacks the exact complete cohort')
    identities = {(reporting._model(row), row['dataset'], row['quality_metric']) for row in adaptive.values()}
    if len(identities) != 1:
        raise ValueError('matched comparison requires one model/dataset/metric')
    for request, row in adaptive.items():
        other = uniform[request]
        for field in ('model_tag', 'group_id', 'dataset', 'quality_metric', 'category',
                      'input_visual_tokens', 'input_text_tokens', *reporting.RUNTIME_FIELDS):
            if (field in row or field in other) and (field not in row or field not in other or row[field] != other[field]):
                raise ValueError('matched-control runtime or cohort mismatch: ' + field)
    adaptive_total = sum(row['retained_visual_tokens'] for row in adaptive.values())
    uniform_total = sum(row['retained_visual_tokens'] for row in uniform.values())
    if adaptive_total != uniform_total:
        raise ValueError('matched controls must have exactly equal total visual tokens')
    contrast = paired_ci(adaptive_rows, uniform_rows, adaptive_rows[0]['quality_metric'],
                         repetitions=reporting.BOOTSTRAP_REPETITIONS, seed=reporting.BOOTSTRAP_SEED)
    return {**contrast, 'evidence': 'development', 'accepted': False, 'confirmed': False,
            'contrast': 'adaptive_minus_exact_total_uniform', 'cohort_complete': True,
            'adaptive_visual_tokens': adaptive_total, 'uniform_visual_tokens': uniform_total, 'token_error': 0,
            'adaptive_minus_uniform_pp': contrast['delta_pp'], 'uniform_minus_adaptive_loss_pp': -contrast['delta_pp'],
            'inference_note': 'Exploratory paired interval, not full-relative safety certification; degenerate intervals do not establish equivalence.'}


def matched_report(study, selected, model_rows):
    """Return None until every selected exact-total control is physically complete."""
    measured = {model: matrix.collect(study, model, 'matched') for model in matrix.MODELS}
    if any(value is None for value in measured.values()):
        return None
    panels = []
    for model in matrix.MODELS:
        features = normalized_features(study, model, 'screening')
        grouped = defaultdict(list)
        for feature in features:
            grouped[feature['dataset']].append(feature)
        for dataset, group in sorted(grouped.items()):
            choice = selected['domains'][model + '|' + dataset]
            counts = choice['assignment']['counts'] if choice else {r['row_hash']: r['input_visual_tokens'] for r in group}
            adaptive = projected_rows(group, physical_index(model_rows[model]), counts, 'adaptive',
                                      assignments=choice['assignment']['rows'] if choice else None)
            uniform = [row for row in measured[model] if row['dataset'] == dataset]
            panels.append({'model': model, 'dataset': dataset, **compare_matched(adaptive, uniform,
                           expected_request_ids=[r['row_hash'] for r in group])})
    result = {'format': FORMAT + '_matched_report', 'evidence': 'development',
              'selection_sha256': selected['sha256'], 'panels': panels}
    write_immutable(Path(study['output_root']) / 'reports/matched.json', result)
    return result


def confirmation_status(study):
    """Cache the identity-only audit; final admission revalidates all inputs."""
    path = Path(study['output_root']) / 'data/confirmation_audit.json'
    if path.exists():
        result = freshness.checked(read(path))
        if result.get('study_sha256') != study['sha256']:
            raise ValueError('fresh identity audit belongs to another study')
        return result
    result = freshness.audit(study)
    write_immutable(path, result)
    return result


def advance_confirmation(study, selected, audit):
    if selected is None:
        return {'status': 'awaiting_development_selection', 'complete': False,
                'fresh_identity_audit_passed': False, 'audit_sha256': audit['sha256'],
                'identity_eligible': audit['eligible'], 'blocking_reasons': audit['blocking_reasons']}
    return confirmation.advance(study, selected, audit)


def tick(study):
    matrix.verify(study)
    freeze_pipeline(study)
    root = Path(study['output_root'])
    audit = confirmation_status(study)
    execution = matrix.status(study)
    indexes, model_rows = {}, {}
    for model in matrix.MODELS:
        if all(execution[f'fit-{band}-{model}']['status'] == 'complete' for band in ('high', 'low')):
            fit = sum((matrix.collect(study, model, 'fit-' + band) for band in ('high', 'low')), [])
            indexes[model] = fit_model(study, model, fit)
        if all(execution[f'screen-{band}-{model}']['status'] == 'complete' for band in ('high', 'low')):
            model_rows[model] = sum((matrix.collect(study, model, 'screen-' + band) for band in ('high', 'low')), [])
    selected = None
    matched = None
    if len(indexes) == len(matrix.MODELS) and len(model_rows) == len(matrix.MODELS):
        selected = develop(study, model_rows, indexes)
        for model in matrix.MODELS:
            publish_matched(study, model, normalized_features(study, model, 'screening'), model_rows[model], selected['domains'])
        matched = matched_report(study, selected, model_rows)
    confirmation_result = advance_confirmation(study, selected, audit)
    systems_result = systems.advance(study, confirmation_result)
    confirmed = confirmation_result.get('complete') is True
    systems_complete = systems_result.get('complete') is True
    complete = matched is not None and confirmed and systems_complete
    blocked = matched is not None and confirmation_result['status'] == 'blocked'
    stage = ('complete' if complete else 'development_complete_confirmation_blocked' if blocked else
             'systems_pending' if confirmed else 'development_complete_confirmation_pending' if matched is not None else
             'matched_controls_pending' if selected else 'development_running')
    status = {'format': FORMAT + '_status', 'updated_utc_epoch': time.time(), 'execution': execution,
              'fitted_models': sorted(indexes), 'selection_frozen': selected is not None,
              'selected_family': selected['selected_family'] if selected else None,
              'matched_controls_complete': matched is not None,
              'complete': complete, 'systems_complete': systems_complete,
              'terminal_blocked': blocked,
              'confirmation': confirmation_result, 'systems': systems_result,
              'identity_audit': {'path': str(root / 'data/confirmation_audit.json'),
                  'sha256': audit['sha256'], 'eligible': audit['eligible'],
                  'blocking_reasons': audit['blocking_reasons']},
              'status': stage}
    atomic_json(root / 'coordinator/status.json', status)
    return status


def watch(study):
    root = Path(study['output_root']) / 'coordinator'
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'driver.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study['deadline_utc_epoch']:
            result = tick(study)
            print(json.dumps(result, sort_keys=True), flush=True)
            if result['complete'] or result['terminal_blocked']:
                break
            time.sleep(60)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('freeze', 'tick', 'watch', 'status'), required=True)
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    study = matrix.prepare(args.cwd)
    if args.phase == 'watch':
        watch(study)
    elif args.phase == 'freeze':
        print(json.dumps(freeze_pipeline(study), indent=2))
    elif args.phase == 'status':
        print(json.dumps(matrix.status(study), indent=2))
    else:
        print(json.dumps(tick(study), indent=2))


if __name__ == '__main__':
    main()
