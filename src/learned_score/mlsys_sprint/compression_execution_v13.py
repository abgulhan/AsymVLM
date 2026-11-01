"""Immutable v13 task publication using the unchanged physical v5 runner."""
import copy
from pathlib import Path
import re
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import digest, file_digest
from .worker import read_plan, verify_sources


def python(study, module):
    return [str(Path(study['cwd']) / '.conda/asymvlm/bin/python'), '-B', '-u', '-m',
            'learned_score.mlsys_sprint.' + module]


def make_task(study, name, argv, artifacts, *, model, resource='gpu', minimum_start_seconds=300):
    from .adaptive_workflow_v4 import environment
    return {'id': 'compression-v13-' + name, 'tag': 'compression-v13', 'resource': resource,
        'model_tag': model, 'minimum_start_seconds': minimum_start_seconds, 'depends_on': [],
        'argv': argv, 'env': {**environment(Path(study['cwd'])), 'PYTHONDONTWRITEBYTECODE': '1'},
        'artifacts': [{'path': str(p)} for p in artifacts]}


def publish(study, name, tasks, extra_pins=None):
    from .compression_finalize_v13 import verify
    verify(study)
    if not re.fullmatch(r'[a-zA-Z0-9_-]+', name):
        raise ValueError('unsafe plan name')
    root = Path(study['output_root'])
    path = root / 'plans' / name / 'plan.json'
    if path.exists():
        plan = read_plan(path)
        verify_sources(plan)
        if plan['tasks'] != tasks:
            raise ValueError('cannot replace published task commands')
        return path
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('v13 publication deadline reached')
    pins = {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json'),
            **(extra_pins or {})}
    plan = seal({'format': 'mlsys_compression_execution_v13', 'cwd': study['cwd'],
        'tasks': tasks, 'input_hashes': pins, 'unexpected_retries': 1,
        'created_utc_epoch': study['created_utc_epoch'], 'deadline_utc_epoch': study['deadline_utc_epoch'],
        'evidence': study['evidence']})
    write_immutable(path, plan)
    verify_sources(read_plan(path))
    return path


def publish_schedule(study, model, stage, split, schedule):
    from .compression_finalize_v13 import verify
    from . import compression_execution_v11 as old
    from .adaptive_experiment_v4 import validate_schedule, validate_cell_count
    from .selector_experiment_v5 import source_hashes
    from .compression_study_v12 import verify as verify_v12
    source = verify_v12(verify(study))
    root, spec = Path(study['output_root']), study['models'][model]
    manifest, feature_root, metadata, features = old.development_features(source, model, split)
    validate_schedule(schedule, manifest_sha256=study['manifest_sha256'],
        checkpoint_sha256=metadata['checkpoint_sha256'], split=split, allowed_ids=set(features))
    if schedule['study_sha256'] != study['sha256'] or schedule['answers_used_to_choose_counts'] is not False:
        raise ValueError('physical schedule is not bound to an answer-free v13 assignment')
    pins = {str(Path(study['cwd']) / 'learned_score' / name): sha
            for name, sha in source_hashes(spec['adapter']).items()}
    for request, geometry in schedule['rows'].items():
        for cell in geometry['cells']:
            validate_cell_count(cell, features[request], geometry)
        path = feature_root / 'features' / (request + '.json')
        pins[str(path)] = file_digest(path)
    schedule_path = root / model / 'schedules' / (stage + '-execution.json')
    write_immutable(schedule_path, schedule)
    pins[str(schedule_path)] = file_digest(schedule_path)
    destination, tasks = root / model / stage, []
    for rank in range(study['world_size']):
        argv = python(study, 'selector_experiment_v5') + ['--phase', 'physical',
            '--candidate', 'current_rank32', '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'],
            '--split', split, '--manifest', study['manifest'], '--schedule', str(schedule_path),
            '--adapter', spec['adapter'], '--model', spec['model_path'], '--model-revision', spec['revision'],
            '--output', str(destination), '--rank', str(rank), '--world-size', str(study['world_size']),
            '--query-control', 'correct', '--feature-root', str(feature_root)]
        if model == 'llava_next':
            argv.append('--reference-clip')
        task = make_task(study, f'{stage}-{model}-{rank}', argv,
                         [destination / f'rank{rank}.complete.json'], model=model)
        task.update(resume_checkpoint=str(destination / f'rank{rank}.metadata.json'), resume_args=['--resume'])
        tasks.append(task)
    return publish(study, stage + '-' + model, tasks, pins)


STAGES = ('learned-controls', 'div-deploy', 'div-matched')
ARMS = ('scorer_static', 'scorer_adaptive', 'divprune_static', 'divprune_adaptive',
        'scorer_matched', 'divprune_matched')
STAGE_ARMS = {'learned-controls': ('scorer_static', 'scorer_adaptive', 'scorer_matched'),
              'div-deploy': ('divprune_static', 'divprune_adaptive'),
              'div-matched': ('divprune_matched',)}


def assignment_catalog(rows, report, profiles):
    """Build deployment arms and cohort-reconciled diagnostic controls separately."""
    from .compression_policy_v11 import factorial_assignments
    from .compression_policy_v10 import canonical_dataset
    report = report.get('result', report)
    counts = {}
    for form in ('static', 'adaptive'):
        variant = profiles['forms'][form]['variant_id']
        counts[form] = report['procedures'][variant]['counts']
    ids = {row['row_hash'] for row in rows}
    if len(ids) != len(rows):
        raise ValueError('duplicate development input identity')
    for form, chosen in counts.items():
        if set(chosen) != ids or any(type(chosen[r['row_hash']]) is not int or
                not 1 <= chosen[r['row_hash']] <= r['input_visual_tokens'] for r in rows):
            raise ValueError('incomplete or invalid exact counts: ' + form)
    normalized = [{**r, 'dataset': canonical_dataset(r['dataset'])} for r in rows]
    factorial = factorial_assignments(normalized, counts['adaptive'], 'divprune')
    counts['matched'] = factorial['counts']['uniform']
    arms = {selector + '_' + form: {'selector': selector, 'counts': dict(counts[form]),
        'diagnostic_only': form == 'matched', 'deployable_assignment': form != 'matched'}
        for selector in ('scorer', 'divprune') for form in ('static', 'adaptive', 'matched')}
    return seal({'format': 'mlsys_compression_v13_assignments', 'cells': arms, 'counts': counts,
        'profile_variants': {f: profiles['forms'][f]['variant_id'] for f in ('static', 'adaptive')},
        'exact_aggregate_matched_within_every_dataset': True,
        'answers_used_to_choose_counts': False, 'quality_requires_physical_inference': True,
        'publication_independent_of_promotion_eligibility': True})


def _verify_publication(study, model, publication):
    if publication['study_sha256'] != study['sha256'] or publication['model'] != model:
        raise ValueError('publication belongs to another study/model')
    for path, sha in publication['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('published comparison input changed: ' + path)
    for path in publication['plans']:
        verify_sources(read_plan(path))


def publish_comparison(study, model, report, profiles):
    """Publish all six arms even when no development rule qualifies for deployment."""
    from .compression_finalize_v13 import verify
    from .compression_study_v12 import verify as verify_v12, collect_features
    from .compression_study_v11 import physical_inventory
    from .compression_reuse_v11 import split_missing
    from .adaptive_planning_v4 import _geometry
    from . import compression_execution_v11 as old
    source_v12 = verify(study)
    source = verify_v12(source_v12)
    root = Path(study['output_root'])
    publication_path = root / model / 'development/comparison-publication.json'
    expected = {'study_sha256': study['sha256'], 'model': model,
                'report_sha256': report['sha256'], 'profiles_sha256': profiles['sha256']}
    if publication_path.exists():
        publication = checked(publication_path)
        if any(publication.get(k) != v for k, v in expected.items()):
            raise ValueError('published comparison belongs to another frozen decision')
        _verify_publication(study, model, publication)
        return publication['plans']
    rows_path = root / model / 'development/rows.json'
    rows = checked(rows_path)['rows']
    catalog = assignment_catalog(rows, report, profiles)
    catalog_path = root / model / 'development/assignments.json'
    write_immutable(catalog_path, catalog)
    features = collect_features(source_v12, model)
    if features is None:
        raise ValueError('all audited v12 feature shards are required')
    inventory = {k: v for k, v in physical_inventory(source, model).items() if k[1] != 'cdpruner'}
    paths, pins, receipts = [], {str(rows_path): file_digest(rows_path),
        str(catalog_path): file_digest(catalog_path)}, {}
    for phase in STAGES:
        for split in old.SPLITS:
            _, _, metadata, native = old.development_features(source, model, split)
            scheduled = {}
            for request, feature in sorted(native.items()):
                n = feature['input_visual_tokens']
                cells = []
                for key in STAGE_ARMS[phase]:
                    assignment = catalog['cells'][key]
                    count = assignment['counts'][request]
                    cells.append({'key': key, 'selector': 'full' if count == n else assignment['selector'],
                        'selected_count': count, 'mode': 'exact', 'policy': 'v13_excluded_fold_counts'})
                scheduled[request] = {**_geometry(feature), 'cells': cells}
            master = seal({'format': 'mlsys_compression_execution_v13_schedule',
                'study_sha256': study['sha256'], 'model': model, 'split': split,
                'candidate': 'current_rank32', 'manifest_sha256': study['manifest_sha256'],
                'checkpoint_sha256': metadata['checkpoint_sha256'],
                'cache_contract': metadata['cache_contract'], 'features_sha256': digest(native),
                'rows': scheduled, 'answers_used_to_choose_counts': False,
                'training_labels_used_inside_other_folds': True, 'diagnostic_publication': True,
                'promotion_eligibility_required': False, 'evidence': study['evidence']})
            stage = split + '-' + phase
            missing, receipt = split_missing(master, inventory, features)
            for suffix, value in (('master', master), ('reuse', receipt)):
                path = root / model / 'schedules' / (stage + '-' + suffix + '.json')
                write_immutable(path, value)
                pins[str(path)] = file_digest(path)
            receipts[stage] = receipt['counts']
            if missing['rows']:
                path = publish_schedule(study, model, stage, split, missing)
                paths.append(str(path))
                pins[str(path)] = file_digest(path)
                execution = root / model / 'schedules' / (stage + '-execution.json')
                pins[str(execution)] = file_digest(execution)
            else:
                path = root / model / 'schedules' / (stage + '-all-reused.json')
                write_immutable(path, seal({'receipt_sha256': receipt['sha256'], 'all_reused': True}))
                pins[str(path)] = file_digest(path)
    write_immutable(publication_path, seal({**expected, 'plans': paths, 'input_hashes': pins,
        'stage_priority': list(STAGES), 'reuse_counts': receipts, 'diagnostic_only_publication': True,
        'complete_publication_not_complete_inference': True}))
    return paths


def collect_model(study, model):
    """Return all six complete physical cohorts plus the all-token reference, or None."""
    from .compression_finalize_v13 import verify
    from .compression_study_v12 import verify as verify_v12
    from .compression_study_v11 import physical_inventory
    from .compression_report_v12 import materialize
    from .compression_report_v11 import _verify_physical_provenance
    from .compression_matrix_v8 import scan_execution
    from .compression_policy_v10 import canonical_dataset
    source = verify_v12(verify(study))
    root = Path(study['output_root'])
    publication_path = root / model / 'development/comparison-publication.json'
    if not publication_path.exists():
        return None
    publication = checked(publication_path)
    _verify_publication(study, model, publication)
    catalog = checked(root / model / 'development/assignments.json')
    bundles = []
    for phase in STAGES:
        for split in ('fit', 'screening'):
            stage = split + '-' + phase
            master = checked(root / model / 'schedules' / (stage + '-master.json'))
            receipt = checked(root / model / 'schedules' / (stage + '-reuse.json'))
            if (master['study_sha256'] != study['sha256'] or master['model'] != model
                    or master['split'] != split or master['promotion_eligibility_required'] is not False):
                raise ValueError('physical master identity changed')
            fresh = []
            if receipt['counts']['missing_unique_cells']:
                execution = checked(root / model / 'schedules' / (stage + '-execution.json'))
                if execution['sha256'] != receipt['missing_schedule_sha256']:
                    raise ValueError('missing execution schedule changed')
                fresh = scan_execution(study, model, stage)
                if fresh is None:
                    return None
            bundles.append((master, receipt, fresh))
    # Historical inventories are large. Do not rescan them on every coordinator
    # tick while a declared stage still lacks complete physical shards.
    inventory = physical_inventory(source, model)
    measured = [row for master, receipt, fresh in bundles
                for row in materialize(master, receipt, fresh, inventory)]
    ids = set(catalog['counts']['adaptive'])
    references = [{**row, 'method': 'full', 'dataset': canonical_dataset(row['dataset'])}
        for (request, selector, _), row in inventory.items() if request in ids and selector == 'full']
    if len(references) != len(ids) or {r['request_id'] for r in references} != ids:
        raise ValueError('missing complete physical all-token reference')
    for arm in ARMS:
        values = [r for r in measured if r['method'] == arm]
        if len(values) != len(ids) or {r['request_id'] for r in values} != ids:
            raise ValueError('incomplete or duplicated physical method cohort: ' + arm)
        for row in values:
            if row['selected_count'] != catalog['cells'][arm]['counts'][row['request_id']]:
                raise ValueError('physical count differs from frozen assignment')
    for row in [*references, *measured]:
        _verify_physical_provenance({**row, 'physical_measurement_performed': False})
        actual = row.get('metrics', {}).get('cache', {}).get('kv_capacity_bytes')
        expected = (row['input_text_tokens'] + row['retained_visual_tokens'] +
                    study['generation_reserve']) * study['bytes_per_token'][model]
        if type(actual) is not int or actual != expected:
            raise ValueError('measured KV capacity differs from exact accounting')
    for dataset in {r['dataset'] for r in measured}:
        totals = [sum(r['metrics']['cache']['kv_capacity_bytes'] for r in measured
                    if r['dataset'] == dataset and r['method'] == arm)
                  for arm in ('scorer_adaptive', 'divprune_adaptive', 'scorer_matched', 'divprune_matched')]
        if len(set(totals)) != 1:
            raise ValueError('diagnostic controls do not exactly match allocated KV')
    return [*references, *measured]

