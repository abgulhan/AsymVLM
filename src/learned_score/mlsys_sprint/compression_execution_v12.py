"""Immutable v12 task publication using the unchanged physical v5 runner."""
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
    return {'id': 'compression-v12-' + name, 'tag': 'compression-v12', 'resource': resource,
        'model_tag': model, 'minimum_start_seconds': minimum_start_seconds, 'depends_on': [],
        'argv': argv, 'env': {**environment(Path(study['cwd'])), 'PYTHONDONTWRITEBYTECODE': '1'},
        'artifacts': [{'path': str(p)} for p in artifacts]}


def publish(study, name, tasks, extra_pins=None):
    from .compression_study_v12 import verify
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
        raise ValueError('v12 publication deadline reached')
    pins = {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json'),
            **(extra_pins or {})}
    plan = seal({'format': 'mlsys_compression_execution_v12', 'cwd': study['cwd'],
        'tasks': tasks, 'input_hashes': pins, 'unexpected_retries': 1,
        'created_utc_epoch': study['created_utc_epoch'], 'deadline_utc_epoch': study['deadline_utc_epoch'],
        'evidence': study['evidence']})
    write_immutable(path, plan)
    verify_sources(read_plan(path))
    return path


def publish_schedule(study, model, stage, split, schedule):
    from .compression_study_v12 import verify
    from . import compression_execution_v11 as old
    from .adaptive_experiment_v4 import validate_schedule, validate_cell_count
    from .selector_experiment_v5 import source_hashes
    source = verify(study)
    root, spec = Path(study['output_root']), study['models'][model]
    manifest, feature_root, metadata, features = old.development_features(source, model, split)
    validate_schedule(schedule, manifest_sha256=study['manifest_sha256'],
        checkpoint_sha256=metadata['checkpoint_sha256'], split=split, allowed_ids=set(features))
    if schedule['study_sha256'] != study['sha256'] or schedule['answers_used_to_choose_counts'] is not False:
        raise ValueError('physical schedule is not bound to an answer-free v12 assignment')
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


def publish_comparison(study, model, report, selection):
    """OOF counts versus exact-matched uniform controls; no interpolated scores."""
    from .compression_study_v12 import verify, collect_features
    from .compression_study_v11 import physical_inventory
    from .compression_policy_v11 import factorial_assignments
    from .compression_reuse_v11 import split_missing
    from .adaptive_planning_v4 import _geometry
    from . import compression_execution_v11 as old
    source = verify(study)
    root = Path(study['output_root'])
    publication_path = root / model / 'development/comparison-publication.json'
    expected = {'study_sha256': study['sha256'], 'model': model,
                'report_sha256': report['sha256'], 'selection_sha256': selection['sha256']}
    if publication_path.exists():
        publication = checked(publication_path)
        if any(publication.get(key) != value for key, value in expected.items()):
            raise ValueError('published comparison belongs to another frozen decision')
        for path, sha in publication['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('published comparison input changed: ' + path)
        for path in publication['plans']:
            verify_sources(read_plan(path))
        return publication['plans']
    if selection.get('rule_kind') not in report.get('procedures', report.get('per_kind', {})):
        raise ValueError('no selected shared rule; do not force physical comparison')
    rows = checked(root / model / 'development/rows.json')['rows']
    procedures = report.get('procedures', report.get('per_kind', {}))
    counts = procedures[selection['rule_kind']]['counts']
    factorial = factorial_assignments(rows, counts, 'divprune')
    write_immutable(root / model / 'development/factorial.json', seal(factorial))
    features = collect_features(study, model)
    inventory = physical_inventory(source, model)
    # CD cross-site historical disagreement must not be hidden by cache reuse.
    inventory = {k: v for k, v in inventory.items() if k[1] != 'cdpruner'}
    paths = []
    for split in old.SPLITS:
        _, _, metadata, native = old.development_features(source, model, split)
        scheduled = {}
        for request, feature in sorted(native.items()):
            n = feature['input_visual_tokens']
            cells = [{'key': key, 'selector': 'full' if cell['counts'][request] == n else cell['selector'],
                'selected_count': cell['counts'][request], 'mode': 'exact', 'policy': 'v12_excluded_fold_counts'}
                for key, cell in factorial['cells'].items()]
            scheduled[request] = {**_geometry(feature), 'cells': cells}
        master = seal({'format': 'mlsys_compression_execution_v12_schedule',
            'study_sha256': study['sha256'], 'model': model, 'split': split, 'candidate': 'current_rank32',
            'manifest_sha256': study['manifest_sha256'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
            'cache_contract': metadata['cache_contract'], 'features_sha256': digest(native),
            'rows': scheduled, 'answers_used_to_choose_counts': False,
            'training_labels_used_inside_other_folds': True, 'evidence': study['evidence']})
        stage = split + '-matched'
        missing, receipt = split_missing(master, inventory, features)
        write_immutable(root / model / 'schedules' / (stage + '-master.json'), master)
        write_immutable(root / model / 'schedules' / (stage + '-reuse.json'), receipt)
        if missing['rows']:
            paths.append(str(publish_schedule(study, model, stage, split, missing)))
        else:
            write_immutable(root / model / 'schedules' / (stage + '-all-reused.json'),
                            seal({'receipt_sha256': receipt['sha256'], 'all_reused': True}))
    pins = {}
    for path in [root / model / 'development/factorial.json',
                 *[root / model / 'schedules' / (split + '-matched-' + suffix + '.json')
                   for split in old.SPLITS for suffix in ('master', 'reuse')],
                 *map(Path, paths)]:
        pins[str(path)] = file_digest(path)
    for split in old.SPLITS:
        for suffix in ('execution', 'all-reused'):
            path = root / model / 'schedules' / (split + '-matched-' + suffix + '.json')
            if path.exists():
                pins[str(path)] = file_digest(path)
    write_immutable(publication_path, seal({**expected, 'plans': paths, 'input_hashes': pins,
        'complete_publication_not_complete_inference': True}))
    return paths
