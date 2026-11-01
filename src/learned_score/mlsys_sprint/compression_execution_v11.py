"""Versioned development execution; frozen v10 inputs and outputs are read-only.

Plan publication never starts a process. Physical execution delegates to the
unchanged exact-count v5/v4 runner, preserving the original scorer and positions.
"""
import argparse
import json
import math
from pathlib import Path
import re
import time

from . import compression_matrix_v8 as matrix
from . import adaptive_planning_v4 as planning
from .followup_common import write_immutable
from .state import digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources

FORMAT = 'mlsys_compression_execution_v11'
MODELS = ('qwen25', 'llava_next')
SPLITS = ('fit', 'screening')
read, checked, seal = matrix.read, matrix.checked, matrix.seal


def plan_names():
    """Closed catalog for future publication; never start or create work."""
    stages = tuple(split + suffix for split in SPLITS
                   for suffix in ('-curves', '-selector-grid')) + tuple(
                   f'{split}-matched-p{target:03d}' for split in SPLITS for target in (50,70,90)) + ('systems',)
    return [f'{stage}-{model}' for stage in stages for model in MODELS]


def verify(study):
    planning._checked(study)
    cwd = Path(study['cwd']).resolve()
    root = resolve_inside(study['output_root'], cwd / 'reproduction_results')
    if (root.parent != cwd / 'reproduction_results'
            or not re.fullmatch(r'mlsys_compression_[0-9]{8}_v11', root.name)):
        raise ValueError('v11 must use its own output namespace')
    old = checked(study['v10_study'])
    if (root == Path(old['output_root']) or study['models'] != old['models']
            or study['manifest'] != old['manifest'] or study['manifest_sha256'] != old['manifest_sha256']
            or study['old_study'] != old['old_study']):
        raise ValueError('v11 changed frozen model, manifest, or source study')
    if type(study['world_size']) is not int or not 1 <= study['world_size'] <= 64:
        raise ValueError('invalid shard count')
    for path, expected in study['input_hashes'].items():
        if file_digest(path) != expected:
            raise ValueError('frozen v11 input changed: ' + path)
    if file_digest(study['manifest']) != study['manifest_sha256']:
        raise ValueError('development manifest changed')
    return old


def development_features(study, model, split):
    if model not in MODELS or split not in SPLITS:
        raise ValueError('only the two models and declared development splits are allowed')
    verify(study)
    manifest, output, checkpoint, metadata, features = planning._features(checked(study['old_study']), model, split)
    if (manifest.file_sha256 != study['manifest_sha256']
            or checkpoint != file_digest(study['models'][model]['checkpoint'])):
        raise ValueError('cached feature model or manifest changed')
    return manifest, output / (split + '_features'), metadata, features


def source_pins(study, model):
    from .selector_experiment_v5 import source_hashes
    cwd, root = Path(study['cwd']), Path(study['output_root'])
    pins = {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json'),
            str(Path(__file__).resolve()): file_digest(__file__),
            study['v10_study']: file_digest(study['v10_study'])}
    pins.update({str(cwd / 'learned_score' / name): sha
                 for name, sha in source_hashes(study['models'][model]['adapter']).items()})
    return pins


def _stage(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,90}', value):
        raise ValueError('unsafe stage name')
    return value


def selector_schedule(study, model, split, selector, *, retentions=tuple(range(10, 101, 10)), exclude_pairs=()):
    """Make only missing physical cells; exclusions must be externally audited.

    The caller supplies (request_id, selector, actual_count) tuples after checking
    old source/checkpoint/manifest identities. No automatic score-based reuse.
    """
    if selector not in ('scorer', 'divprune', 'cdpruner') or (selector == 'cdpruner' and model != 'llava_next'):
        raise ValueError('unsupported native selector/model adaptation')
    if not retentions or any(type(p) is not int or not 1 <= p <= 100 for p in retentions):
        raise ValueError('integer retention percentages required')
    _, _, metadata, features = development_features(study, model, split)
    exclusions = set(tuple(v) for v in exclude_pairs)
    if any(len(v) != 3 or v[0] not in features or v[1] not in ('scorer', 'divprune', 'cdpruner', 'full')
           or type(v[2]) is not int or not 1 <= v[2] <= features[v[0]]['input_visual_tokens'] for v in exclusions):
        raise ValueError('invalid physical reuse exclusions')
    rows = {}
    for request, feature in sorted(features.items()):
        n, cells, emitted = feature['input_visual_tokens'], [], set()
        for p in sorted(set(retentions)):
            count = (n*p+99)//100
            physical_selector = 'full' if count == n else selector
            identity = (request, physical_selector, count)
            if identity in exclusions or identity in emitted:
                continue
            emitted.add(identity)
            cells.append({'key': f'{selector}_p{p:03d}', 'selector': physical_selector,
                'selected_count': count, 'keep_percent': p, 'mode': 'uniform', 'policy': 'fixed_retention'})
        if cells:
            rows[request] = {**planning._geometry(feature), 'cells': cells}
    return seal({'format': FORMAT+'_schedule', 'study_sha256': study['sha256'],
        'model': model, 'split': split, 'candidate': 'current_rank32',
        'manifest_sha256': study['manifest_sha256'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
        'cache_contract': metadata['cache_contract'], 'features_sha256': digest(features),
        'rows': rows, 'answers_used_to_choose_counts': False, 'evidence': 'development only',
        'reuse_exclusions_sha256': digest(sorted(exclusions)),
        'selector_scope': 'projected_feature_adaptations_not_official_pipeline'})


def publish_schedule(study, model, stage, split, schedule, feature_root=None):
    """Publish any exact-count development schedule, without a policy dependency."""
    verify(study)
    _stage(stage)
    if model not in MODELS or split not in SPLITS:
        raise ValueError('physical publication is development-only')
    planning._checked(schedule)
    root, spec = Path(study['output_root']), study['models'][model]
    plan_path = root / f'plans/{stage}-{model}/plan.json'
    if plan_path.exists():
        plan = read_plan(plan_path)
        verify_sources(plan)
        if checked(root / model / f'schedules/{stage}-execution.json') != schedule:
            raise ValueError('cannot replace a published physical schedule')
        return plan_path
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('cannot publish work after the study deadline')
    manifest, default_features, metadata, features = development_features(study, model, split)
    feature_root = Path(feature_root or default_features).resolve()
    if feature_root != default_features.resolve():
        raise ValueError('physical execution requires the original audited development feature root')
    for key, value in {'study_sha256': study['sha256'], 'model': model, 'split': split,
                       'candidate': 'current_rank32', 'manifest_sha256': study['manifest_sha256'],
                       'checkpoint_sha256': metadata['checkpoint_sha256'], 'cache_contract': metadata['cache_contract']}.items():
        if schedule.get(key) != value:
            raise ValueError('physical schedule binding changed: ' + key)
    if schedule.get('answers_used_to_choose_counts') is not False or not schedule.get('rows'):
        raise ValueError('schedule must contain preassigned answer-free counts')
    from .adaptive_experiment_v4 import validate_schedule, validate_cell_count
    validate_schedule(schedule, manifest_sha256=study['manifest_sha256'], checkpoint_sha256=metadata['checkpoint_sha256'],
                      split=split, allowed_ids=set(features))
    pins = source_pins(study, model)
    for request, row in schedule['rows'].items():
        feature = features[request]
        if any(row.get(key) != value for key, value in planning._geometry(feature).items()):
            raise ValueError('schedule geometry differs from cached native geometry')
        for cell in row['cells']:
            validate_cell_count(cell, feature, row)
            if model != 'llava_next' and cell['selector'] == 'cdpruner':
                raise ValueError('Qwen reference-CLIP CDPruner adaptation is unsupported')
        path = feature_root / 'features' / (request + '.json')
        pins[str(path)] = file_digest(path)
    paths = {name: root / model / f'schedules/{stage}{suffix}.json'
             for name, suffix in [('master', ''), ('execution', '-execution'), ('reuse', '-reuse')]}
    write_immutable(paths['master'], schedule)
    write_immutable(paths['execution'], schedule)
    write_immutable(paths['reuse'], seal({'master_schedule_sha256': schedule['sha256'], 'cells': {}}))
    destination, tasks = root / model / stage, []
    for rank in range(study['world_size']):
        argv = matrix.previous.python(study, 'selector_experiment_v5') + ['--phase', 'physical',
            '--candidate', 'current_rank32', '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'],
            '--split', split, '--manifest', study['manifest'], '--schedule', str(paths['execution']),
            '--adapter', spec['adapter'], '--model', spec['model_path'], '--model-revision', spec['revision'],
            '--output', str(destination), '--rank', str(rank), '--world-size', str(study['world_size']),
            '--query-control', 'correct', '--feature-root', str(feature_root)]
        if model == 'llava_next':
            argv.append('--reference-clip')
        task = matrix.previous.task(study, f'{stage}-{model}-{rank}', argv,
            [destination / f'rank{rank}.complete.json'], resume=destination / f'rank{rank}.metadata.json')
        task.update(id=f'compression-v11-{stage}-{model}-{rank}', tag='compression-v11', model_tag=model)
        tasks.append(task)
    pins.update({str(p): file_digest(p) for p in (*paths.values(), Path(study['manifest']), Path(spec['checkpoint']))})
    plan = seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': tasks, 'input_hashes': pins,
                 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
                 'deadline_utc_epoch': study['deadline_utc_epoch'], 'evidence': 'development only'})
    write_immutable(plan_path, plan)
    verify_sources(read_plan(plan_path))
    return plan_path


def prepare(study, model, *, selector, exclude_pairs=()):
    """Explicit selector grid only; use selector_schedule for split-local reuse."""
    paths = []
    for split in SPLITS:
        _, _, _, native_features = development_features(study, model, split)
        exclusions = [v for v in exclude_pairs if v[0] in native_features]
        schedule = selector_schedule(study,model,split,selector,exclude_pairs=exclusions)
        if schedule['rows']:
            paths.append(str(publish_schedule(study,model,split+'-selector-grid',split,schedule)))
    return {'model': model, 'status': 'prepared_not_launched', 'plans': paths}


def collect_stage(study, model, stage):
    verify(study)
    _stage(stage)
    plan_path = Path(study['output_root']) / f'plans/{stage}-{model}/plan.json'
    if not plan_path.exists():
        return None
    verify_sources(read_plan(plan_path))
    rows = matrix.scan_execution(study, model, stage)
    if rows is None:
        return None
    for row in rows:
        if row.get('selector') != row['schedule_cell']['selector'] or row.get('selected_count') != row['schedule_cell']['selected_count']:
            raise ValueError('physical selector/count differs from schedule')
        quality = row.get('quality_score')
        if type(quality) not in (int, float) or not math.isfinite(quality) or not 0 <= quality <= 1:
            raise ValueError('physical quality must be a finite score in [0, 1]')
    return rows


def collect(study, model):
    rows = []
    for split in SPLITS:
        current = collect_stage(study, model, split + '-selector-grid')
        if current is None:
            return None
        rows.extend(current)
    return rows


def prepare_features(study, model, selectors=('scorer', 'divprune')):
    from . import compression_features_v11 as features_api
    verify(study)
    root, spec = Path(study['output_root']), study['models'][model]
    selectors = features_api.validate_selectors(selectors, model)
    paths = []
    for split in SPLITS:
        _, feature_root, _, features = development_features(study, model, split)
        stage, tasks = split + '-curves', []
        destination = root / model / stage
        for rank in range(study['world_size']):
            argv = matrix.previous.python(study, 'compression_features_v11') + ['--study', str(root / 'study.json'),
                '--model-tag', model, '--adapter', spec['adapter'], '--split', split,
                '--rank', str(rank), '--world-size', str(study['world_size']), '--selectors', *selectors]
            task = matrix.previous.task(study, f'{stage}-{model}-{rank}', argv,
                [destination / f'rank{rank}.complete.json'], resume=destination / f'rank{rank}.metadata.json')
            task.update(id=f'compression-v11-{stage}-{model}-{rank}', tag='compression-v11', model_tag=model)
            tasks.append(task)
        pins = {**source_pins(study, model), **features_api.source_pins(spec['adapter'])}
        pins.update({str(feature_root / 'features' / (r + '.json')):
                     file_digest(feature_root / 'features' / (r + '.json')) for r in features})
        path = root / f'plans/{stage}-{model}/plan.json'
        if not path.exists() and time.time() >= study['deadline_utc_epoch']:
            raise ValueError('cannot publish coverage work after study deadline')
        write_immutable(path, seal({'format': FORMAT + '_feature_plan', 'cwd': study['cwd'], 'tasks': tasks,
            'input_hashes': pins, 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
            'deadline_utc_epoch': study['deadline_utc_epoch'], 'answers_loaded': False}))
        verify_sources(read_plan(path))
        paths.append(str(path))
    return {'model': model, 'status': 'prepared_not_launched', 'plans': paths}


def collect_features(study, model):
    from . import compression_features_v11 as features_api
    return features_api.collect(study, model)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--model-tag', choices=MODELS, required=True)
    p.add_argument('--phase', choices=('prepare', 'collect', 'prepare-features', 'collect-features'), required=True)
    p.add_argument('--selector', choices=('scorer','divprune','cdpruner'))
    args = p.parse_args(argv)
    function = {'prepare': prepare, 'collect': collect, 'prepare-features': prepare_features,
                'collect-features': collect_features}[args.phase]
    if args.phase=='prepare' and not args.selector:
        p.error('physical grid preparation requires an explicitly selected patch selector')
    value = function(checked(args.study), args.model_tag,**({'selector':args.selector} if args.phase=='prepare' else {}))
    print(json.dumps(value, sort_keys=True))


if __name__ == '__main__':
    main()

