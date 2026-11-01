"""One-day, immutable-input coverage sprint; explicit prepare/watch execute work.

Only development outcomes are read here. Independent evaluation stays closed
until identity admission succeeds. A completed feature job is not a completed
experiment, and a failed scientific candidate never becomes a forced winner.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import time

from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261022_v12')
SOURCE = Path('reproduction_results/mlsys_compression_20261020_v11/study.json')
FORMAT = 'mlsys_compression_study_v12'
MODELS = ('qwen25', 'llava_next')
KINDS = ('static', 'missing_mass', 'rank32_coverage', 'native_coverage')


def bytes_per_slot(config, dtype_bytes=2):
    """Real model-wide K+V storage per batch-one token; not hidden size alone."""
    config = config.get('text_config', config)
    layers = int(config['num_hidden_layers'])
    heads = int(config.get('num_key_value_heads', config['num_attention_heads']))
    dim = int(config.get('head_dim', config['hidden_size'] // config['num_attention_heads']))
    if min(layers, heads, dim, dtype_bytes) < 1:
        raise ValueError('positive KV geometry required')
    return 2 * layers * heads * dim * dtype_bytes


def verify(study, *, sources=True):
    if study.get('sha256') != digest({k: v for k, v in study.items() if k != 'sha256'}):
        raise ValueError('v12 study content changed')
    cwd, root = Path(study['cwd']).resolve(), Path(study['output_root']).resolve()
    if (study.get('format') != FORMAT or root != cwd / ROOT
            or study.get('scorer_training') is not False
            or study.get('dataset_is_policy_input') is not False
            or study.get('confirmation_labels_opened') is not False
            or not 0 < study['deadline_utc_epoch'] - study['created_utc_epoch'] <= 86400):
        raise ValueError('invalid bounded v12 scope')
    source = checked(study['source_study'])
    if (study['models'] != source['models'] or study['manifest'] != source['manifest']
            or file_digest(study['source_study']) != study['source_study_sha256']):
        raise ValueError('frozen source model/manifest changed')
    if sources:
        for name, expected in study['input_hashes'].items():
            if file_digest(name) != expected:
                raise ValueError('pinned v12 input changed: ' + name)
    return source


def prepare(cwd, *, now=None, hours=24, world_size=16):
    cwd = Path(cwd).resolve()
    path = cwd / ROOT / 'study.json'
    if path.exists():
        value = checked(path)
        verify(value)
        return value
    if not 0 < hours <= 24 or type(world_size) is not int or not 1 <= world_size <= 64:
        raise ValueError('bounded duration and shard count required')
    from . import compression_execution_v11 as source_execution
    source_path = cwd / SOURCE
    old = checked(source_path)
    source_execution.verify(old)
    pins = {**old['input_hashes'], str(source_path): file_digest(source_path)}
    modules = ('compression_study_v12', 'compression_features_v12', 'compression_policy_v12',
               'compression_execution_v12', 'compression_operations_v12', 'compression_text_v12',
               'compression_report_v12', 'compression_systems_v12', 'compression_admission_v12',
               'compression_full_suite_v12')
    for name in modules:
        source = Path(__file__).with_name(name + '.py')
        pins[str(source.resolve())] = file_digest(source)
    from transformers import AutoConfig
    cache_bytes, resolved_configs = {}, {}
    for model, spec in old['models'].items():
        config_path = Path(spec['model_path']) / 'config.json'
        pins[str(config_path)] = file_digest(config_path)
        # Hugging Face config.json may omit architecture defaults (Mistral's
        # layer/head dimensions here). Resolve with the same local config class
        # used by inference rather than guessing missing KV dimensions.
        resolved = AutoConfig.from_pretrained(spec['model_path'], local_files_only=True,
            trust_remote_code=False).to_dict()
        resolved_configs[model] = resolved
        cache_bytes[model] = bytes_per_slot(resolved)
    now = time.time() if now is None else now
    value = seal({'format': FORMAT, 'cwd': str(cwd), 'output_root': str(cwd / ROOT),
        'source_study': str(source_path), 'source_study_sha256': file_digest(source_path),
        'models': old['models'], 'manifest': old['manifest'], 'manifest_sha256': old['manifest_sha256'],
        'created_utc_epoch': now, 'deadline_utc_epoch': now + hours * 3600,
        'method_freeze_utc_epoch': now + min(hours, 8) * 3600,
        'identity_audit_deadline_utc_epoch': now + min(hours, 2) * 3600,
        'world_size': world_size, 'input_hashes': pins, 'keep_percent': list(range(10, 101, 10)),
        'coverage_geometries': ['rank32', 'native'], 'allocation_procedures': list(KINDS),
        'max_mean_loss_pp': 1., 'max_domain_loss_pp': 2., 'cost_tie_fraction': .01,
        'maximum_latency_overhead_fraction': .05, 'outer_folds': 5, 'inner_folds': 3,
        'max_new_tokens': 128, 'generation_reserve': 127, 'bytes_per_token': cache_bytes,
        'resolved_model_configs': resolved_configs,
        'text_eviction': False, 'scorer_training': False, 'dataset_is_policy_input': False,
        'text_pilot_hours': 4, 'text_profiles': [512, 256], 'text_max_new_tokens': 4096,
        'datasets': old['datasets'], 'full_suite_scope': 'complete eligible official evaluation splits',
        'full_suite_targets': ['quality_loss_1pp', 50, 70, 90],
        'baseline_scope': old['baseline_scope'], 'confirmation_labels_opened': False,
        'independent_confirmation': False, 'evidence': 'development on previously inspected data'})
    write_immutable(path, value)
    verify(value)
    return value


def publish_features(study, model):
    from .compression_execution_v12 import publish, make_task, python
    verify(study)
    if model not in MODELS:
        raise ValueError('unsupported model')
    root = Path(study['output_root'])
    tasks = []
    for rank in range(study['world_size']):
        argv = python(study, 'compression_features_v12') + ['--study', str(root / 'study.json'),
            '--model-tag', model, '--rank', str(rank), '--world-size', str(study['world_size']), '--resume']
        tasks.append(make_task(study, f'features-{model}-{rank}', argv,
            [root / model / 'features' / f'rank{rank}.complete.json'], model=model))
    return publish(study, 'features-' + model, tasks)


def collect_features(study, model):
    root = Path(study['output_root']) / model / 'features'
    if any(not (root / f'rank{i}.complete.json').exists() for i in range(study['world_size'])):
        return None
    from .compression_features_v12 import collect
    return collect(study, model)


def development_rows(study, model, features):
    from .compression_diagnostics_v11 import load_development
    from .compression_features_v12 import assert_physical_prefix_parity
    source = verify(study)
    rows, _, index, provenance = load_development(model, source['v10_study'])
    if set(features) != {r['row_hash'] for r in rows}:
        raise ValueError('native features do not cover the exact physical cohort')
    merged = []
    for original in rows:
        key = original['row_hash']
        feature = features[key]
        for field in ('input_visual_tokens', 'input_text_tokens', 'group_id'):
            if feature[field] != original[field]:
                raise ValueError('native/physical geometry mismatch: ' + field)
        hashes = {}
        for count in map(int, original['quality_by_count']):
            selector = 'full' if count == original['input_visual_tokens'] else 'scorer'
            physical = index[(key, selector, count)]
            indices = physical['selected_patch_indices']
            if len(indices) != count or len(set(indices)) != count or indices != sorted(indices):
                raise ValueError('invalid cached physical patch positions')
            hashes[count] = digest(indices)
        assert_physical_prefix_parity(feature['curves'], hashes)
        merged.append({**copy.deepcopy(original), 'curves': copy.deepcopy(feature['curves'])})
    return merged, provenance


def calibrate(study, model, features):
    from .compression_policy_v12 import development_report
    root = Path(study['output_root'])
    path = root / model / 'development/report.json'
    if path.exists():
        return checked(path)
    if time.time() >= study['method_freeze_utc_epoch']:
        return None
    rows, provenance = development_rows(study, model, features)
    result = development_report(rows, model=model, bytes_per_token=study['bytes_per_token'][model],
        max_mean_loss_pp=study['max_mean_loss_pp'], max_domain_loss_pp=study['max_domain_loss_pp'],
        generation_reserve=study['generation_reserve'], outer_folds=study['outer_folds'],
        inner_folds=study['inner_folds'])
    report = seal({'study_sha256': study['sha256'], 'model': model, 'rows_sha256': digest(rows),
        'provenance': provenance, 'result': result, 'evidence': study['evidence']})
    write_immutable(root / model / 'development/rows.json', seal({'rows': rows, 'provenance': provenance}))
    write_immutable(path, report)
    return report


def identity_status(study):
    """Publish a CPU-only fresh audit; never delay GPU feature publication."""
    from .compression_execution_v12 import publish, make_task, python
    from .compression_admission_v12 import full_suite_readiness
    verify(study)
    root = Path(study['output_root'])
    path = root/'admission/status.json'
    if path.exists():
        value = checked(path)
        if value.get('study_sha256') != study['sha256'] or value.get('labels_loaded') is not False:
            raise ValueError('admission is not bound to this answer-free v12 study')
        result = seal({'format': FORMAT+'_identity_status', 'study_sha256': study['sha256'],
            'source_path': str(path), 'source_sha256': file_digest(path), 'source_status': value,
            'labels_opened': False, 'status': value['status'],
            'independent_confirmation_authorized': bool(value.get('manifests'))})
    else:
        argv = python(study, 'compression_admission_v12') + ['--study', str(root/'study.json')]
        task = make_task(study, 'identity-admission', argv, [path], model='cpu', resource='cpu')
        task['minimum_start_seconds'] = 60
        plan = publish(study, 'identity-admission', [task])
        result = seal({'format': FORMAT+'_identity_status', 'study_sha256': study['sha256'],
            'status': 'pending_bounded_identity_audit', 'plan_path': str(plan),
            'argv': argv, 'resource': 'cpu', 'deadline_utc_epoch': study['identity_audit_deadline_utc_epoch'],
            'labels_opened': False, 'independent_confirmation_authorized': False,
            'execution_note': 'Run dedicated CPU task concurrently with GPU extraction; GPU queues exclude this plan.'})
    atomic_json(root/'data/admission-status.json', result)
    atomic_json(root/'full-suite/status.json', full_suite_readiness(study, result.get('source_status')))
    return result


def shared_selection(study, reports):
    """Single procedure across models. Timing remains a separate admission gate."""
    from .compression_policy_v12 import select_shared
    return select_shared({model: report['result'] for model, report in reports.items()},
                         cost_tie_fraction=study['cost_tie_fraction'])


def advance(study):
    verify(study)
    root, now = Path(study['output_root']), time.time()
    status = {'time': now, 'study_sha256': study['sha256'], 'stage': 'features', 'models': {},
              'independent_confirmation': False, 'full_suite': 'blocked_identity_admission'}
    if now >= study['deadline_utc_epoch']:
        status['stage'] = 'day_deadline'
        atomic_json(root / 'status.json', status)
        return status
    from .compression_full_suite_v12 import advance as advance_full
    status['full_suite'] = advance_full(study)
    reports = {}
    for model in MODELS:
        publish_features(study, model)
        completed_report = root / model / 'development/report.json'
        if completed_report.exists():
            report = checked(completed_report)
            if report['study_sha256'] != study['sha256'] or report['model'] != model:
                raise ValueError('completed calibration belongs to another study/model')
            reports[model] = report
            status['models'][model] = 'calibrated'
            continue
        features = collect_features(study, model)
        if features is None:
            status['models'][model] = 'awaiting_features'
            continue
        report = calibrate(study, model, features)
        status['models'][model] = 'calibrated' if report else 'method_freeze_without_complete_calibration'
        if report:
            reports[model] = report
    selected_path = root / 'selection.json'
    if len(reports) == len(MODELS):
        if not selected_path.exists():
            decision = shared_selection(study, reports)
            write_immutable(selected_path, seal({'format': 'mlsys_compression_selection_v12',
                'study_sha256': study['sha256'], 'created_utc_epoch': now, **decision,
                'evidence': 'development selection; physical controls and timing required'}))
        selection = checked(selected_path)
        status['selected_rule'] = selection['rule_kind']
        if selection['rule_kind'] is None:
            status['stage'] = 'no_shared_compression_rule_meets_constraints'
            status['safe_action'] = 'all_image_tokens; no forced scientific winner'
        else:
            from .compression_execution_v12 import publish_comparison
            from .compression_text_v12 import advance as advance_text
            from .compression_systems_v12 import advance as advance_systems
            status['text_pilot'] = advance_text(study, selected_path)
            status['systems'] = {}
            for model in MODELS:
                publish_comparison(study, model, reports[model]['result'], selection)
                status['systems'][model] = advance_systems(study, model, selection)
            status['stage'] = 'selected_physical_comparisons'
    elif now >= study['method_freeze_utc_epoch']:
        status['stage'] = 'method_frozen_existing_baseline_incomplete_evidence'
    atomic_json(root / 'status.json', status)
    write_report(study, reports, status)
    return status


def write_report(study, reports, status):
    """Machine-readable paper tables; incomplete cells are never fabricated."""
    from .compression_report_v12 import generate
    return generate(study, reports, status)


def watch(study, once=False):
    root = Path(study['output_root'])
    with (root / 'coordinator.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while time.time() < study['deadline_utc_epoch']:
            try:
                value = advance(study)
                print(json.dumps(value), flush=True)
            except Exception as error:
                atomic_json(root / 'coordinator-error.json', {'time': time.time(),
                    'type': type(error).__name__, 'error': str(error)})
                raise
            if once:
                return value
            time.sleep(min(30, max(0, study['deadline_utc_epoch'] - time.time())))
    return {'stage': 'day_deadline'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'once', 'watch', 'status'))
    parser.add_argument('--study', type=Path, default=ROOT / 'study.json')
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.phase == 'prepare':
        study = prepare(args.cwd)
        identity_status(study)
        value = {'study': study['output_root'], 'plans': [str(publish_features(study, m)) for m in MODELS]}
    else:
        study = checked(args.study)
        verify(study)
        value = (json.loads((Path(study['output_root']) / 'status.json').read_text())
                 if args.phase == 'status' else watch(study, once=args.phase == 'once'))
    print(json.dumps(value, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
