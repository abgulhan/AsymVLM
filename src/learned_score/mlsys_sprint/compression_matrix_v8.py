"""Physical patch-retention matrix, independently frozen from policy fitting.

All schedules are answer-free. Existing development outputs may replace a
measurement only after their complete original stage has passed its audits.
This module never reads historical evaluation labels or submits a job.
"""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

from . import adaptive_planning_v4 as planning
from . import selector_study_v5 as previous
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

ROOT = Path('reproduction_results/mlsys_compression_20261018_v8')
FORMAT = 'mlsys_compression_matrix_v8'
MODELS = ('qwen25', 'llava_next')
STAGES = ('fit-high', 'screen-high', 'fit-low', 'screen-low')
KEEP = (30, 40, 50, 60, 70, 80, 90, 100)
WORLD = 16


def read(path):
    return json.loads(Path(path).read_text())


def seal(value):
    return {**value, 'sha256': digest(value)}


def checked(path):
    return planning._checked(read(path))


def count_menu(n):
    if type(n) is not int or n < 1:
        raise ValueError('positive original visual-token count required')
    return {p: (n * p + 99) // 100 for p in KEEP}


def plan_names():
    return [f'{stage}-{model}' for stage in (*STAGES, 'matched', 'confirmation-features',
                                           'confirmation', 'systems') for model in MODELS]


def prepare(cwd, *, hours=24):
    cwd = Path(cwd).resolve()
    root = cwd / ROOT
    path = root / 'study.json'
    if path.exists():
        study = checked(path)
        verify(study)
        return study
    if not 0 < hours <= 168:
        raise ValueError('execution window must be positive and at most one week')
    old_path = cwd / 'reproduction_results/mlsys_adaptive_20261016_v4/study.json'
    v5_path = cwd / 'reproduction_results/mlsys_selector_20261017_v5/study.json'
    old, v5 = checked(old_path), checked(v5_path)
    previous.verify_study(v5)
    from .adaptive_data_v4 import load_manifest
    manifest = load_manifest(old['manifest'])
    summary_path = old_path.parent / 'data/summary.json'
    summary = read(summary_path)
    pins = {str(old_path): file_digest(old_path), str(v5_path): file_digest(v5_path),
            str(summary_path): file_digest(summary_path), old['manifest']: file_digest(old['manifest']),
            str(Path(__file__).resolve()): file_digest(__file__)}
    for model in MODELS:
        pins[old['models'][model]['checkpoint']] = file_digest(old['models'][model]['checkpoint'])
    created = time.time()
    study = seal({'format': FORMAT, 'cwd': str(cwd), 'output_root': str(root),
        'created_utc_epoch': created, 'deadline_utc_epoch': created + hours * 3600,
        'models': {m: old['models'][m] for m in MODELS}, 'old_study': str(old_path),
        'v5_study': str(v5_path), 'manifest': old['manifest'], 'manifest_sha256': manifest.file_sha256,
        'input_hashes': pins, 'world_size': WORLD, 'keep_percent': list(KEEP),
        'objective': 'minimum aggregate retained image tokens subject to full-relative quality loss',
        'maximum_measured_loss_pp': 1.0, 'strict_upper_loss_limit_pp': 1.0,
        'confidence': .95, 'bootstrap_repetitions': 10000, 'bootstrap_seed': 20261018,
        'fit_allowances_pp': [0., .25, .5], 'maximum_scalar_bins': 3, 'minimum_bin_groups': 20,
        'candidate_families': ['normalized_entropy', 'local_shell', 'nearest_neighbor_diversity', 'combined_ridge'],
        'local_shell': 'top_mass_by_percent[60] - top_mass_by_percent[40]',
        'max_new_tokens': 128, 'generated_budget': 512, 'text_eviction': False,
        'scorer_training': False, 'evidence': 'development',
        'confirmation': {'non_mme_target_groups': 1000, 'one_question_per_group': True,
            'mme': 'all eligible unseen intact pairs, category stratified',
            'historical_audit_complete': summary['historical_audit_complete'],
            'historical_unresolved_artifacts': summary['historical_unresolved_artifacts'],
            'status': 'requires_fresh_identity_audit'},
        'baseline_scope': {'divprune': 'selector adaptation',
            'cdpruner': 'LLaVA-NeXT independent-reference-CLIP selector adaptation; unavailable on Qwen'},
        'split_roles': {'fit': 'development fitting', 'screening': 'development selection',
                        'evaluation': 'previously inspected; not opened by this study'}})
    write_immutable(path, study)
    return study


def verify(study):
    planning._checked(study)
    if study.get('format') != FORMAT or Path(study['output_root']) != Path(study['cwd']) / ROOT:
        raise ValueError('wrong v8 study or output root')
    for path, sha in study['input_hashes'].items():
        if file_digest(path) != sha:
            raise ValueError('frozen matrix input changed: ' + path)


def features(study, model, split):
    if model not in MODELS or split not in ('fit', 'screening'):
        raise ValueError('matrix only opens declared development features')
    return planning._features(checked(study['old_study']), model, split)


def make_schedule(study, model, stage, metadata, feature_rows):
    if stage not in STAGES:
        raise ValueError('unknown matrix stage')
    split = 'fit' if stage.startswith('fit') else 'screening'
    keep = (70, 80, 90, 100) if stage.endswith('high') else (30, 40, 50, 60)
    selectors = ('scorer',) if split == 'fit' else ('scorer', 'divprune') + (('cdpruner',) if model == 'llava_next' else ())
    rows = {}
    for request, feature in sorted(feature_rows.items()):
        menu, cells = count_menu(feature['input_visual_tokens']), []
        for p in keep:
            if p == 100:
                cells.append({'key': 'full_p100', 'selector': 'full', 'selected_count': menu[p],
                              'keep_percent': p, 'policy': 'full', 'mode': 'full'})
            else:
                for selector in selectors:
                    cells.append({'key': f'{selector}_p{p:03d}', 'selector': selector,
                                  'selected_count': menu[p], 'keep_percent': p,
                                  'policy': 'fixed_retention', 'mode': 'uniform'})
        rows[request] = {**planning._geometry(feature), 'cells': cells}
    return seal({'format': FORMAT + '_schedule', 'study_sha256': study['sha256'],
        'stage': stage, 'model': model, 'split': split, 'candidate': 'current_rank32',
        'manifest_sha256': study['manifest_sha256'], 'checkpoint_sha256': metadata['checkpoint_sha256'],
        'cache_contract': metadata['cache_contract'], 'features_sha256': digest(feature_rows),
        'rows': rows, 'answers_used_to_choose_counts': False})


def runtime_contract(metadata):
    """Selection/runtime identity; deliberately excludes schedule and GPU name."""
    backbone = copy.deepcopy(metadata['backbone'])
    return {'backbone': backbone, 'cache_contract': metadata['cache_contract'],
            'checkpoint_sha256': metadata['checkpoint_sha256'], 'max_new_tokens': metadata['max_new_tokens'],
            'generated_budget': metadata['generated_budget'], 'source_sha256': metadata['source_sha256']}


def cache_inventory(study, model):
    """Only v5 stages with selected-index evidence are eligible for exact reuse."""
    from .recovery_study_v6 import cache_inventory as inventory
    result = inventory(study, model)
    for key, value in result.items():
        row = value['row']
        indices = row.get('selected_patch_indices')
        if (not isinstance(indices, list) or len(indices) != key[2]
                or indices != sorted(set(indices)) or digest(indices) != row.get('selected_patch_indices_sha256')):
            raise ValueError('reusable row lacks exact selected-index proof')
        directory = Path(value['path']).parent.parent
        meta_path = directory / 'rank0.metadata.json'
        value['metadata_path'], value['metadata_sha256'] = str(meta_path), file_digest(meta_path)
        value['runtime_contract_sha256'] = digest(runtime_contract(read(meta_path)))
    return result


def split_reuse(master, inventory):
    execution = copy.deepcopy({k: v for k, v in master.items() if k != 'sha256'})
    execution['rows'] = {}
    reused = {}
    for request, row in master['rows'].items():
        missing = []
        for cell in row['cells']:
            source = inventory.get((request, cell['selector'], cell['selected_count']))
            if source is None:
                missing.append(cell)
            else:
                old = source['row']
                if any(old.get(k) != row[k] for k in ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')):
                    raise ValueError('cached physical request geometry or identity changed')
                reused[f"{request}.{cell['key']}"] = {k: source[k] for k in
                    ('path', 'sha256', 'metadata_path', 'metadata_sha256', 'runtime_contract_sha256')}
        if missing:
            execution['rows'][request] = {**row, 'cells': missing}
    execution['master_schedule_sha256'] = master['sha256']
    return seal(execution), seal({'master_schedule_sha256': master['sha256'], 'cells': reused})


def publish(study, model, stage, inventory=None):
    verify(study)
    root = Path(study['output_root'])
    paths = {name: root / model / 'schedules' / f'{stage}{suffix}.json'
             for name, suffix in (('master', ''), ('execution', '-execution'), ('reuse', '-reuse'))}
    plan_path = root / f'plans/{stage}-{model}/plan.json'
    if plan_path.exists():
        from .worker import read_plan, verify_sources
        verify_sources(read_plan(plan_path))
        return plan_path
    split = 'fit' if stage.startswith('fit') else 'screening'
    _, _, _, metadata, values = features(study, model, split)
    master = make_schedule(study, model, stage, metadata, values)
    execution, reuse = split_reuse(master, {} if split == 'fit' else (inventory if inventory is not None else cache_inventory(study, model)))
    for name, value in (('master', master), ('execution', execution), ('reuse', reuse)):
        write_immutable(paths[name], value)
    return publish_execution(study, model, stage, split, paths, execution)


def publish_execution(study, model, stage, split, paths, execution, *, feature_root=None, manifest=None):
    from .selector_experiment_v5 import source_hashes
    from .worker import read_plan, verify_sources
    root, spec = Path(study['output_root']), study['models'][model]
    destination = root / model / stage
    tasks = []
    manifest = manifest or study['manifest']
    feature_root = feature_root or Path(study['old_study']).parent / model / f'{split}_features'
    for rank in range(study['world_size'] if execution['rows'] else 0):
        argv = previous.python(study, 'selector_experiment_v5') + ['--phase', 'physical',
            '--candidate', 'current_rank32', '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'],
            '--split', split, '--manifest', str(manifest), '--schedule', str(paths['execution']),
            '--adapter', spec['adapter'], '--model', spec['model_path'], '--model-revision', spec['revision'],
            '--output', str(destination), '--rank', str(rank), '--world-size', str(study['world_size']),
            '--query-control', 'correct', '--feature-root', str(feature_root)]
        if model == 'llava_next':
            argv.append('--reference-clip')
        task = previous.task(study, f'{stage}-{model}-{rank}', argv,
            [destination / f'rank{rank}.complete.json'], resume=destination / f'rank{rank}.metadata.json')
        task.update(id=f'compression-v8-{stage}-{model.replace("_", "-")}-{rank}',
                    tag='compression-v8', model_tag=model)
        tasks.append(task)
    pins = {**study['input_hashes'], str(root / 'study.json'): file_digest(root / 'study.json')}
    for relative, sha in source_hashes(spec['adapter']).items():
        pins[str(Path(study['cwd']) / 'learned_score' / relative)] = sha
    for path in (*paths.values(), manifest, spec['checkpoint']):
        pins[str(path)] = file_digest(path)
    for value in checked(paths['reuse'])['cells'].values():
        pins[value['path']], pins[value['metadata_path']] = value['sha256'], value['metadata_sha256']
    plan_path = root / f'plans/{stage}-{model}/plan.json'
    write_immutable(plan_path, seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': tasks,
        'input_hashes': pins, 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
        'deadline_utc_epoch': study['deadline_utc_epoch']}))
    verify_sources(read_plan(plan_path))
    return plan_path


def scan_execution(study, model, stage):
    """Audit complete physical stages, with arbitrary v8 shard counts."""
    from .adaptive_report_v4 import audit_cache, audit_stopping
    from .selector_experiment_v5 import source_hashes
    root = Path(study['output_root']) / model
    schedule = checked(root / f'schedules/{stage}-execution.json')
    if not schedule['rows']:
        return []
    directory, world = root / stage, study['world_size']
    if not all((directory / f'rank{r}.complete.json').is_file() for r in range(world)):
        return None
    ordered = sorted(schedule['rows'], key=lambda r: (schedule['rows'][r]['dataset'], r))
    found, contexts = {}, set()
    for rank in range(world):
        meta = read(directory / f'rank{rank}.metadata.json')
        marker = read(directory / f'rank{rank}.complete.json')
        assigned = ordered[rank::world]
        selector = meta.get('backbone', {}).get('selector_v5', {})
        if (meta.get('format') != 'mlsys_selector_experiment_v5' or meta.get('phase') != 'physical'
                or meta.get('split') != schedule['split'] or meta.get('limit') is not None
                or meta.get('world_size') != world or meta.get('manifest_sha256') != schedule['manifest_sha256']
                or meta.get('schedule_content_sha256') != schedule['sha256']
                or meta.get('schedule_sha256') != file_digest(root / f'schedules/{stage}-execution.json')
                or meta.get('checkpoint_sha256') != schedule['checkpoint_sha256']
                or meta.get('source_sha256') != source_hashes(study['models'][model]['adapter'])
                or meta.get('cache_contract') != schedule['cache_contract']
                or meta.get('max_new_tokens') != 128 or meta.get('generated_budget') != 512
                or selector.get('candidate') != 'current_rank32' or selector.get('query_control') != 'correct'
                or selector.get('coverage_fraction') != 0
                or selector.get('warmstart_sha256') != schedule['checkpoint_sha256']
                or marker.get('identity_sha256') != digest(meta)
                or marker.get('requests') != len(assigned)
                or marker.get('results') != sum(len(schedule['rows'][r]['cells']) for r in assigned)):
            raise ValueError('physical shard differs from frozen v8 matrix')
        contract_hash = digest(runtime_contract(meta))
        contexts.add(contract_hash)
        for request in assigned:
            planned = schedule['rows'][request]
            for cell in planned['cells']:
                path = directory / 'rows' / f"{request}.{cell['key']}.json"
                row = read(path)
                if (row.get('status') != 'ok' or row.get('identity_sha256') != digest(meta)
                        or row.get('schedule_cell') != cell or row.get('request_id') != request
                        or any(row.get(k) != planned[k] for k in ('group_id', 'dataset'))):
                    raise ValueError('physical row identity or schedule changed')
                indices = row.get('selected_patch_indices')
                if (not isinstance(indices, list) or len(indices) != cell['selected_count']
                        or indices != sorted(set(indices)) or indices[0] < 0
                        or indices[-1] >= planned['input_visual_tokens']
                        or digest(indices) != row.get('selected_patch_indices_sha256')):
                    raise ValueError('physical row lacks exact selected patch positions')
                audit_cache(row, schedule['cache_contract'], cell, planned)
                audit_stopping(row, meta)
                found[path.name] = {**row, 'model_tag': model, 'runtime_contract_sha256': contract_hash,
                    'physical_source_path': str(path), 'physical_source_sha256': file_digest(path)}
    if len(contexts) != 1:
        raise ValueError('mixed inference configurations across matrix shards')
    if {p.name for p in (directory / 'rows').glob('*.json')} != set(found):
        raise ValueError('unexpected or missing physical result files')
    return list(found.values())


def collect(study, model, stage):
    root = Path(study['output_root']) / model
    path = root / f'schedules/{stage}.json'
    if not path.exists():
        return None
    master, reuse = checked(path), checked(root / f'schedules/{stage}-reuse.json')
    if reuse['master_schedule_sha256'] != master['sha256']:
        raise ValueError('cache reuse belongs to another master schedule')
    physical = scan_execution(study, model, stage)
    if physical is None:
        return None
    index = {(r['request_id'], r['method']): r for r in physical}
    results = []
    for request, planned in master['rows'].items():
        for cell in planned['cells']:
            cached = reuse['cells'].get(f"{request}.{cell['key']}")
            if cached:
                if (file_digest(cached['path']) != cached['sha256']
                        or file_digest(cached['metadata_path']) != cached['metadata_sha256']):
                    raise ValueError('reused physical measurement changed')
                row = read(cached['path'])
                if (row['request_id'], row['selector'], row['selected_count']) != (request, cell['selector'], cell['selected_count']):
                    raise ValueError('reuse request/selector/count mismatch')
                row = {**row, 'method': cell['key'], 'schedule_cell': cell,
                    'model_tag': model, 'runtime_contract_sha256': cached['runtime_contract_sha256'],
                    'physical_source_path': cached['path'], 'physical_source_sha256': cached['sha256'],
                    'physical_measurement_performed': False, 'timing_reused_not_an_independent_observation': True}
            else:
                row = index[(request, cell['key'])]
            results.append(row)
    if len({r['runtime_contract_sha256'] for r in results}) != 1:
        raise ValueError('new and reused outputs have different inference contracts')
    return results


def status(study):
    root = Path(study['output_root'])
    result = {}
    for stage in STAGES:
        for model in MODELS:
            path = root / model / f'schedules/{stage}-execution.json'
            if not path.exists():
                result[f'{stage}-{model}'] = {'status': 'not_published'}
                continue
            schedule = checked(path)
            directory = root / model / stage
            completed = sum((directory / f'rank{r}.complete.json').exists() for r in range(study['world_size']))
            result[f'{stage}-{model}'] = {'status': 'complete' if not schedule['rows'] or completed == study['world_size'] else 'running_or_queued',
                'completed_shards': completed, 'shards': study['world_size'] if schedule['rows'] else 0,
                'new_physical_cells': sum(len(r['cells']) for r in schedule['rows'].values()),
                'cached_cells': len(checked(root / model / f'schedules/{stage}-reuse.json')['cells']),
                'written_rows': len(list((directory / 'rows').glob('*.json')))}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('prepare', 'publish', 'status'), required=True)
    parser.add_argument('--cwd', type=Path, default=Path.cwd())
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args(argv)
    study = prepare(args.cwd, hours=args.hours)
    if args.phase == 'publish':
        inventories = {m: cache_inventory(study, m) for m in MODELS}
        for stage in STAGES:
            for model in MODELS:
                publish(study, model, stage, inventories[model])
    print(json.dumps(status(study) if args.phase != 'prepare' else study, indent=2))


if __name__ == '__main__':
    main()
