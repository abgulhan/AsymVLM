"""Independent v9 controlled timing; never a quality acceptance certificate."""
import argparse
from collections import defaultdict
import copy
import json
import math
from pathlib import Path
import statistics
import time

from . import compression_matrix_v8 as matrix
from . import compression_report_v8 as reporting
from . import compression_systems_v9 as repeated
from . import adaptive_report_v4 as audit
from .adaptive_report_v1 import _atomic_text
from .followup_common import write_immutable
from .state import digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources

FORMAT = 'mlsys_compression_systems_run_v9'
GROUPS_PER_PANEL = 8
SAMPLE_SEED = 20261019
MAX_ATTEMPTS = 2
read, checked, seal = matrix.read, matrix.checked, matrix.seal

def expected_sources(adapter):
    return repeated.source_hashes(adapter)

def _path(root, path):
    return resolve_inside(path, root)

def verify(study):
    if study.get('sha256') != digest({k: v for k, v in study.items() if k != 'sha256'}):
        raise ValueError('study content hash mismatch')
    if not Path(study['output_root']).name.endswith('_v9'):
        raise ValueError('systems output must remain separate from frozen historical studies')
    for name, expected in study.get('input_hashes', {}).items():
        if file_digest(name) != expected:
            raise ValueError('frozen study input changed: ' + name)

def make_schedule(study, model, source, manifest):
    """Select deterministic intact groups without inspecting quality outcomes."""
    if (source.get('study_sha256') != study['sha256'] or source.get('model') != model
            or source.get('candidate') != 'current_rank32'):
        raise ValueError('systems source schedule identity mismatch')
    entries = {row['row_hash']: row for row in manifest['splits'][source['split']]}
    groups = defaultdict(lambda: defaultdict(list))
    for request, row in source['rows'].items():
        entry = entries.get(request)
        if entry is None or any(entry.get(key) != row[key] for key in ('dataset', 'group_id')):
            raise ValueError('systems schedule differs from identity manifest')
        groups[row['dataset']][row['group_id']].append(request)
    rows, selected_groups = {}, {}
    for dataset, candidates in sorted(groups.items()):
        if len(candidates) < GROUPS_PER_PANEL:
            raise ValueError('insufficient independent groups for deterministic systems subset')
        selected = sorted(candidates, key=lambda g: (digest([SAMPLE_SEED, model, dataset, g]), g))[:GROUPS_PER_PANEL]
        selected_groups[dataset] = selected
        for group in selected:
            requests = candidates[group]
            metrics = {entries[r].get('quality_metric', entries[r].get('metric')) for r in requests}
            if len(metrics) != 1:
                raise ValueError('systems group mixes quality metrics')
            if 'mme' in metrics and (len(requests) != 2 or len({entries[r].get('category') for r in requests}) != 1):
                raise ValueError('systems MME requires intact categorized pairs')
            for request in sorted(requests):
                row = copy.deepcopy(source['rows'][request])
                methods = [cell['key'] for cell in row['cells']]
                if 'full' not in methods or len(methods) < 2 or len(methods) != len(set(methods)):
                    raise ValueError('systems comparison requires unique methods including full reference')
                row.update(quality_metric=entries[request].get('quality_metric', entries[request].get('metric')),
                           category=entries[request].get('category'), image_sha256=entries[request].get('image_sha256'))
                for cell in row['cells']:
                    if cell.get('budget_policy') is not None:
                        cell.update(budget_quality_metric=row['quality_metric'], budget_category=row['category'])
                rows[request] = row
    value = {k: copy.deepcopy(v) for k, v in source.items() if k != 'sha256'}
    value.update(format=FORMAT + '_schedule', rows=rows, source_schedule_sha256=source['sha256'],
                 selected_groups=selected_groups, groups_per_panel=GROUPS_PER_PANEL, sample_seed=SAMPLE_SEED,
                 sampling_uses_quality_labels=False, methods_reselected=False, warmups=1,
                 repetitions=repeated.REPETITIONS, world_size=1, cache_reuse='none; method-specific physical timing',
                 evidence='development timing only; independent of quality certification')
    return seal(value)

def publish(study, model, schedule_path, manifest_path, feature_root, policy_path):
    verify(study)
    root = Path(study['output_root'])
    source, manifest = checked(schedule_path), read(manifest_path)
    if source['manifest_sha256'] != file_digest(manifest_path):
        raise ValueError('systems manifest file hash mismatch')
    schedule = make_schedule(study, model, source, manifest)
    output = root / model / 'schedules/systems.json'
    write_immutable(output, schedule)
    paths = [Path(__file__), root / 'study.json', Path(schedule_path), Path(manifest_path), Path(policy_path), output]
    pins = {**study['input_hashes'],
            **{str(Path(study['cwd']) / 'learned_score' / name): sha for name, sha in expected_sources(study['models'][model]['adapter']).items()},
            **{str(path.resolve()): file_digest(path) for path in paths},
            **{str(Path(p).resolve()): file_digest(p) for p in (audit.__file__, reporting.__file__)}}
    for request, row in schedule['rows'].items():
        path = Path(feature_root) / 'features' / (request + '.json')
        feature = read(path)
        if (feature.get('checkpoint_sha256') != schedule['checkpoint_sha256']
                or feature.get('request_id') != request
                or any(feature.get(key) != row[key] for key in ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens'))):
            raise ValueError('systems saved feature identity or geometry differs')
        pins[str(path.resolve())] = file_digest(path)
    descriptor = seal({'format': FORMAT + '_input', 'study_sha256': study['sha256'], 'model': model,
                      'manifest_path': str(Path(manifest_path).resolve()), 'feature_root': str(Path(feature_root).resolve()),
                      'policy_path': str(Path(policy_path).resolve()), 'input_hashes': pins,
                      'evidence': 'controlled development timing, not quality confirmation'})
    descriptor_path = root / model / 'systems/input.json'
    write_immutable(descriptor_path, descriptor)
    pins = {**pins, str(descriptor_path): file_digest(descriptor_path)}
    argv = matrix.previous.python(study, 'compression_systems_run_v9') + [
        '--phase', 'measure', '--study', str(root / 'study.json'), '--model-tag', model,
        '--adapter', study['models'][model]['adapter']]
    task = matrix.previous.task(study, 'systems-' + model, argv, [root / model / 'systems/complete.json'])
    task.update(id='compression-v9-systems-' + model.replace('_', '-'), tag='compression-v9', model_tag=model,
                minimum_start_seconds=900)
    plan = seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': [task], 'input_hashes': pins,
                 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
                 'deadline_utc_epoch': study['deadline_utc_epoch'], 'world_size': 1,
                 'systems_status': 'fresh_controlled_development_measurement_requested'})
    plan_path = root / f'plans/systems-{model}/plan.json'
    write_immutable(plan_path, plan)
    verify_sources(read_plan(plan_path))
    return plan_path

def measurement_argv(study, model, descriptor, directory):
    root, spec = Path(study['output_root']), study['models'][model]
    schedule = checked(root / model / 'schedules/systems.json')
    argv = ['--exclusive-gpu', '--phase', 'physical', '--candidate', 'current_rank32',
            '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'],
            '--split', schedule['split'], '--manifest', descriptor['manifest_path'],
            '--schedule', str(root / model / 'schedules/systems.json'), '--adapter', spec['adapter'],
            '--model', spec['model_path'], '--model-revision', spec['revision'], '--output', str(directory),
            '--rank', '0', '--world-size', '1', '--warmup', '0', '--query-control', 'correct',
            '--feature-root', descriptor['feature_root']]
    if model == 'llava_next':
        argv.append('--reference-clip')
    return argv

def _positive(value, label, *, allow_zero=False):
    if (isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
            or value < 0 or (value == 0 and not allow_zero)):
        raise ValueError('invalid systems metric: ' + label)
    return float(value)


def _close(value, expected, label):
    actual = _positive(value, label, allow_zero=True)
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError('systems aggregate differs from raw repetitions: ' + label)


def audit_repetitions(row):
    metrics = row.get('metrics', {})
    values = metrics.get('systems_repetition_metrics')
    if (metrics.get('same_gpu_repetitions') is not True or metrics.get('systems_repetitions') != 3
            or metrics.get('systems_warmups') != 1 or not isinstance(values, list) or len(values) != 3):
        raise ValueError('systems require one warmup and three fresh same-GPU measurements')
    snapshots = [metrics.get('systems_exclusive_gpu_' + position, {}) for position in ('before', 'after')]
    for snapshot in snapshots:
        owner, pids = snapshot.get('owner_pid'), snapshot.get('process_ids')
        if (type(owner) is not int or owner <= 0 or not isinstance(pids, list)
                or any(type(pid) is not int or pid != owner for pid in pids)
                or not isinstance(snapshot.get('gpu_uuid'), str) or not snapshot['gpu_uuid']):
            raise ValueError('missing exclusive GPU process evidence')
    identities = {(value['gpu_uuid'], value['owner_pid']) for value in snapshots}
    if len(identities) != 1:
        raise ValueError('systems cell changed GPU or process')
    for value in values:
        if (value.get('generated_token_ids') != row['generated_token_ids']
                or value.get('prediction') != row['prediction']
                or value.get('runtime', {}).get('cache') != metrics.get('cache')):
            raise ValueError('repeat token, answer, or cache parity failed')
        _positive(value['timing'].get('end_to_end_seconds'), 'end_to_end_seconds')
        _positive(value['runtime'].get('full_request_peak_allocated_bytes'), 'GPU peak', allow_zero=True)
    for field in repeated.TIMING_FIELDS:
        raw = [value['timing'].get(field) for value in values]
        if any(value is not None for value in raw):
            for value in raw:
                _positive(value, field, allow_zero=True)
            _close(row.get(field), statistics.mean(raw), field)
    for field in repeated.RUNTIME_TIMINGS:
        raw = [value['runtime'].get(field) for value in values]
        if any(value is not None for value in raw):
            for value in raw:
                _positive(value, field, allow_zero=True)
            _close(metrics.get(field), statistics.mean(raw), field)
    for field in repeated.POST_PREFILL_FIELDS:
        raw = [value['runtime'].get(field) for value in values]
        if any(value is not None for value in raw):
            for value in raw:
                _positive(value, field, allow_zero=True)
            _close(metrics.get(field), statistics.mean(raw), field)
    seconds = sum(value['timing']['end_to_end_seconds'] for value in values)
    _close(metrics.get('systems_generated_tokens_per_second'), 3 * len(row['generated_token_ids']) / seconds, 'token throughput')
    _close(metrics.get('systems_requests_per_second'), 3 / seconds, 'request throughput')
    _close(metrics.get('full_request_peak_allocated_bytes'),
           max(value['runtime']['full_request_peak_allocated_bytes'] for value in values), 'GPU peak')
    return next(iter(identities))


def scan_attempt(study, model, directory):
    """Audit a whole successful single-process attempt, including its fresh aliases."""
    root, directory = Path(study['output_root']), Path(directory)
    schedule_path = root / model / 'schedules/systems.json'
    schedule = checked(schedule_path)
    marker_path = directory / 'rank0.complete.json'
    if not marker_path.exists():
        return None
    meta_path = directory / 'rank0.metadata.json'
    meta, marker = read(meta_path), read(marker_path)
    selector = meta.get('backbone', {}).get('selector_v5', {})
    if (meta.get('format') != 'mlsys_selector_experiment_v5' or meta.get('phase') != 'physical'
            or meta.get('split') != schedule['split'] or meta.get('limit') is not None
            or meta.get('world_size') != 1 or meta.get('warmup') != 0
            or not isinstance(meta.get('hardware'), str) or not meta['hardware']
            or meta.get('manifest_sha256') != schedule['manifest_sha256']
            or meta.get('schedule_content_sha256') != schedule['sha256']
            or meta.get('schedule_sha256') != file_digest(schedule_path)
            or meta.get('checkpoint_sha256') != schedule['checkpoint_sha256']
            or meta.get('source_sha256') != expected_sources(study['models'][model]['adapter'])
            or meta.get('cache_contract') != schedule['cache_contract']
            or meta.get('max_new_tokens') != 128 or meta.get('generated_budget') != 512
            or selector.get('candidate') != 'current_rank32' or selector.get('query_control') != 'correct'
            or selector.get('coverage_fraction') != 0 or selector.get('warmstart_sha256') != schedule['checkpoint_sha256']
            or marker.get('identity_sha256') != digest(meta) or marker.get('requests') != len(schedule['rows'])
            or marker.get('results') != sum(len(row['cells']) for row in schedule['rows'].values())):
        raise ValueError('systems metadata differs from frozen single-process protocol')
    rows, paths, identities, hardware = [], {}, set(), set()
    for request, planned in sorted(schedule['rows'].items()):
        for cell in planned['cells']:
            path = directory / 'rows' / f"{request}.{cell['key']}.json"
            row = read(path)
            if (row.get('status') != 'ok' or row.get('identity_sha256') != digest(meta)
                    or row.get('schedule_cell') != cell or row.get('request_id') != request
                    or row.get('method') != cell['key'] or row.get('selector') != cell['selector']
                    or row.get('schedule_content_sha256') != schedule['sha256']
                    or any(row.get(key) != planned[key] for key in
                           ('group_id', 'dataset', 'quality_metric', 'category', 'image_sha256'))
                    or row.get('hardware') != meta.get('hardware') or not row.get('host')):
                raise ValueError('systems row identity differs from its frozen physical cell')
            indices = row.get('selected_patch_indices')
            if (not isinstance(indices, list) or len(indices) != cell['selected_count']
                    or any(type(index) is not int for index in indices) or indices != sorted(set(indices))
                    or not indices or indices[0] < 0 or indices[-1] >= planned['input_visual_tokens']
                    or digest(indices) != row.get('selected_patch_indices_sha256')):
                raise ValueError('systems row lacks exact selected-index evidence')
            audit.audit_cache(row, schedule['cache_contract'], cell, planned)
            audit.audit_stopping(row, meta)
            identities.add(audit_repetitions(row))
            hardware.add((row['host'], row['hardware'], row.get('job_id')))
            paths[path.name] = path
            rows.append(row)
    if len(identities) != 1 or len(hardware) != 1:
        raise ValueError('systems methods span different hardware or processes')
    if {path.name for path in (directory / 'rows').glob('*.json')} != set(paths):
        raise ValueError('unexpected or missing systems physical outputs')
    by_key = {(row['request_id'], row['method']): row for row in rows}
    for row in rows:
        if row.get('physical_measurement_performed') is not True:
            raise ValueError('v9 systems requires fresh measurements per named method, never timing aliases')
        if row['schedule_cell'].get('budget_policy') is not None and (
                row.get('budget_in_end_to_end_timing') is not True
                or row.get('budget_policy_sha256') != row['schedule_cell']['budget_policy']['sha256']
                or row.get('budget_selected_count') != row['selected_count']):
            raise ValueError('adaptive systems cell did not time and verify its live budget')
        if row.get('physical_measurement_performed') is True:
            if row.get('physical_reuse_of_cell') is not None or row.get('timing_reused_not_an_independent_observation') is not False:
                raise ValueError('fresh systems row has inconsistent reuse provenance')
        elif row.get('physical_measurement_performed') is False:
            source = by_key.get((row['request_id'], row.get('physical_reuse_of_cell')))
            if (source is None or source.get('physical_measurement_performed') is not True
                    or row.get('physical_source_row_sha256') != digest(source)
                    or row.get('timing_reused_not_an_independent_observation') is not True
                    or any(row.get(key) != source.get(key) for key in ('selector', 'selected_count', 'prediction',
                       'generated_token_ids', 'selected_patch_indices', 'quality_score', 'metrics', *repeated.TIMING_FIELDS))):
                raise ValueError('systems timing alias is not an identical fresh physical action')
        else:
            raise ValueError('systems physical provenance is missing')
    return [{**row, 'model_tag': model, 'runtime_contract_sha256': digest(matrix.runtime_contract(meta)),
             'physical_source_path': str(paths[f"{row['request_id']}.{row['method']}.json"]),
             'systems_fresh_measurement_stage': True} for row in rows]


def measure(study, model, adapter):
    """At most two fresh process attempts; never mix partial timing."""
    if model not in matrix.MODELS or adapter != study['models'][model]['adapter']:
        raise ValueError('systems task model and adapter disagree')
    verify(study)
    root = Path(study['output_root'])
    verify_sources(read_plan(root / f'plans/systems-{model}/plan.json'))
    descriptor = checked(root / model / 'systems/input.json')
    directory = root / model / 'systems'
    complete = directory / 'complete.json'
    if complete.exists():
        collect(study, model)
        return checked(complete)
    if time.time() >= study['deadline_utc_epoch']:
        raise RuntimeError('systems deadline expired')
    attempt = None
    for number in range(MAX_ATTEMPTS):
        candidate = directory / f'attempt{number}'
        try:
            candidate.mkdir()
        except FileExistsError:
            continue
        attempt = candidate
        break
    if attempt is None:
        raise RuntimeError('systems exhausted initial attempt and one fresh retry')
    repeated.main(measurement_argv(study, model, descriptor, attempt))
    rows = scan_attempt(study, model, attempt)
    if rows is None:
        raise ValueError('systems worker returned without complete measurements')
    files = [attempt / 'rank0.metadata.json', attempt / 'rank0.complete.json', *sorted((attempt / 'rows').glob('*.json'))]
    value = seal({'format': FORMAT + '_complete', 'study_sha256': study['sha256'], 'model': model,
                  'attempt_directory': str(attempt), 'input_hashes': {str(path): file_digest(path) for path in files},
                  'schedule_sha256': checked(root / model / 'schedules/systems.json')['sha256'],
                  'logical_rows': len(rows), 'fresh_physical_rows': sum(row['physical_measurement_performed'] for row in rows)})
    write_immutable(complete, value)
    return value

def collect(study, model):
    root = Path(study['output_root'])
    path = root / model / 'systems/complete.json'
    if not path.exists():
        return None
    value = checked(path)
    expected_parent = root / model / 'systems'
    directory = _path(expected_parent, value['attempt_directory'])
    if (value.get('format') != FORMAT + '_complete' or value.get('study_sha256') != study['sha256']
            or value.get('model') != model or directory.parent != expected_parent
            or directory.name not in {'attempt0', 'attempt1'}
            or value.get('schedule_sha256') != checked(root / model / 'schedules/systems.json')['sha256']):
        raise ValueError('systems completion pointer differs from its frozen stage')
    for path, sha in value['input_hashes'].items():
        if file_digest(_path(directory, path)) != sha:
            raise ValueError('completed systems output changed')
    rows = scan_attempt(study, model, directory)
    if (rows is None or len(rows) != value['logical_rows']
            or sum(row['physical_measurement_performed'] for row in rows) != value['fresh_physical_rows']):
        raise ValueError('systems completion counts disagree with physical rows')
    return rows


def summarize(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['model_tag'], row['dataset'], row['method'])].append(row)
    panels = []
    for (model, dataset, method), values in sorted(grouped.items()):
        full = grouped.get((model, dataset, 'full'), [])
        if {row['request_id'] for row in full} != {row['request_id'] for row in values}:
            raise ValueError('systems methods do not share the complete sampled cohort')
        raw = [entry for row in values for entry in row['metrics']['systems_repetition_metrics']]
        seconds = sum(entry['timing']['end_to_end_seconds'] for entry in raw)
        full_seconds = sum(entry['timing']['end_to_end_seconds'] for row in full for entry in row['metrics']['systems_repetition_metrics'])
        memory = reporting.memory_summary(values, full)
        panels.append({'model': model, 'dataset': dataset, 'method': method,
            'questions': len(values), 'groups': len({row['group_id'] for row in values}),
            'logical_repetitions': len(raw), 'fresh_physical_cells': sum(row['physical_measurement_performed'] for row in values),
            'same_process_alias_cells': sum(not row['physical_measurement_performed'] for row in values),
            'mean_end_to_end_seconds': seconds / len(raw),
            'mean_ttft_seconds': statistics.mean(entry['timing']['ttft_seconds'] for entry in raw),
            'requests_per_second': len(raw) / seconds,
            'generated_tokens_per_second': sum(len(entry['generated_token_ids']) for entry in raw) / seconds,
            'same_gpu_full_relative_latency_ratio': full_seconds / seconds,
            'maximum_peak_gpu_bytes': max(entry['runtime']['full_request_peak_allocated_bytes'] for entry in raw),
            'mean_post_prefill_allocated_bytes': statistics.mean(entry['runtime']['post_prefill_allocated_bytes'] for entry in raw)
                if all(isinstance(entry['runtime'].get('post_prefill_allocated_bytes'), (float, int)) for entry in raw) else None,
            **memory,
            'gpu_uuid': values[0]['metrics']['systems_exclusive_gpu_before']['gpu_uuid'],
            'owner_pid': values[0]['metrics']['systems_exclusive_gpu_before']['owner_pid'],
            'hardware': values[0]['hardware'], 'host': values[0]['host']})
    return {'format': FORMAT + '_report', 'status': 'complete', 'measured': True,
        'evidence': 'controlled_systems_on_deterministic_development_subset_not_quality_certification',
        'warmups_per_unique_action': 1, 'measured_repetitions_per_unique_action': 3,
        'groups_per_model_dataset': GROUPS_PER_PANEL, 'sample_seed': SAMPLE_SEED,
        'scope': 'sequential batch-one natural-stopping inference; live entropy and budget calculation included; label scoring and feature diagnostics excluded',
        'matched_control_note': 'Uniform control matches the source development cohort token total; its timing subset is not rebalanced or asserted equal-total.',
        'alias_note': 'Every named method has independent fresh repetitions; fixed/adaptive timing is never aliased.',
        'logical_rows': len(rows), 'fresh_physical_cells': sum(row['physical_measurement_performed'] for row in rows),
        'panels': panels, 'raw_rows': rows}


def render(report):
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    lines = ['# Controlled same-GPU systems measurements', '', report['scope'] + '.', '',
             report['alias_note'], '', 'Allocated KV capacity, measured live KV, and GPU peak are distinct; missing live measurements remain N/A.', '',
             '| Model | Dataset | Method | Inputs | Fresh / aliases | End-to-end (s) | TTFT (s) | Requests/s | Generated tokens/s | Full/candidate latency | Allocated KV (bytes) | Live KV (bytes) | Post-prefill GPU (bytes) | Max GPU peak (bytes) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for panel in report['panels']:
        values = [str(panel[key]).replace('|', '\\|') for key in ('model', 'dataset', 'method')]
        values.extend([str(panel['questions']), f"{panel['fresh_physical_cells']} / {panel['same_process_alias_cells']}"])
        values.extend(number(panel.get(key)) for key in ('mean_end_to_end_seconds', 'mean_ttft_seconds',
            'requests_per_second', 'generated_tokens_per_second', 'same_gpu_full_relative_latency_ratio',
            'mean_allocated_kv_bytes', 'mean_actual_kv_bytes', 'mean_post_prefill_allocated_bytes', 'maximum_peak_gpu_bytes'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def advance(study, model, schedule_path, manifest_path, feature_root, policy_path):
    """Independent per-model stage; no confirmation certificate is required."""
    verify(study)
    root = Path(study['output_root'])
    if not Path(schedule_path).exists():
        return {'status': 'awaiting_frozen_development_schedule', 'model': model, 'complete': False, 'measured': False}
    plan_path = root / f'plans/systems-{model}/plan.json'
    if time.time() >= study['deadline_utc_epoch'] and not plan_path.exists():
        return {'status': 'deadline_expired', 'model': model, 'complete': False, 'measured': False}
    plan = publish(study, model, schedule_path, manifest_path, feature_root, policy_path)
    rows = collect(study, model)
    if rows is None:
        return {'status': 'running_or_queued', 'model': model, 'complete': False, 'measured': False, 'plans': [str(plan)]}
    report = summarize(rows)
    path = root / 'reports' / f'systems-{model}.json'
    write_immutable(path, report)
    _atomic_text(root / 'reports' / f'SYSTEMS-{model}.md', render(report))
    return {'status': 'complete', 'model': model, 'complete': True, 'measured': True,
            'evidence': report['evidence'], 'report_path': str(path), 'report_sha256': file_digest(path),
            'plans': [str(plan)], 'logical_rows': len(rows), 'fresh_physical_cells': report['fresh_physical_cells']}

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure',), required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-tag', choices=matrix.MODELS, required=True)
    parser.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(measure(checked(args.study), args.model_tag, args.adapter), sort_keys=True))

if __name__ == '__main__':
    main()
