"""Fresh, controlled same-GPU systems evidence after adaptive confirmation.

This stage never selects another policy, reuses historical timing, or promotes
its small deterministic timing cohort into a quality certification. All frozen
methods run in one process per model. A retry starts an isolated fresh attempt;
no result from a failed process is merged into a successful measurement.
"""

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
from . import compression_systems_v8 as repeated
from . import adaptive_report_v4 as audit
from .adaptive_report_v1 import _atomic_text
from .followup_common import write_immutable
from .state import digest, file_digest, resolve_inside
from .worker import read_plan, verify_sources


FORMAT = 'mlsys_compression_systems_run_v8'
GROUPS_PER_PANEL = 8
SAMPLE_SEED = 20261018
MAX_ATTEMPTS = 2
read, checked, seal = matrix.read, matrix.checked, matrix.seal


def expected_sources(adapter):
    return {**repeated.physical.source_hashes(adapter),
            'mlsys_sprint/compression_systems_v8.py': file_digest(repeated.__file__)}


def source_pins(study, model):
    root = Path(study['cwd'])
    return {**{str(root / 'learned_score' / path): sha for path, sha in
               expected_sources(study['models'][model]['adapter']).items()},
            **{str(Path(module).resolve()): file_digest(module) for module in
               (__file__, audit.__file__, reporting.__file__)}}


def _path(root, path):
    return resolve_inside(path, root)


def completed_confirmation(study, confirmation_status):
    """Reopen and verify the on-disk certificate; a passed flag alone is not proof."""
    if confirmation_status.get('status') != 'complete':
        return None
    root = Path(study['output_root'])
    result_path = root / 'confirmation/result.json'
    result = checked(result_path)
    if (result.get('format') != 'mlsys_compression_confirmation_run_v8_result'
            or result.get('status') != 'complete' or result.get('study_sha256') != study['sha256']
            or _path(root, result.get('result_path', str(result_path))) != result_path
            or confirmation_status.get('sha256', result['sha256']) != result['sha256']):
        raise ValueError('systems require the frozen complete confirmation result')
    if not all(result.get('confirmation_audit', {}).get(key) is True for key in reporting.CONFIRMATION_CHECKS):
        raise ValueError('systems require all independent fresh-confirmation checks')
    selection = checked(root / 'selection.json')
    gate_path = _path(root, result['gate_path'])
    gate = checked(gate_path)
    if (result['selection_sha256'] != selection['sha256'] or result['gate_sha256'] != gate['sha256']
            or gate.get('study_sha256') != study['sha256']
            or gate.get('selection_sha256') != selection['sha256']):
        raise ValueError('confirmation result has a different policy freeze or admission gate')
    for path, sha in gate.get('input_hashes', {}).items():
        if file_digest(path) != sha:
            raise ValueError('confirmation gate input changed')
    manifest_path = _path(root, result['manifest_path'])
    report_path = _path(root, result['report_path'])
    if file_digest(manifest_path) != result['manifest_sha256'] or file_digest(report_path) != result['report_sha256']:
        raise ValueError('confirmation manifest or report changed')
    report = read(report_path)
    if report.get('format') != reporting.FORMAT or report.get('evidence') != 'confirmation':
        raise ValueError('systems cannot use a development quality report')
    for model, artifact in result['schedules'].items():
        if model not in matrix.MODELS:
            raise ValueError('unknown confirmation model')
        path = _path(root, artifact['path'])
        if file_digest(path) != artifact['sha256'] or checked(path)['sha256'] != artifact['content_sha256']:
            raise ValueError('frozen confirmation schedule changed')
    return result, report


def eligible_panels(report):
    """Only the confirmed compressed primary triggers measurements, never full/baseline."""
    result = []
    seen = set()
    for panel in report.get('panels', []):
        if panel.get('method') != 'adaptive':
            continue
        key = panel.get('model'), panel.get('dataset')
        if key in seen:
            raise ValueError('duplicate adaptive confirmation panel')
        seen.add(key)
        memory = panel.get('memory', {})
        if (panel.get('accepted') is True and panel.get('confirmed') is True
                and panel.get('confirmation_audit_passed') is True and panel.get('cohort_complete') is True
                and panel.get('evidence') == 'confirmation'
                and memory.get('aggregate_patch_reduction_percent', 0) > 0
                and memory.get('fraction_compressed', 0) > 0):
            if key[0] not in matrix.MODELS or not isinstance(key[1], str) or not key[1]:
                raise ValueError('invalid accepted model/dataset panel')
            result.append(key)
    return sorted(result)


def make_schedule(study, model, confirmation_schedule, manifest, accepted_datasets, result):
    """Pick eight independent groups without reading outcomes; retain every frozen cell."""
    if not accepted_datasets or len(set(accepted_datasets)) != len(accepted_datasets):
        raise ValueError('declare nonempty unique accepted datasets')
    source = confirmation_schedule
    if (source.get('study_sha256') != study['sha256'] or source.get('model') != model
            or source.get('candidate') != 'current_rank32' or source.get('manifest_sha256') != result['manifest_sha256']):
        raise ValueError('confirmation schedule identity mismatch')
    entries = {row['row_hash']: row for row in manifest['splits'][source['split']]}
    rows, groups_by_dataset = {}, {}
    expected_methods = {'full', 'adaptive', 'matched_uniform', 'static_scorer', 'static_divprune'}
    if model == 'llava_next':
        expected_methods.add('static_cdpruner')
    for dataset in sorted(accepted_datasets):
        groups = defaultdict(list)
        for request, row in sorted(source['rows'].items()):
            if row['dataset'] != dataset:
                continue
            entry = entries.get(request)
            if entry is None or any(entry.get(key) != row[key] for key in ('dataset', 'group_id')):
                raise ValueError('confirmation schedule differs from identity manifest')
            groups[row['group_id']].append(request)
        if len(groups) < GROUPS_PER_PANEL:
            raise ValueError('insufficient independent groups for fixed systems cohort')
        selected = sorted(groups, key=lambda group: (digest([SAMPLE_SEED, model, dataset, group]), group))[:GROUPS_PER_PANEL]
        groups_by_dataset[dataset] = selected
        for group in selected:
            requests = groups[group]
            metrics = {entries[request].get('quality_metric', entries[request].get('metric')) for request in requests}
            if len(metrics) != 1:
                raise ValueError('one systems group cannot mix quality metrics')
            if 'mme' in metrics and (len(requests) != 2 or len({entries[r].get('category') for r in requests}) != 1
                                     or not entries[requests[0]].get('category')):
                raise ValueError('systems MME sampling must preserve intact categorized image pairs')
            for request in sorted(requests):
                row = source['rows'][request]
                methods = [cell['key'] for cell in row['cells']]
                if len(methods) != len(set(methods)) or set(methods) != expected_methods:
                    raise ValueError('all frozen comparison methods must share the systems cohort')
                rows[request] = {**copy.deepcopy(row),
                    'quality_metric': entries[request].get('quality_metric', entries[request].get('metric')),
                    'category': entries[request].get('category'), 'image_sha256': entries[request].get('image_sha256')}
    return seal({'format': FORMAT + '_schedule', 'study_sha256': study['sha256'],
        'model': model, 'split': source['split'], 'candidate': 'current_rank32',
        'manifest_sha256': source['manifest_sha256'], 'checkpoint_sha256': source['checkpoint_sha256'],
        'cache_contract': source['cache_contract'], 'rows': rows,
        'confirmation_schedule_sha256': source['sha256'], 'confirmation_result_sha256': result['sha256'],
        'selection_sha256': result['selection_sha256'], 'groups_per_panel': GROUPS_PER_PANEL,
        'sample_seed': SAMPLE_SEED, 'selected_groups': groups_by_dataset,
        'sampling_uses_quality_labels': False, 'methods_reselected': False,
        'warmups': 1, 'repetitions': repeated.REPETITIONS, 'world_size': 1,
        'cache_reuse': 'none from previous experiments; identical selector/count aliases within this fresh process only'})


def _base_pins(study, model, result):
    root = Path(study['output_root'])
    paths = [root / 'study.json', root / 'selection.json', root / 'confirmation/result.json',
             result['gate_path'], result['manifest_path'], result['report_path']]
    paths.extend(artifact['path'] for artifact in result['schedules'].values())
    return {**study['input_hashes'], **source_pins(study, model),
            **{str(path): file_digest(path) for path in paths}}


def publish(study, model, result, schedule=None):
    """Publish one fresh-process task, or an explicit no-accepted-primary empty queue."""
    root = Path(study['output_root'])
    plan_path = root / f'plans/systems-{model}/plan.json'
    pins = _base_pins(study, model, result)
    tasks = []
    if schedule is not None:
        schedule_path = root / model / 'schedules/systems.json'
        write_immutable(schedule_path, schedule)
        pins[str(schedule_path)] = file_digest(schedule_path)
        pins[study['models'][model]['checkpoint']] = file_digest(study['models'][model]['checkpoint'])
        feature_root = root / model / 'confirmation-features'
        for request, planned in schedule['rows'].items():
            path = feature_root / 'features' / (request + '.json')
            feature = read(path)
            if (feature.get('request_id') != request or feature.get('checkpoint_sha256') != schedule['checkpoint_sha256']
                    or any(feature.get(key) != planned[key] for key in
                           ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens'))):
                raise ValueError('systems input feature identity differs from confirmation')
            pins[str(path)] = file_digest(path)
        argv = matrix.previous.python(study, 'compression_systems_run_v8') + [
            '--phase', 'measure', '--study', str(root / 'study.json'), '--model-tag', model,
            '--adapter', study['models'][model]['adapter']]
        task = matrix.previous.task(study, 'systems-' + model, argv, [root / model / 'systems/complete.json'])
        task.update(id='compression-v8-systems-' + model.replace('_', '-'), tag='compression-v8', model_tag=model)
        tasks.append(task)
    plan = seal({'format': FORMAT + '_plan', 'cwd': study['cwd'], 'tasks': tasks, 'input_hashes': pins,
                 'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
                 'deadline_utc_epoch': study['deadline_utc_epoch'],
                 'systems_status': 'fresh_measurement_requested' if tasks else 'no_accepted_compressed_primary',
                 'confirmation_result_sha256': result['sha256'], 'world_size': 1})
    write_immutable(plan_path, plan)
    verify_sources(read_plan(plan_path))
    return plan_path


def measurement_argv(study, model, result, directory):
    root, spec = Path(study['output_root']), study['models'][model]
    schedule = checked(root / model / 'schedules/systems.json')
    argv = ['--exclusive-gpu', '--phase', 'physical', '--candidate', 'current_rank32',
            '--warmstart', spec['checkpoint'], '--checkpoint', spec['checkpoint'],
            '--split', schedule['split'], '--manifest', result['manifest_path'],
            '--schedule', str(root / model / 'schedules/systems.json'), '--adapter', spec['adapter'],
            '--model', spec['model_path'], '--model-revision', spec['revision'], '--output', str(directory),
            '--rank', '0', '--world-size', '1', '--warmup', '0', '--query-control', 'correct',
            '--feature-root', str(root / model / 'confirmation-features')]
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
    """Worker entrypoint: at most two fresh attempts, never resume partial timing."""
    if model not in matrix.MODELS or adapter != study['models'][model]['adapter']:
        raise ValueError('systems task model and adapter disagree')
    matrix.verify(study)
    root = Path(study['output_root'])
    verify_sources(read_plan(root / f'plans/systems-{model}/plan.json'))
    result, _ = completed_confirmation(study, {'status': 'complete'})
    directory = root / model / 'systems'
    complete = directory / 'complete.json'
    if complete.exists():
        collect(study, model)
        return checked(complete)
    if time.time() >= study['deadline_utc_epoch']:
        raise RuntimeError('systems execution window expired before a fresh attempt')
    directory.mkdir(parents=True, exist_ok=True)
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
        raise RuntimeError('systems exhausted the initial attempt and one fresh retry')
    repeated.main(measurement_argv(study, model, result, attempt))
    rows = scan_attempt(study, model, attempt)
    if rows is None:
        raise ValueError('systems worker returned without a complete audited measurement')
    files = [attempt / 'rank0.metadata.json', attempt / 'rank0.complete.json',
             *sorted((attempt / 'rows').glob('*.json'))]
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
            **memory,
            'gpu_uuid': values[0]['metrics']['systems_exclusive_gpu_before']['gpu_uuid'],
            'owner_pid': values[0]['metrics']['systems_exclusive_gpu_before']['owner_pid'],
            'hardware': values[0]['hardware'], 'host': values[0]['host']})
    return {'format': FORMAT + '_report', 'status': 'complete', 'measured': True,
        'evidence': 'controlled_systems_on_deterministic_confirmation_subset_not_quality_certification',
        'warmups_per_unique_action': 1, 'measured_repetitions_per_unique_action': 3,
        'groups_per_model_dataset': GROUPS_PER_PANEL, 'sample_seed': SAMPLE_SEED,
        'scope': 'sequential batch-one natural-stopping inference; feature extraction and query-order diagnostics outside timing',
        'matched_control_note': 'Uniform control matches the full confirmation cohort token total; its timing subset is not rebalanced or asserted equal-total.',
        'alias_note': 'Identical selector/count actions share fresh same-process repetitions, not independent timing observations.',
        'logical_rows': len(rows), 'fresh_physical_cells': sum(row['physical_measurement_performed'] for row in rows),
        'panels': panels, 'raw_rows': rows}


def render(report):
    def number(value):
        return 'N/A' if value is None else f'{value:.4f}'
    lines = ['# Controlled same-GPU systems measurements', '', report['scope'] + '.', '',
             report['alias_note'], '', 'Allocated KV capacity, measured live KV, and GPU peak are distinct; missing live measurements remain N/A.', '',
             '| Model | Dataset | Method | Inputs | Fresh / aliases | End-to-end (s) | TTFT (s) | Requests/s | Generated tokens/s | Full/candidate latency | Allocated KV (bytes) | Live KV (bytes) | Max GPU peak (bytes) |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for panel in report['panels']:
        values = [str(panel[key]).replace('|', '\\|') for key in ('model', 'dataset', 'method')]
        values.extend([str(panel['questions']), f"{panel['fresh_physical_cells']} / {panel['same_process_alias_cells']}"])
        values.extend(number(panel.get(key)) for key in ('mean_end_to_end_seconds', 'mean_ttft_seconds',
            'requests_per_second', 'generated_tokens_per_second', 'same_gpu_full_relative_latency_ratio',
            'mean_allocated_kv_bytes', 'mean_actual_kv_bytes', 'maximum_peak_gpu_bytes'))
        lines.append('| ' + ' | '.join(values) + ' |')
    return '\n'.join(lines) + '\n'


def advance(study, confirmation_status):
    """Publish/collect only after final fresh confirmation; never synthesize timing."""
    matrix.verify(study)
    validated = completed_confirmation(study, confirmation_status)
    if validated is None:
        return {'status': 'blocked_by_confirmation' if 'blocked' in confirmation_status.get('status', '') else 'awaiting_confirmation',
                'complete': False, 'measured': False, 'evidence': 'none', 'plans': [],
                'reason': 'fresh accepted compressed adaptive confirmation is required'}
    result, report = validated
    eligible = eligible_panels(report)
    root = Path(study['output_root'])
    if (time.time() >= study['deadline_utc_epoch'] and
            any(not (root / f'plans/systems-{model}/plan.json').exists() for model, _ in eligible)):
        return {'status': 'deadline_expired', 'complete': False, 'measured': False, 'evidence': 'none',
                'eligible_panels': [list(value) for value in eligible], 'plans': [],
                'reason': 'new systems measurement plans are not published after the study deadline'}
    manifest = read(result['manifest_path'])
    plans, measured = [], {}
    for model in matrix.MODELS:
        datasets = [dataset for name, dataset in eligible if name == model]
        schedule = None
        if datasets:
            artifact = result['schedules'].get(model)
            if artifact is None:
                raise ValueError('accepted primary has no frozen confirmation schedule')
            schedule = make_schedule(study, model, checked(artifact['path']), manifest, datasets, result)
        plans.append(str(publish(study, model, result, schedule)))
        if schedule is not None:
            measured[model] = collect(study, model)
    base = {'eligible_panels': [list(value) for value in eligible], 'plans': plans,
            'confirmation_result_sha256': result['sha256'], 'complete': False, 'measured': False}
    if not eligible:
        return {**base, 'status': 'no_accepted_compressed_primary', 'complete': True, 'evidence': 'none',
                'reason': 'scheduling finished without systems measurements; no primary compression method passed fresh confirmation'}
    if any(rows is None for rows in measured.values()):
        return {**base, 'status': 'running_or_queued', 'evidence': 'none',
                'completed_models': sorted(model for model, rows in measured.items() if rows is not None)}
    report = summarize([row for model in sorted(measured) for row in measured[model]])
    report.update(confirmation_result_sha256=result['sha256'], selection_sha256=result['selection_sha256'])
    path = root / 'reports/systems.json'
    write_immutable(path, report)
    _atomic_text(root / 'reports/SYSTEMS.md', render(report))
    return {**base, 'status': 'complete', 'complete': True, 'measured': True,
            'evidence': report['evidence'], 'report_path': str(path), 'report_sha256': file_digest(path),
            'fresh_physical_cells': report['fresh_physical_cells'], 'logical_rows': report['logical_rows']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure', 'advance'), required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-tag', choices=matrix.MODELS)
    parser.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'))
    args = parser.parse_args(argv)
    study = checked(args.study)
    if args.phase == 'measure':
        output = measure(study, args.model_tag, args.adapter)
    else:
        path = Path(study['output_root']) / 'confirmation/result.json'
        output = advance(study, checked(path) if path.exists() else {'status': 'pending'})
    print(json.dumps(output, sort_keys=True))


if __name__ == '__main__':
    main()
