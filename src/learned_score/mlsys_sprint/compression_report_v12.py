"""Two honest v12 tables: development budgets and complete physical controls.

Previously inspected outcomes remain development evidence. Incomplete paired
cohorts are never reduced to a convenient intersection, and an unmeasured
integer count never acquires an interpolated quality value. All cached outcome
files are revalidated through their recorded source hashes before reuse.
"""

import argparse
from collections import defaultdict
import copy
import json
from pathlib import Path
import time

import numpy as np

from . import compression_policy_v12 as policy
from . import compression_report_v11 as previous_report
from . import compression_reuse_v11 as reuse
from .adaptive_report_v1 import _atomic_text
from .adaptive_report_v3 import paired_ci, quality_summary
from .compression_matrix_v8 import checked, seal
from .compression_policy_v10 import canonical_dataset
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_report_v12'
METHODS = ('scorer_uniform', 'scorer_adaptive', 'divprune_uniform', 'divprune_adaptive')
LABELS = {'full': 'All image tokens', 'static': 'AsymVLM static',
    'missing_mass': 'AsymVLM missing importance', 'rank32_coverage': 'AsymVLM current coverage',
    'native_coverage': 'AsymVLM richer-feature coverage', 'scorer_uniform': 'AsymVLM static, matched',
    'scorer_adaptive': 'AsymVLM selected budget', 'divprune_uniform': 'DivPrune static, matched',
    'divprune_adaptive': 'DivPrune + same adaptive counts'}


def _seal_checked(value, label):
    if value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError(label + ' content changed')
    return value


def paired_intervals(rows, counts, reference_counts=None, *, repetitions=10000, seed=20261022):
    """Native quality differences; image groups and MME categories stay intact.

    Macro intervals use equal benchmark weight, independent resampling within
    each benchmark. If an image group crosses benchmarks the macro interval is
    withheld rather than silently assuming independent observations.
    """
    if type(repetitions) is not int or repetitions < 100:
        raise ValueError('at least 100 bootstrap repetitions required')
    if set(counts) != {row['row_hash'] for row in rows}:
        raise ValueError('confidence interval cohort mismatch')
    reference_counts = reference_counts or {r['row_hash']: r['input_visual_tokens'] for r in rows}
    if set(reference_counts) != set(counts):
        raise ValueError('paired reference cohort mismatch')
    views, references, ownership = defaultdict(list), defaultdict(list), defaultdict(set)
    for row in sorted(rows, key=lambda r: r['row_hash']):
        base = {'request_id': row['row_hash'], 'group_id': row['group_id'], 'category': row.get('category')}
        views[row['dataset']].append({**base, 'quality_score': row['quality_by_count'][str(counts[row['row_hash']])]})
        references[row['dataset']].append({**base, 'quality_score': row['quality_by_count'][str(reference_counts[row['row_hash']])]})
        ownership[row['group_id']].add(row['dataset'])
    domains, samples = {}, []
    for dataset in sorted(views):
        metric = policy.METRICS[dataset]
        local_seed = int(digest([seed, dataset])[:8], 16)
        a, b = views[dataset], references[dataset]
        interval = paired_ci(a, b, metric, repetitions=repetitions, seed=local_seed)
        interval['interpretation'] = 'exploratory paired development interval; zero width is not equivalence'
        domains[dataset] = interval
        if interval['ci_lower_pp'] is None:
            continue
        groups = defaultdict(list)
        baseline = {r['request_id']: r for r in b}
        for row in a:
            groups[row['group_id']].append(row)
        rng, draws = np.random.default_rng(local_seed), np.zeros(repetitions)
        if dataset == 'MME':
            categories = defaultdict(list)
            for _, members in sorted(groups.items()):
                if len(members) != 2 or len({r['category'] for r in members}) != 1:
                    raise ValueError('MME intervals require intact category/image pairs')
                actual = [r['quality_score'] for r in members]
                full = [baseline[r['request_id']]['quality_score'] for r in members]
                delta = .25 * (sum(actual) - sum(full)) + .5 * (float(all(actual)) - float(all(full)))
                categories[members[0]['category']].append(delta)
            for category in sorted(categories):
                values = np.asarray(categories[category])
                for start in range(0, repetitions, 256):
                    end = min(start+256, repetitions)
                    indices = rng.integers(0, len(values), size=(end-start, len(values)))
                    draws[start:end] += values[indices].mean(axis=1) / len(categories)
        else:
            sums = np.asarray([sum(r['quality_score'] - baseline[r['request_id']]['quality_score'] for r in members)
                               for _, members in sorted(groups.items())])
            sizes = np.asarray([len(members) for _, members in sorted(groups.items())])
            for start in range(0, repetitions, 256):
                end = min(start+256, repetitions)
                indices = rng.integers(0, len(sums), size=(end-start, len(sums)))
                draws[start:end] = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)
        samples.append(draws * 100)
    macro = {'delta_pp': sum(v['delta_pp'] for v in domains.values()) / len(domains),
        'ci_lower_pp': None, 'ci_upper_pp': None, 'confidence': .95,
        'bootstrap_repetitions': repetitions, 'bootstrap_seed': seed,
        'status': 'insufficient_independent_groups',
        'interpretation': 'conditional on fitted out-of-fold predictions, not a refitted-policy interval'}
    if any(len(datasets) > 1 for datasets in ownership.values()):
        macro['status'] = 'shared_groups_across_benchmarks_macro_interval_withheld'
    elif len(samples) == len(domains):
        low, high = np.quantile(np.mean(samples, axis=0), [.025, .975])
        macro.update(ci_lower_pp=float(low), ci_upper_pp=float(high), status='computed',
                     resampling_unit='image_groups_within_benchmark_and_MME_category')
    return {'domains': domains, 'macro': macro}


def development_table(rows, report, *, repetitions=10000):
    """Quality is physical; allocated capacity is exact slot accounting, not GPU peak."""
    rows = sorted(rows, key=lambda r: r['row_hash'])
    report = report.get('result', report)
    config = report['configuration']
    full = {r['row_hash']: r['input_visual_tokens'] for r in rows}
    procedures = {'full': {'counts': full}, **report['procedures']}
    domain_rows, macro_rows = [], []
    for kind, value in procedures.items():
        counts = value['counts']
        result = policy.summary(rows, counts, bytes_per_token=config['bytes_per_token'],
                                generation_reserve=config['generation_reserve'])
        if kind != 'full' and digest(result) != digest(value['summary']):
            raise ValueError('development summary differs from its physical count assignments')
        intervals = paired_intervals(rows, counts, repetitions=repetitions)
        base = {'model': config['model'], 'procedure': kind, 'label': LABELS[kind],
            'quality_source': 'physical_cached_outcomes_at_exact_selected_counts',
            'kv_source': 'exact_allocated_capacity_from_physical_prefix_count_and_127_output_reserve',
            'same_budget_as_other_procedures': False, 'independent_confirmation': False,
            'allocation': value.get('allocation'), 'thresholds_fitted_without_this_image_group': kind != 'full'}
        macro_rows.append({**base, 'dataset': 'Four-benchmark mean',
            'score_percent': 100 * result['macro_accuracy'],
            'mean_allocated_kv_mib': result['mean_allocated_kv_mib'],
            'kv_reduction_percent': result['kv_reduction_percent'],
            'percent_requests_with_memory_reduction': result['percent_requests_with_memory_reduction'],
            'versus_all_tokens': intervals['macro'], 'quality_constraints_satisfied': policy.meets_constraints(result,
                config['max_mean_loss_pp'], config['max_domain_loss_pp']), 'summary': result})
        for dataset, values in result['domains'].items():
            domain_rows.append({**base, 'dataset': dataset, 'score_percent': 100 * values['accuracy'],
                'mean_allocated_kv_mib': values['mean_allocated_kv_mib'],
                'kv_reduction_percent': values['kv_reduction_percent'],
                'percent_requests_with_memory_reduction': values['percent_requests_with_memory_reduction'],
                'versus_all_tokens': intervals['domains'][dataset], 'summary': values})
    return {'macro_rows': macro_rows, 'dataset_rows': domain_rows,
            'not_exact_budget_matched': True, 'independent_confirmation': False}


def cached_development_table(study, model, rows, report, *, repetitions=10000, write=True):
    """Immutable bootstrap result; repeat watch ticks only validate its inputs."""
    identity = {'study_sha256': study['sha256'], 'model': model, 'report_sha256': report['sha256'],
                'rows_sha256': digest(rows), 'bootstrap_repetitions': repetitions,
                'report_source_sha256': file_digest(__file__)}
    path = Path(study['output_root']) / 'reports' / ('development-table-' + model + '.json')
    if path.exists():
        cached = checked(path)
        if cached.get('identity') != identity:
            raise ValueError('cached development table inputs or bootstrap configuration changed')
        return cached['table']
    table = development_table(rows, report, repetitions=repetitions)
    if write:
        write_immutable(path, seal({'format': FORMAT + '_development_table', 'identity': identity, 'table': table}))
    return table


def materialize(master, receipt, fresh_rows, inventory):
    """Strict master/reuse/alias join; no convenient partial intersections."""
    reuse._checked_schedule(master)
    _seal_checked(receipt, 'reuse receipt')
    if receipt['original_schedule_sha256'] != master['sha256']:
        raise ValueError('receipt belongs to a different master schedule')
    for path, sha in receipt['source_pins'].items():
        if file_digest(path) != sha:
            raise ValueError('cached physical source changed')
    cells = {request + '.' + cell['key']: (request, geometry, cell)
             for request, geometry in master['rows'].items() for cell in geometry['cells']}
    if set(cells) != set(receipt['cells']):
        raise ValueError('master/reuse cohort or cells mismatch')
    fresh = {}
    for row in fresh_rows:
        key = row['request_id'] + '.' + row['method']
        if key in fresh:
            raise ValueError('duplicate fresh physical cell')
        fresh[key] = row
    expected = {key for key, r in receipt['cells'].items() if r['kind'] == 'execute'}
    if set(fresh) != expected:
        raise ValueError('incomplete or extra fresh physical cells; no interpolation allowed')
    if receipt['counts']['missing_unique_cells'] != len(expected):
        raise ValueError('receipt missing-cell counter mismatch')
    resolved = {}

    def resolve(address, visited=()):
        if address in resolved:
            return resolved[address]
        if address in visited or address not in cells:
            raise ValueError('cyclic or missing physical alias')
        request, geometry, cell = cells[address]
        record = receipt['cells'][address]
        if any(record[k] != value for k, value in {'request_id': request, 'cell_key': cell['key'],
                'selector': cell['selector'], 'selected_count': cell['selected_count']}.items()):
            raise ValueError('receipt changed exact cell identity')
        if record['kind'] == 'alias':
            source = resolve(record['alias_of'], (*visited, address))
        elif record['kind'] == 'execute':
            source = fresh[address]
        elif record['kind'] == 'cached_physical':
            source = inventory[(request, cell['selector'], cell['selected_count'])]
            if (str(Path(source['physical_source_path']).resolve()) != str(Path(record['physical_source_path']).resolve())
                    or source['physical_source_sha256'] != record['physical_source_sha256']):
                raise ValueError('inventory and receipt point to different original measurements')
        else:
            raise ValueError('unknown physical receipt kind')
        reuse._check_geometry(master, request, geometry, cell, source)
        actual = reuse._index_hash(source, geometry['input_visual_tokens'], cell['selected_count'])
        if record.get('selected_patch_indices_sha256', actual) != actual:
            raise ValueError('receipt selected patches changed')
        if not source.get('physical_source_path') or not source.get('physical_source_sha256'):
            raise ValueError('physical source evidence is required')
        previous_report._verify_physical_provenance({**source, 'physical_measurement_performed': False})
        result = {**copy.deepcopy(source), 'method': cell['key'], 'schedule_cell': copy.deepcopy(cell),
            'physical_measurement_performed': False,
            'timing_reused_not_an_independent_observation': True,
            'receipt_kind': record['kind'], 'dataset': canonical_dataset(source['dataset'])}
        resolved[address] = result
        return result

    return [resolve(key) for key in sorted(cells)]


def collect_matched(study, model, *, repetitions=10000):
    """Return None until both complete declared splits can be audited in full."""
    from .compression_study_v12 import verify
    from .compression_study_v11 import physical_inventory
    from .compression_matrix_v8 import scan_execution
    from .worker import read_plan, verify_sources
    source = verify(study)
    root, bundles = Path(study['output_root']), []
    cached_path = root / 'reports' / (model + '-matched.json')
    if cached_path.exists():
        cached = checked(cached_path)
        if (cached.get('study_sha256') != study['sha256'] or cached.get('model') != model
                or cached.get('report_source_sha256') != file_digest(__file__)):
            raise ValueError('cached physical report belongs to a changed study or reporter')
        if cached.get('bootstrap_repetitions') != repetitions:
            raise ValueError('cannot silently replace a published confidence interval configuration')
        for path, sha in cached['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('published physical report input changed: ' + path)
        return cached
    factorial_path = root / model / 'development/factorial.json'
    if not factorial_path.exists():
        return None
    pins = {str(factorial_path): file_digest(factorial_path)}
    for split in ('fit', 'screening'):
        stage = split + '-matched'
        master_path = root / model / 'schedules' / (stage + '-master.json')
        receipt_path = root / model / 'schedules' / (stage + '-reuse.json')
        if not master_path.exists() or not receipt_path.exists():
            return None
        master, receipt = checked(master_path), checked(receipt_path)
        pins.update({str(master_path): file_digest(master_path), str(receipt_path): file_digest(receipt_path)})
        if master['study_sha256'] != study['sha256'] or master['model'] != model or master['split'] != split:
            raise ValueError('physical master belongs to another v12 study/model/split')
        fresh = []
        if receipt['counts']['missing_unique_cells']:
            plan_path = root / 'plans' / (stage + '-' + model) / 'plan.json'
            execution_path = root / model / 'schedules' / (stage + '-execution.json')
            if not plan_path.exists() or not execution_path.exists():
                return None
            verify_sources(read_plan(plan_path))
            pins.update({str(plan_path): file_digest(plan_path), str(execution_path): file_digest(execution_path)})
            if checked(execution_path)['sha256'] != receipt['missing_schedule_sha256']:
                raise ValueError('execution schedule differs from missing-cell receipt')
            fresh = scan_execution(study, model, stage)  # v12 root, unchanged physical-v5 contract.
            if fresh is None:
                return None
        bundles.append((master, receipt, fresh))
    inventory = physical_inventory(source, model)
    measured = [row for master, receipt, fresh in bundles
                for row in materialize(master, receipt, fresh, inventory)]
    factorial = checked(factorial_path)
    expected = factorial['counts']
    ids = set(expected['adaptive'])
    full = [row for (request, selector, _), row in inventory.items() if request in ids and selector == 'full']
    if len(full) != len(ids):
        raise ValueError('missing physical all-token reference')
    for row in [*full, *measured]:
        previous_report._verify_physical_provenance({**row, 'physical_measurement_performed': False})
        pins[row['physical_source_path']] = row['physical_source_sha256']
        actual = row.get('metrics', {}).get('cache', {}).get('kv_capacity_bytes')
        calculated = (row['input_text_tokens'] + row['retained_visual_tokens'] + study['generation_reserve']) * study['bytes_per_token'][model]
        if type(actual) is not int or actual != calculated:
            raise ValueError('measured KV capacity missing or differs from exact runtime accounting')
    result = previous_report.report(study, model, [*full, *measured], {'quality_loss_1pp': expected},
                                    alternative_selector='divprune', bootstrap_repetitions=repetitions)
    rows, summaries = [], {}
    for dataset in sorted({canonical_dataset(r['dataset']) for r in full}):
        references = [{**r, 'dataset': dataset} for r in full if canonical_dataset(r['dataset']) == dataset]
        ref_ids = {r['request_id'] for r in references}
        methods = {'full': references, **{method: [r for r in measured if r['dataset'] == dataset and r['method'] == method]
                                         for method in METHODS}}
        budgets = []
        for method, values in methods.items():
            if len(values) != len(ref_ids) or {r['request_id'] for r in values} != ref_ids:
                raise ValueError('physical method lacks the entire declared dataset cohort')
            memory = previous_report.old.memory_summary(values, references)
            if method != 'full':
                budgets.append(memory['aggregate_allocated_kv_bytes'])
            interval = paired_ci(values, references, references[0]['quality_metric'], repetitions=repetitions)
            row = {'model': model, 'dataset': dataset, 'method': method, 'label': LABELS[method],
                'score_percent': quality_summary(values, references[0]['quality_metric'])['score_percent'],
                'mean_allocated_kv_mib': memory['mean_allocated_kv_bytes'] / 2**20,
                'kv_reduction_percent': memory['aggregate_allocated_kv_reduction_percent'],
                'percent_requests_with_memory_reduction': 100 * memory['fraction_compressed'],
                'versus_all_tokens': interval, 'memory': memory, 'quality_is_physical': True,
                'kv_source': 'measured_allocated_key_value_tensor_capacity', 'independent_confirmation': False}
            rows.append(row)
        if len(set(budgets)) != 1:
            raise ValueError('compressed methods do not match exact aggregate measured KV')
        summaries[dataset] = {'aggregate_allocated_kv_bytes_each_compressed_method': budgets[0],
                              'matched_methods': list(METHODS), 'complete_requests': len(ref_ids)}
    return seal({'format': FORMAT + '_matched', 'study_sha256': study['sha256'], 'model': model,
        'rows': rows, 'matched_capacity_audit': summaries, 'factorial_report': result,
        'all_declared_cohorts_complete': True, 'exact_matched_controls_complete': True,
        'independent_confirmation': False, 'interpolation_used': False,
        'bootstrap_repetitions': repetitions, 'report_source_sha256': file_digest(__file__), 'input_hashes': pins,
        'physical_sources_sha256': digest([(r['physical_source_path'], r['physical_source_sha256']) for r in measured])})


def status_table(study, selection=None):
    root = Path(study['output_root'])
    rows = []
    for model in study['models']:
        features = root / model / 'features'
        count = sum((features / f'rank{i}.complete.json').exists() for i in range(study['world_size']))
        fitted = root / model / 'development/report.json'
        rows.append({'model': model, 'features_complete_markers': count, 'features_expected_shards': study['world_size'],
                     'calibration_report_present': fitted.exists(), 'physical_splits': {}})
        for split in ('fit', 'screening'):
            receipt = root / model / 'schedules' / (split + '-matched-reuse.json')
            if receipt.exists():
                value = checked(receipt)
                missing = value['counts']['missing_unique_cells']
                done = sum((root / model / (split + '-matched') / f'rank{i}.complete.json').exists()
                           for i in range(study['world_size']))
                rows[-1]['physical_splits'][split] = {'missing_unique_cells': missing,
                    'cached_unique_cells': value['counts']['cached_unique_cells'],
                    'completed_shard_markers': done, 'expected_shards': study['world_size'] if missing else 0,
                    'status': 'awaiting_audit' if not missing or done == study['world_size'] else 'incomplete'}
            else:
                rows[-1]['physical_splits'][split] = {'status': 'not_published'}
    return {'models': rows, 'selected_rule': selection.get('rule_kind') if selection else None,
        'selection_reason': selection.get('reason') if selection else 'awaiting_both_model_calibrations',
        'native_runtime_gate': ('pending_measured_overhead' if selection and selection.get('provisional_pending_overhead')
                               else 'not_evaluated' if selection is None else 'not_required_for_native_promotion'),
        'full_suite': 'blocked_until_independent_identity_admission', 'independent_confirmation': False}


def deployment_decision(study, selection, reports, *, now=None, write=True):
    """Freeze final deployment separately from the unmodified development winner.

    Native geometry needs measured overhead <=5% on both models. At a failed
    gate, or the eight-hour method-freeze deadline with incomplete timing, use
    the best eligible pre-existing simple definition instead. Independent
    evaluation still requires its separate identity-admission audit.
    """
    root = Path(study['output_root'])
    destination = root / 'deployment.json'
    if destination.exists():
        value = checked(destination)
        if value['study_sha256'] != study['sha256'] or value['selection_sha256'] != selection['sha256']:
            raise ValueError('deployment belongs to another frozen selection')
        for path, sha in value['input_hashes'].items():
            if file_digest(path) != sha:
                raise ValueError('deployment admission evidence changed: ' + path)
        return value
    now = time.time() if now is None else now
    normal = {model: value.get('result', value) for model, value in reports.items()}
    if set(normal) != set(study['models']):
        return {'status': 'awaiting_both_model_calibrations', 'frozen': False, 'rule_kind': None}
    pins, gates = {}, {}
    selection_path = root / 'selection.json'
    if selection_path.exists():
        pins[str(selection_path)] = file_digest(selection_path)
    for model in study['models']:
        development_path = root / model / 'development/report.json'
        if development_path.exists():
            pins[str(development_path)] = file_digest(development_path)
        path = root / 'reports' / ('systems-' + model + '.json')
        complete_path = root / model / 'systems/complete.json'
        if not path.exists() or not complete_path.exists():
            gates[model] = {'measured': False, 'status': 'not_yet_measured'}
            continue
        marker = checked(complete_path)
        if marker.get('study_sha256') != study['sha256'] or marker.get('model') != model:
            raise ValueError('systems completion belongs to another study/model')
        for source_path, sha in marker['input_hashes'].items():
            if file_digest(source_path) != sha:
                raise ValueError('systems measurement evidence changed')
        measured = json.loads(path.read_text()).get('budget_overhead_gate')
        if (not measured or measured.get('measured') is not True or type(measured.get('passed')) is not bool
                or measured.get('maximum_fraction') != study['maximum_latency_overhead_fraction']
                or measured.get('quality_and_selected_patch_parity') is not True):
            raise ValueError('invalid measured live-policy overhead gate')
        gates[model] = measured
        pins.update({str(path): file_digest(path), str(complete_path): file_digest(complete_path), **marker['input_hashes']})
    native = selection['rule_kind'] == 'native_coverage'
    failed = any(g.get('measured') and not g['passed'] for g in gates.values())
    passed = all(g.get('measured') and g['passed'] for g in gates.values())
    if native and not failed and not passed and now < study['method_freeze_utc_epoch']:
        return {'status': 'native_pending_measured_overhead', 'frozen': False,
            'rule_kind': 'native_coverage', 'overhead_gates': gates, 'independent_confirmation': False}
    fallback = native and not passed
    decision = (policy.select_shared(normal, cost_tie_fraction=study['cost_tie_fraction'],
                                     allowed_kinds=policy.KINDS[:-1]) if fallback else selection)
    reason = ('native_measured_overhead_failed_use_existing_rule' if failed else
              'native_overhead_not_measured_by_method_freeze_use_existing_rule') if fallback else (
              'native_passed_both_measured_overhead_gates' if native else 'existing_rule_no_native_promotion_gate')
    value = seal({'format': FORMAT + '_deployment', 'study_sha256': study['sha256'],
        'selection_sha256': selection['sha256'], 'created_utc_epoch': now, 'frozen': True,
        'rule_kind': decision['rule_kind'], 'models': decision['models'], 'reason': reason,
        'status': 'all_tokens_no_eligible_shared_rule' if decision['rule_kind'] is None else 'frozen_for_benchmarking',
        'development_selected_rule': selection['rule_kind'], 'native_rejected': fallback,
        'overhead_gates': gates, 'input_hashes': pins, 'independent_confirmation': False,
        'full_suite_requires_separate_identity_admission': True})
    if write:
        write_immutable(destination, value)
    return value


def markdown(report):
    def interval(value):
        if value['ci_lower_pp'] is None:
            return f"{value['delta_pp']:+.2f}; CI unavailable"
        return f"{value['delta_pp']:+.2f} [{value['ci_lower_pp']:+.2f}, {value['ci_upper_pp']:+.2f}]"
    lines = ['# AsymVLM one-day sprint: quality and allocated KV capacity', '',
        'Development evidence only. Confidence intervals resample image/document groups and preserve MME pairs and categories. '
        'They do not certify an independently confirmed quality-loss bound.', '',
        '## Table 1. Budget-rule development comparison', '',
        'Scores are the equally weighted mean of four native benchmark metrics. Memory is the request-weighted mean allocated KV capacity. '
        'These procedures choose different budgets: use Table 2 for equal-budget accuracy comparisons. '
        'Capacity is calculated from exact physical token counts, not peak GPU memory.', '',
        '| Model | Budget procedure | Score | Mean KV MiB | KV saved | Inputs saving memory | Difference vs all tokens, pp [95% CI] |',
        '|---|---|---:|---:|---:|---:|---|']
    for row in report['development']['macro_rows']:
        lines.append(f"| {row['model']} | {row['label']} | {row['score_percent']:.2f} | {row['mean_allocated_kv_mib']:.2f} | "
            f"{row['kv_reduction_percent']:.1f}% | {row['percent_requests_with_memory_reduction']:.1f}% | {interval(row['versus_all_tokens'])} |")
    if not report['development']['macro_rows']:
        lines.append('| — | Awaiting complete feature extraction and calibration | — | — | — | — | — |')
    lines += ['', 'Full per-dataset development rows and fixed-allocation diagnostics are in the JSON report.', '',
        '## Table 2. Physically evaluated, exactly matched-budget comparisons', '',
        'Within each model/dataset, all four compressed methods have exactly the same aggregate measured allocated KV capacity. '
        'The uncompressed reference intentionally uses a different budget. DivPrune is a selector adaptation in the shared pipeline. '
        'No partial-cohort or interpolated quality is shown.', '',
        '| Model | Dataset | Method | Score | Mean measured KV MiB | KV saved | Inputs saving memory | Difference vs all tokens, pp [95% CI] |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for row in report['matched']['rows']:
        lines.append(f"| {row['model']} | {row['dataset']} | {row['label']} | {row['score_percent']:.2f} | "
            f"{row['mean_allocated_kv_mib']:.2f} | {row['kv_reduction_percent']:.1f}% | "
            f"{row['percent_requests_with_memory_reduction']:.1f}% | {interval(row['versus_all_tokens'])} |")
    if not report['matched']['rows']:
        lines.append('| — | — | Awaiting all declared physical control cells | — | — | — | — | — |')
    lines += ['', 'Exact paired adaptive-minus-static and selector contrasts are stored in each matched-model JSON report. '
        'Cached timing is not controlled latency evidence; allocated KV, occupied KV, and total GPU peak are separate quantities.', '',
        '## Source-based progress', '', '| Model | Feature markers | Calibration | Physical controls |', '|---|---:|---|---|']
    for row in report['status']['models']:
        progress = ', '.join(split + ': ' + values['status'] for split, values in row['physical_splits'].items())
        lines.append(f"| {row['model']} | {row['features_complete_markers']}/{row['features_expected_shards']} | "
                     f"{'available' if row['calibration_report_present'] else 'pending'} | {progress} |")
    lines += ['', 'Shared rule: ' + str(report['status']['selected_rule'] or 'not selected') + '. '
        'Native-geometry overhead gate: ' + report['status']['native_runtime_gate'] + '. '
        'Independent confirmation remains blocked until identity admission passes.', '']
    return '\n'.join(lines)


def generate(study, reports=None, status=None, *, repetitions=10000, write=True):
    from .compression_study_v12 import verify
    verify(study)
    root = Path(study['output_root'])
    reports = dict(reports or {})
    for model in study['models']:
        path = root / model / 'development/report.json'
        if model not in reports and path.exists():
            reports[model] = checked(path)
    selection_path = root / 'selection.json'
    selection = checked(selection_path) if selection_path.exists() else None
    development, matched, matched_sources = {'macro_rows': [], 'dataset_rows': []}, [], {}
    for model, report in reports.items():
        _seal_checked(report, 'development report')
        if report['study_sha256'] != study['sha256'] or report['model'] != model:
            raise ValueError('development report belongs to another study/model')
        stored = checked(root / model / 'development/rows.json')
        if digest(stored['rows']) != report['rows_sha256']:
            raise ValueError('physical development rows changed')
        table = cached_development_table(study, model, stored['rows'], report,
                                         repetitions=repetitions, write=write)
        for key in development:
            development[key].extend(table[key])
        physical = collect_matched(study, model, repetitions=repetitions)
        if physical is not None:
            matched.extend(physical['rows'])
            matched_sources[model] = physical
            if write:
                atomic_json(root / 'reports' / (model + '-matched.json'), physical)
    progress = status_table(study, selection)
    deployment = deployment_decision(study, selection, reports, write=write) if selection else None
    if deployment is not None:
        progress['deployment_status'] = deployment['status']
        progress['deployed_rule'] = deployment['rule_kind'] if deployment.get('frozen') else None
        if selection['rule_kind'] == 'native_coverage':
            progress['native_runtime_gate'] = deployment.get('reason', deployment['status'])
    for model_status in progress['models']:
        if model_status['model'] in matched_sources:
            for values in model_status['physical_splits'].values():
                values['status'] = 'audited_complete'
    value = seal({'format': FORMAT, 'study_sha256': study['sha256'], 'development': development,
        'matched': {'rows': matched, 'complete_models': sorted(matched_sources),
                    'exact_matched_controls_complete': set(matched_sources) == set(study['models'])},
        'status': progress, 'coordinator_status': status, 'selection': selection, 'deployment': deployment,
        'quality_is_physical': True, 'interpolation_used': False, 'independent_confirmation': False,
        'source_hashes': {str(Path(__file__).resolve()): file_digest(__file__)}})
    if write:
        atomic_json(root / 'reports/tables.json', value)
        _atomic_text(root / 'reports/TABLES.md', markdown(value))
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--bootstrap-repetitions', type=int, default=10000)
    args = parser.parse_args(argv)
    result = generate(checked(args.study), repetitions=args.bootstrap_repetitions)
    print(json.dumps({'study_sha256': result['study_sha256'], 'status': result['status'],
                      'exact_matched_controls_complete': result['matched']['exact_matched_controls_complete']}, sort_keys=True))


if __name__ == '__main__':
    main()
