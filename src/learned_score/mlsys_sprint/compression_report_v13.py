"""Complete-cohort finalization reports; no implicit promotion or interpolation."""
from collections import defaultdict
import copy
from pathlib import Path
import time

from . import compression_policy_v12 as policy
from .compression_matrix_v8 import checked, seal
from .compression_report_v12 import paired_intervals
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest
from .adaptive_report_v1 import _atomic_text

DEPLOYABLE = ('scorer_static', 'scorer_adaptive', 'divprune_static', 'divprune_adaptive')
DIAGNOSTICS = ('scorer_matched', 'divprune_matched')
METHODS = (*DEPLOYABLE, *DIAGNOSTICS)
LABELS = {'full': 'All image tokens', 'scorer_static': 'Learned scorer, fixed budget',
    'scorer_adaptive': 'Learned scorer, adaptive coverage',
    'divprune_static': 'DivPrune, fixed budget',
    'divprune_adaptive': 'Learned coverage + DivPrune',
    'scorer_matched': 'Learned scorer, matched uniform control',
    'divprune_matched': 'DivPrune, matched uniform control'}
MODELS = {'qwen25': 'Qwen2.5-VL-3B', 'llava_next': 'LLaVA-NeXT-7B'}


def _index(values):
    result = {r['request_id']: r for r in values}
    if len(result) != len(values):
        raise ValueError('duplicate physical request/method')
    return result


def paired_methods(candidate, reference, *, repetitions=10000):
    """Use the existing grouped bootstrap on two observed quality channels.

    Keys 1/2 are internal channel labels, never visual counts or inference values.
    No synthetic quality values or interpolated retention outcomes are created.
    """
    a, b = _index(candidate), _index(reference)
    if set(a) != set(b):
        raise ValueError('paired comparison requires complete identical cohorts')
    rows = []
    for key in sorted(a):
        for field in ('dataset', 'group_id', 'category', 'input_visual_tokens', 'input_text_tokens'):
            if a[key].get(field) != b[key].get(field):
                raise ValueError('paired input geometry changed: ' + field)
        rows.append({'row_hash': key, 'dataset': a[key]['dataset'], 'group_id': a[key]['group_id'],
            'category': a[key].get('category'),
            'quality_by_count': {'1': a[key]['quality_score'], '2': b[key]['quality_score']}})
    return paired_intervals(rows, {k: 1 for k in a}, {k: 2 for k in b}, repetitions=repetitions)


def physical_summary(study, model, values, references, *, repetitions=10000):
    a, b = _index(values), _index(references)
    if set(a) != set(b):
        raise ValueError('no partial cohort scores are allowed')
    rows, counts = [], {}
    for key in sorted(a):
        item, full = a[key], b[key]
        for field in ('dataset', 'group_id', 'category', 'input_visual_tokens', 'input_text_tokens'):
            if item.get(field) != full.get(field):
                raise ValueError('physical candidate/reference identity mismatch')
        n, k = item['input_visual_tokens'], item['retained_visual_tokens']
        if type(k) is not int or not 1 <= k <= n or item['selected_count'] != k:
            raise ValueError('invalid exact retained count')
        if k == n and item['quality_score'] != full['quality_score']:
            raise ValueError('full-count outcome lacks full-token parity')
        expected = (item['input_text_tokens'] + k + study['generation_reserve']) * study['bytes_per_token'][model]
        if item.get('metrics', {}).get('cache', {}).get('kv_capacity_bytes') != expected:
            raise ValueError('measured KV does not match protected prefix plus selected tokens and output reserve')
        counts[key] = k
        rows.append({'row_hash': key, 'group_id': item['group_id'], 'dataset': item['dataset'],
            'category': item.get('category'), 'input_visual_tokens': n, 'input_text_tokens': item['input_text_tokens'],
            'quality_by_count': {str(n): full['quality_score'], str(k): item['quality_score']}})
    summary = policy.summary(rows, counts, bytes_per_token=study['bytes_per_token'][model],
        generation_reserve=study['generation_reserve'])
    return {'summary': summary, 'counts': counts,
        'intervals': paired_methods(values, references, repetitions=repetitions),
        'quality_constraints_satisfied': policy.meets_constraints(summary, 1., 2.)}


def model_report(study, model, physical, *, repetitions=10000):
    groups = defaultdict(list)
    for row in physical:
        if row['method'] not in ('full', *METHODS):
            raise ValueError('undeclared physical method')
        groups[row['method']].append(row)
    if set(groups) != {'full', *METHODS}:
        raise ValueError('all six declared arms and full reference required')
    source_rows = checked(Path(study['output_root']) / model / 'development/rows.json')['rows']
    expected = {r['row_hash'] for r in source_rows}
    if any(set(_index(v)) != expected for v in groups.values()):
        raise ValueError('physical report does not cover the predeclared complete cohort')
    summaries = {method: physical_summary(study, model, values, groups['full'], repetitions=repetitions)
                 for method, values in groups.items()}
    contrasts = {}
    for a, b in (('scorer_adaptive', 'scorer_matched'), ('divprune_adaptive', 'divprune_matched'),
                 ('scorer_adaptive', 'divprune_adaptive'), ('scorer_static', 'divprune_static')):
        for dataset in policy.DATASETS:
            left = summaries[a]['summary']['domains'][dataset]['allocated_kv_bytes']
            right = summaries[b]['summary']['domains'][dataset]['allocated_kv_bytes']
            if left != right:
                raise ValueError('paired contrast does not exactly match per-dataset KV capacity')
        contrasts[a + '_minus_' + b] = paired_methods(groups[a], groups[b], repetitions=repetitions)
    pins = {r['physical_source_path']: r['physical_source_sha256'] for r in physical}
    return seal({'study_sha256': study['sha256'], 'model': model, 'methods': summaries,
        'contrasts': contrasts, 'input_hashes': pins, 'report_source_sha256': file_digest(__file__),
        'bootstrap_repetitions': repetitions, 'complete': True, 'independent_confirmation': False,
        'quality_source': 'physical_inference_at_exact_counts', 'interpolation_used': False,
        'diagnostic_methods_not_deployable': list(DIAGNOSTICS),
        'divprune_scope': 'selector adaptation; adaptive variant retains learned budgeting cost'})


def choose_final(study, report, reports, profiles):
    base = {'study_sha256': study['sha256'], 'report_sha256': report['sha256'],
        'profiles_sha256': profiles['sha256'], 'independent_confirmation': False,
        'evaluation_constraints': study['evaluation_constraints'], 'time': time.time(),
        'safe_action': 'all_image_tokens', 'selected_method': None,
        'quality_is_development_evidence': True, 'runtime_verified': False}
    if not report['complete']:
        return seal({**base, 'status': 'provisional_no_winner_incomplete_required_comparisons'})
    models = sorted(study['models'])
    candidates = []
    for method in DEPLOYABLE:
        summaries = [report['models'][m]['methods'][method]['summary'] for m in models]
        if all(policy.meets_constraints(s, 1., 2.) for s in summaries):
            candidates.append({'method': method,
                'cost': sum(s['mean_allocated_kv_bytes'] for s in summaries) / len(models),
                'worst_loss': max(s['worst_domain_quality_loss_pp'] for s in summaries),
                'mean_loss': sum(s['mean_quality_loss_pp'] for s in summaries) / len(models)})
    if not candidates:
        return seal({**base, 'status': 'no_compressed_method_meets_original_quality_limits'})
    least = min(c['cost'] for c in candidates)
    near = [c for c in candidates if c['cost'] <= least * 1.01 + 1e-8]
    winner = min(near, key=lambda c: (not c['method'].endswith('_static'), c['worst_loss'], c['mean_loss'], c['method']))
    method = winner['method']
    form = 'static' if method.endswith('_static') else 'adaptive'
    variant = profiles['forms'][form]['variant_id']
    rules = {m: reports[m]['result']['final_rules'][variant] for m in models}
    # A full-only outcome is a safe endpoint, not a compression success.
    if all(report['models'][m]['methods'][method]['summary']['requests_with_memory_reduction'] == 0 for m in models):
        return seal({**base, 'status': 'only_full_token_endpoint_supported', 'candidates': candidates})
    return seal({**base, 'selected_method': method, 'method_id': method, 'variant_id': variant,
        'rule_kind': 'static' if form == 'static' else 'rank32_coverage',
        'selector': method.split('_')[0], 'model_rules': rules, 'candidates': candidates,
        'status': 'development_method_frozen_pending_systems_and_independent_confirmation',
        'static_semantics': 'same learned-curve validation and full-fallback as frozen rule; not scorer-free',
        'hybrid_requires_learned_scorer': method == 'divprune_adaptive'})


def _cell(summary):
    return (f"{100 * summary['macro_accuracy']:.2f} | {summary['mean_allocated_kv_mib']:.2f} | "
            f"{summary['kv_reduction_percent']:.1f}% | {summary['percent_requests_with_memory_reduction']:.1f}%")


def markdown(report):
    lines = ['# Method finalization: development results', '',
        'Previously examined development data; not independent confirmation. Scores average four native benchmark metrics. '
        'KV capacity excludes weights and activations and is not whole-GPU peak memory.', '',
        '## Table 1. Deployable methods', '',
        '| Model | Method | Score | Mean KV MiB | KV saved | Inputs saving KV | Score change vs all tokens [95% CI], points |',
        '|---|---|---:|---:|---:|---:|---|']
    for model in MODELS:
        physical = report['models'].get(model)
        if physical:
            for method in ('full', *DEPLOYABLE):
                value = physical['methods'][method]
                ci = value['intervals']['macro']
                interval = (f"{ci['delta_pp']:+.2f} [{ci['ci_lower_pp']:+.2f}, {ci['ci_upper_pp']:+.2f}]"
                            if ci['ci_lower_pp'] is not None else f"{ci['delta_pp']:+.2f} [unavailable]")
                lines.append(f"| {MODELS[model]} | {LABELS[method]} | {_cell(value['summary'])} | {interval} |")
        else:
            lines.append(f'| {MODELS[model]} | Awaiting complete physical comparisons | — | — | — | — | — |')
    lines += ['', '## Table 2. Exactly matched-budget comparisons', '',
        'All four methods below have identical aggregate KV capacity within each model/benchmark. '
        'Uniform controls are cohort-level comparisons, not deployed rules.', '',
        '| Model | Benchmark | Method | Score | Mean KV MiB | Score change vs all tokens, points |',
        '|---|---|---|---:|---:|---:|']
    for model in MODELS:
        physical = report['models'].get(model)
        if not physical:
            lines.append(f'| {MODELS[model]} | — | Awaiting complete cohort | — | — | — |')
            continue
        for dataset in policy.DATASETS:
            for method in ('scorer_adaptive', 'scorer_matched', 'divprune_adaptive', 'divprune_matched'):
                s = physical['methods'][method]['summary']['domains'][dataset]
                lines.append(f"| {MODELS[model]} | {dataset} | {LABELS[method]} | {100*s['accuracy']:.2f} | "
                             f"{s['mean_allocated_kv_mib']:.2f} | {s['delta_full_pp']:+.2f} |")
    lines += ['', '## Calibration candidates (cached physical quality; differing budgets)', '',
        '| Model | Candidate | Score | Mean KV MiB | KV saved | Mean loss, points | Worst benchmark loss, points |',
        '|---|---|---:|---:|---:|---:|---:|']
    for model, variants in sorted(report.get('development', {}).items()):
        for name, summary in variants.items():
            lines.append(f"| {MODELS[model]} | {name} | {100*summary['macro_accuracy']:.2f} | "
                f"{summary['mean_allocated_kv_mib']:.2f} | {summary['kv_reduction_percent']:.1f}% | "
                f"{summary['mean_quality_loss_pp']:.2f} | {summary['worst_domain_quality_loss_pp']:.2f} |")
    lines += ['', 'Paired allocation/selector contrasts and per-benchmark confidence intervals are in tables.json. '
        'Cached timings are not controlled runtime evidence.', '']
    return '\n'.join(lines)


def refresh(study, *, reports=None, profiles=None, repetitions=10000):
    from .compression_execution_v13 import collect_model
    from .compression_finalize_v13 import verify
    verify(study)
    root = Path(study['output_root'])
    reports = reports or {m: checked(root / m / 'development/report.json') for m in study['models']}
    profiles = profiles or checked(root / 'profiles.json')
    physical = {}
    for model in sorted(study['models']):
        cache = root / 'reports' / (model + '-physical.json')
        if cache.exists():
            result = checked(cache)
            if result['study_sha256'] != study['sha256'] or result['report_source_sha256'] != file_digest(__file__):
                raise ValueError('physical result belongs to changed study/reporter')
            for path, sha in result['input_hashes'].items():
                if file_digest(path) != sha:
                    raise ValueError('physical measurement changed after reporting')
            physical[model] = result
        else:
            values = collect_model(study, model)
            if values is not None:
                result = model_report(study, model, values, repetitions=repetitions)
                write_immutable(cache, result)
                physical[model] = result
    value = seal({'study_sha256': study['sha256'], 'models': physical,
        'development': {m: {v: p['summary'] for v, p in r['result']['procedures'].items()} for m, r in reports.items()},
        'profiles': profiles, 'complete': set(physical) == set(study['models']),
        'independent_confirmation': False, 'bootstrap_repetitions': repetitions})
    atomic_json(root / 'reports/tables.json', value)
    _atomic_text(root / 'reports/TABLES.md', markdown(value))
    return value
