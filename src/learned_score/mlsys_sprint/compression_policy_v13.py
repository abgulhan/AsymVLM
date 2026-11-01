"""Bounded, memory-first finalization of fixed and coverage vision budgets.

Six declared procedures differ only in budget form and calibration safety
margin. Evaluation limits stay fixed. Labels are used for fitting and explicit
development diagnostics, never passed to the deployed rule. All quality comes
from physical counts; no quality monotonicity or interpolation is assumed.
"""

import math

from . import compression_policy_v12 as previous
from .state import digest

FORMAT = 'mlsys_compression_policy_v13'
DATASETS, METRICS = previous.DATASETS, previous.METRICS
KINDS = ('static', 'rank32_coverage')
PROFILES = {'original': (1., 2.), 'conservative': (.5, 1.),
            'extra_conservative': (.25, .5)}
VARIANTS = tuple(f'{kind}__{profile}' for kind in KINDS for profile in PROFILES)
EVALUATION_CONSTRAINTS = {'max_mean_loss_pp': 1., 'max_domain_loss_pp': 2.}
FORMS = {'static': 'static', 'adaptive': 'rank32_coverage'}
grouped_folds = previous.grouped_folds
answer_free, infer, summary = previous.answer_free, previous.infer, previous.summary


def _seal(value):
    return {**value, 'sha256': digest(value)}


def _variant(variant_id):
    if variant_id not in VARIANTS:
        raise ValueError('undeclared variant_id')
    kind, profile = variant_id.split('__')
    mean, domain = PROFILES[profile]
    return {'variant_id': variant_id, 'kind': kind, 'profile': profile,
            'fit_constraints': {'max_mean_loss_pp': mean, 'max_domain_loss_pp': domain}}


def fit_variant(rows, variant_id, *, model, bytes_per_token, generation_reserve=127):
    """Return a sealed v12-compatible inference rule; caps are fitting caps only."""
    variant = _variant(variant_id)
    return previous.fit_rule(rows, variant['kind'], model=model,
        bytes_per_token=bytes_per_token, generation_reserve=generation_reserve,
        **variant['fit_constraints'])


def _eligible(value):
    return previous.meets_constraints(value, **EVALUATION_CONSTRAINTS)


def _cost_tie(value):
    if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < 1:
        raise ValueError('relative cost tie must be finite and in [0,1)')


def choose_variant(candidates, *, cost_tie_fraction=.01, prefer_static=False):
    """Select with fixed evaluation caps, never the candidate's fitting caps."""
    _cost_tie(cost_tie_fraction)
    eligible = [c for c in candidates if _eligible(c['summary'])]
    if not eligible:
        return None
    least = min(c['summary']['mean_allocated_kv_bytes'] for c in eligible)
    near = [c for c in eligible if c['summary']['mean_allocated_kv_bytes'] <=
            least * (1 + cost_tie_fraction) + 1e-8]
    return min(near, key=lambda c: (
        int(c['kind'] != 'static') if prefer_static else 0,
        c['summary']['worst_domain_quality_loss_pp'],
        c['summary']['mean_quality_loss_pp'], c['variant_id']))


def development_report(rows, *, model, bytes_per_token, generation_reserve=127,
                       outer_folds=5, inner_folds=3, seed=20261020):
    """Six excluded-fold estimates plus nested selection; no confirmation claim.

Outer-fold candidate rules never use that fold's labels. Nested choices use
inner validation summaries only. Inner summaries are retained so a later
cross-model reducer can choose a shared profile without outer-label leakage.
"""
    previous._configuration(model, bytes_per_token, generation_reserve, 1., 2.)
    rows = previous.validate_rows(rows, 'rank32_coverage')
    if {r['dataset'] for r in rows} != set(DATASETS):
        raise ValueError('all four development benchmarks required')
    config = dict(model=model, bytes_per_token=bytes_per_token,
                  generation_reserve=generation_reserve)
    summarise = lambda data, counts: summary(data, counts,
        bytes_per_token=bytes_per_token, generation_reserve=generation_reserve)
    outer = grouped_folds(rows, outer_folds, seed)
    candidate_counts = {variant: {} for variant in VARIANTS}
    selected, records = {}, []
    selected_forms = {form: {} for form in FORMS}

    def choose(train, local_seed):
        partition = grouped_folds(train, inner_folds, local_seed)
        candidates = []
        for variant_id in VARIANTS:
            counts, folds = {}, []
            for fold in range(inner_folds):
                fit = [r for r in train if partition[r['group_id']] != fold]
                valid = [r for r in train if partition[r['group_id']] == fold]
                if not fit or not valid:
                    raise ValueError('empty grouped calibration partition')
                rule = fit_variant(fit, variant_id, **config)
                output = infer(rule, answer_free(valid))
                counts.update(output['counts'])
                folds.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
                    'validation_groups': sorted({r['group_id'] for r in valid}),
                    'policy_sha256': rule['sha256'], 'threshold': rule['threshold'],
                    'retention_percent': rule['retention_percent'],
                    'fallbacks': output['fallbacks']})
            candidates.append({**_variant(variant_id), 'summary': summarise(train, counts),
                'folds': folds, 'evaluation_constraints': dict(EVALUATION_CONSTRAINTS)})
        winner = choose_variant(candidates, prefer_static=True)
        forms = {form: choose_variant([c for c in candidates if c['kind'] == kind])
                 for form, kind in FORMS.items()}
        return winner, forms, candidates

    for fold in range(outer_folds):
        train = [r for r in rows if outer[r['group_id']] != fold]
        valid = [r for r in rows if outer[r['group_id']] == fold]
        winner, forms, inner = choose(train, seed + fold + 1)
        rules, outputs = {}, {}
        for variant_id in VARIANTS:
            rules[variant_id] = fit_variant(train, variant_id, **config)
            outputs[variant_id] = infer(rules[variant_id], answer_free(valid))
            candidate_counts[variant_id].update(outputs[variant_id]['counts'])
        full_rule = previous.fit_rule(train, 'static', **config, _force_full=True)
        full_output = infer(full_rule, answer_free(valid))
        winner_id = winner['variant_id'] if winner else None
        output = outputs[winner_id] if winner else full_output
        rule = rules[winner_id] if winner else full_rule
        selected.update(output['counts'])
        for form, candidate in forms.items():
            selected_forms[form].update((outputs[candidate['variant_id']] if candidate else
                                          full_output)['counts'])
        records.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
            'validation_groups': sorted({r['group_id'] for r in valid}),
            'selected_variant_id': winner_id, 'selected_kind': rule['kind'] if winner else 'full',
            'selected_by_form': {form: c['variant_id'] if c else None for form, c in forms.items()},
            'rule': rule, 'inner_candidates': inner, 'validation_assignments': output,
            'validation_summary': summarise(valid, output['counts']),
            'candidate_rules': rules, 'candidate_assignments': outputs,
            'full_assignments': full_output,
            'candidate_validation_summaries': {v: summarise(valid, o['counts']) for v, o in outputs.items()}})
    final_winner, final_forms, inner = choose(rows, seed + 100)
    final_rules = {variant: fit_variant(rows, variant, **config) for variant in VARIANTS}
    final_id = final_winner['variant_id'] if final_winner else None
    final_rule = final_rules[final_id] if final_id else previous.fit_rule(
        rows, 'static', **config, _force_full=True)
    procedures = {variant: {**_variant(variant), 'counts': counts,
        'summary': summarise(rows, counts),
        'allocation': previous.previous.allocation_summary(rows, counts, 'scorer'),
        'evaluation_constraints': dict(EVALUATION_CONSTRAINTS)}
        for variant, counts in candidate_counts.items()}
    thresholds = {}
    for variant in VARIANTS:
        field = 'retention_percent' if _variant(variant)['kind'] == 'static' else 'threshold'
        values = [record['candidate_rules'][variant][field] for record in records]
        thresholds[variant] = {'parameter': field, 'values': values, 'minimum': min(values),
            'maximum': max(values), 'range': max(values) - min(values),
            'distinct_values': len(set(values))}
    return _seal({'format': FORMAT + '_nested_development', 'model': model,
        'configuration': {**config, 'evaluation_constraints': dict(EVALUATION_CONSTRAINTS)},
        'variant_catalog': [_variant(v) for v in VARIANTS],
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seed': seed,
        'folds': records, 'procedures': procedures, 'candidates': procedures,
        'counts': selected, 'summary': summarise(rows, selected),
        'allocation': previous.previous.allocation_summary(rows, selected, 'scorer'),
        'nested_forms': {form: {'counts': counts, 'summary': summarise(rows, counts)}
                         for form, counts in selected_forms.items()},
        'final_variant_id': final_id, 'final_kind': final_rule['kind'] if final_id else 'full',
        'final_rule': final_rule, 'final_rules': final_rules,
        'final_form_choices': {form: c['variant_id'] if c else None for form, c in final_forms.items()},
        'final_inner_candidates': inner, 'threshold_stability': thresholds,
        'evaluation_constraints': dict(EVALUATION_CONSTRAINTS),
        'fit_constraints_are_separate_from_evaluation': True,
        'independent_confirmation': False, 'runtime_promotion_gate_evaluated': False,
        'shared_cross_model_choice_required': True, 'not_exact_budget_matched': True,
        'physical_matched_comparison_required': True,
        'selection_rule': 'fixed evaluation caps; minimum KV; within 1% worst loss, mean loss, ID; static preferred across forms',
        'evidence': 'nested development on previously used data'})


def _comparison(variant_id, summaries):
    return {**_variant(variant_id), 'model_summaries': summaries,
        'feasible_on_all_models': all(_eligible(s) for s in summaries.values()),
        'mean_allocated_kv_bytes_across_models': sum(s['mean_allocated_kv_bytes'] for s in summaries.values()) / len(summaries),
        'worst_domain_quality_loss_pp': max(s['worst_domain_quality_loss_pp'] for s in summaries.values()),
        'mean_quality_loss_pp_across_models': sum(s['mean_quality_loss_pp'] for s in summaries.values()) / len(summaries)}


def _choose_shared(comparisons, kind, cost_tie_fraction):
    eligible = [c for c in comparisons.values() if c['kind'] == kind and c['feasible_on_all_models']]
    if not eligible:
        return None
    least = min(c['mean_allocated_kv_bytes_across_models'] for c in eligible)
    return min((c for c in eligible if c['mean_allocated_kv_bytes_across_models'] <=
                least * (1 + cost_tie_fraction) + 1e-8), key=lambda c: (
        c['worst_domain_quality_loss_pp'], c['mean_quality_loss_pp_across_models'], c['variant_id']))


def select_profiles(reports_by_model, *, cost_tie_fraction=.01):
    """Select one profile per form on BOTH models; no eligible means diagnostics.

Shared nested fold choices below see only inner summaries, not outer-fold
scores. Main selected profiles are explicitly development-selected, not an
independent estimate of model-selection success.
"""
    _cost_tie(cost_tie_fraction)
    if len(reports_by_model) != 2:
        raise ValueError('exactly both model reports required')
    models = sorted(reports_by_model)
    for model, report in reports_by_model.items():
        if (report.get('format') != FORMAT + '_nested_development' or report.get('model') != model or
            report.get('sha256') != digest({k: v for k, v in report.items() if k != 'sha256'})):
            raise ValueError('development report integrity mismatch')
        if set(report['procedures']) != set(VARIANTS):
            raise ValueError('all six frozen variants required')
        if report.get('evaluation_constraints') != EVALUATION_CONSTRAINTS:
            raise ValueError('evaluation constraints changed')
        if set(report['final_rules']) != set(VARIANTS):
            raise ValueError('all six final rules required')
        for variant_id, rule in report['final_rules'].items():
            variant = _variant(variant_id)
            if (rule.get('format') != previous.FORMAT or rule.get('model') != model or
                rule.get('kind') != variant['kind'] or
                any(rule.get(key) != value for key, value in variant['fit_constraints'].items()) or
                rule.get('sha256') != digest({k: v for k, v in rule.items() if k != 'sha256'})):
                raise ValueError('final rule does not match sealed model and fitting profile')
        if ([fold.get('fold') for fold in report['folds']] != list(range(report['outer_folds'])) or
            any(len(fold['inner_candidates']) != len(VARIANTS) or
                {c['variant_id'] for c in fold['inner_candidates']} != set(VARIANTS)
                for fold in report['folds'])):
            raise ValueError('complete ordered outer folds and inner variants required')
    comparisons = {v: _comparison(v, {m: reports_by_model[m]['procedures'][v]['summary']
                                      for m in models}) for v in VARIANTS}
    forms = {}
    for form, kind in FORMS.items():
        winner = _choose_shared(comparisons, kind, cost_tie_fraction)
        variant_id = winner['variant_id'] if winner else kind + '__original'
        forms[form] = {'variant_id': variant_id, 'kind': kind,
            'promotion_eligible': winner is not None, 'diagnostic_only': winner is None,
            'reason': 'minimum_memory_feasible_shared_profile' if winner else 'no_eligible_profile_original_for_diagnostics',
            'models': {m: {'rule': reports_by_model[m]['final_rules'][variant_id],
                          'development_summary': comparisons[variant_id]['model_summaries'][m]}
                       for m in models}, 'comparison': comparisons[variant_id]}
    fold_counts = {r['outer_folds'] for r in reports_by_model.values()}
    if len(fold_counts) != 1:
        raise ValueError('shared nested selection requires matching outer-fold counts')
    nested = {form: {'folds': [], 'counts_by_model': {m: {} for m in models}}
              for form in FORMS}
    for fold in range(next(iter(fold_counts))):
        records = {m: reports_by_model[m]['folds'][fold] for m in models}
        training_groups = set().union(*(set(rec['training_groups']) for rec in records.values()))
        validation_groups = set().union(*(set(rec['validation_groups']) for rec in records.values()))
        if training_groups & validation_groups:
            raise ValueError('cross-model shared selection has training/validation group leakage')
        inner = {m: {c['variant_id']: c['summary'] for c in rec['inner_candidates']}
                 for m, rec in records.items()}
        local = {v: _comparison(v, {m: inner[m][v] for m in models}) for v in VARIANTS}
        for form, kind in FORMS.items():
            winner = _choose_shared(local, kind, cost_tie_fraction)
            variant_id = winner['variant_id'] if winner else None
            nested[form]['folds'].append({'fold': fold, 'variant_id': variant_id,
                'full_only_if_none': True, 'selection_labels': 'inner_validation_only',
                'comparisons': local})
            for model in models:
                output = (records[model]['candidate_assignments'][variant_id] if variant_id else
                          records[model]['full_assignments'])
                nested[form]['counts_by_model'][model].update(output['counts'])
    return _seal({'format': FORMAT + '_shared_profiles', 'forms': forms,
        'comparisons': comparisons, 'nested_shared': nested,
        'models': models, 'cost_tie_fraction': cost_tie_fraction,
        'evaluation_constraints': dict(EVALUATION_CONSTRAINTS),
        'main_profiles_evidence': 'post_selection_development',
        'nested_shared_evidence': 'shared_profile_selected_on_inner_folds_only',
        'nested_shared_summary_requires_original_rows': True,
        'independent_confirmation': False, 'runtime_promotion_gate_evaluated': False,
        'diagnostic_physical_runs_allowed_without_promotion': True})


def failure_diagnostics(rows, counts, *, bytes_per_token, generation_reserve=127):
    """Oracle recovery diagnosis, not a policy or a label-dependent deployment.

Each harmed request is checked at every physically measured higher count;
later quality drops after recovery are retained. Per-question MME diagnostics
are explicitly distinguished from the intact-pair/category gate metric.
"""
    rows = previous.validate_rows(rows, 'rank32_coverage')
    previous._configuration('diagnostic', bytes_per_token, generation_reserve, 1., 2.)
    previous._budget(rows, counts, bytes_per_token, generation_reserve)
    cases, all_risk = [], []
    for row in rows:
        key, n = row['row_hash'], row['input_visual_tokens']
        selected, curve = counts[key], row['curves']['scorer']
        if selected not in curve['counts'] or str(selected) not in row['quality_by_count']:
            raise ValueError('diagnostic assignment requires an exact physical curve count')
        quality, full = row['quality_by_count'][str(selected)], row['quality_by_count'][str(n)]
        risk = curve['uncovered_mass'][curve['counts'].index(selected)]
        harm = full - quality
        all_risk.append((risk, key, harm > 1e-12))
        if harm <= 1e-12:
            continue
        higher = [{'count': k, 'quality': row['quality_by_count'][str(k)],
                   'restores_full_score': row['quality_by_count'][str(k)] >= full - 1e-12,
                   'extra_visual_tokens': k - selected,
                   'extra_kv_bytes': (k - selected) * bytes_per_token}
                  for k in curve['counts'] if k > selected]
        recovery = next((x for x in higher if x['restores_full_score']), None)
        if recovery is None:
            raise AssertionError('full-token physical baseline must restore itself')
        increments = {}
        for percent in (10, 20):
            target = min(n, selected + math.ceil(n * percent / 100))
            endpoint = next(x for x in higher if x['count'] >= target)
            increments[str(percent)] = endpoint
        cases.append({'row_hash': key, 'group_id': row['group_id'], 'dataset': row['dataset'],
            'metric': row['quality_metric'], 'diagnostic_unit': 'request_not_MME_pair' if row['dataset'] == 'MME' else 'request',
            'selected_count': selected, 'full_count': n, 'selected_quality': quality,
            'full_quality': full, 'quality_loss_pp': 100 * harm, 'selected_coverage_risk': risk,
            'first_recovery': recovery, 'recovers_before_full': recovery['count'] < n,
            'higher_count_outcomes': higher, 'additional_retention_percentage_points': increments,
            'later_regression_after_recovery': any(not x['restores_full_score']
                for x in higher if x['count'] > recovery['count'])})
    domain = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        subset = [case for case in cases if case['dataset'] == dataset]
        domain[dataset] = {'harmed_requests': len(subset),
            'recover_before_full': sum(c['recovers_before_full'] for c in subset),
            'require_full_for_first_recovery': sum(not c['recovers_before_full'] for c in subset),
            'recover_with_additional_10pp': sum(c['additional_retention_percentage_points']['10']['restores_full_score'] for c in subset),
            'recover_with_additional_20pp': sum(c['additional_retention_percentage_points']['20']['restores_full_score'] for c in subset),
            'extra_kv_bytes_for_first_recovery': sum(c['first_recovery']['extra_kv_bytes'] for c in subset),
            'later_regression_after_recovery': sum(c['later_regression_after_recovery'] for c in subset)}
    oracle = dict(counts)
    for case in cases:
        oracle[case['row_hash']] = case['first_recovery']['count']
    ordered = sorted(all_risk)
    quartiles = []
    for quartile in range(4):
        subset = ordered[len(ordered) * quartile // 4:len(ordered) * (quartile + 1) // 4]
        quartiles.append({'quartile': quartile + 1, 'requests': len(subset),
            'harmed_requests': sum(x[2] for x in subset),
            'risk_min': min((x[0] for x in subset), default=None),
            'risk_max': max((x[0] for x in subset), default=None)})
    return _seal({'format': FORMAT + '_failure_diagnostics', 'requests': len(rows),
        'harmed_requests': len(cases), 'cases': cases, 'domains': domain,
        'coverage_risk_quartiles': quartiles,
        'current_summary': summary(rows, counts, bytes_per_token=bytes_per_token,
                                   generation_reserve=generation_reserve),
        'oracle_recovery_summary': summary(rows, oracle, bytes_per_token=bytes_per_token,
                                           generation_reserve=generation_reserve),
        'oracle_recovery_is_diagnostic_only': True, 'labels_used_for_diagnostics': True,
        'quality_monotonicity_assumed': False, 'independent_confirmation': False,
        'errors': [], 'evidence': 'answer-informed cached recovery, never deployable'})
