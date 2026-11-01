"""Nested, image-group development validation of answer-free stopping rules.

The selectable rule sees a coverage curve, never a benchmark name or answer.
Quality is used only inside training folds. These reused data remain development
data: neither nesting nor a bootstrap turns them into independent confirmation.
"""
from collections import defaultdict
import copy
import itertools
import math

import numpy as np

from .state import digest

FORMAT = 'mlsys_compression_policy_v11'
DATASETS = ('DocVQA', 'TextVQA', 'ChartQA_TEST', 'MME')
METRICS = {'DocVQA': 'anls', 'TextVQA': 'vqa', 'ChartQA_TEST': 'relaxed', 'MME': 'mme'}
SIMPLE = ('uniform', 'missing_mass', 'uncovered_mass')
FEATURES = ('uncovered_mass', 'missing_mass', 'removed_fraction')


def _invalid_curve(curve):
    """Native stopping curves use ``invalid`` and ``fallback``, not ``valid``."""
    return curve.get('invalid', False) or curve.get('fallback', False)


def grouped_folds(rows, folds, seed):
    if type(folds) is not int or folds < 2:
        raise ValueError('at least two folds required')
    groups, strata = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row['group_id']].append(row)
    if len(groups) < folds:
        raise ValueError('too few independent groups')
    for group, members in groups.items():
        stratum = tuple(sorted({(r['dataset'], r.get('category', '') if r['dataset'] == 'MME' else '')
                                for r in members}))
        strata[stratum].append(group)
    result, loads = {}, [0] * folds
    for stratum, members in sorted(strata.items()):
        order = sorted(members, key=lambda g: digest([seed, stratum, g]))
        offset = min(range(folds), key=lambda k: (loads[k], k))
        for i, group in enumerate(order):
            fold = (offset + i) % folds
            result[group] = fold
            loads[fold] += 1
    return result


def validate_rows(rows, selector, *, labels=True):
    if not rows or len({r['row_hash'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique request rows required')
    pairs = defaultdict(list)
    for row in rows:
        n = row['input_visual_tokens']
        if type(n) is not int or n < 1 or not row.get('group_id'):
            raise ValueError('invalid request geometry')
        if labels:
            if row.get('quality_metric') != METRICS.get(row.get('dataset')):
                raise ValueError('native dataset metric mismatch')
            q = row.get('quality_by_count', {})
            if str(n) not in q or any(type(v) not in (int, float) or not math.isfinite(v)
                                     or not 0 <= v <= 1 for v in q.values()):
                raise ValueError('finite physical quality and full baseline required')
            if row['dataset'] == 'MME':
                if not row.get('category') or any(v not in (0, 1) for v in q.values()):
                    raise ValueError('MME requires binary outcomes and category')
                pairs[row['group_id']].append(row)
        elif any(k in row for k in ('quality_by_count', 'quality_score', 'answer', 'answers', 'labels')):
            raise ValueError('inference must be answer-free')
        curve = row.get('curves', {}).get(selector)
        if curve is None:
            raise ValueError('missing selector curve')
        if any(name in curve and type(curve[name]) is not bool for name in ('invalid', 'fallback')):
            raise ValueError('curve validity flags must be booleans')
        if curve.get('input_visual_tokens') != n:
            raise ValueError('curve geometry mismatch')
        counts = curve.get('counts', [])
        if (not counts or any(type(k) is not int or not 1 <= k <= n for k in counts)
                or counts != sorted(set(counts)) or counts[-1] != n):
            raise ValueError('unique increasing physical counts including full required')
        if labels and any(str(k) not in row['quality_by_count'] for k in counts):
            raise ValueError('all curve counts require physical labels, no interpolation')
        for key in FEATURES:
            values = curve.get(key, [])
            if (len(values) != len(counts) or any(type(v) not in (int, float)
                    or not math.isfinite(v) or not -1e-6 <= v <= 1 + 1e-6 for v in values)
                    or abs(values[-1]) > 1e-6
                    or any(a + 1e-6 < b for a, b in zip(values, values[1:]))):
                raise ValueError('risk curves must be finite, nonincreasing and zero at full')
        if any(u > m + 1e-6 for u, m in zip(curve['uncovered_mass'], curve['missing_mass'])):
            raise ValueError('uncovered evidence cannot exceed removed mass')
        if any(abs(f - (1 - k / n)) > 1e-6 for k, f in zip(counts, curve['removed_fraction'])):
            raise ValueError('removed fraction must match physical counts')
    if any(len(p) != 2 or len({r['category'] for r in p}) != 1 for p in pairs.values()):
        raise ValueError('intact same-category MME image pairs required')
    return sorted(rows, key=lambda r: r['row_hash'])


def _units(rows, counts):
    """Native metric observations, clustered by image, including MME pair score."""
    result, pairs = [], defaultdict(list)
    for row in rows:
        selected = counts[row['row_hash']]
        full = row['quality_by_count'][str(row['input_visual_tokens'])]
        actual = row['quality_by_count'][str(selected)]
        if row['dataset'] == 'MME':
            pairs[(row['category'], row['group_id'])].append((full, actual))
        else:
            result.append((row['dataset'], '', row['group_id'], full, actual))
    for (category, group), pair in pairs.items():
        if len(pair) != 2:
            raise ValueError('incomplete MME pair')
        full, actual = zip(*pair)
        score = lambda x: .5 * float(np.mean(x)) + .5 * float(all(x))
        result.append(('MME', category, group, score(full), score(actual)))
    return result


def summary(rows, counts):
    if set(counts) != {r['row_hash'] for r in rows}:
        raise ValueError('assignment cohort mismatch')
    domains = {}
    units = _units(rows, counts)
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        obs = [r for r in units if r[0] == dataset]
        categories = {x[1] for x in obs}
        weights = np.array([1 / (len(categories) * sum(y[1] == x[1] for y in obs)) for x in obs])
        full, actual = np.array([x[3] for x in obs]), np.array([x[4] for x in obs])
        delta = actual - full
        tokens = sum(counts[r['row_hash']] for r in group)
        all_tokens = sum(r['input_visual_tokens'] for r in group)
        domains[dataset] = {'accuracy': float(weights @ actual), 'full_accuracy': float(weights @ full),
            'delta_full_pp': float(100 * (weights @ delta)),
            'positive_harm_pp': float(100 * (weights @ np.maximum(-delta, 0))),
            'helped_units': int((delta > 1e-12).sum()), 'harmed_units': int((delta < -1e-12).sum()),
            'unchanged_units': int((np.abs(delta) <= 1e-12).sum()),
            'requests': len(group), 'independent_groups': len({r['group_id'] for r in group}),
            'retained_visual_tokens': tokens, 'full_visual_tokens': all_tokens,
            'retained_fraction': tokens / all_tokens}
    return {'domains': domains, 'macro_accuracy': float(np.mean([d['accuracy'] for d in domains.values()])),
        'macro_delta_full_pp': float(np.mean([d['delta_full_pp'] for d in domains.values()])),
        'worst_delta_full_pp': min(d['delta_full_pp'] for d in domains.values()),
        'macro_positive_harm_pp': float(np.mean([d['positive_harm_pp'] for d in domains.values()])),
        'macro_retained_fraction': float(np.mean([d['retained_fraction'] for d in domains.values()])),
        'retained_visual_tokens': sum(counts.values()),
        'full_visual_tokens': sum(r['input_visual_tokens'] for r in rows)}


def _risk(curve, kind, coefficients=None):
    if kind in FEATURES:
        return np.asarray(curve[kind], dtype=float)
    if kind == 'nonnegative_linear':
        beta = np.asarray(coefficients, dtype=float)
        if beta.shape != (3,) or not np.isfinite(beta).all() or (beta < 0).any():
            raise ValueError('three finite nonnegative coefficients required')
        return np.asarray([curve[k] for k in FEATURES], dtype=float).T @ beta
    raise ValueError('unknown stopping statistic')


def _nnls(rows, selector):
    """No-intercept positive-question-harm fit, balanced by dataset/category.

    Three available cues, with a duplicated uniform-weight mass cue removed.
    MME labels are per-question loss here; native pair quality is used for every
    model-selection comparison rather than pretending this is a pair predictor.
    """
    features, targets, weights = [], [], []
    usable = [row for row in rows if not _invalid_curve(row['curves'][selector])]
    strata = defaultdict(int)
    for row in usable:
        strata[(row['dataset'], row.get('category', '') if row['dataset'] == 'MME' else '')] += 1
    datasets = {d for d, _ in strata}
    for row in usable:
        curve = row['curves'][selector]
        stratum = (row['dataset'], row.get('category', '') if row['dataset'] == 'MME' else '')
        categories = sum(d == row['dataset'] for d, _ in strata)
        full = row['quality_by_count'][str(row['input_visual_tokens'])]
        for j, count in enumerate(curve['counts']):
            features.append([curve[k][j] for k in FEATURES])
            targets.append(max(0., full - row['quality_by_count'][str(count)]))
            weights.append(1 / (len(datasets) * categories * strata[stratum] * len(curve['counts'])))
    if not features:
        return [0., 0., 0.]
    w = np.sqrt(weights)
    x, y = np.asarray(features) * w[:, None], np.asarray(targets) * w
    active_features = (0, 2) if np.allclose(x[:, 1], x[:, 2], rtol=0, atol=1e-12) else (0, 1, 2)
    best, best_error = np.zeros(3), float(y @ y)
    # No intercept: retaining every patch has exactly zero predicted harm.
    for width in range(1, len(active_features) + 1):
        for active in itertools.combinations(active_features, width):
            beta, *_ = np.linalg.lstsq(x[:, active], y, rcond=None)
            if (beta < -1e-12).any():
                continue
            candidate = np.zeros(3)
            candidate[list(active)] = np.maximum(beta, 0)
            error = float(np.square(x @ candidate - y).sum())
            if error < best_error - 1e-12:
                best, best_error = candidate, error
    return best.tolist()


def _assign(rows, selector, kind, threshold, target, coefficients=None, support=None):
    counts, fallbacks = {}, {}
    for row in rows:
        curve, n = row['curves'][selector], row['input_visual_tokens']
        reason = ((curve.get('fallback_reason') or 'invalid_curve') if _invalid_curve(curve) else
                  'visual_count_out_of_support' if support and not support[0] <= n <= support[1] else None)
        if reason:
            count = n
        elif kind == 'uniform':
            # Only physically measured menu counts are used in cached screening.
            eligible = [k for k in curve['counts'] if k <= math.ceil(n * target)]
            count = max(eligible) if eligible else curve['counts'][0]
        else:
            risk = _risk(curve, kind, coefficients)
            count = next((k for k, r in zip(curve['counts'], risk) if r <= threshold), n)
        counts[row['row_hash']] = int(count)
        if reason:
            fallbacks[row['row_hash']] = reason
    return counts, fallbacks


def fit_rule(rows, selector, kind, target):
    rows = validate_rows(rows, selector)
    if kind not in (*SIMPLE, 'nonnegative_linear') or target not in (.5, .7, .9):
        raise ValueError('undeclared rule or target')
    beta = _nnls(rows, selector) if kind == 'nonnegative_linear' else None
    threshold = None
    if kind != 'uniform':
        candidates = sorted({0., *(float(v) for r in rows for v in _risk(r['curves'][selector], kind, beta))})
        # Smallest threshold whose TRAINING macro token fraction fits the cap.
        # Binary search uses answer-free costs, not validation outcomes.
        left, right = 0, len(candidates) - 1
        while left < right:
            middle = (left + right) // 2
            assigned, _ = _assign(rows, selector, kind, candidates[middle], target, beta)
            if _fraction(rows, assigned) <= target + 1e-12:
                right = middle
            else:
                left = middle + 1
        threshold = candidates[left]
    counts, fallbacks = _assign(rows, selector, kind, threshold, target, beta)
    fitted_summary = summary(rows, counts)
    value = {'format': FORMAT, 'selector': selector, 'kind': kind, 'target_fraction': target,
        'threshold': threshold, 'coefficients': beta,
        'support_visual_tokens': [min(r['input_visual_tokens'] for r in rows), max(r['input_visual_tokens'] for r in rows)],
        'fit_group_ids': sorted({r['group_id'] for r in rows}),
        'fit_row_hashes': [r['row_hash'] for r in rows], 'fit_data_sha256': digest(rows),
        'fit_summary': fitted_summary, 'fit_fallbacks': fallbacks,
        'fit_allocation': allocation_summary(rows, counts, selector),
        'fit_target_feasible': fitted_summary['macro_retained_fraction'] <= target + 1e-12,
        'fit_budget_excess_fraction': max(0., fitted_summary['macro_retained_fraction'] - target),
        'tiny_fit_objective': 'positive per-question quality loss; equal dataset/category weight' if beta is not None else None,
        'dataset_is_policy_input': False, 'evidence': 'development only'}
    return {**value, 'sha256': digest(value)}


def _fraction(rows, counts):
    domains = {r['dataset'] for r in rows}
    return float(np.mean([sum(counts[r['row_hash']] for r in rows if r['dataset'] == d) /
                          sum(r['input_visual_tokens'] for r in rows if r['dataset'] == d) for d in domains]))


def allocation_summary(rows, counts, selector):
    """Separate genuine varying allocations from fixed-rule equivalents.

    Rounding makes actual fractions differ even under a single fixed percent.
    Therefore we report both, and test all integer percentages for exact
    ceil(N*percent/100) equivalence instead of calling rounding noise adaptive.
    This diagnostic consumes geometry and counts only, never quality labels.
    """
    if not rows or set(counts) != {r['row_hash'] for r in rows}:
        raise ValueError('allocation diagnostic cohort mismatch')
    compatible, fractions, positions, selected = set(range(1, 101)), [], set(), set()
    for row in rows:
        n, count = row['input_visual_tokens'], counts[row['row_hash']]
        if type(count) is not int or not 1 <= count <= n:
            raise ValueError('allocation diagnostic requires exact valid counts')
        compatible &= {percent for percent in range(1, 101) if (n * percent + 99) // 100 == count}
        fractions.append(count / n)
        selected.add(count)
        menu = row['curves'][selector]['counts']
        if count not in menu:
            raise ValueError('allocation diagnostic count missing from measured menu')
        positions.add(menu.index(count))
    kind = ('fixed_retention_equivalent' if compatible else
            'fixed_count_equivalent' if len(selected) == 1 else 'varying_allocation')
    return {'allocation_kind': kind, 'fixed_retention_compatible_percentages': sorted(compatible),
        'distinct_actual_retained_fractions': len(set(fractions)),
        'minimum_actual_retained_fraction': min(fractions), 'maximum_actual_retained_fraction': max(fractions),
        'distinct_selected_counts': len(selected), 'selected_menu_position_indices': sorted(positions),
        'distinct_menu_positions': len(positions),
        'fixed_equivalence_is_not_evidence_of_adaptive_gain': True}


def answer_free(rows):
    return [{k: copy.deepcopy(r[k]) for k in ('row_hash', 'group_id', 'input_visual_tokens', 'curves')} for r in rows]


def predict(rule, rows, *, allow_development_overlap=False):
    if rule.get('sha256') != digest({k: v for k, v in rule.items() if k != 'sha256'}) or rule.get('format') != FORMAT:
        raise ValueError('policy integrity mismatch')
    rows = validate_rows(rows, rule['selector'], labels=False)
    if not allow_development_overlap and any(r['group_id'] in rule['fit_group_ids'] or r['row_hash'] in rule['fit_row_hashes'] for r in rows):
        raise ValueError('training and prediction identities overlap')
    counts, fallbacks = _assign(rows, rule['selector'], rule['kind'], rule['threshold'],
        rule['target_fraction'], rule['coefficients'], rule['support_visual_tokens'])
    return {'counts': counts, 'fallbacks': fallbacks, 'policy_sha256': rule['sha256'],
            'answers_consumed': False, 'retained_visual_tokens': sum(counts.values()),
            'allocation': allocation_summary(rows, counts, rule['selector'])}


def _choose_candidate(candidates):
    """Quality-first, then compression-first on exact measured quality ties.

    Within two percentage points of the least-cost tied procedure, prefer
    simplicity.  Larger token savings are not discarded just to prefer a
    nominal uniform budget. No validation labels enter an individual prediction.
    """
    if not candidates:
        raise ValueError('candidate selection requires eligible procedures')
    best_accuracy = max(r['summary']['macro_accuracy'] for r in candidates)
    same_accuracy = [r for r in candidates if r['summary']['macro_accuracy'] == best_accuracy]
    best_worst = max(r['summary']['worst_delta_full_pp'] for r in same_accuracy)
    same_quality = [r for r in same_accuracy if r['summary']['worst_delta_full_pp'] == best_worst]
    least_cost = min(r['summary']['macro_retained_fraction'] for r in same_quality)
    near_least_cost = [r for r in same_quality if
                      r['summary']['macro_retained_fraction'] <= least_cost + .02 + 1e-12]
    complexity = {'uniform': 0, 'missing_mass': 1, 'uncovered_mass': 1, 'nonnegative_linear': 2}
    return min(near_least_cost, key=lambda r: (complexity[r['kind']],
        r['summary']['macro_positive_harm_pp'], r['kind']))


def nested_validate(rows, selector, target, *, outer_folds=5, inner_folds=3, seed=20261020, allow_tiny=True):
    rows = validate_rows(rows, selector)
    if {r['dataset'] for r in rows} != set(DATASETS):
        raise ValueError('the four development benchmarks are required')
    outer = grouped_folds(rows, outer_folds, seed)
    assigned, records = {}, []

    def choose(train, chosen_seed):
        partition = grouped_folds(train, inner_folds, chosen_seed)
        evaluated = []

        def evaluate(kind):
            predictions, inner_records = {}, []
            for fold in range(inner_folds):
                fit = [r for r in train if partition[r['group_id']] != fold]
                valid = [r for r in train if partition[r['group_id']] == fold]
                rule = fit_rule(fit, selector, kind, target)
                output = predict(rule, answer_free(valid))
                predictions.update(output['counts'])
                inner_records.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
                    'validation_groups': sorted({r['group_id'] for r in valid}),
                    'rule_sha256': rule['sha256'], 'threshold': rule['threshold'], 'coefficients': rule['coefficients']})
            evaluated.append({'kind': kind, 'summary': summary(train, predictions), 'folds': inner_records})

        for kind in SIMPLE:
            evaluate(kind)
        fixed = evaluated[0]['summary']['macro_accuracy']
        reference_cost = evaluated[0]['summary']['macro_retained_fraction']
        eligible_simple = [r for r in evaluated[1:] if
                           r['summary']['macro_retained_fraction'] <= reference_cost + .02 + 1e-12]
        # A learned model is considered only if simple adaptive rules cannot
        # improve inner validation accuracy within the allowed token cost.
        # An apparent gain obtained by overspending must not suppress this test.
        if allow_tiny and (not eligible_simple or
                           max(r['summary']['macro_accuracy'] for r in eligible_simple) <= fixed + 1e-12):
            evaluate('nonnegative_linear')
        # Validation budget drift is reported, not silently reconciled with labels.
        # Prefer procedures within the same 2%-of-full token allowance, or the
        # uniform rule if all adaptive procedures spend more than this allowance.
        eligible = [r for r in evaluated if r['kind'] == 'uniform' or
                    r['summary']['macro_retained_fraction'] <= reference_cost + .02 + 1e-12]
        winner = _choose_candidate(eligible)
        return winner['kind'], evaluated

    for fold in range(outer_folds):
        train = [r for r in rows if outer[r['group_id']] != fold]
        valid = [r for r in rows if outer[r['group_id']] == fold]
        kind, inner = choose(train, seed + fold + 1)
        rule = fit_rule(train, selector, kind, target)
        output = predict(rule, answer_free(valid))
        assigned.update(output['counts'])
        records.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
            'validation_groups': sorted({r['group_id'] for r in valid}), 'rule': rule,
            'inner_candidates': inner, 'validation_assignments': output,
            'validation_summary': summary(valid, output['counts'])})
    final_kind, final_candidates = choose(rows, seed + 100)
    final_rule = fit_rule(rows, selector, final_kind, target)
    value = {'format': FORMAT + '_nested_development', 'selector': selector, 'target_fraction': target,
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seed': seed, 'folds': records,
        'counts': assigned, 'summary': summary(rows, assigned), 'final_rule': final_rule,
        'allocation': allocation_summary(rows, assigned, selector),
        'final_inner_candidates': final_candidates,
        'selection_rule': 'macro accuracy, worst domain, within 2pp of least token cost, simplicity, positive harm, lexical',
        'not_exact_budget_matched': True, 'physical_matched_comparison_required': True,
        'two_pp_full_loss_is_reference_not_gate': True,
        'independent_confirmation': False, 'evidence': 'nested development on previously used data'}
    return {**value, 'sha256': digest(value)}


def factorial_assignments(rows, adaptive_counts, alternative):
    """ONE adaptive and ONE exactly matched uniform vector, shared by selectors.

    This produces schedules, not interpolated quality estimates. Any unmeasured
    integer count MUST go through physical inference before it can be reported.
    """
    if alternative not in ('divprune', 'cdpruner') or set(adaptive_counts) != {r['row_hash'] for r in rows}:
        raise ValueError('unsupported alternative or incomplete counts')
    if any(type(adaptive_counts[r['row_hash']]) is not int or
           not 1 <= adaptive_counts[r['row_hash']] <= r['input_visual_tokens'] for r in rows):
        raise ValueError('factorial counts must be exact in-range positive integers')
    from .compression_policy_v8 import uniform_at_visual_total
    uniform = {}
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        total = sum(adaptive_counts[r['row_hash']] for r in group)
        geometry = [{k: r[k] for k in ('row_hash', 'group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')} for r in group]
        result = uniform_at_visual_total(geometry, total)
        uniform.update(result['counts'])
        if sum(result['counts'].values()) != total:
            raise AssertionError('uniform reconciliation lost tokens')
    return {'counts': {'adaptive': dict(adaptive_counts), 'uniform': uniform},
        'cells': {f'{selector}_{form}': {'selector': selector, 'counts': dict(counts)}
                  for selector in ('scorer', alternative) for form, counts in (('adaptive', adaptive_counts), ('uniform', uniform))},
        'exact_aggregate_matched_within_every_dataset': True, 'quality_requires_physical_inference': True}
