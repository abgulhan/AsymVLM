"""Memory-first calibration of four simple, answer-free vision budgets.

Physical native-metric outcomes constrain calibration; no quality monotonicity
or interpolation is assumed. All threshold breakpoints are swept exactly. The
image scorer and selected token sets are unchanged: native coverage changes
only the geometry used to decide where to stop. Development validation is not
independent confirmation, and a fitted quality constraint is not a guarantee.
"""

from collections import Counter, defaultdict
import copy
import math

from . import compression_policy_v11 as previous
from .state import digest

FORMAT = 'mlsys_compression_policy_v12'
DATASETS = previous.DATASETS
METRICS = previous.METRICS
KINDS = ('static', 'missing_mass', 'rank32_coverage', 'native_coverage')
RETENTIONS = tuple(range(10, 101, 10))
grouped_folds = previous.grouped_folds


def _selector(kind):
    if kind not in KINDS:
        raise ValueError('undeclared allocation rule')
    return 'native' if kind == 'native_coverage' else 'scorer'


def _configuration(model, bytes_per_token, generation_reserve, mean_loss, domain_loss):
    if not isinstance(model, str) or not model:
        raise ValueError('one explicit model identifier required')
    if type(bytes_per_token) is not int or bytes_per_token < 1:
        raise ValueError('positive integer bytes_per_token required')
    if type(generation_reserve) is not int or generation_reserve < 0:
        raise ValueError('nonnegative integer generated-token reserve required')
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0
           for v in (mean_loss, domain_loss)):
        raise ValueError('finite nonnegative quality-loss constraints required')


def validate_rows(rows, kind, *, labels=True):
    selector = _selector(kind)
    rows = previous.validate_rows(rows, selector, labels=labels)
    if kind == 'native_coverage':
        previous.validate_rows(rows, 'scorer', labels=labels)
    if not labels and any('dataset' in r or 'quality_metric' in r or 'category' in r for r in rows):
        raise ValueError('inference must not consume benchmark identity')
    for row in rows:
        if type(row.get('input_text_tokens')) is not int or row['input_text_tokens'] < 0:
            raise ValueError('protected prefix text/structural token count required')
        if kind != 'native_coverage':
            continue
        old, new = row['curves'].get('scorer'), row['curves']['native']
        if old is None:
            raise ValueError('native coverage requires the unchanged scorer curve')
        if previous._invalid_curve(old) or previous._invalid_curve(new):
            continue
        if old['counts'] != new['counts'] or any(abs(a-b) > 1e-8 for a, b in
                zip(old['missing_mass'], new['missing_mass'])):
            raise ValueError('native coverage changed counts or importance weights')
        for field in ('selected_indices_sha256', 'ranking_sha256', 'weights_sha256'):
            if field in old or field in new:
                if old.get(field) != new.get(field):
                    raise ValueError('native coverage changed selected tokens or weights: ' + field)
    return rows


def answer_free(rows):
    """Deliberately drop labels AND benchmark identity from deployed inputs."""
    keys = ('row_hash', 'group_id', 'input_visual_tokens', 'input_text_tokens', 'curves')
    return [{k: copy.deepcopy(row[k]) for k in keys} for row in rows]


def _budget(rows, counts, bytes_per_token, generation_reserve):
    if set(counts) != {r['row_hash'] for r in rows}:
        raise ValueError('assignment cohort mismatch')
    if any(type(counts[r['row_hash']]) is not int or not
           1 <= counts[r['row_hash']] <= r['input_visual_tokens'] for r in rows):
        raise ValueError('exact positive in-range counts required')
    total = sum(r['input_text_tokens'] + counts[r['row_hash']] + generation_reserve for r in rows)
    full = sum(r['input_text_tokens'] + r['input_visual_tokens'] + generation_reserve for r in rows)
    saved = sum(counts[r['row_hash']] < r['input_visual_tokens'] for r in rows)
    return {'allocated_kv_slots': total, 'full_allocated_kv_slots': full,
        'allocated_kv_bytes': total * bytes_per_token,
        'full_allocated_kv_bytes': full * bytes_per_token,
        'mean_allocated_kv_bytes': total * bytes_per_token / len(rows),
        'mean_allocated_kv_mib': total * bytes_per_token / len(rows) / 2**20,
        'mean_full_allocated_kv_bytes': full * bytes_per_token / len(rows),
        'kv_reduction_percent': 100 * (1 - total / full),
        'requests_with_memory_reduction': saved,
        'percent_requests_with_memory_reduction': 100 * saved / len(rows),
        'bytes_per_token': bytes_per_token, 'generation_reserve': generation_reserve,
        'cache_scope': 'allocated_key_value_capacity_excluding_weights_activations_and_auxiliary_memory'}


def summary(rows, counts, *, bytes_per_token, generation_reserve=127):
    result = previous.summary(rows, counts)
    result.update(_budget(rows, counts, bytes_per_token, generation_reserve))
    result['mean_quality_loss_pp'] = -result['macro_delta_full_pp']
    result['worst_domain_quality_loss_pp'] = -result['worst_delta_full_pp']
    for dataset, values in result['domains'].items():
        group = [r for r in rows if r['dataset'] == dataset]
        values.update(_budget(group, {r['row_hash']: counts[r['row_hash']] for r in group},
                              bytes_per_token, generation_reserve))
        values['quality_loss_pp'] = -values['delta_full_pp']
    return result


def meets_constraints(report, max_mean_loss_pp=1., max_domain_loss_pp=2.):
    return (report['mean_quality_loss_pp'] <= max_mean_loss_pp + 1e-9 and
            report['worst_domain_quality_loss_pp'] <= max_domain_loss_pp + 1e-9)


def _risk(curve, kind):
    return curve['missing_mass' if kind == 'missing_mass' else 'uncovered_mass']


def _assign(rows, kind, threshold, retention_percent, support=None):
    selector, counts, fallbacks = _selector(kind), {}, {}
    for row in rows:
        curve, n = row['curves'][selector], row['input_visual_tokens']
        bad_ranker = kind == 'native_coverage' and previous._invalid_curve(row['curves']['scorer'])
        reason = ('invalid_frozen_selector' if bad_ranker else
                  (curve.get('fallback_reason') or 'invalid_curve')
                  if previous._invalid_curve(curve) else
                  'visual_count_out_of_support' if support and not support[0] <= n <= support[1] else None)
        if reason:
            count = n
        elif kind == 'static':
            cap = (n * retention_percent + 99) // 100
            count = max((k for k in curve['counts'] if k <= cap), default=curve['counts'][0])
        else:
            count = next((k for k, risk in zip(curve['counts'], _risk(curve, kind)) if risk <= threshold), n)
        counts[row['row_hash']] = count
        if reason:
            fallbacks[row['row_hash']] = reason
    return counts, fallbacks


class _Sweep:
    """Increment native quality and memory at each threshold event, O(NK log NK).

    MME is scored jointly per image pair and equally across categories. Other
    benchmarks use native mean per-question scores. Quality may go up or down
    on any event: every distinct threshold is checked, without binary search.
    """

    def __init__(self, rows):
        self.rows = {r['row_hash']: r for r in rows}
        self.counts = {r['row_hash']: r['input_visual_tokens'] for r in rows}
        self.quality = {r['row_hash']: r['quality_by_count'][str(r['input_visual_tokens'])] for r in rows}
        self.tokens = sum(self.counts.values())
        domains = sorted({r['dataset'] for r in rows})
        self.delta = {d: 0. for d in domains}
        sizes = Counter(r['dataset'] for r in rows)
        pairs = defaultdict(list)
        for row in rows:
            if row['dataset'] == 'MME':
                pairs[row['group_id']].append(row['row_hash'])
        categories = Counter(self.rows[pair[0]]['category'] for pair in pairs.values())
        self.members, self.weights = {}, {}
        for row in rows:
            key = row['row_hash']
            if row['dataset'] == 'MME':
                self.members[key] = pairs[row['group_id']]
                self.weights[key] = 1 / (len(categories) * categories[row['category']])
            else:
                self.members[key] = [key]
                self.weights[key] = 1 / sizes[row['dataset']]

    def _score(self, row_hash):
        q = [self.quality[key] for key in self.members[row_hash]]
        return .25 * sum(q) + .5 * float(all(q)) if len(q) == 2 else q[0]

    def update(self, row_hash, count):
        if count >= self.counts[row_hash]:
            return
        old = self._score(row_hash)
        self.tokens += count - self.counts[row_hash]
        self.counts[row_hash] = count
        self.quality[row_hash] = self.rows[row_hash]['quality_by_count'][str(count)]
        self.delta[self.rows[row_hash]['dataset']] += self.weights[row_hash] * (self._score(row_hash) - old)

    def loss(self):
        return (-100 * sum(self.delta.values()) / len(self.delta), -100 * min(self.delta.values()))


def _best_threshold(rows, kind, mean_limit, domain_limit):
    selector = _selector(kind)
    events = []
    for row in rows:
        curve = row['curves'][selector]
        bad_ranker = kind == 'native_coverage' and previous._invalid_curve(row['curves']['scorer'])
        if not previous._invalid_curve(curve) and not bad_ranker:
            events.extend((float(risk), row['row_hash'], count)
                          for count, risk in zip(curve['counts'], _risk(curve, kind)))
    events.sort()
    state = _Sweep(rows)
    # A negative threshold is a deployable full-only endpoint, even if some
    # non-full physical count has exactly zero proxy risk.
    best = (state.tokens, 0., 0., -1.)
    index, evaluated = 0, 1
    while index < len(events):
        threshold = events[index][0]
        while index < len(events) and events[index][0] == threshold:
            _, key, count = events[index]
            state.update(key, count)
            index += 1
        mean, worst = state.loss()
        evaluated += 1
        if mean <= mean_limit + 1e-9 and worst <= domain_limit + 1e-9:
            candidate = (state.tokens, mean, worst, threshold)
            if candidate < best:
                best = candidate
    return best[-1], evaluated


def fit_rule(rows, kind, *, model, bytes_per_token, max_mean_loss_pp=1.,
             max_domain_loss_pp=2., generation_reserve=127, _force_full=False):
    """Fit one scalar threshold to minimum actual allocated KV within quality limits.

    Prefix and reserved output slots are fixed by each request. Thus minimizing
    the aggregate retained visual count exactly minimizes aggregate allocated
    KV for this single model; it is NOT a mean of per-request retention ratios.
    """
    _configuration(model, bytes_per_token, generation_reserve, max_mean_loss_pp, max_domain_loss_pp)
    rows = validate_rows(rows, kind)
    threshold, percent, evaluated = None, None, 1
    if kind == 'static':
        candidates = []
        for retention in ((100,) if _force_full else RETENTIONS):
            counts, _ = _assign(rows, kind, None, retention)
            result = summary(rows, counts, bytes_per_token=bytes_per_token, generation_reserve=generation_reserve)
            if meets_constraints(result, max_mean_loss_pp, max_domain_loss_pp):
                candidates.append((result['allocated_kv_bytes'], result['mean_quality_loss_pp'],
                                   result['worst_domain_quality_loss_pp'], retention))
        percent = min(candidates)[-1]
        evaluated = 1 if _force_full else len(RETENTIONS)
    elif _force_full:
        threshold = -1.
    else:
        threshold, evaluated = _best_threshold(rows, kind, max_mean_loss_pp, max_domain_loss_pp)
    counts, fallbacks = _assign(rows, kind, threshold, percent)
    report = summary(rows, counts, bytes_per_token=bytes_per_token, generation_reserve=generation_reserve)
    value = {'format': FORMAT, 'model': model, 'kind': kind, 'selector': _selector(kind),
        'threshold': threshold, 'retention_percent': percent,
        'bytes_per_token': bytes_per_token, 'generation_reserve': generation_reserve,
        'max_mean_loss_pp': max_mean_loss_pp, 'max_domain_loss_pp': max_domain_loss_pp,
        'objective': 'minimum_mean_allocated_kv_subject_to_native_quality_constraints',
        'fit_summary': report, 'fit_constraints_satisfied': meets_constraints(report, max_mean_loss_pp, max_domain_loss_pp),
        'fit_allocation': previous.allocation_summary(rows, counts, _selector(kind)),
        'fit_fallbacks': fallbacks, 'fit_group_ids': sorted({r['group_id'] for r in rows}),
        'fit_row_hashes': [r['row_hash'] for r in rows], 'fit_data_sha256': digest(rows),
        'support_visual_tokens': [min(r['input_visual_tokens'] for r in rows), max(r['input_visual_tokens'] for r in rows)],
        'evaluated_thresholds': evaluated, 'full_only': all(counts[r['row_hash']] == r['input_visual_tokens'] for r in rows),
        'quality_monotonicity_assumed': False, 'dataset_is_policy_input': False,
        'training_constraints_are_not_quality_guarantees': True, 'evidence': 'development only'}
    return {**value, 'sha256': digest(value)}


def infer(rule, rows, *, allow_development_overlap=False):
    if rule.get('format') != FORMAT or rule.get('sha256') != digest({k: v for k, v in rule.items() if k != 'sha256'}):
        raise ValueError('policy integrity mismatch')
    rows = validate_rows(rows, rule['kind'], labels=False)
    if not allow_development_overlap and any(r['row_hash'] in rule['fit_row_hashes'] or
            r['group_id'] in rule['fit_group_ids'] for r in rows):
        raise ValueError('training and prediction identities overlap')
    counts, fallbacks = _assign(rows, rule['kind'], rule['threshold'], rule['retention_percent'], rule['support_visual_tokens'])
    return {'counts': counts, 'fallbacks': fallbacks, 'policy_sha256': rule['sha256'],
        'answers_consumed': False, 'dataset_consumed': False,
        'allocation': previous.allocation_summary(rows, counts, rule['selector']),
        'budget': _budget(rows, counts, rule['bytes_per_token'], rule['generation_reserve'])}


predict = infer


def choose_candidate(candidates, *, max_mean_loss_pp=1., max_domain_loss_pp=2., cost_tie_fraction=.01):
    """Feasible minimum-memory procedure, preferring simpler rules within 1%."""
    eligible = [c for c in candidates if meets_constraints(c['summary'], max_mean_loss_pp, max_domain_loss_pp)]
    if not eligible:
        return None
    least = min(c['summary']['mean_allocated_kv_bytes'] for c in eligible)
    near = [c for c in eligible if c['summary']['mean_allocated_kv_bytes'] <= least * (1 + cost_tie_fraction) + 1e-8]
    return min(near, key=lambda c: (KINDS.index(c['kind']), c['summary']['mean_quality_loss_pp'], c['kind']))


def development_report(rows, *, model, bytes_per_token, max_mean_loss_pp=1., max_domain_loss_pp=2.,
                       generation_reserve=127, outer_folds=5, inner_folds=3, seed=20261020):
    """All four OOF comparisons and nested procedure selection, never confirmation.

    The caller freezes one shared definition across models using these reports
    and the independently measured <=5% runtime gate. Numeric thresholds are
    model-specific. This function neither reads confirmation nor starts jobs.
    """
    for kind in KINDS:
        rows = validate_rows(rows, kind)
    if {r['dataset'] for r in rows} != set(DATASETS):
        raise ValueError('all four development benchmarks required')
    config = dict(model=model, bytes_per_token=bytes_per_token, max_mean_loss_pp=max_mean_loss_pp,
                  max_domain_loss_pp=max_domain_loss_pp, generation_reserve=generation_reserve)
    summarise = lambda data, counts: summary(data, counts, bytes_per_token=bytes_per_token, generation_reserve=generation_reserve)
    outer = grouped_folds(rows, outer_folds, seed)
    selected, candidate_counts, records = {}, {kind: {} for kind in KINDS}, []

    def choose(train, local_seed):
        partition = grouped_folds(train, inner_folds, local_seed)
        candidates = []
        for kind in KINDS:
            counts, folds = {}, []
            for fold in range(inner_folds):
                fit = [r for r in train if partition[r['group_id']] != fold]
                valid = [r for r in train if partition[r['group_id']] == fold]
                rule = fit_rule(fit, kind, **config)
                output = infer(rule, answer_free(valid))
                counts.update(output['counts'])
                folds.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
                    'validation_groups': sorted({r['group_id'] for r in valid}),
                    'policy_sha256': rule['sha256'], 'threshold': rule['threshold'],
                    'retention_percent': rule['retention_percent'], 'fallbacks': output['fallbacks']})
            candidates.append({'kind': kind, 'summary': summarise(train, counts), 'folds': folds})
        winner = choose_candidate(candidates, max_mean_loss_pp=max_mean_loss_pp, max_domain_loss_pp=max_domain_loss_pp)
        return (winner['kind'] if winner else None), candidates

    for fold in range(outer_folds):
        train = [r for r in rows if outer[r['group_id']] != fold]
        valid = [r for r in rows if outer[r['group_id']] == fold]
        winner, inner = choose(train, seed + fold + 1)
        outputs, rules = {}, {}
        for kind in KINDS:
            rules[kind] = fit_rule(train, kind, **config)
            outputs[kind] = infer(rules[kind], answer_free(valid))
            candidate_counts[kind].update(outputs[kind]['counts'])
        rule = rules[winner] if winner else fit_rule(train, 'static', **config, _force_full=True)
        output = outputs[winner] if winner else infer(rule, answer_free(valid))
        selected.update(output['counts'])
        records.append({'fold': fold, 'training_groups': rule['fit_group_ids'],
            'validation_groups': sorted({r['group_id'] for r in valid}),
            'selected_kind': winner or 'full', 'rule': rule, 'inner_candidates': inner,
            'validation_assignments': output, 'validation_summary': summarise(valid, output['counts']),
            'candidate_rules': rules, 'candidate_assignments': outputs})
    final_kind, inner = choose(rows, seed + 100)
    final_rules = {kind: fit_rule(rows, kind, **config) for kind in KINDS}
    final_rule = final_rules[final_kind] if final_kind else fit_rule(rows, 'static', **config, _force_full=True)
    candidates = {kind: {'kind': kind, 'counts': counts, 'summary': summarise(rows, counts),
        'allocation': previous.allocation_summary(rows, counts, _selector(kind))}
        for kind, counts in candidate_counts.items()}
    value = {'format': FORMAT + '_nested_development', 'model': model, 'configuration': config,
        'outer_folds': outer_folds, 'inner_folds': inner_folds, 'seed': seed, 'folds': records,
        'candidates': candidates, 'procedures': candidates, 'counts': selected, 'summary': summarise(rows, selected),
        'allocation': previous.allocation_summary(rows, selected, 'scorer'),
        'final_kind': final_kind or 'full', 'final_rule': final_rule, 'final_rules': final_rules,
        'final_inner_candidates': inner, 'independent_confirmation': False,
        'selection_rule': 'quality constraints, minimum allocated KV, simpler within 1% relative cost',
        'runtime_promotion_gate_evaluated': False, 'shared_cross_model_choice_required': True,
        'not_exact_budget_matched': True, 'physical_matched_comparison_required': True,
        'evidence': 'nested development on previously used data'}
    return {**value, 'sha256': digest(value)}


def select_shared(reports_by_model, *, cost_tie_fraction=.01, allowed_kinds=KINDS):
    """Select one statistic, separately calibrated, conservatively across models.

    Compare equal-model mean allocated bytes, not an unweighted average of
    per-request patch percentages. Native geometry must not worsen either
    model's point quality loss versus the best eligible existing definition.
    A native choice is only provisional until measured runtime overhead is
    verified outside this CPU-only calibration module.
    """
    if type(cost_tie_fraction) not in (int, float) or not 0 <= cost_tie_fraction < 1:
        raise ValueError('relative cost tie must be in [0,1)')
    if not allowed_kinds or not set(allowed_kinds) <= set(KINDS):
        raise ValueError('declared nonempty allowed rule set required')
    if len(reports_by_model) < 2:
        return {'rule_kind': None, 'models': {}, 'reason': 'both_model_reports_required',
                'provisional_pending_overhead': False, 'independent_confirmation': False}
    models = sorted(reports_by_model)
    eligible, comparisons = [], {}
    for kind in KINDS:
        summaries = {model: reports_by_model[model]['procedures'][kind]['summary'] for model in models}
        feasible = all(meets_constraints(values,
            reports_by_model[model]['configuration']['max_mean_loss_pp'],
            reports_by_model[model]['configuration']['max_domain_loss_pp']) for model, values in summaries.items())
        comparisons[kind] = {'feasible_on_all_models': feasible,
            'mean_allocated_kv_bytes_across_models': sum(s['mean_allocated_kv_bytes'] for s in summaries.values()) / len(models),
            'model_summaries': summaries}
        if feasible and kind in allowed_kinds:
            eligible.append(kind)

    def choose(kinds):
        if not kinds:
            return None
        least = min(comparisons[k]['mean_allocated_kv_bytes_across_models'] for k in kinds)
        return min((k for k in kinds if comparisons[k]['mean_allocated_kv_bytes_across_models'] <=
                    least * (1 + cost_tie_fraction) + 1e-8), key=KINDS.index)

    existing = choose([kind for kind in eligible if kind != 'native_coverage'])
    native = comparisons['native_coverage']
    native_reason = None
    if 'native_coverage' in eligible:
        # If every existing rule misses the quality constraints, use current
        # rank32 coverage as the conservative non-regression reference.
        reference = existing or 'rank32_coverage'
        if any(native['model_summaries'][model]['mean_quality_loss_pp'] >
               comparisons[reference]['model_summaries'][model]['mean_quality_loss_pp'] + 1e-9 for model in models):
            native_reason = 'native_increases_quality_loss_on_at_least_one_model'
        elif native['mean_allocated_kv_bytes_across_models'] >= comparisons[reference]['mean_allocated_kv_bytes_across_models'] - 1e-8:
            native_reason = 'native_does_not_reduce_mean_memory'
        if native_reason:
            eligible.remove('native_coverage')
    winner = choose(eligible)
    if winner is None:
        return {'rule_kind': None, 'models': {}, 'reason': 'no_shared_definition_passes_development_constraints',
            'comparisons': comparisons, 'native_rejection_reason': native_reason,
            'provisional_pending_overhead': False, 'independent_confirmation': False,
            'safe_action': 'all_tokens_until_a_supported_method_is_available'}
    return {'rule_kind': winner,
        'models': {model: {'rule': reports_by_model[model]['final_rules'][winner],
                           'development_summary': comparisons[winner]['model_summaries'][model]} for model in models},
        'reason': 'minimum_mean_allocated_kv_feasible_on_both_models_simpler_within_relative_tie',
        'comparisons': comparisons, 'native_rejection_reason': native_reason,
        'cost_tie_fraction': cost_tie_fraction, 'provisional_pending_overhead': winner == 'native_coverage',
        'runtime_overhead_limit_fraction': .05, 'runtime_promotion_gate_evaluated': False,
        'independent_confirmation': False}


def matched_static_assignments(rows, adaptive_counts, *, bytes_per_token, generation_reserve=127):
    """Exact per-dataset KV controls; new integer counts require physical inference."""
    result = previous.factorial_assignments(rows, adaptive_counts, 'divprune')
    for dataset in sorted({r['dataset'] for r in rows}):
        group = [r for r in rows if r['dataset'] == dataset]
        budgets = [_budget(group, {r['row_hash']: counts[r['row_hash']] for r in group},
                          bytes_per_token, generation_reserve)['allocated_kv_bytes']
                   for counts in result['counts'].values()]
        if len(set(budgets)) != 1:
            raise AssertionError('matched controls lost exact allocated KV equality')
    return {**result, 'exact_allocated_kv_matched_within_every_dataset': True,
        'bytes_per_token': bytes_per_token, 'generation_reserve': generation_reserve}
