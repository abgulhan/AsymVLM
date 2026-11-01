"""Live-policy timing, including feature computation, with identical-count replay.

The replay arm uses exactly the live policy's selected counts and frozen scorer,
but skips budget calculation. Its quality/output parity isolates real budgeting
overhead from the latency benefit of processing fewer image tokens. Replay is
an overhead control, not a deployable adaptive method.
"""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import time
from types import SimpleNamespace

from . import compression_systems_v11 as old
from . import compression_features_v12 as features
from . import compression_policy_v12 as policy
from .compression_matrix_v8 import checked, seal
from .compression_study_v12 import verify
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_systems_v12'


def source_hashes(adapter):
    return {**old.source_hashes(adapter), **{
        'mlsys_sprint/' + Path(module.__file__).name: file_digest(module.__file__)
        for module in (features, policy)},
        'mlsys_sprint/compression_systems_v12.py': file_digest(__file__)}


def live_curves(vision, question, scorer, kind):
    """Compute only the deployed statistic, never the development ablation set."""
    import torch
    from torch.nn import functional as F
    from . import compression_stopping_v11 as stopping
    n = len(vision)
    counts = sorted({(n * p + 99) // 100 for p in features.RETENTIONS})
    nonfull = [k for k in counts if k < n]
    try:
        scores = scorer(vision, question).float()
        order = stopping.selector_order('asymvlm', scores=scores,
            keep_count=max(nonfull)) if nonfull else torch.empty(0, dtype=torch.long, device=vision.device)
        if kind in ('native_coverage', 'rank32_coverage'):
            geometry = vision if kind == 'native_coverage' else F.linear(
                F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
            curve = stopping.stopping_curve(visual=geometry, order=order, counts=counts,
                scores=scores, weight_mode='learned_scores', selector='asymvlm', geometry=kind)
        else:
            # Static needs ranking only; mass needs the same frozen weighting,
            # but neither needs pairwise feature similarities.
            weights = stopping.learned_score_weights(scores)
            missing = [0. if k == n else float(1 - weights[order[:k]].sum()) for k in counts]
            missing = [min(1., max(0., v)) for v in missing]
            curve = {'input_visual_tokens': n, 'counts': counts, 'missing_mass': missing,
                'uncovered_mass': missing, 'removed_fraction': [1-k/n for k in counts],
                'invalid': False, 'fallback': False, 'valid': True}
        curve.update(selector='scorer', valid=True, physical_prefix_parity=True)
    except (ValueError, RuntimeError) as exc:
        if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
            raise
        curve = features.previous._fallback(n, 'scorer', f'{type(exc).__name__}: {exc}')
        order = torch.empty(0, dtype=torch.long, device=vision.device)
    # Native validation checks only that ranking, counts and importance weights
    # are unchanged. No second rank32 coverage calculation is necessary online.
    return {'scorer': curve, 'native': curve}, order


def timed_selector(base, scorer, feature, cell, observed):
    rule = cell.get('budget_policy')
    if rule is None:
        return base

    def select(prepared):
        import torch
        from .compression_stopping_v11 import selected_positions
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        if (len(vision) != feature['input_visual_tokens'] or
                prepared.inputs_embeds.shape[1]-len(vision) != feature['input_text_tokens']):
            raise ValueError('live policy geometry changed')
        with torch.no_grad():
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            started = time.perf_counter()
            curves, order = live_curves(vision, question, scorer, rule['kind'])
            row = {'row_hash': feature['request_id'], 'group_id': feature['group_id'],
                'input_visual_tokens': len(vision), 'input_text_tokens': feature['input_text_tokens'],
                'curves': curves}
            assignment = policy.infer(rule, [row])
            count = assignment['counts'][feature['request_id']]
            if count != cell['selected_count']:
                raise ValueError('live assignment differs from excluded-fold physical schedule')
            indices = selected_positions(order, count, input_visual_tokens=len(vision))
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            elapsed = time.perf_counter()-started
        if observed is not None:
            values = indices.cpu().tolist()
            observed.update(selected_patch_indices=values, selected_patch_indices_sha256=digest(values),
                candidate='current_rank32', query_control='correct', input_visual_tokens=len(vision),
                input_text_tokens=feature['input_text_tokens'], retained_visual_tokens=len(indices),
                visual_retention=len(indices)/len(vision), selector_seconds=elapsed,
                diagnostics_in_end_to_end_timing=False, budget_in_end_to_end_timing=True,
                score_and_budget_seconds=elapsed, budget_selected_count=count,
                budget_policy_sha256=rule['sha256'], budget_statistic=rule['kind'],
                budget_fallback_reasons=assignment['fallbacks'], budget_dataset_is_prediction_input=False,
                stopping_implementation='dense measured-menu curve; selected statistic only')
        return indices
    return select


@contextmanager
def bindings():
    coordinator = old.coordinator
    names = ('FORMAT', 'verify', 'repeated', 'expected_sources', 'measurement_argv')
    saved = {name: getattr(coordinator, name) for name in names}
    def argv(study, model, descriptor, directory):
        result = saved['measurement_argv'](study, model, descriptor, directory)
        if '--reference-clip' in result:
            result.remove('--reference-clip')
        return result
    coordinator.FORMAT, coordinator.verify = FORMAT, verify
    coordinator.expected_sources, coordinator.measurement_argv = source_hashes, argv
    coordinator.repeated = SimpleNamespace(main=physical_main, source_hashes=source_hashes,
        REPETITIONS=old.REPETITIONS, TIMING_FIELDS=old.TIMING_FIELDS,
        RUNTIME_TIMINGS=old.RUNTIME_TIMINGS, POST_PREFILL_FIELDS=old.POST_PREFILL_FIELDS)
    try:
        yield coordinator
    finally:
        for name, value in saved.items():
            setattr(coordinator, name, value)


def physical_main(argv):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = p.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('exclusive GPU required for timing')
    args = old.repaired.runtime.parse_args(old.base_timing.previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('fresh single-process timing required')
    original = old.base_timing.timed_selector
    old.base_timing.timed_selector = timed_selector
    try:
        with old.repaired.repeated():
            original_sources = old.repaired.physical.source_hashes
            old.repaired.physical.source_hashes = source_hashes
            try:
                old.repaired.physical.main(rest)
            finally:
                old.repaired.physical.source_hashes = original_sources
    finally:
        old.base_timing.timed_selector = original


def publish(study, model, selection):
    from .compression_execution_v12 import publish as publish_plan, python, make_task
    from . import compression_execution_v11 as source_execution
    root = Path(study['output_root'])
    plan_path = root / f'plans/systems-{model}/plan.json'
    if plan_path.exists():
        return plan_path
    source = verify(study)
    master_path = root / model / 'schedules/screening-matched-master.json'
    if not master_path.exists():
        return None
    master = checked(master_path)
    report = checked(root / model / 'development/report.json')['result']
    kind, rules = selection['rule_kind'], {}
    for fold in report['folds']:
        rule = fold['candidate_rules'][kind]
        for request in fold['candidate_assignments'][kind]['counts']:
            if request in rules or request in rule['fit_row_hashes']:
                raise ValueError('timing rule not excluded-fold')
            rules[request] = rule
    value = copy.deepcopy({k: v for k, v in master.items() if k != 'sha256'})
    for request, row in value['rows'].items():
        count = report['procedures'][kind]['counts'][request]
        selector = 'full' if count == row['input_visual_tokens'] else 'scorer'
        if row['group_id'] in rules[request]['fit_group_ids']:
            raise ValueError('timing group leaked into policy fit')
        row['cells'] = [
            {'key': 'full', 'selector': 'full', 'selected_count': row['input_visual_tokens'], 'mode': 'exact'},
            {'key': 'adaptive_live', 'selector': selector, 'selected_count': count,
             'mode': 'exact', 'budget_policy': rules[request]},
            {'key': 'adaptive_replay', 'selector': selector, 'selected_count': count, 'mode': 'exact'},
            {'key': 'static_uniform', 'selector': selector, 'selected_count': count, 'mode': 'exact'}]
    with bindings() as coordinator:
        subset = coordinator.make_schedule(study, model, seal(value), json.loads(Path(study['manifest']).read_text()))
    value = copy.deepcopy({k: v for k, v in subset.items() if k != 'sha256'})
    totals = match_uniform_subset(value)
    value.update(matched_compressed_visual_totals=totals,
        replay_note='identical-count overhead control; not a deployable policy')
    schedule = seal(value)
    schedule_path = root / model / 'schedules/systems.json'
    write_immutable(schedule_path, schedule)
    _, feature_root, _, _ = source_execution.development_features(source, model, master['split'])
    pins = {**study['input_hashes'], **{str(Path(study['cwd']) / 'learned_score' / name): sha
        for name, sha in source_hashes(study['models'][model]['adapter']).items()}}
    for path in (master_path, schedule_path, root / 'selection.json'):
        pins[str(path)] = file_digest(path)
    for request in schedule['rows']:
        path = feature_root / 'features' / (request + '.json')
        pins[str(path)] = file_digest(path)
    descriptor = seal({'format': FORMAT + '_input', 'study_sha256': study['sha256'], 'model': model,
        'manifest_path': study['manifest'], 'feature_root': str(feature_root),
        'policy_path': str(root / 'selection.json'), 'input_hashes': pins})
    descriptor_path = root / model / 'systems/input.json'
    write_immutable(descriptor_path, descriptor)
    pins[str(descriptor_path)] = file_digest(descriptor_path)
    argv = python(study, 'compression_systems_v12') + ['--study', str(root / 'study.json'), '--model', model]
    task = make_task(study, 'systems-' + model, argv, [root / model / 'systems/complete.json'],
        model=model, resource='isolated_gpu', minimum_start_seconds=900)
    return publish_plan(study, 'systems-' + model, [task], pins)


def match_uniform_subset(value):
    """Rebalance the timing subset without leaking schedule fields into policy."""
    from .compression_policy_v8 import uniform_at_visual_total
    totals = {}
    for dataset in sorted({r['dataset'] for r in value['rows'].values()}):
        members = [{'row_hash': key, **row} for key, row in value['rows'].items() if row['dataset'] == dataset]
        total = sum(row['cells'][1]['selected_count'] for row in members)
        geometry = [{key: row[key] for key in ('row_hash', 'group_id', 'dataset',
            'input_visual_tokens', 'input_text_tokens', 'quality_metric', 'category') if key in row}
            for row in members]
        uniform = uniform_at_visual_total(geometry, total)
        if not uniform['feasible']:
            raise ValueError('timing uniform exact-budget assignment infeasible')
        for row in members:
            count = uniform['counts'][row['row_hash']]
            row['cells'][3].update(selected_count=count,
                selector='full' if count == row['input_visual_tokens'] else 'scorer')
        totals[dataset] = total
    return totals


def overhead_gate(rows, maximum_fraction=.05):
    by_method = {}
    for row in rows:
        by_method.setdefault(row['method'], {})[row['request_id']] = row
    live, replay = by_method['adaptive_live'], by_method['adaptive_replay']
    if set(live) != set(replay) or not live:
        raise ValueError('paired live/replay timing required')
    for key, value in live.items():
        if any(value[field] != replay[key][field] for field in
               ('selected_patch_indices', 'generated_token_ids', 'selected_count')):
            raise ValueError('live/replay selected patches or generated outputs differ')
    def total(values):
        return sum(v['timing']['end_to_end_seconds'] for row in values.values()
                   for v in row['metrics']['systems_repetition_metrics'])
    denominator = total(replay)
    if denominator <= 0:
        raise ValueError('nonpositive replay time')
    overhead = total(live) / denominator - 1
    return {'measured': True, 'relative_end_to_end_overhead': overhead,
        'maximum_fraction': maximum_fraction, 'passed': overhead <= maximum_fraction,
        'comparison': 'live budget computation versus exact same counts/scorer without budget computation',
        'quality_and_selected_patch_parity': True}


def advance(study, model, selection):
    plan = publish(study, model, selection)
    if plan is None:
        return {'status': 'awaiting_comparison_schedule', 'measured': False}
    with bindings() as coordinator:
        rows = coordinator.collect(study, model)
        if rows is None:
            return {'status': 'queued_or_running', 'plan': str(plan), 'measured': False}
        report = coordinator.summarize(rows)
    report.update(scope='live deployed statistic; batch one; one warmup and three repetitions',
        matched_control_note='All compressed arms exactly match aggregate visual tokens per dataset.')
    gate = overhead_gate(rows, study['maximum_latency_overhead_fraction'])
    report['budget_overhead_gate'] = gate
    path = Path(study['output_root']) / 'reports' / ('systems-' + model + '.json')
    atomic_json(path, report)
    return {'status': 'complete', 'measured': True, 'report': str(path), 'budget_overhead_gate': gate}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--model', choices=('qwen25', 'llava_next'), required=True)
    args = p.parse_args(argv)
    study = checked(args.study)
    with bindings() as coordinator:
        result = coordinator.measure(study, args.model, study['models'][args.model]['adapter'])
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
