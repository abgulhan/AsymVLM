"""Selector-correct live timing for the v13 finalization study.

The learned scorer determines coverage budgets even when DivPrune selects the
patches. Both operations are included in live latency. Static rules retain the
frozen invalid-curve and visual-support fallback checks and therefore are not
advertised as scorer-free. Exact-count replay is an overhead control, not a
deployable allocation rule. Historical runtime repairs are process-local.
"""
import argparse
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from . import compression_systems_v12 as previous
from . import compression_policy_v12 as policy
from .compression_matrix_v8 import checked, seal
from .followup_common import write_immutable
from .state import atomic_json, digest, file_digest

FORMAT = 'mlsys_compression_systems_v13'
METHODS = ('full', 'selected_live', 'selected_replay', 'static_uniform')


def verify(study):
    from .compression_finalize_v13 import verify as verify_study
    return verify_study(study)


def source_hashes(adapter):
    from . import compression_finalize_v13 as finalizer
    return {**previous.source_hashes(adapter),
        'mlsys_sprint/compression_systems_v13.py': file_digest(__file__),
        'mlsys_sprint/compression_finalize_v13.py': file_digest(finalizer.__file__)}


def patch_selector(cell):
    """Full fallback must not erase which selector the method actually uses."""
    selector = cell.get('patch_selector', cell.get('selector'))
    if selector not in ('scorer', 'divprune'):
        raise ValueError('live/replay method requires explicit scorer or divprune patch_selector')
    if cell.get('selector') not in (selector, 'full'):
        raise ValueError('physical selector and requested patch selector disagree')
    return selector


def timed_selector(base, scorer, feature, cell, observed):
    rule = cell.get('budget_policy')
    if rule is None:
        return base
    if rule['kind'] not in ('static', 'rank32_coverage'):
        raise ValueError('v13 supports only frozen static or rank32 coverage rules')
    selector = patch_selector(cell)

    def select(prepared):
        import torch
        from learned_score.llava_selectors import divprune_order
        from .compression_stopping_v11 import selected_positions
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        if (len(vision) != feature['input_visual_tokens'] or
                prepared.inputs_embeds.shape[1] - len(vision) != feature['input_text_tokens']):
            raise ValueError('live policy geometry changed')
        with torch.no_grad():
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            started = time.perf_counter()
            curves, learned_order = previous.live_curves(vision, question, scorer, rule['kind'])
            row = {'row_hash': feature['request_id'], 'group_id': feature['group_id'],
                'input_visual_tokens': len(vision), 'input_text_tokens': feature['input_text_tokens'],
                'curves': curves}
            assignment = policy.infer(rule, [row])
            count = assignment['counts'][feature['request_id']]
            if count != cell['selected_count']:
                raise ValueError('live assignment differs from excluded-fold physical schedule')
            # Budget geometry is learned even for the diversity hybrid. Calling
            # its real selector here is essential; v12 always used learned_order.
            order = (divprune_order(vision, count)
                if selector == 'divprune' and count < len(vision) else learned_order)
            indices = selected_positions(order, count, input_visual_tokens=len(vision))
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            elapsed = time.perf_counter() - started
        if observed is not None:
            values = indices.cpu().tolist()
            observed.update(selected_patch_indices=values, selected_patch_indices_sha256=digest(values),
                candidate='current_rank32', query_control='correct', input_visual_tokens=len(vision),
                input_text_tokens=feature['input_text_tokens'], retained_visual_tokens=len(indices),
                visual_retention=len(indices) / len(vision), selector_seconds=elapsed,
                diagnostics_in_end_to_end_timing=False, budget_in_end_to_end_timing=True,
                score_and_budget_seconds=elapsed, budget_selected_count=count,
                budget_policy_sha256=rule['sha256'], budget_statistic=rule['kind'],
                budget_fallback_reasons=assignment['fallbacks'], budget_dataset_is_prediction_input=False,
                budget_scorer_executed=True, budget_geometry='learned_rank32',
                patch_selector=selector, diversity_selection_in_end_to_end_timing=selector == 'divprune',
                scorer_free=False,
                stopping_implementation='frozen dense measured-menu rule including support/invalid checks')
        return indices
    return select


@contextmanager
def bindings():
    coordinator = previous.old.coordinator
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
        REPETITIONS=previous.old.REPETITIONS, TIMING_FIELDS=previous.old.TIMING_FIELDS,
        RUNTIME_TIMINGS=previous.old.RUNTIME_TIMINGS, POST_PREFILL_FIELDS=previous.old.POST_PREFILL_FIELDS)
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
    old = previous.old
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


def selected_rules(report, variant_id):
    report = report.get('result', report)
    rules, counts = {}, {}
    for fold in report['folds']:
        rule = fold['candidate_rules'][variant_id]
        for request, count in fold['candidate_assignments'][variant_id]['counts'].items():
            if request in rules or request in rule['fit_row_hashes']:
                raise ValueError('timing rule not excluded-fold')
            rules[request], counts[request] = rule, count
    if not rules:
        raise ValueError('empty excluded-fold timing assignment')
    return rules, counts


def make_schedule(study, model, master, manifest, report, selection):
    """Use frozen excluded-fold rules and the legacy identity-selected subset."""
    selector, variant = selection['selector'], selection['variant_id']
    if selector not in ('scorer', 'divprune'):
        raise ValueError('unsupported selected patch selector')
    rules, counts = selected_rules(report, variant)
    value = copy.deepcopy({k: v for k, v in master.items() if k != 'sha256'})
    for request, row in value['rows'].items():
        if request not in rules or row['group_id'] in rules[request]['fit_group_ids']:
            raise ValueError('timing group leaked into policy fit or lacks rule')
        if rules[request]['kind'] != selection['rule_kind']:
            raise ValueError('selected profile and timing rule kind disagree')
        count = counts[request]
        choice = 'full' if count == row['input_visual_tokens'] else selector
        common = {'selector': choice, 'patch_selector': selector, 'selected_count': count, 'mode': 'exact'}
        row['cells'] = [
            {'key': 'full', 'selector': 'full', 'selected_count': row['input_visual_tokens'], 'mode': 'exact'},
            {'key': 'selected_live', **common, 'budget_policy': rules[request]},
            {'key': 'selected_replay', **common},
            {'key': 'static_uniform', **common}]
    with bindings() as coordinator:
        subset = coordinator.make_schedule(study, model, seal(value), manifest)
    value = copy.deepcopy({k: v for k, v in subset.items() if k != 'sha256'})
    totals = match_uniform_subset(value)
    value.update(matched_compressed_visual_totals=totals,
        selected_method=selection.get('selected_method', selection.get('method_id')),
        variant_id=variant, patch_selector=selector,
        replay_note='identical counts and selected patch selector; skips budget calculation; not deployable',
        static_control_note='cohort-matched control, not a benchmark-specific runtime policy')
    return seal(value)


def match_uniform_subset(value):
    from .compression_policy_v8 import uniform_at_visual_total
    totals = {}
    for dataset in sorted({r['dataset'] for r in value['rows'].values()}):
        members = [(key, row) for key, row in value['rows'].items() if row['dataset'] == dataset]
        cells = {key: {c['key']: c for c in row['cells']} for key, row in members}
        if any(set(c) != set(METHODS) for c in cells.values()):
            raise ValueError('timing schedule requires exactly full/live/replay/uniform methods')
        total = sum(cells[key]['selected_live']['selected_count'] for key, _ in members)
        if any(cells[key]['selected_live']['selected_count'] != cells[key]['selected_replay']['selected_count']
               or patch_selector(cells[key]['selected_live']) != patch_selector(cells[key]['selected_replay'])
               for key, _ in members):
            raise ValueError('live/replay selector/count mismatch')
        geometry = [{'row_hash': key, **{field: row[field] for field in
            ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens', 'quality_metric', 'category')
            if field in row}} for key, row in members]
        uniform = uniform_at_visual_total(geometry, total)
        if not uniform['feasible']:
            raise ValueError('timing uniform exact-budget assignment infeasible')
        for key, row in members:
            count = uniform['counts'][key]
            control = cells[key]['static_uniform']
            selector = patch_selector(control)
            control.update(selected_count=count,
                selector='full' if count == row['input_visual_tokens'] else selector)
        totals[dataset] = total
    return totals


def publish(study, model, selection, *, report=None, master_path=None, policy_path=None):
    from .compression_execution_v13 import publish as publish_plan, python, make_task
    from . import compression_execution_v11 as source_execution
    root = Path(study['output_root'])
    source_v12 = verify(study)
    source = previous.verify(source_v12)
    plan_path = root / f'plans/systems-{model}/plan.json'
    if plan_path.exists():
        from .worker import read_plan, verify_sources
        verify_sources(read_plan(plan_path))
        descriptor = checked(root / model / 'systems/input.json')
        if descriptor.get('selection_sha256') != digest(selection):
            raise ValueError('cannot change frozen systems method')
        return plan_path
    master_path = Path(master_path) if master_path is not None else root / model / 'schedules/screening-learned-controls-master.json'
    if not master_path.exists():
        return None
    if time.time() >= study['deadline_utc_epoch']:
        raise ValueError('v13 systems publication deadline reached')
    report_path = root / model / 'development/report.json'
    report = checked(report_path) if report is None else report
    master = checked(master_path)
    manifest = json.loads(Path(study['manifest']).read_text())
    schedule = make_schedule(study, model, master, manifest, report, selection)
    schedule_path = root / model / 'schedules/systems.json'
    write_immutable(schedule_path, schedule)
    _, feature_root, _, _ = source_execution.development_features(source, model, master['split'])
    # Freeze the exact supplied decision/report locally rather than pin a
    # mutable status file that can legitimately change after systems complete.
    policy_path = root / model / 'systems/selected-method.json'
    write_immutable(policy_path, seal({'selection': selection, 'report': report,
        'selection_sha256': digest(selection), 'report_sha256': digest(report)}))
    pins = {**study['input_hashes'], **{str(Path(study['cwd']) / 'learned_score' / name): sha
        for name, sha in source_hashes(study['models'][model]['adapter']).items()}}
    for path in (master_path, schedule_path, policy_path):
        pins[str(path)] = file_digest(path)
    for request, row in schedule['rows'].items():
        path = feature_root / 'features' / (request + '.json')
        feature = json.loads(path.read_text())
        if (feature.get('checkpoint_sha256') != schedule['checkpoint_sha256']
                or feature.get('request_id') != request
                or any(feature.get(key) != row[key] for key in
                       ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens'))):
            raise ValueError('timing feature identity or geometry differs')
        pins[str(path)] = file_digest(path)
    descriptor = seal({'format': FORMAT + '_input', 'study_sha256': study['sha256'], 'model': model,
        'manifest_path': study['manifest'], 'feature_root': str(feature_root),
        'policy_path': str(policy_path), 'selection_sha256': digest(selection), 'input_hashes': pins})
    descriptor_path = root / model / 'systems/input.json'
    write_immutable(descriptor_path, descriptor)
    pins[str(descriptor_path)] = file_digest(descriptor_path)
    argv = python(study, 'compression_systems_v13') + ['--study', str(root / 'study.json'), '--model', model]
    task = make_task(study, 'systems-' + model, argv, [root / model / 'systems/complete.json'],
        model=model, resource='isolated_gpu', minimum_start_seconds=900)
    return publish_plan(study, 'systems-' + model, [task], pins)


def overhead_gate(rows, maximum_fraction=.05):
    """Validate paired patches, tokens, selector, and count before using times."""
    if not math.isfinite(maximum_fraction) or maximum_fraction < 0:
        raise ValueError('nonnegative finite maximum overhead required')
    by_method = {}
    for row in rows:
        method = by_method.setdefault(row['method'], {})
        if row['request_id'] in method:
            raise ValueError('duplicate paired timing request')
        method[row['request_id']] = row
    live = by_method.get('selected_live', {})
    replay = by_method.get('selected_replay', {})
    if set(live) != set(replay) or not live:
        raise ValueError('paired live/replay timing required')
    for key, value in live.items():
        if any(value[field] != replay[key][field] for field in
               ('selected_patch_indices', 'generated_token_ids', 'selected_count', 'selector')):
            raise ValueError('live/replay selected patches, selectors, counts or generated outputs differ')
        for entry in (value, replay[key]):
            repetitions = entry['metrics']['systems_repetition_metrics']
            if len(repetitions) != 3 or any(not math.isfinite(v['timing']['end_to_end_seconds'])
                    or v['timing']['end_to_end_seconds'] <= 0 for v in repetitions):
                raise ValueError('three positive end-to-end timing repetitions required')

    def total(values):
        return sum(v['timing']['end_to_end_seconds'] for row in values.values()
                   for v in row['metrics']['systems_repetition_metrics'])

    overhead = total(live) / total(replay) - 1
    return {'measured': True, 'relative_end_to_end_overhead': overhead,
        'maximum_fraction': maximum_fraction, 'passed': overhead <= maximum_fraction,
        'comparison': 'live learned budget plus selected patch selector versus same counts and selector replay',
        'quality_and_selected_patch_parity': True, 'scorer_free_claim': False}


def advance(study, model, selection, **kwargs):
    plan = publish(study, model, selection, **kwargs)
    if plan is None:
        return {'status': 'awaiting_comparison_schedule', 'measured': False}
    with bindings() as coordinator:
        rows = coordinator.collect(study, model)
        if rows is None:
            return {'status': 'queued_or_running', 'plan': str(plan), 'measured': False}
        report = coordinator.summarize(rows)
    report.update(scope='selected live budget and selected patch selector; batch one; one warmup and three repetitions',
        selected_method=selection.get('selected_method', selection.get('method_id')),
        matched_control_note='All compressed arms exactly match aggregate visual tokens per dataset.',
        scorer_free_claim=False)
    gate = overhead_gate(rows, study.get('maximum_latency_overhead_fraction', .05))
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
