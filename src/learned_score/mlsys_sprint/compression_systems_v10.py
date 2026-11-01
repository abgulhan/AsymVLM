"""Exact-matched controlled timing for pooled v10 image budgets.

Reuse the frozen v9b capture/residency repairs without editing historical code.
The process-local binding below replaces only the live policy calculation and
versioned publication checks. Each method receives one warmup and three fresh
measurements; scorer and single-statistic budgeting are inside request timing.
"""
import argparse
from collections import defaultdict
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

from . import compression_execution_v10 as execution
from . import compression_features_v10 as features_api
from . import compression_policy_v10 as policy_api
from . import compression_policy_v8 as geometry
from . import compression_systems_v9 as base_timing
from . import compression_systems_v9b as repaired
from . import compression_systems_run_v9 as coordinator
from .adaptive_utility_v3 import calculate_statistic
from .followup_common import write_immutable
from .state import digest, file_digest
from .worker import read_plan, verify_sources

FORMAT = 'mlsys_compression_systems_v10'
REPETITIONS = repaired.REPETITIONS
TIMING_FIELDS = repaired.TIMING_FIELDS
RUNTIME_TIMINGS = repaired.RUNTIME_TIMINGS
POST_PREFILL_FIELDS = repaired.POST_PREFILL_FIELDS
MATCHED_METHODS = ('matched_static', 'matched_divprune', 'matched_cdpruner')
read, checked, seal = execution.read, execution.checked, execution.seal


def source_hashes(adapter):
    return {**repaired.source_hashes(adapter),
            'mlsys_sprint/compression_systems_v10.py': file_digest(__file__),
            'mlsys_sprint/compression_policy_v10.py': file_digest(policy_api.__file__),
            'mlsys_sprint/compression_features_v10.py': file_digest(features_api.__file__),
            'mlsys_sprint/compression_execution_v10.py': file_digest(execution.__file__)}


def live_assignment(policy, feature, cell, statistic_value):
    """Only token geometry, identity and one scalar enter prediction."""
    row = {key: feature[key] for key in ('group_id', 'input_visual_tokens', 'input_text_tokens')}
    row.update(row_hash=feature['request_id'], model_tag=policy['model'],
               checkpoint_sha256=feature['checkpoint_sha256'],
               benefit_features={} if policy['kind'] == 'fixed' else {policy['statistic']: statistic_value})
    result = policy_api.predict_counts(policy, [row], allow_development_overlap=True)
    assignment = result['rows'][row['row_hash']]
    if any(assignment[key] != cell[key] for key in ('selected_count', 'selector')):
        raise ValueError('live v10 budget disagrees with the frozen timing schedule')
    return assignment


def timed_selector(base, scorer, feature, cell, observed):
    policy = cell.get('budget_policy')
    if policy is None:
        return base
    policy_api.load_policy(policy)

    def select(prepared):
        import torch
        from torch.nn import functional as F
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        if (len(vision) != feature['input_visual_tokens'] or
                prepared.inputs_embeds.shape[1]-len(vision) != feature['input_text_tokens']):
            raise ValueError('live timing geometry differs from the frozen feature')
        with torch.no_grad():
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            started = time.perf_counter()
            scores, value = None, None
            if policy['kind'] == 'adaptive':
                scores = scorer(vision, question)
                if policy['statistic'] == 'normalized_entropy':
                    value = calculate_statistic(scores, 'normalized_entropy')
                else:
                    projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
                    value = features_api.weighted_uncovered(scores, projected)[features_api.STATISTIC]
                    del projected
            assignment = live_assignment(policy, feature, cell, value)
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            elapsed = time.perf_counter()-started
            original = scorer.forward
            instance_forward = 'forward' in scorer.__dict__
            calls = [0]

            def reused_scores(*args, **kwargs):
                calls[0] += 1
                if (calls[0] != 1 or kwargs or len(args) != 2 or
                        args[0].shape != vision.shape or args[1].shape != question.shape):
                    raise ValueError('unexpected scorer call during v10 timed reuse')
                return scores

            if scores is not None:
                scorer.forward = reused_scores
            try:
                indices = base(prepared)
            finally:
                if scores is not None:
                    if instance_forward:
                        scorer.forward = original
                    else:
                        del scorer.forward
            if observed is not None:
                observed.update(budget_statistic=policy['statistic'], budget_statistic_value=value,
                    budget_policy_sha256=policy['sha256'], budget_in_end_to_end_timing=True,
                    score_and_budget_seconds=elapsed, budget_selected_count=assignment['selected_count'],
                    budget_fallback_reasons=assignment['fallback_reasons'],
                    budget_dataset_is_prediction_input=False, budget_policy_kind=policy['kind'])
            return indices
    return select


@contextmanager
def repeated():
    original = base_timing.timed_selector
    base_timing.timed_selector = timed_selector
    try:
        with repaired.repeated():
            previous_sources = repaired.physical.source_hashes
            repaired.physical.source_hashes = source_hashes
            try:
                yield
            finally:
                repaired.physical.source_hashes = previous_sources
    finally:
        base_timing.timed_selector = original


def physical_main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = repaired.runtime.parse_args(base_timing.previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('systems methods require one fresh GPU process, no partial timing resume')
    with repeated():
        repaired.physical.main(rest)


@contextmanager
def _coordinator():
    """Use legacy audits with v10 bindings, restoring all module state on exit."""
    names = ('FORMAT', 'verify', 'repeated', 'expected_sources')
    originals = {name: getattr(coordinator, name) for name in names}
    coordinator.FORMAT = FORMAT
    coordinator.verify = execution.verify
    coordinator.expected_sources = source_hashes
    coordinator.repeated = SimpleNamespace(main=physical_main, source_hashes=source_hashes,
        REPETITIONS=REPETITIONS, TIMING_FIELDS=TIMING_FIELDS, RUNTIME_TIMINGS=RUNTIME_TIMINGS,
        POST_PREFILL_FIELDS=POST_PREFILL_FIELDS)
    try:
        yield
    finally:
        for name, value in originals.items():
            setattr(coordinator, name, value)


def make_schedule(study, model, source, manifest, policy):
    policy_api.load_policy(policy)
    if policy['model'] != model or policy['checkpoint_sha256'] != source['checkpoint_sha256']:
        raise ValueError('systems policy model/checkpoint mismatch')
    with _coordinator():
        selected = coordinator.make_schedule(study, model, source, manifest)
    value = {key: copy.deepcopy(item) for key, item in selected.items() if key != 'sha256'}
    domains = defaultdict(list)
    for request, row in value['rows'].items():
        cells = {cell['key']: cell for cell in row['cells']}
        if not {'full', 'selected', 'matched_static', 'matched_divprune'} <= cells.keys():
            raise ValueError('systems source lacks required full/selected/static/DivPrune methods')
        if set(cells) - {'full', 'selected', *MATCHED_METHODS}:
            raise ValueError('unexpected systems method')
        if cells['selected'].get('policy_sha256') != policy['sha256']:
            raise ValueError('selected systems cell does not bind the pooled policy')
        cells['selected']['budget_policy'] = policy
        domains[row['dataset']].append((request, row, cells))
    totals = {}
    for domain, members in domains.items():
        goal = sum(cells['selected']['selected_count'] for _, _, cells in members)
        rows = [{'row_hash': request, **{key: row[key] for key in
                ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')}}
                for request, row, _ in members]
        uniform = geometry.uniform_at_visual_total(rows, goal)
        if not uniform['feasible']:
            raise ValueError('systems exact aggregate match is infeasible')
        for request, row, cells in members:
            count = uniform['counts'][request]
            for method in MATCHED_METHODS:
                if method not in cells:
                    continue
                cell = cells[method]
                selector = {'matched_static': 'scorer', 'matched_divprune': 'divprune',
                            'matched_cdpruner': 'cdpruner'}[method]
                cell.update(selected_count=count, selector='full' if count == row['input_visual_tokens'] else selector,
                            timing_subset_rebalanced=True)
                cell.pop('keep_percent', None)
        totals[domain] = {method: sum(cells[method]['selected_count'] for _, _, cells in members)
                          for method in members[0][2]}
        if any(total != goal for method, total in totals[domain].items() if method != 'full'):
            raise ValueError('systems methods do not share exact aggregate token counts')
    value.update(policy_sha256=policy['sha256'], matched_visual_totals=totals,
                 matched_control_note='All compressed methods match selected visual-token totals on this exact timing subset.',
                 source_schedule_sha256=source['sha256'])
    return seal(value)


def publish(study, model, schedule_path, policy, feature_root):
    execution.verify(study)
    root, spec = Path(study['output_root']), study['models'][model]
    plan_path = root / f'plans/systems-{model}/plan.json'
    source, manifest = checked(schedule_path), read(study['manifest'])
    if source['manifest_sha256'] != file_digest(study['manifest']):
        raise ValueError('systems source manifest changed')
    if not plan_path.exists() and time.time() >= study['deadline_utc_epoch']:
        raise ValueError('cannot publish systems work after the deadline')
    schedule = make_schedule(study, model, source, manifest, policy)
    output = root / model / 'schedules/systems.json'
    policy_path = root / model / 'systems/policy.json'
    write_immutable(output, schedule)
    write_immutable(policy_path, policy)
    _, expected_features, _, _ = execution.development_features(study, model, source['split'])
    if Path(feature_root).resolve() != expected_features.resolve():
        raise ValueError('systems requires the audited native development feature root')
    pins = {**execution.source_pins(study, model),
            **{str(Path(study['cwd'])/'learned_score'/name): sha for name, sha in source_hashes(spec['adapter']).items()},
            **{str(Path(module.__file__).resolve()): file_digest(module.__file__)
               for module in (coordinator, coordinator.audit, coordinator.reporting)}}
    for path in (Path(schedule_path), Path(study['manifest']), output, policy_path):
        pins[str(path.resolve())] = file_digest(path)
    for request, row in schedule['rows'].items():
        path = Path(feature_root)/'features'/(request+'.json')
        feature = read(path)
        if (feature.get('checkpoint_sha256') != schedule['checkpoint_sha256'] or
                feature.get('request_id') != request or any(feature.get(key) != row[key] for key in
                ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens'))):
            raise ValueError('systems feature identity/geometry mismatch')
        pins[str(path.resolve())] = file_digest(path)
    descriptor = seal({'format': FORMAT+'_input', 'study_sha256': study['sha256'], 'model': model,
        'manifest_path': study['manifest'], 'feature_root': str(Path(feature_root).resolve()),
        'policy_path': str(policy_path), 'input_hashes': pins, 'evidence': 'development timing only'})
    descriptor_path = root/model/'systems/input.json'
    write_immutable(descriptor_path, descriptor)
    pins[str(descriptor_path)] = file_digest(descriptor_path)
    argv = execution.matrix.previous.python(study, 'compression_systems_v10') + [
        '--phase', 'measure', '--study', str(root/'study.json'), '--model-tag', model, '--adapter', spec['adapter']]
    task = execution.matrix.previous.task(study, 'systems-'+model, argv, [root/model/'systems/complete.json'])
    task.update(id='compression-v10-systems-'+model, tag='compression-v10', model_tag=model,
                minimum_start_seconds=900)
    plan = seal({'format': FORMAT+'_plan', 'cwd': study['cwd'], 'tasks': [task], 'input_hashes': pins,
        'unexpected_retries': 1, 'created_utc_epoch': study['created_utc_epoch'],
        'deadline_utc_epoch': study['deadline_utc_epoch'], 'world_size': 1})
    write_immutable(plan_path, plan)
    verify_sources(read_plan(plan_path))
    return plan_path


def measure(study, model, adapter):
    with _coordinator():
        return coordinator.measure(study, model, adapter)


def collect(study, model):
    execution.verify(study)
    path = Path(study['output_root'])/f'plans/systems-{model}/plan.json'
    if not path.exists():
        return None
    verify_sources(read_plan(path))
    with _coordinator():
        return coordinator.collect(study, model)


def summarize(rows):
    with _coordinator():
        result = coordinator.summarize(rows)
    result.update(scope='sequential batch-one inference; frozen v10 live score/statistic/budget work included',
        matched_control_note='All compressed methods match aggregate visual-token counts on the measured subset.')
    for panel in result['panels']:
        values = [r for r in rows if (r['model_tag'], r['dataset'], r['method']) ==
                  (panel['model'], panel['dataset'], panel['method'])]
        overhead = [r.get('score_and_budget_seconds') for r in values]
        panel['mean_score_and_budget_seconds'] = (statistics.mean(overhead)
            if all(isinstance(v, (int, float)) for v in overhead) else None)
        panel['mean_score_and_budget_note'] = 'Last repetition per input; end-to-end timing averages all three repetitions.'
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('measure',), required=True)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--model-tag', choices=execution.MODELS, required=True)
    parser.add_argument('--adapter', choices=('qwen2_5_vl', 'llava_next'), required=True)
    args = parser.parse_args(argv)
    print(json.dumps(measure(checked(args.study), args.model_tag, args.adapter), sort_keys=True))


if __name__ == '__main__':
    main()
