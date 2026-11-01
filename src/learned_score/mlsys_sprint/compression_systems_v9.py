"""Controlled batch-one measurements, including live entropy-budget overhead.

One warmup and three repeats per named method. Unlike the historical quality
runner, this stage never aliases fixed and adaptive timing at the same count.
Only the entropy and frozen budget calculation are timed; label scoring and
the historical feature/nearest-neighbour diagnostics remain outside timing.
"""

import argparse
from contextlib import contextmanager
import copy
import statistics
import time

from . import compression_systems_v8 as previous
from . import compression_systems_runtime_v9 as runtime
from . import compression_policy_v9 as policy_api
from . import selector_experiment_v5 as physical
from .adaptive_utility_v3 import calculate_statistic
from .state import file_digest

REPETITIONS = previous.REPETITIONS
TIMING_FIELDS = previous.TIMING_FIELDS
RUNTIME_TIMINGS = previous.RUNTIME_TIMINGS
exclusive_gpu_snapshot = previous.exclusive_gpu_snapshot
_BASE_SOURCES = physical.source_hashes
POST_PREFILL_FIELDS = ('post_prefill_allocated_bytes', 'post_prefill_reserved_bytes')


def aggregate_measurements(measurements):
    result = previous.aggregate_measurements(measurements)
    for key in POST_PREFILL_FIELDS:
        values = [item[3].get(key) for item in measurements]
        if all(isinstance(value, (int, float)) for value in values):
            result[3][key] = statistics.mean(values)
    return result


def source_hashes(adapter):
    return {**_BASE_SOURCES(adapter),
            'mlsys_sprint/compression_systems_v9.py': file_digest(__file__),
            'mlsys_sprint/compression_systems_runtime_v9.py': file_digest(runtime.__file__),
            'mlsys_sprint/compression_policy_v9.py': file_digest(policy_api.__file__)}


def live_assignment(policy, feature, cell, entropy):
    """Use the normal policy implementation; overlap is labelled timing-only."""
    row = {key: feature[key] for key in
           ('group_id', 'dataset', 'input_visual_tokens', 'input_text_tokens')}
    row.update(row_hash=feature['request_id'], quality_metric=cell['budget_quality_metric'],
               category=cell.get('budget_category'),
               benefit_features={'normalized_entropy': entropy})
    result = policy_api.predict_counts(policy, [row], allow_development_overlap=True)
    assignment = result['rows'][row['row_hash']]
    if assignment['selected_count'] != cell['selected_count'] or assignment['selector'] != cell['selector']:
        raise ValueError('live entropy budget disagrees with frozen timing schedule')
    return assignment


def timed_selector(base, scorer, feature, cell, observed):
    policy = cell.get('budget_policy')
    if policy is None:
        return base
    policy_api.load_policy(policy)
    if policy['kind'] != 'adaptive':
        raise ValueError('only adaptive cells may request entropy budget overhead')

    def select(prepared):
        import torch
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        with torch.no_grad():
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            started = time.perf_counter()
            scores = scorer(vision, question)
            entropy = calculate_statistic(scores, 'normalized_entropy')
            assignment = live_assignment(policy, feature, cell, entropy)
            if vision.is_cuda:
                torch.cuda.synchronize(vision.device)
            elapsed = time.perf_counter() - started
            # Selection reuses the scores needed by the allocator. Do not charge
            # the learned scorer twice, or leave any permanent model changes.
            original = scorer.forward
            was_instance_attribute = 'forward' in scorer.__dict__
            calls = [0]
            def reused_scores(*args, **kwargs):
                calls[0] += 1
                if calls[0] != 1 or kwargs or len(args) != 2 or args[0].shape != vision.shape or args[1].shape != question.shape:
                    raise ValueError('unexpected scorer call during timed score reuse')
                return scores
            scorer.forward = reused_scores
            try:
                indices = base(prepared)
            finally:
                if was_instance_attribute:
                    scorer.forward = original
                else:
                    del scorer.forward
            if observed is not None:
                observed.update(budget_statistic='normalized_entropy', budget_statistic_value=entropy,
                    budget_policy_sha256=policy['sha256'], budget_in_end_to_end_timing=True,
                    score_and_budget_seconds=elapsed, budget_selected_count=assignment['selected_count'],
                    budget_fallback_reasons=assignment['fallback_reasons'])
            return indices
    return select


@contextmanager
def repeated():
    import torch
    import learned_score.mlsys_runtime as cache_runtime
    old_runtime, old_installed, old_sources = physical.runtime, physical.installed, physical.source_hashes
    original = runtime.prior.measured
    original_request = cache_runtime.run_request

    def request(adapter, row, selector, **kwargs):
        first_token = kwargs.get('on_first_token')
        snapshot = {}
        def observed_first_token():
            if first_token is not None:
                first_token()
            device = torch.device(adapter.device)
            if device.type == 'cuda':
                snapshot.update(post_prefill_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                                post_prefill_reserved_bytes=int(torch.cuda.memory_reserved(device)))
        kwargs['on_first_token'] = observed_first_token
        result = original_request(adapter, row, selector, **kwargs)
        result.metrics.update(snapshot)
        return result

    def measured(adapter, row, selector, cell, max_new_tokens):
        if max_new_tokens != 128:
            return original(adapter, row, selector, cell, max_new_tokens)
        before = exclusive_gpu_snapshot()
        warmup = original(adapter, row, selector, cell, max_new_tokens)
        del warmup
        values = [original(adapter, row, selector, cell, max_new_tokens) for _ in range(REPETITIONS)]
        result = aggregate_measurements(values)
        after = exclusive_gpu_snapshot()
        if before['gpu_uuid'] != after['gpu_uuid']:
            raise RuntimeError('systems repetitions changed GPU')
        result[3].update(systems_exclusive_gpu_before=before, systems_exclusive_gpu_after=after,
            budget_calculation_in_end_to_end_timing=True,
            systems_evidence='controlled development timing; not quality certification')
        return result

    @contextmanager
    def installed(extra, args):
        with old_installed(extra, args):
            base_selector = runtime.selector_for
            def selector_for(cell, scorer, feature, observed=None):
                select, internal = base_selector(cell, scorer, feature, observed)
                return timed_selector(select, scorer, feature, cell, observed), internal
            runtime.selector_for = selector_for
            try:
                yield
            finally:
                runtime.selector_for = base_selector

    physical.runtime, physical.installed, physical.source_hashes = runtime, installed, source_hashes
    runtime.prior.measured = measured
    cache_runtime.run_request = request
    try:
        yield
    finally:
        runtime.prior.measured = original
        cache_runtime.run_request = original_request
        physical.runtime, physical.installed, physical.source_hashes = old_runtime, old_installed, old_sources


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = runtime.parse_args(previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('systems methods require one fresh GPU process, no partial timing resume')
    with repeated():
        physical.main(rest)


if __name__ == '__main__':
    main()
