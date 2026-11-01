"""Versioned CDPruner request-boundary repair for controlled v9 timing.

The historical selector installs its native-image capture hook around an entire
measured call. A repeated call contains four separate requests (one warmup and
three measurements), so each request must begin with a fresh capture. Keep the
within-request duplicate-forward guard, frozen selectors, and all v9 timing
and repetition checks unchanged. No previous source or measurement is edited.
"""

import argparse
from contextlib import contextmanager

from . import compression_systems_v9 as previous
from .cd_clip_reference import ReferenceClipRelevance
from .state import file_digest

REPETITIONS = previous.REPETITIONS
TIMING_FIELDS = previous.TIMING_FIELDS
RUNTIME_TIMINGS = previous.RUNTIME_TIMINGS
POST_PREFILL_FIELDS = previous.POST_PREFILL_FIELDS
physical = previous.physical
runtime = previous.runtime
aggregate_measurements = previous.aggregate_measurements
exclusive_gpu_snapshot = previous.exclusive_gpu_snapshot


def source_hashes(adapter):
    return {**previous.source_hashes(adapter),
            'mlsys_sprint/compression_systems_v9a.py': file_digest(__file__)}


def reset_request_capture(adapter):
    """Reset only the known bound CDPruner hook, once at each request boundary."""
    tower = getattr(adapter.model, 'vision_tower', None)
    if tower is None:
        return 0
    captures = []
    for hook in tower._forward_hooks.values():
        owner = getattr(hook, '__self__', None)
        if (isinstance(owner, ReferenceClipRelevance)
                and getattr(hook, '__func__', None) is ReferenceClipRelevance.capture):
            captures.append(owner)
    if len(captures) > 1:
        raise ValueError('multiple CDPruner capture hooks on one request')
    for capture in captures:
        capture.release()
    return len(captures)


@contextmanager
def repeated():
    import learned_score.mlsys_runtime as cache_runtime
    with previous.repeated():
        original_request, original_sources = cache_runtime.run_request, physical.source_hashes
        def request(adapter, row, selector, **kwargs):
            captures = reset_request_capture(adapter)
            result = original_request(adapter, row, selector, **kwargs)
            result.metrics.update(reference_clip_capture_reset_per_request=True,
                                  reference_clip_capture_hook_count=captures)
            return result
        cache_runtime.run_request, physical.source_hashes = request, source_hashes
        try:
            yield
        finally:
            cache_runtime.run_request, physical.source_hashes = original_request, original_sources


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = runtime.parse_args(previous.previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('systems methods require one fresh GPU process, no partial timing resume')
    with repeated():
        physical.main(rest)


if __name__ == '__main__':
    main()
