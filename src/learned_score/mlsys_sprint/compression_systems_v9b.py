"""Remove diagnostic-only GPU residency from controlled systems measurements.

The frozen v5 selector keeps a reverse-query diagnostic closure until its outer
measured call returns. Repetition wraps four requests inside that call, leaving
the preceding request's full visual embeddings alive during the next vision
forward. Systems evidence does not use this diagnostic. Drop only that closure
after each selection and at request boundaries; preserve every selected index,
output, scorer parameter, entropy calculation and v9a CLIP capture guard.
"""

import argparse
from contextlib import contextmanager
from types import FunctionType

from . import compression_systems_v9a as previous
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
            'mlsys_sprint/compression_systems_v9b.py': file_digest(__file__)}


def diagnostic_state(selector_factory):
    """Find only the known source-pinned v5 selector's private diagnostic owner.

    v9 adds one `base_selector` wrapper. Fail closed if future wrappers change
    that structure; do not walk arbitrary model objects or unrelated closures.
    """
    current = selector_factory
    for _ in range(4):
        if not isinstance(current, FunctionType):
            break
        closure = dict(zip(current.__code__.co_freevars,
                           (cell.cell_contents for cell in current.__closure__ or ())))
        if (current.__module__ == physical.__name__ and current.__name__ == 'selector'
                and current.__code__.co_filename == physical.__file__):
            state = closure.get('state')
            if not isinstance(state, dict):
                raise ValueError('frozen selector diagnostic owner is not a state dictionary')
            return state
        current = closure.get('base_selector')
    raise ValueError('unknown selector wrapper; cannot safely release diagnostic-only tensors')


def clear_diagnostic(state):
    """Release only the unused callback, without executing diagnostic GPU work."""
    value = state.get('diagnostic')
    if value is not None and not callable(value):
        raise ValueError('unexpected diagnostic state value')
    state.pop('diagnostic', None)


def without_diagnostic_residency(select, state):
    def clean_select(prepared):
        clear_diagnostic(state)
        try:
            return select(prepared)
        finally:
            # The returned indices have their own live reference. Only the
            # unused diagnostic's full-image embedding copies are released.
            clear_diagnostic(state)
    return clean_select


@contextmanager
def repeated():
    import learned_score.mlsys_runtime as cache_runtime
    with previous.repeated():
        old_request, old_installed, old_sources = cache_runtime.run_request, physical.installed, physical.source_hashes
        active_states = []

        @contextmanager
        def installed(extra, args):
            with old_installed(extra, args):
                base_selector = runtime.selector_for
                state = diagnostic_state(base_selector)
                if active_states:
                    raise RuntimeError('nested systems selector owners are unsupported')
                active_states.append(state)
                def selector_for(cell, scorer, feature, observed=None):
                    select, internal = base_selector(cell, scorer, feature, observed)
                    return without_diagnostic_residency(select, state), internal
                runtime.selector_for = selector_for
                try:
                    yield
                finally:
                    clear_diagnostic(state)
                    runtime.selector_for = base_selector
                    active_states.pop()

        def request(adapter, row, selector, **kwargs):
            if len(active_states) != 1:
                raise RuntimeError('systems request lacks one validated diagnostic owner')
            state = active_states[0]
            clear_diagnostic(state)
            try:
                result = old_request(adapter, row, selector, **kwargs)
                result.metrics.update(selector_diagnostics_disabled_for_timing=True,
                                      diagnostic_state_reset_per_request=True)
                return result
            finally:
                clear_diagnostic(state)

        cache_runtime.run_request, physical.installed, physical.source_hashes = request, installed, source_hashes
        try:
            yield
        finally:
            cache_runtime.run_request, physical.installed, physical.source_hashes = old_request, old_installed, old_sources


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = runtime.parse_args(previous.previous.previous._runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0 or args.resume:
        raise ValueError('systems methods require one fresh GPU process, no partial timing resume')
    with repeated():
        physical.main(rest)


if __name__ == '__main__':
    main()
