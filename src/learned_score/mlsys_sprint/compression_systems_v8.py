"""Repeated same-GPU systems measurements for frozen compression schedules.

One warm-up followed by three measurements per exact count. Repetitions must
produce identical tokens and cache geometry. Runtime feature extraction and
query-order diagnostics remain outside the timed inference, as in v5.
"""

import argparse
from contextlib import contextmanager
import copy
import os
from pathlib import Path
import statistics
import subprocess

from . import selector_experiment_v5 as physical
from .state import file_digest

REPETITIONS = 3
TIMING_FIELDS = ('end_to_end_seconds', 'ttft_seconds', 'compression_seconds',
                 'queue_seconds', 'unattributed_seconds')
RUNTIME_TIMINGS = ('full_request_seconds', 'full_request_ttft_seconds', 'generation_seconds',
                  'decode_seconds', 'prepare_seconds', 'prefill_ttft_seconds',
                  'selection_and_visual_compaction_seconds', 'generation_setup_seconds')


def exclusive_gpu_snapshot():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not visible or ',' in visible:
        raise ValueError('systems worker must expose exactly one GPU')
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    devices = dict(line.replace(' ', '').split(',', 1) for line in inventory.splitlines() if line.strip())
    uuid = devices.get(visible, visible if visible in devices.values() else None)
    if uuid is None:
        raise ValueError('cannot resolve the exclusive GPU identity')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'], text=True, timeout=15)
    pids = {int(line.split(',')[1].strip()) for line in processes.splitlines()
            if line.split(',')[0].strip() == uuid}
    if pids - {os.getpid()}:
        raise RuntimeError('another compute process shares the systems measurement GPU')
    return {'gpu_uuid': uuid, 'process_ids': sorted(pids), 'owner_pid': os.getpid()}


def aggregate_measurements(measurements):
    if len(measurements) != REPETITIONS:
        raise ValueError('systems experiment requires exactly three measured repetitions')
    prediction, ids, timing, runtime = measurements[-1]
    tokens = ids.detach().cpu().tolist()
    for other_prediction, other_ids, _, other_runtime in measurements:
        if other_prediction != prediction or other_ids.detach().cpu().tolist() != tokens:
            raise ValueError('repeated natural-stopping inference is nondeterministic')
        if other_runtime['cache'] != runtime['cache']:
            raise ValueError('repeated inference changed cache geometry or occupancy')
    timing, runtime = copy.deepcopy(timing), copy.deepcopy(runtime)
    for name in TIMING_FIELDS:
        values = [item[2].get(name) for item in measurements]
        if all(isinstance(value, (float, int)) for value in values):
            timing[name] = statistics.mean(values)
    for name in RUNTIME_TIMINGS:
        values = [item[3].get(name) for item in measurements]
        if all(isinstance(value, (float, int)) for value in values):
            runtime[name] = statistics.mean(values)
    runtime['full_request_peak_allocated_bytes'] = max(item[3]['full_request_peak_allocated_bytes'] for item in measurements)
    seconds = sum(item[2]['end_to_end_seconds'] for item in measurements)
    generated = sum(int(item[1].shape[-1]) for item in measurements)
    runtime.update(systems_repetitions=REPETITIONS, systems_warmups=1,
        systems_timing_scope='sequential requests on the same exclusive GPU; batch size one',
        systems_generated_tokens_per_second=generated / seconds,
        systems_requests_per_second=REPETITIONS / seconds,
        systems_repetition_metrics=[{'timing': item[2], 'runtime': item[3],
                                     'prediction': item[0],
                                     'generated_token_ids': item[1].detach().cpu().tolist()[0]}
                                    for item in measurements],
        same_gpu_repetitions=True)
    return prediction, ids, timing, runtime


@contextmanager
def repeated():
    prior = physical.runtime.prior
    original, original_sources = prior.measured, physical.source_hashes

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
        result[3]['systems_exclusive_gpu_before'] = before
        result[3]['systems_exclusive_gpu_after'] = after
        return result

    def sources(adapter):
        return {**original_sources(adapter), 'mlsys_sprint/compression_systems_v8.py': file_digest(__file__)}

    prior.measured, physical.source_hashes = measured, sources
    try:
        yield
    finally:
        prior.measured, physical.source_hashes = original, original_sources


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--exclusive-gpu', action='store_true')
    extra, rest = parser.parse_known_args(argv)
    if not extra.exclusive_gpu:
        raise ValueError('systems runs require a verified exclusive-GPU worker')
    args = physical.runtime.parse_args([v for v in rest] if '--candidate' not in rest else
        _runtime_arguments(rest))
    if args.world_size != 1 or args.rank != 0:
        raise ValueError('systems methods must share a single GPU process')
    with repeated():
        physical.main(rest)


def _runtime_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--warmstart', required=True)
    parser.add_argument('--coverage', action='store_true')
    parser.add_argument('--reference-clip', action='store_true')
    parser.add_argument('--query-control', default='correct')
    _, rest = parser.parse_known_args(argv)
    return rest


if __name__ == '__main__':
    main()
