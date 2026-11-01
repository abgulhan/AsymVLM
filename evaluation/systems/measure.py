#!/usr/bin/env python
"""Systems measurements (paper: systems table): latency, TTFT, throughput, KV bytes, peak memory.

Panel: the first 8 requests of TextVQA, DocVQA, ChartQA and MME (32 requests). For each
batch size and method (all_tokens, divprune, cdpruner, asymvlm_uniform at 70%% keep) the
batch is first checked for token parity against single-request generation, then timed
with 3 warm-ups + 5 measured repetitions (CUDA-synchronized). Phi-3.5 uses eager attention
and is measured at batch size 1 only. Run on an otherwise idle GPU.

Output: ``<out>/systems/<model>/b<size>-<offset>-<method>.json``; table with
``evaluation/make_tables.py systems``.

Example
  PYTHONPATH=src:. python evaluation/systems/measure.py --model qwen25 --batch-sizes 1
"""
import argparse
import gc
import time
from pathlib import Path

from evaluation import common

METHODS = ('all_tokens', 'divprune', 'cdpruner', 'asymvlm_uniform')
KEEP = 70


def panel(entries):
    result = []
    for domain in ('TextVQA', 'DocVQA', 'ChartQA', 'MME'):
        result.extend(sorted((e for e in entries if common.family(e['dataset']) == domain), key=lambda e: e['panel_order'])[:8])
    if len(result) != 32:
        raise ValueError('systems panel requires eight requests in four domains')
    return result


def prepare(runtime, entries, method):
    from learned_score.mlsys_sprint.manuscript_execute_v35 import selector
    items, counts = [], []
    for entry in entries:
        raw, prepared, feature = runtime.geometry(entry)
        n = feature['input_visual_tokens']; k = n if method == 'all_tokens' else (n*KEEP+99)//100
        indices = selector(method, k, runtime.scorer, {})(prepared)
        items.append(runtime.adapter.physically_prune(prepared, indices, 'original'))
        counts.append(dict(request_id=entry['row_hash'], input_visual_tokens=n, retained_visual_tokens=k))
    return items, counts


def timed_batch(runtime, entries, method):
    import torch
    from learned_score.mlsys_runtime.batched_native_mask_v15b import generate_batched
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); start = time.perf_counter()
    with torch.no_grad():
        items, counts = prepare(runtime, entries, method)
        torch.cuda.synchronize(); prepared_at = time.perf_counter(); prep_peak = torch.cuda.max_memory_allocated()
        result = generate_batched(runtime.adapter.model, items, max_new_tokens=128)
        torch.cuda.synchronize(); seconds = time.perf_counter()-start
        metrics = dict(result.metrics)
        metrics.update(end_to_end_seconds=seconds, ttft_seconds=prepared_at-start+metrics['prefill_ttft_seconds'],
                       prepare_and_selection_seconds=prepared_at-start, requests_per_second=len(entries)/seconds,
                       output_tokens_per_second=sum(map(len, result.tokens))/seconds,
                       end_to_end_peak_allocated_bytes=max(prep_peak, metrics['peak_allocated_bytes']))
        value = dict(tokens=result.tokens, metrics=metrics, counts=counts)
        del result, items
    return value


def timed_phi(runtime, entry, method):
    import torch
    from learned_score.mlsys_sprint import adaptive_experiment_v3 as old
    from learned_score.mlsys_sprint.manuscript_execute_v35 import selector
    raw, prepared, feature = runtime.geometry(entry); del prepared
    n = feature['input_visual_tokens']; k = n if method == 'all_tokens' else (n*KEEP+99)//100
    torch.cuda.reset_peak_memory_stats()
    prediction, ids, timing, metrics = old.measured(runtime.adapter, raw, selector(method, k, runtime.scorer, {}),
                                                    old.Cell(method, 'uniform', KEEP), 128)
    return dict(tokens=ids.cpu().tolist(), metrics=dict(timing=timing, runtime=metrics,
                peak_allocated_bytes=torch.cuda.max_memory_allocated()),
                counts=[dict(request_id=entry['row_hash'], input_visual_tokens=n, retained_visual_tokens=k)])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap)
    ap.add_argument('--batch-sizes', nargs='+', type=int, default=[1, 4, 8])
    ap.add_argument('--methods', nargs='+', default=list(METHODS))
    ap.add_argument('--warmups', type=int, default=3); ap.add_argument('--repetitions', type=int, default=5)
    ap.add_argument('--training-manifest', type=Path, default=None)
    a = ap.parse_args()
    import torch
    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime
    from learned_score.mlsys_runtime.batched_native_mask_v15b import validate_batch_parity
    from learned_score.mlsys_scorer.backbone import assert_frozen
    config = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints,
                                 out=Path(a.out)/'systems'/'runtime', training_manifest=a.training_manifest)
    runtime = Runtime(config, a.model); entries = panel(runtime.entries.values())
    root = Path(a.out)/'systems'/a.model
    for size in a.batch_sizes:
        if a.model == 'phi35' and size > 1:
            common.write_json(root/f'batch{size}-unsupported.json', dict(status='unsupported',
                reason='native padded-batch backend requires SDPA; Phi-3.5 uses eager attention'))
            continue
        batches = [entries[o:o+size] for o in range(0, len(entries), size)]
        for i, batch in enumerate(common.shard(list(enumerate(batches)), a.shard, a.num_shards)):
            offset, batch = batch[0]*size, batch[1]
            for method in a.methods:
                path = root/f'b{size}-{offset:02d}-{method}.json'
                if path.exists():
                    continue
                if a.model != 'phi35':
                    with torch.no_grad():
                        items, _ = prepare(runtime, batch, method)
                        parity = validate_batch_parity(runtime.adapter.model, items, max_new_tokens=128)
                        del items
                    if not parity['passed']:
                        common.write_json(path, dict(status='batch_token_parity_failed_no_latency_claim', parity=parity,
                                                     model=a.model, batch_size=size, method=method))
                        continue
                else:
                    parity = dict(passed=True, reference='single-request native Phi pipeline')
                measured, reference, stable = [], None, True
                for repetition in range(a.warmups+a.repetitions):
                    gc.collect(); torch.cuda.empty_cache()
                    value = timed_phi(runtime, batch[0], method) if a.model == 'phi35' else timed_batch(runtime, batch, method)
                    reference = reference or value['tokens']
                    stable &= value['tokens'] == reference
                    if a.model != 'phi35':
                        stable &= value['tokens'] == [r['batch_tokens'] for r in parity['records']]
                    if repetition >= a.warmups:
                        measured.append(value)
                assert_frozen(runtime.adapter.model, runtime.versions); assert_frozen(runtime.scorer, runtime.scorer_versions)
                common.write_json(path, dict(status='ok' if stable else 'repeat_token_mismatch_no_latency_claim',
                    model=a.model, batch_size=size, method=method, request_ids=[e['row_hash'] for e in batch],
                    warmups=a.warmups, repetitions=a.repetitions, measurements=measured, parity=parity,
                    hardware=torch.cuda.get_device_name()))
                print(dict(model=a.model, batch_size=size, offset=offset, method=method), flush=True)


if __name__ == '__main__':
    main()
