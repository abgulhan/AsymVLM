#!/usr/bin/env python
"""Main compression benchmark (paper: comparison figure / tables, 512 examples per dataset).

For one model, runs every (method, keep%) cell physically at batch size 1 on the
canonical panel (first N rows per dataset in locked panel order). Each request writes
one JSON record to

    <out>/records/<model>/<dataset>/<method>/<percent>/<request_id>.json

containing the prediction, the benchmark score (``quality_score``) and the measured
KV-cache bytes. Existing records are skipped, so the command is resumable. Shard the
work over GPUs with ``--shard i --num-shards n`` (one process per GPU).

Methods
  all_tokens            100% reference (run once, keep% ignored)
  asymvlm_uniform       learned scorer, uniform keep%            (AsymVLM)
  divprune, cdpruner    native projected-feature adaptations
  visionzip_native      Qwen2.5-VL native-tower VisionZip port
  visionzip_llava_native, visionzip_phi_native   VisionZip ports for LLaVA-NeXT / Phi-3.5
  fastv_native, vlcache_native                   native FastV / VL-Cache ports

Example
  PYTHONPATH=src:. python evaluation/run_benchmark.py --model qwen25 \
      --methods all_tokens asymvlm_uniform divprune --per-dataset 512 --shard 0 --num-shards 4
"""
import argparse
import time
import traceback
from pathlib import Path

from evaluation import common


def build_runner(runtime, method, out):
    """Return ``fn(entry, percent) -> record`` for one method (mirrors the paper's run_one)."""
    model = runtime.model
    if method in ('all_tokens', 'asymvlm_uniform', 'divprune', 'cdpruner'):
        def run(entry, percent):
            runtime.config = {**runtime.config, 'output_root': str(out)}
            return runtime.run(entry, method, percent)
        return run
    if method == 'visionzip_native':
        if model != 'qwen25':
            raise ValueError('visionzip_native is the Qwen2.5-VL port')
        from evaluation.baselines import visionzip_qwen, visionzip_qwen_physical
        return lambda entry, percent: visionzip_qwen_physical.visionzip_physical(runtime, entry, percent, Path(out), visionzip_qwen)
    if method == 'visionzip_phi_native':
        if model != 'phi35':
            raise ValueError('visionzip_phi_native is the Phi-3.5 port')
        from evaluation.baselines import visionzip_phi
        return lambda entry, percent: visionzip_phi.physical(runtime, entry, percent)
    if model == 'llava_next' and method in ('fastv_native', 'vlcache_native', 'visionzip_llava_native'):
        from evaluation.baselines import native_llava, native_llava_physical
        return lambda entry, percent: native_llava_physical.physical(runtime, native_llava, entry, method, percent, 128)
    if model in ('qwen25', 'phi35') and method in ('fastv_native', 'vlcache_native'):
        return _native_port_runner(runtime, method)
    raise ValueError(f'unsupported method {method} for {model}')


def _native_port_runner(runtime, method):
    """FastV / VL-Cache on Qwen2.5-VL and Phi-3.5 (original-position native ports)."""
    import torch
    from evaluation.baselines import native_ports_qwen_phi as ports
    from learned_score.mlsys_sprint.manuscript_execute_v35 import score
    from learned_score.mlsys_sprint.adaptive_experiment_v3 import benchmark

    def run(entry, percent):
        with ports.native_sdpa(runtime.adapter.model), torch.inference_mode():
            raw, prepared, _ = runtime.geometry(entry)
            n = len(prepared.visual_positions)
            base = dict(request_id=entry['row_hash'], model=runtime.model, dataset=common.family(entry['dataset']),
                        group_id=entry['group_id'], panel_order=entry['panel_order'], method=method,
                        percent=percent, input_visual_tokens=n)
            try:
                ids, timing = ports.generate_native(runtime.adapter.model, prepared, method, percent)
            except ports.UnsupportedBudget as exc:
                # The budget is never silently changed: such cells are reported as unsupported.
                return dict(**base, status='unsupported_budget', reason=str(exc), quality_score_available=False)
        prediction = runtime.adapter.processor.tokenizer.decode(ids[0], skip_special_tokens=True)
        labeled = runtime.labels.row_for_entry(entry)
        if labeled['question'] != raw['question']:
            raise ValueError('label/input identity mismatch')
        refs = benchmark.parse_references(labeled['answer'])
        return dict(**base, status='ok', prediction=prediction, quality_score=score(prediction, refs, entry['metric']),
                    generated_ids=ids[0].cpu().tolist(), runtime=timing, no_scorer_used=True, timing_isolated=False)
    return run


def main():
    p = common.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--methods', nargs='+', default=None, help='default: all_tokens + every method of the model')
    p.add_argument('--percents', nargs='+', type=int, default=list(common.PERCENTS))
    p.add_argument('--datasets', nargs='+', default=list(common.DATASETS))
    p.add_argument('--per-dataset', type=int, default=512)
    p.add_argument('--training-manifest', type=Path, default=None)
    a = p.parse_args()
    methods = a.methods or ['all_tokens', *common.METHODS[a.model]]

    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime
    cfg = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints, out=a.out,
                              training_manifest=a.training_manifest)
    runtime = Runtime(cfg, a.model)
    panels = common.first_panel(runtime.entries.values(), a.per_dataset, a.datasets)
    mine = common.shard(common.interleave(panels), a.shard, a.num_shards)
    runners = {m: build_runner(runtime, m, a.out) for m in methods}
    started, done = time.time(), 0
    for entry in mine:
        for method in methods:
            for percent in ([100] if method == 'all_tokens' else a.percents):
                target = Path(a.out)/'records'/a.model/common.family(entry['dataset'])/method/str(percent)/(entry['row_hash']+'.json')
                if target.exists():
                    continue
                try:
                    common.write_json(target, runners[method](entry, percent))
                    done += 1
                except Exception:
                    print(f'failed {a.model} {method} {percent} {entry["row_hash"]}\n{traceback.format_exc()}', flush=True)
                    raise
        print(f'[{a.model} shard {a.shard}/{a.num_shards}] {done} cells, {time.time()-started:.0f}s', flush=True)


if __name__ == '__main__':
    main()
