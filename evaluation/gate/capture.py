#!/usr/bin/env python
"""Adaptive self-consistency gate, step 1: capture the 10%/20% AsymVLM passes.

Reruns AsymVLM uniform at the given keep levels (default 10,20) with the frozen runtime
and records, for every generated token, the top-1/top-2 probability, entropy and
argmax. One JSON per request is written to ``<out>/gate/records/<model>/<request_id>.json``.
The quality of every keep level used by the gate comes from the benchmark records
(``evaluation/run_benchmark.py``); this script only adds the per-token confidences.

Example
  PYTHONPATH=src:. python evaluation/gate/capture.py --model qwen25 --panel first512 --shard 0 --num-shards 2
"""
import argparse
import json
import time
from pathlib import Path

from evaluation import common

MAX_STEPS = 32


def step_stats(logits):
    import torch
    x = logits[0, -1].float(); lp = torch.log_softmax(x, dim=-1); top = torch.topk(lp, 2); p = lp.exp()
    return dict(top1=float(top.values[0].exp()), top2=float(top.values[1].exp()),
                entropy=float(-(p*lp).sum()), argmax=int(logits[0, -1].argmax()))


def install(steps_ref):
    """Wrap the decoder forward so every decoding step's logits are summarized."""
    from learned_score.mlsys_runtime import generation
    original = generation.decoder_forward_function

    def patched(model):
        forward = original(model)

        def wrapped(**kw):
            out = forward(**kw)
            if steps_ref[0] is not None and len(steps_ref[0]) < MAX_STEPS:
                steps_ref[0].append(step_stats(out.logits))
            return out
        return wrapped
    generation.decoder_forward_function = patched


def main():
    p = common.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument('--panel', choices=('first512', 'public'), default='first512')
    p.add_argument('--levels', default='10,20', help='comma-separated keep percentages')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--training-manifest', type=Path, default=None)
    a = p.parse_args()
    levels = tuple(int(x) for x in a.levels.split(','))
    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime
    cfg = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints,
                              out=Path(a.out)/'gate'/'runtime', training_manifest=a.training_manifest)
    steps_ref = [None]; install(steps_ref)
    runtime = Runtime(cfg, a.model)
    panels = common.select_panel(runtime.entries.values(), a.panel)
    mine = common.shard(common.interleave(panels), a.shard, a.num_shards)
    if a.limit:
        mine = mine[:a.limit]
    out = Path(a.out)/'gate'/'records'/a.model
    started, done = time.time(), 0
    for entry in mine:
        rid = entry['row_hash']; path = out/f'{rid}.json'
        if path.exists() and all(str(l) in json.loads(path.read_text())['cells'] for l in levels):
            continue
        cells = {}
        for level in levels:
            steps_ref[0] = []
            rec = runtime.run(entry, 'asymvlm_uniform', level)
            steps, steps_ref[0] = steps_ref[0], None
            raw = json.loads(Path(rec['raw_path']).read_text())
            ids = raw['generated_token_ids']
            cells[str(level)] = dict(prediction=raw['prediction'], ids=ids, quality=rec['quality_score'], steps=steps,
                                     step_mismatch=sum(s['argmax'] != t for s, t in zip(steps, ids)),
                                     count=rec['retained_visual_tokens'])
        common.write_json(path, dict(format='asymvlm_gate_capture', request_id=rid, model=a.model,
                                     dataset=entry['dataset'], group_id=entry['group_id'], metric=entry['metric'],
                                     input_visual_tokens=rec['input_visual_tokens'], cells=cells))
        done += 1
        if done % 20 == 0:
            print(json.dumps(dict(model=a.model, done=done, elapsed=time.time()-started)), flush=True)
    print(json.dumps(dict(model=a.model, done=done, elapsed=time.time()-started, finished=True)), flush=True)


if __name__ == '__main__':
    main()
