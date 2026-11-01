#!/usr/bin/env python
"""Joint vision pruning + generated-text KV eviction on olmOCR-Bench (generation).

Per (source page, cell): AsymVLM top-k vision selection at ``vision`` %% keep, then
  policy=full               -> unbounded generated-text KV (generate_dynamic)
  policy=protected_recency  -> fixed generated-text budget ``profile`` (recency window)
  policy=h2o / rkv          -> H2O / R-KV-style eviction at the same budget (comparisons)
``donate=True`` lets the vision savings enlarge the text budget (capacity basis = full prefix).
Greedy, max_new_tokens=4096, EOS stop, batch 1. One JSON row per (model, cell, page):
``<out>/joint/rows/<model>/<cell>/<row_hash>.json``. Score with evaluation/joint/score.py.

The olmOCR manifest is built with ``learned_score.mlsys_eval.olmocr.build_olmocr_manifest``
from a local copy of allenai/olmOCR-bench (see docs/REPRODUCE.md); the 64 paper pages are
listed in results/panels/olmocr_joint_panel.json.

Example (paper Table: 4 core cells, both models; ~100 s/request on one GPU)
  PYTHONPATH=src:. python evaluation/joint/generate.py --model qwen25 \
      --olmocr-manifest data/olmocr/manifest.json --panel combined64 --shard 0 --num-shards 4
"""
import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

from evaluation import common

PANEL = common.ROOT/'results'/'panels'/'olmocr_joint_panel.json'
CORE = ('all_tokens_full_history', 'vision70_full_history', 'all_tokens_recency256', 'vision70_recency256')


@contextmanager
def capacity_basis(retained_basis):
    """Separate storage size from original positional coordinates."""
    from learned_score.mlsys_runtime import matched_generation_v3 as generation
    original = generation.MatchedKVCacheV3

    class Cache(original):
        def __init__(self, prefix_length, full_prefix_tokens, budget_profile, policy, **kwargs):
            super().__init__(prefix_length, prefix_length if retained_basis else full_prefix_tokens,
                             budget_profile, policy, **kwargs)
            self.full_prefix_tokens = full_prefix_tokens
    generation.MatchedKVCacheV3 = Cache
    try:
        yield
    finally:
        generation.MatchedKVCacheV3 = original


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap)
    ap.add_argument('--olmocr-manifest', type=Path, required=True, help='olmOCR manifest (split final_test or development)')
    ap.add_argument('--split', default='final_test')
    ap.add_argument('--panel', choices=('dev32', 'new32', 'combined64', 'all'), default='combined64')
    ap.add_argument('--cells', nargs='+', default=list(CORE), help='cell names from the panel file (12 available)')
    ap.add_argument('--max-new-tokens', type=int, default=4096)
    ap.add_argument('--training-manifest', type=Path, default=None)
    a = ap.parse_args()
    import torch
    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime, selector
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
    from learned_score.mlsys_eval.olmocr import OlmOCRDataset
    from learned_score.mlsys_scorer.backbone import assert_frozen
    panel = json.loads(PANEL.read_text())
    cells = {c['name']: c for c in panel['cells']}
    entries = json.loads(a.olmocr_manifest.read_text())['splits'][a.split]
    if a.panel != 'all':
        want = [e['row_hash'] for key in (('dev32', 'new32') if a.panel == 'combined64' else (a.panel,)) for e in panel[key]]
        by = {e['row_hash']: e for e in entries}
        entries = [by[r] for r in want if r in by]
    config = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints,
                                 out=Path(a.out)/'joint'/'runtime', training_manifest=a.training_manifest)
    runtime = Runtime(config, a.model)
    dataset = OlmOCRDataset(str(a.olmocr_manifest), allowed_splits=(a.split,))
    runtime.adapter.dataset = dataset
    units = common.shard([(e, cells[c]) for c in a.cells for e in entries], a.shard, a.num_shards)
    for entry, cell in units:
        path = Path(a.out)/'joint'/'rows'/a.model/cell['name']/(entry['row_hash']+'.json')
        if path.exists():
            continue
        sample = dataset.row_for_entry(entry)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        with torch.no_grad():
            prepared = runtime.adapter.prepare_sample(sample, include_answer=False)
            if prepared.answer_positions.numel():
                raise ValueError('joint inference contains answers')
            n = len(prepared.visual_positions); prefix = int(prepared.inputs_embeds.shape[1])
            k = (n*cell['vision']+99)//100
            indices = selector('asymvlm_uniform' if k < n else 'all_tokens', k, runtime.scorer, {})(prepared)
            pruned = runtime.adapter.physically_prune(prepared, indices, 'original')
            retained_prefix = int(pruned.inputs_embeds.shape[1])
            if cell['policy'] == 'full':
                result = generate_dynamic(runtime.adapter.model, pruned, max_new_tokens=a.max_new_tokens, stop_on_eos=True)
            else:
                staging = min(128, cell['profile']//4) if cell['policy'] == 'rkv' else 1
                with capacity_basis(not cell['donate']):
                    result = generate_matched_v3(runtime.adapter.model, pruned, a.max_new_tokens, full_prefix_tokens=prefix,
                                                 budget_profile=cell['profile'], policy=cell['policy'], staging=staging,
                                                 visual_source_positions=prepared.visual_positions, stop_on_eos=True)
                expected = (prefix if cell['donate'] else retained_prefix)+cell['profile']+1
                if result.metrics['cache']['capacity_tokens_per_layer'] != expected:
                    raise ValueError('KV capacity mismatch')
            ids = result.ids[0].cpu().tolist()
            value = dict(request_id=entry['row_hash'], row_hash=entry['row_hash'], group_id=entry['group_id'],
                         image_sha256=entry['image_sha256'], model=a.model, model_tag=a.model, method=cell['name'],
                         status='ok', prediction=runtime.adapter.processor.decode(ids, skip_special_tokens=True).strip(),
                         generated_ids=ids, generated_tokens=len(ids), max_new_tokens=a.max_new_tokens,
                         input_visual_tokens=n, retained_visual_tokens=k, input_prefix_tokens=prefix,
                         retained_prefix_tokens=retained_prefix, metrics=dict(result.metrics),
                         request_seconds=time.perf_counter()-started, peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                         timing_isolated=False, policy=cell, physical_measurement_performed=True)
            assert_frozen(runtime.adapter.model, runtime.versions); assert_frozen(runtime.scorer, runtime.scorer_versions)
            common.write_json(path, value)
            del result, prepared, pruned
        print(json.dumps(dict(model=a.model, cell=cell['name'], request=entry['row_hash'][:12], tokens=len(ids),
                              seconds=round(time.perf_counter()-started, 1))), flush=True)


if __name__ == '__main__':
    main()
