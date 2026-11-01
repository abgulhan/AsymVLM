#!/usr/bin/env python
"""Score joint olmOCR-Bench rows with the official olmOCR-Bench checks.

Reads ``<out>/joint/rows/<model>/<cell>/*.json`` (evaluation/joint/generate.py), runs the
official evaluator (via ``learned_score.mlsys_eval.olmocr_score``) on pages complete for
all cells, and writes a report with the macro score, paired category-cluster bootstrap
CI vs. the all-token/full-history cell, owned KV-cache MiB and savings, and output length.

The official evaluator needs Playwright browsers and an equation-render cache:
  PLAYWRIGHT_BROWSERS_PATH, OLMOCR_EQUATION_CACHE_DIR (default under $ASYMVLM_DATA/olmocr).

Example
  PYTHONPATH=src:. python evaluation/joint/score.py --olmocr-manifest data/olmocr/manifest.json \
      --evaluator-root data/olmocr/olmocr_repo --report results/joint/report_combined64.json
"""
import argparse
import json
import os
from pathlib import Path
from statistics import mean

from evaluation import common
from evaluation.joint.generate import CORE

BASE = CORE[0]


def owned_bytes(row):
    c = row['metrics']['cache']
    return c.get('total_owned_bytes', c['kv_capacity_bytes'] + c.get('source_metadata_bytes', 0) + c.get('auxiliary_persistent_bytes', 0))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=common.OUT)
    ap.add_argument('--olmocr-manifest', type=Path, required=True)
    ap.add_argument('--split', default='final_test')
    ap.add_argument('--evaluator-root', type=Path, required=True, help='checkout of the official olmOCR evaluator')
    ap.add_argument('--cache-patch', type=Path, default=None, help='evaluator cache patch, if used')
    ap.add_argument('--models', nargs='+', default=['qwen25', 'llava_next'])
    ap.add_argument('--cells', nargs='+', default=list(CORE))
    ap.add_argument('--report', type=Path, default=Path('results/joint/report.json'))
    a = ap.parse_args()
    from learned_score.mlsys_sprint.state import digest
    from learned_score.mlsys_eval.common import seal_manifest
    from learned_score.mlsys_eval.olmocr_score import verify_evaluator, score_candidate, paired_category_cluster_ci
    olm = common.DATA/'olmocr'
    os.environ.setdefault('PLAYWRIGHT_BROWSERS_PATH', str(olm/'browsers'))
    os.environ.setdefault('OLMOCR_EQUATION_CACHE_DIR', str(olm/'equations'))
    manifest = json.loads(a.olmocr_manifest.read_text())
    verify_evaluator(str(a.evaluator_root), manifest, str(a.cache_patch) if a.cache_patch else None)
    report = []
    for model in a.models:
        rows_dir = Path(a.out)/'joint'/'rows'/model
        records = {c: {json.loads(p.read_text())['request_id']: json.loads(p.read_text()) for p in (rows_dir/c).glob('*.json')}
                   for c in a.cells}
        entries = [e for e in manifest['splits'][a.split] if all(e['row_hash'] in records[c] for c in a.cells)]
        if not entries:
            print('no complete pages for', model); continue
        ids = [e['row_hash'] for e in entries]
        sub = seal_manifest({**{k: v for k, v in manifest.items() if k != 'manifest_sha256'},
                             'splits': {'final_test': entries}, 'metadata': {**manifest['metadata'], 'sources': len(entries)}})
        scores = {}
        for c in a.cells:
            rows = [{**records[c][i], 'run_context': f'joint-{model}', 'visual_retention': None, 'generated_budget': None} for i in ids]
            dest = Path(a.out)/'joint'/'scored'/model/c/f'n{len(ids)}'
            path = dest/'official_scores.json'
            scores[c] = json.loads(path.read_text()) if path.exists() else score_candidate(
                sub, rows, dest, split='final_test', policy_frozen_sha256=digest(dict(n=len(ids))))
        bm = mean(owned_bytes(records[BASE][i]) for i in ids)
        for c in a.cells:
            rows = [records[c][i] for i in ids]
            try:
                ci = paired_category_cluster_ci(scores[c]['checks'], scores[BASE]['checks'], repetitions=2000)
            except ValueError as exc:
                ci = dict(status='insufficient', reason=str(exc))
            report.append(dict(model=model, cell=c, sources=len(ids), score=100*scores[c]['official_macro_score'],
                               versus_full=ci, mean_owned_cache_MiB=mean(owned_bytes(r) for r in rows)/2**20,
                               cache_saving_percent=100*(1-mean(owned_bytes(r) for r in rows)/bm),
                               mean_output_tokens=mean(r['generated_tokens'] for r in rows),
                               output_limit_percent=100*mean(r['generated_tokens'] == r['max_new_tokens'] for r in rows),
                               mean_request_seconds=mean(r['request_seconds'] for r in rows)))
            print(json.dumps({k: report[-1][k] for k in ('model', 'cell', 'sources', 'score', 'cache_saving_percent')}), flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(dict(rows=report), indent=1, default=str))


if __name__ == '__main__':
    main()
