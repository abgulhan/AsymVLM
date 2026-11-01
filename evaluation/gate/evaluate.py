#!/usr/bin/env python
"""Adaptive self-consistency gate, step 2: calibrate and evaluate (2-fold cross-calibration).

Gate: run 10% and 20%; if the normalized answers agree and min(min-token-prob) >= t,
keep the 10% answer, else run at cap K. Thresholds are chosen on one image-group fold
and applied to the other (and vice versa), so no example tunes its own threshold.
Per-level quality comes from the physical uniform benchmark records
(``<out>/records/<model>/<dataset>/asymvlm_uniform/<p>/<id>.json``, plus ``all_tokens/100``);
confidences come from ``evaluation/gate/capture.py``. Scores are normalized per dataset
to the all-token score on the same examples.

Writes ``<report>`` with, per model: the operating points at a target mean keep
(e.g. 30% / 20% kept = 70% / 80% removal), chosen (cap, t) per fold, per-dataset
removal and relative quality.

Example
  python evaluation/gate/evaluate.py --out runs --report results/gate/gate_report.json
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from evaluation import common

LV = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
I = list(LV).index


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def level_quality(out, model, dataset, rid):
    q = {}
    for p in LV:
        method = 'all_tokens' if p == 100 else 'asymvlm_uniform'
        f = Path(out)/'records'/model/common.family(dataset)/method/str(p)/f'{rid}.json'
        if f.exists():
            q[p] = json.loads(f.read_text())['quality_score']
    return q


def load(out, model, manifest, panel):
    panels = common.select_panel(common.load_entries(manifest), panel)
    rows = []
    for d, entries in panels.items():
        for e in entries:
            rid = e['row_hash']; f = Path(out)/'gate'/'records'/model/f'{rid}.json'
            if not f.exists():
                continue
            cells = json.loads(f.read_text())['cells']
            if '10' not in cells or '20' not in cells:
                continue
            q = level_quality(out, model, d, rid)
            if len(q) < len(LV):
                continue
            mt = lambda b: min([s['top1'] for s in (cells[b]['steps'][:-1] or cells[b]['steps'])] or [1])
            rows.append(dict(dataset=common.family(d), group=e['group_id'], Q=[q[p] for p in LV],
                             agree=norm(cells['10']['prediction']) == norm(cells['20']['prediction']),
                             conf=min(mt('10'), mt('20')),
                             fold=int(hashlib.sha256(e['group_id'].encode()).hexdigest(), 16) % 2))
    return rows


def arrays(rows):
    return (np.array([r['Q'] for r in rows], float), np.array([r['dataset'] for r in rows]),
            np.array([r['agree'] for r in rows]), np.array([r['conf'] for r in rows]), np.array([r['fold'] for r in rows]))


def decide(agree, conf, t, cap):
    return np.where(agree & (conf >= t), I(10), I(cap))


def choose_t(Q, agree, conf, mask, cap, eps=None, target_kept=None):
    """eps: min kept s.t. mean positive harm vs uniform-cap <= eps. target_kept: max quality s.t. mean kept <= target."""
    best = None
    for t in np.linspace(0, 1, 401):
        fin = decide(agree[mask], conf[mask], t, cap); q = Q[mask][np.arange(mask.sum()), fin]
        kept = np.mean([LV[i] for i in fin])
        if eps is not None:
            if 100*np.maximum(0, Q[mask, I(cap)]-q).mean() <= eps and (best is None or kept < best[1]):
                best = (t, kept)
        elif kept <= target_kept + 1e-9 and (best is None or q.mean() > best[1]):
            best = (t, q.mean())
    return None if best is None else best[0]


def crossfit(rows, cap, eps=None, target_kept=None):
    Q, ds, agree, conf, fold = arrays(rows)
    fin = np.full(len(rows), I(cap)); ts = []
    for k in (0, 1):
        t = choose_t(Q, agree, conf, fold != k, cap, eps, target_kept)
        if t is None:
            return None, None
        ts.append(t); m = fold == k
        fin[m] = decide(agree[m], conf[m], t, cap)
    return fin, ts


def crossfit_target(rows, target_kept, caps=(30, 40, 50)):
    """Per fold, choose (cap, t) maximizing quality on the other fold with mean kept <= target; apply."""
    Q, ds, agree, conf, fold = arrays(rows)
    fin = np.full(len(rows), I(max(caps))); chosen = []
    for k in (0, 1):
        cal = fold != k; best = None
        for cap in caps:
            if cap <= 10:
                continue
            t = choose_t(Q, agree, conf, cal, cap, target_kept=target_kept)
            if t is None:
                continue
            f = decide(agree[cal], conf[cal], t, cap); q = Q[cal][np.arange(cal.sum()), f].mean()
            if best is None or q > best[0]:
                best = (q, cap, t)
        if best is None:
            return None, None
        _, cap, t = best; chosen.append(dict(fold=k, cap=cap, t=float(t)))
        m = fold == k; fin[m] = decide(agree[m], conf[m], t, cap)
    return fin, chosen


def summarize(rows, fin, datasets):
    Q, ds, *_ = arrays(rows)
    kept = np.array([LV[i] for i in fin], float); q = Q[np.arange(len(Q)), fin]
    per = {}
    for d in datasets:
        m = ds == d
        if m.any():
            per[d] = dict(removal=float(100-kept[m].mean()), relative=float(100*q[m].mean()/Q[m, I(100)].mean()), n=int(m.sum()))
    present = [d for d in datasets if d in per]
    per['average'] = dict(removal=float(np.mean([per[d]['removal'] for d in present])),
                          relative=float(np.mean([per[d]['relative'] for d in present])))
    return per


def uniform(rows, p, datasets):
    return summarize(rows, np.full(len(rows), I(p)), datasets)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=common.OUT, help='root holding records/ and gate/records/')
    ap.add_argument('--manifest', type=Path, default=common.DATA/'manifest.json')
    ap.add_argument('--panel', choices=('first512', 'public'), default='first512')
    ap.add_argument('--models', nargs='+', default=list(common.MODELS))
    ap.add_argument('--targets', nargs='+', type=float, default=[30, 20], help='target mean keep %% (100 - removal)')
    ap.add_argument('--caps', nargs='+', type=int, default=[30, 40, 50])
    ap.add_argument('--report', type=Path, default=Path('results/gate/gate_report.json'))
    a = ap.parse_args()
    datasets = list(common.DATASETS)
    report = dict(panel=a.panel, levels=LV, models={})
    for model in a.models:
        rows = load(a.out, model, a.manifest, a.panel)
        if not rows:
            print('no complete rows for', model); continue
        entry = dict(requests=len(rows), uniform={str(p): uniform(rows, p, datasets) for p in (10, 20, 30, 40)}, gate={})
        for target in a.targets:
            fin, chosen = crossfit_target(rows, target, tuple(a.caps))
            if fin is not None:
                entry['gate'][f'removal{int(100-target)}'] = dict(chosen=chosen, **summarize(rows, fin, datasets),
                    accept10=float(np.mean(fin == I(10))))
        report['models'][model] = entry
        print(model, json.dumps({k: v['average'] for k, v in entry['gate'].items()}), flush=True)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
