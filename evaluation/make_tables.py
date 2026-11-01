#!/usr/bin/env python
"""Regenerate the paper tables from run outputs / result JSONs.

Subcommands
  aggregate  per-request benchmark records -> results/comparison_512/aggregate.json
             (mean score per model/method/keep/dataset, relative-to-all-tokens, n)
  quality    aggregate.json -> LaTeX quality table at one keep %% (paper: quality30/50/70)
  gate       gate report (evaluation/gate/evaluate.py) -> adaptive table
  joint      joint report (evaluation/joint/score.py) -> joint olmOCR table
  systems    systems measurements (evaluation/systems/measure.py) -> systems table (batch 1)

Examples
  python evaluation/make_tables.py aggregate --out runs
  python evaluation/make_tables.py quality --keep 30 > tables/quality70.tex   # 70% removed
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import common  # noqa: E402

NAMES = {'asymvlm_uniform': 'AsymVLM', 'all_tokens': 'All tokens', 'divprune': 'DivPrune', 'cdpruner': 'CDPruner',
         'visionzip_native': 'VisionZip', 'visionzip_llava_native': 'VisionZip', 'visionzip_phi_native': 'VisionZip',
         'fastv_native': 'FastV', 'vlcache_native': 'VL-Cache'}
MODEL_NAMES = {'qwen25': 'Qwen2.5-VL-3B', 'llava_next': 'LLaVA-NeXT-7B', 'phi35': 'Phi-3.5-Vision'}


def aggregate(a):
    scores = defaultdict(list)
    for f in Path(a.out, 'records').glob('*/*/*/*/*.json'):
        r = json.loads(f.read_text())
        if r.get('status') != 'ok':
            continue
        model, dataset, method, percent = f.parts[-5:-1]
        scores[(model, method, int(percent), dataset)].append((r['request_id'], float(r['quality_score'])))
    out = defaultdict(dict)
    for (model, method, p, d), v in scores.items():
        base = dict(scores.get((model, 'all_tokens', 100, d), []))
        ids = [i for i, _ in v if i in base]
        mean = statistics.mean(s for _, s in v)
        rel = 100*statistics.mean(dict(v)[i] for i in ids)/statistics.mean(base[i] for i in ids) if ids and statistics.mean(base[i] for i in ids) else None
        out[model].setdefault(method, {}).setdefault(str(p), {})[d] = dict(score=100*mean, relative=rel, n=len(v))
    a.dest.parent.mkdir(parents=True, exist_ok=True)
    a.dest.write_text(json.dumps(out, indent=1))
    print('wrote', a.dest)


def quality(a):
    agg = json.loads(a.source.read_text())
    ds = list(common.DATASETS)
    lines = [r'\begin{tabular}{ll' + 'c'*(len(ds)+1) + '}', r'\toprule',
             'Model & Method & ' + ' & '.join(ds) + r' & Avg. \\', r'\midrule']
    for model in common.MODELS:
        for method, cells in agg.get(model, {}).items():
            cell = cells.get('100' if method == 'all_tokens' else str(a.keep))
            if not cell:
                continue
            vals = [cell.get(d, {}).get('score') for d in ds]
            avg = statistics.mean(v for v in vals if v is not None) if any(v is not None for v in vals) else None
            fmt = lambda v: '--' if v is None else f'{v:.1f}'
            lines.append(f'{MODEL_NAMES[model]} & {NAMES.get(method, method)} & ' + ' & '.join(map(fmt, vals)) + f' & {fmt(avg)} \\\\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'; lines.append(r'\end{tabular}')
    print('\n'.join(lines))


def gate(a):
    rep = json.loads(a.source.read_text())
    print(r'\begin{tabular}{llccc}', r'\toprule', r'Model & Operating point & Removal (\%) & Rel. quality (\%) & Accept@10 \\', r'\midrule', sep='\n')
    for model, m in rep['models'].items():
        for name, g in m['gate'].items():
            print(f"{MODEL_NAMES[model]} & gate {name} & {g['average']['removal']:.1f} & {g['average']['relative']:.1f} & {100*g['accept10']:.0f}\\% \\\\")
        for p in ('30', '20'):
            u = m['uniform'][p]['average']
            print(f"{MODEL_NAMES[model]} & uniform {p}\\% kept & {u['removal']:.1f} & {u['relative']:.1f} & -- \\\\")
    print(r'\bottomrule', r'\end{tabular}', sep='\n')


def joint(a):
    rep = json.loads(a.source.read_text())
    print(r'\begin{tabular}{llccc}', r'\toprule', r'Model & Cell & olmOCR score & KV saving (\%) & Output tokens \\', r'\midrule', sep='\n')
    for r in rep['rows']:
        print(f"{MODEL_NAMES[r['model']]} & {r.get('cell', r.get('method'))} & {r['score']:.1f} & {r['cache_saving_percent']:.1f} & {r['mean_output_tokens']:.0f} \\\\")
    print(r'\bottomrule', r'\end{tabular}', sep='\n')


def systems(a):
    print(r'\begin{tabular}{llcccc}', r'\toprule', r'Model & Method & TTFT (ms) & E2E (ms) & KV (MiB) & Peak (GiB) \\', r'\midrule', sep='\n')
    for model in common.MODELS:
        rows = defaultdict(list)
        for f in sorted(Path(a.out, 'systems', model).glob('b1-*.json')):
            r = json.loads(f.read_text())
            if r.get('status') == 'ok':
                rows[r['method']].extend(r['measurements'])
        for method, ms in rows.items():
            med = lambda key: statistics.median(m['metrics'].get(key) or m['metrics'].get('timing', {}).get(key, float('nan')) for m in ms)
            kv = statistics.median(m['metrics'].get('kv_active_bytes', m['metrics'].get('runtime', {}).get('kv_active_bytes', float('nan'))) for m in ms)
            peak = statistics.median(m['metrics'].get('end_to_end_peak_allocated_bytes', m['metrics'].get('peak_allocated_bytes', float('nan'))) for m in ms)
            print(f"{MODEL_NAMES[model]} & {NAMES.get(method, method)} & {1000*med('ttft_seconds'):.0f} & {1000*med('end_to_end_seconds'):.0f} & {kv/2**20:.0f} & {peak/2**30:.1f} \\\\")
    print(r'\bottomrule', r'\end{tabular}', sep='\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('aggregate'); s.add_argument('--out', type=Path, default=common.OUT)
    s.add_argument('--dest', type=Path, default=Path('results/comparison_512/aggregate.json')); s.set_defaults(fn=aggregate)
    s = sub.add_parser('quality'); s.add_argument('--source', type=Path, default=Path('results/comparison_512/aggregate.json'))
    s.add_argument('--keep', type=int, default=30); s.set_defaults(fn=quality)
    s = sub.add_parser('gate'); s.add_argument('--source', type=Path, default=Path('results/gate/gate_report.json')); s.set_defaults(fn=gate)
    s = sub.add_parser('joint'); s.add_argument('--source', type=Path, default=Path('results/joint/report.json')); s.set_defaults(fn=joint)
    s = sub.add_parser('systems'); s.add_argument('--out', type=Path, default=common.OUT); s.set_defaults(fn=systems)
    a = ap.parse_args(); a.fn(a)


if __name__ == '__main__':
    main()
