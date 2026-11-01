#!/usr/bin/env python
"""Regenerate the quality-vs-compression comparison figure (per benchmark + average).

Input: results/comparison_512/aggregate.json (``make_tables.py aggregate``), optionally a
gate report (``evaluation/gate/evaluate.py``) whose operating points are overlaid as stars.
Output: <dest>/quality_compression_<model>.pdf/.png

Example
  python evaluation/make_figures.py --gate results/gate/gate_report.json --dest figures
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluation import common  # noqa: E402
from evaluation.make_tables import NAMES, MODEL_NAMES  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', type=Path, default=Path('results/comparison_512/aggregate.json'))
    ap.add_argument('--gate', type=Path, default=None)
    ap.add_argument('--dest', type=Path, default=Path('figures'))
    a = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    agg = json.loads(a.source.read_text())
    gate = json.loads(a.gate.read_text())['models'] if a.gate else {}
    a.dest.mkdir(parents=True, exist_ok=True)
    panels = list(common.DATASETS) + ['Average']
    for model, methods in agg.items():
        fig, axes = plt.subplots(1, len(panels), figsize=(3*len(panels), 2.8), sharex=True)
        for ax, d in zip(axes, panels):
            for method, cells in methods.items():
                if method == 'all_tokens':
                    continue
                xs, ys = [], []
                for p in sorted(cells, key=int):
                    vals = [cells[p][k]['relative'] for k in cells[p] if cells[p][k]['relative'] is not None and (d == 'Average' or k == d)]
                    if vals:
                        xs.append(100-int(p)); ys.append(sum(vals)/len(vals))
                ax.plot(xs, ys, marker='o', ms=3, lw=1.6 if method == 'asymvlm_uniform' else 1, label=NAMES.get(method, method))
            for name, g in gate.get(model, {}).get('gate', {}).items():
                pt = g['average'] if d == 'Average' else g.get(d)
                if pt:
                    ax.plot([pt['removal']], [pt['relative']], marker='*', ms=10, color='black', ls='none')
            ax.set_title(d, fontsize=9); ax.set_xlabel('visual tokens removed (%)', fontsize=8); ax.grid(alpha=.3)
        axes[0].set_ylabel('relative to all tokens (%)')
        axes[-1].legend(fontsize=7, loc='lower left')
        fig.suptitle(MODEL_NAMES.get(model, model), fontsize=10); fig.tight_layout()
        for ext in ('pdf', 'png'):
            fig.savefig(a.dest/f'quality_compression_{model}.{ext}', dpi=200)
        plt.close(fig)
        print('wrote', a.dest/f'quality_compression_{model}.pdf')


if __name__ == '__main__':
    main()
