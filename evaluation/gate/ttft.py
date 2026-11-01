"""TTFT/E2E timing for the teaser: all tokens, uniform-30/20 (70/80% removal), adaptive gate.

Frozen learned_score runtime, batch 1, CUDA-synchronized host timers. For each request and repeat:
  full     : prepare -> prefill all tokens -> generate.   TTFT = prepare + prefill to first token.
  u30, u20 : prepare -> select top-k -> prune -> generate.
  gate70/80: prepare once -> 10% pass -> 20% pass -> accept 10% if answers agree & min prob >= t,
             else cap pass. TTFT = time until the FIRST TOKEN OF THE EMITTED ANSWER is available
             (after the decision; for escalations, after the cap pass's first token).
Writes per-request rows + summary with paired bootstrap 95% CIs of TTFT/E2E reduction vs all tokens.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np

from evaluation import common


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap)
    ap.add_argument('--training-manifest', type=Path, default=None)
    ap.add_argument('--panel', choices=('public', 'first512'), default='public'); ap.add_argument('--per-dataset', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=3); ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--gate70', required=True, help='cap,t'); ap.add_argument('--gate80', required=True, help='cap,t')
    a = ap.parse_args()
    import torch
    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime, selector
    from learned_score.mlsys_runtime import generation
    cfg = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints, out=Path(a.out)/'ttft'/'runtime', training_manifest=a.training_manifest)
    rt = Runtime(cfg, a.model); adapter, scorer = rt.adapter, rt.scorer; phi = adapter.name == 'phi3_vision'
    original = generation.decoder_forward_function; probs = []

    def patched(model):
        fwd = original(model)

        def wrapped(**kw):
            out = fwd(**kw)
            if len(probs) < 64:
                probs.append(float(torch.softmax(out.logits[0, -1].float(), -1).max()))
            return out
        return wrapped
    generation.decoder_forward_function = patched
    sync = torch.cuda.synchronize

    def run_pass(prepared, pct, t_start):
        """Returns (text, min_prob, ttft_from_t_start, end_from_t_start)."""
        n = len(prepared.visual_positions); count = n if pct == 100 else (n*pct+99)//100
        probs.clear(); first = {}
        idx = selector('asymvlm_uniform', count, scorer, {})(prepared)
        pruned = adapter.physically_prune(prepared, idx, 'original')
        res = generation.generate_bounded(adapter.model, pruned, 128, 128, slack=1, stop_on_eos=True,
                                          require_fused_attention=not phi,
                                          on_first_token=lambda: first.setdefault('t', time.perf_counter()))
        sync(); end = time.perf_counter()
        ids = res.ids.detach().cpu(); text = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
        p = probs[:ids.shape[1]]; ans = p[:-1] if len(p) > 1 else p
        del res, pruned
        return text, min(ans or [1.0]), first['t']-t_start, end-t_start

    def fixed(raw, pct):
        sync(); t0 = time.perf_counter()
        prepared = adapter.prepare_sample(raw)
        _, _, ttft, e2e = run_pass(prepared, pct, t0)
        return ttft, e2e, pct

    def gate(raw, cap, t):
        sync(); t0 = time.perf_counter()
        prepared = adapter.prepare_sample(raw)
        a10, c10, _, _ = run_pass(prepared, 10, t0)
        a20, c20, _, e20 = run_pass(prepared, 20, t0)
        if norm(a10) == norm(a20) and min(c10, c20) >= t:
            return e20, e20, 10            # answer known after the decision; emitted immediately
        _, _, ttft, e2e = run_pass(prepared, cap, t0)
        return ttft, e2e, cap

    g70 = tuple(map(float, a.gate70.split(','))); g80 = tuple(map(float, a.gate80.split(',')))
    panels = {d: v for d, v in common.select_panel(rt.entries.values(), a.panel).items()}
    entries = [rt.entries[e['row_hash']] for d in sorted(panels) for e in panels[d][:a.per_dataset]]
    fns = {'full': lambda r: fixed(r, 100), 'u30': lambda r: fixed(r, 30), 'u20': lambda r: fixed(r, 20),
           'gate70': lambda r: gate(r, int(g70[0]), g70[1]), 'gate80': lambda r: gate(r, int(g80[0]), g80[1])}
    rows = []
    with torch.no_grad():
        rt.adapter.dataset = rt.inputs
        for e in entries[:a.warmup]:
            raw = rt.inputs.row_for_entry(e)
            for f in fns.values():
                if hasattr(raw.get('image'), 'seek'):
                    raw['image'].seek(0)
                f(raw)
        for e in entries:
            raw = rt.inputs.row_for_entry(e); rec = dict(request_id=e['row_hash'], dataset=e['dataset'])
            for name, f in fns.items():
                tt, ee, lv = [], [], None
                for _ in range(a.repeats):
                    if hasattr(raw.get('image'), 'seek'):
                        raw['image'].seek(0)
                    t, x, lv = f(raw); tt.append(t); ee.append(x)
                rec[name] = dict(ttft=float(np.median(tt)), e2e=float(np.median(ee)), level=lv)
            rows.append(rec); print(json.dumps(dict(i=len(rows), **{k: round(rec[k]['ttft']*1000) for k in fns})), flush=True)
    rng = np.random.default_rng(0)

    def red(key, metric):
        f = np.array([r['full'][metric] for r in rows]); x = np.array([r[key][metric] for r in rows])
        point = 100*(1-x.mean()/f.mean()); boots = []
        for _ in range(2000):
            i = rng.integers(0, len(rows), len(rows)); boots.append(100*(1-x[i].mean()/f[i].mean()))
        return dict(full_ms=1000*f.mean(), method_ms=1000*x.mean(), reduction_percent=point,
                    ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])
    summary = dict(model=a.model, requests=len(rows), gate70=g70, gate80=g80, gpu=torch.cuda.get_device_name(0),
                   ttft={k: red(k, 'ttft') for k in ('u30', 'u20', 'gate70', 'gate80')},
                   e2e={k: red(k, 'e2e') for k in ('u30', 'u20', 'gate70', 'gate80')},
                   gate70_accept10=float(np.mean([r['gate70']['level'] == 10 for r in rows])),
                   gate80_accept10=float(np.mean([r['gate80']['level'] == 10 for r in rows])))
    out = Path(a.out)/'ttft'/a.model; out.mkdir(parents=True, exist_ok=True)
    (out/'rows.json').write_text(json.dumps(rows)); (out/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
