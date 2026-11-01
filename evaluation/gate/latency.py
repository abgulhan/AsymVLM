"""end-to-end latency of the AsymVLM adaptive self-consistency gate vs uniform-40.

Frozen learned_score runtime (PYTHONPATH=src:.). One process, one GPU, batch 1.
Per request, timed with CUDA synchronization:
  uniform40 : load+preprocess+vision (prepare_sample) -> scorer top-40% -> prune -> generate.
  gate      : prepare_sample ONCE -> 10% pass -> 20% pass -> if answers agree and
              min token prob >= t: output 10% answer; else 40% pass.
Both use generate_bounded with the campaign settings (max_new_tokens 128, slack 1, greedy).
Answers are checked against the recorded physical outputs (parity).
"""
import argparse
import json
import re
import time
from pathlib import Path

from evaluation import common
THRESH = {'qwen25': 0.68, 'llava_next': 0.53, 'phi35': 0.65}   # cap40, 1pp (public calibration half)


def norm(s):
    return re.sub(r'[^a-z0-9]+', ' ', (s or '').lower()).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    common.add_common_args(ap)
    ap.add_argument('--training-manifest', type=Path, default=None)
    ap.add_argument('--panel', choices=('public', 'first512'), default='public'); ap.add_argument('--per-dataset', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=3); ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--threshold', type=float, default=None, help='gate min-token-prob threshold (default: calibrated value)')
    a = ap.parse_args()
    import torch
    from learned_score.mlsys_sprint.manuscript_execute_v35 import Runtime, selector
    from learned_score.mlsys_runtime import generation
    config = common.build_config(a.model, manifest=a.manifest, checkpoints=a.checkpoints, out=Path(a.out)/'latency'/'runtime', training_manifest=a.training_manifest)
    rt = Runtime(config, a.model)
    adapter, scorer = rt.adapter, rt.scorer
    phi = adapter.name == 'phi3_vision'
    original = generation.decoder_forward_function
    probs = []

    def patched(model):
        fwd = original(model)

        def wrapped(**kw):
            out = fwd(**kw)
            if len(probs) < 64:
                x = out.logits[0, -1].float()
                probs.append(float(torch.softmax(x, -1).max()))
            return out
        return wrapped
    generation.decoder_forward_function = patched
    eos = adapter.processor.tokenizer.eos_token_id if hasattr(adapter.processor, 'tokenizer') else None

    def sync():
        torch.cuda.synchronize()

    def one_pass(prepared, pct):
        n = len(prepared.visual_positions); count = (n*pct+99)//100
        probs.clear()
        idx = selector('asymvlm_uniform', count, scorer, {})(prepared)
        pruned = adapter.physically_prune(prepared, idx, 'original')
        res = generation.generate_bounded(adapter.model, pruned, 128, 128, slack=1, stop_on_eos=True,
                                          require_fused_attention=not phi)
        ids = res.ids.detach().cpu()
        text = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
        p = list(probs[:ids.shape[1]])
        ans = p[:-1] if len(p) > 1 else p
        del res, pruned
        return text, min(ans or [1.0])

    def uniform40(raw):
        prepared = adapter.prepare_sample(raw)
        text, _ = one_pass(prepared, 40)
        return text, 40

    def gate(raw):
        prepared = adapter.prepare_sample(raw)
        a10, c10 = one_pass(prepared, 10)
        a20, c20 = one_pass(prepared, 20)
        if norm(a10) == norm(a20) and min(c10, c20) >= (a.threshold or THRESH[a.model]):
            return a10, 10
        a40, _ = one_pass(prepared, 40)
        return a40, 40

    panels = {d: v for d, v in common.select_panel(rt.entries.values(), a.panel).items()}
    entries = [rt.entries[e['row_hash']] for d in sorted(panels) for e in panels[d][-a.per_dataset:]]
    out = Path(a.out)/'latency'/a.model; out.mkdir(parents=True, exist_ok=True)
    rows = []
    with torch.no_grad():
        rt.adapter.dataset = rt.inputs
        for i, e in enumerate(entries[:a.warmup]):
            raw = rt.inputs.row_for_entry(e); uniform40(raw); gate(raw)
        for e in entries:
            raw = rt.inputs.row_for_entry(e)
            rec = {'request_id': e['row_hash'], 'dataset': e['dataset']}
            for name, fn in (('uniform40', uniform40), ('gate', gate)):
                times = []
                for _ in range(a.repeats):
                    if hasattr(raw.get('image'), 'seek'):
                        raw['image'].seek(0)
                    sync(); t0 = time.perf_counter(); text, level = fn(raw); sync(); times.append(time.perf_counter()-t0)
                rec[name] = dict(seconds=times, answer=text, level=level)
            rows.append(rec)
            print(json.dumps(dict(model=a.model, i=len(rows), u40=min(rec['uniform40']['seconds']),
                                  gate=min(rec['gate']['seconds']), level=rec['gate']['level'])), flush=True)
    import numpy as np
    u = np.array([np.median(r['uniform40']['seconds']) for r in rows]); g = np.array([np.median(r['gate']['seconds']) for r in rows])
    lv = np.array([r['gate']['level'] for r in rows])
    summary = dict(model=a.model, requests=len(rows), threshold=(a.threshold or THRESH[a.model]), gpu=torch.cuda.get_device_name(0), uniform40_mean_ms=1000*u.mean(), gate_mean_ms=1000*g.mean(), ratio_mean=float(g.mean()/u.mean()),
                   uniform40_p50_ms=1000*np.median(u), gate_p50_ms=1000*np.median(g),
                   uniform40_p90_ms=1000*np.percentile(u, 90), gate_p90_ms=1000*np.percentile(g, 90),
                   accept_at_10=float((lv == 10).mean()),
                   gate_ms_when_accepted=1000*g[lv == 10].mean() if (lv == 10).any() else None,
                   gate_ms_when_escalated=1000*g[lv == 40].mean() if (lv == 40).any() else None)
    (out/'rows.json').write_text(json.dumps(rows)); (out/'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
