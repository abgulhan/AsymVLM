"""Calibrate the self-consistency gate threshold t (CPU only).

Gate: answer at 10% and 20% visual-token keep; if the normalized answers agree
and min(per-token top-1 probability over both answers) >= t, keep 10%;
otherwise fall back to the uniform cap K (e.g. 30%).

`eps` mode: t = argmin mean kept %  s.t.  mean_i max(0, Q_i[K] - Q_i[gate]) * 100 <= eps.
`target` mode: maximize mean quality s.t. mean kept % <= target (optionally over caps).
With --crossfit, thresholds are fit on one image-group fold (sha256(group) % 2)
and applied to the other, so no request tunes its own threshold.

Input: JSON list or JSONL, one record per request:
  {"dataset": "DocVQA", "group": "<image/document id>",
   "answer_10": "...", "answer_20": "...",
   "top1_10": [p_1, ..., p_n], "top1_20": [...],          # per generated token
   "scores": {"10": s, "20": s, ..., "100": s}}            # benchmark score per keep %
Scores are used as given (normalize per dataset beforehand if desired).

    python training/calibrate_adaptive_gate.py records.jsonl --cap 30 --eps 0.5 --crossfit
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

LEVELS = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
I = list(LEVELS).index


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def min_top1(probs):
    # The final (EOS) step is excluded when there is more than one step.
    steps = probs[:-1] or probs
    return min(steps) if steps else 1.0


def load_rows(path):
    text = Path(path).read_text().strip()
    raw = json.loads(text) if text.startswith("[") else [json.loads(l) for l in text.splitlines() if l.strip()]
    rows = []
    for r in raw:
        if not all(str(level) in r["scores"] for level in LEVELS):
            continue
        group = str(r.get("group", r.get("image_id", "")))
        rows.append(dict(dataset=r.get("dataset", "all"), group=group,
                         Q=[float(r["scores"][str(level)]) for level in LEVELS],
                         agree=norm(r["answer_10"]) == norm(r["answer_20"]),
                         conf=min(min_top1(list(r["top1_10"])), min_top1(list(r["top1_20"]))),
                         fold=int(hashlib.sha256(group.encode()).hexdigest(), 16) % 2))
    if not rows:
        raise ValueError("no complete records")
    return rows


def arrays(rows):
    return (np.array([r["Q"] for r in rows], float), np.array([r["dataset"] for r in rows]),
            np.array([r["agree"] for r in rows]), np.array([r["conf"] for r in rows]),
            np.array([r["fold"] for r in rows]))


def decide(agree, conf, t, cap):
    return np.where(agree & (conf >= t), I(10), I(cap))


def choose_t(Q, agree, conf, mask, cap, eps=None, target_kept=None, grid=401):
    best = None
    for t in np.linspace(0, 1, grid):
        fin = decide(agree[mask], conf[mask], t, cap)
        q = Q[mask][np.arange(mask.sum()), fin]
        kept = np.mean([LEVELS[i] for i in fin])
        if eps is not None:
            harm = 100 * np.maximum(0, Q[mask, I(cap)] - q).mean()
            if harm <= eps and (best is None or kept < best[1]):
                best = (t, kept)
        elif kept <= target_kept + 1e-9 and (best is None or q.mean() > best[1]):
            best = (t, q.mean())
    return None if best is None else float(best[0])


def crossfit(rows, cap, eps=None, target_kept=None):
    Q, _, agree, conf, fold = arrays(rows)
    fin = np.full(len(rows), I(cap)); ts = []
    for k in (0, 1):
        t = choose_t(Q, agree, conf, fold != k, cap, eps, target_kept)
        if t is None:
            return None, None
        ts.append(t); m = fold == k
        fin[m] = decide(agree[m], conf[m], t, cap)
    return fin, ts


def crossfit_target(rows, target_kept, caps=(30, 40, 50)):
    Q, _, agree, conf, fold = arrays(rows)
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


def summarize(rows, fin):
    Q, ds, *_ = arrays(rows)
    kept = np.array([LEVELS[i] for i in fin], float); q = Q[np.arange(len(Q)), fin]
    out = {}
    for d in sorted(set(ds)):
        m = ds == d
        out[d] = dict(n=int(m.sum()), mean_kept=float(kept[m].mean()), removal=float(100 - kept[m].mean()),
                      relative_to_full=float(100 * q[m].mean() / max(Q[m, I(100)].mean(), 1e-12)),
                      gate_10pct_rate=float((fin[m] == I(10)).mean()))
    out["all"] = dict(n=len(rows), mean_kept=float(kept.mean()), mean_quality=float(q.mean()))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("records")
    p.add_argument("--cap", type=int, default=30, choices=[l for l in LEVELS if l > 10])
    p.add_argument("--eps", type=float, help="allowed mean positive harm vs uniform cap (score points x100)")
    p.add_argument("--target-kept", type=float, help="target mean kept %% instead of --eps")
    p.add_argument("--caps", type=int, nargs="+", default=[30, 40, 50], help="caps searched in target mode")
    p.add_argument("--crossfit", action="store_true", help="2-fold image-group cross-calibration")
    p.add_argument("--output", type=Path)
    a = p.parse_args(argv)
    if (a.eps is None) == (a.target_kept is None):
        p.error("give exactly one of --eps / --target-kept")
    rows = load_rows(a.records)
    if a.crossfit:
        if a.target_kept is not None:
            fin, chosen = crossfit_target(rows, a.target_kept, tuple(a.caps))
        else:
            fin, chosen = crossfit(rows, a.cap, eps=a.eps)
        if fin is None:
            raise SystemExit("no feasible threshold on some fold")
        result = dict(mode="crossfit", thresholds=chosen, summary=summarize(rows, fin))
    else:
        Q, _, agree, conf, _ = arrays(rows)
        mask = np.ones(len(rows), bool)
        t = choose_t(Q, agree, conf, mask, a.cap, a.eps, a.target_kept)
        if t is None:
            raise SystemExit("no feasible threshold")
        result = dict(mode="full", cap=a.cap, t=t, summary=summarize(rows, decide(agree, conf, t, a.cap)))
    text = json.dumps(result, indent=1)
    print(text)
    if a.output:
        a.output.write_text(text)


if __name__ == "__main__":
    main()
