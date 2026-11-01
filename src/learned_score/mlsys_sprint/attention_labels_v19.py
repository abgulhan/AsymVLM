"""Small, training-only attention labels; no runtime attention or hard sink deletion."""
import math


def probability(values):
    import torch
    x = torch.as_tensor(values).double()
    if x.ndim != 1 or not len(x) or not torch.isfinite(x).all() or (x < 0).any() or x.sum() <= 0:
        raise ValueError('finite nonnegative nonempty attention with positive mass required')
    return x / x.sum()


def middle_layers(count):
    if type(count) is not int or count < 4:
        raise ValueError('at least four decoder layers required')
    return list(range(count // 4, 3 * count // 4))


def cells(rectangles):
    """Native rectangles in [0,1000], not flattened token-index bins."""
    import torch
    r = torch.as_tensor(rectangles).double()
    if r.ndim != 2 or r.shape[1] != 4 or not len(r) or not torch.isfinite(r).all() or (r < 0).any() or (r > 1000).any() or (r[:, :2] >= r[:, 2:]).any():
        raise ValueError('valid native patch rectangles required')
    xy = (((r[:, :2] + r[:, 2:]) / 2000) * 8).long().clamp(0, 7)
    return xy[:, 1] * 8 + xy[:, 0]


def position_bias(records, exclude_image=None):
    """Equal-image mean relative attention; omit every row of the held-out image."""
    import torch
    sums, counts = torch.zeros(64, dtype=torch.float64), torch.zeros(64, dtype=torch.float64)
    seen = set()
    for record in records:
        if record['image_sha256'] == exclude_image:
            continue
        if record['image_sha256'] in seen:
            raise ValueError('bias fit requires one training question per image')
        seen.add(record['image_sha256'])
        p, bins = probability(record['raw_attention']), cells(record['patch_rectangles'])
        if len(p) != len(bins):
            raise ValueError('attention and native geometry differ')
        for b in bins.unique().tolist():
            sums[b] += (len(p) * p[bins == b]).mean()
            counts[b] += 1
    if not seen:
        raise ValueError('leave-image-out bias requires another training image')
    return torch.where(counts > 0, sums / counts.clamp_min(1), torch.ones_like(sums))


def corrected_attention(record, bias):
    import torch
    b = torch.as_tensor(bias).double()
    if b.shape != (64,) or not torch.isfinite(b).all() or (b < 0).any():
        raise ValueError('finite nonnegative 8x8 training bias required')
    return probability(probability(record['raw_attention']) / b[cells(record['patch_rectangles'])].clamp_min(1))


def gradient_target(saliency):
    import torch
    x = torch.as_tensor(saliency).float()
    if x.ndim != 1 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('finite cached gradient labels required')
    return ((x - x.mean()) / x.std(unbiased=False).clamp_min(1e-6)).softmax(0)


def risk_curve(scores, temperature=None):
    import torch
    from .state import digest
    x = torch.as_tensor(scores).double()
    if x.ndim != 1 or not len(x) or not torch.isfinite(x).all():
        raise ValueError('finite one-dimensional scorer output required')
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise ValueError('positive training-only temperature required')
    scale = x.std(unbiased=False).clamp_min(1e-8) if temperature is None else temperature
    weights = ((x - x.mean()) / scale).softmax(0)
    order = torch.argsort(x, descending=True, stable=True)
    removed = torch.cat((weights[order].flip(0).cumsum(0).flip(0), weights.new_zeros(1)))
    values = removed.clamp(0, 1).tolist()
    values[0], values[-1] = 1., 0.
    return dict(removed_weight_by_count=values, weights_sha256=digest(weights.tolist()),
                ranking_sha256=digest(order.tolist()),
                candidate_counts=sorted(set((len(x)*p+99)//100 for p in (50,60,70,80,90,100))))


def screen_gate(rows, expected_ids):
    """Four predeclared cells; choose one stronger control globally, never per row."""
    arms = ('current', 'gradient', 'raw', 'corrected')
    by = {}
    for row in rows:
        method = row['method']
        if not any(method == f'{arm}_{p:03d}' for arm in arms for p in (50,70)):
            continue
        key = (row['request_id'], method)
        if key in by or row.get('status') != 'ok':
            raise ValueError('duplicate or failed screening measurement')
        q = row.get('quality_score')
        if type(q) not in (int, float) or not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError('invalid physical quality')
        by[key] = row
    if len(set(expected_ids)) != len(expected_ids) or not expected_ids:
        raise ValueError('unique nonempty screening panel required')
    expected = {(rid,f'{arm}_{p:03d}') for rid in expected_ids for arm in arms for p in (50,70)}
    if set(by) != expected:
        raise ValueError('complete paired screening matrix required')
    scores = {arm:{} for arm in arms}
    for rid in expected_ids:
        for p in (50,70):
            reference = by[rid,f'current_{p:03d}']
            for arm in arms:
                row = by[rid,f'{arm}_{p:03d}']
                for field in ('dataset','image_sha256','group_id','input_visual_tokens','input_text_tokens','cache_contract','selected_count'):
                    if row[field] != reference[field]:
                        raise ValueError('unmatched screening geometry/count: '+field)
                key = f"{row['dataset']}:{p}"
                scores[arm].setdefault(key, []).append(100*row['quality_score'])
    means = {a:{k:sum(v)/len(v) for k,v in groups.items()} for a,groups in scores.items()}
    overall = {a:sum(groups.values())/len(groups) for a,groups in means.items()}
    control = max(('current','gradient'), key=lambda a:(overall[a], a=='current'))
    candidates = {}
    for arm in ('raw','corrected'):
        delta = {k:means[arm][k]-means[control][k] for k in means[control]}
        gain = sum(delta.values())/len(delta)
        candidates[arm] = dict(mean_gain_pp=gain, cell_delta_pp=delta,
                               passed=gain >= 1-1e-10 and min(delta.values()) >= -1-1e-10)
    passing = [a for a in candidates if candidates[a]['passed']]
    winner = max(passing,key=lambda a:(overall[a],a=='raw')) if passing else None
    return dict(winner=winner,stronger_control=control,means=means,candidates=candidates,
                evidence='development_screening_not_significance_test')
