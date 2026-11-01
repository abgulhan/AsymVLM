"""Frozen-scorer, answer-free prediction of physical compression damage.

Reference-answer quality is supervision/calibration only. No predictor changes
the scorer, its ordering, native geometry, or the candidate retention menu.
"""
from collections import Counter
import math

import numpy as np

from .compression_matrix_v8 import seal
from .state import digest

FORMAT = 'mlsys_budget_predictor_v31'
GRID = tuple(range(10, 101, 10))
FAMILIES = ('scalar', 'trees', 'head')
FEATURES = ('fixed_entropy', 'standardized_entropy', 'fixed_mass_10_30',
    'standardized_mass_10_30', 'fixed_mass_top10', 'standardized_mass_top70',
    'score_std', 'score_excess_kurtosis', 'log_visual_tokens', 'log_question_tokens',
    'pooled_alignment_drop', 'pooled_next20_norm')


def counts(n):
    if type(n) is not int or n < 1:
        raise ValueError('positive native visual-token count required')
    return np.asarray([(n*p+99)//100 for p in GRID], dtype=np.int64)


def _cosine(a, b):
    return float(np.dot(a, b)/max(1e-15, np.linalg.norm(a)*np.linalg.norm(b)))


def vector(feature, temperature):
    if feature.get('answer_inputs') is not False or not math.isfinite(temperature) or temperature <= 0:
        raise ValueError('answer-free input and positive training temperature required')
    s = np.asarray(feature['scores_sorted'], dtype=np.float64)
    pooled = np.asarray(feature['pooled96'], dtype=np.float64)
    n = feature['input_visual_tokens']
    if (s.shape != (n,) or pooled.shape != (96,) or not np.isfinite(s).all()
            or not np.isfinite(pooled).all() or np.any(np.diff(s) > 0)):
        raise ValueError('finite sorted native scores and 96 frozen pooled features required')
    sd = float(s.std())
    def stats(t):
        w = np.exp((s-s.max())/t); w /= w.sum()
        k = counts(n)
        entropy = -float(np.sum(w*np.log(np.maximum(w, 1e-300))))/math.log(n) if n > 1 else 0.
        return entropy, float(w[k[0]:k[2]].sum()), float(w[:k[0]].sum()), float(w[:k[6]].sum())
    f, z = stats(temperature), stats(max(sd, 1e-8))
    top, extra, query = pooled[:32], pooled[32:64], pooled[64:96]
    x = np.asarray([f[0], z[0], f[1], z[1], f[2], z[3], sd,
        float(np.mean(((s-s.mean())/max(sd, 1e-15))**4)-3),
        math.log(n), math.log1p(feature['input_question_tokens']),
        _cosine(top, query)-_cosine(extra, query), float(np.linalg.norm(extra))])
    if not np.isfinite(x).all():
        raise ValueError('nonfinite budget statistics')
    return x, np.concatenate((pooled, x))


def feature_inputs(scorer, visual, query, prompt_count, *, scores=None):
    import torch
    from torch.nn import functional as F
    if not len(visual) or not len(query) or getattr(scorer, 'rank', None) != 32:
        raise ValueError('nonempty native patches/questions and frozen rank-32 scorer required')
    if getattr(scorer, 'mode', 'retrained') != 'retrained':
        raise ValueError('only the frozen query-aware ranking is in scope')
    with torch.no_grad():
        scores = scorer(visual, query).float() if scores is None else scores.float()
        if scores.shape != (len(visual),):
            raise ValueError('precomputed scores must cover the native visual tokens')
        order = torch.argsort(scores, descending=True, stable=True)
        v = F.normalize(F.linear(F.normalize(visual.float(), dim=-1), scorer.visual.weight.float()), dim=-1)
        q = F.normalize(F.linear(F.normalize(query.float(), dim=-1), scorer.question.weight.float()), dim=-1)
        low, high = int(counts(len(v))[2]), int(counts(len(v))[4])
        top = v[order[:low]].mean(0)
        extra = v[order[low:high]].mean(0) if high > low else torch.zeros_like(top)
        pooled = torch.cat((top, extra, q.mean(0)))
        if not torch.isfinite(scores).all() or not torch.isfinite(pooled).all():
            raise ValueError('nonfinite scorer features')
    return dict(answer_inputs=False, input_visual_tokens=len(visual),
        input_text_tokens=int(prompt_count), input_question_tokens=len(query),
        scores_sorted=scores[order].double().cpu().tolist(), pooled96=pooled.double().cpu().tolist(),
        ranking_sha256=digest(order.cpu().tolist()))


def _tree_fit(x, y, depth=0):
    mean = y.mean(0)
    result = dict(value=mean.tolist(), samples=len(x))
    if depth >= 2 or len(x) < 64:
        return result
    total = y.sum(0); base = float((total*total).sum()/len(y)); best = None
    for col in range(x.shape[1]):
        order = np.argsort(x[:, col], kind='stable'); sx = x[order, col]
        cy = y[order].cumsum(0)
        ix = np.arange(31, len(x)-32)
        if not len(ix):
            continue
        gain = (cy[ix]**2).sum(1)/(ix+1) + ((total-cy[ix])**2).sum(1)/(len(x)-ix-1)-base
        gain[sx[ix] == sx[ix+1]] = -np.inf
        pos = int(np.argmax(gain)); i = int(ix[pos]); g = float(gain[pos])
        threshold = float(sx[i]+(sx[i+1]-sx[i])/2)
        key = (g, -col, -threshold)
        if g > 1e-12 and (best is None or key > best[0]):
            best = key, col, threshold
    if best is None:
        return result
    _, col, threshold = best; left = x[:, col] <= threshold
    return dict(column=col, threshold=threshold, samples=len(x),
        left=_tree_fit(x[left], y[left], depth+1), right=_tree_fit(x[~left], y[~left], depth+1))


def _tree_predict(tree, x):
    if 'value' in tree:
        return np.tile(tree['value'], (len(x), 1))
    if not len(x):
        node = tree
        while 'value' not in node:
            node = node['left']
        return np.empty((0, len(node['value'])))
    left = x[:, tree['column']] <= tree['threshold']
    a, b = _tree_predict(tree['left'], x[left]), _tree_predict(tree['right'], x[~left])
    out = np.empty((len(x), a.shape[1])); out[left], out[~left] = a, b
    return out


def fit(features, qualities, family, checkpoint_sha256, temperature=None):
    if family not in FAMILIES or len(features) != len(qualities) or not features:
        raise ValueError('registered family and paired complete training examples required')
    if len({f['group_id'] for f in features}) != len(features):
        raise ValueError('one training question per image/document group required')
    q = np.asarray(qualities, dtype=np.float64)
    if q.shape != (len(features), 10) or not np.isfinite(q).all() or np.any((q < 0)|(q > 1)):
        raise ValueError('ten physical answer-quality labels per training example required')
    temperature = float(temperature or max(1e-8, np.median([np.std(f['scores_sorted']) for f in features])))
    pairs = [vector(f, temperature) for f in features]
    x = np.stack([p[int(family == 'head')] for p in pairs])
    y = np.maximum(0, q[:, -1:]-q[:, :-1])
    mean, scale = x.mean(0), x.std(0); scale = np.where(scale > 1e-8, scale, 1.)
    z = (x-mean)/scale
    result = dict(format=FORMAT, family=family, checkpoint_sha256=checkpoint_sha256,
        temperature=temperature, feature_names=list(FEATURES), menu=list(GRID), answer_inputs=False,
        ranking_frozen=True, training_ids=[f['request_id'] for f in features],
        training_groups=[f['group_id'] for f in features],
        target='max(0,full_answer_quality-compressed_answer_quality)',
        mean=mean.tolist(), scale=scale.tolist(), quality_monotonicity_assumed=False)
    if family == 'scalar':
        cuts = np.unique(np.quantile(x[:, 2], [.2, .4, .6, .8]))
        bins = np.searchsorted(cuts, x[:, 2], side='right')
        means = [y[bins == i].mean(0) if np.any(bins == i) else y.mean(0) for i in range(len(cuts)+1)]
        result.update(cuts=cuts.tolist(), means=np.asarray(means).tolist(), scalar_column=2,
                      bin_counts=[int((bins == i).sum()) for i in range(len(cuts)+1)])
    elif family == 'trees':
        initial = y.mean(0); prediction = np.tile(initial, (len(y), 1)); trees = []
        for _ in range(16):
            tree = _tree_fit(z, y-prediction); trees.append(tree)
            prediction += .1*_tree_predict(tree, z)
        result.update(initial=initial.tolist(), trees=trees, learning_rate=.1)
    else:
        import torch
        from torch import nn
        torch.manual_seed(42); torch.set_num_threads(2)
        net = nn.Sequential(nn.Linear(108, 32), nn.ReLU(), nn.Linear(32, 9))
        optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
        inputs, target = torch.tensor(z, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)
        generator = torch.Generator().manual_seed(42)
        for _ in range(100):
            for ix in torch.randperm(len(x), generator=generator).split(64):
                optimizer.zero_grad(set_to_none=True)
                loss = nn.functional.mse_loss(net(inputs[ix]), target[ix])
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite neural budget loss')
                loss.backward(); optimizer.step()
        result.update(w1=net[0].weight.detach().double().tolist(), b1=net[0].bias.detach().double().tolist(),
            w2=net[2].weight.detach().double().tolist(), b2=net[2].bias.detach().double().tolist(),
            recipe=dict(architecture=[108,32,9], seed=42, epochs=100, batch_size=64,
                        lr=.001, weight_decay=.01, loss='MSE_actual_positive_quality_damage'))
    return seal(result)


def validate(model):
    if (model.get('format') != FORMAT or model.get('family') not in FAMILIES
            or model.get('menu') != list(GRID) or model.get('answer_inputs') is not False
            or model.get('sha256') != digest({k:v for k,v in model.items() if k != 'sha256'})):
        raise ValueError('invalid or changed frozen budget predictor')


def predict(model, features):
    validate(model)
    if not features:
        return np.empty((0, 9))
    family = model['family']
    x = np.stack([vector(f, model['temperature'])[int(family == 'head')] for f in features])
    z = (x-np.asarray(model['mean']))/np.asarray(model['scale'])
    if family == 'scalar':
        y = np.asarray(model['means'])[np.searchsorted(model['cuts'], x[:, 2], side='right')]
    elif family == 'trees':
        y = np.tile(model['initial'], (len(x), 1))
        for tree in model['trees']:
            y += model['learning_rate']*_tree_predict(tree, z)
    else:
        # Rowwise reductions keep decisions independent of worker batching.
        w1, b1, w2, b2 = (np.asarray(model[k]) for k in ('w1','b1','w2','b2'))
        y = np.stack([np.sum(w2*np.maximum(0, np.sum(w1*r[None,:], axis=1)+b1)[None,:], axis=1)+b2 for r in z])
    if y.shape != (len(x), 9) or not np.isfinite(y).all():
        raise ValueError('invalid predicted loss surface')
    return np.clip(y, 0, 1)


def choices(prediction, threshold):
    if not math.isfinite(threshold):
        raise ValueError('finite frozen cutoff required')
    passed = np.asarray(prediction) <= threshold
    return np.where(passed.any(1), passed.argmax(1), 9)


def assign(model, features, threshold):
    selected = choices(predict(model, features), threshold)
    return {f['request_id']: int(counts(f['input_visual_tokens'])[i]) for f,i in zip(features, selected)}


def calibrate(model, features, qualities, allowance_pp):
    if allowance_pp not in (1, 2):
        raise ValueError('only predeclared one/two point targets')
    if set(model['training_groups']) & {f['group_id'] for f in features}:
        raise ValueError('training/calibration image overlap')
    q = np.asarray(qualities, dtype=float)
    if q.shape != (len(features), 10) or not np.isfinite(q).all() or np.any((q<0)|(q>1)):
        raise ValueError('complete physical calibration grid required')
    losses = predict(model, features)
    candidates = np.r_[np.nextafter(losses.min(), -np.inf), np.unique(losses)]
    ks = np.stack([counts(f['input_visual_tokens']) for f in features])
    units = np.asarray([f['cache_contract']['total_owned_bytes_per_slot'] for f in features])
    protected = np.asarray([f['input_text_tokens']+f['cache_contract']['reserved_generated_slots'] for f in features])
    domains = sorted({f['dataset'] for f in features}); best = None
    harm = np.maximum(0, q[:, -1:]-q)
    for tau in candidates:
        ix = choices(losses, float(tau)); observed = harm[np.arange(len(q)), ix]
        ds = {d:float(100*np.mean(observed[[f['dataset']==d for f in features]])) for d in domains}
        if any(v > allowance_pp+1e-10 for v in ds.values()):
            continue
        cost = int(np.sum((ks[np.arange(len(q)), ix]+protected)*units))
        key = cost, max(ds.values()), float(tau)
        if best is None or key < best[0]:
            best = key, dict(threshold=float(tau), mean_positive_damage_pp=ds,
                total_owned_cache_bytes=cost, allowance_pp=allowance_pp,
                calibration_requests=len(features), all_token_fraction=float(np.mean(ix==9)),
                prediction_is_not_an_accuracy_guarantee=True)
    if best is None:
        raise AssertionError('all-token policy must always be feasible')
    return seal(best[1])


def exact_uniform(features, adaptive):
    from .budget_router_v21 import exact_controls
    return exact_controls(features, adaptive)


def bootstrap_delta(a, b, groups, *, repetitions=10000, seed=42):
    values = np.asarray(a, dtype=float)-np.asarray(b, dtype=float)
    if len(values) != len(groups) or not len(values) or not np.isfinite(values).all():
        raise ValueError('complete paired outcomes and image groups required')
    unique = sorted(set(groups)); index = {g:i for i,g in enumerate(unique)}
    sums, sizes = np.zeros(len(unique)), np.zeros(len(unique))
    for g, v in zip(groups, values):
        sums[index[g]] += v; sizes[index[g]] += 1
    rng = np.random.default_rng(seed); samples = []
    for start in range(0, repetitions, 256):
        draw = rng.integers(0, len(unique), size=(min(256,repetitions-start),len(unique)))
        samples.extend((100*sums[draw].sum(1)/sizes[draw].sum(1)).tolist())
    return dict(delta_pp=float(100*values.mean()), ci95_pp=np.quantile(samples,[.025,.975]).tolist(),
        ci98_333_pp=np.quantile(samples,[.05/6,1-.05/6]).tolist(), groups=len(unique), requests=len(values))
