"""Small, answer-free predictors of the measured benefit of extra patches.

The patch scorer is frozen. Quality labels are used only by fit(), never by
predict(), price_for_target(), or assign(). Prices are fixed before evaluation.
"""
from collections import defaultdict
import math

import numpy as np

from . import evidence_allocation_v18 as allocation
from .state import digest

MENUS = {'binary': (30, 100), 'three': (20, 60, 100)}
TARGETS = (40, 60)
UNIFORM_GRID = (20, 30, 40, 50, 60, 80, 100)
FORMAT = 'mlsys_benefit_router_v21'


def counts(n, menu):
    if type(n) is not int or n < 1 or menu not in MENUS:
        raise ValueError('positive native token count and registered menu required')
    return [(n*p+99)//100 for p in MENUS[menu]]


def vector(feature, sha, family='simple'):
    if feature.get('answer_inputs') is not False or family not in ('simple', 'context'):
        raise ValueError('registered answer-free feature family required')
    n, p = feature['input_visual_tokens'], feature['input_text_tokens']
    if type(n) is not int or n < 1 or type(p) is not int or p < 0:
        raise ValueError('invalid native input sizes')
    curve = feature['curves'][sha]
    out = []
    for c in (curve, curve['fixed_temperature_curve']):
        allocation.choose_count(c, 0.)
        risk = np.asarray(c['removed_weight_by_count'], dtype=float)
        if len(risk) != n+1:
            raise ValueError('risk curve length differs from native token geometry')
        w = risk[:-1]-risk[1:]
        if np.min(w) < -1e-10 or abs(w.sum()-1) > 1e-8:
            raise ValueError('invalid importance distribution')
        w = np.maximum(w, 0.)
        out.extend(float(risk[(n*q+99)//100]) for q in (20, 30, 40, 60))
        positive = w[w > 0]
        out.append(float(-np.sum(positive*np.log(positive))/math.log(n)) if n > 1 else 0.)
    out.extend((math.log(n), math.log1p(p)))
    if family == 'context':
        context = curve.get('pooled_context')
        if not isinstance(context, list) or len(context) != 96:
            raise ValueError('96 frozen, answer-free pooled context features required')
        out.extend(context)
    value = np.asarray(out, dtype=np.float64)
    if not np.isfinite(value).all():
        raise ValueError('nonfinite routing features')
    return value


def physical_lookup(features, rows, sha, required_percentages=UNIFORM_GRID):
    fs = {f['request_id']: f for f in features}
    if not fs or len(fs) != len(features):
        raise ValueError('unique nonempty feature panel required')
    output = {}
    for row in rows:
        if row.get('status') != 'ok' or row.get('request_id') not in fs:
            raise ValueError('unsuccessful or out-of-panel physical row')
        f = fs[row['request_id']]
        allocation._same_request(f, row)
        if row['selector'] != 'full' and (row['selector'] != 'scorer' or row['checkpoint_sha256'] != sha):
            raise ValueError('only the frozen scorer or all tokens may supply labels')
        if row['selector'] == 'full' and row['selected_count'] != f['input_visual_tokens']:
            raise ValueError('non-full all-token reference')
        key = row['request_id'], row['selected_count']
        if key in output and abs(output[key]['quality_score']-row['quality_score']) > 1e-10:
            raise ValueError('same physical count has conflicting answer scores')
        output.setdefault(key, row)
    for rid, f in fs.items():
        for percent in required_percentages:
            if (rid, (f['input_visual_tokens']*percent+99)//100) not in output:
                raise ValueError('missing physical label; quality interpolation prohibited')
    return output


def _panel(features):
    ordered = sorted(features, key=lambda f: f['request_id'])
    if not ordered or len({f['request_id'] for f in ordered}) != len(ordered):
        raise ValueError('unique nonempty panel required')
    return ordered


def fit(features, lookup, sha, menu, family='simple'):
    fs = _panel(features)
    x = np.stack([vector(f, sha, family) for f in fs])
    y = []
    groups = defaultdict(int)
    for f in fs:
        groups[f['dataset']] += 1
        ks = counts(f['input_visual_tokens'], menu)
        q = [lookup[f['request_id'], k]['quality_score'] for k in ks]
        y.append([v-q[0] for v in q[1:]])
    y = np.asarray(y, dtype=float)  # Signed benefits; never clip negative labels.
    weights = np.array([len(fs)/(len(groups)*groups[f['dataset']]) for f in fs])
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x-mean)**2, axis=0, weights=weights))
    scale = np.where(scale > 1e-8, scale, 1.)
    z = (x-mean)/scale
    intercept = np.average(y, axis=0, weights=weights)
    alpha = 1. if family == 'simple' else 10.
    coef = np.linalg.solve(z.T@(weights[:, None]*z)+alpha*np.eye(z.shape[1]),
                           z.T@(weights[:, None]*(y-intercept)))
    result = dict(format=FORMAT, family=family, menu=menu, checkpoint_sha256=sha,
        means=mean.tolist(), scales=scale.tolist(), coefficients=coef.tolist(),
        intercept=intercept.tolist(), regularization=alpha,
        training_request_ids=[f['request_id'] for f in fs],
        training_group_ids=sorted({f['group_id'] for f in fs}),
        labels='signed_physical_quality_difference_from_smallest_menu_count',
        price_units='quality_fraction_per_extra_allocated_KV_MiB',
        answer_inputs=False)
    result['sha256'] = digest(result)
    return result


def predict(model, features):
    if (model.get('format') != FORMAT or model.get('answer_inputs') is not False or
            model.get('sha256') != digest({k: v for k, v in model.items() if k != 'sha256'})):
        raise ValueError('changed or invalid router checkpoint')
    x = np.stack([vector(f, model['checkpoint_sha256'], model['family']) for f in features])
    z = (x-np.asarray(model['means']))/np.asarray(model['scales'])
    y = z@np.asarray(model['coefficients'])+np.asarray(model['intercept'])
    if not np.isfinite(y).all():
        raise ValueError('nonfinite benefit prediction')
    return np.concatenate([np.zeros((len(features), 1)), np.clip(y, -1., 1.)], axis=1)


def _geometry(model, features):
    ks = np.asarray([counts(f['input_visual_tokens'], model['menu']) for f in features], dtype=np.int64)
    costs = np.asarray([f['cache_contract']['total_owned_bytes_per_slot'] for f in features], dtype=float)
    if not np.isfinite(costs).all() or np.any(costs <= 0):
        raise ValueError('positive actual KV bytes per slot required')
    extra_mib = (ks-ks[:, :1])*costs[:, None]/2**20
    return ks, extra_mib


def assign(model, features, price):
    if not math.isfinite(price) or price < 0:
        raise ValueError('finite nonnegative fixed price required')
    fs = _panel(features)
    benefits = predict(model, fs)
    ks, cost = _geometry(model, fs)
    chosen = np.argmax(benefits-price*cost, axis=1)  # Ordered menu: cheaper wins exact ties.
    return {f['request_id']: int(ks[i, chosen[i]]) for i, f in enumerate(fs)}


def price_for_target(model, features, target):
    if target not in TARGETS:
        raise ValueError('unregistered target')
    fs = _panel(features)
    benefits = predict(model, fs)
    ks, cost = _geometry(model, fs)
    cap = sum((f['input_visual_tokens']*target+99)//100 for f in fs)
    def total(price):
        selected = np.argmax(benefits-price*cost, axis=1)
        return int(ks[np.arange(len(fs)), selected].sum())
    if total(0.) <= cap:
        return 0.
    low, high = 0., 1.
    while total(high) > cap:
        high *= 2
        if not math.isfinite(high):
            raise ValueError('target infeasible for fixed menu')
    for _ in range(60):
        mid = (low+high)/2
        if total(mid) > cap:
            low = mid
        else:
            high = mid
    return high


def exact_controls(features, adaptive):
    if set(adaptive) != {f['request_id'] for f in features}:
        raise ValueError('count map must cover the complete panel')
    domains = defaultdict(list)
    for f in features:
        k = adaptive[f['request_id']]
        if type(k) is not int or not 1 <= k <= f['input_visual_tokens']:
            raise ValueError('invalid adaptive count')
        domains[f['dataset']].append(f)
    result = {}
    for panel in domains.values():
        if len({f['cache_contract']['total_owned_bytes_per_slot'] for f in panel}) != 1:
            raise ValueError('token matching requires identical within-domain KV architectures')
        result.update(allocation.exact_uniform(panel, sum(adaptive[f['request_id']] for f in panel)))
    return result


def folds(features):
    domain_groups = defaultdict(set)
    for f in features:
        domain_groups[f['dataset']].add(f['group_id'])
    result = {}
    for domain, groups in sorted(domain_groups.items()):
        for i, group in enumerate(sorted(groups, key=lambda g: digest(['router-v21-fold', domain, g]))):
            result[domain, group] = i % 5
    return {f['request_id']: result[f['dataset'], f['group_id']] for f in features}


def development_fit(features, rows, sha, family='simple'):
    lookup = physical_lookup(features, rows, sha)
    fold = folds(features)
    result = {}
    for menu in MENUS:
        model = fit(features, lookup, sha, menu, family)
        prices = {str(t): price_for_target(model, features, t) for t in TARGETS}
        oof = []
        for j in range(5):
            train = [f for f in features if fold[f['request_id']] != j]
            held = [f for f in features if fold[f['request_id']] == j]
            if not held or not train:
                raise ValueError('five nonempty disjoint folds required')
            m = fit(train, lookup, sha, menu, family)
            for target in TARGETS:
                price = price_for_target(m, train, target)
                chosen = assign(m, held, price)
                for f in held:
                    rid = f['request_id']; k = chosen[rid]
                    oof.append(dict(request_id=rid, group_id=f['group_id'], dataset=f['dataset'], fold=j,
                        target=target, selected_count=k, input_visual_tokens=f['input_visual_tokens'],
                        quality=lookup[rid, k]['quality_score'], full_quality=lookup[rid, f['input_visual_tokens']]['quality_score']))
        result[f'{menu}_{family}'] = dict(model=model, prices=prices, oof=oof)
    return result


def assess(rows, candidates):
    """Screening only: complete, physically measured exact-matched comparisons."""
    by = defaultdict(dict)
    for r in rows:
        key = r['dataset'], r['method']
        if r['request_id'] in by[key] or r.get('status') != 'ok':
            raise ValueError('duplicate or failed screening row')
        by[key][r['request_id']] = r
    output = {}
    for name, candidate in candidates.items():
        points = []
        for domain in sorted({r['dataset'] for r in rows}):
            full = by[domain, 'full']
            for target in TARGETS:
                a, u = by[domain, f'{name}_{target}_adaptive'], by[domain, f'{name}_{target}_uniform']
                if not full or a.keys() != full.keys() or u.keys() != full.keys():
                    raise ValueError('complete screening cohort required')
                if any(sum(r[k] for r in a.values()) != sum(r[k] for r in u.values())
                       for k in ('expected_cache_bytes', 'selected_count')):
                    raise ValueError('adaptive and uniform must exactly match tokens and KV bytes')
                aq = 100*np.mean([r['quality_score'] for r in a.values()])
                fq = 100*np.mean([r['quality_score'] for r in full.values()])
                uq = 100*np.mean([r['quality_score'] for r in u.values()])
                akv = sum(r['expected_cache_bytes'] for r in a.values())
                memory_refs = []
                grid = []
                for percent in UNIFORM_GRID:
                    ref = full if percent == 100 else by[domain, f'keep{percent:03d}']
                    if ref.keys() != full.keys():
                        raise ValueError('complete measured uniform curve required')
                    score = 100*np.mean([r['quality_score'] for r in ref.values()])
                    kv = sum(r['expected_cache_bytes'] for r in ref.values())
                    grid.append(dict(retention=percent, score=score, kv_bytes=kv))
                # Compare with the cheapest measured uniform point attaining
                # adaptive quality minus 0.5 pp, not an arbitrarily costly arm.
                eligible = [r for r in grid if r['score'] >= aq-.5-1e-10]
                reference = min(eligible, key=lambda r: (r['kv_bytes'], -r['score'], r['retention'])) if eligible else None
                saving = 100*(1-akv/reference['kv_bytes']) if reference else None
                if reference and saving >= 10-1e-10 and aq >= reference['score']-.5-1e-10:
                    memory_refs.append(reference['retention'])
                safe = fq-aq <= 2.+1e-10
                points.append(dict(dataset=domain, target=target, loss_vs_full_pp=fq-aq,
                    adaptive_minus_uniform_pp=aq-uq, allocated_kv_bytes=akv,
                    accuracy_win=bool(safe and aq-uq >= 2.-1e-10),
                    memory_win=bool(safe and memory_refs), memory_references=memory_refs,
                    memory_saving_vs_cheapest_similar_uniform_percent=saving,
                    memory_reference=reference, passed_full_loss=bool(safe)))
        safe = all(p['passed_full_loss'] for p in points)
        wins = sum(p['accuracy_win'] or p['memory_win'] for p in points)
        output[name] = dict(points=points, wins=wins, passed=bool(safe and wins),
            safe_all_points=safe, worst_domain_loss_pp=max(p['loss_vs_full_pp'] for p in points))
    def key(name):
        r = output[name]; m = candidates[name]['model']
        return (not r['passed'], not r['safe_all_points'], -r['wins'], r['worst_domain_loss_pp'],
                m['family'] != 'simple', len(MENUS[m['menu']]), name)
    winner = min(output, key=key)
    return dict(winner=winner, candidates=output, passed=output[winner]['passed'],
        diagnostic_only=not output[winner]['passed'], final_labels_used=False)
