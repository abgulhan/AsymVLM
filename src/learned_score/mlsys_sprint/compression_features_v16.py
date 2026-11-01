"""Answer-free importance-floor and representation-floor stopping curves.

The frozen learned scores still choose every retained patch.  Beta changes
only the nonnegative importance weights, p_beta=(1-beta)*p+beta/N.  Lambda
changes only the stopping statistic, R=(1-lambda)*U+lambda*M.  Selected patches
have zero deficit, including when lambda is nonzero.  Neither quantity is an
accuracy probability.  Historical sources and checkpoints are read-only.

One nearest-retained-similarity pass per geometry computes both learned and
uniform weighted coverage.  All beta mixtures then use linearity, and lambda
needs only the saved scalar curves.  Similarities occupy at most N-by-64.
"""

import copy
import math

from . import compression_features_v11 as v11
from . import compression_features_v12 as v12
from . import compression_stopping_v11 as stopping
from .state import digest


FORMAT = 'mlsys_importance_representation_floor_features_v16'
RETENTIONS = v11.RETENTIONS
CURVE_KEYS = ('scorer', 'scorer_floor010', 'native')
IMPORTANCE_FLOOR = 0.1
REPRESENTATION_FLOOR = 0.1
CONTRACT = {
    'retention_percentages': RETENTIONS,
    'selector': v12.CONTRACT['selector'],
    'learned_weights': v11.CONTRACT['scorer_weights'],
    'importance_floor': '(1-beta)*learned_weight+beta/N; ranking unchanged',
    'representation_floor': '(1-lambda)*uncovered_mass+lambda*missing_mass; selected deficit zero',
    'primary_importance_floor': IMPORTANCE_FLOOR,
    'primary_representation_floor': REPRESENTATION_FLOOR,
    'rank32_geometry': v12.CONTRACT['rank32_geometry'],
    'native_geometry': v12.CONTRACT['native_geometry'],
    'similarity_block_size': stopping.MAX_BLOCK_SIZE,
    'curve_keys': CURVE_KEYS,
    'quality_monotonicity_assumed': False,
    'answers_consumed': False,
}


def _fraction(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + ' must be finite and in [0, 1]')
    return float(value)


def importance_weights(scores, beta=0.0):
    """FP64 normalized weights with a beta/N floor; no score/rank mutation."""
    beta = _fraction(beta, 'beta')
    weights = stopping.learned_score_weights(scores)
    return weights if beta == 0 else (1. - beta) * weights + beta / len(weights)


def _menus(n, retentions):
    if not retentions or any(type(p) is not int or not 1 <= p <= 100 for p in retentions):
        raise ValueError('nonempty integer retention percentages in [1, 100] required')
    return sorted({(n * p + 99) // 100 for p in retentions} | {n})


def _geometry_statistics(visual, scores, order, counts, geometry):
    """Reuse the old FP32 similarity arithmetic; reduce two weightings once."""
    import torch
    visual = stopping._visual(visual)
    n = len(visual)
    probability = stopping.learned_score_weights(scores).to(visual.device)
    uniform = torch.full((n,), 1. / n, dtype=torch.float64, device=visual.device)
    coverage = torch.zeros(n, dtype=torch.float32, device=visual.device)
    selected = torch.zeros(n, dtype=torch.bool, device=visual.device)
    missing, uncovered, uniform_missing, uniform_uncovered = [], [], [], []
    processed = 0
    for count in counts:
        if count == n:
            for values in (missing, uncovered, uniform_missing, uniform_uncovered):
                values.append(0.)
            continue
        for start in range(processed, count, stopping.MAX_BLOCK_SIZE):
            chosen = order[start:min(start + stopping.MAX_BLOCK_SIZE, count)]
            similarity = (visual @ visual[chosen].T).clamp_(0., 1.)
            coverage = torch.maximum(coverage, similarity.max(dim=1).values)
            selected[chosen] = True
            coverage[selected] = 1.
        processed = count
        deficit = (1. - coverage).double()
        for weights, masses, risks in ((probability, missing, uncovered),
                                       (uniform, uniform_missing, uniform_uncovered)):
            mass = min(1., max(0., float(weights[~selected].sum())))
            masses.append(mass)
            risks.append(min(mass, max(0., float((weights * deficit).sum()))))
    return {
        'format': stopping.FORMAT, 'input_visual_tokens': n, 'counts': list(counts),
        'missing_mass': missing, 'uncovered_mass': uncovered,
        'removed_fraction': [1. - count / n for count in counts],
        'selector': 'scorer', 'geometry': geometry, 'weight_mode': 'learned_scores',
        'weights_sha256': stopping._digest(probability.cpu().tolist()),
        'order_prefix_sha256': stopping._digest(order.cpu().tolist()),
        'order_prefix_length': len(order), 'block_size': stopping.MAX_BLOCK_SIZE,
        'invalid': False, 'fallback': False, 'fallback_reason': None,
        'answers_consumed': False,
        'risk_interpretation': 'answer_free_proxy_not_calibrated_accuracy_probability',
        'uniform_missing_mass': uniform_missing, 'uniform_uncovered_mass': uniform_uncovered,
        'importance_floor_beta': 0., 'representation_floor_lambda': 0.,
        'valid': True, 'physical_prefix_parity': True,
        'selected_indices_sha256': {
            str(count): digest(stopping.selected_positions(order, count, input_visual_tokens=n).cpu().tolist())
            for count in counts},
        'ranking_sha256': digest(order.cpu().tolist()),
    }


def _with_importance_floor(curve, scores, beta):
    beta = _fraction(beta, 'beta')
    result = copy.deepcopy(curve)
    result['importance_floor_beta'] = beta
    if beta == 0 or result['fallback']:
        return result
    if curve.get('importance_floor_beta', 0.) != 0 or curve.get('representation_floor_lambda', 0.) != 0:
        raise ValueError('importance mixing requires an unmodified base curve')
    result['weight_mode'] = 'learned_scores_uniform_floor'
    result['weights_sha256'] = stopping._digest(importance_weights(scores, beta).cpu().tolist())
    for name in ('missing_mass', 'uncovered_mass'):
        result[name] = [(1. - beta) * old + beta * uniform
                        for old, uniform in zip(curve[name], curve['uniform_' + name])]
    # Both bases are bounded; avoid a one-ulp violation after mixing.
    result['uncovered_mass'] = [min(m, max(0., u)) for m, u in
                                zip(result['missing_mass'], result['uncovered_mass'])]
    return result


def representation_floor_curve(curve, representation_floor=REPRESENTATION_FLOOR):
    """Pure CPU lambda transform, preserving hashes and input curve bytes.

    The output's ``uncovered_mass`` is the new stopping risk, not the original
    cosine-only deficit; its definition and original values remain explicit.
    Lambda=0 is an exact deep-copy identity.  Never apply this twice.
    """
    value = _fraction(representation_floor, 'lambda')
    result = copy.deepcopy(curve)
    if value == 0:
        return result
    if curve.get('representation_floor_lambda', 0.) != 0:
        raise ValueError('representation floor cannot be applied twice')
    masses, risks, counts = curve['missing_mass'], curve['uncovered_mass'], curve['counts']
    if len(masses) != len(risks) or len(risks) != len(counts) or not counts:
        raise ValueError('curve lengths differ')
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1
           for values in (masses, risks) for v in values):
        raise ValueError('finite bounded curve values required')
    if (any(u > m + 1e-12 for u, m in zip(risks, masses))
            or any(b > a + 1e-12 for values in (masses, risks) for a, b in zip(values, values[1:]))
            or masses[-1] != 0 or risks[-1] != 0):
        raise ValueError('monotone bounded curves ending at zero required')
    result['cosine_uncovered_mass'] = list(risks)
    result['uncovered_mass'] = [min(m, (1. - value) * u + value * m) for u, m in zip(risks, masses)]
    result['representation_floor_lambda'] = value
    result['stopping_risk_definition'] = CONTRACT['representation_floor']
    return result


def _fallback(n, geometry, reason, beta=0.):
    curve = stopping.full_fallback_curve(n, reason, selector='scorer', geometry=geometry)
    curve.update(valid=False, physical_prefix_parity=False,
                 selected_indices_sha256={str(n): digest(list(range(n)))},
                 uniform_missing_mass=[0.], uniform_uncovered_mass=[0.],
                 importance_floor_beta=beta, representation_floor_lambda=0.)
    return curve


def compute_feature_curves(scores, visual_features, projected_visual_features, *, retentions=RETENTIONS):
    """Return scorer, scorer_floor010 and native curves on identical prefixes.

    Rank32-invalid data cause full fallback for every candidate.  Native-only
    invalid geometry affects only native coverage.  OOM is always propagated,
    never recorded as a scientific fallback.  Shape/device errors are rejected.
    """
    import torch
    if not isinstance(visual_features, torch.Tensor) or visual_features.ndim != 2 or not all(visual_features.shape):
        raise ValueError('nonempty N-by-D native visual features required')
    if (not isinstance(projected_visual_features, torch.Tensor) or projected_visual_features.ndim != 2
            or not projected_visual_features.shape[1] or len(projected_visual_features) != len(visual_features)):
        raise ValueError('rank32 and native feature row counts must agree')
    n = len(visual_features)
    if not isinstance(scores, torch.Tensor) or scores.shape != (n,):
        raise ValueError('one learned score per visual patch required')
    if len({scores.device, visual_features.device, projected_visual_features.device}) != 1:
        raise ValueError('all feature tensors must share a device')
    counts = _menus(n, retentions)
    rank_geometry, native_geometry = CONTRACT['rank32_geometry'], CONTRACT['native_geometry']
    with torch.no_grad():
        try:
            # Native values also enter the physical selector's finite-value audit.
            if not bool(torch.isfinite(visual_features).all()):
                raise ValueError('native visual features must be finite')
            stopping._scores(scores, n)
            nonfull = [count for count in counts if count < n]
            order = (stopping.selector_order('asymvlm', scores=scores, keep_count=max(nonfull))
                     if nonfull else torch.empty(0, device=scores.device, dtype=torch.long))
            rank32 = _geometry_statistics(projected_visual_features, scores, order, counts, rank_geometry)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
                raise
            reason = f'{type(exc).__name__}: {exc}'
            return {'scorer': _fallback(n, rank_geometry, reason),
                    'scorer_floor010': _fallback(n, rank_geometry, reason, IMPORTANCE_FLOOR),
                    'native': _fallback(n, native_geometry, 'rank32 reference invalid: ' + reason)}
        try:
            native = _geometry_statistics(visual_features, scores, order, counts, native_geometry)
        except (ValueError, RuntimeError) as exc:
            if isinstance(exc, RuntimeError) and 'out of memory' in str(exc).lower():
                raise
            native = _fallback(n, native_geometry, f'{type(exc).__name__}: {exc}')
        rank32['coverage_geometry'] = 'rank32'
        native['coverage_geometry'] = 'native'
        return {'scorer': rank32,
                'scorer_floor010': _with_importance_floor(rank32, scores, IMPORTANCE_FLOOR),
                'native': native}


def assert_physical_prefix_parity(curves, physical_hashes):
    """Audit each stored selection, without accepting any quality labels."""
    expected = {str(k): value for k, value in physical_hashes.items()}
    for key in CURVE_KEYS:
        for count, fingerprint in curves[key]['selected_indices_sha256'].items():
            if expected.get(count) != fingerprint:
                raise ValueError('physical selected-index parity failed: ' + key + ' count ' + count)
    return True


def summarize(prepared, scorer):
    """Use only predecoder visual and semantic-query embeddings, never answers."""
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.models import Rank32Scorer
    from .adaptive_utility_v3 import calculate_statistic
    if type(scorer) is not Rank32Scorer:
        raise ValueError('unchanged frozen Rank32Scorer required')
    with torch.no_grad():
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        scores = scorer(vision, question).float()
        projected = F.linear(F.normalize(vision.float(), dim=-1), scorer.visual.weight.float())
        curves = compute_feature_curves(scores, vision, projected)
    return {'curves': curves, 'input_visual_tokens': len(vision),
            'input_text_tokens': int(prepared.inputs_embeds.shape[1] - len(vision)),
            'normalized_entropy': calculate_statistic(scores, 'normalized_entropy')}
