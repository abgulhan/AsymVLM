"""Isolated facility-location diagnostics; no answers, models, or job control.

The incumbent importance ranking is NOT marginal-gain greedy. These functions
make that comparison explicit. Bounds concern feature coverage, not accuracy.
"""
from collections import OrderedDict
import heapq
import itertools
import math
import time

import torch


class Coverage:
    def __init__(self, vectors, weights, *, dtype=torch.float64, block=64):
        if not isinstance(vectors, torch.Tensor) or vectors.ndim != 2 or not all(vectors.shape):
            raise ValueError('nonempty N by D tensor required')
        if dtype not in (torch.float32, torch.float64) or type(block) is not int or not 1 <= block <= 64:
            raise ValueError('FP32/FP64 and block size 1..64 required')
        self.z = vectors.detach().to(dtype=dtype)
        if not bool(torch.isfinite(self.z).all()):
            raise ValueError('nonfinite vectors')
        norms = self.z.norm(dim=1)
        if bool((norms <= 1e-12).any()):
            raise ValueError('zero vector')
        self.z = self.z / norms[:, None]
        self.n, self.block = len(self.z), block
        self.w = torch.as_tensor(weights, device=self.z.device, dtype=torch.float64).detach()
        if (self.w.shape != (self.n,) or not bool(torch.isfinite(self.w).all())
                or bool((self.w < 0).any()) or abs(float(self.w.sum())-1) > 1e-8):
            raise ValueError('finite nonnegative weights summing to one required')
        self.columns = OrderedDict()
        self.column_limit = max(1, min(256, (32 * 1024**2)//(self.n*self.z.element_size())))

    def column(self, j):
        if j not in self.columns:
            result = (self.z @ self.z[j]).clamp(0, 1)
            result[j] = 1
            self.columns[j] = result
            if len(self.columns) > self.column_limit:
                self.columns.popitem(last=False)
        else:
            self.columns.move_to_end(j)
        return self.columns[j]

    def best(self, selected):
        ids = list(selected)
        if len(ids) != len(set(ids)) or any(type(j) is not int or not 0 <= j < self.n for j in ids):
            raise ValueError('distinct valid patch indices required')
        result = torch.zeros(self.n, device=self.z.device, dtype=self.z.dtype)
        for start in range(0, len(ids), self.block):
            part = ids[start:start+self.block]
            result = torch.maximum(result, (self.z @ self.z[part].T).clamp(0, 1).amax(1))
        result[ids] = 1
        return result

    def value(self, selected):
        return float((self.w * self.best(selected).double()).sum())

    def gains(self, best, *, selected=()):
        gains = torch.empty(self.n, device=self.z.device, dtype=torch.float64)
        for start in range(0, self.n, self.block):
            stop = min(start+self.block, self.n)
            sim = (self.z @ self.z[start:stop].T).clamp(0, 1)
            sim[torch.arange(start, stop, device=self.z.device), torch.arange(stop-start, device=self.z.device)] = 1
            gains[start:stop] = (self.w[:, None] * (sim-best[:, None]).clamp_min(0).double()).sum(0)
        gains[list(selected)] = 0
        return gains

    def greedy(self, count, *, lazy=True, seconds=60.):
        if type(count) is not int or not 1 <= count <= self.n or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('valid count and positive finite timeout required')
        start = time.monotonic()
        best = self.best([])
        order, values, recomputations = [], [], 0
        initial = self.gains(best).cpu().tolist()
        heap = [(-g, j, 0) for j, g in enumerate(initial)]
        heapq.heapify(heap)
        while len(order) < count:
            if time.monotonic()-start > seconds:
                raise TimeoutError('coverage greedy exceeded per-example selection limit')
            if lazy:
                neg, j, stamp = heapq.heappop(heap)
                if stamp != len(order):
                    gain = float((self.w*(self.column(j)-best).clamp_min(0).double()).sum())
                    heapq.heappush(heap, (-gain, j, len(order)))
                    recomputations += 1
                    continue
            else:
                gains = self.gains(best)
                gains[order] = -1
                j = int(gains.argmax())
            order.append(j)
            best = torch.maximum(best, self.column(j))
            best[order] = 1
            values.append(float((self.w*best.double()).sum()))
        return {'order': order, 'coverage': values, 'seconds': time.monotonic()-start,
                'gain_recomputations': recomputations, 'lazy': lazy}

    def curve(self, order, counts, *, tau=None):
        if len(order) != self.n or sorted(order) != list(range(self.n)):
            raise ValueError('full deterministic permutation required')
        if not counts or counts != sorted(set(counts)) or any(type(k) is not int or not 1 <= k <= self.n for k in counts):
            raise ValueError('sorted unique nonempty count menu required')
        if tau is not None and (not math.isfinite(tau) or not 0 <= tau <= 1):
            raise ValueError('tolerance must be in [0,1]')
        best, processed, result = self.best([]), 0, []
        for count in counts:
            if count == self.n:
                risk = 0.
            else:
                for start in range(processed, count, self.block):
                    part = order[start:min(count, start+self.block)]
                    best = torch.maximum(best, (self.z @ self.z[part].T).clamp(0, 1).amax(1))
                    best[part] = 1
                processed = count
                risk = float((self.w * (1-best).double()).sum())
            result.append({'count': count, 'risk': risk})
            if tau is not None and risk <= tau:
                break
        return result

    def distortion(self, selected):
        if not selected:
            raise ValueError('distortion requires a nonempty selected set')
        cosine = torch.full((self.n,), -1., device=self.z.device, dtype=self.z.dtype)
        for start in range(0, len(selected), self.block):
            cosine = torch.maximum(cosine, (self.z @ self.z[selected[start:start+self.block]].T).clamp(-1,1).amax(1))
        cosine[selected] = 1
        risk = float((self.w*(1-cosine.clamp_min(0)).double()).sum())
        distance = float((self.w*(2*(1-cosine)).double()).sum())
        return {'risk': risk, 'squared_distortion': distance,
                'negative_nearest_match_mass': float(self.w[cosine < 0].sum()),
                'two_risk_lower': 2*risk, 'four_risk_upper': 4*risk}

    def curvature(self):
        singletons = self.gains(self.best([]))
        other = torch.zeros(self.n, device=self.z.device, dtype=self.z.dtype)
        for start in range(0, self.n, self.block):
            stop = min(start+self.block, self.n)
            sim = (self.z @ self.z[start:stop].T).clamp(0,1)
            sim[torch.arange(start,stop,device=self.z.device), torch.arange(stop-start,device=self.z.device)] = 0
            other = torch.maximum(other, sim.amax(1))
        valid = singletons > 0
        ratios = self.w[valid] * (1-other[valid]).double() / singletons[valid]
        c = min(1., max(0., 1-float(ratios.min())))
        return {'curvature': c, 'all_candidates_checked': True,
                'ordinary_greedy_fixed_count_bound': -math.expm1(-c)/c if c else 1.,
                'accuracy_guarantee': False, 'formal_interval_arithmetic': False}

    def upper_bound(self, selected, count):
        if not len(selected) <= count <= self.n:
            raise ValueError('selected size <= comparison count <= N required')
        best = self.best(selected)
        value = float((self.w*best.double()).sum())
        gain = self.gains(best, selected=selected)
        # A numerical guard, not a claimed interval-arithmetic proof.
        guard = 16*torch.finfo(self.z.dtype).eps*self.n
        upper = min(1., value+float(gain.topk(count).values.sum())+guard)
        return {'coverage': value, 'optimum_coverage_upper': upper,
                'coverage_to_upper_ratio': value/upper if upper else 1.,
                'roundoff_guard': guard, 'formal_interval_arithmetic': False}


def count_menu(n, step):
    if type(n) is not int or n < 1 or type(step) is not int or not 1 <= step <= 100:
        raise ValueError('positive count and percentage step required')
    return sorted({(n*p+99)//100 for p in range(step, 101, step)} | {n})


def first_feasible(curve, tau):
    if not math.isfinite(tau) or not 0 <= tau <= 1:
        raise ValueError('finite coverage tolerance in [0,1] required')
    passing = [item['count'] for item in curve if item['risk'] <= tau]
    if not passing:
        raise ValueError('no passing count; caller must include the full-retention endpoint')
    return min(passing)


def exhaustive_frontier(objective):
    if objective.n > 12:
        raise ValueError('exhaustive diagnostic capped at twelve patches')
    result = []
    for count in range(1, objective.n+1):
        best = max((objective.value(list(s)), tuple(-j for j in s), s)
                   for s in itertools.combinations(range(objective.n), count))
        result.append({'count': count, 'coverage': best[0], 'selected': list(best[2])})
    return result


def certification_feasibility(*, policies=3, domains=4, models=3, alpha=.01, delta=.05):
    if any(type(v) is not int or v < 1 for v in (policies,domains,models)) or not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError('positive family sizes, alpha/delta in (0,1) required')
    tests = policies*models*(domains+1)
    log_term = math.log(tests/delta)
    return {'target_expected_positive_harm':alpha, 'confidence':1-delta,
        'simultaneous_tests':tests,
        'hoeffding_zero_empirical_harm_groups_per_test':math.ceil(log_term/(2*alpha**2)),
        'zero_any_harm_indicator_groups_per_test':math.ceil(math.log(delta/tests)/math.log1p(-alpha)),
        'indicator_bound_note':'bounds probability of any harm, conservatively bounds fractional mean harm; requires zero observed harmed groups',
        'assumptions':['fixed candidate family before calibration', 'independent calibration source groups',
                       'matching target distribution', 'loss scaled to [0,1]'],
        'current_data_certified':False,
        'pointwise_or_distribution_shift_guarantee':False}
