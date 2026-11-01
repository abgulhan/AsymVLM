"""Versioned allocation controls for the selector pilot; frozen v4 is unchanged.

Only the nominal search grid and operating-retention support allowance differ
from v4. Reconciliation still proceeds one physical token at a time: nominal
five-point choices are *not* a claim that every final count lies on that grid.
All utilities are estimates requiring exact-count physical evaluation.
"""

from __future__ import annotations

import heapq
from collections.abc import Mapping

import numpy as np

from . import adaptive_allocation_v4 as original


CONTROL_FORMAT = "mlsys_selector_allocation_controls_v5"
OPERATING_RETENTION_MARGIN = .005


def allocate(policy, rows, uniform_counts, *, budget_step_pp=1,
             operating_retention_margin=OPERATING_RETENTION_MARGIN):
    """Price tokens on a one/five-point grid and conserve the uniform total.

The extra support margin is a preregistered experimental tolerance, not a
measured guarantee. Zero is accepted for strict-v4 regression controls. Input
statistic support, image geometry, fit/evaluation identity separation, and
the uncertainty penalty are unchanged. Inputs must remain answer-free.
"""
    if type(budget_step_pp) is not int or budget_step_pp not in (1, 5):
        raise ValueError("budget_step_pp must be the predeclared 1 or 5")
    margin = original._finite(operating_retention_margin, "operating_retention_margin")
    if margin not in (0., OPERATING_RETENTION_MARGIN):
        raise ValueError("operating_retention_margin must be 0 or the predeclared .005")
    policy = original.load_policy(policy)
    rows = original._geometry_rows(rows, inference=True)
    if not isinstance(uniform_counts, Mapping) or set(uniform_counts) != {r["row_hash"] for r in rows}:
        raise ValueError("uniform count identities differ")
    fitted_rows, fitted_groups = set(policy["calibration_rows"]), set(policy["calibration_groups"])
    uniform, sizes, reasons, curves, bounds, choices = [], [], [], [], [], []
    support = policy["support"]
    for row in rows:
        if row["row_hash"] in fitted_rows or row["group_id"] in fitted_groups:
            raise ValueError("assignment identities overlap fitting")
        n = row["input_visual_tokens"]
        u = original._integer(uniform_counts[row["row_hash"]], "uniform count", 1)
        if u > n:
            raise ValueError("uniform count exceeds image count")
        why = []
        checks = (("statistic", row["statistic_value"], 1e-12), ("input_visual_tokens", n, 0),
                  ("uniform_retention", u / n, 1 / n + margin))
        for name, value, tolerance in checks:
            if not support[name][0] - tolerance <= value <= support[name][1] + tolerance:
                why.append(name)
        probes = original.local_probe_counts(n, u)
        lo, hi = (u, u) if why else (min(probes.values()), max(probes.values()))
        counts = ({u} if why else
                  {min(hi, max(lo, u + n * offset // 100)) for offset in range(-10, 11, budget_step_pp)}
                  | {u, lo, hi})
        sizes.append(n)
        uniform.append(u)
        reasons.append(why)
        curves.append(original._curves(policy, row["statistic_value"]))
        bounds.append((lo, hi))
        choices.append(sorted(counts, key=lambda count: (abs(count - u), count)))
    goal = sum(uniform)

    def utility(index, count):
        offset = (count - uniform[index]) / sizes[index]
        gains = np.asarray([np.interp(offset, np.asarray(original.OFFSETS) / 100, curve)
                            for curve in curves[index]])
        return float(gains.mean() - .5 * gains.std())

    utilities = [np.asarray([utility(i, count) for count in counts]) for i, counts in enumerate(choices)]
    scale = max(1., float(np.median(uniform)))

    def choose(price):
        result = []
        for i, counts in enumerate(choices):
            values = utilities[i] - price * (np.asarray(counts) - uniform[i]) / scale
            best = int(np.flatnonzero(values >= values.max() - 1e-12)[0])
            result.append(counts[best])
        return result

    def reconcile(counts):
        counts = list(counts)
        direction = 1 if sum(counts) < goal else -1
        residual = abs(goal - sum(counts))
        heap = []

        def push(i):
            proposed = counts[i] + direction
            if bounds[i][0] <= proposed <= bounds[i][1]:
                heapq.heappush(heap, (utility(i, counts[i]) - utility(i, proposed), rows[i]["row_hash"], i, proposed))

        for i in range(len(rows)):
            push(i)
        for _ in range(residual):
            if not heap:
                raise AssertionError("exact token reconciliation exhausted capacity")
            _, _, i, proposed = heapq.heappop(heap)
            counts[i] = proposed
            push(i)
        return counts, residual

    candidates = []
    if any(lo != hi for lo, hi in bounds) and support["statistic"][0] != support["statistic"][1]:
        low, high = -1., 1.
        for _ in range(64):
            if sum(choose(low)) >= goal:
                break
            low *= 2
        else:
            raise ValueError("resource-price lower bracket failed")
        for _ in range(64):
            if sum(choose(high)) <= goal:
                break
            high *= 2
        else:
            raise ValueError("resource-price upper bracket failed")
        for _ in range(80):
            middle = (low + high) / 2
            if sum(choose(middle)) > goal:
                low = middle
            else:
                high = middle
        for price in (low, high):
            nominal = choose(price)
            counts, residual = reconcile(nominal)
            candidates.append((counts, residual, price, nominal))
    best, repaired, price, best_score, best_nominal = list(uniform), 0, 0., 0., list(uniform)
    for candidate, residual, candidate_price, nominal in candidates:
        score = sum(utility(i, count) for i, count in enumerate(candidate))
        key = (sum(a != b for a, b in zip(candidate, uniform)), tuple(candidate))
        best_key = (sum(a != b for a, b in zip(best, uniform)), tuple(best))
        if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and key < best_key):
            best, repaired, price, best_score, best_nominal = candidate, residual, candidate_price, score, nominal
    if sum(best) != goal or any(not lo <= count <= hi for count, (lo, hi) in zip(best, bounds)):
        raise AssertionError("allocation violates exact budget or local bounds")
    records = {}
    for i, row in enumerate(rows):
        records[row["row_hash"]] = {**row, "uniform_selected_count": uniform[i], "adaptive_selected_count": best[i],
                                   "changed_from_uniform": best[i] != uniform[i], "uniform_retention": uniform[i] / sizes[i],
                                   "adaptive_retention": best[i] / sizes[i], "out_of_support": bool(reasons[i]),
                                   "fallback": bool(reasons[i]), "support_failures": reasons[i],
                                   "fallback_kind": "uniform_out_of_support" if reasons[i] else None,
                                   "predicted_conservative_gain": utility(i, best[i])}
    # Retain the v4 assignment wire format for its audited shuffle/report paths.
    # Controls are inside the assignment seal, and cannot be changed afterwards.
    return original._seal({"format": original.FORMAT + "_assignments", "policy_sha256": policy["policy_sha256"],
                          "statistic": policy["statistic"], "target_visual_tokens": goal, "uniform_visual_tokens": goal,
                          "adaptive_visual_tokens": sum(best), "token_error": sum(best) - goal, "rows": records,
                          "changed_rows": sum(a != b for a, b in zip(best, uniform)),
                          "uniform_fallback_rows": sum(bool(r) for r in reasons), "full_fallback_rows": 0,
                          "integer_reconciliation_tokens": repaired, "resource_price": price,
                          "predicted_conservative_total_gain": best_score,
                          "decision": "adaptive_predicted_benefit" if best != uniform else "uniform_no_supported_predicted_gain",
                          "quality_scope": "predicted utility only; exact-count physical inference required",
                          "allocation_controls": {"format": CONTROL_FORMAT, "budget_step_pp": budget_step_pp,
                              "operating_retention_margin": margin, "existing_retention_allowance": "one image token",
                              "physical_validation_required": True, "risk_penalty": .5,
                              "offset_range_pp": [-10, 10], "reconciliation_step_tokens": 1,
                              "grid_scope": "nominal search choices before exact-token reconciliation",
                              "off_grid_rows_after_reconciliation": sum(best[i] not in choices[i] for i in range(len(rows))),
                              "nominal_counts": {row["row_hash"]: best_nominal[i] for i, row in enumerate(rows)},
                              "nominal_choices": {row["row_hash"]: sorted(choices[i]) for i, row in enumerate(rows)}}},
                         "assignments_sha256")


def shuffle_assignments(assignments, *, seed=42):
    """The unchanged, answer-free, exact-multiset v4 negative control."""
    return original.shuffle_assignments(assignments, seed=seed)
