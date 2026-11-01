"""Paired request comparisons, resampled at document/image cluster level."""

from collections import defaultdict
import math


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def paired_cluster_bootstrap(records, baseline_method, *, method=None,
                             repetitions=10000, seed=2027, value_key="quality_score",
                             confidence=0.95):
    """Question-weighted mean difference with document-level paired resampling.

    Records must represent exactly one model/dataset/split/metric/action comparison.
    Repeated measurements of a request are first averaged. Request sets and group
    identities must match exactly; failed/incomplete rows block the comparison.
    """
    import numpy as np

    if repetitions < 100 or not 0 < confidence < 1:
        raise ValueError("at least 100 bootstrap replicates and a valid confidence are required")
    records = list(records)
    methods = {row["method"] for row in records}
    choices = methods - {baseline_method}
    if method is None:
        if len(choices) != 1:
            raise ValueError("select one candidate method for a paired comparison")
        method = next(iter(choices))
    rows = [row for row in records if row["method"] in {baseline_method, method}]
    contexts = {(row["model"], row["dataset"], row["split"], row.get("quality_metric")) for row in rows}
    if len(contexts) != 1 or not next(iter(contexts))[3]:
        raise ValueError("do not pool models, datasets, splits, or different/undefined quality metrics")
    values, groups = defaultdict(list), {}
    configurations = defaultdict(set)
    for row in rows:
        if row.get("status", "complete") not in {"complete", "ok", "success"}:
            raise ValueError("incomplete/failed requests cannot silently disappear from a paired comparison")
        value = row.get(value_key)
        if not _number(value):
            raise ValueError(f"missing/nonfinite {value_key}")
        if value_key == "quality_score" and not 0 <= value <= 1:
            raise ValueError("quality scores must be normalized to [0,1]")
        request = str(row["request_id"])
        group = str(row["group_id"])
        if request in groups and groups[request] != group:
            raise ValueError("paired request changed its document/image group")
        groups[request] = group
        configuration = row.get("configuration_id")
        configurations[row["method"]].add(
            ("policy", configuration) if configuration else
            ("fixed", row.get("visual_retention"), row.get("generated_budget")))
        values[row["method"], request].append(value)
    if any(len(configs) != 1 for configs in configurations.values()):
        raise ValueError("multiple action configurations must be compared separately")
    baseline_ids = {request for name, request in values if name == baseline_method}
    candidate_ids = {request for name, request in values if name == method}
    if not baseline_ids or baseline_ids != candidate_ids:
        raise ValueError("baseline and candidate must cover exactly the same nonempty request set")
    differences, baseline_values, candidate_values = defaultdict(list), [], []
    for request in sorted(baseline_ids):
        baseline = float(np.mean(values[baseline_method, request]))
        candidate = float(np.mean(values[method, request]))
        baseline_values.append(baseline)
        candidate_values.append(candidate)
        differences[groups[request]].append(candidate - baseline)
    cluster_names = sorted(differences)
    if len(cluster_names) < 2:
        raise ValueError("at least two independent document/image groups are required")
    sums = np.array([sum(differences[name]) for name in cluster_names])
    counts = np.array([len(differences[name]) for name in cluster_names])
    rng = np.random.default_rng(seed)
    estimates = []
    for start in range(0, repetitions, 256):
        draw = rng.integers(0, len(cluster_names), size=(min(256, repetitions - start), len(cluster_names)))
        estimates.extend((sums[draw].sum(axis=1) / counts[draw].sum(axis=1)).tolist())
    alpha = (1 - confidence) / 2
    low, high = np.quantile(estimates, [alpha, 1 - alpha])
    return {"method": method, "baseline_method": baseline_method, "n_requests": len(baseline_ids),
            "n_groups": len(cluster_names), "baseline_mean": float(np.mean(baseline_values)),
            "candidate_mean": float(np.mean(candidate_values)),
            "delta": float(sums.sum() / counts.sum()), "ci_lower": float(low), "ci_upper": float(high),
            "confidence": confidence, "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "estimand": "request_weighted_mean", "resampling_unit": "document_or_image_group"}
