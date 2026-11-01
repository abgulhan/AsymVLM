"""Pure full-reference compression reports; no fitting, label reads, or writes.

Unlike the historical matched-budget comparisons, this module deliberately
permits unequal retained counts and cache capacities. Callers must first audit
physical metadata, then supply the complete, predeclared request cohort.
"""

from collections import defaultdict
import math
import statistics

import numpy as np

from .adaptive_report_v3 import quality_summary


FORMAT = "mlsys_compression_report_v8"
BOOTSTRAP_REPETITIONS = 10000
BOOTSTRAP_SEED = 20261018
CONFIDENCE = .95
LOSS_LIMIT_PP = 1.
RUNTIME_FIELDS = (
    "model", "image_sha256", "max_new_tokens", "generated_budget", "query_control",
    "runtime_contract_sha256", "prompt_sha256", "model_revision",
    "generation_config_sha256", "inference_config_sha256",
)
CONFIRMATION_CHECKS = (
    "fresh_identity_audit_passed", "policies_frozen_before_labels", "expected_complete_cohort",
)


def _finite(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("missing or nonfinite " + label)
    if minimum is not None and value < minimum:
        raise ValueError("negative " + label)
    return float(value)


def _model(row):
    return row.get("model_tag", row.get("model"))


def _index(rows):
    result = {}
    for row in rows:
        request = row.get("request_id")
        if not isinstance(request, str) or not request or request in result:
            raise ValueError("missing or duplicate request identity")
        if row.get("status", "ok") != "ok":
            raise ValueError("only successful audited physical rows may be compared")
        if not all(isinstance(row.get(k), str) and row[k] for k in ("group_id", "dataset", "quality_metric")) or not _model(row):
            raise ValueError("missing group, dataset, metric, or model identity")
        quality = _finite(row.get("quality_score"), "quality score")
        if not 0 <= quality <= 1:
            raise ValueError("quality must be normalized to [0,1]")
        for key in ("input_visual_tokens", "retained_visual_tokens", "input_text_tokens"):
            if type(row.get(key)) is not int or row[key] < (0 if key == "input_text_tokens" else 1):
                raise ValueError("invalid " + key)
        if row["retained_visual_tokens"] > row["input_visual_tokens"]:
            raise ValueError("retained count exceeds input visual tokens")
        if "selected_count" in row and (type(row["selected_count"]) is not int
                                         or row["selected_count"] != row["retained_visual_tokens"]):
            raise ValueError("selected and retained counts disagree")
        result[request] = row
    return result


def _optional_mean(rows, getter, label):
    values = [getter(row) for row in rows]
    for value in values:
        if value is not None:
            _finite(value, label, minimum=0)
    return statistics.mean(values) if values and all(v is not None for v in values) else None


def _cache(row, field):
    return row.get("metrics", {}).get("cache", {}).get(field)


def _full_fallback(row):
    cell = row.get("schedule_cell", {})
    flags = []
    for value in (row, cell):
        if "full_fallback" in value:
            if type(value["full_fallback"]) is not bool:
                raise ValueError("full_fallback must be boolean")
            flags.append(value["full_fallback"])
        if "fallback_kind" in value:
            flags.append(value["fallback_kind"] in {"full", "full_token", "full_tokens"})
    if flags:
        if len(set(flags)) != 1:
            raise ValueError("contradictory full-fallback provenance")
        return flags[0]
    if row.get("selector") == "full" or row.get("method") == "full":
        return False  # The structural reference is not an adaptive fallback.
    return None


def memory_summary(rows, full_rows=None):
    """Keep patch, allocated-KV, live-KV, and GPU-peak quantities separate.

Missing measurements stay None; live bytes are never inferred from capacity.
Fallback use requires an explicit flag, distinct from simply retaining all K.
"""
    rows = list(rows)
    _index(rows)
    n = len(rows)
    original = sum(row["input_visual_tokens"] for row in rows)
    kept = sum(row["retained_visual_tokens"] for row in rows)
    fallback = [_full_fallback(row) for row in rows]
    if any(flag is True and row["retained_visual_tokens"] != row["input_visual_tokens"]
           for flag, row in zip(fallback, rows)):
        raise ValueError("full fallback did not retain all image tokens")
    result = {
        "questions": n, "aggregate_input_visual_tokens": original,
        "aggregate_retained_visual_tokens": kept,
        "aggregate_patch_reduction_percent": 100 * (1 - kept / original) if original else None,
        "mean_per_input_patch_reduction_percent": statistics.mean(
            100 * (1 - row["retained_visual_tokens"] / row["input_visual_tokens"]) for row in rows) if n else None,
        "fraction_compressed": sum(row["retained_visual_tokens"] < row["input_visual_tokens"] for row in rows) / n if n else None,
        "fraction_at_full_tokens": sum(row["retained_visual_tokens"] == row["input_visual_tokens"] for row in rows) / n if n else None,
        "fraction_full_fallback": sum(fallback) / n if n and all(flag is not None for flag in fallback) else None,
        "fallback_status_known_rows": sum(flag is not None for flag in fallback),
    }
    for output, field in (("allocated_kv", "kv_capacity_bytes"), ("owned_cache", "total_owned_bytes"),
                          ("actual_kv", "kv_active_bytes")):
        mean = _optional_mean(rows, lambda row: _cache(row, field), field)
        result["mean_" + output + "_bytes"] = mean
        result["aggregate_" + output + "_bytes"] = sum(_cache(row, field) for row in rows) if mean is not None else None
    result["mean_peak_gpu_bytes"] = _optional_mean(rows,
        lambda row: row.get("metrics", {}).get("full_request_peak_allocated_bytes"), "GPU peak")
    result["aggregate_allocated_kv_reduction_percent"] = None
    result["mean_per_input_allocated_kv_reduction_percent"] = None
    result["full_mean_actual_kv_bytes"] = None
    result["full_mean_peak_gpu_bytes"] = None
    if full_rows is not None:
        full = _index(list(full_rows))
        if set(full) != {row["request_id"] for row in rows}:
            raise ValueError("memory comparison requires identical cohorts")
        references = [full[row["request_id"]] for row in rows]
        for row, reference in zip(rows, references):
            if reference["retained_visual_tokens"] != reference["input_visual_tokens"]:
                raise ValueError("memory full reference was compressed")
            if (row["input_visual_tokens"], row["input_text_tokens"]) != (reference["input_visual_tokens"], reference["input_text_tokens"]):
                raise ValueError("memory comparison input geometry changed")
        mean_full = _optional_mean(references, lambda row: _cache(row, "kv_capacity_bytes"), "full allocated KV")
        result["full_aggregate_allocated_kv_bytes"] = sum(_cache(row, "kv_capacity_bytes") for row in references) if mean_full is not None else None
        if mean_full is not None and result["mean_allocated_kv_bytes"] is not None:
            if mean_full <= 0 or any(_cache(row, "kv_capacity_bytes") <= 0 for row in references):
                raise ValueError("full allocated KV must be positive")
            result["aggregate_allocated_kv_reduction_percent"] = 100 * (1 - result["mean_allocated_kv_bytes"] / mean_full)
            result["mean_per_input_allocated_kv_reduction_percent"] = statistics.mean(
                100 * (1 - _cache(a, "kv_capacity_bytes") / _cache(b, "kv_capacity_bytes"))
                for a, b in zip(rows, references)) if n else None
        result["full_mean_actual_kv_bytes"] = _optional_mean(references, lambda row: _cache(row, "kv_active_bytes"), "full live KV")
        result["full_mean_peak_gpu_bytes"] = _optional_mean(references,
            lambda row: row.get("metrics", {}).get("full_request_peak_allocated_bytes"), "full GPU peak")
    return result


summarize_memory = memory_summary


def systems_summary(rows):
    rows = list(rows)
    result = {"timing_claim": "descriptive_only_not_controlled_speed_evidence",
              "timing_reused_rows": sum(row.get("timing_reused_not_an_independent_observation") is True for row in rows),
              "all_timing_isolated": bool(rows) and all(row.get("timing_isolated") is True for row in rows)}
    for field in ("end_to_end_seconds", "ttft_seconds", "selector_seconds", "compression_seconds"):
        result["mean_" + field] = _optional_mean(rows,
            lambda row: row.get(field, row.get("metrics", {}).get(field)), field)
    result["mean_selection_and_compaction_seconds"] = _optional_mean(rows,
        lambda row: row.get("metrics", {}).get("selection_and_visual_compaction_seconds"), "selection and compaction")
    return result


def _clusters(candidate, full, metric):
    groups = defaultdict(list)
    for request in sorted(candidate):
        groups[candidate[request]["group_id"]].append(request)
    if metric != "mme":
        return groups, None
    strata = defaultdict(list)
    for group, requests in sorted(groups.items()):
        categories = {candidate[request].get("category") for request in requests}
        if len(requests) != 2 or len(categories) != 1 or not next(iter(categories)):
            raise ValueError("MME requires complete, single-category two-question image pairs")
        a = [candidate[request]["quality_score"] for request in requests]
        b = [full[request]["quality_score"] for request in requests]
        if any(value not in (0, 1) for value in a + b):
            raise ValueError("MME question accuracy must be binary")
        loss = .5 * (statistics.mean(b) - statistics.mean(a)) + .5 * (int(all(b)) - int(all(a)))
        strata[next(iter(categories))].append(loss)
    return groups, strata


def _upper_loss(candidate, full, metric, *, repetitions, seed, structural_full):
    groups, strata = _clusters(candidate, full, metric)
    losses = {key: full[key]["quality_score"] - candidate[key]["quality_score"] for key in candidate}
    base = {"confidence": CONFIDENCE, "interval_sides": "upper_one_sided",
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "independent_groups": len(groups), "upper_loss_pp": None}
    if structural_full:
        if any(candidate[key] != full[key] or candidate[key]["retained_visual_tokens"] != candidate[key]["input_visual_tokens"]
               for key in candidate):
            raise ValueError("structural full baseline requires the identical all-token observations")
        return {**base, "status": "computed", "upper_loss_pp": 0., "bound_method": "structural_identity",
                "resampling_unit": "not_required_identical_full_reference"}
    if all(value == 0 for value in losses.values()):
        if metric == "mme":
            categories = len(strata)
            bounds = {category: {"groups": len(values), "weight": 1 / categories,
                       "alpha": (1 - CONFIDENCE) / categories,
                       "upper_harm_probability": -math.expm1(math.log((1 - CONFIDENCE) / categories) / len(values))}
                      for category, values in sorted(strata.items())}
            upper = 100 * sum(item["weight"] * item["upper_harm_probability"] for item in bounds.values())
            return {**base, "status": "computed", "upper_loss_pp": upper,
                    "bound_method": "zero_harm_exact_binomial_category_bonferroni",
                    "category_bounds": bounds, "resampling_unit": "image_pairs_stratified_by_category",
                    "bound_assumption": "independent image pairs within each category; worst-case loss per harmed pair is one"}
        sizes = [len(requests) for requests in groups.values()]
        probability = -math.expm1(math.log(1 - CONFIDENCE) / len(groups))
        if len(set(sizes)) != 1:
            return {**base, "status": "zero_difference_variable_group_sizes_inconclusive",
                    "bound_method": "zero_harm_variable_group_weights_not_certified",
                    "upper_group_harm_probability": probability,
                    "resampling_unit": "document_or_image_groups"}
        return {**base, "status": "computed", "upper_loss_pp": 100 * probability,
                "upper_harm_probability": probability, "bound_method": "zero_harm_exact_binomial",
                "resampling_unit": "document_or_image_groups",
                "bound_assumption": "independent equal-size groups; worst-case loss per harmed group is one"}
    rng = np.random.default_rng(seed)
    estimates = np.zeros(repetitions, dtype=float)
    if metric == "mme":
        if any(len(values) < 2 for values in strata.values()):
            return {**base, "status": "insufficient_image_pairs_per_category",
                    "bound_method": "cluster_bootstrap_inconclusive"}
        for category, values in sorted(strata.items()):
            values = np.asarray(values, dtype=float)
            for start in range(0, repetitions, 256):
                stop = min(start + 256, repetitions)
                draws = rng.integers(0, len(values), size=(stop - start, len(values)))
                estimates[start:stop] += values[draws].mean(axis=1) / len(strata)
        unit = "image_pairs_stratified_by_category"
    else:
        if len(groups) < 2:
            return {**base, "status": "insufficient_independent_groups", "bound_method": "cluster_bootstrap_inconclusive"}
        sums = np.asarray([sum(losses[key] for key in requests) for _, requests in sorted(groups.items())])
        counts = np.asarray([len(requests) for _, requests in sorted(groups.items())])
        for start in range(0, repetitions, 256):
            stop = min(start + 256, repetitions)
            draws = rng.integers(0, len(sums), size=(stop - start, len(sums)))
            estimates[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        unit = "document_or_image_groups"
    if np.ptp(estimates) <= 1e-12:
        return {**base, "status": "degenerate_nonzero_or_cancelling_cluster_losses_inconclusive",
                "bound_method": "cluster_bootstrap_zero_variance", "resampling_unit": unit}
    return {**base, "status": "computed", "upper_loss_pp": float(np.quantile(100 * estimates, CONFIDENCE)),
            "bound_method": "paired_cluster_percentile_bootstrap", "resampling_unit": unit}


def compare_to_full(candidate_rows, full_rows, *, expected_request_ids=None,
                    repetitions=BOOTSTRAP_REPETITIONS, seed=BOOTSTRAP_SEED,
                    confidence=CONFIDENCE, evidence="development", confirmation_audit=None,
                    structural_full=False, runtime_fields=()):
    """Full-minus-compressed quality loss, with no equal-budget requirement.

Numerical passing is not confirmation. Fresh-cohort and pre-label freeze audits
must be explicit, true booleans supplied by the independently audited study.
"""
    if repetitions != BOOTSTRAP_REPETITIONS or type(repetitions) is not int or seed != BOOTSTRAP_SEED or type(seed) is not int or confidence != CONFIDENCE:
        raise ValueError("v8 fixes 10000 bootstrap draws, seed 20261018, and one-sided 95% confidence")
    if evidence not in {"development", "confirmation"}:
        raise ValueError("evidence must be development or confirmation")
    candidate, full = _index(list(candidate_rows)), _index(list(full_rows))
    base = {"format": FORMAT + "_comparison", "evidence": evidence,
            "evidence_label": "development only; not fresh confirmation" if evidence == "development" else "confirmation audit required",
            "cohort_complete": False, "questions": len(candidate), "mean_loss_pp": None,
            "upper_loss_pp": None, "passes_numeric_rule": False, "accepted": False,
            "confirmed": False, "loss_limit_pp": LOSS_LIMIT_PP,
            "budget_comparison": "unequal_budgets_permitted_against_full_tokens"}
    if expected_request_ids is None:
        return {**base, "status": "missing_predeclared_cohort"}
    expected_list = list(expected_request_ids)
    expected = set(expected_list)
    if (not expected or len(expected) != len(expected_list)
            or any(not isinstance(value, str) or not value for value in expected)):
        raise ValueError("expected cohort must contain unique nonempty request identities")
    if set(candidate) != expected or set(full) != expected:
        return {**base, "status": "missing_or_incomplete_cohort",
                "missing_candidate": sorted(expected - candidate.keys()), "missing_full": sorted(expected - full.keys()),
                "unexpected_candidate": sorted(candidate.keys() - expected), "unexpected_full": sorted(full.keys() - expected)}
    identities = {(_model(row), row["dataset"], row["quality_metric"]) for row in candidate.values()}
    if len(identities) != 1:
        raise ValueError("one comparison requires one model, dataset, and quality metric")
    for key, a in candidate.items():
        b = full[key]
        if (_model(a), a["dataset"], a["quality_metric"], a["group_id"], a.get("category"), a["input_visual_tokens"], a["input_text_tokens"]) != (
                _model(b), b["dataset"], b["quality_metric"], b["group_id"], b.get("category"), b["input_visual_tokens"], b["input_text_tokens"]):
            raise ValueError("paired model/dataset/group/category/geometry/metric mismatch")
        if b["retained_visual_tokens"] != b["input_visual_tokens"]:
            raise ValueError("full reference was compressed")
        for field in {*RUNTIME_FIELDS, *runtime_fields}:
            if (field in a or field in b) and (field not in a or field not in b or a[field] != b[field]):
                raise ValueError("paired runtime identity mismatch: " + field)
    ordered = sorted(candidate)
    a, b = [candidate[key] for key in ordered], [full[key] for key in ordered]
    metric = a[0]["quality_metric"]
    candidate_score, full_score = quality_summary(a, metric), quality_summary(b, metric)
    loss = full_score["score_percent"] - candidate_score["score_percent"]
    bound = _upper_loss(candidate, full, metric, repetitions=repetitions, seed=seed, structural_full=structural_full)
    numeric = bound["status"] == "computed" and loss <= LOSS_LIMIT_PP and bound["upper_loss_pp"] < LOSS_LIMIT_PP
    audit = confirmation_audit if isinstance(confirmation_audit, dict) else {}
    verified = evidence == "confirmation" and all(audit.get(key) is True for key in CONFIRMATION_CHECKS)
    status = bound["status"]
    if evidence == "confirmation" and not verified:
        status = "unverified_confirmation"
    return {**base, **bound, "status": status, "uncertainty_status": bound["status"],
            "cohort_complete": True, "model": _model(a[0]), "dataset": a[0]["dataset"],
            "quality_metric": metric, "candidate_quality_percent": candidate_score["score_percent"],
            "full_quality_percent": full_score["score_percent"], "candidate_quality": candidate_score,
            "full_quality": full_score, "mean_loss_pp": loss, "passes_numeric_rule": bool(numeric),
            "accepted": bool(numeric and verified), "confirmed": bool(numeric and verified),
            "confirmation_audit_passed": bool(verified),
            "evidence_label": "fresh confirmation audited" if verified else base["evidence_label"],
            "confirmation_checks": {key: audit.get(key) is True for key in CONFIRMATION_CHECKS},
            "memory": memory_summary(a, b), "systems": systems_summary(a),
            "full_systems": systems_summary(b)}


def summarize(rows, *, full_method="full", evidence="development", expected_cohorts=None,
              confirmation_audit=None, expected_methods=None, runtime_fields=()):
    """Produce two table-ready collections from already audited physical rows.

expected_cohorts maps (model_tag, normalized_dataset) to predeclared request IDs.
expected_methods, when supplied, also surfaces entirely missing candidate panels.
"""
    grouped = defaultdict(list)
    for row in rows:
        if (not isinstance(row.get("method"), str) or not row["method"]
                or not _model(row) or not isinstance(row.get("dataset"), str) or not row["dataset"]):
            raise ValueError("normalized model, dataset, and method labels are required")
        grouped[(_model(row), row["dataset"], row["method"])].append(row)
    cohorts = expected_cohorts or {}
    panels = []
    keys = set(grouped)
    if expected_methods is not None:
        keys |= {(model, dataset, method) for model, dataset in cohorts for method in expected_methods}
    for model, dataset, method in sorted(keys):
        candidate = grouped.get((model, dataset, method), [])
        full = grouped.get((model, dataset, full_method), [])
        result = compare_to_full(candidate, full, expected_request_ids=cohorts.get((model, dataset)),
            evidence=evidence, confirmation_audit=confirmation_audit,
            structural_full=method == full_method, runtime_fields=runtime_fields)
        panels.append({"model": model, "dataset": dataset, "method": method, **result})
    return {"format": FORMAT, "evidence": evidence, "panels": panels,
            "quality_compression": [{key: value for key, value in panel.items() if key not in {"systems", "full_systems"}}
                                    for panel in panels],
            "systems": [{"model": panel["model"], "dataset": panel["dataset"], "method": panel["method"],
                         **panel.get("systems", {}), **{key: panel.get("memory", {}).get(key) for key in (
                             "mean_allocated_kv_bytes", "mean_actual_kv_bytes", "mean_peak_gpu_bytes")}}
                        for panel in panels]}


def render_tables(report):
    def number(value):
        return "N/A" if value is None else f"{value:.3f}"
    def text(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    def display_identity(panel):
        model = {"qwen25": "Qwen2.5-VL-3B", "llava_next": "LLaVA-NeXT-7B"}.get(panel["model"], panel["model"])
        method = "All image patches / full KV cache" if panel["method"] == "full" else panel["method"]
        for source, label in (("normalized_entropy", "entropy"), ("local_shell", "local score mass"),
                              ("nearest_neighbor_diversity", "nearest-neighbor diversity"),
                              ("combined_ridge", "three-statistic predictor")):
            method = method.replace(source, label)
        return [text(model), text(panel["dataset"]), text(method)]
    lines = ["# Compression against full-token reference", "",
             "Evidence: " + text(report.get("evidence", "development")) + ". Development results are not confirmation. Loss is full minus compressed quality in percentage points; upper bounds are one-sided 95%.", "",
             "The reference keeps every image patch and all text tokens in the same protected-prefix inference backend. It is not a separate stock-DynamicCache performance baseline.", "",
             "## Quality and compression", "",
             "| Model | Dataset | Method | N | Quality (%) | Loss (pp) | Upper loss (pp) | Aggregate patch reduction (%) | Mean per-input patch reduction (%) | Compressed fraction | Full-fallback fraction | Numeric pass | Accepted | Status |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for panel in report["quality_compression"]:
        memory = panel.get("memory", {})
        values = display_identity(panel)
        values += [str(panel["questions"])] + [number(panel.get(key)) for key in (
            "candidate_quality_percent", "mean_loss_pp", "upper_loss_pp")]
        values += [number(memory.get(key)) for key in ("aggregate_patch_reduction_percent",
            "mean_per_input_patch_reduction_percent", "fraction_compressed", "fraction_full_fallback")]
        values += [str(panel["passes_numeric_rule"]), str(panel["accepted"]), text(panel["status"])]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "## Systems measurements", "",
              "Allocated KV capacity, measured live KV, and full-request GPU peak are different quantities. Timings are descriptive; no controlled speedup is inferred. Missing measurements are N/A.", "",
              "| Model | Dataset | Method | Mean allocated KV (bytes) | Mean live KV (bytes) | Mean GPU peak (bytes) | Mean end-to-end (s) | Mean TTFT (s) | Mean selector (s) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for panel in report["systems"]:
        values = display_identity(panel)
        values += [number(panel.get(key)) for key in ("mean_allocated_kv_bytes", "mean_actual_kv_bytes",
            "mean_peak_gpu_bytes", "mean_end_to_end_seconds", "mean_ttft_seconds", "mean_selector_seconds")]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


tables = render_tables
