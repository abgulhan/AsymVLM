"""Exact-budget physical 2x2 reporting, separating ranking and allocation.

Consumes audited development outputs only. No label loading, policy fitting,
timing comparisons, or claims of independent confirmation occur here.
"""

from collections import defaultdict
import json
from pathlib import Path

from . import compression_report_v8 as old
from . import compression_study_v8 as projection
from .adaptive_report_v3 import paired_ci, quality_summary
from .compression_policy_v10 import canonical_dataset
from .compression_matrix_v8 import seal
from .state import digest, file_digest

FORMAT = "mlsys_compression_report_v11"


def _verify_physical_provenance(row, *, visited=()):
    """A reuse view is physical evidence, but is never a new measurement.

    Revalidate its original content hash and all reported outcomes/memory. Some
    older source schemas predate runtime-contract hashes; additive audited
    runtime metadata is permitted, but existing fields cannot change.
    """
    if row.get("physical_measurement_performed") is True:
        return False
    source_path, expected = row.get("physical_source_path"), row.get("physical_source_sha256")
    if not isinstance(source_path, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError("only physically measured outputs or hash-verified original physical reuse may be reported")
    path = Path(source_path).resolve()
    if str(path) in visited or len(visited) >= 8:
        raise ValueError("cyclic or excessive physical provenance chain")
    if file_digest(path) != expected:
        raise ValueError("reused physical source content hash changed")
    original = json.loads(path.read_text())
    _verify_physical_provenance(original, visited=(*visited, str(path)))
    required = ("request_id", "group_id", "quality_metric", "quality_score", "input_visual_tokens",
                "input_text_tokens", "selected_count", "retained_visual_tokens", "selector")
    for field in required:
        if field not in row or field not in original or row[field] != original[field]:
            raise ValueError("physical reuse changed original outcome or geometry: " + field)
    model_matches = (row["model"] == original["model"] if "model" in row and "model" in original
                     else old._model(row) == old._model(original))
    if canonical_dataset(row["dataset"]) != canonical_dataset(original["dataset"]) or not model_matches:
        raise ValueError("physical reuse changed model or dataset")
    if (row.get("category") or None) != (original.get("category") or None):
        raise ValueError("physical reuse changed category")
    for field in ("prediction", "generated_token_ids", "selected_patch_indices", "metrics"):
        if row.get(field) != original.get(field):
            raise ValueError("physical reuse changed original measurement: " + field)
    for field in old.RUNTIME_FIELDS:
        if field in original and row.get(field) != original[field]:
            raise ValueError("physical reuse changed runtime identity: " + field)
    return True


def _harm(candidate, full):
    a, b = old._index(candidate), old._index(full)
    metric = candidate[0]["quality_metric"]
    groups, strata = old._clusters(a, b, metric)
    if metric == "mme":
        losses = [sum(max(0., v) for v in values) / len(values) for values in strata.values()]
        gains = [sum(max(0., -v) for v in values) / len(values) for values in strata.values()]
        harm, gain = sum(losses) / len(losses), sum(gains) / len(gains)
    else:
        losses = [b[k]["quality_score"] - a[k]["quality_score"] for k in a]
        harm, gain = sum(max(0., v) for v in losses) / len(losses), sum(max(0., -v) for v in losses) / len(losses)
    return {"positive_harm_pp": 100 * harm, "pruning_gain_pp": 100 * gain,
            "questions_improved": sum(a[k]["quality_score"] > b[k]["quality_score"] for k in a),
            "questions_harmed": sum(a[k]["quality_score"] < b[k]["quality_score"] for k in a),
            "questions_unchanged": sum(a[k]["quality_score"] == b[k]["quality_score"] for k in a),
            "independent_groups": len(groups)}


def _paired(candidate, reference, repetitions):
    a, b = old._index(candidate), old._index(reference)
    if set(a) != set(b):
        raise ValueError("paired contrast requires identical cohorts")
    for key in a:
        for field in ("model_tag", "group_id", "dataset", "quality_metric", "category",
                      "input_visual_tokens", "input_text_tokens", *old.RUNTIME_FIELDS):
            if (field in a[key] or field in b[key]) and a[key].get(field) != b[key].get(field):
                raise ValueError("paired runtime or input identity mismatch: " + field)
    if sum(r["retained_visual_tokens"] for r in a.values()) != sum(r["retained_visual_tokens"] for r in b.values()):
        raise ValueError("all four compressed cells must match exact aggregate token count")
    return {**paired_ci(candidate, reference, candidate[0]["quality_metric"], repetitions=repetitions, seed=20261020),
        "interval_interpretation": "exploratory paired image/document-group interval; zero-width intervals do not establish equivalence",
        "token_error": 0, "confirmed": False}


def report(study, model, physical_rows, expected_countvectors, *, alternative_selector,
           bootstrap_repetitions=10000):
    """Report common A/U vectors applied to scorer and one alternative selector.

    expected_countvectors is {target: {'adaptive': {request: count},
                                      'uniform': {request: count}}}.
    Each target must cover the identical predeclared cohort in all four cells.
    A/U totals must match exactly WITHIN each dataset. Full references are shared.
    """
    if alternative_selector not in ("divprune", "cdpruner"):
        raise ValueError("supported independent alternative selector required")
    if model == "qwen25" and alternative_selector == "cdpruner":
        raise ValueError("Qwen CDPruner is not supported by the audited pipeline")
    if not expected_countvectors or not isinstance(study.get("sha256"), str):
        raise ValueError("frozen study identity and predeclared count vectors required")
    if type(bootstrap_repetitions) is not int or bootstrap_repetitions < 100:
        raise ValueError("at least 100 grouped bootstrap repetitions required")
    normalized = []
    verified_reuse_sources = {}
    for raw in physical_rows:
        reused = _verify_physical_provenance(raw)
        if reused:
            verified_reuse_sources[raw["physical_source_path"]] = raw["physical_source_sha256"]
        if old._model(raw) != model:
            raise ValueError("physical row belongs to a different model")
        normalized.append({**raw, "dataset": canonical_dataset(raw["dataset"])})
    index = projection.physical_index(normalized)
    panels, macro = [], {}
    all_ids = None
    for target, vectors in sorted(expected_countvectors.items(), key=lambda item: str(item[0])):
        if set(vectors) != {"adaptive", "uniform"} or not vectors["adaptive"] or set(vectors["adaptive"]) != set(vectors["uniform"]):
            raise ValueError("adaptive and uniform must cover the exact same nonempty cohort")
        ids = set(vectors["adaptive"])
        if all_ids is None:
            all_ids = ids
        elif ids != all_ids:
            raise ValueError("targets must use the same predeclared cohort")
        full = {}
        methods = {selector + "_" + allocation: {} for selector in ("scorer", alternative_selector)
                   for allocation in ("adaptive", "uniform")}
        for request in sorted(ids):
            full_options = [r for (rid, selector, count), r in index.items() if rid == request and selector == "full"]
            if len(full_options) != 1:
                raise ValueError("one physical all-token reference is required for each request")
            source = full_options[0]
            n = source["input_visual_tokens"]
            if source["selected_count"] != n:
                raise ValueError("full reference was pruned")
            full[request] = source
            for allocation, counts in vectors.items():
                k = counts[request]
                if type(k) is not int or not 1 <= k <= n:
                    raise ValueError("invalid frozen integer token count")
                for selector in ("scorer", alternative_selector):
                    measured = index.get((request, "full" if k == n else selector, k))
                    if measured is None:
                        raise ValueError(f"missing physical 2x2 cell: {request} {selector} {k}")
                    method = selector + "_" + allocation
                    methods[method][request] = {**measured, "method": method,
                        "timing_reused_not_an_independent_observation": True}
        by_dataset = defaultdict(list)
        for request, r in full.items():
            by_dataset[r["dataset"]].append(request)
        for dataset, requests in sorted(by_dataset.items()):
            total = {allocation: sum(v[request] for request in requests) for allocation, v in vectors.items()}
            if total["adaptive"] != total["uniform"]:
                raise ValueError("per-dataset adaptive and uniform token totals must be identical")
            refs = [full[r] for r in requests]
            views = {method: [values[r] for r in requests] for method, values in methods.items()}
            summaries = {}
            for method, values in views.items():
                # The full-reference validator also checks runtime identity and
                # an intact predeclared cohort. Its one-sided bound is retained
                # as development uncertainty, never a promotion criterion here.
                checked = old.compare_to_full(values, refs, expected_request_ids=requests)
                summaries[method] = {
                    "quality_percent": checked["candidate_quality_percent"],
                    "full_quality_percent": checked["full_quality_percent"],
                    "full_minus_method_pp": checked["mean_loss_pp"],
                    "full_relative_upper_loss_pp": checked["upper_loss_pp"],
                    "full_relative_uncertainty_status": checked["uncertainty_status"],
                    "full_relative_bound_method": checked["bound_method"],
                    "memory": checked["memory"], **_harm(values, refs),
                    "systems_claim": "no timing comparison; mixed cached outputs are not controlled systems measurements",
                    "peak_memory_interpretation": "physical descriptive measurement only; controlled exclusive-run memory evidence is separate",
                }
            contrasts = {
                "scorer_adaptive_minus_uniform": _paired(views["scorer_adaptive"], views["scorer_uniform"], bootstrap_repetitions),
                "alternative_adaptive_minus_uniform": _paired(views[alternative_selector + "_adaptive"], views[alternative_selector + "_uniform"], bootstrap_repetitions),
                "alternative_minus_scorer_at_uniform": _paired(views[alternative_selector + "_uniform"], views["scorer_uniform"], bootstrap_repetitions),
                "alternative_minus_scorer_at_adaptive": _paired(views[alternative_selector + "_adaptive"], views["scorer_adaptive"], bootstrap_repetitions),
            }
            panels.append({"target": target, "dataset": dataset, "questions": len(requests),
                "independent_groups": len({r["group_id"] for r in refs}),
                "aggregate_visual_tokens_each_compressed_cell": total["adaptive"],
                "full": quality_summary(refs, refs[0]["quality_metric"]),
                "methods": summaries, "contrasts": contrasts})
        target_panels = [p for p in panels if p["target"] == target]
        macro[str(target)] = {name: sum(p["contrasts"][name]["delta_pp"] for p in target_panels) / len(target_panels)
                              for name in target_panels[0]["contrasts"]}
    return seal({"format": FORMAT, "study_sha256": study["sha256"], "model": model,
        "alternative_selector": alternative_selector, "panels": panels,
        "equal_benchmark_mean_delta_pp": macro, "macro_confidence_interval": None,
        "counts_sha256": digest(expected_countvectors),
        "physical_rows_sha256": digest(normalized), "evidence": "development only; not independent confirmation",
        "verified_reused_physical_sources": verified_reuse_sources,
        "reuse_note": "original physical outcome hashes revalidated; reuse views are not new inference or independent measurements",
        "baseline_scope": "selector adaptations, not official end-to-end pipelines",
        "all_four_compressed_cells_exact_budget_matched": True,
        "full_reference_budget_scope": "uncompressed full reference intentionally has a different token count",
        "source_hashes": {str(Path(p).resolve()): file_digest(p) for p in (__file__, old.__file__, projection.__file__)}})


def markdown(result):
    """Two compact tables: accuracy effects, then allocation/measured memory."""
    alternative = result["alternative_selector"]
    label = {"scorer": "AsymVLM learned selector", "divprune": "DivPrune adaptation", "cdpruner": "CDPruner adaptation"}
    methods = ("scorer_uniform", "scorer_adaptive", alternative + "_uniform", alternative + "_adaptive")
    def interval(item):
        if item["ci_lower_pp"] is None:
            return f"{item['delta_pp']:+.2f} (CI unavailable)"
        return f"{item['delta_pp']:+.2f} [{item['ci_lower_pp']:+.2f}, {item['ci_upper_pp']:+.2f}]"
    lines = [f"# {result['model']}: exact-budget selector and allocation comparison", "",
        "Development only; these data are not independent confirmation. Baselines are selector adaptations, not official end-to-end pipelines. "
        "All four compressed cells use identical aggregate visual-token counts within each benchmark. The all-token reference is uncompressed.", "",
        "## Table 1. Accuracy and adaptive-budget effect", "",
        f"Alternative patch selector: {label[alternative]}. Parentheses show adaptive minus uniform in percentage points with exploratory paired 95% intervals; resampling keeps image/document groups intact.", "",
        "| Dataset | Target | All tokens | AsymVLM uniform | AsymVLM adaptive | Alternative uniform | Alternative adaptive | AsymVLM budget effect [95% CI] | Alternative budget effect [95% CI] |",
        "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for panel in result["panels"]:
        qualities = " | ".join(f"{panel['methods'][method]['quality_percent']:.2f}" for method in methods)
        lines.append(f"| {panel['dataset']} | {panel['target']} | {panel['full']['score_percent']:.2f} | {qualities} | "
                     f"{interval(panel['contrasts']['scorer_adaptive_minus_uniform'])} | "
                     f"{interval(panel['contrasts']['alternative_adaptive_minus_uniform'])} |")
    lines += ["", "## Table 2. Allocation and measured cache reduction", "",
        "Patch reduction and allocated KV-cache reduction are different quantities. Missing physical memory measurements remain unmeasured. "
        "No latency, throughput, or controlled GPU-peak claim is inferred from mixed cached outputs.", "",
        "| Dataset | Target | Patch selector | Budget rule | Retained visual tokens | Patch reduction | Inputs with fewer patches | Mean allocated KV reduction | Positive harm vs all tokens (pp) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for panel in result["panels"]:
        for method in methods:
            selector, allocation = method.rsplit("_", 1)
            values = panel["methods"][method]
            memory = values["memory"]
            kv = memory["mean_per_input_allocated_kv_reduction_percent"]
            kvtext = "unmeasured" if kv is None else f"{kv:.2f}%"
            lines.append(f"| {panel['dataset']} | {panel['target']} | {label[selector]} | {allocation} | "
                f"{memory['aggregate_retained_visual_tokens']} | {memory['aggregate_patch_reduction_percent']:.2f}% | "
                f"{100*memory['fraction_compressed']:.1f}% | {kvtext} | {values['positive_harm_pp']:.2f} |")
    lines += ["", "Positive harm counts losses without cancelling them against pruning-induced improvements. "
        "For MME it is computed on the native image-pair/category metric. A zero-width bootstrap interval does not prove population equivalence. "
        "Exact alternative-minus-AsymVLM contrasts at both count vectors are in the JSON report.", ""]
    return "\n".join(lines)
