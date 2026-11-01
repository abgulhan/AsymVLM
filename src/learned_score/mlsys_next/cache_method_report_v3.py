"""Versioned policy-matrix reporting; source-level uncertainty and seed averaging.

Official Boolean checks are scored separately for every physical run. Random
seeds are averaged within each page/check BEFORE source-cluster resampling.
Development selection and descriptive mechanism diagnostics are not final tests.
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import mean

from learned_score.mlsys_eval.common import canonical_hash, file_sha256, write_new_json
from learned_score.mlsys_eval.olmocr_score import score_candidate, verify_evaluator
from .cache_method_report import write_new_text


POLICIES = ("protected_recency", "h2o", "h2o_recent_adjusted", "protected_h2o", "rkv", "protected_rkv", "protected_random")
RANDOM_SEEDS = (42, 43, 44)
POLICY_LABELS = {"none": "All tokens, no eviction", "protected_recency": "AsymVLM: protected prefix + recent text",
                 "h2o": "H2O adaptation", "h2o_recent_adjusted": "H2O adaptation, adjusted recent allocation",
                 "protected_h2o": "AsymVLM: protected prefix + H2O text selection", "rkv": "R-KV adaptation",
                 "protected_rkv": "AsymVLM: protected prefix + R-KV text selection",
                 "protected_random": "Protected prefix + random text selection (3-seed mean)"}


def policy(row):
    value = row.get("cache_policy", row.get("cache_method"))
    return "none" if value == "fullref" else value


def staging(row):
    return row.get("staging_slots", row["metrics"]["cache"].get("overflow_allowance"))


def logical_key(row):
    return "full_v100_off" if policy(row) == "none" else f"{policy(row)}__{row['comparison_group']}"


def verify_budget_matching(rows, expected_policies=POLICIES):
    expected = set(expected_policies)
    if not expected or not expected <= set(POLICIES):
        raise ValueError("invalid declared cache policy matrix")
    paired, cross_vision = defaultdict(lambda: defaultdict(dict)), defaultdict(set)
    for row in rows:
        name = policy(row)
        if name == "none":
            continue
        if name not in expected:
            raise ValueError("undeclared policy")
        seed = row.get("random_seed")
        if (name == "protected_random" and seed not in RANDOM_SEEDS) or (name != "protected_random" and seed is not None):
            raise ValueError("invalid policy random seed")
        key = row["model"], row["comparison_group"], row["request_id"]
        if seed in paired[key][name]:
            raise ValueError("duplicate request/policy/seed")
        slots, size = row["kv_budget_slots"], row["kv_budget_bytes"]
        cache, slack = row["metrics"]["cache"], staging(row)
        profile = re.fullmatch(r"(?:base|full_prefix_plus_)?(512|1024)", str(row["budget_profile"]))
        if not profile or slack not in (1, 128):
            raise ValueError("invalid fixed physical budget or staging size")
        if any(type(v) is not int or v <= 0 for v in (slots, size)):
            raise ValueError("physical KV capacities must be positive integers")
        if (slots != row["input_prefix_tokens"] + int(profile[1]) + 1
                or cache["capacity_tokens_per_layer"] != slots or cache["kv_capacity_bytes"] != size
                or row["kv_retained_slots"] != slots - slack
                or not 0 <= cache["active_tokens"] <= slots
                or not 0 <= cache["peak_active_tokens"] <= slots):
            raise ValueError("physical allocation/occupancy violates M=Pfull+B+1")
        paired[key][name][seed] = (slots, size)
        cross_vision[row["model"], row["request_id"], int(profile[1])].add((slots, size))
    if not paired or any(len(v) != 1 for v in cross_vision.values()):
        raise ValueError("physical budget differs across vision/staging conditions")
    for cells in paired.values():
        if set(cells) != expected:
            raise ValueError("incomplete per-request policy matrix")
        capacities = set()
        for name, runs in cells.items():
            if set(runs) != (set(RANDOM_SEEDS) if name == "protected_random" else {None}):
                raise ValueError("incomplete random-seed matrix")
            capacities.update(runs.values())
        if len(capacities) != 1:
            raise ValueError("policies have unequal physical KV budgets")


def _check_index(checks):
    result = {}
    for row in checks:
        key = row["request_id"], row["category"], row["test_id"]
        value = row["passed"]
        if key in result or not isinstance(value, (bool, int, float)) or not 0 <= value <= 1:
            raise ValueError("duplicate or invalid averaged official check")
        result[key] = row
    return result


def macro(checks):
    values = defaultdict(list)
    for row in checks:
        values[row["category"]].append(float(row["passed"]))
    if not values:
        raise ValueError("empty official checks")
    categories = {key: mean(v) for key, v in sorted(values.items())}
    return {"official_macro_score": mean(categories.values()), "category_scores": categories,
            "category_check_counts": {k: len(v) for k, v in values.items()}}


def aggregate_seed_scores(scores, *, method, expected_seeds=RANDOM_SEEDS):
    if set(scores) != set(expected_seeds):
        raise ValueError("seed aggregation requires the exact predeclared seeds")
    indexed = {seed: _check_index(score["checks"]) for seed, score in scores.items()}
    first = next(iter(indexed.values()))
    if any(rows.keys() != first.keys() for rows in indexed.values()):
        raise ValueError("random seeds have different pages/checks")
    checks = []
    for key, row in first.items():
        identity = (row["group_id"], row["model"], row["split"], row["manifest_sha256"], row["run_context"])
        if any((r[key]["group_id"], r[key]["model"], r[key]["split"], r[key]["manifest_sha256"], r[key]["run_context"]) != identity
               for r in indexed.values()):
            raise ValueError("random seeds changed paired identity/context")
        if any(type(r[key]["passed"]) is not bool for r in indexed.values()):
            raise ValueError("seed inputs must be official Boolean checks")
        checks.append({**row, "method": method, "passed": mean(int(r[key]["passed"]) for r in indexed.values())})
    first_score = next(iter(scores.values()))
    return {**{k: first_score[k] for k in ("model", "split", "manifest_sha256", "run_context", "quality_metric")},
            "method": method, "checks": checks, **macro(checks), "random_seeds": list(expected_seeds),
            "aggregation": "mean within page/check before paired source-cluster resampling; not extra pages"}


def paired_difference(candidate, reference, *, repetitions=10000, seed=2027):
    import numpy as np
    left, right = _check_index(candidate["checks"]), _check_index(reference["checks"])
    if not left or left.keys() != right.keys():
        raise ValueError("paired scores must contain identical nonempty official checks")
    contexts = {(r["model"], r["split"], r["manifest_sha256"], r["run_context"])
                for r in [*left.values(), *right.values()]}
    if len(contexts) != 1:
        raise ValueError("different models, splits, manifests or run contexts")
    grouped = defaultdict(lambda: defaultdict(list))
    for key, row in left.items():
        if row["group_id"] != right[key]["group_id"]:
            raise ValueError("paired source identity changed")
        grouped[row["group_id"]][row["category"]].append(float(row["passed"]) - float(right[key]["passed"]))
    categories = sorted({c for values in grouped.values() for c in values})
    counts = {c: sum(c in v for v in grouped.values()) for c in categories}
    point = 100 * (macro(left.values())["official_macro_score"] - macro(right.values())["official_macro_score"])
    result = {"delta_pp": point, "n_pages": len({k[0] for k in left}), "n_source_groups": len(grouped),
              "sources_by_category": counts, "ci_lower_pp": None, "ci_upper_pp": None,
              "estimand": "equal-category macro of per-check seed-mean pass rates"}
    if min(counts.values()) < 2:
        return {**result, "ci_unavailable_reason": "a category has fewer than two independent sources; its uncertainty is not estimable"}
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap repetitions required")
    strata = defaultdict(list)
    for group, values in grouped.items():
        strata[tuple(sorted(values))].append(group)
    rng = np.random.default_rng(seed)
    samples = np.zeros(repetitions)
    # Vectorized source draws preserve cross-category correlation within a source.
    totals = {c: np.zeros(repetitions) for c in categories}
    sizes = {c: np.zeros(repetitions) for c in categories}
    for signature, groups in sorted(strata.items()):
        draws = rng.integers(0, len(groups), size=(repetitions, len(groups)))
        for c in signature:
            sums = np.asarray([sum(grouped[g][c]) for g in groups])
            ns = np.asarray([len(grouped[g][c]) for g in groups])
            totals[c] += sums[draws].sum(axis=1)
            sizes[c] += ns[draws].sum(axis=1)
    for c in categories:
        samples += totals[c] / sizes[c] / len(categories)
    low, high = np.quantile(samples * 100, [.025, .975])
    return {**result, "ci_lower_pp": float(low), "ci_upper_pp": float(high),
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed,
            "singleton_strata_warning": any(len(v) == 1 for v in strata.values()),
            "resampling": "paired source clusters stratified by category-membership signature"}


def _logical_scores(rows, scores):
    bundles = defaultdict(list)
    for row in rows:
        bundles[logical_key(row)].append(row)
    if {r["method"] for r in rows} != set(scores):
        raise ValueError("official score cells differ from physical cells")
    output = {}
    for key, records in bundles.items():
        cells = {r["method"]: r.get("random_seed") for r in records}
        if policy(records[0]) == "protected_random":
            if len(cells) != 3 or len(set(cells.values())) != 3:
                raise ValueError("ambiguous random seed cells")
            output[key] = aggregate_seed_scores({seed: scores[cell] for cell, seed in cells.items()}, method=key)
        elif len(cells) != 1:
            raise ValueError("ambiguous deterministic policy cell")
        else:
            output[key] = scores[next(iter(cells))]
    return bundles, output


def quality_table(panel, scores):
    rows = list(panel["rows"].values())
    verify_budget_matching(rows, panel["metadata"].get("cache_policies", POLICIES))
    bundles, logical_scores = _logical_scores(rows, scores)
    if "full_v100_off" not in bundles:
        raise ValueError("missing unrestricted full-token reference")
    requests = {r["request_id"] for r in bundles["full_v100_off"]}
    output = []
    for key, records in sorted(bundles.items()):
        first, score = records[0], logical_scores[key]
        if {r["request_id"] for r in records} != requests or score["run_context"] != panel["context"]:
            raise ValueError("quality comparison changed requests or run context")
        replicas = 3 if policy(first) == "protected_random" else 1
        if len(records) != replicas * len(requests):
            raise ValueError("duplicate/missing page-seed rows")
        comparisons = {}
        if policy(first) != "none":
            comparisons["all_tokens_no_eviction"] = paired_difference(score, logical_scores["full_v100_off"])
            baseline = f"protected_recency__{first['comparison_group']}"
            comparisons["protected_recency_same_budget"] = paired_difference(score, logical_scores[baseline])
            h2o = f"h2o__{first['comparison_group']}"
            if h2o in logical_scores:
                comparisons["h2o_same_budget"] = paired_difference(score, logical_scores[h2o])
        output.append({"model": first["model"], "model_name": panel["metadata"].get("model_name"),
                       "method": key, "cache_policy": policy(first), "method_label": POLICY_LABELS[policy(first)],
                       "comparison_group": first["comparison_group"], "visual_retention": first["visual_retention"],
                       "budget_profile": first["budget_profile"], "staging_slots": None if policy(first) == "none" else staging(first),
                       "requests": len(requests), "physical_runs": len(records), "seeds_per_request": replicas,
                       "score_percent": 100 * score["official_macro_score"],
                       "category_scores_percent": {k: 100 * v for k, v in score["category_scores"].items()},
                       "mean_outputs_at_cap": sum(r["generated_tokens"] == r["max_new_tokens"] for r in records) / replicas,
                       "eviction_active_fraction": mean(r["metrics"]["cache"]["compaction_events"] > 0 for r in records),
                       "paired_comparisons": comparisons, "stage": "development", "run_context": panel["context"],
                       "selection_not_confirmation": True})
    return output


def systems_table(rows):
    groups = defaultdict(list)
    for row in rows:
        execution = row["execution_context"]
        if not execution.get("cluster") or not execution.get("hostname"):
            raise ValueError("missing platform/node provenance")
        key = (row["model"], logical_key(row), row["hardware"], execution["cluster"], execution["hostname"],
               execution.get("torch_version"), execution.get("cuda_version"), row["run_context"])
        groups[key].append(row)
    result = []
    for key, records in sorted(groups.items(), key=lambda x: str(x[0])):
        first = records[0]
        def cache_stat(field, fn=max):
            values = [r["metrics"]["cache"].get(field) for r in records]
            return None if any(v is None for v in values) else fn(values)
        result.append({"model": key[0], "method": key[1], "method_label": POLICY_LABELS[policy(first)],
                       "cache_policy": policy(first), "hardware": key[2], "cluster": key[3], "hostname": key[4],
                       "torch_version": key[5], "cuda_version": key[6], "run_context": key[7],
                       "requests": len({r["request_id"] for r in records}), "physical_runs": len(records),
                       "budget_profile": first["budget_profile"], "visual_retention": first["visual_retention"],
                       "staging_slots": None if policy(first) == "none" else staging(first),
                       "eviction_active_fraction": mean(r["metrics"]["cache"]["compaction_events"] > 0 for r in records),
                       "mean_generated_tokens": mean(r["generated_tokens"] for r in records),
                       "mean_end_to_end_seconds": mean(r["end_to_end_seconds"] for r in records),
                       "mean_ttft_seconds": mean(r["ttft_seconds"] for r in records),
                       "maximum_gpu_allocated_bytes": max(r["peak_allocated_bytes"] for r in records),
                       "mean_allocated_kv_bytes": cache_stat("kv_capacity_bytes", mean),
                       "maximum_allocated_kv_bytes": cache_stat("kv_capacity_bytes"),
                       "maximum_active_kv_bytes": cache_stat("peak_active_kv_bytes"),
                       "mean_final_active_slots": cache_stat("active_tokens", mean),
                       "maximum_auxiliary_bytes": cache_stat("auxiliary_persistent_bytes"),
                       "maximum_attention_scratch_bytes_estimate": cache_stat("attention_statistics_scratch_peak_bytes_estimate"),
                       "maximum_selection_scratch_bytes_estimate": cache_stat("selection_scratch_peak_bytes_estimate"),
                       "maximum_rolling_rotary_scratch_bytes_estimate": cache_stat("rolling_rotary_scratch_peak_bytes_estimate"),
                       "maximum_owned_cache_bytes": cache_stat("allocated_cache_and_auxiliary_bytes"),
                       "timing_isolated": False, "serving_throughput": None,
                       "timing_caution": "parallel natural-stop quality measurements, not isolated speedups"})
    return result


def select_shared_policy(reports, *, primary_group="v100_b512", candidates=("protected_h2o", "protected_rkv"), isolated_latencies=None):
    if len(reports) != 2:
        raise ValueError("shared selection requires both model development reports")
    model_rows = []
    for report in reports:
        rows = {r["cache_policy"]: r for r in report["quality_table"] if r["comparison_group"] == primary_group}
        if any(name not in rows for name in (*candidates, "protected_recency")) or any(r["requests"] != 100 or r["stage"] != "development" for r in rows.values()):
            raise ValueError("selection requires complete 100-page development policy matrix")
        model_rows.append(rows)
    if len({r["protected_recency"]["model"] for r in model_rows}) != 2:
        raise ValueError("shared selection requires two different model architectures")
    ranking = []
    for name in candidates:
        gains = [rows[name]["score_percent"] - rows["protected_recency"]["score_percent"] for rows in model_rows]
        latency = (isolated_latencies or {}).get(name)
        if latency is not None and (not latency.get("isolated") or not latency.get("fixed_output_length") or latency["mean_seconds"] <= 0):
            raise ValueError("latency tie-break requires isolated, fixed-output measurements")
        ranking.append({"cache_policy": name, "model_gains_pp": gains, "worst_gain_pp": min(gains),
                        "mean_gain_pp": mean(gains), "isolated_latency_seconds": None if latency is None else latency["mean_seconds"]})
    # Missing latency cannot silently defeat a measured method: use the tie-break
    # only if every candidate has the same qualified isolated measurement scope.
    use_latency = bool(ranking) and all(r["isolated_latency_seconds"] is not None for r in ranking)
    ranking.sort(key=lambda r: (-r["worst_gain_pp"], -r["mean_gain_pp"],
                               r["isolated_latency_seconds"] if use_latency else 0, r["cache_policy"]))
    winner = ranking[0] if ranking else None
    promote = winner is not None and winner["mean_gain_pp"] >= .5 - 1e-12 and winner["worst_gain_pp"] >= -1e-12
    tied = [r for r in ranking if winner is not None and abs(r["worst_gain_pp"] - winner["worst_gain_pp"]) <= 1e-12
            and abs(r["mean_gain_pp"] - winner["mean_gain_pp"]) <= 1e-12]
    needs_latency = promote and len(tied) > 1 and not use_latency
    promote = promote and not needs_latency
    return {"complete": True, "selected_policy": winner["cache_policy"] if promote else "protected_recency",
            "updated": promote, "ranking": ranking, "primary_comparison_group": primary_group,
            "selection_rule": "worst-model gain, mean gain, qualified isolated latency, lexical name; update only mean>=0.5pp and neither model negative",
            "eligible_policies": list(candidates), "latency_tiebreak_used": use_latency,
            "selection_ready": not needs_latency, "requires_isolated_latency_tiebreak": needs_latency,
            "development_only": True, "final_labels_opened": False,
            "source_reports": [r["run_context"] for r in reports]}


def repetition_fraction(text, n=8):
    words = text.split()
    windows = [tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))]
    return 0. if not windows else 1 - len(set(windows)) / len(windows)


def mechanism_diagnostics(candidate, reference, candidate_rows, reference_rows):
    """Already-scored development diagnostics, never a causal/significance claim."""
    left, right = _check_index(candidate["checks"]), _check_index(reference["checks"])
    paired_difference(candidate, reference, repetitions=100)  # identity validation, singleton guard preserved
    counts = Counter(r["category"] for r in left.values())
    sources, pages = defaultdict(float), defaultdict(float)
    for key, row in left.items():
        value = 100 * (float(row["passed"]) - float(right[key]["passed"])) / counts[row["category"]] / len(counts)
        sources[row["group_id"]] += value
        pages[row["request_id"]] += value
    leave_one_out = {}
    for group in sorted(sources):
        a = [r for r in left.values() if r["group_id"] != group]
        b = [r for r in right.values() if r["group_id"] != group]
        leave_one_out[group] = (None if {r["category"] for r in a} != set(counts)
                               else 100 * (macro(a)["official_macro_score"] - macro(b)["official_macro_score"]))
    arows, brows = {r["request_id"]: r for r in candidate_rows}, {r["request_id"]: r for r in reference_rows}
    if len(arows) != len(candidate_rows) or len(brows) != len(reference_rows) or arows.keys() != brows.keys() or arows.keys() != pages.keys():
        raise ValueError("diagnostics require one physical run per paired page")
    cap = defaultdict(lambda: {"pages": 0, "contribution_pp": 0.})
    repeat = {"candidate": [], "reference": []}
    for request, row in arows.items():
        other = brows[request]
        key = f"candidate_{'capped' if row['generated_tokens'] == row['max_new_tokens'] else 'uncapped'}__reference_{'capped' if other['generated_tokens'] == other['max_new_tokens'] else 'uncapped'}"
        cap[key]["pages"] += 1
        cap[key]["contribution_pp"] += pages[request]
        for name, item in (("candidate", row), ("reference", other)):
            repeat[name].append(repetition_fraction(item["prediction"]))
    valid = [v for v in leave_one_out.values() if v is not None]
    return {"source_contribution_pp": dict(sorted(sources.items(), key=lambda x: (-x[1], x[0]))),
            "page_contribution_pp": dict(sorted(pages.items(), key=lambda x: (-x[1], x[0]))),
            "leave_one_source_out_pp": leave_one_out,
            "leave_one_source_out_range_pp": [min(valid), max(valid)] if valid else None,
            "leave_one_source_out_is_not_ci": True, "output_cap_strata": dict(cap),
            "repeated_eight_word_windows": {name: {"mean_fraction": mean(values), "pages_above_half": sum(v > .5 for v in values)}
                                           for name, values in repeat.items()},
            "caution": "post-hoc descriptive development diagnostics; cap strata are post-treatment, not causal effects"}


def render_tables(quality, systems):
    def safe(v):
        return str(v).replace("|", "\\|").replace("\n", " ")
    def delta(row, name):
        comparison = row["paired_comparisons"].get(name)
        if comparison is None:
            return "reference"
        point = f"{comparison['delta_pp']:+.2f}"
        return point + (" (CI unavailable)" if comparison["ci_lower_pp"] is None else
                        f" [{comparison['ci_lower_pp']:+.2f}, {comparison['ci_upper_pp']:+.2f}]")
    lines = ["## Table 1. Development document quality at equal physical KV capacity", "",
             "Official category-balanced olmOCR scores (%); differences are percentage points. M = original full prefix + stated extra slots + 1. "
             "Staging space is included in M, not added on top. Random-policy seeds 42/43/44 are averaged within each page/check before source-level confidence intervals; they do not increase the sample size.", "",
             "| Model | Policy | Vision / extra slots / staging | Pages | Score | Difference vs protected recency [95% CI] | Difference vs H2O [95% CI] | Difference vs all tokens [95% CI] | Mean capped pages |",
             "|---|---|---|---:|---:|---|---|---|---:|"]
    for row in quality:
        cells = [row["model"], row["method_label"], f"{row['visual_retention']:.0%} / {row['budget_profile']} / {row['staging_slots']}", row["requests"],
                 f"{row['score_percent']:.2f}", delta(row, "protected_recency_same_budget"), delta(row, "h2o_same_budget"),
                 delta(row, "all_tokens_no_eviction"), f"{row['mean_outputs_at_cap']:.1f}"]
        lines.append("| " + " | ".join(map(safe, cells)) + " |")
    lines += ["", "CI unavailable means source independence is insufficient for that interval; point gains are not significance claims.", "",
              "## Table 2. Observed memory and runtime in quality runs", "",
              "Nodes/platforms are not pooled. K/V tensor capacity, occupied K/V, auxiliary memory and measured CUDA peaks are distinct. "
              "Selection scratch is an analytical estimate, not a measured allocator peak; attention/rotary scratch estimates are retained separately in the JSON report. "
              "Parallel natural-stop measurements are not isolated speedups or serving throughput. H2O statistics are recomputed after shared attention for numerical parity; this is not optimized upstream timing.", "",
              "| Model | Policy | Platform / node | Pages / runs | Staging | Eviction active | Mean output tokens | Mean allocated KV (MiB) | Peak occupied KV (MiB) | Resident auxiliary / selection scratch estimate (MiB) | Peak GPU (MiB) | Mean latency / first token (s) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in systems:
        def mib(key):
            return "unavailable" if row[key] is None else f"{row[key] / 2**20:.2f}"
        cells = [row["model"], row["method_label"], f"{row['cluster']} / {row['hostname']}", f"{row['requests']} / {row['physical_runs']}", row["staging_slots"],
                 f"{row['eviction_active_fraction']:.0%}", f"{row['mean_generated_tokens']:.1f}", mib("mean_allocated_kv_bytes"), mib("maximum_active_kv_bytes"),
                 mib("maximum_auxiliary_bytes") + " / " + mib("maximum_selection_scratch_bytes_estimate"), mib("maximum_gpu_allocated_bytes"), f"{row['mean_end_to_end_seconds']:.2f} / {row['mean_ttft_seconds']:.2f}"]
        lines.append("| " + " | ".join(map(safe, cells)) + " |")
    return "\n".join(lines + ["", "Development results are used for policy selection, not final confirmation. Final annotation access is separately gated.", ""])


def score_panel(panel, manifest, output, *, evaluator_root, cache_patch=None):
    if not panel.get("complete") or panel["metadata"]["split"] != "scorer_dev" or len(panel["entries"]) != 100:
        raise ValueError("requires complete 100-page development panel before opening annotations")
    rows = list(panel["rows"].values())
    if {r["request_id"] for r in rows} != {e["row_hash"] for e in panel["entries"]}:
        raise ValueError("generation rows differ from the complete development manifest")
    verify_budget_matching(rows, panel["metadata"].get("cache_policies", POLICIES))
    evaluator = verify_evaluator(evaluator_root, manifest, cache_patch)
    root, scores = Path(output) / panel["context"], {}
    for cell in sorted({r["method"] for r in rows}):
        records = [r for r in rows if r["method"] == cell]
        path = root / cell / "official_scores.json"
        if path.exists():
            result = json.loads(path.read_text())
            if any(result.get(k) != v for k, v in {"run_context": panel["context"], "method": cell,
                                                  "manifest_sha256": manifest["manifest_sha256"], "split": "scorer_dev"}.items()):
                raise ValueError("existing score changed verified context")
        else:
            result = score_candidate(manifest, [{**r, "generated_budget": None} for r in records], path.parent, split="scorer_dev")
        scores[cell] = result
    quality, systems = quality_table(panel, scores), systems_table(rows)
    bundles, logical_scores = _logical_scores(rows, scores)
    diagnostics = {}
    for key, records in bundles.items():
        if policy(records[0]) == "protected_recency":
            h2o = f"h2o__{records[0]['comparison_group']}"
            if h2o in bundles:
                diagnostics[records[0]["comparison_group"]] = mechanism_diagnostics(logical_scores[key], logical_scores[h2o], records, bundles[h2o])
    result = {"format": "mlsys_cache_method_report_v3", "run_context": panel["context"],
              "manifest_sha256": manifest["manifest_sha256"], "source_sha256": file_sha256(__file__),
              "integrity_artifacts": panel["integrity_artifacts"], "evaluator": evaluator,
              "quality_table": quality, "systems_table": systems, "mechanism_diagnostics": diagnostics,
              "final_labels_closed": True, "same_physical_kv_capacity_verified": True,
              "random_seeds_do_not_increase_pages": True, "isolated_speedup_claim": False,
              "tables_markdown": render_tables(quality, systems)}
    write_new_text(root / "tables.md", result["tables_markdown"])
    write_new_json(root / "report.complete.json", result)
    return result
