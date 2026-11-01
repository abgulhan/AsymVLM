"""Two-table, development-only reporting for matched-total-KV cache methods.

The generation collector must verify every sealed request before this module
opens annotations.  All-token outputs are unrestricted references, not another
matched-budget method.  Natural-stop quality runs do not establish speedups.
"""

from collections import defaultdict
import argparse
import json
import os
from pathlib import Path
from statistics import mean
import tempfile

from learned_score.mlsys_eval.common import file_sha256, load_manifest, write_new_json
from learned_score.mlsys_eval.olmocr_score import paired_category_cluster_ci, score_candidate, verify_evaluator


COMPRESSED_METHODS = frozenset(("asymvlm", "h2o", "streamingllm"))


def _group(row):
    return row["model"], row["comparison_group"]


def verify_budget_matching(rows, expected_methods=COMPRESSED_METHODS):
    """Require identical per-request total KV allowance, not identical text slots."""
    expected_methods = frozenset(expected_methods)
    if not {"asymvlm", "h2o"}.issubset(expected_methods) or not expected_methods.issubset(COMPRESSED_METHODS):
        raise ValueError("unsupported cache comparison method set")
    paired = defaultdict(dict)
    for row in rows:
        method = row["cache_method"]
        if method == "none":
            continue
        if method not in COMPRESSED_METHODS:
            raise ValueError("unknown cache comparison method")
        key = (*_group(row), row["request_id"])
        if method in paired[key]:
            raise ValueError("duplicate request/cache-method comparison")
        values = row["kv_budget_slots"], row["kv_budget_bytes"]
        if any(type(value) is not int or value <= 0 for value in values):
            raise ValueError("total KV allowances must be positive integer counts")
        if row["metrics"]["cache"]["kv_capacity_bytes"] != row["kv_budget_bytes"]:
            raise ValueError("allocated K/V tensors do not match the declared allowance")
        paired[key][method] = values
    if not paired:
        raise ValueError("no matched-budget compressed methods")
    for methods in paired.values():
        if set(methods) != expected_methods or len(set(methods.values())) != 1:
            raise ValueError("cache methods do not share the same complete per-request total KV allowance")


def paired_difference(candidate, reference):
    point = 100 * (candidate["official_macro_score"] - reference["official_macro_score"])
    try:
        result = paired_category_cluster_ci(candidate["checks"], reference["checks"])
    except ValueError as error:
        # A singleton category genuinely cannot support this interval.  Do not
        # replace source-cluster uncertainty with a spuriously narrower page CI.
        if "fewer than two independent sources" not in str(error):
            raise
        return {"delta_pp": point, "ci_lower_pp": None, "ci_upper_pp": None,
                "ci_unavailable_reason": str(error)}
    return {"delta_pp": point, "ci_lower_pp": 100 * result["ci_lower"],
            "ci_upper_pp": 100 * result["ci_upper"], "bootstrap": result}


def quality_table(panel, scores):
    rows = list(panel["rows"].values())
    verify_budget_matching(rows, panel["metadata"]["cache_methods"])
    methods = defaultdict(list)
    for row in rows:
        methods[row["method"]].append(row)
    if set(methods) != set(scores):
        raise ValueError("official score cells differ from verified generation cells")
    references, asymvlm = {}, {}
    for cell, records in methods.items():
        first = records[0]
        if first["cache_method"] == "none":
            key = first["model"]
            if key in references:
                raise ValueError("ambiguous unrestricted reference")
            references[key] = cell
        elif first["cache_method"] == "asymvlm":
            key = _group(first)
            if key in asymvlm:
                raise ValueError("ambiguous matched-budget AsymVLM reference")
            asymvlm[key] = cell
    output = []
    for cell, records in sorted(methods.items()):
        first, score = records[0], scores[cell]
        requests = {row["request_id"] for row in records}
        if len(requests) != len(records):
            raise ValueError("quality table expects one output per request/method")
        if score["run_context"] != panel["context"]:
            raise ValueError("official scores belong to another verified run")
        reference_key = first["model"]
        reference = references.get(reference_key)
        if reference is None:
            raise ValueError("missing all-token unrestricted reference")
        if requests != {row["request_id"] for row in methods[reference]}:
            raise ValueError("quality comparison does not cover identical requests")
        comparisons = {}
        if cell != reference:
            comparisons["all_tokens_no_eviction"] = paired_difference(score, scores[reference])
        if first["cache_method"] in COMPRESSED_METHODS:
            same_budget = asymvlm[_group(first)]
            if cell != same_budget:
                comparisons["asymvlm_same_total_kv_budget"] = paired_difference(score, scores[same_budget])
        output.append({"model": first["model"], "dataset": "olmOCR-Bench", "stage": "development",
                       "method": cell, "method_label": first["method_label"],
                       "cache_method": first["cache_method"], "visual_retention": first["visual_retention"],
                       "budget_profile": first["budget_profile"],
                       "comparison_group": first["comparison_group"], "requests": len(records),
                       "score_percent": 100 * score["official_macro_score"],
                       "quality_metric": score["quality_metric"],
                       "category_scores_percent": {key: 100 * value for key, value in score["category_scores"].items()},
                       "outputs_at_cap": sum(row["generated_tokens"] == row["max_new_tokens"] for row in records),
                       "eviction_active_requests": sum(row["metrics"]["cache"]["compaction_events"] > 0 for row in records),
                       "eviction_active_fraction": mean(row["metrics"]["cache"]["compaction_events"] > 0 for row in records),
                       "paired_comparisons": comparisons, "run_context": panel["context"],
                       "control_status": "fresh_same_request_controls",
                       "unrestricted_reference": first["cache_method"] == "none"})
    return output


def systems_table(rows):
    """Describe observed quality-run costs without pooling nodes or software."""
    grouped = defaultdict(list)
    for row in rows:
        execution = row["execution_context"]
        if not execution.get("cluster") or not execution.get("hostname"):
            raise ValueError("systems rows require explicit platform/node provenance")
        key = (row["model"], row["method"], row["hardware"], execution["cluster"],
               execution["hostname"], execution.get("torch_version"), execution.get("cuda_version"),
               row["run_context"])
        grouped[key].append(row)
    output = []
    for key, records in sorted(grouped.items(), key=lambda item: str(item[0])):
        first = records[0]

        def measured(field, reduction, *, metrics=False):
            values = [(row["metrics"] if metrics else row).get(field) for row in records]
            if any(value is None for value in values):
                return None
            return reduction(values)

        output.append({"model": key[0], "method": key[1], "method_label": first["method_label"],
                       "cache_method": first["cache_method"], "hardware": key[2],
                       "cluster": key[3], "hostname": key[4], "torch_version": key[5], "cuda_version": key[6],
                       "run_context": key[7], "requests": len(records),
                       "visual_retention": first["visual_retention"], "budget_profile": first["budget_profile"],
                       "comparison_group": first["comparison_group"],
                       "mean_generated_tokens": mean(row["generated_tokens"] for row in records),
                       "eviction_active_requests": sum(row["metrics"]["cache"]["compaction_events"] > 0 for row in records),
                       "eviction_active_fraction": mean(row["metrics"]["cache"]["compaction_events"] > 0 for row in records),
                       "mean_kv_budget_bytes": measured("kv_budget_bytes", mean),
                       "minimum_kv_budget_bytes": measured("kv_budget_bytes", min),
                       "maximum_kv_budget_bytes": measured("kv_budget_bytes", max),
                       "mean_end_to_end_seconds": mean(row["end_to_end_seconds"] for row in records),
                       "mean_ttft_seconds": mean(row["ttft_seconds"] for row in records),
                       "maximum_gpu_allocated_bytes": max(row["peak_allocated_bytes"] for row in records),
                       "maximum_active_kv_bytes": max(row["metrics"]["cache"]["peak_active_kv_bytes"] for row in records),
                       "maximum_allocated_kv_bytes": max(row["metrics"]["cache"]["kv_capacity_bytes"] for row in records),
                       "maximum_auxiliary_bytes": max(row["metrics"]["cache"]["auxiliary_persistent_bytes"] for row in records),
                       "maximum_attention_scratch_bytes_estimate": max(row["metrics"]["cache"]["attention_statistics_scratch_peak_bytes_estimate"] for row in records),
                       "maximum_rolling_rotary_scratch_bytes_estimate": max(row["metrics"]["cache"]["rolling_rotary_scratch_peak_bytes_estimate"] for row in records),
                       "maximum_temporary_scratch_bytes_estimate": max(sum(row["metrics"]["cache"][field] for field in
                                                                         ("attention_statistics_scratch_peak_bytes_estimate", "rolling_rotary_scratch_peak_bytes_estimate")) for row in records),
                       "maximum_owned_cache_bytes": max(row["metrics"]["cache"]["allocated_cache_and_auxiliary_bytes"] for row in records),
                       "quality_run_only": True, "isolated_latency_measurement": False,
                       "serving_throughput": None,
                       "timing_caution": "natural-stop, parallel quality run; not a matched-output speedup experiment"})
    return output


def render_tables(quality, systems, *, unsupported_cache_methods=None):
    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    def difference(row, key):
        value = row["paired_comparisons"].get(key)
        if value is None:
            return "reference" if key == "all_tokens_no_eviction" or row["cache_method"] == "asymvlm" else "not budget-matched"
        point = f"{value['delta_pp']:+.2f}"
        if value["ci_lower_pp"] is None:
            return point + " (CI unavailable)"
        return point + f" [{value['ci_lower_pp']:+.2f}, {value['ci_upper_pp']:+.2f}]"

    lines = ["## Table 1. Document quality at matched total KV-cache budgets", "",
             "Scores are official olmOCR category-macro percentages on development pages. Differences are percentage points. "
             "Within each budget group, compressed methods receive exactly the same total KV allowance for each request, "
             "including image, prompt, and generated-text tokens. The allowance is the original full prefix plus the stated extra slots, "
             "with one incoming-token slot equally reserved for every method. No-eviction rows are unrestricted references.", "",
             "| Model | Cache method | Visual tokens kept | Budget profile | Pages | Score | Difference vs all tokens, no eviction [95% CI] | Difference vs AsymVLM, same KV allowance [95% CI] | Outputs at cap |",
             "|---|---|---:|---|---:|---:|---|---|---:|"]
    for row in quality:
        cells = [row["model"], row["method_label"], f"{row['visual_retention']:.0%}", row["budget_profile"], row["requests"],
                 f"{row['score_percent']:.2f}", difference(row, "all_tokens_no_eviction"),
                 difference(row, "asymvlm_same_total_kv_budget"), row["outputs_at_cap"]]
        lines.append("| " + " | ".join(map(safe, cells)) + " |")
    reasons = sorted({comparison["ci_unavailable_reason"] for row in quality
                      for comparison in row["paired_comparisons"].values() if comparison.get("ci_unavailable_reason")})
    if reasons:
        lines.extend(["", "Confidence-interval limitation: " + "; ".join(reasons) + ". Point differences are not significance claims."])
    lines.extend(["", "## Table 2. Observed memory and runtime", "",
                  "These are parallel, natural-stop quality-run measurements. Output lengths differ; latency is not evidence of an isolated speedup, "
                  "and serving throughput is not inferred. Platforms/nodes are never pooled. MiB = 2^20 bytes. "
                  "Total KV allowance varies with request prefix length and is shown as a mean; measured cache and GPU columns are maxima. "
                  "Auxiliary memory is reported separately from K/V tensors: resident allocations and an analytical estimate of transient attention-statistics or rolling-rotary scratch "
                  "are distinct. The CUDA peak-GPU measurement includes all allocations; equality of K/V tensor budgets does not imply equality of total memory.", "",
                  "| Model | Cache method | Vision kept / budget | Platform / node | Pages | Eviction activated | Mean output tokens | Mean KV allowance (MiB) | Peak active KV (MiB) | Allocated KV (MiB) | Auxiliary resident / transient estimate (MiB) | Peak GPU (MiB) | Mean latency / first token (s) |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for row in systems:
        def mib(key):
            value = row.get(key)
            return "unrestricted" if key == "mean_kv_budget_bytes" and value is None else "unavailable" if value is None else f"{value / 2**20:.2f}"
        cells = [row["model"], row["method_label"], f"{row['visual_retention']:.0%} / {row['budget_profile']}",
                 f"{row['cluster']} / {row['hostname']}", row["requests"],
                 f"{row['eviction_active_requests']}/{row['requests']} ({row['eviction_active_fraction']:.0%})",
                 f"{row['mean_generated_tokens']:.1f}",
                 mib("mean_kv_budget_bytes"), mib("maximum_active_kv_bytes"), mib("maximum_allocated_kv_bytes"),
                 mib("maximum_auxiliary_bytes") + " / " + mib("maximum_temporary_scratch_bytes_estimate"), mib("maximum_gpu_allocated_bytes"),
                 f"{row['mean_end_to_end_seconds']:.2f} / {row['mean_ttft_seconds']:.2f}"]
        lines.append("| " + " | ".join(map(safe, cells)) + " |")
    lines.extend(["", "H2O and StreamingLLM are explicitly labeled adaptations in this shared VLM inference runtime, not official end-to-end reproductions.",
                  "H2O attention statistics are recomputed after the shared SDPA attention operation to preserve numerical parity; these costs are included, "
                  "but are not timings of an optimized upstream H2O implementation.",
                  "All results are development-only; final-test labels remain unopened.", ""])
    if unsupported_cache_methods:
        lines.extend(["Excluded, unsupported methods (not zero scores or failed requests): " +
                      "; ".join(f"{key}: {value}" for key, value in sorted(unsupported_cache_methods.items())), ""])
    return "\n".join(lines)


def score_panel(panel, manifest, output, *, evaluator_root, cache_patch=None):
    if not panel.get("complete") or panel["metadata"]["split"] != "scorer_dev" or len(panel["entries"]) != 100:
        raise ValueError("official scoring requires the complete 100-page development panel")
    rows = list(panel["rows"].values())
    verify_budget_matching(rows, panel["metadata"]["cache_methods"])
    evaluator = verify_evaluator(evaluator_root, manifest, cache_patch)
    root = Path(output) / panel["context"]
    scores = {}
    for method in sorted({row["method"] for row in rows}):
        records = [row for row in rows if row["method"] == method]
        target = root / method / "official_scores.json"
        if target.exists():
            result = json.loads(target.read_text())
            if (result.get("run_context") != panel["context"] or result.get("method") != method
                    or result.get("manifest_sha256") != manifest["manifest_sha256"] or result.get("split") != "scorer_dev"):
                raise ValueError("existing official score changed verified context")
        else:
            # The old official-export wrapper requires a constant generated
            # allowance. Here the actual allowance deliberately varies with
            # visual-prefix length. That field is unused by the evaluator:
            # normalize ONLY this scoring projection, preserving sealed rows.
            scoring_records = [{**row, "generated_budget": None} for row in records]
            result = score_candidate(manifest, scoring_records, target.parent, split="scorer_dev")
        scores[method] = result
    quality, systems = quality_table(panel, scores), systems_table(rows)
    result = {"format": "mlsys_matched_cache_method_report_v1", "run_context": panel["context"],
              "manifest_sha256": manifest["manifest_sha256"], "source_sha256": file_sha256(__file__),
              "integrity_artifacts": panel["integrity_artifacts"], "evaluator": evaluator,
              "quality_table": quality, "systems_table": systems,
              "unsupported_cache_methods": panel["metadata"].get("unsupported_cache_methods", {}),
              "tables_markdown": render_tables(quality, systems, unsupported_cache_methods=panel["metadata"].get("unsupported_cache_methods")),
              "final_labels_closed": True, "historical_controls_used": False,
              "scoring_projection": "generated_budget normalized to None only for legacy prediction exporter; actual total budgets verified from unchanged sealed rows",
              "same_total_kv_allowance_verified": True, "isolated_speedup_claim": False}
    write_new_text(root / "tables.md", result["tables_markdown"])
    write_new_json(root / "report.complete.json", result)
    return result


def write_new_text(path, text):
    """Atomic immutable text publication with the same safety as JSON reports."""
    path = Path(path)
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--evaluator-root", type=Path, required=True)
    parser.add_argument("--cache-patch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from learned_score.mlsys_sprint.cache_methods import collect_panel
    panel = collect_panel(args.protocol, args.run_dir, require_complete=True)
    protocol = json.loads(args.protocol.read_text())
    base = json.loads(Path(protocol["base_protocol"]).read_text())
    manifest = load_manifest(base["manifest"])
    result = score_panel(panel, manifest, args.output, evaluator_root=args.evaluator_root, cache_patch=args.cache_patch)
    print(json.dumps({"output": str(args.output / panel["context"] / "report.complete.json"),
                      "quality_rows": len(result["quality_table"]), "systems_rows": len(result["systems_table"])}))


if __name__ == "__main__":
    main()
