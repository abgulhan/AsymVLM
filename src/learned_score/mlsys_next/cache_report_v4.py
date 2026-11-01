"""Exploratory v4 cache coverage reporting on one immutable development subset.

This module never opens final annotations.  Existing v3 Boolean checks are
filtered to the exact v4 requests before comparison.  Random seeds are averaged
within checks, never treated as additional pages.  A relative quality floor is
a descriptive engineering constraint, not a significance or equivalence claim.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from learned_score.mlsys_eval.common import (
    canonical_hash, file_sha256, load_manifest, seal_manifest, write_new_json,
)
from learned_score.mlsys_eval.olmocr_score import score_candidate, verify_evaluator
from .cache_method_report import write_new_text
from .cache_method_report_v3 import aggregate_seed_scores, macro, paired_difference


RANDOM_SEEDS = (42, 43, 44)
PATTERNS = ("protected_grid", "protected_weyl")
FORMAT = "mlsys_cache_coverage_report_v4"


def _rows(panel):
    values = panel["rows"]
    return list(values.values()) if isinstance(values, dict) else list(values)


def _entries(protocol):
    entries = protocol.get("development_entries", protocol.get("entries"))
    if not isinstance(entries, list) or not entries:
        raise ValueError("v4 protocol requires explicit development entries")
    return entries


def cell_key(cell):
    """Logical identity keeps physical budget, cadence and recent allocation."""
    return "__".join(str(cell.get(k, default)) for k, default in (
        ("policy", None), ("budget", 512), ("staging", 128),
        ("recent_mode", "eight"), ("phase", 0)))


def _is_random(policy):
    return "random" in policy


def _check_requests(score, entries):
    identities = {entry["row_hash"]: entry["group_id"] for entry in entries}
    checks = score["checks"]
    if {check["request_id"] for check in checks} != set(identities):
        raise ValueError("score is not the exact requested development subset")
    seen = set()
    for check in checks:
        key = check["request_id"], check["category"], check["test_id"]
        if key in seen or check["group_id"] != identities[check["request_id"]]:
            raise ValueError("duplicate check or changed source identity")
        seen.add(key)


def align_score(score, entries, context, manifest_sha256, *, subset=False):
    """Explicitly recontextualize verified same-page checks, retaining provenance."""
    wanted = {entry["row_hash"] for entry in entries}
    if score.get("split") != "scorer_dev":
        raise ValueError("only already-scored development checks may be reused")
    checks = [dict(check) for check in score["checks"]
              if not subset or check["request_id"] in wanted]
    candidate = {**score, "checks": checks}
    _check_requests(candidate, entries)
    original = {key: score.get(key) for key in ("run_context", "manifest_sha256", "method")}
    for check in checks:
        if check.get("split") != "scorer_dev":
            raise ValueError("final checks cannot enter v4 development")
        if any(check.get(key) != score.get(key) for key in ("model", "split", "manifest_sha256", "run_context")):
            raise ValueError("check provenance differs from its verified score")
        check.update(run_context=context, manifest_sha256=manifest_sha256)
    return {**candidate, "run_context": context, "manifest_sha256": manifest_sha256,
            "source_context": original, **macro(checks)}


def development_subset(protocol, manifest):
    if protocol.get("split") != "scorer_dev" or protocol.get("final_labels_opened", False):
        raise ValueError("v4 report is development-only")
    entries = _entries(protocol)
    parent = {entry["row_hash"]: entry for entry in manifest["splits"]["scorer_dev"]}
    if len(entries) != len({entry["row_hash"] for entry in entries}):
        raise ValueError("duplicate development page")
    if len(entries) != len({entry["group_id"] for entry in entries}):
        raise ValueError("pilot must use distinct source documents")
    if any(parent.get(entry["row_hash"]) != entry for entry in entries):
        raise ValueError("v4 entries changed their parent development identity")
    declared = protocol.get("development_request_ids")
    if declared is not None and set(declared) != set(parent_entry["row_hash"] for parent_entry in entries):
        raise ValueError("declared v4 request IDs disagree")
    return seal_manifest({"format": manifest["format"], "dataset": manifest["dataset"],
                          "splits": {"scorer_dev": entries},
                          "metadata": {**manifest["metadata"],
                                       "parent_manifest_sha256": manifest["manifest_sha256"],
                                       "evaluation_stage": "exploratory_development",
                                       "final_status": "not_in_this_manifest",
                                       "counts": {"scorer_dev": len(entries)},
                                       "category_counts": {},
                                       "previously_examined_development": True}})


def validate_panel(panel, protocol):
    if not panel.get("complete"):
        raise ValueError("complete sealed page matrix required before annotation access")
    entries = _entries(protocol)
    wanted = {entry["row_hash"]: entry["group_id"] for entry in entries}
    if {entry["row_hash"] for entry in panel["entries"]} != set(wanted):
        raise ValueError("panel and protocol request assignments disagree")
    cells = panel.get("cells", protocol["cells"])
    # Keys are excluded because some producers serialize the computed property.
    normalized = lambda cell: {k: v for k, v in cell.items() if k != "key"}
    declared = {canonical_hash(normalized(cell)) for cell in cells}
    if declared != {canonical_hash(normalized(cell)) for cell in protocol["cells"]}:
        raise ValueError("panel cell matrix differs from frozen protocol")
    seen, capacities = set(), defaultdict(set)
    for row in _rows(panel):
        cell = normalized(row["cell"])
        identity = row["request_id"], canonical_hash(cell)
        if identity in seen or identity[0] not in wanted or identity[1] not in declared:
            raise ValueError("duplicate, extra, or undeclared physical cell")
        seen.add(identity)
        if row.get("group_id") != wanted[identity[0]] or row.get("split") != "scorer_dev":
            raise ValueError("physical row changed development source identity")
        if row.get("run_context") != panel["context"]:
            raise ValueError("physical row is not bound to the sealed panel context")
        if row.get("visual_retention") != 1.0 or row.get("max_new_tokens") != protocol["max_new_tokens"]:
            raise ValueError("generation configuration differs from frozen full-vision pilot")
        if row.get("cache_policy") != cell["policy"]:
            raise ValueError("physical policy differs from cell")
        slots = row["input_prefix_tokens"] + cell["budget"] + 1
        cache = row["metrics"]["cache"]
        if (row["kv_budget_slots"] != slots or cache["capacity_tokens_per_layer"] != slots
                or row["kv_retained_slots"] != slots - cell["staging"]
                or cache["kv_capacity_bytes"] != row["kv_budget_bytes"]
                or not 0 <= cache["active_tokens"] <= slots
                or not 0 <= cache["peak_active_tokens"] <= slots):
            raise ValueError("physical KV budget or staging contract failed")
        capacities[identity[0], cell["budget"]].add((slots, row["kv_budget_bytes"]))
    if len(seen) != len(wanted) * len(declared) or any(len(values) != 1 for values in capacities.values()):
        raise ValueError("incomplete or unequal-budget physical matrix")


def _pinned_json(path, pins):
    path = Path(path)
    expected = pins.get(str(path), pins.get(str(path.resolve())))
    if not expected or file_sha256(path) != expected:
        raise ValueError(f"unbound or changed reference artifact: {path}")
    return json.loads(path.read_text())


def _reference_report_path(protocol):
    if protocol.get("reference_report"):
        return Path(protocol["reference_report"])
    candidates = []
    for name in protocol.get("input_hashes", {}):
        path = Path(name)
        if path.name == "report.complete.json" and "/evaluation/qwen25/" in str(path):
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError("exactly one pinned Qwen v3 reference report is required")
    return candidates[0]


def reuse_v3_references(protocol, entries, context, manifest_sha256):
    """Reuse only sealed physical rows and pinned, already-scored Boolean checks."""
    pins = protocol["input_hashes"]
    report_path = _reference_report_path(protocol)
    report = _pinned_json(report_path, pins)
    artifacts = {str(report_path): file_sha256(report_path)}
    wanted = {entry["row_hash"] for entry in entries}
    scores, records, descriptors = {}, {}, {}
    # The original report binds every physical row used to produce these checks.
    for name, expected in report["integrity_artifacts"].items():
        path = Path(name)
        if "/attempts/" not in name or path.suffix != ".json":
            continue
        row = json.loads(path.read_text())
        if row.get("request_id") not in wanted:
            continue
        if file_sha256(path) != expected:
            raise ValueError("v3 physical reference changed")
        artifacts[str(path)] = expected
        records.setdefault(row["method"], []).append(row)
    for method, rows in records.items():
        if len(rows) != len(wanted) or {row["request_id"] for row in rows} != wanted:
            raise ValueError("v3 reference does not contain exactly the same pilot pages")
        path = report_path.parent / method / "official_scores.json"
        source = _pinned_json(path, pins)
        if (source["run_context"] != report["run_context"] or source["method"] != method
                or source["manifest_sha256"] != report["manifest_sha256"]
                or source["model"] != rows[0]["model"]):
            raise ValueError("v3 score is not bound to its original report")
        artifacts[str(path)] = file_sha256(path)
        scores[method] = align_score(source, entries, context, manifest_sha256, subset=True)
        row = rows[0]
        descriptors[method] = {"policy": row["cache_policy"], "budget": None if row["cache_policy"] in ("none", "fullref") else 512,
                               "staging": row.get("staging_slots"), "recent_mode": "original_v3",
                               "seed": row.get("random_seed"), "phase": 0, "reference_v3": True}
    if "full_v100_off" not in scores:
        raise ValueError("same-page full-cache reference is missing")
    return scores, records, descriptors, artifacts


def logical_scores(scores, descriptors):
    bundles = defaultdict(dict)
    for method, score in scores.items():
        cell = descriptors[method]
        key = "full_v100_off" if cell["policy"] in ("none", "fullref") else cell_key(cell)
        if cell.get("reference_v3") and key != "full_v100_off":
            key = "v3__" + key
        seed = cell.get("seed")
        if seed in bundles[key]:
            raise ValueError("duplicate physical seed or deterministic cell")
        bundles[key][seed] = (score, cell, method)
    result, cells, physical = {}, {}, {}
    for key, values in bundles.items():
        first = next(iter(values.values()))
        if _is_random(first[1]["policy"]):
            result[key] = aggregate_seed_scores({seed: value[0] for seed, value in values.items()}, method=key)
        elif set(values) == {None}:
            result[key] = first[0]
        else:
            raise ValueError("deterministic policy unexpectedly has seeds")
        cells[key] = {k: v for k, v in first[1].items() if k != "seed"}
        physical[key] = [value[2] for value in values.values()]
    return result, cells, physical


def validate_reference_pairing(reference_records, new_rows):
    current = {}
    fields = ("model", "group_id", "image_sha256", "input_prefix_tokens", "input_visual_tokens",
              "selected_visual_indices_sha256", "max_new_tokens", "visual_retention")
    for row in new_rows:
        values = tuple(row.get(field) for field in fields)
        if any(value is None for value in values):
            raise ValueError("missing physical pairing provenance")
        if row["request_id"] in current and current[row["request_id"]] != values:
            raise ValueError("new physical cells changed preprocessing")
        current[row["request_id"]] = values
    for rows in reference_records.values():
        for row in rows:
            if tuple(row.get(field) for field in fields) != current.get(row["request_id"]):
                raise ValueError("v3 reference changed model, image, tokens, or generation configuration")


def _memory(rows):
    # Average seeds inside each request before averaging pages.
    by_request = defaultdict(list)
    for row in rows:
        by_request[row["request_id"]].append(row)
    def average(getter):
        values = []
        for replicas in by_request.values():
            items = [getter(row) for row in replicas]
            if any(value is None for value in items):
                return None
            values.append(mean(items))
        return mean(values) if values else None
    return {"mean_allocated_kv_bytes": average(lambda row: row["metrics"]["cache"].get("kv_capacity_bytes")),
            "mean_peak_gpu_allocated_bytes": average(lambda row: row.get("peak_allocated_bytes")),
            "mean_peak_active_kv_bytes": average(lambda row: row["metrics"]["cache"].get("peak_active_kv_bytes")),
            "mean_generated_tokens": average(lambda row: row["generated_tokens"]),
            "mean_output_cap_fraction": average(lambda row: int(row["generated_tokens"] == row["max_new_tokens"])),
            "hardware": sorted({str(row.get("hardware")) for row in rows}),
            "timing_isolated": False, "isolated_speedup_claim": False}


def select_configuration(stage, quality, *, previous_selection=None, floor=.95):
    candidates = [row for row in quality if row["policy"] in PATTERNS and not row.get("reference_v3")]
    if not candidates:
        raise ValueError("no deterministic candidate available for selection")
    if stage in ("stage1", "stage2"):
        candidates = [row for row in candidates if row["budget"] == 512]
        if stage == "stage1":
            if {row["policy"] for row in candidates} != set(PATTERNS):
                raise ValueError("stage1 requires both deterministic patterns")
            candidates = [row for row in candidates if row["recent_mode"] == "eight"]
        else:
            if previous_selection is None or previous_selection["policy"] not in PATTERNS:
                raise ValueError("stage2 requires immutable stage1 pattern nomination")
            candidates = [row for row in candidates if row["policy"] == previous_selection["policy"]]
            if {row["recent_mode"] for row in candidates} != {"eight", "two_thirds"}:
                raise ValueError("stage2 requires both fixed recent allocations")
        chosen = min(candidates, key=lambda row: (-row["score_percent"],
                                                  PATTERNS.index(row["policy"]),
                                                  row["recent_mode"] != "eight"))
        rule = "highest development official category macro; exact ties grid, then eight recent tokens"
    elif stage == "stage3":
        if previous_selection is None:
            raise ValueError("stage3 requires a frozen 512-token configuration")
        candidates = [row for row in candidates if (row["policy"], row["recent_mode"], row.get("phase", 0)) ==
                      (previous_selection["policy"], previous_selection["recent_mode"], previous_selection.get("phase", 0))]
        if {row["budget"] for row in candidates} != {128, 256, 512}:
            raise ValueError("stage3 must compare all three predeclared budgets")
        eligible = [row for row in candidates if row["relative_quality_to_full"] >= floor]
        chosen = min(eligible, key=lambda row: (row["budget"], -row["score_percent"])) if eligible else None
        rule = "smallest measured budget with development score at least 95% of the same-page full-cache score"
    else:
        raise ValueError("unknown v4 report stage")
    if chosen is None:
        return {"policy": previous_selection["policy"], "recent_mode": previous_selection["recent_mode"],
                "phase": previous_selection.get("phase", 0), "budget": None,
                "deployment_eligible": False, "relative_quality_floor": floor,
                "selection_rule": rule, "development_only": True, "statistical_claim": False}
    return {key: chosen[key] for key in ("policy", "recent_mode", "phase", "budget")} | {
        "method": chosen["method"], "score_percent": chosen["score_percent"],
        "relative_quality_to_full": chosen["relative_quality_to_full"],
        "deployment_eligible": chosen["relative_quality_to_full"] >= floor,
        "relative_quality_floor": floor, "selection_rule": rule,
        "development_only": True, "statistical_claim": False}


def method_label(cell):
    labels = {"none": "All tokens, no eviction", "fullref": "All tokens, no eviction",
              "protected_recency": "AsymVLM: protected prefix + recent text",
              "protected_priority_random": "Protected prefix + persistent random history (3-seed mean)",
              "protected_random": "Original protected random selection (3-seed mean)",
              "protected_grid": "Protected prefix + nested-grid history",
              "protected_weyl": "Protected prefix + nonperiodic history coverage",
              "protected_h2o": "Protected prefix + H2O text selection",
              "protected_rkv": "Protected prefix + R-KV text selection",
              "h2o": "H2O adaptation", "h2o_recent_adjusted": "H2O adaptation, adjusted recent allocation",
              "rkv": "R-KV adaptation"}
    label = labels.get(cell["policy"], cell["policy"])
    if cell.get("reference_v3") and cell["policy"] not in ("none", "fullref"):
        label += " (original v3)"
    return label


def recent_allocation_label(cell):
    if cell["policy"] in ("none", "fullref"):
        return "not applicable"
    if cell.get("reference_v3"):
        return "original method allocation"
    available = cell["budget"] + 1 - cell["staging"]
    if cell["policy"] == "protected_recency":
        return f"all {available} retained text tokens"
    if cell["policy"] in (*PATTERNS, "protected_priority_random"):
        count = 8 if cell["recent_mode"] == "eight" else 2 * available // 3
        return f"latest {count} text tokens"
    return "method-specific allocation"


def render_tables(quality, selection):
    lines = ["# Exploratory cache-history coverage pilot", "",
             "All rows use the same previously examined development pages; no final annotations are accessed.", "",
             "| Method | Budget | Recent allocation | Score (%) | Versus full (pp) | Paired 95% CI (pp) | Relative full quality | Meets 95% floor |",
             "|---|---:|---|---:|---:|---|---:|---|"]
    for row in quality:
        comparison = row["versus_full"]
        interval = ("unavailable" if comparison["ci_lower_pp"] is None else
                    f"[{comparison['ci_lower_pp']:+.3f}, {comparison['ci_upper_pp']:+.3f}]")
        lines.append(f"| {row['method_label']} | {row['budget']} | {row['recent_allocation']} | {row['score_percent']:.3f} | "
                     f"{row['versus_full']['delta_pp']:+.3f} | {interval} | {row['relative_quality_to_full']:.4f} | {row['quality_floor_pass']} |")
    lines += ["", "| Method | Mean allocated KV (MiB) | KV reduction versus full (%) | Mean peak GPU allocation (MiB) | Mean capped fraction |",
              "|---|---:|---:|---:|---:|"]
    def mib(value):
        return "unavailable" if value is None else f"{value / 2**20:.3f}"
    for row in quality:
        memory = row["memory"]
        cap = memory.get("mean_output_cap_fraction")
        savings = memory.get("allocated_kv_reduction_vs_full_percent")
        lines.append(f"| {row['method_label']} (budget {row['budget']}, {row['recent_allocation']}) | {mib(memory.get('mean_allocated_kv_bytes'))} | "
                     f"{'unavailable' if savings is None else f'{savings:.3f}'} | "
                     f"{mib(memory.get('mean_peak_gpu_allocated_bytes'))} | {'unavailable' if cap is None else f'{cap:.3f}'} |")
    lines += ["", "Random seeds are averaged within each page/check before paired source-cluster resampling.",
              "Memory measurements come from concurrent natural-stop quality runs; they do not establish isolated speedups.",
              "Intervals are unavailable when any category has fewer than two independent source documents; small changes are not established wins.",
              "The 95% floor means 0.95 times full-cache score, not a 95% confidence guarantee.",
              "Protected prefix means image and original prompt tokens are never evicted. Budgets are extra slots beyond the original prefix; staging is inside that maximum.",
              "Eight means the latest eight generated tokens; two_thirds reserves two-thirds of the post-compaction generated-text cache for recent tokens. Recent-only policies retain only the newest text.",
              "Original v3 random redraws per-head scores each compaction; v4 persistent random uses fixed token priorities and a shared mask. They are different controls.",
              "Selection: " + json.dumps(selection, sort_keys=True), ""]
    return "\n".join(lines)


def build_report(panel, protocol, manifest, output, *, evaluator_root, cache_patch=None,
                 repetitions=10000, previous_report=None):
    validate_panel(panel, protocol)
    subset = development_subset(protocol, manifest)
    entries, context = _entries(protocol), panel["context"]
    root = Path(output) / context
    refs, records, descriptors, artifacts = reuse_v3_references(protocol, entries, context, subset["manifest_sha256"])
    scores = dict(refs)
    rows = _rows(panel)
    validate_reference_pairing(records, rows)
    evaluator = verify_evaluator(evaluator_root, subset, cache_patch)
    for method in sorted({row["method"] for row in rows}):
        physical = [row for row in rows if row["method"] == method]
        path = root / method / "official_scores.json"
        if path.exists():
            score = json.loads(path.read_text())
            if any(score.get(key) != value for key, value in {
                "run_context": context, "manifest_sha256": subset["manifest_sha256"],
                "method": method, "split": "scorer_dev"}.items()):
                raise ValueError("existing v4 score changed sealed context")
        else:
            score = score_candidate(subset, [{**row, "generated_budget": None} for row in physical],
                                    path.parent, split="scorer_dev")
        _check_requests(score, entries)
        scores[method], records[method], descriptors[method] = score, physical, physical[0]["cell"]
        artifacts[str(path)] = file_sha256(path)
    logical, cells, physical_methods = logical_scores(scores, descriptors)
    memory = {key: _memory([row for name in methods for row in records[name]])
              for key, methods in physical_methods.items()}
    previous_selection = None
    if previous_report is not None:
        if previous_report.get("format") != FORMAT or previous_report["request_ids"] != sorted(entry["row_hash"] for entry in entries):
            raise ValueError("previous report belongs to a different pilot")
        previous_selection = previous_report["selection"]
        if protocol.get("stage_selection") != previous_selection:
            raise ValueError("previous report does not match frozen stage selection")
        for key, old in previous_report["logical_scores"].items():
            if key in logical:
                continue
            logical[key] = align_score(old, entries, context, subset["manifest_sha256"])
            cells[key] = previous_report["cells"][key]
            memory[key] = previous_report["memory"][key]
    full = logical["full_v100_off"]
    full_score = full["official_macro_score"]
    if full_score <= 0:
        raise ValueError("relative full-cache quality is undefined at zero baseline score")
    quality = []
    for key, score in sorted(logical.items()):
        cell = cells[key]
        full_bytes, measured_bytes = memory["full_v100_off"].get("mean_allocated_kv_bytes"), memory[key].get("mean_allocated_kv_bytes")
        memory[key] = {**memory[key], "allocated_kv_reduction_vs_full_percent":
                       None if not full_bytes or measured_bytes is None else 100 * (1 - measured_bytes / full_bytes)}
        quality.append({"method": key, "method_label": method_label(cell), "recent_allocation": recent_allocation_label(cell),
                        **cell, "phase": cell.get("phase", 0),
                        "requests": len(entries), "source_documents": len(entries),
                        "score_percent": 100 * score["official_macro_score"],
                        "category_scores_percent": {k: 100 * value for k, value in score["category_scores"].items()},
                        "relative_quality_to_full": score["official_macro_score"] / full_score,
                        "quality_floor_pass": score["official_macro_score"] >= protocol.get("relative_quality_floor", .95) * full_score,
                        "versus_full": paired_difference(score, full, repetitions=repetitions),
                        "paired_comparisons": {}, "memory": memory[key]})
    for row in quality:
        key, cell = row["method"], cells[row["method"]]
        for other, other_cell in cells.items():
            if other == key or other_cell.get("budget") != cell.get("budget"):
                continue
            if _is_random(other_cell["policy"]) or other_cell["policy"] in ("protected_recency", "h2o_recent_adjusted", "rkv"):
                row["paired_comparisons"][other] = paired_difference(logical[key], logical[other], repetitions=repetitions)
    selection = select_configuration(protocol["stage"], quality, previous_selection=previous_selection,
                                     floor=protocol.get("relative_quality_floor", .95))
    report = {"format": FORMAT, "stage": protocol["stage"], "run_context": context,
              "protocol_sha256": protocol["sha256"], "source_sha256": file_sha256(__file__),
              "manifest_sha256": subset["manifest_sha256"], "request_ids": sorted(entry["row_hash"] for entry in entries),
              "requests": len(entries), "source_documents": len(entries), "evaluator": evaluator,
              "logical_scores": logical, "cells": cells, "memory": memory, "quality_table": quality,
              "selection": selection, "final_labels_opened": False,
              "previously_examined_development": True, "isolated_speedup_claim": False,
              "random_seeds_are_not_additional_pages": True,
              "integrity_artifacts": {**panel["integrity_artifacts"], **artifacts},
              "tables_markdown": render_tables(quality, selection)}
    write_new_text(root / "tables.md", report["tables_markdown"])
    write_new_json(root / "report.complete.json", report)
    return report


def report_stage(protocol_path, stage_root=None, stage=None, *, repetitions=10000,
                 evaluator_root=None, cache_patch=None):
    from learned_score.mlsys_sprint.cache_methods_v4 import collect_panel
    protocol_path = Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    if stage is not None and stage != protocol["stage"]:
        raise ValueError("requested report stage differs from immutable protocol")
    stage_root = Path(stage_root) if stage_root is not None else protocol_path.parent
    panel = collect_panel(protocol_path, require_complete=True, model="qwen25")
    manifest = load_manifest(protocol["manifest"])
    base = json.loads(Path(protocol["base_protocol"]).read_text())
    previous = None
    if protocol.get("previous_report"):
        path = Path(protocol["previous_report"])
        if file_sha256(path) != protocol["previous_report_sha256"]:
            raise ValueError("previous immutable stage report changed")
        previous = json.loads(path.read_text())
    if protocol["stage"] != "stage1" and previous is None:
        raise ValueError("later stages require an immutable previous report")
    evaluator_root = evaluator_root or protocol.get("evaluator_root", base.get("evaluator_root"))
    cache_patch = cache_patch or protocol.get("cache_patch", base.get("cache_patch"))
    if evaluator_root is None:
        evaluator_root = base.get("olmocr_evaluator_root")
    if evaluator_root is None:
        raise ValueError("pinned official evaluator root missing")
    report = build_report(panel, protocol, manifest, stage_root / "evaluation", evaluator_root=evaluator_root,
                          cache_patch=cache_patch, repetitions=repetitions, previous_report=previous)
    report_path = (stage_root / "evaluation" / panel["context"] / "report.complete.json").resolve()
    selection = {"selection": report["selection"], "passed": True, "final_labels_opened": False,
                 "source_report": str(report_path), "source_report_sha256": file_sha256(report_path),
                 "protocol_sha256": protocol["sha256"]}
    selection["sha256"] = canonical_hash(selection)
    write_new_json(stage_root / "selection.json", selection)
    write_new_json(stage_root / "evaluation" / "latest.json", {
        "report": str(report_path), "report_sha256": file_sha256(report_path),
        "selection": str((stage_root / "selection.json").resolve()), "complete": True,
        "final_labels_opened": False})
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--stage-root", type=Path)
    parser.add_argument("--repetitions", type=int, default=10000)
    parser.add_argument("--evaluator-root", type=Path)
    parser.add_argument("--cache-patch", type=Path)
    args = parser.parse_args(argv)
    report = report_stage(args.protocol, args.stage_root, repetitions=args.repetitions,
                          evaluator_root=args.evaluator_root, cache_patch=args.cache_patch)
    print(json.dumps({"stage": report["stage"], "requests": report["requests"], "selection": report["selection"]}, sort_keys=True))


if __name__ == "__main__":
    main()
