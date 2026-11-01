"""Pinned official olmOCR scoring and paired category/source-cluster intervals.

Final annotations require an explicit frozen-policy SHA. This module never
uses OCR checks as VQA accuracy and never changes the official scoring logic.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

from .common import canonical_hash, file_sha256, load_manifest, write_new_json
from .olmocr import export_predictions, prepare_official_subset


def official_macro(checks):
    categories = defaultdict(list)
    for check in checks:
        value = check["passed"]
        if not isinstance(value, bool):
            raise ValueError("official checks must contain the evaluator's majority-repeat Boolean result")
        categories[check["category"]].append(int(value))
    if not categories:
        raise ValueError("no official checks")
    scores = {category: sum(values) / len(values) for category, values in sorted(categories.items())}
    return {"official_macro_score": sum(scores.values()) / len(scores), "category_scores": scores,
            "category_check_counts": {category: len(values) for category, values in categories.items()},
            "quality_metric": "olmocr_official_category_macro_including_baseline"}


def paired_category_cluster_ci(candidate, baseline, *, repetitions=10000, seed=2027):
    """Preserve source clusters, stratified by their category-membership signature.

    A source that contributes to several categories is resampled once, jointly;
    category scores are then equally weighted as in the official macro metric.
    This avoids both page-level pseudoreplication and metric reweighting.
    """
    import numpy as np
    if repetitions < 100:
        raise ValueError("at least 100 bootstrap replicates are required")
    def index(rows):
        result = {}
        for row in rows:
            key = row["request_id"], row["category"], row["test_id"]
            if key in result:
                raise ValueError("duplicate official check")
            if not isinstance(row.get("passed"), bool):
                raise ValueError("missing official Boolean check result")
            result[key] = row
        return result
    left, right = index(candidate), index(baseline)
    if not left or left.keys() != right.keys():
        raise ValueError("candidate and baseline must contain exactly the same nonempty checks")
    contexts = {(row.get("model"), row.get("split"), row.get("manifest_sha256"), row.get("run_context"))
                for row in list(left.values()) + list(right.values())}
    if len(contexts) != 1:
        raise ValueError("do not compare different models, splits, manifests, or verified run contexts")
    grouped = defaultdict(lambda: defaultdict(list))
    for key, row in left.items():
        other = right[key]
        if row["group_id"] != other["group_id"]:
            raise ValueError("paired check changed source group")
        grouped[row["group_id"]][row["category"]].append(int(row["passed"]) - int(other["passed"]))
    categories = sorted({category for value in grouped.values() for category in value})
    counts_by_category = {category: sum(category in value for value in grouped.values()) for category in categories}
    if any(value < 2 for value in counts_by_category.values()):
        raise ValueError("a category has fewer than two independent sources; its uncertainty is not estimable")
    strata = defaultdict(list)
    for group, value in grouped.items():
        strata[tuple(sorted(value))].append(group)
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repetitions):
        totals, counts = Counter(), Counter()
        for signature, groups in sorted(strata.items()):
            for position in rng.integers(0, len(groups), size=len(groups)):
                for category, differences in grouped[groups[int(position)]].items():
                    totals[category] += sum(differences)
                    counts[category] += len(differences)
        values.append(sum(totals[category] / counts[category] for category in categories) / len(categories))
    low, high = np.quantile(values, [.025, .975])
    point = official_macro(candidate)["official_macro_score"] - official_macro(baseline)["official_macro_score"]
    return {"delta": point, "ci_lower": float(low), "ci_upper": float(high), "confidence": .95,
            "bootstrap_repetitions": repetitions, "bootstrap_seed": seed, "n_source_groups": len(grouped),
            "groups_by_category": counts_by_category, "resampling": "paired source-cluster bootstrap stratified by category-membership signature",
            "stratum_group_counts": {" + ".join(signature): len(groups) for signature, groups in sorted(strata.items())},
            "singleton_strata_warning": any(len(groups) == 1 for groups in strata.values()),
            "estimand": "official equal-category macro pass-rate difference"}


def verify_evaluator(evaluator_root, payload, cache_patch=None):
    root = Path(evaluator_root).resolve()
    locks = sorted(root.glob("acquisition*.json"))
    inventories = [json.loads(path.read_text()) for path in locks]
    inventories = [item for item in inventories if item["evaluator_revision"] == payload["metadata"]["evaluator_revision"]]
    if not inventories:
        raise ValueError("no matching pinned evaluator acquisition")
    inventory = max(inventories, key=lambda item: len(item.get("files", [])))
    patch = json.loads(Path(cache_patch).read_text()) if cache_patch else None
    for item in inventory["files"]:
        expected = item["sha256"]
        if patch and item["path"] == "olmocr/bench/katex/render.py":
            if patch["source_before_sha256"] != expected:
                raise ValueError("cache patch does not bind to pinned evaluator source")
            expected = patch["source_after_sha256"]
        if file_sha256(root / item["path"]) != expected:
            raise ValueError(f"official evaluator source changed: {item['path']}")
    sys.path.insert(0, str(root))
    return {"evaluator_revision": inventory["evaluator_revision"], "cache_patch": patch}


def score_candidate(payload, records, output_dir, *, split, policy_frozen_sha256=None):
    if split == "final_test" and not re.fullmatch(r"[0-9a-f]{64}", policy_frozen_sha256 or ""):
        raise ValueError("final annotation scoring is closed until an explicit frozen-policy SHA is supplied")
    records = list(records)
    contexts = {(row["model"], row["method"], row.get("run_context")) for row in records}
    if len(contexts) != 1 or not next(iter(contexts))[2]:
        raise ValueError("score one model/method/verified run_context at a time; use completeness-checked collector records")
    model, method, run_context = next(iter(contexts))
    if len({(row.get("visual_retention"), row.get("generated_budget"), row.get("max_new_tokens")) for row in records}) != 1:
        raise ValueError("score one fixed action/output-cap configuration at a time")
    root = Path(output_dir)
    # Verify complete predictions BEFORE opening any evaluation annotations.
    export_predictions(payload, records, root / "candidate", split=split)
    prepare_official_subset(payload, root, split=split)
    from olmocr.bench.benchmark import evaluate_candidate
    from olmocr.bench.tests import BaselineTest, load_tests
    tests, categories = [], {}
    for filename in sorted(root.glob("*.jsonl")):
        loaded = load_tests(str(filename))
        for test in loaded:
            if test.id in categories:
                raise ValueError("official annotation test IDs are not unique")
            categories[test.id] = filename.stem
        tests.extend(loaded)
    rows = {(row["pdf"], row["page"]): row for row in payload["splits"][split]}
    pdfs = sorted({row["pdf"] for row in payload["splits"][split]})
    for pdf in pdfs:
        if not any(test.type == "baseline" for test in tests if test.pdf == pdf):
            test = BaselineTest(id=f"{pdf}_baseline", pdf=pdf, page=1, type="baseline")
            tests.append(test)
            categories[test.id] = "baseline"
    evaluated = evaluate_candidate(str(root / "candidate"), tests, pdfs, force=False)
    if evaluated[2]:
        raise ValueError(f"official evaluator reported {len(evaluated[2])} candidate errors")
    checks = []
    for pdf, pages in evaluated[6].items():
        for page, outcomes in pages.items():
            entry = rows[pdf, page]
            for test, passed, _ in outcomes:
                checks.append({"request_id": entry["row_hash"], "group_id": entry["group_id"],
                               "category": categories[test.id], "test_id": test.id, "passed": bool(passed),
                               "model": model, "method": method, "split": split, "run_context": run_context,
                               "manifest_sha256": payload["manifest_sha256"]})
    checks.sort(key=lambda row: (row["request_id"], row["category"], row["test_id"]))
    if len(checks) != len(tests):
        raise ValueError("official evaluator omitted or duplicated check results")
    pages = defaultdict(list)
    for check in checks:
        pages[check["request_id"], check["category"]].append(check["passed"])
    result = {"format": "asymvlm_olmocr_official_scores_v1", "manifest_sha256": payload["manifest_sha256"],
              "policy_frozen_sha256": policy_frozen_sha256, "model": model, "method": method, "split": split,
              "run_context": run_context, "checks": checks, **official_macro(checks),
              "per_page_category": [{"request_id": request, "category": category, "passed": sum(values), "checks": len(values),
                                     "score": sum(values) / len(values)} for (request, category), values in sorted(pages.items())],
              "repeat_rule": "official evaluator: test passes iff strictly more than half of repeated outputs pass",
              "not_vqa_accuracy": True, "subset_not_full_benchmark": True,
              "evaluation_stage": payload.get("metadata", {}).get("evaluation_stage", "development" if split != "final_test" else "final"),
              "smoke_only": payload.get("metadata", {}).get("evaluation_stage") == "smoke",
              "parent_manifest_sha256": payload.get("metadata", {}).get("parent_manifest_sha256")}
    write_new_json(root / "official_scores.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score")
    score.add_argument("--manifest", type=Path, required=True)
    score.add_argument("--records", type=Path, nargs="+", required=True)
    score.add_argument("--method", required=True)
    score.add_argument("--evaluator-root", type=Path, required=True)
    score.add_argument("--cache-patch", type=Path)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--split", choices=("scorer_dev", "final_test"), required=True)
    score.add_argument("--policy-frozen-sha256")
    compare = sub.add_parser("compare")
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--baseline", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.command == "compare":
        candidate, baseline = json.loads(args.candidate.read_text()), json.loads(args.baseline.read_text())
        result = paired_category_cluster_ci(candidate["checks"], baseline["checks"], repetitions=args.repetitions)
        write_new_json(args.output, result)
        print(json.dumps(result, indent=2))
    else:
        payload = load_manifest(args.manifest)
        verify_evaluator(args.evaluator_root, payload, args.cache_patch)
        records = [json.loads(line) for path in args.records for line in path.read_text().splitlines() if line.strip()]
        records = [row for row in records if row["method"] == args.method]
        result = score_candidate(payload, records, args.output_dir, split=args.split, policy_frozen_sha256=args.policy_frozen_sha256)
        print(json.dumps({key: value for key, value in result.items() if key not in {"checks", "per_page_category"}}, indent=2))


if __name__ == "__main__":
    main()
