"""Answer-free reconciliation of frozen adaptive counts to a static token total.

Each dataset is reconciled independently. Only cached entropy, patch counts,
and identities are consumed; no evaluation quality is consulted. The initial
policy uses 1%-steps; final single-token residuals are explicitly recorded.
"""

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

from .adaptive_vision_policy import AdaptiveVisionPolicy
from .state import digest
from learned_score.mlsys_eval.common import write_new_json

FORMAT = "mlsys_adaptive_budget_assignments_v1"


def build_assignments(policy, rows, *, manifest_sha256, target=.7, low=.6, high=.8):
    if policy.statistic != "normalized_entropy":
        raise ValueError("cached assignment builder requires normalized entropy")
    if not 0 < low <= target <= high < 1:
        raise ValueError("invalid budget range")
    percentages = [round(value * 100) for value in (low, target, high)]
    if any(not math.isclose(value * 100, percent, abs_tol=1e-9) for value, percent in zip((low, target, high), percentages)):
        raise ValueError("budget range must use integer percentages")
    lo, goal, hi = percentages
    groups = defaultdict(list)
    if not rows or len({row["row_hash"] for row in rows}) != len(rows):
        raise ValueError("assignment identities must be nonempty and unique")
    for row in rows:
        if row.get("split") != "scorer_dev":
            raise ValueError("assignment preparation is restricted to development inputs")
        policy.validate_evaluation_identity(row_hash=row["row_hash"], group_id=row["group_id"],
                                            image_sha256=row["image_sha256"], split="scorer_selection")
        decision = policy.decide_from_statistic(row["normalized_entropy"], row["input_visual_tokens"])
        if not decision.fallback and not low <= decision.retention <= high:
            raise ValueError("policy retention is outside the predeclared reconciliation range")
        groups[row["dataset"]].append((row, decision))
    output, summary = {}, {}
    for dataset, values in sorted(groups.items()):
        values.sort(key=lambda pair: (pair[1].oriented_value, pair[0]["row_hash"]))
        counts = [row["input_visual_tokens"] for row, _ in values]
        target_tokens = sum(max(1, n * goal // 100) for n in counts)
        minima = [n if decision.fallback else max(1, n * lo // 100) for n, (_, decision) in zip(counts, values)]
        maxima = [n if decision.fallback else max(1, n * hi // 100) for n, (_, decision) in zip(counts, values)]
        if not sum(minima) <= target_tokens <= sum(maxima):
            raise ValueError("full-fallback counts make exact token matching infeasible")
        candidates = []
        for shift in range(lo - hi, hi - lo + 1):
            selected = [n if decision.fallback else max(1, n * min(hi, max(lo, round(decision.retention * 100) + shift)) // 100)
                        for n, (_, decision) in zip(counts, values)]
            if sum(selected) <= target_tokens:
                candidates.append((sum(selected), -abs(shift), shift, selected))
        _, _, shift, selected = max(candidates, key=lambda item: item[:3])
        initial = list(selected)
        residual = target_tokens - sum(selected)
        while residual:
            progressed = False
            for index in reversed(range(len(values))):
                if selected[index] < maxima[index]:
                    selected[index] += 1
                    residual -= 1
                    progressed = True
                    if not residual:
                        break
            if not progressed:
                raise ValueError("token reconciliation cannot reach its target")
        for (row, decision), kept, before in zip(values, selected, initial):
            output[row["row_hash"]] = {
                "row_hash": row["row_hash"], "group_id": row["group_id"], "image_sha256": row["image_sha256"],
                "dataset": dataset, "input_visual_tokens": row["input_visual_tokens"],
                "statistic_value": decision.statistic_value, "policy_selected_count": decision.selected_count,
                "policy_retention": decision.retention, "selected_count": kept, "fallback": decision.fallback,
                "integer_residual_tokens": kept - before,
            }
        summary[dataset] = {"requests": len(values), "input_visual_tokens": sum(counts),
                            "static_visual_tokens": target_tokens, "adaptive_visual_tokens": sum(selected),
                            "token_error": 0., "percent_shift": shift,
                            "integer_residual_tokens": sum(selected) - sum(initial),
                            "full_fallback_requests": sum(decision.fallback for _, decision in values)}
    payload = {"format": FORMAT, "policy_sha256": digest(policy.payload), "manifest_sha256": manifest_sha256,
               "target_retention": target, "minimum_retention": low, "maximum_retention": high,
               "labels_used": False, "rows": output, "datasets": summary,
               "scope": "previously_examined_development; not fresh confirmation"}
    return {**payload, "sha256": digest(payload)}


def load_assignments(path, policy, manifest_sha256, entries):
    from learned_score.mlsys_eval.common import dataset_family
    payload = json.loads(Path(path).read_text())
    if payload.get("sha256") != digest({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError("budget assignment digest mismatch")
    if (payload.get("format") != FORMAT or payload.get("labels_used") is not False
            or payload.get("policy_sha256") != digest(policy.payload)
            or payload.get("manifest_sha256") != manifest_sha256
            or set(payload["rows"]) != {entry["row_hash"] for entry in entries}):
        raise ValueError("budget assignments do not match policy, manifest, or request subset")
    original = []
    for entry in entries:
        row = payload["rows"][entry["row_hash"]]
        if any(row.get(key) != entry.get(key) for key in ("row_hash", "group_id", "image_sha256")):
            raise ValueError("budget assignment identity mismatch")
        if row.get("dataset") != dataset_family(entry["dataset"]):
            raise ValueError("budget assignment dataset mismatch")
        original.append({**row, "split": "scorer_dev", "normalized_entropy": row["statistic_value"]})
    expected = build_assignments(policy, original, manifest_sha256=manifest_sha256,
                                 target=payload["target_retention"], low=payload["minimum_retention"],
                                 high=payload["maximum_retention"])
    if expected != payload:
        raise ValueError("budget assignments are not the deterministic label-free reconciliation")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--request-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    bundle = json.loads(args.input.read_text())
    wanted = {row["row_hash"] for row in json.loads(args.request_ids.read_text())}
    rows = [row for row in bundle["rows"] if row["row_hash"] in wanted]
    if {row["row_hash"] for row in rows} != wanted:
        raise ValueError("screening subset differs from cached feature identities")
    policy = AdaptiveVisionPolicy.load(args.policy)
    result = build_assignments(policy, rows, manifest_sha256=bundle["identity"]["calibration_manifest_sha256"])
    write_new_json(args.output, result)
    print(json.dumps({"output": str(args.output), "datasets": result["datasets"]}), flush=True)


if __name__ == "__main__":
    main()
