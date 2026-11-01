"""A bounded metadata-only second pass over unresolved historical identities.

This does not infer dataset names from directory names. Explicit runtime metadata
and a checked derived-file relationship may resolve an issue. All other issues,
including binary archives, remain open and keep final VQA evaluation closed.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import re

from .common import canonical_hash, file_sha256, write_new_json
from .historical_resolver import LEGACY_VALIDATION, OUT_OF_SCOPE, iter_identity_records


def checked_audit(path):
    payload = json.loads(Path(path).read_text())
    if canonical_hash({key: value for key, value in payload.items() if key != "audit_sha256"}) != payload["audit_sha256"]:
        raise ValueError(f"audit contents changed: {path}")
    return payload


def explicit_context(path):
    """Return only dataset-bearing metadata lines, never model responses."""
    candidates = [path.with_suffix(".log"), path.parent / "rank0.log",
                  path.parent / "run_manifest.txt", path.parent / "environment.txt"]
    patterns = [(re.compile(r"^Dataset:\s*([A-Za-z0-9_./-]+)(?:,|\s|$)"), None),
                (re.compile(r"^ASYMVLM_DATASET=([A-Za-z0-9_./-]+)\s*$"), None),
                (re.compile(r"^Loading MMDU from .*(?:MMDU\.tsv|mmdu)"), "MMDU"),
                (re.compile(r"^protocol=.*\bMME\b"), "MME")]
    evidence = []
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        matched = []
        with candidate.open(errors="replace") as source:
            for line_number, line in enumerate(source, 1):
                for pattern, fixed in patterns:
                    match = pattern.search(line)
                    if match:
                        matched.append({"line": line_number, "dataset": fixed or match[1]})
        if matched:
            evidence.append({"path": str(candidate), "sha256": file_sha256(candidate), "matches": matched})
    names = {item["dataset"] for record in evidence for item in record["matches"]}
    return names, evidence


def refine(primary_path, original_path, supplementary_paths):
    primary, original = checked_audit(primary_path), checked_audit(original_path)
    if primary["input_audit_sha256"] != original["audit_sha256"]:
        raise ValueError("resolved and original primary audit do not match")
    inventory = {row["path"]: row for row in original["files"]}
    unresolved_paths = {row["path"] for row in primary["unresolved"]}
    safe_paths = {path for path in inventory if path not in unresolved_paths}
    # This inspected producer copies dataset_index from rank*.jsonl to failures.
    producer = Path("scripts/aggregate_llava_methods.py")
    producer_evidence = {"path": str(producer), "sha256": file_sha256(producer),
                         "relationship": "failure_cases.dataset_index is copied from the same directory's rank*.jsonl"}
    remaining, resolved = [], []
    for issue in primary["unresolved"]:
        path = Path(issue["path"])
        expected = inventory.get(str(path), {}).get("sha256")
        if not expected or file_sha256(path) != expected:
            remaining.append({**issue, "reason": "artifact_changed_since_identity_audit"})
            continue
        names, evidence = explicit_context(path)
        if len(names) == 1 and next(iter(names)) in OUT_OF_SCOPE:
            resolved.append({"path": str(path), "sha256": expected,
                             "resolution": "explicit_runtime_dataset_outside_DocVQA_TextVQA", "evidence": evidence})
            continue
        if len(names) == 1 and next(iter(names)) in LEGACY_VALIDATION:
            # The prior pass already excludes the complete named validation
            # sources; do not pretend an index was matched to one question.
            if primary.get("resolution_counts", {}).get("entire_named_legacy_validation_source_excluded", 0):
                resolved.append({"path": str(path), "sha256": expected,
                                 "resolution": "explicit_legacy_validation_context_conservatively_excluded_in_full",
                                 "dataset": next(iter(names)), "evidence": evidence})
                continue
        if path.name == "failure_cases.json":
            siblings = sorted(path.parent.glob("rank*.jsonl"))
            if siblings and all(str(sibling) in safe_paths and file_sha256(sibling) == inventory[str(sibling)]["sha256"] for sibling in siblings):
                failures = {row["dataset_index"] for row in iter_identity_records(path) if "dataset_index" in row}
                represented = {row["dataset_index"] for sibling in siblings for row in iter_identity_records(sibling) if "dataset_index" in row}
                if failures and failures <= represented:
                    resolved.append({"path": str(path), "sha256": expected,
                                     "resolution": "derived_failure_indices_already_present_in_audited_sibling_results",
                                     "producer": producer_evidence, "failure_indices": sorted(failures),
                                     "audited_siblings": [{"path": str(sibling), "sha256": inventory[str(sibling)]["sha256"]} for sibling in siblings]})
                    continue
        remaining.append({**issue, "explicit_dataset_names": sorted(names), "metadata_evidence": evidence})
    supplementary = []
    for path in supplementary_paths:
        audit = checked_audit(path)
        supplementary.append({"path": str(path), "audit_sha256": audit["audit_sha256"],
                              "complete": audit["complete"], "unresolved": audit["unresolved"],
                              "unsupported_files": audit["unsupported_files"], "identity_count": audit["identity_count"]})
    result = {"format": "asymvlm_historical_context_status_v1", "final_vqa_status": "closed",
              "final_manifest_created": False, "complete": False, "labels_decoded": False,
              "primary_audit_sha256": primary["audit_sha256"], "identity_count": primary["identity_count"],
              "additional_resolved_files": resolved, "remaining_primary": remaining,
              "primary_resolution_counts": dict(Counter(row["resolution"] for row in resolved)),
              "supplementary_scopes": supplementary,
              "required_before_final_release": [
                  "Resolve every remaining source-local index using an explicit historical dataset/source binding.",
                  "Audit identities in unsupported binary/TSV archives without opening evaluation outcomes; do not infer dataset identity from filenames alone.",
                  "Merge certified exclusion identities from every archive scope and verify document/image disjointness.",
                  "Create a separate final-only manifest referencing, never modifying, the frozen training manifest."],
              "training_manifest_unchanged": True}
    result["status_sha256"] = canonical_hash(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary", type=Path, required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--supplementary", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = refine(args.primary, args.original, args.supplementary)
    write_new_json(args.output, result)
    print(json.dumps({"final_vqa_status": result["final_vqa_status"],
                      "additionally_resolved": len(result["additional_resolved_files"]),
                      "remaining_primary": len(result["remaining_primary"]),
                      "resolution_counts": result["primary_resolution_counts"]}), flush=True)


if __name__ == "__main__":
    main()
