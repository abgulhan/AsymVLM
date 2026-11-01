"""Label-free legacy index reconciliation and a separate final-only manifest.

Unknown index conventions remain unresolved. Named legacy validation datasets
are conservatively excluded in full; they are not silently relabeled as fresh.
"""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re

from .audit import IdentityJSONReader, row_identity_keys
from .common import canonical_hash, dataset_family, file_sha256, load_manifest, seal_manifest, write_new_json
from .protocol import _take_unique, finalize_image_provenance, source_inventory


CANONICAL = {"pixparse/docvqa-single-page-questions": "DocVQA", "lmms-lab/textvqa": "TextVQA"}
LEGACY_VALIDATION = {"DocVQA_VAL": "DocVQA", "TextVQA_VAL": "TextVQA", "DocVQA-extra": "DocVQA",
                     "HF_DOCVQA_EXTRA_TRAIN": "DocVQA"}
OUT_OF_SCOPE = {"MME", "ChartQA_TEST", "OCRVQA_TESTCORE", "OCRVQA_TEST", "LLaVABench", "MMDU",
                "POPE", "MMBench_DEV_EN", "MMMU_DEV_VAL", "allenai/olmOCR-bench"}


class FastIdentityJSONReader(IdentityJSONReader):
    """Skip unneeded primitive array runs in C, retaining nested-object IDs.

    Large score/feature vectors contain no object keys. Skipping their numeric
    runs together avoids decoding either their values or evaluation outcomes.
    """
    _array_boundary = re.compile(r'[\[\]{}"]')

    def value(self, retain=False):
        self.whitespace()
        if self.peek() != "[":
            return (yield from super().value(retain))
        self.take()
        while self.peek():
            char = self.peek()
            if char == "]":
                self.take()
                return None
            if char == "}":
                raise ValueError("unexpected object close inside JSON array")
            if char in '[{"':
                yield from self.value(False)
                continue
            match = self._array_boundary.search(self.buffer, self.position)
            self.position = match.start() if match else len(self.buffer)
        raise ValueError("unterminated JSON array")


def iter_identity_records(path):
    with Path(path).open(encoding="utf-8") as source:
        yield from FastIdentityJSONReader(source).records()


def resolve_record(record, source_rows, index_lookup, whole_validation):
    """Return (identity exclusion keys, resolution reason) or (None, reason)."""
    direct = row_identity_keys(record)
    if direct:
        return direct, "explicit_identity"
    dataset = record.get("dataset", record.get("dataset_name", record.get("source_dataset")))
    if dataset in OUT_OF_SCOPE:
        return set(), "outside_DocVQA_TextVQA_scope"
    if dataset in LEGACY_VALIDATION:
        return whole_validation[LEGACY_VALIDATION[dataset]], "entire_named_legacy_validation_source_excluded"
    if dataset not in CANONICAL:
        return None, "unknown_or_missing_dataset"
    index = record.get("dataset_index")
    if not isinstance(index, int):
        return None, "canonical_source_without_integer_dataset_index"
    matches = index_lookup.get((dataset, index), [])
    if not matches:
        return None, "source_index_out_of_range"
    # These repository-native outputs use entry.dataset_index, which indexes
    # the HF source split (hf_docvqa.NativeMixtureDataset._load_source). When the
    # split was omitted, exclude BOTH source splits conservatively.
    keys = set().union(*(row_identity_keys(row) for row in matches))
    return keys, "canonical_HF_index_both_source_splits_excluded"


def reconcile_history(audit, source_rows, *, progress=None):
    unsigned = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if canonical_hash(unsigned) != audit.get("audit_sha256"):
        raise ValueError("historical audit hash mismatch")
    index_lookup, whole_validation = defaultdict(list), defaultdict(set)
    for row in source_rows:
        index_lookup[row["dataset"], int(row["dataset_index"])].append(row)
        if row["source_split"] == "validation":
            whole_validation[dataset_family(row["dataset"])].update(row_identity_keys(row))
    keys = set(audit["identity_keys"])
    inventory = {row["path"]: row for row in audit["files"]}
    remaining, resolved = [], []
    reasons = Counter()
    excluded_validation_families = set()
    issues = {row["path"]: row for row in audit["unresolved"]}
    for row in audit["files"]:
        if row["identity_objects"] > row["resolvable_objects"]:
            issues.setdefault(row["path"], {"path": row["path"], "reason": "only local/ambiguous IDs"})
    issues = list(issues.values())
    for number, issue in enumerate(issues):
        path = Path(issue["path"])
        if issue["reason"] != "only local/ambiguous IDs":
            remaining.append(issue)
            continue
        expected = inventory.get(str(path), {}).get("sha256")
        if not expected or file_sha256(path) != expected:
            remaining.append({"path": str(path), "reason": "artifact_changed_since_initial_audit"})
            continue
        records = list(iter_identity_records(path))
        names = {row.get("dataset", row.get("dataset_name", row.get("source_dataset"))) for row in records}
        names.discard(None)
        unresolved_types = Counter()
        for row in records:
            if not any(row.get(key) for key in ("dataset", "dataset_name", "source_dataset")) and len(names) == 1:
                row = {**row, "dataset": next(iter(names))}
            found, reason = resolve_record(row, source_rows, index_lookup, whole_validation)
            reasons[reason] += 1
            if found is None:
                unresolved_types[reason] += 1
            elif reason == "entire_named_legacy_validation_source_excluded":
                dataset = row.get("dataset", row.get("dataset_name", row.get("source_dataset")))
                family = LEGACY_VALIDATION[dataset]
                if family not in excluded_validation_families:
                    keys.update(found)
                    excluded_validation_families.add(family)
            else:
                keys.update(found)
        if unresolved_types:
            remaining.append({"path": str(path), "reason": "unresolved_index_objects", "types": dict(unresolved_types)})
        else:
            resolved.append({"path": str(path), "sha256": expected, "identity_objects": len(records)})
        if progress and (number + 1) % 100 == 0:
            progress({"processed_files": number + 1, "total": len(issues), "remaining_files": len(remaining)})
    unsupported = []
    scheduling_only = []
    for filename in audit.get("unsupported_files", []):
        path = Path(filename)
        # Only these inspected scheduling-log schemas are exempted. Other TSV
        # files (including dataset rows) stay unsupported, not deserialized.
        known = (path.name == "jobs.tsv" and path.parent.name in {"clustera_fit_slices", "clustera_remainder"}) or path.name == "clusterb_by_gpu_20260730T0645Z.tsv"
        if known:
            scheduling_only.append({"path": str(path), "sha256": file_sha256(path), "role": "PBS_submission_log_no_sample_identities"})
        else:
            unsupported.append(filename)
    result = {"format": "asymvlm_resolved_identity_audit_v1", "scope": ["DocVQA", "TextVQA"],
              "input_audit_sha256": audit["audit_sha256"], "labels_loaded": False,
              "identity_keys": sorted(keys), "identity_count": len(keys), "resolved_files": resolved,
              "unresolved": remaining, "unsupported_files": unsupported, "scheduling_logs": scheduling_only,
              "resolution_counts": dict(reasons), "complete": not remaining and not unsupported,
              "index_semantics_source": "learned_score/multimodel/hf_docvqa.py: NativeMixtureDataset.row_for_entry",
              "legacy_validation_policy": "exclude all official validation rows for any named legacy validation source"}
    result["audit_sha256"] = canonical_hash(result)
    return result


def build_final_only(training, source_rows, audit, *, questions_per_dataset=1000):
    if not audit.get("complete"):
        raise ValueError("final selection remains closed: unresolved historical indices")
    forbidden = set(audit["identity_keys"])
    for rows in training["splits"].values():
        for row in rows:
            forbidden.update(row_identity_keys(row))
    final = []
    for family in ("DocVQA", "TextVQA"):
        for split in ("validation", "train"):
            needed = questions_per_dataset - sum(dataset_family(row["dataset"]) == family for row in final)
            candidates = [row for row in source_rows if dataset_family(row["dataset"]) == family and row["source_split"] == split]
            final.extend(_take_unique(candidates, needed, forbidden,
                                     role="unused_official_validation" if split == "validation" else "unused_official_train_evaluation_source"))
        actual = sum(dataset_family(row["dataset"]) == family for row in final)
        if actual != questions_per_dataset:
            raise ValueError(f"only {actual}/{questions_per_dataset} demonstrably unused {family} document groups remain")
    return seal_manifest({"format": "asymvlm_mlsys_final_only_v1", "splits": {"final_test": final},
                          "metadata": {"training_manifest_sha256": training["manifest_sha256"],
                                       "audit_sha256": audit["audit_sha256"], "historical_audit_complete": True,
                                       "historical_audit_scope": audit.get("scope", ["DocVQA", "TextVQA"]),
                                       "final_status": "newly_selected_unopened", "previous_confirmation_is_fresh": False,
                                       "training_manifest_unchanged": True,
                                       "source_counts": dict(Counter(f"{dataset_family(row['dataset'])}/{row['source_split']}" for row in final))}})


def validate_final_binding(payload, training, audit):
    """Recheck identities after image hashing, without changing training data."""
    if payload["metadata"]["training_manifest_sha256"] != training["manifest_sha256"]:
        raise ValueError("final manifest is bound to a different training manifest")
    if not audit.get("complete"):
        raise ValueError("historical identity resolution is incomplete")
    forbidden = set(audit["identity_keys"])
    for rows in training["splits"].values():
        for row in rows:
            forbidden.update(row_identity_keys(row))
    for row in payload["splits"]["final_test"]:
        keys = row_identity_keys(row)
        if keys & forbidden:
            raise ValueError(f"final identity/image overlaps prior data: {row['row_hash']}")
        forbidden.update(keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--final-output", type=Path)
    args = parser.parse_args()
    audit = json.loads(args.audit.read_text())
    sources, _ = source_inventory(args.project_root, load_manifest(args.native))
    result = reconcile_history(audit, sources, progress=lambda row: print(json.dumps(row), flush=True))
    write_new_json(args.output, result)
    print(json.dumps({key: result[key] for key in ("complete", "identity_count", "resolution_counts")}))
    print(json.dumps({"unresolved_files": len(result["unresolved"]), "unsupported_files": len(result["unsupported_files"])}))
    if args.final_output:
        training = load_manifest(args.training)
        payload = build_final_only(training, sources, result)
        payload = finalize_image_provenance(payload, args.project_root)
        validate_final_binding(payload, training, result)
        write_new_json(args.final_output, payload)


if __name__ == "__main__":
    main()
