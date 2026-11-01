"""Identity-only confirmation eligibility; never silently reuse development.

The native source_catalog.jsonl contains questions/answers and is deliberately
NOT opened. Parquet inventory reads identifier columns only; TSV inventory
reads index/image-path/category columns only. Historical ambiguity blocks
publication rather than being inferred away from filenames.
"""

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from learned_score.mlsys_eval import audit as identity_audit, common, protocol
from learned_score.mlsys_eval.audit import row_identity_keys
from learned_score.mlsys_eval.common import canonical_hash, file_sha256, load_manifest, seal_manifest, write_new_json
from learned_score.mlsys_eval.protocol import source_inventory, finalize_image_provenance
from . import adaptive_data_v3 as data, compression_matrix_v8 as matrix


FORMAT = "mlsys_compression_confirmation_audit_v8"
FAMILIES = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
FORBIDDEN = frozenset({"question", "answer", "answers", "prediction", "quality_score", "references"})


def seal(value):
    return {**value, "sha256": canonical_hash(value)}


def checked(value):
    if value.get("sha256") != canonical_hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("confirmation audit changed after creation")
    return value


def identity_rows(manifest):
    rows = [row for split in manifest["splits"].values() for row in split]
    if any(FORBIDDEN & row.keys() for row in rows):
        raise ValueError("identity manifest contains question/answer/outcome fields")
    return rows


def assess(candidates, exclusions, history, *, minimum_groups=1000, expected_mme_categories=None):
    """Pure provisional counts; unresolved history never becomes fresh evidence."""
    from .adaptive_data_v4 import MME_CATEGORIES
    expected_mme_categories = tuple(expected_mme_categories or MME_CATEGORIES)
    if type(minimum_groups) is not int or minimum_groups < 1:
        raise ValueError("confirmation group minimum must be positive")
    if any(FORBIDDEN & row.keys() for row in [*candidates, *exclusions]):
        raise ValueError("confirmation selection accepts identity records only")
    blocked = []
    unknown = copy.deepcopy(history.get("unresolved", []))
    unsupported = list(history.get("unsupported_files", []))
    if history.get("labels_loaded") is not False:
        blocked.append({"code": "history_not_certified_label_free"})
    if history.get("complete") is not True or unknown or unsupported:
        blocked.append({"code": "incomplete_historical_identity_audit", "unresolved_artifacts": len(unknown),
                        "unsupported_artifacts": len(unsupported)})
    scope = {data.dataset_family(name) for name in history.get("scope", [])}
    missing_scope = sorted(set(FAMILIES) - scope)
    if missing_scope:
        blocked.append({"code": "historical_audit_missing_benchmark_scope", "datasets": missing_scope})
    forbidden = set(history.get("identity_keys", []))
    for row in exclusions:
        forbidden.update(row_identity_keys(row))
        forbidden.update(data._split_keys(row))
    by_family, available = defaultdict(list), defaultdict(list)
    for row in candidates:
        family = data.dataset_family(row["dataset"])
        if family not in FAMILIES:
            continue
        by_family[family].append(row)
        if not (row_identity_keys(row) | data._split_keys(row)) & forbidden:
            available[family].append(row)
    counts = {}
    for family in FAMILIES:
        raw, remaining = by_family[family], available[family]
        count = {"source_rows": len(raw), "source_groups": len({r["group_id"] for r in raw}),
                 "excluded_rows": len(raw) - len(remaining), "provisional_remaining_rows": len(remaining),
                 "provisional_remaining_groups": len({r["group_id"] for r in remaining}),
                 "source_splits": dict(Counter(r["source_split"] for r in raw)),
                 "freshness_certified": False}
        if family != "MME" and count["provisional_remaining_groups"] < minimum_groups:
            blocked.append({"code": "insufficient_unexamined_groups", "dataset": family,
                            "available": count["provisional_remaining_groups"], "required": minimum_groups})
        if family == "MME":
            groups = defaultdict(list)
            for row in remaining:
                groups[(row.get("category"), row["group_id"])].append(row)
            pairs = Counter(category for (category, _), members in groups.items()
                            if len(members) == 2 and len({r["row_hash"] for r in members}) == 2)
            count["complete_pairs_by_category"] = dict(pairs)
            if set(pairs) != set(expected_mme_categories) or any(v < 2 for v in pairs.values()):
                blocked.append({"code": "insufficient_intact_mme_pairs", "pairs_by_category": dict(pairs),
                                "required_categories": list(expected_mme_categories), "minimum_pairs_per_category": 2})
        counts[family] = count
    return seal({"format": FORMAT, "status": "eligible_pending_image_provenance" if not blocked else "blocked",
        "eligible": not blocked, "labels_loaded": False, "annotations_opened": False,
        "minimum_non_mme_groups": minimum_groups, "counts": counts,
        "blocking_reasons": blocked, "unknown_artifacts": unknown, "unsupported_artifacts": unsupported,
        "history_scope": sorted(scope), "historical_exclusion_keys": len(history.get("identity_keys", [])),
        "all_exclusion_keys": len(forbidden), "fresh_confirmation_manifest_created": False,
        "candidate_identity_sha256": canonical_hash(candidates), "excluded_identity_sha256": canonical_hash(exclusions),
        "count_caveat": "Remaining counts are provisional under conservative identity exclusions; image-byte deduplication and full source-hash verification precede publication."})


def collect_inputs(study):
    """Read existing local identities; no download, expanded catalog, or labels."""
    matrix.verify(study)
    root = Path(study["cwd"])
    old_root = Path(study["old_study"]).parent
    summary_path = old_root / "data/summary.json"
    summary = json.loads(summary_path.read_text())
    history_path = Path(summary["historical_audit_path"])
    if file_sha256(history_path) != summary["historical_audit_sha256"]:
        raise ValueError("historical audit changed from the frozen development provenance")
    history = json.loads(history_path.read_text())
    if history.get("audit_sha256") != canonical_hash({k: v for k, v in history.items() if k != "audit_sha256"}):
        raise ValueError("historical identity audit content hash mismatch")
    paths = [root / "reproduction_results/native_protocol/manifest.json",
             root / "reproduction_results/mlsys_20261012/manifest.json",
             root / "reproduction_results/mlsys_adaptive_20261016_v3/data/manifest.json", Path(study["manifest"])]
    manifests = [load_manifest(p) for p in paths]
    exclusions = [row for manifest in manifests for row in identity_rows(manifest)]
    candidates, files = source_inventory(root, manifests[0])
    for family in ("ChartQA_TEST", "MME"):
        path = root / ".data/LMUData/" / (family + ".tsv")
        rows = data._identity_tsv(path, family)
        candidates.extend(rows)
        files.append({"path": str(path), "rows": len(rows), "sha256": file_sha256(path),
                      "sha256_verified_this_run": True, "columns_read": ["index", "image_path", "category"]})
    dependencies = [Path(module.__file__) for module in (data, identity_audit, common, protocol)]
    pins = {str(p): file_sha256(p) for p in (*paths, summary_path, history_path, Path(__file__), *dependencies)}
    return candidates, exclusions, history, {"input_hashes": pins, "source_files": files,
        "existing_v4_fit_candidate_pool_only": summary["source_candidate_counts"],
        "label_catalog_opened": False, "source_catalog_expanded": False,
        "parquet_columns_read": {"DocVQA": ["question_id", "other_metadata"], "TextVQA": ["question_id", "image_id"]}}


def audit(study):
    candidates, exclusions, history, provenance = collect_inputs(study)
    result = assess(candidates, exclusions, history,
                    minimum_groups=study["confirmation"]["non_mme_target_groups"])
    return seal({**{k: v for k, v in result.items() if k != "sha256"}, "study_sha256": study["sha256"],
                 "provenance": provenance})


def prepare_manifest(study, result):
    """Publish only after complete history, required cohort size, and image audit."""
    checked(result)
    if (result.get("study_sha256") != study["sha256"] or result.get("eligible") is not True
            or result.get("blocking_reasons") or result.get("unknown_artifacts")
            or result.get("unsupported_artifacts")):
        raise ValueError("fresh confirmation blocked; see identity audit blocking_reasons")
    current = audit(study)
    if current != result:
        raise ValueError("confirmation inputs changed after the eligibility audit")
    candidates, exclusions, history, _ = collect_inputs(study)
    root = Path(study["cwd"])
    # Image bytes may be inspected for deduplication; annotation columns remain unopened.
    parquet = [copy.deepcopy(row) for row in candidates if row.get("source_format") == "parquet"]
    tsv = [copy.deepcopy(row) for row in candidates if row.get("source_format") == "tsv"]
    forbidden = set(history["identity_keys"])
    for row in exclusions:
        forbidden.update(row_identity_keys(row) | data._split_keys(row))
    parquet = [r for r in parquet if not (row_identity_keys(r) | data._split_keys(r)) & forbidden]
    tsv = [r for r in tsv if not (row_identity_keys(r) | data._split_keys(r)) & forbidden]
    enriched = finalize_image_provenance({"splits": {"evaluation": parquet}}, root)["splits"]["evaluation"] if parquet else []
    data._hash_tsv_images(tsv, root)
    enriched += tsv
    selected = []
    for family in FAMILIES:
        pool = sorted((r for r in enriched if data.dataset_family(r["dataset"]) == family),
                      key=lambda r: canonical_hash(["compression-v8-confirmation", r["row_hash"]]))
        if family == "MME":
            groups = defaultdict(list)
            for row in pool:
                groups[row["group_id"]].append(row)
            additions = []
            for members in groups.values():
                keys = {k for row in members for k in row_identity_keys(row) | data._split_keys(row)}
                if len(members) == 2 and not keys & forbidden:
                    additions.extend(members)
                    forbidden.update(keys)
        else:
            additions = []
            for row in pool:
                keys = row_identity_keys(row) | data._split_keys(row)
                if keys & forbidden:
                    continue
                additions.append(row)
                forbidden.update(keys)
                if len(additions) == result["minimum_non_mme_groups"]:
                    break
            if len(additions) != result["minimum_non_mme_groups"]:
                raise ValueError("insufficient image-disjoint confirmation groups after image hashing")
        selected.extend(data._decorate(r, "fresh_confirmation_v8") for r in additions)
    # Reapply all scope/count exclusions after image-byte provenance enrichment.
    final_check = assess(selected, exclusions, history, minimum_groups=result["minimum_non_mme_groups"])
    if not final_check["eligible"]:
        raise ValueError("final confirmation identity audit failed")
    payload = seal_manifest({"format": "mlsys_adaptive_data_v3", "dataset": "ASYMVLM_COMPRESSION_V8_CONFIRMATION",
        "metadata": {"study_version": "compression_v8_confirmation", "identity_audit_sha256": result["sha256"],
                     "fresh_confirmation": True, "selection_uses_labels": False, "source_catalog_path": None},
        "splits": {"fit": [], "screening": [], "evaluation": selected, "phi_train": []}})
    path = Path(study["output_root"]) / "data/confirmation_manifest.json"
    write_new_json(path, payload)
    return path


def verified_manifest(study, result):
    """Return manifest and sealed proof; recheck frozen sources on every resume.

    Ineligible audits fail before any output creation or image/annotation read.
    The complete audit is rederived, never accepted solely because its seal is
    internally consistent. The subsequent proof binds all selected source and
    image hashes as well as the manifest, study, and original audit.
    """
    checked(result)
    if not result.get("eligible") or result.get("blocking_reasons"):
        raise ValueError("fresh confirmation blocked; see identity audit blocking_reasons")
    if result.get("study_sha256") != study["sha256"] or audit(study) != result:
        raise ValueError("confirmation inputs changed after eligibility audit")
    path = Path(study["output_root"]) / "data/confirmation_manifest.json"
    if not path.exists():
        path = prepare_manifest(study, result)
    payload = load_manifest(path)
    metadata = payload.get("metadata", {})
    if (metadata.get("fresh_confirmation") is not True or metadata.get("selection_uses_labels") is not False
            or metadata.get("identity_audit_sha256") != result["sha256"]
            or set(payload["splits"]) != {"fit", "screening", "evaluation", "phi_train"}
            or any(payload["splits"][s] for s in ("fit", "screening", "phi_train"))):
        raise ValueError("manifest lacks original fresh-confirmation admission")
    rows = identity_rows(payload)
    image_groups = {}
    seen_groups = set()
    for row in rows:
        family = data.dataset_family(row["dataset"])
        group = family, row["group_id"]
        image = row.get("image_sha256")
        if not image or image_groups.get(image, group) != group:
            raise ValueError("confirmation image overlaps independent groups")
        image_groups[image] = group
        if family != "MME" and group in seen_groups:
            raise ValueError("confirmation requires one question per independent group")
        seen_groups.add(group)
    for family in FAMILIES[:-1]:
        if sum(data.dataset_family(r["dataset"]) == family for r in rows) != result["minimum_non_mme_groups"]:
            raise ValueError("confirmation cohort size differs from preregistered minimum")
    candidates, exclusions, history, _ = collect_inputs(study)
    # Bind each published row to an audited source identity before hashing.
    lookup = {r["row_hash"]: r for r in candidates}
    source_pins = {}
    for row in rows:
        original = lookup.get(row["row_hash"])
        if original is None or any(row.get(k) != original.get(k) for k in
                ("dataset", "question_id", "source_file", "source_row_index", "source_split")):
            raise ValueError("confirmation row is not an audited source identity")
        source = Path(study["cwd"]) / row["source_file"]
        if str(source) not in source_pins:
            source_pins[str(source)] = file_sha256(source)
        if row.get("source_shard_sha256") != source_pins[str(source)]:
            raise ValueError("confirmation source shard changed")
    parquet = [copy.deepcopy(row) for row in rows if row.get("source_format") == "parquet"]
    for row in parquet:
        row.pop("image_sha256", None)
    rebuilt = finalize_image_provenance({"splits": {"evaluation": parquet}}, Path(study["cwd"]))["splits"]["evaluation"] if parquet else []
    tsv = [copy.deepcopy(row) for row in rows if row.get("source_format") == "tsv"]
    data._hash_tsv_images(tsv, Path(study["cwd"]))
    rebuilt += tsv
    images = {r["row_hash"]: r["image_sha256"] for r in rebuilt}
    if len(images) != len(rows) or any(images.get(r["row_hash"]) != r.get("image_sha256") for r in rows):
        raise ValueError("confirmation image provenance changed")
    final = assess(rows, exclusions, history, minimum_groups=result["minimum_non_mme_groups"])
    if not final["eligible"]:
        raise ValueError("final confirmation source/image exclusions failed")
    proof = seal({"format": "mlsys_compression_confirmation_proof_v8", "study_sha256": study["sha256"],
        "audit_sha256": result["sha256"], "manifest_path": str(path),
        "manifest_sha256": file_sha256(path), "manifest_content_sha256": payload["manifest_sha256"],
        "fresh_identity_audit_passed": True, "image_exclusions_verified": True,
        "labels_loaded": False, "annotations_opened": False,
        "input_hashes": {**result["provenance"]["input_hashes"], **source_pins},
        "selected_images_sha256": canonical_hash(images),
        "counts": {family: {**count, "freshness_certified": True} for family, count in final["counts"].items()}})
    write_new_json(path.with_name("confirmation_proof.json"), proof)
    return path, proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=matrix.ROOT / "study.json")
    parser.add_argument("--publish-audit", action="store_true")
    parser.add_argument("--prepare-manifest", action="store_true")
    args = parser.parse_args(argv)
    study = matrix.checked(args.study)
    result = audit(study)
    if args.publish_audit:
        write_new_json(Path(study["output_root"]) / "data/confirmation_audit.json", result)
    if args.prepare_manifest:
        print(prepare_manifest(study, result))
    else:
        print(json.dumps({k: result[k] for k in ("status", "eligible", "counts", "blocking_reasons")}, indent=2))


if __name__ == "__main__":
    main()
