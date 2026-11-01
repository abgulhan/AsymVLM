"""Official-train-only scorer data and independently audited final questions."""

import copy
from collections import Counter
from pathlib import Path

from .audit import row_identity_keys
from .common import canonical_hash, dataset_family, file_sha256, seal_manifest


SOURCE_SPECS = {
    "docvqa_33136ef4": ("pixparse/docvqa-single-page-questions",
                        "33136ef456fa5a3fe68568d6e31dda4eeff95b9b"),
    "textvqa_9c0699cd": ("lmms-lab/textvqa",
                         "9c0699cd19768ac5ab97568f6b3cbac4c0062884"),
}


def selection_key(row, salt="mlsys-2027-v1"):
    return canonical_hash([salt, row["dataset"], row["source_split"],
                           str(row["question_id"])])


def source_inventory(project_root, native_manifest):
    """Read provenance/IDs only: no questions, answers, OCR text, or image bytes."""
    import pyarrow.parquet as pq

    root = Path(project_root).resolve()
    known_shards = {
        row["source_file"]: row["source_shard_sha256"]
        for rows in native_manifest["splits"].values() for row in rows
        if row.get("source_file") and row.get("source_shard_sha256")
    }
    entries, files = [], []
    for directory, (dataset, revision) in SOURCE_SPECS.items():
        data_root = root / ".data/native_sources" / directory / "data"
        for split in ("train", "validation"):
            offset = 0
            for path in sorted(data_root.glob(f"{split}-*.parquet")):
                relative = str(path.relative_to(root))
                shard_sha = known_shards.get(relative)
                parquet = pq.ParquetFile(path)
                columns = ["question_id", "other_metadata"] if "docvqa" in directory else ["question_id", "image_id"]
                metadata_rows = parquet.read(columns=columns).to_pylist()
                files.append({"path": relative, "rows": len(metadata_rows),
                              "bytes": path.stat().st_size,
                              "sha256": shard_sha, "sha256_verified_this_run": False})
                for index, raw in enumerate(metadata_rows):
                    meta = raw.get("other_metadata") or {}
                    image_id = str(raw.get("image_id") or meta.get("image") or "")
                    if not image_id:
                        raise ValueError(f"source lacks image identity: {path}:{index}")
                    document_id = meta.get("ucsf_document_id")
                    if dataset_family(dataset) == "DocVQA" and not document_id:
                        raise ValueError(f"DocVQA source lacks UCSF document ID: {path}:{index}")
                    family = dataset_family(dataset)
                    group = f"{family}:document:{document_id}" if document_id else f"{family}:image:{image_id}"
                    entry = {"dataset": dataset, "dataset_revision": revision,
                             "dataset_index": offset + index, "source_split": split,
                             "question_id": str(raw["question_id"]), "image_id": image_id,
                             "source_file": relative, "source_row_index": index,
                             "source_format": "parquet", "group_id": group,
                             "group_provenance": "ucsf_document_id" if document_id else "image_id"}
                    if document_id:
                        entry["document_id"] = str(document_id)
                    if shard_sha:
                        entry["source_shard_sha256"] = shard_sha
                    entry["row_hash"] = canonical_hash(entry)
                    entry["row_hash_kind"] = "source_locator_v1"
                    entry["selection_sha256"] = selection_key(entry)
                    entries.append(entry)
                offset += len(metadata_rows)
    return entries, files


def _enrich_native(row, lookup):
    key = (dataset_family(row["dataset"]), row["source_split"], str(row["question_id"]))
    if key not in lookup:
        raise ValueError(f"native row has no official source identity: {key}")
    source = lookup[key]
    if Path(source["image_id"]).stem != Path(row["image_id"]).stem:
        raise ValueError(f"native/source image identity disagreement: {key}")
    result = dict(row)
    for name in ("group_id", "group_provenance", "document_id"):
        if name in source:
            result[name] = source[name]
    result["provenance_role"] = "retained_native_official_train"
    return result


def _take_unique(candidates, count, excluded, *, role):
    chosen = []
    if count == 0:
        return chosen
    for source in sorted(candidates, key=selection_key):
        keys = row_identity_keys(source)
        if keys & excluded:
            continue
        row = dict(source, provenance_role=role)
        chosen.append(row)
        excluded.update(keys)
        if len(chosen) == count:
            break
    return chosen


def build_manifest(native, source_rows, audit, *, training_per_dataset=2800,
                   development_per_dataset=500, final_per_dataset=1000,
                   training_only=False):
    """Preserve development, repair train/document overlap, never reopen old finals."""
    lookup = {(dataset_family(row["dataset"]), row["source_split"], str(row["question_id"])): row
              for row in source_rows}
    development = [_enrich_native(row, lookup)
                   for name in ("scorer_screening", "scorer_selection")
                   for row in native["splits"].get(name, [])]
    expected = {"DocVQA": development_per_dataset, "TextVQA": development_per_dataset}
    if dict(Counter(dataset_family(row["dataset"]) for row in development)) != expected:
        raise ValueError("native development does not have the required per-dataset count")
    for row in development:
        row["provenance_role"] = "previously_inspected_development"
    reserved = set().union(*(row_identity_keys(row) for row in development))
    old_train = [_enrich_native(row, lookup) for row in native["splits"].get("scorer_train", [])
                 if row.get("source_split") == "train" and dataset_family(row["dataset"]) in expected]
    training, removed = [], []
    for row in old_train:
        if row_identity_keys(row) & reserved:
            removed.append(row["row_hash"])
        else:
            training.append(row)
    # All previous native identities, including historically 'sealed' rows, are
    # excluded from both replacement data and fresh final selection.
    prior = set(audit.get("identity_keys", []))
    for rows in native["splits"].values():
        for row in rows:
            prior.update(row_identity_keys(row))
    used = reserved | set().union(*(row_identity_keys(row) for row in training))
    for family in expected:
        count = sum(dataset_family(row["dataset"]) == family for row in training)
        missing = training_per_dataset - count
        if missing < 0:
            raise ValueError(f"too many original training rows for {family}")
        candidates = [row for row in source_rows if dataset_family(row["dataset"]) == family
                      and row["source_split"] == "train"]
        replacement = _take_unique(candidates, missing, used | prior,
                                   role="official_train_replacement_for_document_separation")
        if len(replacement) != missing:
            raise ValueError(f"insufficient unused official-train document groups for {family}: "
                             f"need {missing}, found {len(replacement)}")
        training.extend(replacement)
        for row in replacement:
            used.update(row_identity_keys(row))
    final = []
    if not training_only:
        if not audit.get("complete"):
            raise ValueError("fresh final selection blocked: unresolved historical identity audit")
        forbidden = used | prior
        for family in expected:
            for split, role in (("validation", "unused_official_validation"),
                                ("train", "unused_official_train_evaluation_source")):
                need = final_per_dataset - sum(dataset_family(row["dataset"]) == family for row in final)
                pool = [row for row in source_rows if dataset_family(row["dataset"]) == family
                        and row["source_split"] == split]
                final.extend(_take_unique(pool, need, forbidden, role=role))
            count = sum(dataset_family(row["dataset"]) == family for row in final)
            if count != final_per_dataset:
                raise ValueError(f"insufficient unused {family} final groups: {count}/{final_per_dataset}")
    payload = {"format": "asymvlm_mlsys_disjoint_v1", "dataset": "ASYMVLM_NATIVE_MIXTURE",
               "splits": {"scorer_train": sorted(training, key=selection_key),
                          "scorer_dev": sorted(development, key=selection_key),
                          "final_test": sorted(final, key=selection_key)},
               "metadata": {"selection": "deterministic SHA order, one question per new source group",
                            "native_manifest_sha256": native.get("manifest_sha256"),
                            "audit_sha256": audit.get("audit_sha256"),
                            "historical_audit_complete": bool(audit.get("complete")),
                            "excluded_validation_derived_training_rows": sum(
                                row.get("source_split") != "train"
                                for row in native["splits"].get("scorer_train", [])),
                            "replaced_training_row_hashes": removed,
                            "previous_confirmation_is_fresh": False,
                            "final_status": "not_selected" if training_only else "newly_selected_unopened",
                            "final_source_counts": dict(Counter(
                                f"{dataset_family(row['dataset'])}/{row['source_split']}" for row in final)),
                            "source_catalog_path": native.get("metadata", {}).get("source_catalog_path"),
                            "source_catalog_sha256": native.get("metadata", {}).get("source_catalog_sha256"),
                            "source_catalog_partial": True}}
    validate_manifest(payload, require_image_hash=False)
    return seal_manifest(payload)


def validate_manifest(payload, *, require_image_hash=True):
    group_owner, image_owner, question_owner = {}, {}, {}
    for split, rows in payload["splits"].items():
        for row in rows:
            required = {"dataset", "dataset_revision", "source_split", "question_id", "row_hash", "group_id"}
            if require_image_hash:
                required.add("image_sha256")
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{split}: missing provenance {sorted(missing)}")
            if split == "scorer_train" and (row["source_split"] != "train" or "extra" in row["dataset"].lower()):
                raise ValueError("scorer training must use official train sources only")
            question = (dataset_family(row["dataset"]), str(row["question_id"]))
            if question in question_owner:
                raise ValueError(f"duplicate question identity: {question}")
            question_owner[question] = split
            for key, owner in ((row["group_id"], group_owner), (row.get("image_sha256"), image_owner)):
                if key and key in owner and owner[key] != split:
                    raise ValueError(f"document/image group overlaps splits: {key}")
                if key:
                    owner[key] = split
    return True


def finalize_image_provenance(payload, project_root):
    """Add selected images' hashes, reading no answer/question columns."""
    import pyarrow.parquet as pq

    payload = copy.deepcopy(payload)
    root = Path(project_root).resolve()
    pending = {}
    for rows in payload["splits"].values():
        for row in rows:
            if not row.get("image_sha256"):
                pending.setdefault(row["source_file"], []).append(row)
    import hashlib
    for source_file, rows in sorted(pending.items()):
        path = root / source_file
        digest = file_sha256(path)
        table = pq.read_table(path, columns=["image"])
        for row in rows:
            if row.get("source_shard_sha256") and row["source_shard_sha256"] != digest:
                raise ValueError(f"source shard changed: {source_file}")
            image = table["image"][int(row["source_row_index"])].as_py()
            if not image.get("bytes"):
                raise ValueError(f"source image has no bytes: {source_file}")
            row["source_shard_sha256"] = digest
            row["image_sha256"] = hashlib.sha256(image["bytes"]).hexdigest()
    validate_manifest(payload)
    return seal_manifest(payload)
