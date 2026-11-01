"""Versioned, independent-domain, identity-only admission for compression v9.

No old manifest is changed. Unresolved source scope remains a blocker, not an
invitation to reuse examples. Downloads, identity admission, and label-bearing
runtime materialization are separate phases. All writable state lives in v9.
"""

import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import stat
import time
import zipfile
import xml.etree.ElementTree as ET

from learned_score.mlsys_eval import audit as old_audit, historical_context, historical_resolver, protocol
from learned_score.mlsys_eval.common import canonical_hash, file_sha256, load_manifest, seal_manifest, write_new_json
from . import adaptive_data_v3 as data
from .state import atomic_json


FORMAT = "mlsys_compression_data_v9"
DEFAULT_ROOT = Path("reproduction_results/mlsys_compression_20261018_v9")
FAMILIES = ("DocVQA", "TextVQA", "ChartQA_TEST", "MME")
COUNTS = {"evaluation": 1000, "screening": 1000, "fit": 400}
FORBIDDEN = frozenset({"question", "query", "answer", "answers", "label", "prediction", "quality_score", "references"})
PREVIEW_IMAGES = frozenset({"10095.png", "10149.png", "10154.png", "10183.png", "10184.png"})
DOC_REPO = "pixparse/docvqa-single-page-questions"
DOC_REV = "33136ef456fa5a3fe68568d6e31dda4eeff95b9b"
TEXT_REPO = "lmms-lab/textvqa"
TEXT_REV = "9c0699cd19768ac5ab97568f6b3cbac4c0062884"
CHART_REPO = "ahmed-masry/ChartQA"


def seal(value):
    return {**value, "sha256": canonical_hash(value)}


def checked(value):
    if value.get("sha256") != canonical_hash({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("v9 data artifact content hash mismatch")
    return value


def checked_source(value):
    if value.get("record_sha256") != canonical_hash({k: v for k, v in value.items() if k != "record_sha256"}):
        raise ValueError("source object record content hash mismatch")
    return value


def family(name):
    value = data.dataset_family(name)
    return "ChartQA_TEST" if "chartqa" in str(value).lower() else value


def identity_only(rows):
    if any(FORBIDDEN & set(row) for row in rows):
        raise ValueError("identity admission cannot contain annotations or outcomes")
    return rows


def qualify_keys(row, *, proven_family=None):
    """Only explicitly bound datasets namespace IDs; ambiguous IDs are wildcard.

    Candidate matching also checks their wildcard aliases. Image-byte hashes
    are always global, including across differently named datasets.
    """
    domain = proven_family
    if domain is None:
        name = row.get("dataset", row.get("dataset_name", row.get("source_dataset")))
        domain = family(name) if name else None
    result = {f"{k}:{row[k]}" for k in ("row_hash", "image_sha256", "group_id") if row.get(k)}
    for field, aliases in (("question_id", ("question_id", "questionId", "questionID")),
                           ("image_id", ("image_id", "imageId", "imageID")),
                           ("document_id", ("document_id", "ucsf_document_id"))):
        for alias in aliases:
            value = row.get(alias)
            if value is None or value == "":
                continue
            value = Path(str(value)).stem if field == "image_id" else str(value)
            result.add(f"{domain or '*'}:{field}:{value}")
            if field == "image_id" and domain == "DocVQA":
                document = re.sub(r"_\d+$", "", value)
                result.add(f"DocVQA:document_id:{document}")
    return result


def candidate_keys(row):
    keys = qualify_keys(row)
    for key in list(keys):
        pieces = key.split(":", 2)
        if len(pieces) == 3 and pieces[1] in {"question_id", "image_id", "document_id"}:
            keys.add("*:" + ":".join(pieces[1:]))
    return keys


def legacy_keys(keys):
    """Never narrow old unqualified exclusions without original provenance."""
    return {"*:" + key if key.split(":", 1)[0] in {"question_id", "image_id", "document_id"} else key for key in keys}


def _validate_output(project_root, output_root):
    root = Path(project_root).resolve()
    out = Path(output_root)
    out = (root / out).resolve() if not out.is_absolute() else out.resolve()
    required = root / DEFAULT_ROOT
    if out != required and required not in out.parents:
        raise ValueError("v9 data output must be inside the dedicated v9 working directory")
    if out == root or any(p.is_symlink() for p in [out, *out.parents] if p != root and root in p.parents):
        raise ValueError("unsafe v9 writable directory")
    return root, out


def _read_audit(path, hash_field="audit_sha256"):
    value = json.loads(Path(path).read_text())
    if value.get(hash_field) != canonical_hash({k: v for k, v in value.items() if k != hash_field}):
        raise ValueError(f"historical audit changed: {path}")
    return value


def _metadata_scope(path):
    """Read only recognized dataset-bearing log lines, never response lines."""
    names, evidence = historical_context.explicit_context(path)
    candidates = [path.with_suffix(".txt"), path.with_suffix(".log"), path.parent / "rank0.log"]
    patterns = [re.compile(r"^Dataset:\s*([A-Za-z0-9_./-]+)(?:,|\s|$)"),
                re.compile(r"^file\.py: Loading .*/([A-Za-z0-9_-]+)\.tsv\s*$")]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        matches = []
        with candidate.open(errors="replace") as stream:
            for number, line in enumerate(stream, 1):
                for pattern in patterns:
                    match = pattern.search(line)
                    if match:
                        names.add(match[1])
                        matches.append({"line": number, "dataset": match[1]})
        if matches:
            evidence.append({"path": str(candidate), "sha256": file_sha256(candidate), "matches": matches})
    if path.suffix in {".json", ".jsonl"} and path.is_file():
        explicit = set()
        with path.open(encoding="utf-8") as source:
            for record in DatasetScopeReader(source).records():
                explicit.update(str(value) for value in record.values())
        if explicit:
            names.update(explicit)
            evidence.append({"path": str(path), "sha256": file_sha256(path),
                             "explicit_dataset_fields": sorted(explicit), "annotations_read": False})
    if path.suffix == ".xlsx" and path.is_file():
        records = xlsx_identity_rows(path)
        explicit = {str(record[key]) for record in records for key in
                    ("dataset", "dataset_name", "source_dataset") if record.get(key)}
        path_domains = set()
        for record in records:
            for key in ("source_file", "image_path", "source_image_path"):
                value = record.get(key, "")
                # Only literal source directory components, never workbook names.
                components = set(Path(str(value)).parts)
                path_domains.update(components & {"MME", "MMLongBench_DOC", "MMDU", "DocVQA_VAL", "TextVQA_VAL", "ChartQA_TEST"})
        if explicit or path_domains:
            names.update(explicit | path_domains)
            evidence.append({"path": str(path), "sha256": file_sha256(path), "identity_records": len(records),
                             "identity_projection_sha256": canonical_hash(records),
                             "explicit_dataset_fields": sorted(explicit), "explicit_source_path_domains": sorted(path_domains),
                             "columns_read": sorted({key for record in records for key in record}), "annotations_read": False})
    return sorted({family(name) for name in names}), evidence


def xlsx_identity_rows(path):
    """Read only allowlisted identity columns from zipped workbook XML.

    Worksheet cells outside identity columns and unreferenced shared strings
    are discarded. No pickle is deserialized and no prediction column is read.
    """
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    allowed = {"index", "dataset_index", "question_id", "image_id", "image_path", "source_image_path",
               "source_file", "dataset", "dataset_name", "source_dataset", "group_id", "category"}
    with zipfile.ZipFile(path) as archive:
        members = safe_zip_members(archive)
        sheets = sorted(name for name in members if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name))

        def shared_values(indices):
            if not indices:
                return {}
            if "xl/sharedStrings.xml" not in members:
                raise ValueError("workbook refers to missing shared-string table")
            found, index = {}, 0
            with archive.open("xl/sharedStrings.xml") as source:
                for _, element in ET.iterparse(source, events=("end",)):
                    if element.tag == namespace + "si":
                        if index in indices:
                            found[index] = "".join(e.text or "" for e in element.iter(namespace + "t"))
                        element.clear()
                        index += 1
            if set(found) != set(indices):
                raise ValueError("workbook shared-string index out of range")
            return found

        def cells(element):
            projected = {}
            for cell in element.findall(namespace + "c"):
                column = re.sub(r"\d+$", "", cell.attrib.get("r", ""))
                kind = cell.attrib.get("t")
                if kind == "inlineStr":
                    value = "".join(item.text or "" for item in cell.iter(namespace + "t"))
                else:
                    value = cell.findtext(namespace + "v", default="")
                projected[column] = (kind, value)
            return projected

        result = []
        for sheet in sheets:
            with archive.open(sheet) as source:
                header = next((cells(element) for _, element in ET.iterparse(source, events=("end",))
                               if element.tag == namespace + "row"), {})
            shared_header = shared_values({int(value) for kind, value in header.values() if kind == "s"})
            header = {column: shared_header[int(value)] if kind == "s" else value
                      for column, (kind, value) in header.items()}
            selected_columns = {column: name for column, name in header.items() if name in allowed}
            projected, needed = [], set()
            with archive.open(sheet) as source:
                first = True
                for _, element in ET.iterparse(source, events=("end",)):
                    if element.tag != namespace + "row":
                        continue
                    if first:
                        first = False
                    else:
                        record = {}
                        for cell in element.findall(namespace + "c"):
                            column = re.sub(r"\d+$", "", cell.attrib.get("r", ""))
                            if column not in selected_columns:
                                continue
                            kind = cell.attrib.get("t")
                            value = ("".join(item.text or "" for item in cell.iter(namespace + "t"))
                                     if kind == "inlineStr" else cell.findtext(namespace + "v", default=""))
                            record[selected_columns[column]] = (kind, value)
                            if kind == "s":
                                needed.add(int(value))
                        if record:
                            projected.append(record)
                    element.clear()
            shared = shared_values(needed)
            result.extend({name: shared[int(value)] if kind == "s" else value for name, (kind, value) in record.items()}
                          for record in projected)
        return result


def _revalidate_context_record(record, root, original_inventory):
    path = root / record["path"]
    if file_sha256(path) != record["sha256"]:
        raise ValueError("context-resolved source changed")
    if original_inventory.get(record["path"], {}).get("sha256") != record["sha256"]:
        raise ValueError("context record not bound to original audit")
    for pin in record.get("evidence", []) + record.get("audited_siblings", []):
        if file_sha256(root / pin["path"]) != pin["sha256"]:
            raise ValueError("context evidence changed")
    if record.get("producer"):
        pin = record["producer"]
        if file_sha256(root / pin["path"]) != pin["sha256"]:
            raise ValueError("derived historical producer changed")


def history_inventory(root):
    """Revalidate previous refinement and retain domain-local unresolved scope.

    Old exclusion keys stay conservative. Supplemental unknown binary artifacts
    are NOT inferred to be unrelated merely from a filename and are NOT unpickled.
    A caller can see exactly which missing provenance prevents each admission.
    """
    root = Path(root)
    base = root / "reproduction_results/mlsys_20261012"
    original = _read_audit(base / "historical_identity_audit.json")
    primary = _read_audit(base / "resolved_historical_audit.json")
    context = _read_audit(base / "historical_context_status.json", "status_sha256")
    if primary["input_audit_sha256"] != original["audit_sha256"] or context["primary_audit_sha256"] != primary["audit_sha256"]:
        raise ValueError("historical audit chain mismatch")
    original_inventory = {row["path"]: row for row in original["files"]}
    resolved, issues, keys = [], [], legacy_keys(primary["identity_keys"])
    for item in context["additional_resolved_files"]:
        _revalidate_context_record(item, root, original_inventory)
        resolved.append(copy.deepcopy(item))
    resolved_paths = {item["path"] for item in resolved}
    for item in primary["unresolved"]:
        if item["path"] in resolved_paths:
            continue
        path = root / item["path"]
        expected = original_inventory.get(item["path"], {}).get("sha256")
        if not expected or file_sha256(path) != expected:
            raise ValueError("unresolved historical artifact changed")
        domains, evidence = _metadata_scope(path)
        issues.append({"path": str(path), "sha256": expected, "possible_domains": domains or list(FAMILIES),
                       "reason": "unresolved_identity_indices", "metadata_evidence": evidence})
    supplemental_pins = []
    for entry in context["supplementary_scopes"]:
        path = root / entry["path"]
        audit = _read_audit(path)
        if audit["audit_sha256"] != entry["audit_sha256"]:
            raise ValueError("supplementary audit changed")
        supplemental_pins.append({"path": str(path), "sha256": file_sha256(path)})
        keys.update(legacy_keys(audit.get("identity_keys", [])))
        for issue in audit.get("unresolved", []):
            source = root / issue["path"]
            names, evidence = _metadata_scope(source)
            issues.append({"path": str(source), "sha256": file_sha256(source),
                           "possible_domains": names or list(FAMILIES), "reason": "supplementary_unresolved_identity",
                           "metadata_evidence": evidence})
        for filename in audit.get("unsupported_files", []):
            source = root / filename
            names, evidence = _metadata_scope(source)
            issues.append({"path": str(source), "sha256": file_sha256(source),
                           "possible_domains": names or list(FAMILIES), "reason": "unsupported_identity_archive",
                           "metadata_evidence": evidence})
    pins = {str(base / name): file_sha256(base / name) for name in
            ("historical_identity_audit.json", "resolved_historical_audit.json", "historical_context_status.json")}
    pins.update({p["path"]: p["sha256"] for p in supplemental_pins})
    # Bind raw evidence, not just an audit which once referred to that evidence.
    for item in original_inventory.values():
        path = (root / item["path"]).resolve()
        if file_sha256(path) != item["sha256"]:
            raise ValueError("original historical source changed since audit")
        pins[str(path)] = item["sha256"]
    for item in resolved:
        for pin in item.get("evidence", []) + item.get("audited_siblings", []) + ([item["producer"]] if item.get("producer") else []):
            pins[str((root / pin["path"]).resolve())] = pin["sha256"]
    for issue in issues:
        pins[str(Path(issue["path"]).resolve())] = issue["sha256"]
        for evidence in issue.get("metadata_evidence", []):
            pins[str((root / evidence["path"]).resolve())] = evidence["sha256"]
    return seal({"format": FORMAT, "labels_loaded": False, "identity_keys": sorted(keys),
                 "resolved_context_artifacts": len(resolved), "resolved_evidence": resolved,
                 "issues": issues, "input_hashes": pins,
                 "scope_note": "Unknown scope remains a blocker for every candidate domain; no filename-only scope inference."})


def group_closure(candidates, exclusions):
    """An excluded question excludes every question in its document/image."""
    identity_only(candidates)
    forbidden = set(exclusions)
    bad = {(family(row["dataset"]), row["group_id"]) for row in candidates if candidate_keys(row) & forbidden}
    return [row for row in candidates if (family(row["dataset"]), row["group_id"]) not in bad], bad


def assess_domains(candidates, history, exclusions=(), *, counts=COUNTS):
    checked(history)
    identity_only(candidates)
    keys = set(history["identity_keys"])
    for row in identity_only(list(exclusions)):
        keys.update(qualify_keys(row))
    remaining, closed = group_closure(candidates, keys)
    domains = {}
    for domain in FAMILIES:
        source = [r for r in candidates if family(r["dataset"]) == domain]
        available = [r for r in remaining if family(r["dataset"]) == domain]
        groups = {r["group_id"] for r in available}
        blockers = [item for item in history["issues"] if domain in item["possible_domains"]]
        reasons = ([{"code": "unresolved_history", "artifacts": len(blockers)}] if blockers else [])
        if domain == "MME":
            reasons.append({"code": "mme_no_new_official_split", "detail": "Existing MME is retained as development/transfer; no fresh admission is fabricated."})
        elif len(groups) < counts["evaluation"] + counts["screening"]:
            reasons.append({"code": "insufficient_independent_groups", "available": len(groups),
                            "required": counts["evaluation"] + counts["screening"]})
        domains[domain] = {"status": "blocked" if reasons else "eligible_pending_image_audit", "eligible": not reasons,
                           "source_rows": len(source), "source_groups": len({r["group_id"] for r in source}),
                           "remaining_rows": len(available), "remaining_groups": len(groups),
                           "group_closed_exclusions": sum(group[0] == domain for group in closed),
                           "blocking_reasons": reasons, "unresolved_artifacts": blockers}
    return seal({"format": FORMAT, "labels_loaded": False, "domains": domains,
                 "history_sha256": history["sha256"], "candidate_sha256": canonical_hash(candidates),
                 "exclusion_sha256": canonical_hash(sorted(keys)), "requested_counts": counts}), remaining


def select_groups(rows, *, counts=COUNTS, excluded_image_hashes=()):
    """Reserve final first, then representative calibration, then fitting.

    Call only after byte-hashing every eligible image. Duplicate image bytes
    cannot occur in different groups or roles. One question per group is kept.
    """
    identity_only(rows)
    if any(not row.get("image_sha256") for row in rows):
        raise ValueError("image-byte provenance is required before split reservation")
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["group_id"]].append(row)
    seen = set(excluded_image_hashes)
    selected = {"fit": [], "screening": [], "evaluation": [], "phi_train": []}
    ordered = sorted(grouped, key=lambda group: canonical_hash(["compression-v9-groups", group]))
    cursor = 0
    for role in ("evaluation", "screening", "fit"):
        target = counts[role]
        while cursor < len(ordered) and len(selected[role]) < target:
            members = grouped[ordered[cursor]]
            cursor += 1
            hashes = {r["image_sha256"] for r in members}
            if hashes & seen:
                continue
            pick = min(members, key=lambda r: canonical_hash(["compression-v9-question", r["row_hash"]]))
            selected[role].append(dict(pick))
            seen.update(hashes)
        if role != "fit" and len(selected[role]) != target:
            raise ValueError(f"insufficient independent images after deduplication for {role}")
    return selected


def _native_rows(root, extra_doc=None):
    native = load_manifest(root / "reproduction_results/native_protocol/manifest.json")
    rows, files = protocol.source_inventory(root, native)
    if extra_doc:
        import pyarrow.parquet as pq
        offset = 0
        for path in sorted(Path(extra_doc).glob("validation-*.parquet")):
            shard_sha = file_sha256(path)
            raw_rows = pq.read_table(path, columns=["question_id", "other_metadata"]).to_pylist()
            for index, raw in enumerate(raw_rows):
                meta = raw["other_metadata"]
                doc, image = str(meta["ucsf_document_id"]), str(meta["image"])
                entry = {"dataset": DOC_REPO, "dataset_revision": DOC_REV, "dataset_index": offset + index,
                         "source_split": "validation", "question_id": str(raw["question_id"]), "image_id": image,
                         "source_file": str(path.resolve()), "source_row_index": index, "source_format": "parquet",
                         "source_shard_sha256": shard_sha, "document_id": doc, "group_id": f"DocVQA:document:{doc}",
                         "group_provenance": "ucsf_document_id"}
                entry["row_hash"] = canonical_hash(entry)
                rows.append(entry)
            offset += len(raw_rows)
            files.append({"path": str(path), "sha256": shard_sha, "rows": len(raw_rows), "columns_read": ["question_id", "other_metadata"]})
    return rows, files


class ChartIdentityReader(old_audit.IdentityJSONReader):
    """Project only image filename and question type from author JSON arrays."""

    wanted_keys = frozenset({"imgname", "image", "type"})

    def value(self, retain=False):
        self.whitespace()
        if self.peek() != "{":
            return (yield from super().value(retain))
        self.take()
        record = {}
        self.whitespace()
        if self.peek() == "}":
            self.take()
            return None
        while True:
            self.whitespace()
            key = self.string(True)
            self.whitespace()
            if self.take() != ":":
                raise ValueError("invalid ChartQA metadata object")
            keep = key in self.wanted_keys
            child = yield from self.value(keep)
            if keep and isinstance(child, (str, int)):
                record[key] = child
            self.whitespace()
            separator = self.take()
            if separator == "}":
                break
            if separator != ",":
                raise ValueError("invalid ChartQA object separator")
        if record:
            yield record
        return None


class DatasetScopeReader(ChartIdentityReader):
    """Unlike old identity reader, retain dataset-only parent metadata too."""

    wanted_keys = frozenset({"dataset", "dataset_name", "source_dataset"})


def chart_inventory(archive, *, revision):
    archive = Path(archive)
    digest = file_sha256(archive)
    rows = []
    with zipfile.ZipFile(archive) as handle:
        members = safe_zip_members(handle)
        for name in sorted(members):
            match = re.search(r"(?:^|/)(train|val)/(?:train|val)_(human|augmented)\.json$", name)
            if not match:
                continue
            split, kind = match.groups()
            with handle.open(name) as source:
                projected = ChartIdentityReader(io.TextIOWrapper(source, encoding="utf-8")).records()
                for index, raw in enumerate(projected):
                    image = str(raw.get("imgname", raw.get("image", "")))
                    if not image or Path(image).name != image:
                        raise ValueError("ChartQA source lacks a safe image filename")
                    if image in PREVIEW_IMAGES:
                        continue
                    image_member = str(Path(name).parent / "png" / image)
                    if image_member not in members:
                        raise ValueError("ChartQA JSON image is absent from archive")
                    entry = {"dataset": "ChartQA_TEST", "source_dataset": CHART_REPO, "dataset_revision": revision,
                             "source_split": "validation" if split == "val" else "train", "question_id": f"{split}:{kind}:{index}",
                             "dataset_index": index, "image_id": image, "group_id": f"ChartQA:image:{image}",
                             "group_provenance": "official_imgname_and_image_sha256", "source_format": "chartqa_archive",
                             "source_file": str(archive.resolve()), "source_row_index": index, "source_shard_sha256": digest,
                             "archive_annotation_member": name, "archive_image_member": image_member, "chart_question_type": kind}
                    entry["row_hash"] = canonical_hash(entry)
                    rows.append(entry)
    if not rows:
        raise ValueError("author archive contains no supported train/validation metadata")
    return rows


def safe_zip_members(handle):
    names = set()
    for member in handle.infolist():
        path = Path(member.filename)
        if path.is_absolute() or ".." in path.parts or member.filename in names:
            raise ValueError("unsafe or duplicate archive member")
        if stat.S_ISLNK(member.external_attr >> 16):
            raise ValueError("symbolic links are prohibited in source archives")
        names.add(member.filename)
    return names


def acquire_sources(root, out, *, deadline):
    """HF downloads resume in working storage; never use home-directory caches."""
    from huggingface_hub import HfApi, hf_hub_download
    source_root = out / "data/sources"
    source_root.mkdir(parents=True, exist_ok=True)
    api = HfApi()
    pins_path = source_root / "pins.json"
    if pins_path.exists():
        pins = checked(json.loads(pins_path.read_text()))
    else:
        chart_info = api.dataset_info(CHART_REPO, revision="main", files_metadata=True)
        doc_info = api.dataset_info(DOC_REPO, revision=DOC_REV, files_metadata=True)
        chart_files = [item.rfilename for item in chart_info.siblings if item.rfilename.lower().endswith(".zip")]
        if len(chart_files) != 1:
            raise ValueError("expected exactly one author ChartQA full archive")
        records = [{"repo_id": CHART_REPO, "revision": chart_info.sha, "filename": chart_files[0], "role": "chartqa_full"}]
        records += [{"repo_id": DOC_REPO, "revision": DOC_REV, "filename": item.rfilename, "role": "docvqa_validation"}
                    for item in doc_info.siblings if re.fullmatch(r"data/validation-.*\.parquet", item.rfilename)]
        if len(records) < 2:
            raise ValueError("pinned DocVQA source exposes no validation shards")
        pins = seal({"format": FORMAT, "records": records, "labels_loaded": False})
        write_new_json(pins_path, pins)
    completed = []
    # DocVQA shards then ChartQA; every completed object has an immutable digest.
    for item in sorted(pins["records"], key=lambda record: record["role"]):
        if time.time() >= deadline:
            break
        marker = source_root / "completed_v2" / (canonical_hash(item) + ".json")
        if marker.exists():
            record = checked_source(json.loads(marker.read_text()))
            if any(record.get(key) != value for key, value in item.items()) or file_sha256(record["path"]) != record["sha256"]:
                raise ValueError("completed download source changed")
            completed.append({k: v for k, v in record.items() if k != "record_sha256"})
            continue
        previous_status = source_root / "status.json"
        if previous_status.exists():
            previous = checked(json.loads(previous_status.read_text()))
            matched = [r for r in previous.get("records", []) if all(r.get(k) == v for k, v in item.items())]
            if len(matched) == 1 and file_sha256(matched[0]["path"]) == matched[0]["sha256"]:
                record = matched[0]
                write_new_json(marker, {**record, "record_sha256": canonical_hash(record)})
                completed.append(record)
                continue
        local = source_root / item["role"]
        path = Path(hf_hub_download(item["repo_id"], item["filename"], repo_type="dataset", revision=item["revision"],
                                    local_dir=local, cache_dir=source_root / "hub-cache"))
        record = {**item, "path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        # Object hash and seal have different fields; never overwrite the object hash.
        write_new_json(marker, {**record, "record_sha256": canonical_hash(record)})
        completed.append(record)
        atomic_json(source_root / "progress.json", {"status": "downloading", "completed": completed,
                                                     "expected": len(pins["records"]), "deadline": deadline})
        print(json.dumps({"event": "source_completed", "role": item["role"], "filename": item["filename"], "bytes": record["bytes"]}), flush=True)
    result = seal({"format": FORMAT, "status": "complete" if len(completed) == len(pins["records"]) else "deadline_reached",
                   "records": completed, "pins_sha256": pins["sha256"], "labels_loaded": False})
    atomic_json(source_root / "status.json", result)
    return result


def collect_sources(root, out):
    rows, sources = _native_rows(root, out / "data/sources/docvqa_validation/data")
    for name in ("ChartQA_TEST", "MME"):
        path = root / ".data/LMUData" / (name + ".tsv")
        rows.extend(data._identity_tsv(path, name))
        sources.append({"path": str(path), "sha256": file_sha256(path), "columns_read": ["index", "image_path", "category"]})
    completed_dir = out / "data/sources/completed_v2"
    completed = [checked_source(json.loads(marker.read_text())) for marker in sorted(completed_dir.glob("*.json"))]
    if not completed and (out / "data/sources/status.json").exists():
        completed = checked(json.loads((out / "data/sources/status.json").read_text())).get("records", [])
    for record in completed:
        path = Path(record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError("completed source changed")
        if record["role"] == "chartqa_full":
            rows.extend(chart_inventory(path, revision=record["revision"]))
            sources.append(record)
    return identity_only(rows), sources


def frozen_exclusions(root):
    paths = ["reproduction_results/native_protocol/manifest.json", "reproduction_results/mlsys_20261012/manifest.json",
             "reproduction_results/mlsys_adaptive_20261016_v3/data/manifest.json",
             "reproduction_results/mlsys_adaptive_20261016_v4/data/manifest.json"]
    exclusions = []
    for name in paths:
        manifest = load_manifest(root / name)
        exclusions.extend(row for split in manifest["splits"].values() for row in split)
    return identity_only(exclusions)


def enrich_images(rows, root, out):
    """Only image bytes and source metadata are accessed here."""
    rows = copy.deepcopy(rows)
    native = [r for r in rows if r["source_format"] == "parquet"]
    if native:
        enriched = protocol.finalize_image_provenance({"splits": {"pending": native}}, root)["splits"]["pending"]
        index = {r["row_hash"]: r for r in enriched}
        rows = [index.get(r["row_hash"], r) for r in rows]
    tsv = [r for r in rows if r["source_format"] == "tsv"]
    data._hash_tsv_images(tsv, root)
    archives = defaultdict(list)
    for row in rows:
        if row["source_format"] == "chartqa_archive":
            archives[row["source_file"]].append(row)
    for filename, selected in archives.items():
        with zipfile.ZipFile(filename) as handle:
            safe_zip_members(handle)
            hashes = {}
            for row in selected:
                member = row["archive_image_member"]
                if member not in hashes:
                    digest = hashlib.sha256()
                    with handle.open(member) as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                    hashes[member] = digest.hexdigest()
                row["image_sha256"] = hashes[member]
    return rows


def _prepare(project_root, output_root=DEFAULT_ROOT, *, download=False, deadline=None, counts=COUNTS):
    root, out = _validate_output(project_root, output_root)
    config_path = out / "data/admission_config.json"
    if config_path.exists():
        config = checked(json.loads(config_path.read_text()))
        if config["counts"] != counts:
            raise ValueError("admission counts cannot change after reservation starts")
    else:
        config = seal({"format": FORMAT, "created": time.time(), "deadline": deadline or time.time() + 4 * 3600,
                       "counts": counts, "labels_loaded": False, "preview_chart_images_excluded": sorted(PREVIEW_IMAGES)})
        write_new_json(config_path, config)
    if download and time.time() < config["deadline"]:
        acquire_sources(root, out, deadline=config["deadline"])
    history = history_inventory(root)
    write_new_json(out / "data/audits" / (history["sha256"] + ".json"), history)
    rows, source_files = collect_sources(root, out)
    exclusions = frozen_exclusions(root)
    # Prior resolution excluded every named legacy DocVQA validation source.
    # Newly downloaded shards do not become fresh just because the old local
    # source inventory lacked those shards. Exact historical reconstruction is
    # required to replace this conservative exclusion in a future audit.
    exclusions += [row for row in rows if family(row["dataset"]) == "DocVQA" and row["source_split"] == "validation"]
    assessment, remaining = assess_domains(rows, history, exclusions, counts=counts)
    manifests = {}
    domains = copy.deepcopy(assessment["domains"])
    old_images = {r["image_sha256"] for r in exclusions if r.get("image_sha256")}
    old_images.update(key.split(":", 1)[1] for key in history["identity_keys"] if key.startswith("image_sha256:"))
    for domain in FAMILIES:
        if not domains[domain]["eligible"]:
            continue
        identity_path = out / "data" / domain / "identity_manifest.json"
        try:
            if identity_path.exists():
                manifest = load_manifest(identity_path)
                if manifest["metadata"]["audit_sha256"] != assessment["sha256"]:
                    raise ValueError("live audit changed after identity split freeze")
            else:
                if time.time() >= config["deadline"]:
                    raise ValueError("data admission deadline reached")
                available = [r for r in remaining if family(r["dataset"]) == domain]
                # Exclude inspected ChartQA preview groups and byte matches to ALL old TSV images.
                if domain == "ChartQA_TEST":
                    old_chart = [r for r in rows if family(r["dataset"]) == domain and r["source_format"] == "tsv"]
                    old_images.update(r["image_sha256"] for r in enrich_images(old_chart, root, out))
                    available = [r for r in available if r["source_format"] == "chartqa_archive"]
                enriched = enrich_images(available, root, out)
                splits = select_groups(enriched, counts=counts, excluded_image_hashes=old_images)
                for role, members in splits.items():
                    for row in members:
                        row.update(metric=data.METRICS[domain], quality_metric=data.METRICS[domain],
                                   evaluation_stage="fresh_confirmation_v9" if role == "evaluation" else "fresh_calibration_v9" if role == "screening" else "additional_fit_v9")
                manifest = seal_manifest({"format": "mlsys_adaptive_data_v3", "dataset": "ASYMVLM_COMPRESSION_V9_" + domain,
                    "metadata": {"fresh_identity_audit_passed": True, "audit_sha256": assessment["sha256"],
                                 "history_sha256": history["sha256"], "selection_uses_labels": False,
                                 "source_catalog_path": None, "role_aliases": {"evaluation": "final", "screening": "calibration", "fit": "additional_fit"}},
                    "splits": splits})
                write_new_json(identity_path, manifest)
            if len(manifest["splits"]["fit"]) < 16:
                raise ValueError("at least 16 independent additional-fit groups are required")
            runtime_path = materialize_chart(identity_path, root, out) if domain == "ChartQA_TEST" else identity_path
            runtime_manifest = load_manifest(runtime_path)
            input_hashes = dict(history["input_hashes"])
            input_hashes[str(identity_path)] = file_sha256(identity_path)
            input_hashes[str(runtime_path)] = file_sha256(runtime_path)
            for members in runtime_manifest["splits"].values():
                for row in members:
                    input_hashes[row["source_file"]] = row["source_shard_sha256"]
            domains[domain].update(status="admitted", counts={k: len(v) for k, v in manifest["splits"].items()},
                                   manifest_path=str(runtime_path), manifest_sha256=file_sha256(runtime_path),
                                   fresh_identity_audit_passed=True, image_exclusions_verified=True,
                                   input_hashes=input_hashes)
            manifests[domain] = str(runtime_path)
        except (ValueError, FileNotFoundError) as error:
            domains[domain].update(status="blocked", eligible=False)
            domains[domain]["blocking_reasons"].append({"code": "image_or_source_admission_failed", "detail": str(error)})
    domains = {domain: seal(record) for domain, record in domains.items()}
    result = seal({"format": FORMAT, "status": "partially_admitted" if manifests else "blocked",
                   "domains": domains, "manifests": manifests, "labels_loaded_for_admission": False,
                   "labels_loaded": "ChartQA_TEST" in manifests, "assessment_sha256": assessment["sha256"],
                   "source_files": source_files, "history_sha256": history["sha256"], "deadline": config["deadline"]})
    write_new_json(out / "data/audits" / (result["sha256"] + ".json"), result)
    atomic_json(out / "data/status.json", result)
    return result


def prepare(project_root, output_root=DEFAULT_ROOT, *, download=False, deadline=None, counts=COUNTS):
    """Publish fail-closed status even when source verification raises.

    A stale previously eligible status must never survive a failed re-audit.
    """
    root, out = _validate_output(project_root, output_root)
    try:
        return _prepare(root, out, download=download, deadline=deadline, counts=counts)
    except (ValueError, OSError, KeyError, zipfile.BadZipFile) as error:
        issue = {"code": "data_audit_failed", "detail": str(error), "exception_type": type(error).__name__}
        result = seal({"format": FORMAT, "status": "blocked", "labels_loaded": False, "manifests": {},
            "domains": {name: seal({"status": "blocked", "eligible": False, "blocking_reasons": [issue]}) for name in FAMILIES}})
        atomic_json(out / "data/status.json", result)
        return result


def materialize_chart(manifest_path, project_root, output_root=DEFAULT_ROOT):
    """Programmatic label conversion after immutable identity reservation.

    Never prints/returns questions or answers. The CPU coordinator must still
    freeze its pre-label policy before any evaluation-stage runtime is launched.
    The TSV is one-record-per-line and is read by the established runtime only
    for explicitly permitted manifest row offsets.
    """
    root, out = _validate_output(project_root, output_root)
    identity = load_manifest(manifest_path)
    if identity["metadata"].get("fresh_identity_audit_passed") is not True:
        raise ValueError("runtime conversion requires completed identity admission")
    destination = out / "data/ChartQA_TEST/runtime_manifest.json"
    if destination.exists():
        result = load_manifest(destination)
        if result["metadata"].get("identity_manifest_sha256") != identity["manifest_sha256"]:
            raise ValueError("runtime manifest belongs to different identity reservation")
        return destination
    output = copy.deepcopy(identity)
    groups = defaultdict(list)
    for role, rows in output["splits"].items():
        for row in rows:
            if row["source_format"] != "chartqa_archive":
                raise ValueError("ChartQA materializer received another source format")
            groups[(row["source_file"], row["archive_annotation_member"])].append((role, row))
    runtime_rows = []
    for (filename, member), entries in sorted(groups.items()):
        with zipfile.ZipFile(filename) as archive:
            safe_zip_members(archive)
            if file_sha256(filename) != entries[0][1]["source_shard_sha256"]:
                raise ValueError("ChartQA archive changed before runtime materialization")
            # Conversion is a mechanical process, not policy selection or evaluation.
            annotations = json.loads(archive.read(member))
            for role, row in entries:
                raw = annotations[row["source_row_index"]]
                if raw["imgname"] != row["image_id"]:
                    raise ValueError("ChartQA annotation/image identity mismatch")
                image = out / "data/ChartQA_TEST/images" / (row["image_sha256"] + ".png")
                if not image.exists():
                    image.parent.mkdir(parents=True, exist_ok=True)
                    with image.open("xb") as target, archive.open(row["archive_image_member"]) as source:
                        shutil.copyfileobj(source, target)
                if file_sha256(image) != row["image_sha256"]:
                    raise ValueError("materialized ChartQA image hash mismatch")
                question, answer = str(raw["query"]), str(raw["label"])
                # Runtime CSV reader permits quoting but not literal embedded newlines.
                question = question.replace("\r", " ").replace("\n", " ")
                answer = answer.replace("\r", " ").replace("\n", " ")
                row["original_source_provenance"] = {k: row[k] for k in ("source_file", "source_row_index", "source_shard_sha256", "archive_annotation_member")}
                row["image_path"] = str(image)
                runtime_rows.append((row, question, answer))
    tsv = out / "data/ChartQA_TEST/runtime.tsv"
    temporary = tsv.with_suffix(".tsv.partial")
    with temporary.open("w", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(["index", "question", "answer"])
        for index, (row, question, answer) in enumerate(runtime_rows):
            writer.writerow([index, question, answer])
            row.update(source_format="tsv", source_file=str(tsv), source_row_index=index, dataset_index=index)
    if tsv.exists() and file_sha256(tsv) != file_sha256(temporary):
        raise ValueError("different ChartQA runtime TSV already exists")
    os.replace(temporary, tsv)
    digest = file_sha256(tsv)
    for row, _, _ in runtime_rows:
        row["source_shard_sha256"] = digest
    output["metadata"].update(identity_manifest_sha256=identity["manifest_sha256"],
                                annotation_materialization="mechanical_conversion_after_identity_freeze")
    write_new_json(destination, seal_manifest(output))
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["audit", "prepare", "download", "materialize-chart"], default="audit")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--deadline", type=float)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)
    root, out = _validate_output(args.root, args.output_root)
    if args.phase == "download":
        result = acquire_sources(root, out, deadline=args.deadline or time.time() + 4 * 3600)
    elif args.phase == "materialize-chart":
        if not args.manifest:
            parser.error("--manifest required for materialize-chart")
        result = {"runtime_manifest": str(materialize_chart(args.manifest, root, out))}
    else:
        result = prepare(root, out, download=args.phase == "prepare", deadline=args.deadline)
    # Keep logs compact; full identity evidence is persisted, not dumped to users.
    summary = {k: result[k] for k in ("status", "manifests", "runtime_manifest") if k in result}
    if "domains" in result:
        summary["domains"] = {domain: {k: record[k] for k in
            ("status", "eligible", "remaining_groups", "blocking_reasons", "counts") if k in record}
            for domain, record in result["domains"].items()}
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
