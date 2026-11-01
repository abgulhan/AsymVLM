"""Audited multi-source data for the adaptive-budget v3 development study.

Preparation selects identities, never uses answers or predictions for selection.
Previously examined rows and incomplete historical audits remain explicitly
development evidence; this module never promotes them to fresh confirmation.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from learned_score.mlsys_eval.audit import row_identity_keys
from learned_score.mlsys_eval.common import (
    canonical_hash, dataset_family as native_dataset_family, file_sha256,
    load_manifest as load_sealed_manifest, seal_manifest, write_new_json,
)


DEFAULT_OUTPUT = Path("reproduction_results/mlsys_adaptive_20261016_v3")
BASE_MANIFEST = Path("reproduction_results/mlsys_20261012/manifest.json")
V1_CALIBRATION = Path("reproduction_results/mlsys_adaptive_vision_20261015_v1/qwen25/calibration")
HISTORY_AUDIT = Path("reproduction_results/mlsys_20261012/resolved_historical_audit.json")
METRICS = {"DocVQA": "anls", "TextVQA": "vqa", "ChartQA_TEST": "relaxed", "MME": "mme"}


def dataset_family(name):
    family = native_dataset_family(name)
    return {"chartqa_test": "ChartQA_TEST", "mme": "MME"}.get(family, family)


def _order(entry, salt):
    return canonical_hash([salt, entry["dataset"], entry["row_hash"]])


def _decorate(entry, stage):
    result = dict(entry)
    result["metric"] = METRICS[dataset_family(entry["dataset"])]
    result["quality_metric"] = result["metric"]
    result["evaluation_stage"] = stage
    return result


def _split_keys(row):
    """Cross-dataset image hashes, but dataset-qualified numeric question IDs."""
    family = dataset_family(row["dataset"])
    result = {f"{key}:{row[key]}" for key in ("row_hash", "image_sha256", "group_id") if row.get(key)}
    for key in ("question_id", "image_id", "document_id"):
        if row.get(key) is not None:
            result.add(f"{family}:{key}:{row[key]}")
    return result


def select_rows(rows, count, *, salt, excluded=(), one_per_group=False):
    """Deterministic identity-only selection; never use a quality field."""
    selected, forbidden = [], set(excluded)
    for row in sorted(rows, key=lambda item: _order(item, salt)):
        keys = row_identity_keys(row)
        if keys & forbidden:
            continue
        selected.append(dict(row))
        if one_per_group:
            forbidden.update(keys)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"insufficient disjoint candidates: {len(selected)}/{count}")
    return selected


def select_mme_pairs(rows, pairs_per_category=8):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["category"], row["group_id"])].append(row)
    categories = sorted({category for category, _ in groups})
    selected = []
    for category in categories:
        candidates = [(group, members) for (cat, group), members in groups.items()
                      if cat == category and len(members) == 2]
        candidates.sort(key=lambda item: canonical_hash(["adaptive-v3-mme", category, item[0]]))
        if len(candidates) < pairs_per_category:
            raise ValueError(f"MME category {category} lacks {pairs_per_category} complete image pairs")
        for _, members in candidates[:pairs_per_category]:
            selected.extend(sorted(members, key=lambda row: row["row_hash"]))
    return selected


def _identity_tsv(path, dataset):
    # Conversion excludes question, answer and base64 image columns entirely.
    import pandas as pd
    columns = ["index", "image_path"] + (["category"] if dataset == "MME" else [])
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=str, keep_default_na=False)
    digest = file_sha256(path)
    rows = []
    for position, raw in enumerate(frame.to_dict("records")):
        category = raw.get("category", "")
        source_image = raw["image_path"]
        group = f"{dataset}:image:" + canonical_hash([category, source_image])
        row = {"dataset": dataset, "dataset_revision": digest, "source_split": "test",
               "dataset_index": int(raw["index"]), "question_id": raw["index"],
               "image_id": source_image, "group_id": group,
               "group_provenance": "category_and_source_image_path",
               "source_format": "tsv", "source_file": str(path.resolve()),
               "source_row_index": position, "source_shard_sha256": digest,
               "source_image_path": source_image, "category": category}
        row["row_hash"] = canonical_hash(row)
        rows.append(row)
    return rows


def _resolve_tsv_image(row, root):
    raw = Path(row["source_image_path"])
    base = root / ".data/LMUData/images" / row["dataset"]
    candidates = [raw if raw.is_absolute() else base / raw,
                  base / f"{row['dataset_index']}.jpg",
                  base / f"{row['dataset_index']}.png"]
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            if not resolved.is_relative_to(root.resolve()):
                raise ValueError("benchmark image escapes project root")
            return resolved
    raise FileNotFoundError(f"missing materialized {row['dataset']} image {row['dataset_index']}")


def _hash_tsv_images(rows, root):
    for row in rows:
        path = _resolve_tsv_image(row, root)
        row["image_path"] = str(path)
        row["image_sha256"] = file_sha256(path)
        if row["dataset"] == "ChartQA_TEST":
            # Materialized ChartQA filenames include question indices; two
            # questions can therefore point to identical image bytes at
            # different paths. Cluster those questions together in reporting.
            row["group_id"] = "ChartQA_TEST:image:" + row["image_sha256"]
            row["group_provenance"] = "image_sha256"


def _fresh_text_candidates(root, base, history, count):
    from learned_score.mlsys_eval.protocol import source_inventory, finalize_image_provenance
    native = load_sealed_manifest(root / "reproduction_results/native_protocol/manifest.json")
    sources, _ = source_inventory(root, native)
    excluded = set(history.get("identity_keys", []))
    for rows in base["splits"].values():
        for row in rows:
            excluded.update(row_identity_keys(row))
    candidates = [row for row in sources if dataset_family(row["dataset"]) == "TextVQA"
                  and row["source_split"] == "train"]
    chosen = select_rows(candidates, count, salt="adaptive-v3-text-evaluation",
                         excluded=excluded, one_per_group=True)
    for row in chosen:
        row["provenance_role"] = "unused_in_resolved_history_but_global_audit_incomplete"
    # Image bytes are read for provenance, not question/answer columns.
    sealed = finalize_image_provenance({"splits": {"evaluation": chosen}}, root)
    return sealed["splits"]["evaluation"]


def validate_payload(payload, original_training):
    splits = payload["splits"]
    required = {"dataset", "group_id", "row_hash", "image_sha256", "metric",
                "source_file", "source_shard_sha256", "source_row_index"}
    train_keys = set().union(*(_split_keys(row) for row in original_training))
    for name, rows in splits.items():
        seen = set()
        for row in rows:
            if required - row.keys():
                raise ValueError(f"missing data provenance in {name}: {required - row.keys()}")
            if row["row_hash"] in seen:
                raise ValueError(f"duplicate row within {name}")
            seen.add(row["row_hash"])
            if row["metric"] not in {"anls", "vqa", "relaxed", "mme"}:
                raise ValueError("unknown benchmark metric")
            if name != "phi_train" and _split_keys(row) & train_keys:
                raise ValueError(f"scorer-training identity overlaps {name}")
    fit_keys = set().union(*(_split_keys(row) for row in splits["fit"]))
    for name in ("screening", "evaluation"):
        for row in splits[name]:
            if _split_keys(row) & fit_keys:
                raise ValueError(f"fit identity overlaps {name}")
    original_by_hash = {row["row_hash"]: row for row in original_training}
    for row in splits["phi_train"]:
        source = original_by_hash.get(row["row_hash"])
        if source is None or source["source_split"] != "train":
            raise ValueError("Phi training must be a subset of original official training identities")
    screen_ids = {row["row_hash"] for row in splits["screening"]}
    for row in splits["evaluation"]:
        if row["row_hash"] in screen_ids and row["evaluation_stage"] != "development_reused_screening":
            raise ValueError("screening reuse must be explicitly labeled development")
    return True


def prepare(output_root=DEFAULT_OUTPUT, *, project_root=None, evaluation_count=120,
            mme_pairs_per_category=8, phi_per_dataset=256):
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    output_root = Path(output_root)
    if not output_root.is_absolute():
        output_root = root / output_root
    base_path = root / BASE_MANIFEST
    base = load_sealed_manifest(base_path)
    calibration = root / V1_CALIBRATION
    policy = json.loads((calibration / "policy.json").read_text())
    screen = json.loads((calibration / "screening_rows.json").read_text())
    fit_ids = set(policy["calibration_row_hashes"])
    screen_ids = {row["row_hash"] for row in screen}
    dev = {row["row_hash"]: row for row in base["splits"]["scorer_dev"]}
    if fit_ids & screen_ids or fit_ids | screen_ids != set(dev):
        raise ValueError("frozen fit/screening split does not partition existing development")
    fit = [_decorate(dev[key], "development_fit") for key in sorted(fit_ids)]
    screening = [_decorate(dev[key], "development_screening") for key in sorted(screen_ids)]
    history_path = root / HISTORY_AUDIT
    history = json.loads(history_path.read_text())
    doc = select_rows([row for row in screening if dataset_family(row["dataset"]) == "DocVQA"],
                      evaluation_count, salt="adaptive-v3-doc-evaluation")
    doc = [_decorate(row, "development_reused_screening") for row in doc]
    text = _fresh_text_candidates(root, base, history, evaluation_count)
    # A completed two-dataset resolver still cannot certify unrelated TSV data.
    text_stage = ("fresh_confirmation" if history.get("complete")
                  else "development_historical_audit_incomplete")
    text = [_decorate(row, text_stage) for row in text]
    chart_rows = _identity_tsv(root / ".data/LMUData/ChartQA_TEST.tsv", "ChartQA_TEST")
    chart = select_rows(chart_rows, evaluation_count, salt="adaptive-v3-chart-evaluation",
                        one_per_group=True)
    mme_rows = _identity_tsv(root / ".data/LMUData/MME.tsv", "MME")
    mme = select_mme_pairs(mme_rows, mme_pairs_per_category)
    _hash_tsv_images(chart + mme, root)
    chart = [_decorate(row, "development_transfer_historical_status_unverified") for row in chart]
    mme = [_decorate(row, "development_transfer_historical_status_unverified") for row in mme]
    training = base["splits"]["scorer_train"]
    phi = []
    for family in ("DocVQA", "TextVQA"):
        selected = select_rows([row for row in training if dataset_family(row["dataset"]) == family],
                               phi_per_dataset, salt="adaptive-v3-phi-train", one_per_group=True)
        phi.extend(_decorate(row, "official_scorer_training") for row in selected)
    evaluation = sorted(doc + text + chart + mme, key=lambda row: (row["dataset"], row["row_hash"]))
    splits = {"fit": fit, "screening": screening, "evaluation": evaluation, "phi_train": phi}
    metadata = {"source_catalog_path": base["metadata"].get("source_catalog_path"),
                "source_catalog_sha256": base["metadata"].get("source_catalog_sha256"),
                "original_training_manifest": str(base_path),
                "original_training_manifest_sha256": file_sha256(base_path),
                "policy_sha256": file_sha256(calibration / "policy.json"),
                "screening_ids_sha256": file_sha256(calibration / "screening_rows.json"),
                "historical_audit_path": str(history_path),
                "historical_audit_sha256": file_sha256(history_path),
                "historical_audit_complete": bool(history.get("complete")),
                "historical_unresolved_artifacts": len(history.get("unresolved", [])),
                "counts": {name: len(rows) for name, rows in splits.items()},
                "evaluation_counts": dict(Counter(dataset_family(row["dataset"]) for row in evaluation)),
                "evaluation_status": {"DocVQA": "development_reused_screening", "TextVQA": text_stage,
                                      "ChartQA_TEST": "development_transfer_historical_status_unverified",
                                      "MME": "development_transfer_historical_status_unverified"},
                "mme_pairs_per_category": mme_pairs_per_category,
                "phi_training_per_dataset": phi_per_dataset,
                "label_selection": "identity-only; no quality/prediction/answer outcome used",
                "deadline_utc": "2026-10-17T02:35:06Z"}
    payload = seal_manifest({"format": "mlsys_adaptive_data_v3", "dataset": "ASYMVLM_ADAPTIVE_V3",
                             "splits": splits, "metadata": metadata})
    validate_payload(payload, training)
    path = output_root / "data/manifest.json"
    write_new_json(path, payload)
    summary = {"manifest": str(path), "manifest_sha256": file_sha256(path), **metadata}
    write_new_json(output_root / "data/summary.json", summary)
    return summary


def load_manifest(path):
    path = Path(path)
    payload = load_sealed_manifest(path)
    if payload.get("format") != "mlsys_adaptive_data_v3":
        raise ValueError("not an adaptive v3 manifest")
    return SimpleNamespace(payload=payload, splits=payload["splits"], metadata=payload["metadata"],
                           file_sha256=file_sha256(path), path=str(path), dataset=payload["dataset"])


class ManifestDatasetV3:
    name = "ASYMVLM_ADAPTIVE_V3"

    def __init__(self, manifest_path, allowed_splits=("fit", "screening", "evaluation"), *, project_root=None):
        from learned_score.mlsys_eval.datasets import EntryDataset
        self.manifest = load_manifest(manifest_path)
        self.root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
        if set(allowed_splits) - self.manifest.splits.keys():
            raise ValueError("unknown permitted v3 split")
        self.allowed = {}
        for split in allowed_splits:
            for entry in self.manifest.splits[split]:
                # Evaluation can explicitly reuse screening rows with a different stage label.
                self.allowed.setdefault(entry["row_hash"], []).append(entry)
        self.native = EntryDataset(manifest_path, allowed_splits=allowed_splits, project_root=self.root)
        self._verified_sources = set()
        self._tsv_rows = {}

    def _tsv_row(self, entry):
        path = Path(entry["source_file"]).resolve()
        approved_source = (self.root / ".data/LMUData" / (entry["dataset"] + ".tsv")).resolve()
        # Existing MME is an explicitly named read-only link to the shared
        # VLMEvalKit dataset. Permit that exact target, not arbitrary siblings.
        if not path.is_relative_to(self.root) and path != approved_source:
            raise ValueError("TSV source escapes project root")
        if path not in self._verified_sources:
            if file_sha256(path) != entry["source_shard_sha256"]:
                raise ValueError("TSV source hash mismatch")
            self._verified_sources.add(path)
        key = (str(path), entry["source_row_index"])
        if key not in self._tsv_rows:
            # Parse labels only for explicitly permitted row offsets; skip all
            # other TSV lines as bytes. Benchmarked sources have one record/line.
            wanted = {row["source_row_index"] for choices in self.allowed.values() for row in choices
                      if row.get("source_format") == "tsv" and row["source_file"] == str(path)}
            csv.field_size_limit(64 * 1024 * 1024)
            with path.open(newline="") as handle:
                header = next(csv.reader([handle.readline()], delimiter="\t"))
                for offset, line in enumerate(handle):
                    if offset not in wanted:
                        continue
                    values = next(csv.reader([line], delimiter="\t"))
                    if len(values) != len(header):
                        raise ValueError("multiline or malformed TSV rows are not supported")
                    self._tsv_rows[(str(path), offset)] = dict(zip(header, values))
            if key not in self._tsv_rows:
                raise ValueError("selected TSV row is absent")
        raw = self._tsv_rows[key]
        if str(raw["index"]) != str(entry["dataset_index"]):
            raise ValueError("TSV row identity mismatch")
        image = Path(entry["image_path"])
        if file_sha256(image) != entry["image_sha256"]:
            raise ValueError("TSV image hash mismatch")
        return {"question": raw["question"], "answer": raw["answer"], "image": str(image)}

    def read_entry(self, entry):
        if entry not in self.allowed.get(entry["row_hash"], []):
            raise ValueError("entry outside explicitly allowed v3 splits")
        if entry.get("source_format") == "tsv":
            row = self._tsv_row(entry)
        else:
            # EntryDataset stores the last alias, so use its identical source
            # entry when screening/evaluation differ only in stage annotation.
            row = self.native.row_for_entry(self.native.allowed[entry["row_hash"]])
        return {**row, "_manifest_row_hash": entry["row_hash"], "_source_dataset": entry["dataset"],
                "group_id": entry["group_id"], "metric": entry["metric"],
                "category": entry.get("category", "")}

    row_for_entry = read_entry

    def build_prompt(self, row):
        suffix = ("\nAnswer yes or no." if row.get("metric") == "mme"
                  else "\nAnswer the question using a single word or phrase.")
        return [{"type": "image", "value": row["image"]},
                {"type": "text", "value": row["question"] + suffix}]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
