"""Read-only protocol validation, independent of PyTorch."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

OFFICIAL_TRAIN_DATASETS = {
    "pixparse/docvqa-single-page-questions", "lmms-lab/textvqa",
}
DEVELOPMENT_SPLITS = {"scorer_screening", "scorer_selection", "scorer_dev",
                      "development", "controller_calibration"}


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path, expected_training_count=5600):
    path = Path(path)
    payload = json.loads(path.read_text())
    splits = payload.get("splits", {})
    training = splits.get("scorer_train", [])
    if len(training) != expected_training_count:
        raise ValueError(f"expected {expected_training_count} official training rows, found {len(training)}")
    for entry in training:
        if entry.get("dataset") not in OFFICIAL_TRAIN_DATASETS or entry.get("source_split") != "train":
            raise ValueError("scorer training permits only official DocVQA/TextVQA train; validation/DocVQA-extra forbidden")
    if expected_training_count == 5600:
        counts = {name: sum(entry["dataset"] == name for entry in training)
                  for name in OFFICIAL_TRAIN_DATASETS}
        if set(counts.values()) != {2800}:
            raise ValueError("training requires 2800 official rows per dataset")
    owners = {}
    required = {"dataset", "dataset_index", "dataset_revision", "source_split",
                "question_id", "image_id", "image_sha256", "row_hash"}
    for split, entries in splits.items():
        rows = set()
        for entry in entries:
            if required - entry.keys():
                raise ValueError(f"missing manifest provenance fields in {split}")
            if entry["row_hash"] in rows:
                raise ValueError(f"duplicate row in {split}")
            rows.add(entry["row_hash"])
            identities = [("image_sha256", entry["image_sha256"]),
                          ("row_hash", entry["row_hash"]),
                          (entry["dataset"] + ":image", str(entry["image_id"]))]
            if entry.get("group_id"):
                identities.append((entry["dataset"] + ":group", str(entry["group_id"])))
            for key in identities:
                if not key[1]:
                    raise ValueError("empty identity in manifest")
                if key in owners and owners[key] != split:
                    raise ValueError(f"image/document/row overlap between {owners[key]} and {split}")
                owners[key] = split
    metadata = payload.get("metadata", {})
    return SimpleNamespace(splits=splits, metadata=metadata, dataset=payload.get("dataset", "mixed"),
                           file_sha256=digest_file(path), path=str(path), payload=payload)


def entries_for_split(manifest, split, *, training=False):
    if training and split != "scorer_train":
        raise ValueError("label extraction/training is restricted to scorer_train")
    if not training and split not in DEVELOPMENT_SPLITS:
        raise ValueError("evaluation CLI is development-only; confirmation/final/reserve labels are closed")
    entries = manifest.splits.get(split)
    if not entries:
        raise ValueError(f"empty or missing allowed split: {split}")
    return sorted(entries, key=lambda entry: (entry["dataset"], entry["row_hash"]))


def native_resolution(adapter, model_config, requested_max_visual_patches):
    if adapter == "llava_next":
        grid = getattr(model_config, "image_grid_pinpoints", None)
        if not grid or grid == [[336, 336]] or grid == [(336, 336)]:
            raise ValueError("LLaVA-NeXT requires its native any-resolution grid, not the 336-only shortcut")
        return {"image_grid_pinpoints": [list(pair) for pair in grid]}
    if adapter == "qwen2_5_vl":
        if requested_max_visual_patches <= 0:
            raise ValueError("invalid Qwen visual-token limit")
        return {"max_visual_patches": requested_max_visual_patches}
    raise ValueError("MLSys scorer experiments support only Qwen2.5-VL and LLaVA-NeXT")
