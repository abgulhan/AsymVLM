"""Lazy adapters for immutable manifest rows and both supported VLM families."""

import hashlib
import io
import json
import re
from pathlib import Path

from .common import file_sha256, load_manifest


class EntryDataset:
    name = "ASYMVLM_NATIVE_MIXTURE"

    def __init__(self, manifest_path, *, allowed_splits=("scorer_train", "scorer_dev"),
                 project_root=None):
        self.manifest_path = Path(manifest_path)
        self.payload = load_manifest(manifest_path)
        self.metadata, self.splits = self.payload["metadata"], self.payload["splits"]
        self.root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
        missing = set(allowed_splits) - self.splits.keys()
        if missing:
            raise ValueError(f"unknown permitted splits: {sorted(missing)}")
        self.allowed = {row["row_hash"]: row for name in allowed_splits for row in self.splits[name]}
        self._catalog = None
        self._source_table = None
        self._source_path = None
        self._verified = set()

    def _catalog_records(self):
        if self._catalog is not None:
            return self._catalog
        self._catalog = {}
        value = self.metadata.get("source_catalog_path")
        if not value:
            return self._catalog
        path = Path(value)
        if not path.is_absolute():
            path = self.root / path
        expected = self.metadata.get("source_catalog_sha256")
        if expected and file_sha256(path) != expected:
            raise ValueError("source catalog hash mismatch")
        pattern = re.compile(r'"row_hash"\s*:\s*"([a-f0-9]{64})"')
        with path.open() as source:
            for line in source:
                # Select identity before decoding a record; unopened confirmation
                # answer payloads in a reused catalog are never materialized.
                match = pattern.search(line)
                if match and match[1] in self.allowed:
                    record = json.loads(line)
                    self._catalog[record["row_hash"]] = record
        return self._catalog

    def row_for_entry(self, entry):
        digest = entry["row_hash"]
        if digest not in self.allowed or dict(entry) != self.allowed[digest]:
            raise ValueError("entry is outside this dataset's explicitly permitted splits")
        record = self._catalog_records().get(digest)
        if record:
            image = Path(record["image_path"])
            if not image.is_absolute():
                image = self.root / image
            if file_sha256(image) != entry["image_sha256"]:
                raise ValueError("materialized image hash mismatch")
            question, answer, image_value = record["question"], record["answer"], str(image)
        else:
            import pyarrow.parquet as pq
            path = (self.root / entry["source_file"]).resolve()
            if not path.is_relative_to(self.root.resolve()):
                raise ValueError("source path escapes project root")
            expected = entry.get("source_shard_sha256")
            if not expected:
                raise ValueError("source shard hash is required before loading labels")
            if path not in self._verified:
                if file_sha256(path) != expected:
                    raise ValueError("source shard hash mismatch")
                self._verified.add(path)
            if self._source_path != path:
                self._source_table = pq.read_table(path, columns=["image", "question_id", "question", "answers"])
                self._source_path = path
            row = self._source_table.slice(int(entry["source_row_index"]), 1).to_pylist()[0]
            if str(row["question_id"]) != str(entry["question_id"]):
                raise ValueError("source question identity mismatch")
            image_bytes = row["image"]["bytes"]
            if hashlib.sha256(image_bytes).hexdigest() != entry["image_sha256"]:
                raise ValueError("source image hash mismatch")
            question, answer, image_value = row["question"], json.dumps(row["answers"]), io.BytesIO(image_bytes)
        return {"question": question, "answer": answer, "image": image_value,
                "_manifest_row_hash": digest, "_source_dataset": entry["dataset"],
                "group_id": entry["group_id"]}

    def build_prompt(self, row):
        return [{"type": "image", "value": row["image"]},
                {"type": "text", "value": row["question"] + "\nAnswer the question using a single word or phrase."}]
