"""Identity-only historical audit.

The streaming JSON reader discards non-identity scalar values without decoding or
retaining them. In particular prediction, answer, quality, and annotation text
are not materialized. No pickle files are deserialized. A report explicitly
lists unsupported artifacts; these cannot establish a fresh final evaluation.
"""

import json
import re
from pathlib import Path

from .common import canonical_hash, dataset_family, file_sha256


IDENTITY_FIELDS = frozenset({
    "dataset", "dataset_name", "source_dataset", "source_split", "split",
    "question_id", "questionId", "questionID", "image_id", "imageId",
    "imageID", "doc_id", "document_id", "ucsf_document_id", "image_sha256",
    "row_hash", "sample_id", "request_id", "group_id", "index", "dataset_index",
    "source_row_index", "source_file", "pdf", "page", "url", "source_url", "id",
})


class IdentityJSONReader:
    """Incremental JSON/JSONL parser that only retains allowlisted scalars."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.position = 0

    def peek(self):
        if self.position == len(self.buffer):
            self.buffer = self.stream.read(65536)
            self.position = 0
        return self.buffer[self.position:self.position + 1]

    def take(self):
        char = self.peek()
        self.position += bool(char)
        return char

    def whitespace(self):
        while self.peek() and self.peek() in " \r\n\t":
            self.take()

    def string(self, retain):
        if self.take() != '"':
            raise ValueError("expected JSON string")
        if not retain:
            # Skip label text in buffer-sized runs without decoding or copying.
            while self.peek():
                special = re.search(r'["\\]', self.buffer[self.position:])
                if special is None:
                    self.position = len(self.buffer)
                    continue
                self.position += special.start()
                char = self.take()
                if char == '"':
                    return None
                if not self.take():
                    raise ValueError("unterminated JSON escape")
            raise ValueError("unterminated JSON string")
        chars = ['"'] if retain else None
        escaped = False
        while True:
            char = self.take()
            if not char:
                raise ValueError("unterminated JSON string")
            if retain:
                chars.append(char)
            if char == '"' and not escaped:
                return json.loads("".join(chars)) if retain else None
            if char == "\\" and not escaped:
                escaped = True
            else:
                escaped = False

    def value(self, retain=False):
        self.whitespace()
        char = self.peek()
        if char == "{":
            self.take()
            record = {}
            self.whitespace()
            if self.peek() != "}":
                while True:
                    self.whitespace()
                    key = self.string(True)
                    self.whitespace()
                    if self.take() != ":":
                        raise ValueError("expected JSON colon")
                    child = yield from self.value(key in IDENTITY_FIELDS)
                    if key in IDENTITY_FIELDS and child is not None:
                        record[key] = child
                    self.whitespace()
                    separator = self.take()
                    if separator == "}":
                        break
                    if separator != ",":
                        raise ValueError("expected JSON object separator")
            else:
                self.take()
            if any(key in record for key in IDENTITY_FIELDS - {
                "dataset", "dataset_name", "source_dataset", "source_split", "split"
            }):
                yield record
            return None
        if char == "[":
            self.take()
            self.whitespace()
            if self.peek() != "]":
                while True:
                    yield from self.value(False)
                    self.whitespace()
                    separator = self.take()
                    if separator == "]":
                        break
                    if separator != ",":
                        raise ValueError("expected JSON array separator")
            else:
                self.take()
            return None
        if char == '"':
            return self.string(retain)
        token = [] if retain else None
        while self.peek() and self.peek() not in " \t\r\n,]}":
            char = self.take()
            if retain:
                token.append(char)
        if retain:
            return json.loads("".join(token))
        return None

    def records(self):
        self.whitespace()
        while self.peek():
            yield from self.value()
            self.whitespace()


def iter_identity_records(path):
    with Path(path).open(encoding="utf-8") as source:
        yield from IdentityJSONReader(source).records()


def row_identity_keys(row):
    """Conservative cross-revision identities; source row offsets are not IDs."""
    result = set()
    for key in ("row_hash", "image_sha256", "group_id"):
        if row.get(key):
            result.add(f"{key}:{row[key]}")
    # Numeric question IDs in legacy files are sometimes unqualified. Excluding
    # globally can over-exclude, but cannot silently create a false clean split.
    for key in ("question_id", "questionId", "questionID"):
        if row.get(key) is not None:
            result.add(f"question_id:{row[key]}")
    family = dataset_family(row.get("dataset", row.get("dataset_name", "")))
    for key in ("image_id", "imageId", "imageID"):
        if row.get(key):
            image = Path(str(row[key])).stem
            result.add(f"image_id:{image}")
            # UCSF document names survive row/schema changes. Numeric final
            # suffix is a page number, not a separate independent document.
            if family == "DocVQA" or re.fullmatch(r"[A-Za-z]+\d+_\d+", image):
                result.add("document_id:" + re.sub(r"_\d+$", "", image))
    for key in ("ucsf_document_id", "document_id"):
        if row.get(key):
            result.add(f"document_id:{row[key]}")
    return result


def audit_paths(paths, *, exclude=()):
    excluded = {str(Path(path).resolve()) for path in exclude}
    files = set()
    unsupported = []
    for root in map(Path, paths):
        candidates = root.rglob("*") if root.is_dir() else [root]
        for path in candidates:
            if not path.is_file() or path.is_symlink():
                continue
            if str(path.resolve()) in excluded:
                continue
            if path.suffix in {".json", ".jsonl"}:
                files.add(path)
            elif path.suffix in {".pkl", ".pickle", ".xlsx", ".parquet", ".tsv"}:
                unsupported.append(str(path))
    keys = set()
    inventory, unresolved = [], []
    for path in sorted(files):
        count = 0
        identifiable = 0
        try:
            for record in iter_identity_records(path):
                found = row_identity_keys(record)
                keys.update(found)
                identifiable += bool(found)
                count += 1
            inventory.append({"path": str(path), "sha256": file_sha256(path),
                              "identity_objects": count, "resolvable_objects": identifiable})
            if count and not identifiable:
                unresolved.append({"path": str(path), "reason": "only local/ambiguous IDs"})
        except (ValueError, OSError, UnicodeError) as error:
            unresolved.append({"path": str(path), "reason": str(error)})
    report = {"format": "asymvlm_identity_audit_v1", "labels_loaded": False,
              "files": inventory, "unresolved": unresolved,
              "unsupported_files": sorted(unsupported),
              "roots": [str(Path(path)) for path in paths],
              "coverage": "only_explicitly_supplied_paths",
              "identity_keys": sorted(keys), "identity_count": len(keys),
              "complete": not unresolved and not unsupported}
    report["audit_sha256"] = canonical_hash(report)
    return report
