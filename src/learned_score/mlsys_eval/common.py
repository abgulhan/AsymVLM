"""Small hashing and immutable-output helpers shared by protocol commands."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new_json(path, value):
    """Publish complete JSON, refusing to replace a different existing artifact."""
    path = Path(path)
    encoded = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if path.exists():
        if path.read_text() != encoded:
            raise FileExistsError(f"refusing to overwrite different artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as target:
            target.write(encoded)
            target.flush()
            os.fsync(target.fileno())
        # Hard-link publication is atomic and fails if another process won the race.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def seal_manifest(payload):
    payload = dict(payload)
    payload.pop("manifest_sha256", None)
    payload["manifest_sha256"] = canonical_hash(payload)
    return payload


def load_manifest(path):
    payload = json.loads(Path(path).read_text())
    expected = payload.get("manifest_sha256")
    unsigned = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if not expected or canonical_hash(unsigned) != expected:
        raise ValueError("manifest contents do not match manifest_sha256")
    return payload


def dataset_family(name):
    name = str(name).lower()
    if "docvqa" in name:
        return "DocVQA"
    if "textvqa" in name:
        return "TextVQA"
    if "olmocr" in name:
        return "olmOCR-Bench"
    return str(name)
