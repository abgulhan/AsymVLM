"""Content-addressed run state and crash-safe JSON files."""

import hashlib
import json
import os
import tempfile
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def atomic_json(path, value):
    """Never expose a partially written completion record."""
    path = Path(path)
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def source_inventory(root):
    """Hash current experiment code plus the legacy mechanics it calls."""
    root = Path(root)
    files = []
    for relative in ("learned_score/mlsys_sprint", "learned_score/mlsys_runtime", "learned_score/mlsys_scorer", "learned_score/mlsys_eval", "learned_score/multimodel", "learned_score/tests"):
        base = root / relative
        if base.exists():
            files.extend(base.rglob("*.py"))
    files.extend((root / "scripts").glob("*mlsys*"))
    files.extend((root / "configs").glob("mlsys*.json"))
    return {str(p.relative_to(root)): file_digest(p) for p in sorted(set(files)) if p.is_file()}


def resolve_inside(path, root):
    root = Path(root).resolve()
    result = Path(path)
    result = (root / result).resolve() if not result.is_absolute() else result.resolve()
    if result != root and root not in result.parents:
        raise ValueError(f"output path escapes experiment root: {path}")
    return result
