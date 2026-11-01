"""Read-only protocol/model loading shared by isolated runtime follow-ups."""

import json
from pathlib import Path
from types import SimpleNamespace

from . import benchmark
from .state import digest, file_digest, atomic_json


FORMAT = "mlsys_runtime_followup_v1"


def load_followup_protocol(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("runtime follow-up protocol digest mismatch")
    if (value.get("format") != FORMAT or value.get("split") != "scorer_dev"
            or value.get("final_pages_opened") is not False
            or value.get("max_new_tokens") != 4096
            or len(value.get("native_request_ids", [])) != 8
            or len(set(value["native_request_ids"])) != 8
            or len(value.get("micro_request_ids", [])) != 4
            or len(set(value["micro_request_ids"])) != 4):
        raise ValueError("unsupported development follow-up protocol")
    for path, expected in value["input_hashes"].items():
        if file_digest(path) != expected:
            raise ValueError(f"frozen follow-up input changed: {path}")
    return value


def model_spec(protocol, name):
    matches = [m for m in protocol["models"] if m["name"] == name]
    if len(matches) != 1:
        raise ValueError("model is not uniquely specified in the protocol")
    return matches[0]


def verify_native_gate(protocol, name):
    path = Path(protocol["native_gates"][name])
    if not path.is_file():
        raise ValueError("long-document branch quarantined: native audit is absent")
    record = json.loads(path.read_text())
    if (record.get("format") != "mlsys_native_long_audit_v1"
            or record.get("protocol_sha256") != protocol["sha256"]
            or record.get("model_name") != name or record.get("passed") is not True
            or record.get("request_ids") != protocol["native_request_ids"]
            or record.get("max_new_tokens") != 4096):
        raise ValueError("long-document branch quarantined: native audit failed or changed")
    for filename, expected in record.get("page_artifacts", {}).items():
        if file_digest(filename) != expected:
            raise ValueError("native page artifact changed after audit publication")
    if len(record.get("page_artifacts", {})) != 8:
        raise ValueError("native audit does not bind all eight page artifacts")
    return {"path": str(path), "sha256": file_digest(path), "passed": True}


def protocol_manifest(protocol):
    config = SimpleNamespace(manifest=Path(protocol["manifest"]), dataset_kind="olmocr")
    manifest = benchmark.load_benchmark_manifest(config)
    entries = benchmark.entries_for_split(manifest, "scorer_dev")
    if (len(entries) != 100 or sorted(e["row_hash"] for e in entries) != protocol["development_request_ids"]
            or not set(protocol["native_request_ids"] + protocol["micro_request_ids"]).issubset(protocol["development_request_ids"])):
        raise ValueError("follow-up page identities differ from the frozen development manifest")
    return manifest, entries


def load_model(protocol, name, device, *, with_scorer=False):
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    from learned_score.mlsys_scorer.protocol import load_manifest
    spec = model_spec(protocol, name)
    manifest, entries = protocol_manifest(protocol)
    args = SimpleNamespace(manifest=Path(protocol["manifest"]), dataset_kind="olmocr", split="scorer_dev",
                           adapter=spec["adapter"], model=spec["model_path"], model_revision=spec["revision"],
                           max_visual_patches=spec.get("max_visual_patches", 1280), device=device,
                           attention_implementation="sdpa", backbone_dtype="bfloat16",
                           dataset_splits=("scorer_dev",))
    adapter, dataset, backbone = benchmark.load_benchmark_backbone(args, manifest)
    scorer = None
    if with_scorer:
        scorer, checkpoint = load_scorer(spec["checkpoint"], device)
        benchmark.require_completed_checkpoint(spec["checkpoint"], checkpoint)
        benchmark.validate_checkpoint_identity(
            checkpoint, load_manifest(protocol["scorer_training_manifest"]).file_sha256, backbone)
        if checkpoint["variant"] != "rank32" or checkpoint["identity"]["seed"] != 42:
            raise ValueError("follow-up requires the frozen rank32 seed-42 scorer")
    return adapter, dataset, backbone, scorer, entries


def require_cuda(device):
    import torch
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("physical follow-up experiments require an assigned CUDA device")


def first_divergence(expected, actual):
    """Zero-based emitted-token index, including a length-only divergence."""
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index
    return None if len(expected) == len(actual) else min(len(expected), len(actual))


def repeated_word_fraction(text, width=8):
    words = text.split()
    count = max(0, len(words) - width + 1)
    return 0.0 if not count else 1 - len({tuple(words[i:i + width]) for i in range(count)}) / count


def write_immutable(path, payload):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise ValueError(f"existing immutable artifact differs: {path}")
        return
    atomic_json(path, payload)
