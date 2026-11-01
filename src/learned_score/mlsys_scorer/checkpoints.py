"""Atomic restart checkpoints with configuration and data binding."""

import hashlib
import json
import os
import uuid
from pathlib import Path


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    with temporary.open("w") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_torch_save(path, value):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_checkpoint(path, scorer, optimizer, identity, epoch, next_index, steps):
    import torch
    payload = {"format": "asymvlm_mlsys_scorer_v1", "identity": identity,
               "identity_sha256": canonical_digest(identity),
               "variant": identity["variant"], "hidden_size": scorer.hidden_size,
               "state_dict": {key: value.detach().cpu() for key, value in scorer.state_dict().items()},
               "optimizer": optimizer.state_dict(), "epoch": epoch,
               "next_index": next_index, "steps": steps,
               "torch_version": str(torch.__version__)}
    atomic_torch_save(path, payload)
    atomic_json(Path(path).with_suffix(".json"), {key: value for key, value in payload.items()
                                               if key not in {"state_dict", "optimizer"}})


def load_checkpoint(path, identity=None, device="cpu"):
    import torch
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") != "asymvlm_mlsys_scorer_v1":
        raise ValueError("not an MLSys scorer checkpoint")
    if canonical_digest(payload["identity"]) != payload.get("identity_sha256"):
        raise ValueError("checkpoint identity hash mismatch")
    if identity is not None and payload["identity"] != identity:
        raise ValueError("resume configuration, data, seed, or resolution differs")
    return payload


def load_scorer(path, device="cpu"):
    from .models import build_scorer, require_finite
    payload = load_checkpoint(path, device=device)
    scorer = build_scorer(payload["variant"], payload["hidden_size"]).to(device).float()
    scorer.load_state_dict(payload["state_dict"], strict=True)
    for parameter in scorer.parameters():
        require_finite(parameter, "checkpoint weights")
    scorer.eval()
    return scorer, payload
