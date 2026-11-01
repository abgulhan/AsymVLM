"""Answer-gradient visual-token saliency cache for native scorer training."""

import hashlib
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from .objectives import saliency_from_answer_ce_gradient


class SaliencyCache:
    def __init__(self, root: Path, model_identity: str):
        self.root = Path(root)
        self.model_hash = hashlib.sha256(
            model_identity.encode("utf-8")
        ).hexdigest()

    def _path(self, row_hash: str) -> Path:
        if len(row_hash) != 64 or any(
            character not in "0123456789abcdef" for character in row_hash
        ):
            raise ValueError("row_hash must be lowercase SHA-256")
        return self.root / self.model_hash[:16] / f"{row_hash}.pt"

    def load(self, row_hash: str, expected_tokens: int):
        path = self._path(row_hash)
        if not path.exists():
            return None
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload.get("model_sha256") != self.model_hash
            or payload.get("row_hash") != row_hash
        ):
            raise RuntimeError("saliency cache provenance mismatch")
        values = payload["normalized_saliency"].float()
        if values.shape != (expected_tokens,):
            raise RuntimeError("saliency cache visual-token count mismatch")
        return values

    def save(self, row_hash: str, values: torch.Tensor) -> Path:
        path = self._path(row_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        # V3 and V4 intentionally share the model-specific cache and may
        # finish the same row concurrently on separate GPUs.
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        torch.save(
            {
                "format": "asymvlm_answer_gradient_saliency_v1",
                "model_sha256": self.model_hash,
                "row_hash": row_hash,
                "normalized_saliency": values.detach().float().cpu(),
            },
            temporary,
        )
        if path.exists():
            temporary.unlink()
        else:
            temporary.replace(path)
        return path


def compute_answer_gradient_saliency(adapter, prepared):
    """Compute relu(-d answer CE / d visual-key bias), then z-normalize."""

    if prepared.answer_positions.numel() == 0:
        raise ValueError("answer-gradient saliency requires answer positions")
    sequence_length = prepared.inputs_embeds.shape[1]
    key_bias = torch.zeros(
        sequence_length,
        device=prepared.inputs_embeds.device,
        dtype=prepared.inputs_embeds.dtype,
        requires_grad=True,
    )
    student = adapter.student_forward_with_key_penalty(prepared, key_bias)
    predictors = prepared.answer_positions - 1
    targets = prepared.native_inputs["input_ids"][
        :, prepared.answer_positions
    ]
    logits = student["logits"][:, predictors].float()
    answer_ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)
    )
    gradient = torch.autograd.grad(
        answer_ce, key_bias, retain_graph=False, create_graph=False
    )[0]
    return saliency_from_answer_ce_gradient(
        gradient[prepared.visual_positions]
    )
