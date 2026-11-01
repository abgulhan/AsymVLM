"""Versioned query-context scorer; existing frozen Rank32 artifacts stay intact."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from learned_score.mlsys_scorer.models import Rank32Scorer, _inputs, require_finite


class QueryContextRank32Scorer(Rank32Scorer):
    """One positional, residual question block, retaining the old projections.

    Residual outputs start at zero so a warm start reproduces the original
    ranker's scores exactly. Context parameters become active during training.
    No image-independent answer, prompt-template, or label input is accepted.
    """

    def __init__(self, hidden_size):
        super().__init__(hidden_size)
        self.attention_norm = nn.LayerNorm(self.rank)
        self.attention = nn.MultiheadAttention(self.rank, 4, dropout=0.0, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(self.rank)
        self.feedforward = nn.Sequential(nn.Linear(self.rank, 128), nn.GELU(), nn.Linear(128, self.rank))
        nn.init.zeros_(self.attention.out_proj.weight)
        nn.init.zeros_(self.attention.out_proj.bias)
        nn.init.zeros_(self.feedforward[-1].weight)
        nn.init.zeros_(self.feedforward[-1].bias)

    def position_encoding(self, count, device):
        positions = torch.arange(count, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(torch.arange(0, self.rank, 2, device=device, dtype=torch.float32)
                                * (-math.log(10000.0) / self.rank))
        values = torch.zeros(count, self.rank, device=device, dtype=torch.float32)
        values[:, 0::2] = torch.sin(positions * frequencies)
        values[:, 1::2] = torch.cos(positions * frequencies)
        return values

    def forward(self, vision, question):
        vision, question = _inputs(vision, question)
        visual = F.normalize(F.linear(F.normalize(vision, dim=-1), self.visual.weight.float()), dim=-1)
        query = F.linear(F.normalize(question, dim=-1), self.question.weight.float())
        positioned = (self.attention_norm(F.normalize(query, dim=-1))
                      + self.position_encoding(len(query), query.device))[None]
        attended = self.attention(positioned, positioned, positioned, need_weights=False)[0][0]
        query = query + attended
        query = F.normalize(query + self.feedforward(self.feedforward_norm(query)), dim=-1)
        return require_finite((visual @ query.T).max(dim=-1).values, "contextual rank32 scores")


def build_selector(hidden_size, contextual=False, warmstart_state=None):
    scorer = QueryContextRank32Scorer(hidden_size) if contextual else Rank32Scorer(hidden_size)
    if warmstart_state is not None:
        if set(warmstart_state) != {"visual.weight", "question.weight"}:
            raise ValueError("warm start must be an unmodified Rank32 projection checkpoint")
        if contextual:
            missing, unexpected = scorer.load_state_dict(warmstart_state, strict=False)
            if unexpected or any(name in {"visual.weight", "question.weight"} for name in missing):
                raise ValueError("warm-start projections were not restored")
        else:
            scorer.load_state_dict(warmstart_state, strict=True)
    return scorer.float()


def load_selector(path, device="cpu"):
    """Return (FP32 eval scorer, payload), accepting old and versioned weights."""
    from learned_score.mlsys_scorer.checkpoints import canonical_digest, load_scorer
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format") == "asymvlm_mlsys_scorer_v1":
        return load_scorer(path, device)
    if payload.get("format") != "asymvlm_selector_training_v5":
        raise ValueError("unknown selector checkpoint format")
    if payload.get("identity_sha256") != canonical_digest(payload["identity"]):
        raise ValueError("selector checkpoint identity digest mismatch")
    encoder = payload["identity"].get("query_encoder")
    if encoder not in {"token_max", "residual_self_attention_32_4_128"}:
        raise ValueError("unknown query encoder")
    scorer = build_selector(payload["hidden_size"], encoder != "token_max").to(device)
    scorer.load_state_dict(payload["state_dict"], strict=True)
    for parameter in scorer.parameters():
        require_finite(parameter, "selector checkpoint weights")
    return scorer.eval(), payload
