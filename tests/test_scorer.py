"""Query-aware rank-32 scorer: shapes, semantics, and released checkpoints."""

import hashlib
from pathlib import Path

import pytest
import torch

models = pytest.importorskip("learned_score.mlsys_scorer.models")
Rank32Scorer = models.Rank32Scorer
OriginalScorer = models.OriginalScorer

ROOT = Path(__file__).resolve().parents[1]
CKPT_DIR = ROOT / "checkpoints"
MODELS = ["qwen25", "llava_next", "phi35"]


@pytest.fixture(params=[Rank32Scorer, OriginalScorer], ids=["rank32", "original"])
def scorer(request):
    torch.manual_seed(0)
    return request.param(48).eval()


@pytest.mark.parametrize("n,q", [(1, 1), (17, 3), (196, 9)])
def test_forward_shape_and_finite(scorer, n, q):
    torch.manual_seed(1)
    with torch.no_grad():
        scores = scorer(torch.randn(n, 48), torch.randn(q, 48))
    assert scores.shape == (n,)
    assert scores.dtype == torch.float32
    assert torch.isfinite(scores).all()
    # Cosine similarity of normalized projections is bounded.
    if isinstance(scorer, Rank32Scorer):
        assert scores.abs().max() <= 1 + 1e-5


def test_bfloat16_inputs_are_scored_in_fp32():
    scorer = Rank32Scorer(16).eval()
    with torch.no_grad():
        scores = scorer(torch.randn(5, 16).bfloat16(), torch.randn(2, 16).bfloat16())
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()


def test_max_over_question_tokens(scorer):
    """Score(patch) = max over question tokens; adding a token aligned with a
    patch makes that patch's score the maximum attainable value."""
    torch.manual_seed(2)
    vision, question = torch.randn(20, 48), torch.randn(4, 48)
    with torch.no_grad():
        base = scorer(vision, question)
        target = int(base.argmin())
        augmented = scorer(vision, torch.cat([question, vision[target:target + 1]]))
        # Monotone: adding a question token never lowers any score.
        assert (augmented >= base - 1e-6).all()
        assert augmented[target] > base[target]
        # Per-token decomposition equals the max.
        per_token = torch.stack([scorer(vision, question[i:i + 1]) for i in range(len(question))])
        torch.testing.assert_close(per_token.max(0).values, base)
    if isinstance(scorer, Rank32Scorer):
        # identical direction -> cosine 1 in projected space
        assert abs(float(augmented[target]) - 1.0) < 1e-5


def test_permutation_equivariance_over_patches(scorer):
    torch.manual_seed(3)
    vision, question = torch.randn(30, 48), torch.randn(5, 48)
    perm = torch.randperm(30)
    with torch.no_grad():
        torch.testing.assert_close(scorer(vision[perm], question), scorer(vision, question)[perm])
        # invariance to question-token order
        torch.testing.assert_close(scorer(vision, question.flip(0)), scorer(vision, question))


def test_rank32_projection_shapes():
    scorer = Rank32Scorer(64)
    assert scorer.visual.weight.shape == (32, 64)
    assert scorer.question.weight.shape == (32, 64)
    assert sum(p.numel() for p in scorer.parameters()) == 2 * 32 * 64


@pytest.mark.parametrize("bad", [(torch.randn(3, 8), torch.randn(2, 9)),
                                 (torch.randn(0, 8), torch.randn(2, 8)),
                                 (torch.randn(3, 8), torch.full((1, 8), float("nan")))])
def test_rejects_invalid_inputs(bad):
    with pytest.raises((ValueError, FloatingPointError)):
        Rank32Scorer(8)(*bad)


# --------------------------------------------------------------------------- #
# Released checkpoints
# --------------------------------------------------------------------------- #

def _state_dict(payload):
    if isinstance(payload, torch.nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "scorer", "model_state_dict"):
            if isinstance(payload.get(key), dict):
                return payload[key]
        if all(isinstance(v, torch.Tensor) for v in payload.values()):
            return payload
    raise AssertionError(f"unrecognized checkpoint layout: {type(payload)}")


def _load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # older pickles with non-tensor metadata
        return torch.load(path, map_location="cpu", weights_only=False)


def _projection(state, name):
    matches = [v for k, v in state.items() if k.endswith(f"{name}.weight")]
    assert matches, f"no '{name}.weight' in {sorted(state)}"
    return matches[0]


@pytest.mark.parametrize("model", MODELS)
def test_released_checkpoint_loads_and_scores(model):
    path = CKPT_DIR / model / "scorer.pt"
    if not path.exists():
        pytest.skip(f"{path.relative_to(ROOT)} not present")
    state = _state_dict(_load(path))
    visual, question = _projection(state, "visual"), _projection(state, "question")
    assert visual.ndim == 2 and visual.shape[0] == 32
    assert question.shape == visual.shape
    hidden = visual.shape[1]
    for tensor in state.values():
        if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
            assert torch.isfinite(tensor).all()
    scorer = Rank32Scorer(hidden)
    scorer.load_state_dict({"visual.weight": visual.float(), "question.weight": question.float()})
    scorer.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        scores = scorer(torch.randn(64, hidden), torch.randn(7, hidden))
    assert scores.shape == (64,) and torch.isfinite(scores).all()
    # A trained scorer should not be the identical-init random projection.
    assert not torch.equal(visual, question)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _manifest_entries():
    sums = CKPT_DIR / "SHA256SUMS"
    if not sums.exists():
        return []
    entries = []
    for line in sums.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 2 and not line.lstrip().startswith("#"):
            entries.append((parts[0].lower(), parts[-1].lstrip("*")))
    return entries


def test_checkpoint_sha256sums():
    entries = _manifest_entries()
    if not entries:
        pytest.skip("checkpoints/SHA256SUMS not present")
    checked = 0
    for expected, name in entries:
        candidates = [CKPT_DIR / name, ROOT / name]
        path = next((c for c in candidates if c.exists()), None)
        assert path is not None, f"SHA256SUMS lists missing file {name}"
        assert _sha256(path) == expected, f"sha256 mismatch for {name}"
        checked += 1
    assert checked


@pytest.mark.parametrize("model,hidden", [("qwen25", 2048), ("llava_next", 4096), ("phi35", 3072)])
def test_asymvlm_load_scorer_cpu(model, hidden):
    pruning = pytest.importorskip("asymvlm.pruning")
    config = pytest.importorskip("asymvlm.config")
    if not config.get_spec(model).scorer_path().exists():
        pytest.skip("checkpoint not present")
    scorer = pruning.load_scorer(model, device="cpu")
    assert not scorer.training
    assert all(not p.requires_grad for p in scorer.parameters())
    torch.manual_seed(0)
    vision, question = torch.randn(40, hidden), torch.randn(6, hidden)
    with torch.no_grad():
        scores = scorer(vision, question)
        assert scores.shape == (40,) and torch.isfinite(scores).all()
        perm = torch.randperm(40)
        torch.testing.assert_close(scorer(vision[perm], question), scores[perm])
