"""Optional GPU smoke test (NOT collected by the CPU suite).

Runs one TextVQA-style image+question through each released model at 40%
visual-token keep, then through the adaptive self-consistency gate.

Usage:
    python tests/gpu_smoke.py [--models qwen25 llava_next phi35] [--image path.jpg]
    pytest -q tests/gpu_smoke.py          # same checks, as tests

Requires: a CUDA GPU, the model snapshots already present in the Hugging Face
cache (HF_HOME), and the released scorer checkpoints under checkpoints/.
Every check is skipped (not failed) if any requirement is missing or if the
public asymvlm API it needs is not available.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    sys.path.insert(0, str(p))

import pytest  # noqa: E402

QUESTION = "What word is written on the sign?"
MODELS = ["qwen25", "llava_next", "phi35"]


def _synthetic_image():
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (672, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([180, 140, 500, 300], fill="red")
    draw.text((280, 210), "STOP", fill="white")
    import tempfile
    path = Path(tempfile.gettempdir()) / "asymvlm_gpu_smoke.png"
    image.save(path)
    return str(path)  # adapters accept an image path


def _requirements(model):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    config = pytest.importorskip("asymvlm.config")
    pruning = pytest.importorskip("asymvlm.pruning")
    spec = config.get_spec(model)
    if not spec.scorer_path().exists():
        pytest.skip(f"missing scorer checkpoint {spec.scorer_checkpoint}")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(spec.hf_id, revision=spec.revision, local_files_only=True)
    except Exception:
        pytest.skip(f"{spec.hf_id} not in the local Hugging Face cache")
    return spec, pruning


_RUNTIMES = {}


def _runtime(spec, pruning):
    if spec.key not in _RUNTIMES:
        _RUNTIMES.clear()  # keep at most one backbone resident
        import torch
        torch.cuda.empty_cache()
        _RUNTIMES[spec.key] = pruning.load_model_and_scorer(spec.key, device="cuda:0")
    return _RUNTIMES[spec.key]


@pytest.mark.parametrize("model", MODELS)
def test_prune_40_and_generate(model):
    spec, pruning = _requirements(model)
    runtime = _runtime(spec, pruning)
    prepared = pruning.prepare(runtime, _synthetic_image(), QUESTION)
    n = len(prepared.visual_positions)
    indices = pruning.select_tokens(prepared, 0.4, runtime.scorer)
    assert len(indices) == pruning.keep_count(n, 0.4) and len(indices.unique()) == len(indices)
    text, result = pruning.prune_and_generate(runtime, prepared, 0.4, max_new_tokens=16)
    assert isinstance(text, str) and text.strip()
    print(f"[{model}] visual {n} -> {len(indices)} (40%): {text!r}")


@pytest.mark.parametrize("model", MODELS)
def test_self_consistency_gate(model):
    spec, pruning = _requirements(model)
    gate = pytest.importorskip("asymvlm.gate")
    runtime = _runtime(spec, pruning)
    prepared = pruning.prepare(runtime, _synthetic_image(), QUESTION)
    answer, kept = gate.self_consistency_gate(runtime, prepared, max_new_tokens=16)
    assert kept in (gate.GATE_PROBES[0], gate.GATE_CAP)
    assert isinstance(answer, str)
    print(f"[{model}] gate kept {kept}%: {answer!r}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()
    selected = ",".join(args.models)
    extra = ["-k", " or ".join(args.models)] if selected != ",".join(MODELS) else []
    sys.exit(pytest.main(["-q", "-s", "-rs", __file__, "-p", "no:cacheprovider", *extra]))


if __name__ == "__main__":
    main()
