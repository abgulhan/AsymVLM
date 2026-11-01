"""Quick start: answer one question with AsymVLM uniform pruning and the adaptive gate.

    python examples/quickstart.py                                  # synthetic image, Qwen2.5-VL-3B
    python examples/quickstart.py --model llava_next --image page.png --question "What is the title?"

Needs one GPU; downloads the pinned model snapshot on first use (see docs/INSTALL.md).
"""
import argparse
import tempfile
from pathlib import Path


def synthetic_image():
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (672, 448), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([180, 140, 500, 300], fill="red")
    draw.text((280, 210), "STOP", fill="white")
    path = Path(tempfile.gettempdir()) / "asymvlm_quickstart.png"
    image.save(path)
    return str(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="qwen25", choices=("qwen25", "llava_next", "phi35"))
    ap.add_argument("--image", help="image path (default: a synthetic STOP sign)")
    ap.add_argument("--question", default="What word is written on the sign?")
    ap.add_argument("--keep", type=float, default=0.4, help="fraction of visual tokens kept for AsymVLM uniform")
    a = ap.parse_args()
    from asymvlm import load_model_and_scorer, prepare, prune_and_generate, self_consistency_gate
    image = a.image or synthetic_image()
    rt = load_model_and_scorer(a.model)
    prepared = prepare(rt, image, a.question)
    n = len(prepared.visual_positions)
    text, _ = prune_and_generate(rt, prepared, keep_ratio=a.keep)
    print(f"[{a.model}] visual tokens: {n}")
    print(f"AsymVLM uniform ({100 * a.keep:.0f}% kept): {text}")
    answer, kept = self_consistency_gate(rt, prepared)
    print(f"AsymVLM adaptive ({kept}% kept): {answer}")


if __name__ == "__main__":
    main()
