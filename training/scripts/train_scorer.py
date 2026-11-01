"""Step 3: fit the rank-32 scorer (W_v, W_q) on cached targets.  CPU or one GPU.

Thin wrapper over the released trainers (identical math, seeds and checkpoint
format `asymvlm_mlsys_scorer_v1`):
  * qwen25 (reference stage), llava_next -> `python -m learned_score.mlsys_scorer train`
        rank32, AdamW lr 1e-3, wd 0.01, grad-clip 1.0, 1 epoch over 5600 rows, seed 42
  * phi35 -> `learned_score.mlsys_sprint.phi_adaptive_v3 --phase train`
        rank32, AdamW lr 1e-3, wd 0.01, grad-clip 1.0, 1 epoch over 512 rows, seed 42
Loss: mlsys_scorer.models.saliency_loss = CE(softmax(z(saliency)), log_softmax(z(scores))).
Batch size is 1 (one image-question per step); the VLM is not loaded.

    python training/scripts/train_scorer.py --model llava_next --manifest data/manifest.json \
        --cache work/llava_next/targets --output work/llava_next/scorer
Output: <output>/latest.pt (+ epoch_zero.pt, epoch_001.pt, run_metadata.json, complete.json).
For Qwen2.5-VL this is the *reference* scorer; run refine_qwen_scorer.py next.
"""
import argparse
from pathlib import Path

from _common import MODELS, load_config, setup_path

setup_path()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True, help="output dir of extract_targets.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.model)
    if args.model == "phi35":
        from learned_score.mlsys_sprint import phi_adaptive_v3 as phi
        return phi.main(["--phase", "train", "--manifest", str(args.manifest), "--cache", str(args.cache),
                         "--output", str(args.output), "--device", args.device, "--model", config["hf_model_id"]]
                        + (["--resume"] if args.resume else []))
    stage = config.get("stage1_reference", config)
    from learned_score.mlsys_scorer.__main__ import main as scorer_main
    return scorer_main(["train", "--manifest", str(args.manifest), "--cache", str(args.cache),
                        "--output", str(args.output), "--device", args.device,
                        "--variant", stage["variant"], "--seed", str(stage["seed"]),
                        "--epochs", str(stage["epochs"]), "--learning-rate", str(stage["learning_rate"]),
                        "--save-every", str(stage["save_every"])] + (["--resume"] if args.resume else []))


if __name__ == "__main__":
    main()
