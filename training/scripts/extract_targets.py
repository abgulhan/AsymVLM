"""Step 2: extract answer-gradient saliency targets with the frozen VLM (GPU).

Thin wrapper over the released extractors; nothing is reimplemented:
  * qwen25, llava_next -> `python -m learned_score.mlsys_scorer extract`
    (multimodel.saliency.compute_answer_gradient_saliency)
  * phi35              -> `learned_score.mlsys_sprint.phi_adaptive_v3 --phase extract`
    (phi_saliency: ReLU(-dCE_answer / d visual-key attention bias), z-scored)

Each training row becomes one `<row_hash>.pt` holding the frozen visual and
question embeddings (bf16) and the per-patch saliency vector (fp32).  Shard
across GPUs with --rank/--world-size (one process per GPU); finished rows are
skipped so the job is restartable.

    python training/scripts/extract_targets.py --model llava_next \
        --manifest data/manifest.json --output work/llava_next/targets --rank 0 --world-size 4
"""
import argparse
from pathlib import Path

from _common import MODELS, hf_model, load_config, setup_path

setup_path()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="scorer-training manifest (phi35: the 512-row phi_train manifest)")
    parser.add_argument("--output", type=Path, required=True, help="target cache directory")
    parser.add_argument("--hf-model", help="HF id or local snapshot (default: pinned id from config; HF_HOME honored)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--limit", type=int, help="smoke-test limit (qwen25/llava_next only)")
    args = parser.parse_args(argv)
    config = load_config(args.model)
    model = hf_model(config, args.hf_model)
    if args.model == "phi35":
        from learned_score.mlsys_sprint import phi_adaptive_v3 as phi
        argv = ["--phase", "extract", "--manifest", str(args.manifest), "--output", str(args.output),
                "--model", model, "--model-revision", config["hf_revision"],
                "--rank", str(args.rank), "--world-size", str(args.world_size), "--device", args.device]
        return phi.main(argv)
    from learned_score.mlsys_scorer.__main__ import main as scorer_main
    argv = ["extract", "--manifest", str(args.manifest), "--output", str(args.output),
            "--adapter", config["adapter"], "--model", model, "--model-revision", config["hf_revision"],
            "--device", args.device, "--rank", str(args.rank), "--world-size", str(args.world_size),
            "--backbone-dtype", config["backbone_dtype"]]
    if args.model == "qwen25":
        argv += ["--max-visual-patches", str(config["max_visual_patches"])]
    if args.limit:
        argv += ["--limit", str(args.limit)]
    return scorer_main(argv)


if __name__ == "__main__":
    main()
