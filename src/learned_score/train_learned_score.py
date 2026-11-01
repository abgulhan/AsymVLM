"""
Train a learned importance scorer for vision token pruning.

Learns a per-dimension weighting w: [D] that improves upon raw cosine
similarity for determining which vision tokens to keep.

Training:
  1. Compute importance = ((v_norm * w) @ t_norm.T).max(dim=-1)
  2. Apply score masking to attention scores (pruned tokens get -C penalty)
  3. MSE loss between masked and full-attention hidden states
  4. Gradient flows: loss → attention → score_mask → importance → w

Usage:
    python train_learned_score.py --dataset DocVQA_VAL --keep-ratio 0.5 --epochs 3
"""

import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F
from PIL import Image

# Setup paths
_script_dir = os.path.dirname(os.path.abspath(__file__))
_dual_kvcache_dir = os.path.join(_script_dir, "..")
_project_root = os.path.join(_dual_kvcache_dir, "..")
sys.path.insert(0, _script_dir)
sys.path.insert(0, _dual_kvcache_dir)
sys.path.insert(0, _project_root)

from scorer import LearnedImportanceScorer, ScoreMaskCache, compute_score_mask


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_phi3_inputs(processor, message):
    prompt = "\n".join([msg["value"] for msg in message if msg["type"] == "text"])
    images = [
        Image.open(msg["value"]).convert("RGB")
        for msg in message if msg["type"] == "image"
    ]
    placeholder = ""
    for idx in range(1, len(images) + 1):
        placeholder += f"<|image_{idx}|>\n"
    messages = [{"role": "user", "content": placeholder + prompt}]
    prompt_text = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return processor(prompt_text, images, return_tensors="pt")


def compute_image_text_boundary(input_ids, default=757):
    if input_ids is None:
        return default
    image_mask = (input_ids < 0) & (input_ids > -1_000_000_000)
    if image_mask.any():
        return int(image_mask.nonzero()[:, 1].max().item() + 1)
    return default


def parse_args():
    p = argparse.ArgumentParser(description="Train learned importance scorer")

    p.add_argument("--model-path", type=str,
                   default="microsoft/Phi-3.5-vision-instruct")
    p.add_argument("--dataset", type=str, default="DocVQA_VAL")
    p.add_argument("--train-start", type=int, default=0)
    p.add_argument("--train-end", type=int, default=None)
    p.add_argument("--subset-size", type=int, default=None)

    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--keep-ratio", type=float, default=0.5,
                   help="Fraction of vision tokens to keep")
    p.add_argument("--mask-C", type=float, default=5.0,
                   help="Score masking strength")
    p.add_argument("--mask-tau", type=float, default=1.0,
                   help="Score masking temperature")
    p.add_argument("--aggregation", type=str, default="max",
                   choices=["max", "mean", "sum"])
    p.add_argument("--num-crops", type=int, default=16)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output-dir", type=str, default="./learned_score_ckpts")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", type=str, default="learned-score")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--start-epoch", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # VLMEvalKit
    vlmeval_path = os.path.join(_project_root, "StreamingVLM", "VLMEvalKit")
    sys.path.insert(0, vlmeval_path)

    from vlmeval.dataset import build_dataset
    from transformers import AutoProcessor
    try:
        from vlmeval.vlm.Phi3_5v.modeling_phi3_v import Phi3VForCausalLM
    except ImportError:
        from transformers import AutoModelForCausalLM as Phi3VForCausalLM

    # Dataset
    print(f"Loading dataset: {args.dataset}")
    dataset = build_dataset(args.dataset)
    total_samples = len(dataset)

    train_start = args.train_start
    train_end = min(
        args.train_end if args.train_end is not None else int(total_samples * 0.8),
        total_samples,
    )
    train_indices = list(range(train_start, train_end))
    print(f"Training: [{train_start}, {train_end}) = {len(train_indices)} samples")

    # Model (frozen)
    print(f"Loading model: {args.model_path}")
    processor = AutoProcessor.from_pretrained(
        args.model_path, trust_remote_code=True, num_crops=args.num_crops,
    )
    model = Phi3VForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="eager",
    ).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    hidden_dim = model.config.hidden_size
    print(f"Hidden dim: {hidden_dim}")

    # Scorer (learnable)
    scorer = LearnedImportanceScorer(hidden_dim=hidden_dim).to(device)

    if args.resume:
        print(f"Resuming from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        scorer.load_state_dict(ckpt["scorer"])
        print(f"  Loaded scorer weights")

    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.lr)

    # W&B
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, config=vars(args))
        except Exception as e:
            print(f"W&B init failed: {e}")

    epoch_size = args.subset_size or len(train_indices)
    total_steps = epoch_size * args.epochs

    print(f"\nTraining config:")
    print(f"  Keep ratio: {args.keep_ratio:.0%}")
    print(f"  Mask C: {args.mask_C}")
    print(f"  Mask tau: {args.mask_tau}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Num crops: {args.num_crops}")
    print(f"  Steps: {epoch_size} x {args.epochs} epochs = {total_steps}")
    print(f"  Scorer params: {sum(p.numel() for p in scorer.parameters())}")
    print()

    global_step = args.start_epoch * epoch_size
    for epoch in range(args.start_epoch, args.epochs):
        epoch_loss = 0.0
        epoch_pool = random.sample(train_indices, min(epoch_size, len(train_indices)))

        for step, idx in enumerate(epoch_pool):
            line = dataset[idx]
            message = dataset.build_prompt(line)
            inputs = build_phi3_inputs(processor, message)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask")
            pixel_values = inputs.get("pixel_values")
            image_sizes = inputs.get("image_sizes")
            boundary = compute_image_text_boundary(input_ids)

            # Get inputs_embeds (frozen)
            with torch.no_grad():
                inputs_embeds = model.model.vision_embed_tokens(
                    input_ids, pixel_values=pixel_values, image_sizes=image_sizes,
                )

            seq_len = inputs_embeds.shape[1]
            vision_embeds = inputs_embeds[:, :boundary, :]
            text_embeds = inputs_embeds[:, boundary:, :]

            # Compute importance with learned scorer
            importance = scorer(vision_embeds.detach(), text_embeds.detach(),
                                args.aggregation)

            # Compute score mask
            score_mask = compute_score_mask(
                importance, boundary, seq_len,
                keep_ratio=args.keep_ratio,
                C=args.mask_C, tau=args.mask_tau,
                device=device, dtype=inputs_embeds.dtype,
            )

            # Baseline forward (no masking)
            with torch.no_grad():
                baseline_out = model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                )
                baseline_hidden = baseline_out.hidden_states[-1]

            # Masked forward (with score mask)
            cache = ScoreMaskCache()
            cache.set_score_mask(score_mask)

            masked_out = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            masked_hidden = masked_out.hidden_states[-1]

            # MSE loss on text positions (after vision boundary)
            bl_text = baseline_hidden[:, boundary:, :]
            mk_text = masked_hidden[:, boundary:, :]

            loss = F.mse_loss(mk_text, bl_text)

            # L2 regularization: keep w close to 1 (cross-sim baseline)
            w_reg = 0.001 * (scorer.w - 1.0).pow(2).mean()
            loss = loss + w_reg

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Clamp w to prevent divergence
            with torch.no_grad():
                scorer.w.clamp_(0.0, 3.0)

            epoch_loss += loss.item()
            global_step += 1

            if (step + 1) % args.log_every == 0:
                avg_loss = epoch_loss / (step + 1)
                w = scorer.w.data
                print(
                    f"[Epoch {epoch+1}] Step {step+1}/{len(epoch_pool)} | "
                    f"loss={avg_loss:.4f} | "
                    f"w_mean={w.mean():.3f} | w_std={w.std():.3f} | "
                    f"w_min={w.min():.3f} | w_max={w.max():.3f}"
                )

                if wandb_run:
                    wandb_run.log({
                        "train/loss": loss.item(),
                        "train/avg_loss": avg_loss,
                        "train/w_mean": w.mean().item(),
                        "train/w_std": w.std().item(),
                    }, step=global_step)

        # Save checkpoint
        ckpt = {
            "scorer": scorer.state_dict(),
            "config": {
                "hidden_dim": hidden_dim,
                "keep_ratio": args.keep_ratio,
                "mask_C": args.mask_C,
                "mask_tau": args.mask_tau,
                "aggregation": args.aggregation,
                "num_crops": args.num_crops,
            },
            "args": vars(args),
        }
        ckpt_path = os.path.join(args.output_dir, f"scorer_epoch{epoch+1}.pt")
        torch.save(ckpt, ckpt_path)
        print(f"Saved: {ckpt_path}")

        w = scorer.w.data
        print(f"\n[Epoch {epoch+1}] Final stats:")
        print(f"  w: mean={w.mean():.3f}, std={w.std():.3f}, "
              f"min={w.min():.3f}, max={w.max():.3f}")
        print(f"  Loss: {epoch_loss / len(epoch_pool):.4f}")
        print()

    if wandb_run:
        wandb_run.finish()

    print("Training complete!")


if __name__ == "__main__":
    main()
