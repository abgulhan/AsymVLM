"""Restartable scorer extraction, training, and development evaluation."""

import argparse
import json
import random
import time
from pathlib import Path

from . import RETENTIONS, VARIANTS
from .checkpoints import atomic_json, atomic_torch_save, canonical_digest
from .protocol import digest_file, entries_for_split, load_manifest


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("extract", "train", "train-original", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--device", default="cuda:0")
        sub.add_argument("--limit", type=int, help="development smoke limit; recorded in metadata")
        if command != "train":
            sub.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next"), required=True)
            sub.add_argument("--model", help="canonical repository or pinned local snapshot")
            sub.add_argument("--model-revision")
            sub.add_argument("--max-visual-patches", type=int, default=1280)
            sub.add_argument("--backbone-dtype", choices=("bfloat16", "float32"), default="bfloat16")
        if command in {"extract", "evaluate"}:
            sub.add_argument("--rank", type=int, default=0)
            sub.add_argument("--world-size", type=int, default=1)
        if command in {"train", "train-original"}:
            sub.add_argument("--seed", type=int, required=True)
            sub.add_argument("--epochs", type=int, default=1)
            sub.add_argument("--learning-rate", type=float, default=1e-3)
            sub.add_argument("--save-every", type=int, default=100)
            sub.add_argument("--resume", action="store_true")
        if command == "train":
            sub.add_argument("--cache", type=Path, required=True)
            sub.add_argument("--variant", choices=("rank32", "rank32_coverage"), default="rank32")
        if command == "evaluate":
            sub.add_argument("--checkpoint", type=Path, required=True)
            sub.add_argument("--split", default="scorer_dev")
            sub.add_argument("--retentions", type=float, nargs="+", default=list(RETENTIONS))
            sub.add_argument("--max-new-tokens", type=int, default=32)
            sub.add_argument("--coverage", action="store_true", help="fixed 80/20 coverage on the SAME rank32 checkpoint")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if hasattr(args, "rank") and not 0 <= args.rank < args.world_size:
        parser.error("require 0 <= rank < world-size")
    if hasattr(args, "epochs") and (args.epochs <= 0 or args.save_every <= 0 or args.learning_rate <= 0):
        parser.error("epochs/save-every/learning-rate must be positive")
    if hasattr(args, "retentions") and (not args.retentions or any(r not in (*RETENTIONS, 1.0) for r in args.retentions)):
        parser.error("evaluation retentions are fixed at .5/.7/.9; 1.0 allowed for full-token control")
    return args


def selected_entries(args, manifest, split, training):
    entries = entries_for_split(manifest, split, training=training)
    if args.limit:
        entries = entries[:args.limit]
    return entries


def validate_cache(record, entry, manifest, backbone_identity=None):
    from .models import require_finite
    if record.get("format") != "asymvlm_mlsys_saliency_v1":
        raise ValueError("unrecognized saliency-cache format")
    if record.get("manifest_sha256") != manifest.file_sha256 or record.get("row_hash") != entry["row_hash"]:
        raise ValueError("cache manifest/row provenance differs")
    if backbone_identity is not None and record.get("backbone") != backbone_identity:
        raise ValueError("mixed backbone/resolution in saliency cache")
    for key in ("vision", "question", "saliency"):
        require_finite(record[key], f"cached {key}")
    vision, question, saliency = (record[k] for k in ("vision", "question", "saliency"))
    if vision.ndim != 2 or question.ndim != 2 or not len(question) or saliency.shape != (len(vision),):
        raise ValueError("invalid saliency-cache tensor dimensions")
    if vision.shape[1] != question.shape[1] or not len(vision):
        raise ValueError("invalid cached hidden/token dimensions")


def extract(args, manifest):
    import torch
    from learned_score.multimodel.saliency import compute_answer_gradient_saliency
    from .backbone import assert_frozen, frozen_versions, load_backbone, prepared_inputs, read_entry
    from .models import require_finite
    entries = selected_entries(args, manifest, "scorer_train", True)
    adapter, dataset, identity = load_backbone(args, manifest)
    versions = frozen_versions(adapter.model)
    args.output.mkdir(parents=True, exist_ok=True)
    for entry in entries[args.rank::args.world_size]:
        path = args.output / (entry["row_hash"] + ".pt")
        if path.exists():
            validate_cache(torch.load(path, map_location="cpu", weights_only=True), entry, manifest, identity)
            continue
        prepared, vision, question = prepared_inputs(adapter, read_entry(dataset, entry), True)
        saliency = require_finite(compute_answer_gradient_saliency(adapter, prepared), "answer-gradient saliency")
        record = {"format": "asymvlm_mlsys_saliency_v1", "backbone": identity,
                  "manifest_sha256": manifest.file_sha256, "row_hash": entry["row_hash"],
                  "dataset": entry["dataset"], "image_sha256": entry["image_sha256"],
                  "vision": vision.to("cpu", dtype=torch.bfloat16),
                  "question": question.to("cpu", dtype=torch.bfloat16),
                  "saliency": saliency.detach().float().cpu()}
        validate_cache(record, entry, manifest, identity)
        assert_frozen(adapter.model, versions)
        atomic_torch_save(path, record)
        print(json.dumps({"event": "extracted", "row_hash": entry["row_hash"], "visual_tokens": len(vision),
                          "question_tokens": len(question), "saliency_std": float(saliency.float().std(unbiased=False))}), flush=True)
        del prepared, vision, question, saliency, record
    atomic_json(args.output / f"rank{args.rank}.complete.json",
                {"manifest_sha256": manifest.file_sha256, "backbone": identity,
                 "rank": args.rank, "world_size": args.world_size,
                 "rows": len(entries[args.rank::args.world_size]), "limit": args.limit})


def original_loss(adapter, prepared, scorer, vision, question, retention):
    from learned_score.llava_objectives import answer_distillation_loss
    from learned_score.multimodel.config import MaskingConfig
    from learned_score.multimodel.masking import visual_key_penalty
    teacher = adapter.teacher_forward(prepared)
    scores = scorer(vision, question)
    penalty = visual_key_penalty(scores, prepared.visual_positions,
                                 prepared.inputs_embeds.shape[1],
                                 MaskingConfig(keep_ratio=retention, C=5.0, tau=0.1))
    student = adapter.student_forward_with_key_penalty(prepared, penalty)
    losses = answer_distillation_loss(student["logits"], teacher["logits"],
                                     student["hidden"], teacher["hidden"],
                                     prepared.answer_positions, prepared.native_inputs["input_ids"],
                                     kl_temperature=1.0, kl_weight=1.0, hidden_weight=1.0, ce_weight=0.1)
    return losses["total"].float() + 0.001 * (scorer.w.float() - 1).square().mean()


def train(args, manifest):
    import torch
    from .backbone import assert_frozen, frozen_versions, load_backbone, prepared_inputs, read_entry
    from .checkpoints import load_checkpoint, save_checkpoint
    from .models import build_scorer, optimizer_step, saliency_loss
    entries = selected_entries(args, manifest, "scorer_train", True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    original = args.command == "train-original"
    variant = "original" if original else args.variant
    if original:
        adapter, dataset, backbone = load_backbone(args, manifest)
        versions = frozen_versions(adapter.model)
        hidden = adapter.hidden_size
        cache_hash = None
    else:
        first = torch.load(args.cache / (entries[0]["row_hash"] + ".pt"), map_location="cpu", weights_only=True)
        validate_cache(first, entries[0], manifest)
        backbone, hidden = first["backbone"], first["vision"].shape[1]
        # Binding all cache file bytes detects stale/replaced labels on resume.
        cache_hash = canonical_digest([(entry["row_hash"], digest_file(args.cache / (entry["row_hash"] + ".pt")))
                                       for entry in entries])
    scorer = build_scorer(variant, hidden).to(args.device).float()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=args.learning_rate, weight_decay=0.01)
    identity = {"variant": variant, "manifest_sha256": manifest.file_sha256,
                "backbone": backbone, "seed": args.seed, "epochs": args.epochs,
                "learning_rate": args.learning_rate, "weight_decay": 0.01,
                "retentions": list(RETENTIONS), "rows": len(entries),
                "cache_sha256": cache_hash, "limit": args.limit,
                "objective": "answer_distillation" if original else "answer_gradient_saliency",
                "coverage_fraction": 0.2 if variant == "rank32_coverage" else 0.0}
    args.output.mkdir(parents=True, exist_ok=True)
    latest = args.output / "latest.pt"
    start_epoch = next_index = steps = 0
    if latest.exists():
        if not args.resume:
            raise FileExistsError("training output already exists; use --resume with identical configuration")
        payload = load_checkpoint(latest, identity, args.device)
        scorer.load_state_dict(payload["state_dict"])
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch, next_index, steps = payload["epoch"], payload["next_index"], payload["steps"]
    else:
        if args.resume:
            raise FileNotFoundError("--resume requires latest.pt")
        save_checkpoint(latest, scorer, optimizer, identity, 0, 0, 0)
        save_checkpoint(args.output / "epoch_zero.pt", scorer, optimizer, identity, 0, 0, 0)
    atomic_json(args.output / "run_metadata.json", identity)
    for epoch in range(start_epoch, args.epochs):
        order = list(entries)
        random.Random(args.seed + epoch).shuffle(order)
        offset = next_index if epoch == start_epoch else 0
        for index in range(offset, len(order)):
            entry = order[index]
            optimizer.zero_grad(set_to_none=True)
            if original:
                prepared, vision, question = prepared_inputs(adapter, read_entry(dataset, entry), True)
                ratio = RETENTIONS[(epoch + index) % len(RETENTIONS)]
                loss = original_loss(adapter, prepared, scorer, vision, question, ratio)
            else:
                record = torch.load(args.cache / (entry["row_hash"] + ".pt"), map_location=args.device, weights_only=True)
                validate_cache(record, entry, manifest, backbone)
                loss = saliency_loss(scorer(record["vision"], record["question"]), record["saliency"])
            loss_value = float(loss.detach())
            optimizer_step(loss, scorer, optimizer)
            steps += 1
            if original:
                assert_frozen(adapter.model, versions)
                del prepared, vision, question
            else:
                del record
            del loss
            if steps % args.save_every == 0:
                save_checkpoint(latest, scorer, optimizer, identity, epoch, index + 1, steps)
                print(json.dumps({"event": "checkpoint", "epoch": epoch, "next_index": index + 1,
                                  "steps": steps, "loss": loss_value}), flush=True)
        save_checkpoint(latest, scorer, optimizer, identity, epoch + 1, 0, steps)
        save_checkpoint(args.output / f"epoch_{epoch + 1:03d}.pt", scorer, optimizer, identity, epoch + 1, 0, steps)
    atomic_json(args.output / "complete.json", {"steps": steps, "identity_sha256": canonical_digest(identity)})


def evaluate(args, manifest):
    import torch
    from benchmark_dual_kv_v2 import check_answer, parse_answer_field
    from .backbone import assert_frozen, frozen_versions, load_backbone, prepared_inputs, read_entry
    from .checkpoints import load_scorer
    from .models import select_tokens
    entries = selected_entries(args, manifest, args.split, False)
    scorer, checkpoint = load_scorer(args.checkpoint, args.device)
    adapter, dataset, backbone = load_backbone(args, manifest)
    if checkpoint["identity"]["manifest_sha256"] != manifest.file_sha256:
        raise ValueError("evaluation manifest differs from training manifest")
    if checkpoint["identity"]["backbone"] != backbone:
        raise ValueError("evaluation model/resolution differs from training")
    variant = checkpoint["variant"]
    if args.coverage:
        if variant == "original":
            raise ValueError("coverage experiment requires the rank32 checkpoint")
        variant = "rank32_coverage"
    versions = frozen_versions(adapter.model)
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_identity = {"checkpoint_sha256": digest_file(args.checkpoint), "manifest_sha256": manifest.file_sha256,
                           "split": args.split, "retentions": args.retentions, "variant": variant,
                           "max_new_tokens": args.max_new_tokens, "backbone": backbone}
    for entry in entries[args.rank::args.world_size]:
        path = args.output / (entry["row_hash"] + ".json")
        if path.exists():
            if json.loads(path.read_text()).get("identity") != evaluation_identity:
                raise ValueError("existing evaluation row has different provenance")
            continue
        row = read_entry(dataset, entry)
        prepared, vision, question = prepared_inputs(adapter, row, False)
        references = parse_answer_field(row["answer"])
        metric = "anls" if "docvqa" in entry["dataset"].lower() else "vqa_score"
        methods = {}
        for retention in args.retentions:
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.no_grad():
                scores = scorer(vision, question)
                selected = select_tokens(scores, vision, retention, variant)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.synchronize()
            selector_seconds = time.perf_counter() - started
            pruned = adapter.physically_prune(prepared, selected, "original")
            ids = adapter.generate_from_pruned(pruned, {"max_new_tokens": args.max_new_tokens})
            prediction = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
            methods[f"{retention:.2f}"] = {"prediction": prediction,
                                          "score": float(check_answer(prediction, references, metric)),
                                          "visual_tokens": len(selected), "visual_tokens_original": len(vision),
                                          "selector_seconds": selector_seconds,
                                          "generated_tokens": int(ids.shape[-1])}
        assert_frozen(adapter.model, versions)
        atomic_json(path, {"format": "asymvlm_mlsys_development_row_v1", "identity": evaluation_identity,
                           "row_hash": entry["row_hash"], "dataset": entry["dataset"], "metric": metric,
                           "image_sha256": entry["image_sha256"], "group_id": entry.get("group_id"),
                           "references": references, "methods": methods})
        print(json.dumps({"event": "evaluated", "row_hash": entry["row_hash"]}), flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json", {"identity": evaluation_identity,
                                                              "rows": len(entries[args.rank::args.world_size])})


def main(argv=None):
    args = parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "extract":
        extract(args, manifest)
    elif args.command in {"train", "train-original"}:
        train(args, manifest)
    else:
        evaluate(args, manifest)


if __name__ == "__main__":
    main()
