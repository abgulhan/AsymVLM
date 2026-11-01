"""One-pass, warm-start selector pilot with answer-free query inputs.

All changes live in new modules. Existing checkpoint and saliency-cache bytes
are read-only. Cached saliency can be reused only after auditing original
manifest membership, per-row provenance, and backbone/resolution equivalence.
"""

import argparse
from collections import Counter
from contextlib import contextmanager
from functools import partial
import fcntl
import json
import math
from pathlib import Path
import random
import time

import torch

from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
from learned_score.mlsys_scorer.checkpoints import atomic_json, atomic_torch_save, canonical_digest, load_checkpoint
from learned_score.mlsys_scorer.models import optimizer_step, require_finite, saliency_loss
from learned_score.mlsys_scorer.protocol import OFFICIAL_TRAIN_DATASETS, digest_file, load_manifest
from .selector_models_v5 import build_selector, load_selector


FORMAT = "asymvlm_selector_training_v5"
RECIPES = ("saliency_control", "query_saliency", "answer_soft", "query_answer_soft", "answer_hard")
TARGETS = (40, 50, 60, 70)


def source_hashes():
    from . import selector_models_v5
    from learned_score import llava_objectives
    from learned_score.mlsys_scorer import models, backbone, checkpoints, protocol
    from learned_score.multimodel import saliency
    modules = (selector_models_v5, llava_objectives, models, backbone, checkpoints, protocol, saliency)
    paths = [Path(__file__), *(Path(module.__file__) for module in modules)]
    learned_root = Path(__file__).resolve().parents[1]
    paths.extend(learned_root / relative for relative in (
        "mlsys_sprint/phi_adaptive_v3.py", "mlsys_sprint/phi_adaptive_v3_compat.py", "mlsys_sprint/phi_adaptive_v4.py",
        "multimodel/adapters/llava.py", "multimodel/adapters/llava_next.py", "multimodel/adapters/qwen2_vl.py",
        "multimodel/adapters/qwen2_5_vl.py", "mlsys_eval/datasets.py", "mlsys_eval/common.py"))
    return {str(path.resolve()): digest_file(path) for path in paths}


def pilot_manifest(path):
    manifest = load_manifest(path, expected_training_count=512)
    entries = manifest.splits["scorer_train"]
    if Counter(entry["dataset"] for entry in entries) != Counter({name: 256 for name in OFFICIAL_TRAIN_DATASETS}):
        raise ValueError("selector pilot requires exactly 256 official training rows per dataset")
    return manifest


def cache_membership(path, entries):
    """Audit membership against historical scorer_train or Phi training split."""
    payload = json.loads(Path(path).read_text())
    candidates = payload.get("splits", {}).get("scorer_train", payload.get("splits", {}).get("phi_train", []))
    old = {row["row_hash"]: row for row in candidates}
    if len(old) != len(candidates):
        raise ValueError("duplicate historical training rows")
    for row in entries:
        original = old.get(row["row_hash"])
        if original is None or any(original.get(key) != row.get(key) for key in
                                  ("dataset", "dataset_index", "dataset_revision", "source_split", "question_id",
                                   "image_id", "image_sha256", "row_hash")):
            raise ValueError("pilot row is not the same official training row in the cache manifest")
    return digest_file(path)


def validate_record(record, entry, manifest_hash, expected_backbone=None):
    from learned_score.mlsys_scorer.__main__ import validate_cache
    from types import SimpleNamespace
    validate_cache(record, entry, SimpleNamespace(file_sha256=manifest_hash))
    if record.get("dataset") != entry["dataset"] or record.get("image_sha256") != entry["image_sha256"]:
        raise ValueError("cached dataset/image identity mismatch")
    if expected_backbone is not None:
        assert_same_backbone(record["backbone"], expected_backbone)


def assert_same_backbone(left, right):
    """Source instrumentation can differ; model and physical geometry cannot."""
    keys = ("adapter", "model", "resolved_revision", "resolution", "backbone_dtype", "attention_implementation",
            "position_policy", "question_only")
    if any(left.get(key) != right.get(key) for key in keys):
        raise ValueError("cached/warm-start/native model or resolution differs")


def question_only_inputs(prepared):
    semantic, visual, answers = prepared.semantic_text_positions, prepared.visual_positions, prepared.answer_positions
    if not len(semantic) or not len(visual) or not len(answers):
        raise ValueError("training requires nonempty visual, question, and answer spans")
    if semantic.ndim != 1 or visual.ndim != 1 or answers.ndim != 1:
        raise ValueError("prepared positions must be one-dimensional")
    if int(semantic.max()) >= int(answers.min()) or torch.isin(semantic, visual).any():
        raise ValueError("reference answer or image tokens leaked into scorer question inputs")
    return prepared.inputs_embeds[0, visual].detach(), prepared.inputs_embeds[0, semantic].detach()


def nominal_training_count(prepared, target, entry=None, adapter=None, reserved_generated_slots=127):
    """Training operating point, not an evaluation aggregate-budget claim."""
    if target not in TARGETS:
        raise ValueError("unregistered training compression target")
    n = len(prepared.visual_positions)
    # prompt_text_positions excludes teacher-forced answers and includes all
    # protected newline/crop-boundary tokens in the model adapter's contract.
    protected = len(prepared.prompt_text_positions) + reserved_generated_slots
    raw = ((n + protected) * (100 - target)) // 100 - protected
    counts = (entry or {}).get("training_visual_counts", {})
    supplied = counts.get(adapter, counts)
    if str(target) in supplied:
        value = supplied[str(target)]
        if type(value) is not int or not 1 <= value <= n:
            raise ValueError("invalid manifest training count")
        return value, {"target": target, "count_source": "manifest", "nominal_feasible": raw >= 1}
    return max(1, min(n, raw)), {"target": target, "count_source": "nominal_per_input",
                              "nominal_feasible": raw >= 1, "protected_slots": protected}


def attention_key_penalty(scores, positions, sequence_length, keep_count, *, hard=False, C=5.0, tau=0.1):
    """Stable exact-count cutoff with sigmoid or straight-through key bias.

    Hard means a -C bias on precisely N-K visual keys in the forward pass;
    neither variant physically removes tokens during training.
    """
    require_finite(scores, "mask scores")
    if (scores.ndim != 1 or positions.ndim != 1 or len(scores) != len(positions)
            or type(keep_count) is not int or not 1 <= keep_count <= len(scores)
            or len(positions.unique()) != len(positions) or positions.dtype != torch.long
            or int(positions.min()) < 0 or int(positions.max()) >= sequence_length
            or not math.isfinite(C) or C < 0 or not math.isfinite(tau) or tau <= 0):
        raise ValueError("invalid visual-key mask geometry or hyperparameters")
    penalty = scores.new_zeros(sequence_length)
    if keep_count == len(scores) or C == 0:
        return penalty + scores.sum() * 0.0
    selected = torch.argsort(scores, descending=True, stable=True)[:keep_count]
    threshold = scores[selected[-1]].detach()
    soft_drop = torch.sigmoid((threshold - scores) / tau)
    if hard:
        hard_drop = torch.ones_like(scores).scatter(0, selected, 0.0)
        soft_drop = hard_drop - soft_drop.detach() + soft_drop
    return penalty.scatter(0, positions, -C * soft_drop)


def answer_loss(adapter, prepared, scorer, keep_count, *, hard=False):
    """Same KL+hidden+CE objective for all answer-preservation recipes."""
    from learned_score.llava_objectives import answer_distillation_loss
    vision, question = question_only_inputs(prepared)
    teacher = adapter.teacher_forward(prepared)
    if any(value.requires_grad for value in teacher.values() if torch.is_tensor(value)):
        raise ValueError("teacher outputs must be detached")
    scores = scorer(vision, question)
    penalty = attention_key_penalty(scores, prepared.visual_positions, prepared.inputs_embeds.shape[1],
                                    keep_count, hard=hard)
    student = adapter.student_forward_with_key_penalty(prepared, penalty)
    losses = answer_distillation_loss(student["logits"], teacher["logits"], student["hidden"], teacher["hidden"],
                                     prepared.answer_positions, prepared.native_inputs["input_ids"],
                                     kl_temperature=1.0, kl_weight=1.0, hidden_weight=1.0, ce_weight=0.1)
    return require_finite(losses["total"].float(), "answer-preservation loss"), {
        key: float(value.detach()) for key, value in losses.items()}


@contextmanager
def decoder_memory(adapter, adapter_name):
    """Recompute decoder layers without turning on dropout or training weights."""
    if adapter_name == "phi3_vision":
        # Phi's key-bias hooks must survive recomputation; save-on-CPU avoids
        # changing that hook lifetime in its frozen historical adapter.
        with torch.autograd.graph.save_on_cpu(pin_memory=True):
            yield
        return
    model = adapter.model
    decoder = model.language_model.model if adapter_name == "llava_next" else model.model
    if any(parameter.requires_grad for parameter in model.parameters()) or any(m.training for m in model.modules()):
        raise ValueError("checkpointing requires an eval-mode frozen backbone")
    from torch.utils.checkpoint import checkpoint
    old = (decoder.training, getattr(decoder, "gradient_checkpointing", False),
           getattr(decoder, "_gradient_checkpointing_func", None))
    decoder.training, decoder.gradient_checkpointing = True, True
    decoder._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False, preserve_rng_state=True)
    try:
        yield
    finally:
        decoder.training, decoder.gradient_checkpointing = old[:2]
        if old[2] is None:
            del decoder._gradient_checkpointing_func
        else:
            decoder._gradient_checkpointing_func = old[2]


def _load_backbone(args, manifest):
    if args.adapter == "phi3_vision":
        from .phi_adaptive_v4 import load_phi_backbone
        return load_phi_backbone(args, manifest)
    from learned_score.mlsys_scorer.backbone import load_backbone
    return load_backbone(args, manifest)


def _extract_record(adapter, dataset, backbone, entry, manifest_hash):
    prepared = adapter.prepare_sample(read_entry(dataset, entry), include_answer=True)
    vision, question = question_only_inputs(prepared)
    if backbone["adapter"] == "phi3_vision":
        from .phi_adaptive_v3 import phi_saliency
        values = phi_saliency(adapter, prepared)
    else:
        from learned_score.multimodel.saliency import compute_answer_gradient_saliency
        with decoder_memory(adapter, backbone["adapter"]):
            values = compute_answer_gradient_saliency(adapter, prepared)
    return {"format": "asymvlm_mlsys_saliency_v1", "backbone": backbone,
            "manifest_sha256": manifest_hash, "row_hash": entry["row_hash"],
            "dataset": entry["dataset"], "image_sha256": entry["image_sha256"],
            "vision": vision.to("cpu", dtype=torch.bfloat16), "question": question.to("cpu", dtype=torch.bfloat16),
            "saliency": values.detach().float().cpu()}


def checkpoint(path, scorer, optimizer, identity, next_index, metrics):
    payload = {"format": FORMAT, "identity": identity, "identity_sha256": canonical_digest(identity),
               "variant": "query_rank32" if identity["query_encoder"] != "token_max" else "rank32",
               "hidden_size": scorer.hidden_size, "state_dict": {k: v.detach().cpu() for k, v in scorer.state_dict().items()},
               "optimizer": optimizer.state_dict(), "next_index": next_index, "steps": next_index,
               "metrics": metrics, "torch_rng_state": torch.get_rng_state(),
               "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "torch_version": str(torch.__version__)}
    atomic_torch_save(path, payload)
    return payload


def _train_identity(args, manifest, warm, order, cache_hash):
    return {"format": FORMAT, "recipe": args.recipe, "query_encoder":
            "residual_self_attention_32_4_128" if args.recipe.startswith("query_") else "token_max",
            "objective": "answer_gradient_saliency" if "saliency" in args.recipe else "answer_preservation",
            "mask_type": "none" if "saliency" in args.recipe else ("hard_straight_through" if args.recipe == "answer_hard" else "soft"),
            "manifest_sha256": manifest.file_sha256, "warmstart_sha256": digest_file(args.warmstart),
            "warmstart_identity_sha256": warm["identity_sha256"], "backbone": warm["identity"]["backbone"],
            "cache_sha256": cache_hash, "cache_manifest_sha256": digest_file(args.cache_manifest) if args.cache_manifest else None,
            "seed": args.seed, "epochs": 1, "rows": len(order), "row_order_sha256": canonical_digest([e["row_hash"] for e in order]),
            "learning_rate": 1e-4, "weight_decay": 0.01, "max_grad_norm": 1.0,
            "targets": list(TARGETS), "reserved_generated_slots": 127,
            "mask_C": 5.0, "mask_tau": 0.1, "kl_weight": 1.0, "hidden_weight": 1.0, "ce_weight": 0.1,
            "soft_training_is_physical_compression": False, "answer_inputs_to_scorer": False,
            "runtime_request": {key: getattr(args, key, None) for key in (
                "adapter", "model", "model_revision", "max_visual_patches", "backbone_dtype", "num_crops")},
            "training_memory_strategy": "save_on_cpu" if args.adapter == "phi3_vision" else "nonreentrant_decoder_checkpointing",
            "source_sha256": source_hashes()}


def train_run(args):
    """Restart automatically from the identical per-example optimizer state."""
    manifest = pilot_manifest(args.manifest)
    entries = sorted(manifest.splits["scorer_train"], key=lambda row: (row["dataset"], row["row_hash"]))
    random.Random(args.seed).shuffle(entries)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    warm = load_checkpoint(args.warmstart, device="cpu")
    if warm.get("variant") not in {"rank32", "rank32_coverage"}:
        raise ValueError("pilot warm start must be the frozen learned Rank32 scorer")
    if warm["identity"]["backbone"]["adapter"] != args.adapter:
        raise ValueError("warm-start model family differs")
    native = None
    versions = None

    def backbone():
        nonlocal native, versions
        if native is None:
            native = _load_backbone(args, manifest)
            assert_same_backbone(native[2], warm["identity"]["backbone"])
            versions = frozen_versions(native[0].model)
        return native

    saliency = "saliency" in args.recipe or args.extract_only
    cache_hash = None
    cache_paths = {}
    if saliency:
        historical_hash = cache_membership(args.cache_manifest, entries) if args.cache_manifest else None
        hashes = []
        generated_root = args.output / "saliency_cache"
        for entry in entries:
            path = args.cache / (entry["row_hash"] + ".pt") if args.cache else generated_root / (entry["row_hash"] + ".pt")
            expected_hash = historical_hash if args.cache else manifest.file_sha256
            if path.exists():
                record = torch.load(path, map_location="cpu", weights_only=True)
                validate_record(record, entry, expected_hash, warm["identity"]["backbone"])
            else:
                if args.cache:
                    raise FileNotFoundError(f"read-only source cache missing training row: {path}")
                adapter, dataset, identity = backbone()
                record = _extract_record(adapter, dataset, identity, entry, manifest.file_sha256)
                validate_record(record, entry, manifest.file_sha256, warm["identity"]["backbone"])
                assert_frozen(adapter.model, versions)
                atomic_torch_save(path, record)
                print(json.dumps({"event": "saliency_extracted", "row_hash": entry["row_hash"]}), flush=True)
            hashes.append((entry["row_hash"], digest_file(path)))
            cache_paths[entry["row_hash"]] = (path, expected_hash, hashes[-1][1])
            del record
        cache_hash = canonical_digest(hashes)
        if args.extract_only:
            atomic_json(args.output / "extraction.complete.json", {"format": FORMAT, "manifest_sha256": manifest.file_sha256,
                        "warmstart_sha256": digest_file(args.warmstart), "cache_sha256": cache_hash, "rows": len(entries)})
            return
    identity = _train_identity(args, manifest, warm, entries, cache_hash)
    scorer = build_selector(warm["hidden_size"], args.recipe.startswith("query_"), warm["state_dict"]).to(args.device)
    scorer.train()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4, weight_decay=0.01)
    latest = args.output / "latest.pt"
    next_index, metrics = 0, []
    if latest.exists():
        payload = torch.load(latest, map_location=args.device, weights_only=True)
        if (payload.get("format") != FORMAT or payload.get("identity") != identity
                or payload.get("identity_sha256") != canonical_digest(identity)):
            raise ValueError("resume source/config/data/checkpoint identity changed")
        scorer.load_state_dict(payload["state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        next_index, metrics = payload["next_index"], payload["metrics"]
        if type(next_index) is not int or not 0 <= next_index <= len(entries) or len(metrics) != next_index:
            raise ValueError("invalid resume position")
        torch.set_rng_state(payload["torch_rng_state"].cpu())
        if torch.cuda.is_available() and payload["cuda_rng_states"]:
            torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng_states"]])
    else:
        checkpoint(latest, scorer, optimizer, identity, 0, [])
        checkpoint(args.output / "epoch_zero.pt", scorer, optimizer, identity, 0, [])
    if (args.output / "complete.json").exists():
        marker = json.loads((args.output / "complete.json").read_text())
        if (next_index != len(entries) or marker.get("identity_sha256") != canonical_digest(identity)
                or marker.get("checkpoint_sha256") != digest_file(latest)):
            raise ValueError("completed run changed")
        return marker
    atomic_json(args.output / "run_metadata.json", identity)
    initial_sources = identity["source_sha256"]
    for index in range(next_index, len(entries)):
        if source_hashes() != initial_sources:
            raise ValueError("training source changed while run was active")
        entry = entries[index]
        optimizer.zero_grad(set_to_none=True)
        started = time.monotonic()
        diagnostics = {}
        if saliency:
            path, expected_hash, expected_file_hash = cache_paths[entry["row_hash"]]
            if digest_file(path) != expected_file_hash:
                raise ValueError("saliency cache bytes changed during training")
            record = torch.load(path, map_location=args.device, weights_only=True)
            validate_record(record, entry, expected_hash, warm["identity"]["backbone"])
            loss = saliency_loss(scorer(record["vision"], record["question"]), record["saliency"])
            optimizer_step(loss, scorer, optimizer, max_grad_norm=1.0)
            del record
        else:
            adapter, dataset, _ = backbone()
            prepared = adapter.prepare_sample(read_entry(dataset, entry), include_answer=True)
            target = TARGETS[index % len(TARGETS)]
            keep_count, diagnostics = nominal_training_count(prepared, target, entry, args.adapter)
            with decoder_memory(adapter, args.adapter):
                loss, parts = answer_loss(adapter, prepared, scorer, keep_count, hard=args.recipe == "answer_hard")
                optimizer_step(loss, scorer, optimizer, max_grad_norm=1.0)
            diagnostics.update(parts)
            diagnostics.update(visual_count=len(prepared.visual_positions), keep_count=keep_count)
            assert_frozen(adapter.model, versions)
            del prepared
        metrics.append({"step": index + 1, "row_hash": entry["row_hash"], "loss": float(loss.detach()),
                        "seconds": time.monotonic() - started, **diagnostics})
        del loss
        # Save every example: one-hour debug allocations can resume without
        # replaying an expensive teacher/student forward or duplicating updates.
        checkpoint(latest, scorer, optimizer, identity, index + 1, metrics)
        print(json.dumps({"event": "selector_training", "recipe": args.recipe, **metrics[-1]}), flush=True)
    marker = {"format": FORMAT, "identity_sha256": canonical_digest(identity), "steps": len(entries),
              "checkpoint_sha256": digest_file(latest), "manifest_sha256": manifest.file_sha256,
              "wall_time_in_training_steps": sum(row["seconds"] for row in metrics),
              "evidence_status": "development training; no evaluation labels consumed"}
    atomic_json(args.output / "complete.json", marker)
    return marker


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--warmstart", type=Path, required=True)
    parser.add_argument("--recipe", choices=RECIPES, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next", "phi3_vision"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, choices=(42, 43), default=42)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--cache-manifest", type=Path)
    parser.add_argument("--extract-only", action="store_true")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--backbone-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--num-crops", type=int, default=16)
    args = parser.parse_args(argv)
    if bool(args.cache) != bool(args.cache_manifest):
        parser.error("read-only cache reuse requires --cache and --cache-manifest together")
    if args.max_visual_patches < 1 or args.num_crops < 1:
        parser.error("resolution limits must be positive")
    args.dataset_splits = ("scorer_train",)
    args.attention_implementation = "eager"
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another worker owns this exact training output") from exc
        train_run(args)


if __name__ == "__main__":
    main()
