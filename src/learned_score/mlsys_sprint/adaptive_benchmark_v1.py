"""Development-only adaptive vision + fixed recent-text cache benchmark.

Future AsymVLM runs use this entry point, not the frozen fixed-vision runners::

    python -m learned_score.mlsys_sprint.adaptive_benchmark_v1 \
      --manifest MANIFEST --adapter qwen2_5_vl --model SNAPSHOT \
      --checkpoint SCORER --budget-policy CALIBRATED_POLICY --output NEW_DIRECTORY

The policy must be calibrated for this exact scorer. No guessed thresholds,
final-label calibration, or silent static deployment is permitted. Full-image
references and explicitly requested fixed-image controls are labeled separately.
This runner is not a replacement for the frozen cache-method confirmation.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import platform
import socket

from . import benchmark
from .adaptive_vision_policy import AdaptiveVisionPolicy
from learned_score.mlsys_scorer.checkpoints import atomic_json, canonical_digest
from learned_score.mlsys_scorer.protocol import digest_file, entries_for_split, load_manifest


@dataclass(frozen=True)
class AdaptiveCell:
    vision_budget_mode: str
    selector: str
    visual_retention: float | None
    generated_budget: int | None
    policy_tag: str = ""

    @property
    def key(self):
        text = "off" if self.generated_budget is None else str(self.generated_budget)
        vision = (f"adaptive_{self.policy_tag}" if self.vision_budget_mode == "adaptive"
                  else f"{self.vision_budget_mode}_v{round(self.visual_retention * 100)}")
        return f"{self.selector}_{vision}_text{text}"

    @property
    def method(self):
        return self.key


def make_cells(policy, *, generated_budget=512, selector="scorer", fixed_retentions=()):
    if selector not in ("scorer", "scorer_coverage"):
        raise ValueError("adaptive AsymVLM requires a learned selector")
    if selector != policy.payload["selector"]:
        raise ValueError("patch selector differs from adaptive-policy calibration")
    if generated_budget not in (None, 128, 256, 512, 1024):
        raise ValueError("unsupported fixed generated-text budget")
    if any(not math.isfinite(r) or not 0 < r < 1 or abs(r * 100 - round(r * 100)) > 1e-8
           for r in fixed_retentions):
        raise ValueError("fixed controls must use integer-percent retentions below 100%")
    tag = f"{policy.payload['statistic']}_{policy.payload['form']}_{canonical_digest(policy.payload)[:12]}"
    cells = [AdaptiveCell("all", "full", 1., None)]
    if generated_budget is not None:
        cells.append(AdaptiveCell("all", "full", 1., generated_budget))
    cells.append(AdaptiveCell("adaptive", selector, None, generated_budget, tag))
    cells.extend(AdaptiveCell("fixed_control", selector, r, generated_budget)
                 for r in dict.fromkeys(fixed_retentions))
    return cells


def select_exact_count(scores, vision, count, *, coverage=False):
    """Never round-trip an integer count through a floating-point fraction."""
    import torch
    from torch.nn import functional as F
    if scores.ndim != 1 or vision.ndim != 2 or len(scores) != len(vision):
        raise ValueError("scores and visual features must describe the same patches")
    if type(count) is not int or not 1 <= count <= len(scores):
        raise ValueError("invalid exact patch count")
    if not bool(torch.isfinite(scores).all()) or not bool(torch.isfinite(vision).all()):
        raise ValueError("patch scores and features must be finite")
    if count == len(scores):
        return torch.arange(count, device=vision.device, dtype=torch.long)
    order = torch.argsort(scores.float(), descending=True, stable=True)
    if not coverage:
        return order[:count].sort().values
    learned_count = max(1, count * 4 // 5)
    selected = torch.empty(count, dtype=torch.long, device=vision.device)
    selected[:learned_count] = order[:learned_count]
    normalized = F.normalize(vision.float(), dim=-1)
    distance = torch.full((len(scores),), float("inf"), device=vision.device)
    for start in range(0, learned_count, 64):
        batch = normalized[selected[start:min(start + 64, learned_count)]]
        torch.minimum(distance, (1 - normalized @ batch.T).clamp(0, 2).min(dim=1).values, out=distance)
    distance[selected[:learned_count]] = -float("inf")
    for offset in range(learned_count, count):
        index = distance.argmax().reshape(1)
        selected[offset:offset + 1] = index
        candidate = (1 - normalized @ normalized.index_select(0, index).T).squeeze(1).clamp(0, 2)
        torch.minimum(distance, candidate, out=distance)
        distance.index_fill_(0, index, -float("inf"))
    return selected.sort().values


def make_selector(cell, policy, scorer=None, scorer_variant=None, observed=None, assignment=None):
    import torch

    @torch.no_grad()
    def select(prepared):
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        count = len(vision)
        if not count:
            raise ValueError("input has no prunable image tokens")
        decision = None
        if cell.vision_budget_mode == "all":
            kept = count
            selected = torch.arange(count, device=vision.device, dtype=torch.long)
        else:
            if scorer is None:
                raise ValueError("adaptive/fixed learned selection requires a verified scorer")
            if cell.selector == "scorer_coverage" and scorer_variant == "original":
                raise ValueError("coverage requires the rank32 scorer")
            scores = scorer(vision, question)
            if cell.vision_budget_mode == "adaptive":
                decision = policy.decide(scores)
                kept = decision.selected_count
                if assignment is not None:
                    if (assignment["input_visual_tokens"] != count
                            or abs(assignment["statistic_value"] - decision.statistic_value) > 1e-6
                            or assignment["fallback"] != decision.fallback):
                        raise ValueError("live image scores differ from frozen budget assignment")
                    kept = assignment["selected_count"]
            elif cell.vision_budget_mode == "fixed_control":
                kept = max(1, count * round(cell.visual_retention * 100) // 100)
            else:
                raise ValueError("unknown vision budget mode")
            selected = select_exact_count(scores, vision, kept, coverage=cell.selector == "scorer_coverage")
        if (selected.dtype != torch.long or selected.ndim != 1 or len(selected) != kept
                or len(selected.unique()) != kept or int(selected.min()) < 0 or int(selected.max()) >= count
                or bool((selected[1:] <= selected[:-1]).any())):
            raise RuntimeError("selector violated exact count or chronological patch identity")
        if observed is not None:
            observed.update({"vision_budget_mode": cell.vision_budget_mode,
                             "input_visual_tokens": count, "retained_visual_tokens": kept,
                             "input_text_tokens": int(prepared.inputs_embeds.shape[1] - count),
                             "input_prefix_tokens": int(prepared.inputs_embeds.shape[1]),
                             "visual_retention": kept / count,
                             "requested_visual_retention": decision.retention if decision else cell.visual_retention,
                             "adaptive_decision": decision.asdict() if decision else None,
                             "aggregate_reconciled_selected_count": kept if assignment is not None and decision else None})
        return selected
    return select


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("vqa", "olmocr"), default="vqa")
    parser.add_argument("--scorer-training-manifest", type=Path)
    parser.add_argument("--split", choices=("scorer_dev", "scorer_screening", "scorer_selection", "development"), default="scorer_dev")
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--budget-policy", type=Path, required=True,
                        help="frozen development-calibrated policy bound to this exact checkpoint")
    parser.add_argument("--request-ids", type=Path, help="explicit development subset; list of row identity objects")
    parser.add_argument("--budget-assignments", type=Path, help="answer-free exact aggregate token reconciliation")
    parser.add_argument("--selector", choices=("scorer", "scorer_coverage"), default="scorer")
    parser.add_argument("--fixed-baseline-retentions", nargs="*", type=float, default=[])
    parser.add_argument("--text-budget", choices=("off", "128", "256", "512", "1024"), default="512")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-output-tokens", action="store_true")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.repetitions < 1 or args.warmup < 0 or args.max_visual_patches < 1:
        parser.error("invalid generation, repetition, warmup, or image-token limit")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.rank < args.world_size:
        parser.error("require 0 <= rank < world-size")
    if args.dataset_kind == "olmocr" and not args.scorer_training_manifest:
        parser.error("olmOCR requires --scorer-training-manifest")
    args.generated_budget = None if args.text_budget == "off" else int(args.text_budget)
    args.attention_implementation, args.dataset_splits = "sdpa", (args.split,)
    return args


def source_hashes():
    root = Path(__file__).parents[1]
    names = ["mlsys_sprint/adaptive_benchmark_v1.py", "mlsys_sprint/adaptive_vision_policy.py",
             "mlsys_sprint/adaptive_budget_assignments_v1.py",
             "mlsys_sprint/benchmark.py", "mlsys_next/entropy.py",
             "mlsys_scorer/models.py", "mlsys_scorer/backbone.py", "mlsys_scorer/protocol.py",
             "mlsys_scorer/checkpoints.py", "mlsys_eval/timing.py"]
    names += [f"mlsys_runtime/{name}.py" for name in ("generation", "cache", "masking", "policies")]
    return {name: digest_file(root / name) for name in names}


def main(argv=None):
    args = parse_args(argv)
    import torch
    from learned_score.mlsys_eval.common import dataset_family
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    from learned_score.mlsys_scorer.checkpoints import load_scorer, load_checkpoint

    policy_file_hash = digest_file(args.budget_policy)
    policy = AdaptiveVisionPolicy.load(args.budget_policy)
    if digest_file(args.budget_policy) != policy_file_hash:
        raise ValueError("adaptive policy changed while loading")
    checkpoint_hash = digest_file(args.checkpoint)
    checkpoint = load_checkpoint(args.checkpoint)
    benchmark.require_completed_checkpoint(args.checkpoint, checkpoint)
    policy.validate_identity(checkpoint_sha256=checkpoint_hash, scorer_variant=checkpoint["variant"],
                             model=checkpoint["identity"]["backbone"]["model"], adapter=args.adapter,
                             selector=args.selector)
    manifest = benchmark.load_benchmark_manifest(args)
    training_sha = (load_manifest(args.scorer_training_manifest).file_sha256
                    if args.scorer_training_manifest else manifest.file_sha256)
    entries = entries_for_split(manifest, args.split)
    if args.request_ids:
        requested = json.loads(args.request_ids.read_text())
        if not isinstance(requested, list) or not requested:
            raise ValueError("request subset must be a nonempty identity list")
        requested_ids = [item["row_hash"] for item in requested]
        if len(requested_ids) != len(set(requested_ids)):
            raise ValueError("duplicate request in explicit development subset")
        by_id = {entry["row_hash"]: entry for entry in entries}
        if set(requested_ids) - by_id.keys():
            raise ValueError("request subset includes identities outside the permitted development split")
        for item in requested:
            if any(item.get(key) != by_id[item["row_hash"]].get(key) for key in ("group_id", "image_sha256")):
                raise ValueError("request subset document/image identity mismatch")
        requested_set = set(requested_ids)
        entries = [entry for entry in entries if entry["row_hash"] in requested_set]
    assignments = None
    if args.budget_assignments:
        from .adaptive_budget_assignments_v1 import load_assignments
        assignments = load_assignments(args.budget_assignments, policy, manifest.file_sha256, entries)
        if args.limit:
            raise ValueError("--limit would break frozen aggregate token matching")
    if args.limit:
        entries = entries[:args.limit]
    entries = entries[args.rank::args.world_size]
    if not entries:
        raise ValueError("no development rows assigned to this worker")
    overlap = {entry["row_hash"]: policy.validate_evaluation_identity(
        row_hash=entry["row_hash"], group_id=entry.get("group_id", entry["image_sha256"]),
        image_sha256=entry["image_sha256"], split=args.split) for entry in entries}
    cells = make_cells(policy, generated_budget=args.generated_budget, selector=args.selector,
                       fixed_retentions=args.fixed_baseline_retentions)
    normalizer = None
    if not args.fixed_output_tokens and any("textvqa" in row["dataset"].lower() for row in entries):
        from vlmeval.dataset.utils.vqa_eval import process_answer
        normalizer = process_answer
    adapter, dataset, backbone = benchmark.load_benchmark_backbone(args, manifest)
    versions = frozen_versions(adapter.model)
    compatibility = benchmark.validate_checkpoint_identity(checkpoint, training_sha, backbone)
    policy.validate_identity(checkpoint_sha256=checkpoint_hash, scorer_variant=checkpoint["variant"],
                             model=backbone["model"], adapter=backbone["adapter"], selector=args.selector)
    scorer, loaded = load_scorer(args.checkpoint, args.device)
    if digest_file(args.checkpoint) != checkpoint_hash or loaded["identity_sha256"] != checkpoint["identity_sha256"]:
        raise ValueError("scorer checkpoint changed while loading")
    if digest_file(args.budget_policy) != policy_file_hash:
        raise ValueError("adaptive policy changed while loading the model")
    scorer.requires_grad_(False)
    scorer_versions = frozen_versions(scorer)
    scorer_variant = checkpoint["variant"]
    if args.selector == "scorer_coverage" and scorer_variant == "original":
        raise ValueError("coverage requires a rank32 checkpoint")
    hardware = (torch.cuda.get_device_name(adapter.device) if torch.device(adapter.device).type == "cuda"
                else platform.processor() or "CPU")
    identity = {"format": "mlsys_adaptive_vision_benchmark_v1", "manifest_sha256": manifest.file_sha256,
                "dataset_kind": args.dataset_kind, "backbone": backbone, "checkpoint_sha256": checkpoint_hash,
                "scorer_training_manifest_sha256": training_sha, "checkpoint_compatibility": compatibility,
                "budget_policy": policy.payload, "budget_policy_sha256": canonical_digest(policy.payload),
                "budget_policy_file_sha256": policy_file_hash,
                "source_sha256": source_hashes(), "hardware": hardware, "split": args.split,
                "batch_size": 1, "max_new_tokens": args.max_new_tokens, "slack": 1,
                "text_cache_policy": "protected_retained_image_and_prompt_plus_recent_generated_text",
                "physical_cache_budget": "retained_prefix+min(fixed_text_allowance+1,max_new_tokens-1)",
                "freed_visual_slots_donated_to_text": False, "aggregate_visual_tokens_reconciled": False,
                "stage": "development", "confirmation_claim": False,
                "fixed_output_tokens": args.fixed_output_tokens, "repetitions": args.repetitions,
                "warmup": args.warmup, "limit": args.limit, "world_size": args.world_size,
                "request_subset_sha256": digest_file(args.request_ids) if args.request_ids else None,
                "budget_assignments_sha256": digest_file(args.budget_assignments) if args.budget_assignments else None,
                "cells": [cell.key for cell in cells]}
    identity["aggregate_visual_tokens_reconciled"] = assignments is not None
    identity["aggregate_token_target"] = assignments["target_retention"] if assignments else None
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / f"rank{args.rank}.metadata.json"
    if metadata_path.exists():
        if not args.resume or json.loads(metadata_path.read_text()) != identity:
            raise ValueError("resume requires unchanged adaptive policy, data, code, and runtime identity")
    else:
        atomic_json(metadata_path, identity)
    for _ in range(args.warmup):
        warmup_row = read_entry(dataset, entries[0])
        for cell in cells:
            benchmark.measured_request(adapter, warmup_row, make_selector(cell, policy, scorer, scorer_variant,
                                       assignment=assignments["rows"][entries[0]["row_hash"]] if assignments else None),
                                       cell, max_new_tokens=min(args.max_new_tokens, 8), slack=1)
    completed = 0
    for entry in entries:
        row = read_entry(dataset, entry)
        references = benchmark.parse_references(row["answer"]) if args.dataset_kind == "vqa" else None
        metric = (manifest.metadata["official_metric"] if args.dataset_kind == "olmocr" else
                  "anls" if "docvqa" in entry["dataset"].lower() else "vqa_score")
        for repetition in range(args.repetitions):
            offset = (int(entry["row_hash"][:8], 16) + repetition) % len(cells)
            for cell in cells[offset:] + cells[:offset]:
                path = args.output / "rows" / f"{entry['row_hash']}.{cell.key}.repeat{repetition}.json"
                attempt = 0
                if path.exists():
                    previous = json.loads(path.read_text())
                    if not args.resume or previous.get("identity_sha256") != canonical_digest(identity):
                        raise ValueError("existing row does not match this adaptive run")
                    if previous.get("status") == "ok":
                        completed += 1
                        continue
                    if previous.get("attempt", 0) >= 1:
                        raise RuntimeError("failed row exhausted its one retry")
                    attempt = previous.get("attempt", 0) + 1
                observed = {}
                record = {"format": "mlsys_adaptive_request_result_v1", "identity_sha256": canonical_digest(identity),
                          "model": backbone["model"], "dataset": dataset_family(entry["dataset"]), "split": args.split,
                          "stage": "timing_diagnostic" if args.fixed_output_tokens else "development",
                          "request_id": entry["row_hash"], "group_id": entry.get("group_id", entry["image_sha256"]),
                          "image_sha256": entry["image_sha256"], "method": cell.key, "selector": cell.selector,
                          "vision_budget_mode": cell.vision_budget_mode, "generated_budget": cell.generated_budget,
                          "text_budget_mode": "disabled" if cell.generated_budget is None else "fixed",
                          "calibration_reuse": overlap[entry["row_hash"]], "quality_metric": metric,
                          "budget_policy_sha256": canonical_digest(policy.payload) if cell.vision_budget_mode == "adaptive" else None,
                          "aggregate_visual_tokens_reconciled": assignments is not None and cell.vision_budget_mode == "adaptive",
                          "hardware": hardware, "host": socket.gethostname(), "pbs_job_id": os.getenv("PBS_JOBID"), "batch_size": 1,
                          "serving_measurement": False, "max_new_tokens": args.max_new_tokens,
                          "repetition": repetition, "repeat": repetition + 1, "attempt": attempt}
                try:
                    prediction, ids, timing, runtime = benchmark.measured_request(
                        adapter, row, make_selector(cell, policy, scorer, scorer_variant, observed,
                        assignment=assignments["rows"][entry["row_hash"]] if assignments else None), cell,
                        max_new_tokens=args.max_new_tokens, slack=1, fixed_output_tokens=args.fixed_output_tokens)
                    assert_frozen(adapter.model, versions)
                    assert_frozen(scorer, scorer_versions)
                    score = (None if args.fixed_output_tokens or args.dataset_kind == "olmocr" else
                             benchmark.quality_score(prediction, references, metric, normalizer))
                    record.update({"status": "ok", "prediction": prediction, "quality_score": score,
                                   "quality_status": "timing_diagnostic_not_scored" if args.fixed_output_tokens else
                                   "awaiting_official_score" if args.dataset_kind == "olmocr" else "scored",
                                   "generated_tokens": int(ids.shape[-1]), "natural_stop_enabled": not args.fixed_output_tokens,
                                   "peak_allocated_bytes": runtime.get("full_request_peak_allocated_bytes"),
                                   "metrics": runtime, **timing, **observed})
                    del ids
                except Exception as error:
                    record.update({"status": "failed", "quality_score": None, "error_type": type(error).__name__,
                                   "error": str(error), **observed})
                    atomic_json(path, record)
                    raise
                atomic_json(path, record)
                completed += 1
                print(json.dumps({"event": "adaptive_measurement", "request_id": entry["row_hash"],
                                  "method": cell.key, "retained_visual_tokens": record["retained_visual_tokens"]}), flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json",
                {"identity_sha256": canonical_digest(identity), "request_results": completed,
                 "requests": len(entries), "cells": len(cells), "repetitions": args.repetitions})


if __name__ == "__main__":
    main()
