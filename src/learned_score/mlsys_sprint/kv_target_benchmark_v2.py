"""Physical development comparison at explicitly matched allocated KV bytes.

The entropy rule is transferred, without new quality fitting, to 40/50/60%
cache-reduction targets. All methods use the same bounded-cache runtime.
DivPrune and CDPruner are projected-feature adaptations, not official full
pipeline reproductions. Existing adaptive/cache experiments remain untouched.
"""

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket

from . import adaptive_benchmark_v1 as previous
from . import benchmark
from .adaptive_vision_policy import AdaptiveVisionPolicy
from .state import atomic_json, digest, file_digest
from learned_score.mlsys_scorer.protocol import entries_for_split, load_manifest


FORMAT = "mlsys_kv_target_benchmark_v2"
ROW_FORMAT = "mlsys_kv_target_request_v2"
METHOD_LABELS = {
    "full": "All image tokens, shared cache runtime",
    "scorer_adaptive": "AsymVLM adaptive entropy budget",
    "scorer_uniform": "AsymVLM uniform budget",
    "divprune_uniform": "DivPrune projected-feature adaptation, uniform budget",
    "cdpruner_uniform": "CDPruner projected-relevance stable FP64 adaptation, uniform budget",
}


@dataclass(frozen=True)
class Cell:
    selector: str
    allocation_mode: str
    target_percent: int
    generated_budget: int = 512

    @property
    def key(self):
        return f"{self.selector}_{self.allocation_mode}_kv{self.target_percent}_text{self.generated_budget}"

    @property
    def method(self):
        return self.key

    @property
    def label(self):
        return METHOD_LABELS["full" if self.selector == "full" else f"{self.selector}_{self.allocation_mode}"]


def make_cells(targets=(40, 50, 60)):
    if tuple(targets) != (40, 50, 60):
        raise ValueError("v2 requires the predeclared 40/50/60% allocated-cache targets")
    return [Cell("full", "all", 0)] + [Cell(selector, mode, target)
        for target in targets for selector, mode in (
            ("scorer", "adaptive"), ("scorer", "uniform"),
            ("divprune", "uniform"), ("cdpruner", "uniform"))]


def contract_from_config(config, *, max_new_tokens=128):
    from .kv_target_assignments_v2 import make_cache_contract
    config = getattr(config, "text_config", config)
    return make_cache_contract(
        num_hidden_layers=config.num_hidden_layers,
        num_key_value_heads=getattr(config, "num_key_value_heads", config.num_attention_heads),
        head_dim=getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads,
        element_size_bytes=2, max_new_tokens=max_new_tokens, generated_budget=512, slack=1)


def make_selector(cell, policy, scorer, assigned, observed=None):
    import torch
    from torch.nn import functional as F

    @torch.no_grad()
    def select(prepared):
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        n = len(vision)
        text_count = int(prepared.inputs_embeds.shape[1] - n)
        if (n != assigned["input_visual_tokens"] or text_count != assigned["input_text_tokens"]
                or not n or not bool(torch.isfinite(vision).all())
                or not bool(torch.isfinite(question).all())):
            raise ValueError("live image/prompt inputs differ from frozen cache assignment")
        decision = None
        if cell.allocation_mode == "all":
            count = n
            selected = torch.arange(n, dtype=torch.long, device=vision.device)
        else:
            count = assigned[f"{cell.allocation_mode}_selected_count"]
            if type(count) is not int or not 1 <= count <= n:
                raise ValueError("invalid exact assigned image count")
            if cell.selector == "scorer":
                scores = scorer(vision, question)
                if cell.allocation_mode == "adaptive":
                    decision = policy.decide(scores)
                    if (abs(decision.statistic_value - assigned["statistic_value"]) > 1e-6
                            or decision.fallback != assigned["fallback"]):
                        raise ValueError("live entropy/support differs from frozen cache assignment")
                selected = previous.select_exact_count(scores, vision, count)
            elif cell.selector == "divprune":
                from learned_score.llava_selectors import divprune_order
                selected = divprune_order(vision, count).sort().values
            elif cell.selector == "cdpruner":
                from .stable_cdpruner import stable_cdpruner_order
                if not len(question):
                    raise ValueError("CDPruner adaptation requires nonempty question embeddings")
                relevance = (F.normalize(vision.float(), dim=-1)
                             @ F.normalize(question.float(), dim=-1).T).max(dim=-1).values
                selected = stable_cdpruner_order(vision, relevance, count).sort().values
            else:
                raise ValueError("unsupported exact-budget selector")
        if (selected.ndim != 1 or selected.dtype != torch.long or len(selected) != count
                or len(selected.unique()) != count or int(selected.min()) < 0
                or int(selected.max()) >= n or bool((selected[1:] <= selected[:-1]).any())):
            raise RuntimeError("selector violated exact count, uniqueness, or original patch order")
        if observed is not None:
            observed.update(input_visual_tokens=n, input_text_tokens=text_count,
                            input_prefix_tokens=n + text_count, retained_visual_tokens=count,
                            visual_retention=count / n,
                            adaptive_decision=decision.asdict() if decision else None)
        return selected
    return select


def verify_cache(runtime, assigned, cell):
    expected = assigned[f"{cell.allocation_mode if cell.allocation_mode != 'all' else 'full'}_expected_cache_bytes"]
    cache = runtime["cache"]
    if (runtime.get("peak_cache_bytes") != expected or cache.get("total_owned_bytes") != expected
            or cache.get("scratch_bytes") != 0 or cache.get("compaction_events") != 0):
        raise ValueError("physical allocated cache differs from frozen byte target")
    return expected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "checkpoint", "budget-policy", "request-ids", "kv-target-assignments", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--adapter", required=True, choices=("qwen2_5_vl", "llava_next"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, choices=(128,), default=128)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.rank < args.world_size or args.warmup < 0 or args.max_visual_patches < 1:
        parser.error("invalid rank, warmup, or image resolution")
    args.dataset_kind, args.split, args.dataset_splits = "vqa", "scorer_dev", ("scorer_dev",)
    args.backbone_dtype, args.attention_implementation = "bfloat16", "sdpa"
    return args


def source_hashes():
    root = Path(__file__).parents[1]
    names = set(previous.source_hashes()) | {
        "mlsys_sprint/kv_target_benchmark_v2.py", "mlsys_sprint/kv_target_assignments_v2.py",
        "mlsys_sprint/stable_cdpruner.py", "llava_selectors.py"}
    return {name: file_digest(root / name) for name in sorted(names)}


def main(argv=None):
    args = parse_args(argv)
    import torch
    from .kv_target_assignments_v2 import load_assignments
    from learned_score.mlsys_eval.common import dataset_family
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    from learned_score.mlsys_scorer.checkpoints import load_checkpoint, load_scorer

    frozen_files = {str(p): file_digest(p) for p in (
        args.budget_policy, args.request_ids, args.kv_target_assignments, args.checkpoint)}
    policy = AdaptiveVisionPolicy.load(args.budget_policy)
    checkpoint = load_checkpoint(args.checkpoint)
    benchmark.require_completed_checkpoint(args.checkpoint, checkpoint)
    policy.validate_identity(checkpoint_sha256=frozen_files[str(args.checkpoint)],
        scorer_variant=checkpoint["variant"], model=checkpoint["identity"]["backbone"]["model"],
        adapter=args.adapter, selector="scorer")
    manifest = load_manifest(args.manifest)
    requested = json.loads(args.request_ids.read_text())
    wanted = {item["row_hash"]: item for item in requested}
    if len(wanted) != len(requested) or not wanted:
        raise ValueError("request subset must contain unique development identities")
    entries = [e for e in entries_for_split(manifest, args.split) if e["row_hash"] in wanted]
    if {e["row_hash"] for e in entries} != set(wanted):
        raise ValueError("requested identity is outside permitted development split")
    for entry in entries:
        if any(entry[k] != wanted[entry["row_hash"]][k] for k in ("group_id", "image_sha256")):
            raise ValueError("development subset identity mismatch")
        if policy.validate_evaluation_identity(row_hash=entry["row_hash"], group_id=entry["group_id"],
                image_sha256=entry["image_sha256"], split=args.split):
            raise ValueError("evaluation row overlaps policy fitting")
    assignments = load_assignments(args.kv_target_assignments, policy, manifest.file_sha256, entries)
    cells = make_cells(tuple(sorted(int(t) for t in assignments["targets"])))
    entries = entries[args.rank::args.world_size]
    if not entries:
        raise ValueError("empty worker shard")
    from vlmeval.dataset.utils.vqa_eval import process_answer
    adapter, dataset, backbone = benchmark.load_benchmark_backbone(args, manifest)
    versions = frozen_versions(adapter.model)
    compatibility = benchmark.validate_checkpoint_identity(checkpoint, manifest.file_sha256, backbone)
    if contract_from_config(adapter.model.config) != assignments["cache_contract"]:
        raise ValueError("live model/cache dimensions differ from frozen byte accounting")
    scorer, loaded = load_scorer(args.checkpoint, args.device)
    if loaded["identity_sha256"] != checkpoint["identity_sha256"]:
        raise ValueError("checkpoint changed while loading")
    scorer.requires_grad_(False)
    scorer_versions = frozen_versions(scorer)
    if any(file_digest(Path(p)) != h for p, h in frozen_files.items()):
        raise ValueError("frozen experimental input changed during model loading")
    hardware = torch.cuda.get_device_name(adapter.device)
    identity = {"format": FORMAT, "manifest_sha256": manifest.file_sha256,
        "checkpoint_sha256": frozen_files[str(args.checkpoint)], "backbone": backbone,
        "checkpoint_compatibility": compatibility, "budget_policy": policy.payload,
        "budget_policy_sha256": digest(policy.payload),
        "budget_policy_file_sha256": frozen_files[str(args.budget_policy)],
        "request_subset_sha256": frozen_files[str(args.request_ids)],
        "kv_target_assignments_sha256": frozen_files[str(args.kv_target_assignments)],
        "cache_contract": assignments["cache_contract"], "source_sha256": source_hashes(),
        "hardware": hardware, "split": args.split, "stage": "development", "confirmation_claim": False,
        "policy_scope": "frozen_entropy_rule_transfer_no_target_specific_quality_fit",
        "baseline_scope": "projected_feature_selector_adaptations_shared_cache_runtime",
        "batch_size": 1, "max_new_tokens": args.max_new_tokens, "slack": 1,
        "generated_budget": 512, "repetitions": 1, "world_size": args.world_size,
        "warmup": args.warmup, "cells": [c.key for c in cells], "labels": {c.key: c.label for c in cells}}
    metadata_path = args.output / f"rank{args.rank}.metadata.json"
    if metadata_path.exists():
        if not args.resume or json.loads(metadata_path.read_text()) != identity:
            raise ValueError("resume requires identical source, policy, inputs, hardware, and runtime")
    else:
        atomic_json(metadata_path, identity)

    def assignment_for(cell, entry):
        # Every target records the same full-input context and baseline bytes.
        return assignments["targets"][str(cell.target_percent or 40)]["rows"][entry["row_hash"]]

    for _ in range(args.warmup):
        row = read_entry(dataset, entries[0])
        for cell in cells:
            benchmark.measured_request(adapter, row, make_selector(cell, policy, scorer,
                assignment_for(cell, entries[0])), cell, max_new_tokens=8, slack=1)
    completed = 0
    for entry in entries:
        row = read_entry(dataset, entry)
        references = benchmark.parse_references(row["answer"])
        metric = "anls" if "docvqa" in entry["dataset"].lower() else "vqa_score"
        offset = int(entry["row_hash"][:8], 16) % len(cells)
        for cell in cells[offset:] + cells[:offset]:
            path = args.output / "rows" / f"{entry['row_hash']}.{cell.key}.repeat0.json"
            attempt = 0
            if path.exists():
                prior = json.loads(path.read_text())
                if not args.resume or prior.get("identity_sha256") != digest(identity):
                    raise ValueError("existing output differs from frozen experimental identity")
                if prior.get("status") == "ok":
                    completed += 1
                    continue
                if prior.get("attempt", 0) >= 1:
                    raise RuntimeError("physical row exhausted its single retry")
                attempt = prior.get("attempt", 0) + 1
            observed, assigned = {}, assignment_for(cell, entry)
            record = {"format": ROW_FORMAT, "identity_sha256": digest(identity),
                "model": backbone["model"], "dataset": dataset_family(entry["dataset"]),
                "request_id": entry["row_hash"], "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                "split": args.split, "stage": "development", "confirmation_claim": False,
                "method": cell.key, "method_label": cell.label, "selector": cell.selector,
                "allocation_mode": cell.allocation_mode, "target_reduction": cell.target_percent / 100,
                "generated_budget": 512, "max_new_tokens": args.max_new_tokens, "quality_metric": metric,
                "policy_scope": identity["policy_scope"], "calibration_reuse": False,
                "hardware": hardware, "host": socket.gethostname(), "pbs_job_id": os.getenv("PBS_JOBID"),
                "serving_measurement": False, "timing_isolated": False, "batch_size": 1,
                "repetition": 0, "repeat": 1, "attempt": attempt}
            try:
                prediction, ids, timing, runtime = benchmark.measured_request(adapter, row,
                    make_selector(cell, policy, scorer, assigned, observed), cell,
                    max_new_tokens=args.max_new_tokens, slack=1)
                expected_bytes = verify_cache(runtime, assigned, cell)
                assert_frozen(adapter.model, versions)
                assert_frozen(scorer, scorer_versions)
                record.update(status="ok", prediction=prediction,
                    quality_score=benchmark.quality_score(prediction, references, metric, process_answer),
                    quality_status="scored", generated_tokens=int(ids.shape[-1]),
                    expected_cache_bytes=expected_bytes, metrics=runtime,
                    peak_allocated_bytes=runtime["full_request_peak_allocated_bytes"],
                    natural_stop_enabled=True, **timing, **observed)
                del ids
            except Exception as error:
                record.update(status="failed", quality_score=None, error_type=type(error).__name__, error=str(error), **observed)
                atomic_json(path, record)
                raise
            atomic_json(path, record)
            completed += 1
            print(json.dumps({"event": "kv_target_measurement", "request_id": entry["row_hash"],
                "method": cell.key, "cache_bytes": expected_bytes}), flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json", {
        "identity_sha256": digest(identity), "requests": len(entries), "request_results": completed,
        "cells": len(cells), "repetitions": 1})


if __name__ == "__main__":
    main()
