"""Versioned physical measurements for the one-statistic allocation study.

Features never contain answers. Calibration labels are separate atomic files.
All evaluated selectors share the same protected-prefix cache and decoder.
"""

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import socket
import time

from . import benchmark
from .adaptive_benchmark_v1 import select_exact_count
from .kv_target_benchmark_v2 import contract_from_config
from .state import atomic_json, digest, file_digest


FORMAT = "mlsys_adaptive_experiment_v3"
KNOTS = (10, 25, 40, 55, 70)
TARGETS = (60, 40, 50, 70)


@dataclass(frozen=True)
class Cell:
    selector: str
    allocation: str
    percent: int
    generated_budget: int = 512

    @property
    def key(self):
        return f"{self.selector}_{self.allocation}_{self.percent:03d}"

    @property
    def method(self):
        return self.key


def cells_for(phase, assignments=None, targets=TARGETS):
    if phase == "features":
        return []
    if phase == "calibration":
        return [Cell("scorer", "retention", knot) for knot in KNOTS]
    if phase not in {"screening", "evaluation"}:
        raise ValueError("unknown measurement phase")
    if assignments is None:
        raise ValueError("physical evaluation needs frozen exact assignments")
    selected = [target for target in targets if str(target) in assignments["targets"]]
    if phase == "screening":
        selected = [target for target in selected if target in (60, 70)]
    methods = [("scorer", "adaptive"), ("scorer", "uniform")]
    if phase == "evaluation":
        methods.extend((("divprune", "uniform"), ("cdpruner", "uniform")))
    return ([] if phase == "screening" else [Cell("full", "full", 0)]) + [
        Cell(selector, allocation, target) for target in selected
        for selector, allocation in methods]


def score_prediction(prediction, references, metric):
    if metric in {"anls", "vqa", "vqa_score"}:
        return benchmark.quality_score(prediction, references, "vqa_score" if metric == "vqa" else metric)
    if not references:
        raise ValueError("quality evaluation requires references")
    if metric == "relaxed":
        from vlmeval.dataset.utils.vqa_eval import relaxed_correctness
        return float(max(relaxed_correctness(answer, prediction) for answer in references))
    if metric == "mme":
        from vlmeval.dataset.utils.yorn import YOrN_Extraction
        answer = YOrN_Extraction(prediction)
        return float(any(answer.lower() == reference.strip().lower() for reference in references))
    raise ValueError(f"unsupported metric {metric}")


def feature_summary(prepared, scorer):
    import torch
    from torch.nn import functional as F
    from .adaptive_utility_v3 import calculate_statistic
    vision = prepared.inputs_embeds[0, prepared.visual_positions]
    question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
    with torch.no_grad():
        scores = scorer(vision, question).float()
        n = len(scores)
        if not n or not bool(torch.isfinite(scores).all()):
            raise ValueError("empty or nonfinite scorer scores")
        standardized = (scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-8)
        probabilities = F.softmax(standardized, dim=0).sort(descending=True).values
        cumulative = torch.cat((probabilities.new_zeros(1), probabilities.cumsum(0)))
        mass = [float(cumulative[n * percent // 100]) for percent in range(101)]
        return {
            "input_visual_tokens": n,
            "input_text_tokens": int(prepared.inputs_embeds.shape[1] - n),
            "normalized_entropy": calculate_statistic(scores, "normalized_entropy"),
            "nearest_neighbor_diversity": calculate_statistic(scores, "nearest_neighbor_diversity", vision=vision),
            "top_mass_by_percent": mass,
        }


def select_for(cell, scorer, feature, assigned=None, observed=None):
    import torch
    from torch.nn import functional as F

    @torch.no_grad()
    def select(prepared):
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        n, p = len(vision), int(prepared.inputs_embeds.shape[1] - len(vision))
        if (n, p) != (feature["input_visual_tokens"], feature["input_text_tokens"]):
            raise ValueError(f"input geometry changed: live={(n,p)}, frozen={(feature['input_visual_tokens'],feature['input_text_tokens'])}")
        if cell.allocation == "full":
            count = n
        elif cell.allocation == "retention":
            count = max(1, n * cell.percent // 100)
        else:
            if assigned is None:
                raise ValueError("missing immutable assignment")
            count = assigned[f"{cell.allocation}_selected_count"]
        if type(count) is not int or not 1 <= count <= n:
            raise ValueError("invalid retained token count")
        if cell.selector == "full":
            indices = torch.arange(n, device=vision.device)
        elif cell.selector == "scorer":
            indices = select_exact_count(scorer(vision, question), vision, count)
        elif cell.selector == "divprune":
            from learned_score.llava_selectors import divprune_order
            indices = divprune_order(vision, count).sort().values
        elif cell.selector == "cdpruner":
            from .stable_cdpruner import stable_cdpruner_order
            relevance = (F.normalize(vision.float(), dim=-1) @ F.normalize(question.float(), dim=-1).T).max(dim=-1).values
            indices = stable_cdpruner_order(vision, relevance, count).sort().values
        else:
            raise ValueError("unknown patch selector")
        if (indices.dtype != torch.long or indices.ndim != 1 or len(indices) != count
                or len(indices.unique()) != count or int(indices.min()) < 0 or int(indices.max()) >= n
                or bool((indices[1:] <= indices[:-1]).any())):
            raise ValueError("selected indices violate exact count or original ordering")
        if observed is not None:
            observed.update(input_visual_tokens=n, input_text_tokens=p,
                            retained_visual_tokens=count, visual_retention=count/n)
        return indices
    return select


def expected_cache_bytes(feature, cell, contract, assigned=None):
    n, p = feature["input_visual_tokens"], feature["input_text_tokens"]
    count = n if cell.allocation == "full" else (
        max(1, n * cell.percent // 100) if cell.allocation == "retention"
        else assigned[f"{cell.allocation}_selected_count"])
    return (count + p + contract["reserved_generated_slots"]) * contract["total_owned_bytes_per_slot"]


def check_cache(runtime, expected):
    cache = runtime["cache"]
    if (cache.get("total_owned_bytes") != expected or runtime.get("peak_cache_bytes") != expected
            or cache.get("scratch_bytes") != 0 or cache.get("compaction_events") != 0):
        raise ValueError(f"cache mismatch: expected={expected}, measured={cache}")


def load_runtime(args, manifest):
    from .adaptive_data_v3 import ManifestDatasetV3
    from learned_score.mlsys_scorer import backbone
    from learned_score.mlsys_scorer.checkpoints import load_checkpoint, load_scorer
    if args.adapter == "phi3_vision":
        from .phi_adaptive_v3 import load_phi_backbone
        adapter, dataset, identity = load_phi_backbone(args, manifest)
    else:
        # Preserve the trained checkpoint's original integration identity. The
        # evaluation dataset changes, never the frozen backbone implementation.
        from learned_score.mlsys_scorer.protocol import load_manifest
        original = load_manifest(args.training_manifest)
        adapter, _, identity = backbone.load_backbone(args, original)
        dataset = ManifestDatasetV3(args.manifest, allowed_splits=(args.split,))
        adapter.dataset = dataset
    checkpoint = load_checkpoint(args.checkpoint)
    benchmark.require_completed_checkpoint(args.checkpoint, checkpoint)
    if args.adapter != "phi3_vision":
        benchmark.validate_checkpoint_identity(checkpoint, file_digest(args.training_manifest), identity)
    else:
        expected = checkpoint["identity"]["backbone"]
        for key in ("model", "adapter", "requested_revision", "resolution", "backbone_dtype"):
            if expected.get(key) != identity.get(key):
                raise ValueError(f"Phi checkpoint/runtime {key} mismatch")
    scorer, _ = load_scorer(args.checkpoint, args.device)
    if scorer.hidden_size != adapter.hidden_size:
        raise ValueError("scorer hidden dimension differs from VLM")
    scorer.requires_grad_(False)
    return adapter, dataset, identity, scorer


def measured(adapter, row, selector, cell, max_new_tokens):
    run = None
    if adapter.name == "phi3_vision":
        from learned_score.mlsys_runtime import run_request
        def run(*args, **kwargs):
            kwargs["require_fused_attention"] = False
            return run_request(*args, **kwargs)
    return benchmark.measured_request(adapter, row, selector, cell,
        max_new_tokens=max_new_tokens, slack=1, run=run)


def source_hashes(adapter):
    from .adaptive_benchmark_v1 import source_hashes as previous
    root = Path(__file__).parents[1]
    names = set(previous()) | {"mlsys_sprint/adaptive_experiment_v3.py",
        "mlsys_sprint/adaptive_data_v3.py", "mlsys_sprint/adaptive_utility_v3.py",
        "mlsys_sprint/kv_target_benchmark_v2.py", "mlsys_sprint/kv_target_assignments_v2.py",
        "mlsys_sprint/stable_cdpruner.py", "llava_selectors.py"}
    if adapter == "phi3_vision":
        names.add("mlsys_sprint/phi_adaptive_v3.py")
    return {name: file_digest(root/name) for name in sorted(names)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("features", "calibration", "screening", "evaluation"))
    parser.add_argument("--split", required=True, choices=("fit", "screening", "evaluation"))
    for name in ("manifest", "checkpoint", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, default=Path("reproduction_results/mlsys_20261012/manifest.json"))
    parser.add_argument("--assignments", type=Path)
    parser.add_argument("--feature-root", type=Path)
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next", "phi3_vision"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--num-crops", type=int, default=16)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not 0 <= args.rank < args.world_size or args.warmup < 0 or (args.limit is not None and args.limit < 1):
        parser.error("invalid rank/warmup/limit")
    if args.phase in {"screening", "evaluation"} and (args.assignments is None or args.feature_root is None):
        parser.error("evaluation requires assignments and feature-root")
    if args.phase == "calibration" and args.split != "fit":
        parser.error("calibration labels may only be generated on fit rows")
    args.backbone_dtype = "bfloat16"
    args.attention_implementation = "sdpa" if args.adapter != "phi3_vision" else "eager"
    args.dataset_splits = (args.split,) if args.adapter == "phi3_vision" else ("scorer_dev",)
    return args


def _immutable(path, payload, *, resume):
    if path.exists():
        if not resume or json.loads(path.read_text()) != payload:
            raise ValueError(f"immutable metadata mismatch: {path}")
    else:
        atomic_json(path, payload)


def main(argv=None):
    args = parse_args(argv)
    import torch
    from .adaptive_data_v3 import load_manifest
    from learned_score.mlsys_scorer.backbone import frozen_versions, assert_frozen
    manifest = load_manifest(args.manifest)
    entries = sorted(manifest.splits[args.split], key=lambda entry: (entry["dataset"], entry["row_hash"]))
    if args.limit:
        entries = entries[:args.limit]
    entries = entries[args.rank::args.world_size]
    if not entries:
        raise ValueError("empty experiment shard")
    assignments = json.loads(args.assignments.read_text()) if args.assignments else None
    if assignments is not None:
        if assignments.get("sha256") != digest({key: value for key, value in assignments.items() if key != "sha256"}):
            raise ValueError("assignment content hash mismatch")
        if assignments.get("manifest_sha256") != manifest.file_sha256 or assignments.get("split") != args.split:
            raise ValueError("assignment manifest/split mismatch")
        if assignments.get("checkpoint_sha256") != file_digest(args.checkpoint):
            raise ValueError("assignment scorer differs")
    cells = cells_for(args.phase, assignments)
    adapter, dataset, backbone, scorer = load_runtime(args, manifest)
    contract = contract_from_config(adapter.model.config)
    if assignments is not None and assignments.get("cache_contract") != contract:
        raise ValueError("assignment cache dimensions differ")
    versions, scorer_versions = frozen_versions(adapter.model), frozen_versions(scorer)
    identity = {"format": FORMAT, "manifest_sha256": manifest.file_sha256,
        "checkpoint_sha256": file_digest(args.checkpoint), "backbone": backbone,
        "phase": args.phase, "split": args.split, "world_size": args.world_size,
        "limit": args.limit, "warmup": args.warmup, "source_sha256": source_hashes(args.adapter),
        "cache_contract": contract, "hardware": torch.cuda.get_device_name(adapter.device),
        "max_new_tokens": 128, "cells": [cell.key for cell in cells],
        "assignments_sha256": file_digest(args.assignments) if args.assignments else None,
        "feature_root": str(args.feature_root) if args.feature_root else None,
        "baseline_scope": "projected_feature_adaptations_not_official_pipeline",
        "timing_isolated": False, "text_eviction_claim": False}
    _immutable(args.output/f"rank{args.rank}.metadata.json", identity, resume=args.resume)
    identity_sha = digest(identity)
    completed = 0
    started = time.monotonic()
    for row_index, entry in enumerate(entries):
        row = dataset.row_for_entry(entry)
        if hasattr(row.get("image"), "seek"):
            row["image"].seek(0)
        with torch.no_grad():
            prepared = adapter.prepare_sample(row, include_answer=False)
            features = feature_summary(prepared, scorer)
        del prepared
        feature = {"format": "mlsys_answer_free_features_v3", "request_id": entry["row_hash"],
            "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
            "dataset": entry["dataset"], "split": args.split, "identity_sha256": identity_sha,
            "checkpoint_sha256": identity["checkpoint_sha256"], "cache_contract": contract,
            "answers_consumed_by_feature_extractor": False, **features}
        feature_path = args.output/"features"/(entry["row_hash"]+".json")
        if args.feature_root:
            saved = json.loads((args.feature_root/"features"/(entry["row_hash"]+".json")).read_text())
            differences = {key: abs(features[key]-saved[key]) for key in ("normalized_entropy", "nearest_neighbor_diversity")}
            mass_difference = max(abs(a-b) for a,b in zip(features["top_mass_by_percent"], saved["top_mass_by_percent"]))
            if (saved["checkpoint_sha256"] != identity["checkpoint_sha256"]
                    or any(saved[key] != features[key] for key in ("input_visual_tokens", "input_text_tokens"))
                    or max(*differences.values(), mass_difference) > 1e-6):
                atomic_json(args.output/f"rank{args.rank}.feature_mismatch.json", {
                    "request_id": entry["row_hash"], "live": features,
                    "saved": {key: saved[key] for key in features}, "differences": differences,
                    "max_mass_difference": mass_difference, "hardware": identity["hardware"], "host": socket.gethostname()})
                raise ValueError("live features differ from frozen features; see explicit mismatch diagnostic")
        if feature_path.exists():
            previous = json.loads(feature_path.read_text())
            if not args.resume or previous != feature:
                raise ValueError("feature resume mismatch")
        else:
            atomic_json(feature_path, feature)
        if args.phase == "features":
            completed += 1
            continue
        references = benchmark.parse_references(row["answer"])
        metric = entry.get("quality_metric", entry.get("metric"))
        offset = int(entry["row_hash"][:8], 16) % len(cells)
        for cell in cells[offset:] + cells[:offset]:
            if cell.allocation in {"full", "retention"}:
                assigned = None
            else:
                target = assignments["targets"][str(cell.percent)]
                if entry["row_hash"] not in target["rows"]:
                    if entry["dataset"] in target.get("infeasible_datasets", {}):
                        continue
                    raise ValueError("missing exact assignment")
                assigned = target["rows"][entry["row_hash"]]
            path = args.output/"rows"/f"{entry['row_hash']}.{cell.key}.json"
            if path.exists():
                previous = json.loads(path.read_text())
                if not args.resume or previous.get("identity_sha256") != identity_sha:
                    raise ValueError("result resume identity mismatch")
                if previous["status"] == "ok":
                    completed += 1
                    continue
                if previous.get("attempt", 0) >= 1:
                    raise RuntimeError("physical result exhausted one retry")
                attempt = previous.get("attempt", 0)+1
            else:
                attempt = 0
            observed = {}
            record = {"format": FORMAT, "identity_sha256": identity_sha,
                "request_id": entry["row_hash"], "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                "dataset": entry["dataset"], "category": entry.get("category"), "model": backbone["model"],
                "method": cell.key, "selector": cell.selector, "allocation": cell.allocation,
                "percent": cell.percent, "phase": args.phase, "quality_metric": metric,
                "evidence_status": entry.get("evidence_status", entry.get("evaluation_stage", "development")),
                "host": socket.gethostname(), "job_id": os.getenv("PBS_JOBID"),
                "hardware": identity["hardware"], "timing_isolated": False, "attempt": attempt,
                "assignment": assigned, "max_new_tokens": 128, "generated_budget": 512}
            try:
                if row_index == 0 and args.warmup:
                    for _ in range(args.warmup):
                        measured(adapter, row, select_for(cell, scorer, feature, assigned), cell, 8)
                prediction, ids, timing, runtime = measured(adapter, row,
                    select_for(cell, scorer, feature, assigned, observed), cell, 128)
                expected = expected_cache_bytes(feature, cell, contract, assigned)
                check_cache(runtime, expected)
                assert_frozen(adapter.model, versions)
                assert_frozen(scorer, scorer_versions)
                record.update(status="ok", prediction=prediction,
                    quality_score=score_prediction(prediction, references, metric),
                    generated_tokens=int(ids.shape[-1]), expected_cache_bytes=expected,
                    metrics=runtime, **timing, **observed)
                del ids
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__, error=str(error), quality_score=None)
                atomic_json(path, record)
                raise
            atomic_json(path, record)
            completed += 1
        print(json.dumps({"event": "adaptive_v3_progress", "phase": args.phase,
            "rank": args.rank, "requests": row_index+1, "results": completed,
            "elapsed_seconds": time.monotonic()-started}), flush=True)
    atomic_json(args.output/f"rank{args.rank}.complete.json", {
        "identity_sha256": identity_sha, "requests": len(entries), "results": completed,
        "elapsed_seconds": time.monotonic()-started})


if __name__ == "__main__":
    main()
