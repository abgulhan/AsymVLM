"""Batch-one, end-to-end development benchmark for joint visual/text budgets.

Example: python -m learned_score.mlsys_sprint.benchmark --manifest manifest.json
  --adapter qwen2_5_vl --model /pinned/model/snapshot --output results
  --selectors full divprune cdpruner --limit 2 --max-new-tokens 32

This is an isolated request benchmark, not a serving/continuous-batching test.
"""

import argparse
import ast
import copy
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

from learned_score.mlsys_scorer.checkpoints import atomic_json, canonical_digest
from learned_score.mlsys_scorer.protocol import digest_file, entries_for_split, load_manifest


SELECTORS = ("full", "divprune", "cdpruner", "scorer", "scorer_coverage")
RETENTIONS = (0.7, 0.9, 1.0)
ALLOWED_RETENTIONS = (0.5, *RETENTIONS)
TEXT_BUDGETS = (None, 256, 512)


@dataclass(frozen=True)
class Cell:
    selector: str
    visual_retention: float
    generated_budget: int | None

    @property
    def key(self):
        budget = "off" if self.generated_budget is None else self.generated_budget
        return f"{self.selector}_v{int(round(self.visual_retention * 100))}_t{budget}"

    @property
    def method(self):
        if self.selector == "full" and self.generated_budget is None:
            return "all_tokens"
        return self.key


def make_grid(selectors, retentions, text_budgets):
    if not selectors or any(value not in SELECTORS for value in selectors):
        raise ValueError("unknown/empty selector menu")
    if not retentions or any(value not in ALLOWED_RETENTIONS for value in retentions):
        raise ValueError("visual retention must be .5, .7, .9 or 1.0")
    if not text_budgets or any(value not in TEXT_BUDGETS for value in text_budgets):
        raise ValueError("text budget must be off, 256 or 512")
    # Every run includes an actual, paired all-token reference.
    cells = [Cell("full", 1.0, None)]
    for selector in dict.fromkeys(selectors):
        ratios = [1.0] if selector == "full" else retentions
        for ratio in dict.fromkeys(ratios):
            for budget in dict.fromkeys(text_budgets):
                cell = Cell(selector, ratio, budget)
                if cell not in cells:
                    cells.append(cell)
    return cells


def validate_checkpoint_identity(checkpoint, manifest_sha256, backbone):
    """Inference can use SDPA while the scorer's training used eager attention."""
    identity = checkpoint["identity"]
    if identity.get("manifest_sha256") != manifest_sha256:
        raise ValueError("checkpoint training manifest differs from benchmark manifest")
    expected = dict(identity.get("backbone", {}))
    actual = dict(backbone)
    memory = expected.pop("training_memory_strategy", None)
    if memory is not None:
        from learned_score.mlsys_scorer.memory_training import memory_strategy_identity
        if memory != memory_strategy_identity(memory.get("strategy")):
            raise ValueError("training-memory strategy or launcher source is not verified")
    expected.pop("attention_implementation", None)
    actual.pop("attention_implementation", None)
    # Released checkpoints are metadata-sanitized: model paths were rewritten to ${HF_HOME}/... and the
    # source tree was renamed for anonymity. Compare everything else (revision, resolution, dtype, ...).
    for key in [k for k, v in expected.items() if isinstance(v, str) and v.startswith("${HF_HOME}")]:
        expected.pop(key, None); actual.pop(key, None)
    for key in ("source_hashes", "source_sha256"):
        expected.pop(key, None); actual.pop(key, None)
    if expected != actual:
        raise ValueError("checkpoint model, resolution, dtype, revision or source hashes differ")
    return {"training_attention": identity["backbone"].get("attention_implementation"),
            "inference_attention": backbone.get("attention_implementation"),
            "allowed_difference": "attention_backend_only" if memory is None else "attention_backend_and_verified_training_memory_only",
            "verified_training_memory_strategy": memory}


def require_completed_checkpoint(path, checkpoint):
    """A mutable mid-training checkpoint is not a production benchmark input."""
    marker = Path(path).with_suffix(".complete.json")
    if not marker.is_file():
        marker = Path(path).parent / "complete.json"
    if not marker.is_file():
        raise ValueError("scorer training has no complete.json readiness marker")
    completion = json.loads(marker.read_text())
    if (completion.get("identity_sha256") != checkpoint.get("identity_sha256")
            or completion.get("steps") != checkpoint.get("steps")
            or checkpoint.get("epoch") != checkpoint["identity"]["epochs"]
            or checkpoint.get("next_index") != 0):
        raise ValueError("scorer completion marker does not match a completed checkpoint")


def parse_references(answer):
    """Read lists without executing dataset text; preserve annotator duplicates."""
    if isinstance(answer, (list, tuple)):
        values = answer
    else:
        value = str(answer).strip()
        if value.startswith("[") and value.endswith("]"):
            values = ast.literal_eval(value)
            if not isinstance(values, (list, tuple)):
                raise ValueError("answer list is malformed")
        else:
            values = [value]
    return [str(value).strip() for value in values if str(value).strip()]


def quality_score(prediction, references, metric, normalize=None):
    """Official leave-one-annotator-out VQA accuracy or DocVQA ANLS, in [0,1]."""
    if not references:
        raise ValueError("quality evaluation requires ground-truth references")
    if metric == "anls":
        from benchmark_dual_kv_v2 import anls_score
        return float(max(anls_score(prediction, answer) for answer in references))
    if metric != "vqa_score":
        raise ValueError(f"unsupported quality metric: {metric}")
    if normalize is None:
        # Deliberately no binary-accuracy fallback when this dependency is absent.
        from vlmeval.dataset.utils.vqa_eval import process_answer
        normalize = process_answer
    predicted = normalize(prediction)
    answers = [normalize(answer) for answer in references]
    total = sum(predicted == answer for answer in answers)
    return sum(min(1.0, (total - (predicted == answer)) / 3.0)
               for answer in answers) / len(answers)


def load_benchmark_manifest(args):
    if args.dataset_kind == "vqa":
        return load_manifest(args.manifest)
    from learned_score.mlsys_eval.common import load_manifest as load_sealed, canonical_hash
    from learned_score.mlsys_eval.olmocr import TRANSCRIPTION_PROMPT
    payload = load_sealed(args.manifest)
    if payload.get("format") != "asymvlm_olmocr_pages_v1":
        raise ValueError("olmOCR requires its separately audited page manifest")
    metadata = payload["metadata"]
    if (metadata.get("transcription_prompt") != TRANSCRIPTION_PROMPT
            or metadata.get("prompt_sha256") != canonical_hash(TRANSCRIPTION_PROMPT)):
        raise ValueError("olmOCR transcription prompt differs from the fixed benchmark protocol")
    if not metadata.get("source_groups_verified"):
        raise ValueError("olmOCR page source groups have not been verified")
    owners = {}
    for split, rows in payload["splits"].items():
        seen = set()
        for row in rows:
            if row["row_hash"] in seen:
                raise ValueError("duplicate olmOCR page identity")
            seen.add(row["row_hash"])
            for key in ("row_hash", "group_id", "image_sha256"):
                identity = (key, row[key])
                if not row[key] or identity in owners and owners[identity] != split:
                    raise ValueError("olmOCR source/page overlaps development and final splits")
                owners[identity] = split
    return SimpleNamespace(path=str(args.manifest), file_sha256=digest_file(args.manifest),
                           payload=payload, metadata=metadata, splits=payload["splits"])


def load_benchmark_backbone(args, manifest):
    from learned_score.mlsys_scorer import backbone
    if args.dataset_kind == "vqa":
        return backbone.load_backbone(args, manifest)
    from learned_score.mlsys_eval.olmocr import OlmOCRDataset
    dataset = OlmOCRDataset(manifest.path, allowed_splits=(args.split,))
    # Temporary in-process dataset injection avoids editing training's frozen,
    # hashed loader. Model and preprocessing construction remain identical.
    original = backbone.make_dataset
    backbone.make_dataset = lambda manifest, allowed_splits=None: dataset
    try:
        return backbone.load_backbone(args, manifest)
    finally:
        backbone.make_dataset = original


def make_selector(cell, scorer=None, scorer_variant=None, observed=None):
    import torch
    from torch.nn import functional as F
    from learned_score.mlsys_scorer.models import require_finite, select_tokens

    @torch.no_grad()
    def select(prepared):
        vision = prepared.inputs_embeds[0, prepared.visual_positions]
        question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        count = len(vision)
        if not count:
            raise ValueError("benchmark image has no prunable visual tokens")
        if observed is not None:
            observed.update({"input_visual_tokens": count,
                             "input_text_tokens": int(prepared.inputs_embeds.shape[1] - count),
                             "input_prefix_tokens": int(prepared.inputs_embeds.shape[1])})
        keep = max(1, int(count * cell.visual_retention))
        if keep == count or cell.selector == "full":
            selected = torch.arange(count, dtype=torch.long, device=vision.device)
        elif cell.selector in {"scorer", "scorer_coverage"}:
            if scorer is None:
                raise ValueError("learned selector requires a verified checkpoint")
            variant = "rank32_coverage" if cell.selector == "scorer_coverage" else scorer_variant
            if variant == "rank32_coverage" and scorer_variant == "original":
                raise ValueError("coverage variant requires rank32, not original scorer")
            selected = select_tokens(scorer(vision, question), vision, cell.visual_retention, variant)
        else:
            from learned_score.llava_selectors import cdpruner_order, divprune_order
            require_finite(vision, "baseline visual embeddings")
            if cell.selector == "divprune":
                selected = divprune_order(vision, keep)
            elif cell.selector == "cdpruner":
                require_finite(question, "baseline question embeddings")
                relevance = (F.normalize(vision.float(), dim=-1)
                             @ F.normalize(question.float(), dim=-1).T).max(dim=-1).values
                selected = cdpruner_order(vision, relevance, keep)
            else:
                raise ValueError(f"unsupported selector: {cell.selector}")
        selected = selected.sort().values
        # Do not silently repair degenerate external selectors or underbudget rows.
        if selected.ndim != 1 or selected.dtype != torch.long or len(selected) != keep:
            raise RuntimeError("selector violated the exact retained-token count")
        if len(selected.unique()) != keep or int(selected.min()) < 0 or int(selected.max()) >= count:
            raise RuntimeError("selector returned duplicate or invalid patch indices")
        return selected
    return select


def measured_request(adapter, row, select_visual, cell, *, max_new_tokens,
                     slack=32, fixed_output_tokens=False, run=None, timer_factory=None):
    """One live cache at a time; return only plain metrics, CPU IDs and text."""
    import torch
    from learned_score.mlsys_eval.timing import TimingContract
    from learned_score.mlsys_runtime import run_request
    run = run_request if run is None else run
    timer_factory = TimingContract if timer_factory is None else timer_factory
    synchronize = (lambda: torch.cuda.synchronize(adapter.device)) if torch.device(adapter.device).type == "cuda" else None
    # Parquet-backed rows may expose one reusable BytesIO object.
    if hasattr(row.get("image"), "seek"):
        row["image"].seek(0)
    timer = timer_factory(synchronize=synchronize)
    result = None
    try:
        result = run(adapter, row, select_visual,
                     max_new_tokens=max_new_tokens,
                     generated_budget=max_new_tokens if cell.generated_budget is None else cell.generated_budget,
                     slack=slack, stop_on_eos=not fixed_output_tokens,
                     on_first_token=timer.mark_first_token,
                     require_fused_attention=True)
        runtime = copy.deepcopy(result.metrics)
        # These are observed host intervals. Device compaction time is recorded
        # independently in runtime metrics and is never added to total latency.
        timer.stages["preprocess"] = runtime.get("prepare_seconds", 0.0)
        timer.stages["selector"] = runtime.get("selection_and_visual_compaction_seconds", 0.0)
        timer.stages["prefill"] = runtime.get("prefill_ttft_seconds", 0.0)
        compression = runtime.get("compaction_cpu_seconds", 0.0)
        timer.stages["compression"] = compression
        timer.stages["decode"] = max(0.0, runtime.get("decode_seconds", 0.0) - compression)
        with timer.stage("postprocess"):
            ids = result.ids.detach().cpu()
            prediction = adapter.processor.decode(ids[0], skip_special_tokens=True).strip()
        timing = timer.finish()
        cache = runtime["cache"]
        runtime["peak_cache_bytes"] = cache["total_owned_bytes"]
        runtime["physical_cache_measurement"] = "allocated_KV_capacity_plus_scratch_and_source_metadata"
        runtime["logical_active_cache_bytes"] = cache["kv_active_bytes"]
        runtime["eviction_active"] = bool(cache["compaction_events"])
        runtime["cache_baseline_scope"] = "shared_preallocated_storage_not_stock_DynamicCache"
        runtime["text_eviction_disabled"] = cell.generated_budget is None
        runtime["stage_granularity"] = "preprocess_includes_vision;selector_includes_visual_gather"
        return prediction, ids, timing, runtime
    finally:
        # GenerationResult owns the whole GPU cache. Never accumulate these
        # objects in output lists or retain them into the next benchmark mode.
        if result is not None:
            del result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-kind", choices=("vqa", "olmocr"), default="vqa")
    parser.add_argument("--scorer-training-manifest", type=Path,
                        help="original training manifest for checkpoints evaluated on separate olmOCR pages")
    parser.add_argument("--split", choices=("scorer_dev", "scorer_screening", "scorer_selection", "development"), default="scorer_dev")
    parser.add_argument("--adapter", choices=("qwen2_5_vl", "llava_next"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-revision")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--selectors", nargs="+", choices=SELECTORS, default=["full", "divprune", "cdpruner"])
    parser.add_argument("--retentions", nargs="+", type=float, default=list(RETENTIONS))
    parser.add_argument("--text-budgets", nargs="+", choices=("off", "256", "512"), default=["off", "256", "512"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backbone-dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-visual-patches", type=int, default=1280)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--slack", type=int, default=32)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fixed-output-tokens", action="store_true", help="timing diagnostic; no quality score reported")
    args = parser.parse_args(argv)
    if args.max_new_tokens < 1 or args.slack < 1 or args.repetitions < 1 or args.warmup < 0:
        parser.error("invalid generation/repetition/warmup limit")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if not 0 <= args.rank < args.world_size:
        parser.error("require 0 <= rank < world-size")
    if any(value.startswith("scorer") for value in args.selectors) and args.checkpoint is None:
        parser.error("learned selectors require --checkpoint")
    if args.dataset_kind == "olmocr" and args.checkpoint and not args.scorer_training_manifest:
        parser.error("olmOCR learned selectors require --scorer-training-manifest")
    args.text_budgets = [None if value == "off" else int(value) for value in args.text_budgets]
    make_grid(args.selectors, args.retentions, args.text_budgets)
    args.attention_implementation = "sdpa"
    args.dataset_splits = (args.split,)
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    from learned_score.mlsys_eval.common import dataset_family
    from learned_score.mlsys_scorer.backbone import assert_frozen, frozen_versions, read_entry
    from learned_score.mlsys_scorer.checkpoints import load_scorer
    manifest = load_benchmark_manifest(args)
    training_sha = (load_manifest(args.scorer_training_manifest).file_sha256
                    if args.scorer_training_manifest else manifest.file_sha256)
    entries = entries_for_split(manifest, args.split)
    if args.limit:
        entries = entries[:args.limit]
    entries = entries[args.rank::args.world_size]
    if not entries:
        raise ValueError("no development rows assigned to this worker")
    # Fail before loading the model rather than silently switch TextVQA metrics.
    normalizer = None
    if not args.fixed_output_tokens and any("textvqa" in entry["dataset"].lower() for entry in entries):
        from vlmeval.dataset.utils.vqa_eval import process_answer
        normalizer = process_answer
    adapter, dataset, backbone = load_benchmark_backbone(args, manifest)
    versions = frozen_versions(adapter.model)
    scorer = scorer_variant = checkpoint_hash = checkpoint_compatibility = None
    if args.checkpoint:
        scorer, checkpoint = load_scorer(args.checkpoint, args.device)
        require_completed_checkpoint(args.checkpoint, checkpoint)
        checkpoint_compatibility = validate_checkpoint_identity(checkpoint, training_sha, backbone)
        scorer_variant, checkpoint_hash = checkpoint["variant"], digest_file(args.checkpoint)
    hardware = torch.cuda.get_device_name(adapter.device) if torch.device(adapter.device).type == "cuda" else platform.processor() or "CPU"
    cells = make_grid(args.selectors, args.retentions, args.text_budgets)
    identity = {"format": "mlsys_batch_one_benchmark_v1", "manifest_sha256": manifest.file_sha256,
                "dataset_kind": args.dataset_kind, "scorer_training_manifest_sha256": training_sha if args.checkpoint else None,
                "backbone": backbone, "checkpoint_sha256": checkpoint_hash,
                "checkpoint_compatibility": checkpoint_compatibility,
                "benchmark_source_sha256": digest_file(Path(__file__)),
                "runtime_source_sha256": {name: digest_file(Path(__file__).parents[1] / "mlsys_runtime" / name)
                                          for name in ("generation.py", "cache.py", "masking.py", "policies.py")},
                "hardware": hardware, "split": args.split, "batch_size": 1,
                "max_new_tokens": args.max_new_tokens, "slack": args.slack,
                "fixed_output_tokens": args.fixed_output_tokens, "repetitions": args.repetitions,
                "warmup": args.warmup, "cells": [cell.key for cell in cells], "limit": args.limit}
    args.output.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output / f"rank{args.rank}.metadata.json"
    if metadata_path.exists():
        if not args.resume:
            raise FileExistsError("output exists; use --resume with unchanged configuration")
        if json.loads(metadata_path.read_text()) != identity:
            raise ValueError("resume benchmark identity differs")
    else:
        atomic_json(metadata_path, identity)
    # Warm up all methods outside measurements, with a short fixed cap.
    warmup_row = read_entry(dataset, entries[0])
    for _ in range(args.warmup):
        for cell in cells:
            measured_request(adapter, warmup_row, make_selector(cell, scorer, scorer_variant), cell,
                             max_new_tokens=min(args.max_new_tokens, 8), slack=args.slack)
    del warmup_row
    completed = 0
    for entry in entries:
        row = read_entry(dataset, entry)
        references = parse_references(row["answer"]) if args.dataset_kind == "vqa" else None
        metric = (manifest.metadata["official_metric"] if args.dataset_kind == "olmocr" else
                  "anls" if "docvqa" in entry["dataset"].lower() else "vqa_score")
        for repetition in range(args.repetitions):
            # Rotate methods by request/repetition to avoid a fixed warm-order advantage.
            offset = (int(entry["row_hash"][:8], 16) + repetition) % len(cells)
            order = cells[offset:] + cells[:offset]
            for cell in order:
                path = args.output / "rows" / f"{entry['row_hash']}.{cell.key}.repeat{repetition}.json"
                attempt = 0
                if path.exists():
                    previous = json.loads(path.read_text())
                    if previous.get("identity_sha256") != canonical_digest(identity):
                        raise ValueError("row provenance differs from this run")
                    if previous.get("status") == "ok":
                        completed += 1
                        continue
                    if not args.resume or previous.get("attempt", 0) >= 1:
                        raise RuntimeError("failed row already exhausted its one restart retry")
                    attempt = previous.get("attempt", 0) + 1
                observed = {}
                record = {"format": "mlsys_request_result_v1", "identity_sha256": canonical_digest(identity),
                          "model": backbone["model"], "dataset": dataset_family(entry["dataset"]),
                          "split": args.split, "stage": "timing_diagnostic" if args.fixed_output_tokens else "development",
                          "request_id": entry["row_hash"], "group_id": entry.get("group_id", entry["image_sha256"]),
                          "image_sha256": entry["image_sha256"], "method": cell.method, "selector": cell.selector,
                          "visual_retention": cell.visual_retention, "generated_budget": cell.generated_budget,
                          "quality_metric": metric, "max_new_tokens": args.max_new_tokens,
                          "hardware": hardware, "batch_size": 1, "serving_measurement": False,
                          "repetition": repetition, "repeat": repetition + 1, "attempt": attempt,
                          "timing_boundary": "image_preprocessing_to_final_token",
                          "includes_output_detokenization": True,
                          "baseline_implementation": "adapted_projected_feature_selector" if cell.selector in {"divprune", "cdpruner"} else None}
                try:
                    prediction, ids, timing, runtime = measured_request(
                        adapter, row, make_selector(cell, scorer, scorer_variant, observed), cell,
                        max_new_tokens=args.max_new_tokens, slack=args.slack,
                        fixed_output_tokens=args.fixed_output_tokens)
                    assert_frozen(adapter.model, versions)
                    score = (None if args.fixed_output_tokens or args.dataset_kind == "olmocr" else
                             quality_score(prediction, references, metric, normalizer))
                    quality_status = ("timing_diagnostic_not_scored" if args.fixed_output_tokens else
                                      "awaiting_official_score" if args.dataset_kind == "olmocr" else "scored")
                    record.update({"status": "ok", "prediction": prediction, "quality_score": score,
                                   "quality_status": quality_status,
                                   "generated_tokens": int(ids.shape[-1]), "natural_stop_enabled": not args.fixed_output_tokens,
                                   "peak_allocated_bytes": runtime.get("full_request_peak_allocated_bytes"),
                                   "metrics": runtime, **timing, **observed})
                    del ids
                except Exception as error:
                    record.update({"status": "failed", "quality_score": None,
                                   "error_type": type(error).__name__, "error": str(error), **observed})
                    atomic_json(path, record)
                    raise
                atomic_json(path, record)
                completed += 1
                print(json.dumps({"event": "measurement", "request_id": entry["row_hash"],
                                  "cell": cell.key, "repetition": repetition,
                                  "seconds": record["end_to_end_seconds"], "generated_tokens": record["generated_tokens"]}), flush=True)
    atomic_json(args.output / f"rank{args.rank}.complete.json",
                {"identity_sha256": canonical_digest(identity), "request_results": completed,
                 "requests": len(entries), "cells": len(cells), "repetitions": args.repetitions})


if __name__ == "__main__":
    main()
