"""Exact-count selector study on the unchanged, audited physical v4 runtime.

One process has one frozen scorer and one query-control setting. This makes
v4's physical reuse key (selector, K) safe; different scorers never alias.
"""

import argparse
import copy
from contextlib import contextmanager
import json
from pathlib import Path
import time

from .state import digest, file_digest
from . import adaptive_experiment_v4 as runtime


FORMAT = "mlsys_selector_experiment_v5"
CANDIDATES = {"current_rank32", "rank32_coverage", "saliency_control", "query_saliency",
              "answer_soft", "query_answer_soft", "answer_hard"}
_BASE_SOURCE_HASHES = runtime.source_hashes


def source_hashes(adapter):
    """Public immutable-runtime source contract, also checked by result audits."""
    result = _BASE_SOURCE_HASHES(adapter)
    for name in ("selector_experiment_v5.py", "selector_models_v5.py", "selector_training_v5.py",
                 "baseline_audit_v5.py", "cd_clip_reference.py", "cd_clip_native.py", "stable_cdpruner.py"):
        result["mlsys_sprint/" + name] = file_digest(Path(__file__).with_name(name))
    return result


def validate_indices(indices, count, size):
    import torch
    if (indices.dtype != torch.long or indices.ndim != 1 or len(indices) != count
            or len(indices.unique()) != count or int(indices.min()) < 0
            or int(indices.max()) >= size or bool((indices[1:] <= indices[:-1]).any())):
        raise ValueError("selector violated unique exact count / original positions")


def query_donors(entries):
    """Deterministic, different-image donor within the same dataset, no labels."""
    groups = {}
    for entry in sorted(entries, key=lambda e: e["row_hash"]):
        groups.setdefault(entry["dataset"], []).append(entry)
    result = {}
    for values in groups.values():
        for i, entry in enumerate(values):
            donor = next((values[(i + j) % len(values)] for j in range(1, len(values))
                          if values[(i + j) % len(values)]["image_sha256"] != entry["image_sha256"]), None)
            if donor is None:
                raise ValueError("query shuffle needs a different image in each dataset")
            result[entry["row_hash"]] = donor
    return result


@contextmanager
def installed(extra, args):
    """Scoped dependency injection; no frozen historical source is modified."""
    import torch
    from .selector_models_v5 import load_selector
    from .adaptive_benchmark_v1 import select_exact_count
    from learned_score.mlsys_scorer.backbone import frozen_versions
    if extra.candidate not in CANDIDATES or bool(extra.coverage) != (extra.candidate == "rank32_coverage"):
        raise ValueError("candidate and coverage configuration disagree")
    prior = runtime.prior
    saved = {name: getattr(runtime, name) for name in
             ("FORMAT", "source_hashes", "selector_for", "compare_features")}
    old_load, old_measured = prior.load_runtime, prior.measured
    state = {}
    configuration = {
        "candidate": extra.candidate, "warmstart_sha256": file_digest(extra.warmstart),
        "checkpoint_sha256": file_digest(args.checkpoint), "coverage_fraction": .2 if extra.coverage else 0.,
        "query_control": extra.query_control, "generation_query": "unchanged original",
        "diagnostics": ["selected_patch_indices", "reverse_query_selection_overlap"],
        "baseline_scope": "audited selector adaptations, not official end-to-end pipelines",
    }
    if extra.reference_clip:
        from .cd_clip_reference import configuration as clip_configuration
        configuration["reference_clip"] = clip_configuration()

    def sources(adapter):
        return source_hashes(adapter)

    def load(call_args, manifest):
        original_args = copy.copy(call_args)
        original_args.checkpoint = extra.warmstart
        adapter, dataset, backbone, _ = old_load(original_args, manifest)
        scorer, checkpoint = load_selector(call_args.checkpoint, adapter.device)
        if checkpoint.get("format") == "asymvlm_selector_training_v5":
            if checkpoint["identity"].get("recipe") != extra.candidate:
                raise ValueError("new checkpoint recipe differs from declared candidate")
            marker = json.loads((Path(call_args.checkpoint).parent / "complete.json").read_text())
            if (marker.get("checkpoint_sha256") != file_digest(call_args.checkpoint)
                    or marker.get("identity_sha256") != checkpoint["identity_sha256"]
                    or checkpoint.get("next_index") != 512 or marker.get("steps") != 512):
                raise ValueError("physical evaluation requires all 512 completed training steps")
            if checkpoint["identity"].get("warmstart_sha256") != file_digest(extra.warmstart):
                raise ValueError("new scorer has a different warm start")
        else:
            if extra.candidate not in {"current_rank32", "rank32_coverage"}:
                raise ValueError("refined candidate requires a completed v5 checkpoint")
            prior.benchmark.require_completed_checkpoint(call_args.checkpoint, checkpoint)
        if scorer.hidden_size != adapter.hidden_size:
            raise ValueError("new scorer hidden dimension differs from VLM")
        scorer.eval().requires_grad_(False)
        frozen_versions(scorer)
        state.update(adapter=adapter, dataset=dataset, scorer=scorer)
        if extra.reference_clip:
            from .cd_clip_reference import ReferenceClipRelevance
            state["clip"] = ReferenceClipRelevance(adapter, configuration["reference_clip"]["clip_snapshot"])
        if extra.query_control == "shuffled":
            state["donors"] = query_donors(manifest.splits[call_args.split])
        return adapter, dataset, {**backbone, "selector_v5": configuration}, scorer

    def compare_geometry(live, saved_feature, checkpoint_sha256):
        # Old features define geometry, not scores from the newly trained ranker.
        if saved_feature["checkpoint_sha256"] != file_digest(extra.warmstart):
            raise ValueError("geometry feature checkpoint is not the frozen warm start")
        if extra.candidate in {"current_rank32", "rank32_coverage"}:
            # Allocation controls reuse old entropy calibration and require the
            # complete live score-feature parity check, not geometry alone.
            return saved["compare_features"](live, saved_feature, checkpoint_sha256)
        if any(live[key] != saved_feature[key] for key in ("input_visual_tokens", "input_text_tokens")):
            raise ValueError("native visual/prompt geometry changed")

    def selector(cell, scorer, feature, observed=None):
        count = runtime.validate_cell_count(cell, feature)
        internal = prior.Cell(cell["selector"], "full" if cell["selector"] == "full" else "uniform", 0)
        donor_question = None
        if cell["selector"] == "scorer" and extra.query_control == "shuffled":
            donor = state["donors"][feature["request_id"]]
            # Preparing a donor is diagnostic setup, excluded from inference timing.
            with torch.no_grad():
                donor_prepared = state["adapter"].prepare_sample(state["dataset"].row_for_entry(donor), include_answer=False)
                donor_question = donor_prepared.inputs_embeds[0, donor_prepared.semantic_text_positions].detach().clone()
            del donor_prepared
            if observed is not None:
                observed["scorer_query_donor"] = donor["row_hash"]

        @torch.no_grad()
        def select(prepared):
            vision = prepared.inputs_embeds[0, prepared.visual_positions]
            question = prepared.inputs_embeds[0, prepared.semantic_text_positions]
            if (len(vision), prepared.inputs_embeds.shape[1] - len(vision)) != (
                    feature["input_visual_tokens"], feature["input_text_tokens"]):
                raise ValueError("runtime image/prompt geometry differs from frozen schedule")
            if cell["selector"] == "scorer":
                effective_question = question if donor_question is None else donor_question
                if vision.is_cuda:
                    torch.cuda.synchronize(vision.device)
                start = time.perf_counter()
                scores = scorer(vision, effective_question)
                indices = select_exact_count(scores, vision, count, coverage=extra.coverage)
                if vision.is_cuda:
                    torch.cuda.synchronize(vision.device)
                elapsed = time.perf_counter() - start
                def diagnostic():
                    with torch.no_grad():
                        reverse = select_exact_count(scorer(vision, effective_question.flip(0)), vision, count,
                                                     coverage=extra.coverage)
                    if observed is not None:
                        observed.update(selector_seconds=elapsed,
                            reversed_query_overlap=len(set(indices.tolist()) & set(reverse.tolist())) / count,
                            diagnostics_in_end_to_end_timing=False)
                state["diagnostic"] = diagnostic
            elif cell["selector"] == "cdpruner":
                if "clip" not in state:
                    raise ValueError("CDPruner requires independently encoded reference CLIP relevance")
                from .stable_cdpruner import stable_cdpruner_order
                indices = stable_cdpruner_order(vision, state["clip"](prepared), count).sort().values
            else:
                indices = saved["selector_for"](cell, scorer, feature, observed)[0](prepared)
            validate_indices(indices, count, len(vision))
            if observed is not None:
                values = indices.detach().cpu().tolist()
                observed.update(selected_patch_indices=values, selected_patch_indices_sha256=digest(values),
                    candidate=extra.candidate, query_control=extra.query_control,
                    input_visual_tokens=len(vision), input_text_tokens=prepared.inputs_embeds.shape[1] - len(vision),
                    retained_visual_tokens=count, visual_retention=count / len(vision))
            return indices
        return select, internal

    def measured(adapter, row, selector, cell, max_new_tokens):
        if cell.selector != "cdpruner":
            state.pop("diagnostic", None)
            result = old_measured(adapter, row, selector, cell, max_new_tokens)
            diagnostic = state.pop("diagnostic", None)
            if diagnostic is not None:
                diagnostic()
            return result
        clip = state["clip"]
        clip.release()
        handle = adapter.model.vision_tower.register_forward_hook(clip.capture)
        try:
            return old_measured(adapter, row, selector, cell, max_new_tokens)
        finally:
            handle.remove()
            clip.release()

    runtime.FORMAT, runtime.source_hashes = FORMAT, sources
    runtime.selector_for, runtime.compare_features = selector, compare_geometry
    prior.load_runtime, prior.measured = load, measured
    try:
        yield
    finally:
        for name, value in saved.items():
            setattr(runtime, name, value)
        prior.load_runtime, prior.measured = old_load, old_measured


def main(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--warmstart", required=True, type=Path)
    parser.add_argument("--coverage", action="store_true")
    parser.add_argument("--reference-clip", action="store_true")
    parser.add_argument("--query-control", choices=("correct", "shuffled"), default="correct")
    extra, rest = parser.parse_known_args(argv)
    args = runtime.parse_args(rest)
    if args.phase != "physical" or (extra.reference_clip and args.adapter != "llava_next"):
        raise ValueError("versioned selector runner is physical-only; reference CLIP is LLaVA-only")
    schedule = json.loads(args.schedule.read_text())
    if schedule.get("candidate") != extra.candidate:
        raise ValueError("candidate differs from immutable schedule")
    with installed(extra, args):
        runtime.main(rest)


if __name__ == "__main__":
    main()
