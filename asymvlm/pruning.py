"""AsymVLM uniform visual-token pruning on top of the frozen ``learned_score`` runtime.

Pipeline for one request:
    prepared = prepare(runtime, image, question)        # adapter-native inputs
    idx      = select_tokens(prepared, 0.4, runtime.scorer)  # top-k scorer, original order
    result   = prune_and_generate(runtime, prepared, 0.4)    # physical pruning + bounded KV decode

Heavy dependencies (torch, transformers) are imported lazily.
"""
import math
from dataclasses import dataclass
from types import SimpleNamespace

from .config import GENERATION_DEFAULTS, get_spec


class PromptDataset:
    """Minimal prompt builder accepted by the frozen adapters.

    Rows are dicts with ``image`` (path or file object) and ``question`` (str).
    Benchmark-specific prompt suffixes must already be part of ``question``.
    """

    def build_prompt(self, row):
        return [dict(type="image", value=row["image"]), dict(type="text", value=row["question"])]

    def row_for_entry(self, row):
        return row


@dataclass
class Runtime:
    spec: object
    adapter: object
    scorer: object
    device: str

    @property
    def model(self):
        return self.adapter.model

    def decode(self, ids):
        ids = ids[0] if getattr(ids, "ndim", 1) == 2 else ids
        return self.adapter.processor.decode(ids, skip_special_tokens=True).strip()


def _snapshot(spec):
    from huggingface_hub import snapshot_download
    return snapshot_download(spec.hf_id, revision=spec.revision)


def load_scorer(model, device="cpu"):
    """Load the trained rank-32 scorer for ``model`` (no VLM needed)."""
    spec = get_spec(model)
    if spec.scorer_format == "query_selector_v24":
        # Query-aware retrained scorer (the one used for all Qwen2.5-VL results).
        from learned_score.mlsys_sprint.query_selector_v24 import load
        scorer, _ = load(str(spec.scorer_path()), device)
    else:
        from learned_score.mlsys_scorer.checkpoints import load_scorer as _load
        scorer, _ = _load(str(spec.scorer_path()), device)
    return scorer.eval().requires_grad_(False)


def load_model_and_scorer(model, device="cuda:0", dtype="bfloat16"):
    """Load a pinned VLM backbone, its frozen adapter, and the trained scorer.

    Mirrors ``learned_score.mlsys_sprint.budget_execute_v31.load_runtime``: the
    backbone is loaded exactly as during evaluation (Qwen/LLaVA with SDPA, Phi
    with eager attention and 16 crops) and the per-backbone scorer is attached.
    """
    import torch
    spec = get_spec(model)
    dataset = PromptDataset()
    path = _snapshot(spec)
    if spec.adapter == "phi3_vision":
        from learned_score.mlsys_sprint import phi_adaptive_v3, phi_adaptive_v4
        args = SimpleNamespace(model=path, model_revision=spec.revision, num_crops=spec.num_crops,
                               backbone_dtype=dtype, device=device)
        with phi_adaptive_v4.installed():
            adapter = phi_adaptive_v3.PhiAdaptiveAdapterV3(dataset, device).load_model_and_processor(args)
            phi_adaptive_v4.validate_stopping(adapter)
    else:
        from learned_score.multimodel.adapters import LlavaNextAdapter, Qwen2_5_VLAdapter
        cls = {"qwen2_5_vl": Qwen2_5_VLAdapter, "llava_next": LlavaNextAdapter}[spec.adapter]
        config = SimpleNamespace(model=path, model_revision=spec.revision, processor_revision=spec.revision,
                                 max_visual_patches=spec.max_visual_patches or 1280,
                                 llava_next_grid_pinpoints=None,
                                 attn_implementation=spec.attention_implementation)
        adapter = cls(dataset=dataset, device=device).load_model_and_processor(config)
        adapter.model.to(dtype=getattr(torch, dtype))
        adapter.model.eval().requires_grad_(False)
    scorer = load_scorer(spec, device)
    if scorer.hidden_size != adapter.hidden_size:
        raise ValueError("scorer hidden dimension differs from the VLM")
    return Runtime(spec=spec, adapter=adapter, scorer=scorer, device=device)


def prepare(runtime, image, question):
    """Run the adapter's native preprocessing (no answer tokens)."""
    import torch
    with torch.no_grad():
        prepared = runtime.adapter.prepare_sample(dict(image=image, question=question), include_answer=False)
    if prepared.answer_positions.numel():
        raise ValueError("answers must not enter patch selection")
    return prepared


def keep_count(num_visual, keep_ratio):
    """k = ceil(N * r), clipped to [1, N]."""
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    return max(1, min(num_visual, math.ceil(num_visual * keep_ratio - 1e-9)))


def scorer_scores(prepared, scorer):
    """Per-visual-token scores: max over question tokens of cosine similarity in the learned 32-d space."""
    import torch
    with torch.no_grad():
        visual = prepared.inputs_embeds[0, prepared.visual_positions]
        query = prepared.inputs_embeds[0, prepared.semantic_text_positions]
        return scorer(visual, query).float()


def select_tokens(prepared, keep_ratio, scorer):
    """Top-k scorer tokens, returned in original (ascending) order.

    Identical to ``selector('asymvlm_uniform', k, scorer, {})`` in the frozen
    manuscript runtime (stable argsort, ties broken by position).
    """
    import torch
    n = len(prepared.visual_positions)
    k = keep_count(n, keep_ratio)
    if k == n:
        return torch.arange(n, device=prepared.inputs_embeds.device, dtype=torch.long)
    from learned_score.mlsys_sprint.manuscript_execute_v35 import selector
    return selector("asymvlm_uniform", k, scorer, {})(prepared)


def prune(runtime, prepared, indices):
    """Physically remove unselected visual tokens (original position ids retained)."""
    return runtime.adapter.physically_prune(prepared, indices, "original")


def generate(runtime, pruned, max_new_tokens=None, generated_budget=None, **kwargs):
    """Greedy bounded-KV generation (``learned_score.mlsys_runtime.generation.generate_bounded``)."""
    import torch
    from learned_score.mlsys_runtime.generation import generate_bounded
    kwargs.setdefault("require_fused_attention", runtime.spec.adapter != "phi3_vision")
    kwargs.setdefault("slack", GENERATION_DEFAULTS["slack"])
    with torch.no_grad():
        return generate_bounded(runtime.model, pruned,
                                max_new_tokens or GENERATION_DEFAULTS["max_new_tokens"],
                                generated_budget or GENERATION_DEFAULTS["generated_budget"], **kwargs)


def prune_and_generate(runtime, prepared, keep_ratio, **generation_kwargs):
    """AsymVLM uniform: keep ``ceil(N*keep_ratio)`` scorer-selected visual tokens and decode.

    Returns ``(text, result)`` where ``result`` is the ``GenerationResult``.
    """
    indices = select_tokens(prepared, keep_ratio, runtime.scorer)
    result = generate(runtime, prune(runtime, prepared, indices), **generation_kwargs)
    result.metrics.update(original_visual_tokens=int(len(prepared.visual_positions)),
                          retained_visual_tokens=int(indices.numel()))
    return runtime.decode(result.ids.detach().cpu()), result


def answer(runtime, image, question, keep_ratio=0.4, **generation_kwargs):
    """Convenience: prepare + prune + generate for one image/question."""
    return prune_and_generate(runtime, prepare(runtime, image, question), keep_ratio, **generation_kwargs)[0]
