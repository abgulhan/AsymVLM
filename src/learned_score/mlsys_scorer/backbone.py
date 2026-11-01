"""Native-resolution integration with existing, frozen VLM adapters."""

from pathlib import Path
from types import SimpleNamespace

from .protocol import digest_file, native_resolution


MODELS = {
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "llava_next": "llava-hf/llava-v1.6-mistral-7b-hf",
}


def make_dataset(manifest, allowed_splits=None):
    # The MLSys audit loader supports newly selected official parquet rows;
    # the historical NativeMixtureDataset cannot validate all their hashes.
    from learned_score.mlsys_eval.datasets import EntryDataset
    candidates = allowed_splits or ("scorer_train", "scorer_dev", "scorer_screening", "scorer_selection")
    permitted = tuple(name for name in candidates if name in manifest.splits)
    return EntryDataset(manifest.path, allowed_splits=permitted)


def load_backbone(args, manifest):
    import torch
    from learned_score.multimodel.adapters import LlavaNextAdapter, Qwen2_5_VLAdapter
    classes = {"qwen2_5_vl": Qwen2_5_VLAdapter, "llava_next": LlavaNextAdapter}
    dataset = make_dataset(manifest, getattr(args, "dataset_splits", None))
    config = SimpleNamespace(model=args.model or MODELS[args.adapter],
                             model_revision=args.model_revision,
                             processor_revision=args.model_revision,
                             max_visual_patches=args.max_visual_patches,
                             llava_next_grid_pinpoints=None,
                             attn_implementation=getattr(args, "attention_implementation", "eager"))
    adapter = classes[args.adapter](dataset=dataset, device=args.device).load_model_and_processor(config)
    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.backbone_dtype]
    adapter.model.to(dtype=dtype)
    adapter.model.eval().requires_grad_(False)
    resolution = native_resolution(args.adapter, adapter.model.config, args.max_visual_patches)
    learned_root = Path(__file__).resolve().parents[1]
    source_paths = ("mlsys_scorer/backbone.py", "mlsys_scorer/models.py",
                    "mlsys_scorer/__main__.py", "multimodel/saliency.py",
                    "multimodel/objectives.py", "multimodel/adapters/llava_next.py",
                    "multimodel/adapters/qwen2_5_vl.py", "multimodel/adapters/qwen2_vl.py",
                    "multimodel/adapters/llava.py")
    identity = {"adapter": args.adapter, "model": MODELS[args.adapter],
                "model_path": config.model,
                "requested_revision": args.model_revision,
                "resolved_revision": getattr(adapter.model.config, "_commit_hash", None),
                "processor_revision": getattr(adapter.processor, "_commit_hash", None),
                "resolution": resolution, "backbone_dtype": args.backbone_dtype,
                "attention_implementation": config.attn_implementation, "position_policy": "original",
                "question_only": True,
                "source_sha256": {name: digest_file(learned_root / name) for name in source_paths}}
    return adapter, dataset, identity


def frozen_versions(model):
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("the backbone must be frozen")
    return tuple((id(parameter), parameter._version) for parameter in model.parameters())


def assert_frozen(model, versions):
    if frozen_versions(model) != versions or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("backbone parameters or gradient state mutated during scorer training")


def read_entry(dataset, entry):
    row = dataset.row_for_entry(entry)
    if row.get("_manifest_row_hash", entry["row_hash"]) != entry["row_hash"]:
        raise ValueError("dataset row does not match manifest identity")
    return row


def prepared_inputs(adapter, row, include_answer):
    prepared = adapter.prepare_sample(row, include_answer=include_answer)
    vision = prepared.inputs_embeds[0, prepared.visual_positions].detach()
    question = prepared.inputs_embeds[0, prepared.semantic_text_positions].detach()
    if include_answer:
        if not prepared.answer_positions.numel():
            raise ValueError("training sample has no answer-token targets")
        if (prepared.semantic_text_positions >= prepared.answer_positions.min()).any():
            raise ValueError("answer leakage into scorer question embeddings")
    return prepared, vision, question
