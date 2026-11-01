"""CLIP-relevance CDPruner adaptation with native LLaVA-NeXT packing.

This is NOT the source-faithful cropwise CDPruner pipeline. Relevance follows
the vendored reference, while selection is global after native unpadding and
keeps original positions/protected newlines. All new outputs declare this.
"""

import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import json
from pathlib import Path
import socket
import os

from .state import atomic_json, digest, file_digest
from .stable_cdpruner import stable_cdpruner_order


CLIP_SNAPSHOT = Path(os.environ.get("ASYMVLM_CLIP_SNAPSHOT", os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub", "models--openai--clip-vit-large-patch14-336/snapshots/ce19dc912ca5cd21c8a653c79e251e808ccabcd1")))
SUFFIX = "\nAnswer the question using a single word or phrase."


def clip_relevance(image, text):
    """Vendored negative-mean convention BEFORE min-max DPP normalization."""
    import torch
    from torch.nn import functional as F
    if image.ndim != 2 or text.ndim != 2 or image.shape[1] != text.shape[1] or not len(text):
        raise ValueError("CLIP relevance needs matching patch/segment embeddings")
    if not torch.isfinite(image).all() or not torch.isfinite(text).all():
        raise ValueError("nonfinite CLIP embeddings")
    return -(F.normalize(image.float(), dim=-1) @ F.normalize(text.float(), dim=-1).T).mean(-1)


def segment_inputs(inputs, length):
    """Match reference segmentation, including its zero padding/mask."""
    import torch
    count = inputs["input_ids"].shape[1]
    if count <= 0 or length <= 0 or any(x.shape != (1, count) for x in inputs.values()):
        raise ValueError("single nonempty CLIP question required")
    padding = ((count - 1) // length + 1) * length - count
    return {k: torch.cat((v, v.new_zeros((1, padding))), 1).reshape(-1, length)
            for k, v in inputs.items()}


def configuration(snapshot=CLIP_SNAPSHOT):
    snapshot = Path(snapshot)
    names = ("config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
    return {"format": "mlsys_cd_clip_native_v1", "label": "CDPruner, CLIP-relevance native-layout adaptation",
            "official_end_to_end_reproduction": False, "selection_layout": "global_after_native_anyres_unpadding",
            "relevance": "negative_mean_normalized_CLIP_patch_segment_cosine",
            "question_scope": "raw_question_without_generic_answer_suffix",
            "position_policy": "original", "protected_newlines": True,
            "kernel": "stable_FP64_conditional_DPP", "clip_snapshot": str(snapshot),
            "clip_sha256": {name: file_digest(snapshot / name) for name in names},
            "source_sha256": {"wrapper": file_digest(Path(__file__)),
                              "stable_dpp": file_digest(Path(__file__).with_name("stable_cdpruner.py"))}}


class ClipRelevance:
    def __init__(self, adapter, snapshot):
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast, CLIPVisionModelWithProjection
        if adapter.name != "llava_next" or adapter.model.config.vision_feature_select_strategy != "default":
            raise ValueError("CLIP relevance is validated only for native LLaVA-NeXT patch features")
        if not isinstance(adapter.model.config.vision_feature_layer, int):
            raise ValueError("multiple vision layers require a separate relevance mapping")
        # Load the reference projection on CPU, then discard its duplicate tower.
        vision = CLIPVisionModelWithProjection.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32)
        native = adapter.model.vision_tower.vision_model
        for name, parameter in native.state_dict().items():
            reference = vision.vision_model.state_dict().get(name)
            # The frozen native adapter loads FP16 before converting to BF16.
            if reference is None or not torch.equal(parameter.detach().cpu(), reference.to(torch.float16).to(parameter.dtype)):
                raise ValueError(f"native vision tower differs from reference CLIP: {name}")
        self.projection = vision.visual_projection.to(adapter.device).float().eval().requires_grad_(False)
        del vision
        self.text = CLIPTextModelWithProjection.from_pretrained(snapshot, local_files_only=True, torch_dtype=torch.float32)
        self.text = self.text.to(adapter.device).eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        self.adapter, self.captured = adapter, None

    def capture(self, module, args, output):
        if self.captured is not None:
            raise RuntimeError("unexpected second image encoder forward in one request")
        layer = self.adapter.model.config.vision_feature_layer
        self.captured = output.hidden_states[layer][:, 1:].detach()

    def release(self):
        self.captured = None

    def __call__(self, prepared):
        import torch
        if self.captured is None or prepared.native_inputs["image_sizes"].shape[0] != 1:
            raise ValueError("expected one captured native image")
        model = self.adapter.model
        features = model.vision_tower.vision_model.post_layernorm(self.captured)
        aligned = self.projection(features.float())
        # Apply exactly the same native packing to CLIP features and then map
        # the adapter's prunable positions. Newline entries are never selected.
        packed, _ = model.pack_image_features(
            [aligned], prepared.native_inputs["image_sizes"],
            vision_feature_select_strategy="default", image_newline=aligned.new_zeros(aligned.shape[-1]))
        all_visual = prepared.native_inputs["input_ids"][0].eq(model.config.image_token_index).nonzero().flatten()
        local = torch.searchsorted(all_visual, prepared.visual_positions)
        if len(packed) != len(all_visual) or not torch.equal(all_visual[local], prepared.visual_positions):
            raise ValueError("CLIP feature/native patch mapping failed")
        question = prepared.metadata["question"]
        if question.endswith(SUFFIX):
            question = question[:-len(SUFFIX)]
        inputs = self.tokenizer(text=question, return_tensors="pt", truncation=False)
        inputs = {k: v.to(self.adapter.device) for k, v in segment_inputs(inputs, self.text.config.max_position_embeddings).items()}
        with torch.no_grad():
            text = self.text(**inputs).text_embeds
        return clip_relevance(packed[local], text)


@dataclass(frozen=True)
class Cell:
    base: object
    config_hash: str
    @property
    def selector(self): return self.base.selector
    @property
    def visual_retention(self): return self.base.visual_retention
    @property
    def generated_budget(self): return self.base.generated_budget
    @property
    def key(self): return self.base.key if self.base.method == "all_tokens" else self.base.key + "_clipnative_" + self.config_hash[:12]
    @property
    def method(self): return "all_tokens" if self.base.method == "all_tokens" else self.key


@contextmanager
def wrapper(config):
    import torch
    from . import benchmark
    keys = ("load_benchmark_backbone", "make_selector", "make_grid", "measured_request", "atomic_json")
    old = {key: getattr(benchmark, key) for key in keys}
    state = {}
    binding = {"configuration": config, "configuration_sha256": digest(config)}
    def load(args, manifest):
        adapter, dataset, identity = old["load_benchmark_backbone"](args, manifest)
        state["clip"] = ClipRelevance(adapter, config["clip_snapshot"])
        return adapter, dataset, identity
    def grid(*args, **kwargs):
        return [Cell(cell, digest(config)) for cell in old["make_grid"](*args, **kwargs)]
    def selector(cell, scorer=None, scorer_variant=None, observed=None):
        if cell.selector != "cdpruner":
            return old["make_selector"](cell, scorer, scorer_variant, observed)
        @torch.no_grad()
        def select(prepared):
            vision = prepared.inputs_embeds[0, prepared.visual_positions]
            relevance = state["clip"](prepared)
            count = len(vision)
            keep = max(1, int(count * cell.visual_retention))
            selected = stable_cdpruner_order(vision, relevance, keep).sort().values
            if len(selected) != keep or len(selected.unique()) != keep:
                raise ValueError("native CLIP selector violated exact count")
            if observed is not None:
                observed.update(input_visual_tokens=count, input_text_tokens=prepared.inputs_embeds.shape[1] - count,
                                input_prefix_tokens=prepared.inputs_embeds.shape[1])
            return selected
        return select
    def measured(adapter, row, selector, cell, **kwargs):
        clip = state["clip"]
        if cell.selector != "cdpruner":
            return old["measured_request"](adapter, row, selector, cell, **kwargs)
        clip.release()
        handle = adapter.model.vision_tower.register_forward_hook(clip.capture)
        try:
            return old["measured_request"](adapter, row, selector, cell, **kwargs)
        finally:
            handle.remove()
            clip.release()
    def write(path, payload):
        if isinstance(payload, dict) and payload.get("format") == "mlsys_request_result_v1":
            payload = {**payload, "cd_clip_native_binding": binding}
            if payload["format"] == "mlsys_request_result_v1":
                payload["execution_context"] = {"hostname": socket.gethostname(), "pbs_job_id": os.getenv("PBS_JOBID"),
                    "cluster": "clustera" if socket.gethostname().startswith("x") else "clusterb"}
                if payload.get("selector") == "cdpruner":
                    payload["baseline_implementation"] = config["label"]
        return old["atomic_json"](path, payload)
    for name, value in zip(keys, (load, selector, grid, measured, write)):
        setattr(benchmark, name, value)
    try:
        yield benchmark
    finally:
        for key, value in old.items(): setattr(benchmark, key, value)


def main(argv=None):
    from . import benchmark
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--clip-snapshot", type=Path, default=CLIP_SNAPSHOT)
    extra, rest = parser.parse_known_args(argv)
    args = benchmark.parse_args(rest)
    if args.adapter != "llava_next" or args.dataset_kind != "vqa":
        raise ValueError("this isolated baseline is LLaVA VQA development only")
    config = configuration(extra.clip_snapshot)
    args.output.mkdir(parents=True, exist_ok=True)
    binding = {"configuration": config, "configuration_sha256": digest(config)}
    with (args.output / ".clip_native.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = args.output / "clip_native.metadata.json"
        if path.exists():
            if json.loads(path.read_text()) != binding:
                raise ValueError("CLIP baseline configuration changed")
        elif any(p.name != ".clip_native.lock" for p in args.output.iterdir()):
            raise ValueError("use a new empty output directory")
        else: atomic_json(path, binding)
    with wrapper(config) as module:
        module.main(rest)


if __name__ == "__main__": main()
