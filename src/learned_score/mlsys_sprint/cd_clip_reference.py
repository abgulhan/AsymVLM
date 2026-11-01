"""Independent reference-CLIP relevance; native LLaVA layout remains unchanged.

The tuned native vision tower is not assumed equal to pretrained CLIP. A
separate frozen FP32 CLIP encoder receives the exact native crop tensor. Its
additional image/text computation is inside end-to-end request timing.
"""

from contextlib import contextmanager
from pathlib import Path

from . import cd_clip_native as native
from .state import file_digest

_native_configuration = native.configuration


def configuration(snapshot=native.CLIP_SNAPSHOT):
    result = _native_configuration(snapshot)
    result.update({"format": "mlsys_cd_clip_reference_v1",
                   "label": "CDPruner, reference-CLIP native-layout adaptation",
                   "reference_vision": "independent_frozen_FP32_CLIPVisionModelWithProjection",
                   "reference_post_layernorm": True, "reference_visual_projection": True,
                   "vision_inputs": "exact_native_stacked_crop_pixel_values_before_encoder_cast",
                   "extra_reference_image_forward_in_timing": True,
                   "clip_model_residency": "independent_vision_and_text_resident_for_all_methods",
                   "native_vision_weights_assumed_equal": False,
                   "reference_attention_implementation": "sdpa"})
    result["source_sha256"]["reference_wrapper"] = file_digest(Path(__file__))
    return result


class ReferenceClipRelevance(native.ClipRelevance):
    def __init__(self, adapter, snapshot):
        import torch
        from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast, CLIPVisionModelWithProjection
        if (adapter.name != "llava_next" or adapter.model.config.vision_feature_select_strategy != "default"
                or type(adapter.model.config.vision_feature_layer) is not int):
            raise ValueError("reference CLIP requires audited single-layer native LLaVA-NeXT patch layout")
        self.vision = CLIPVisionModelWithProjection.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
        self.vision = self.vision.to(adapter.device).float().eval().requires_grad_(False)
        self.text = CLIPTextModelWithProjection.from_pretrained(
            snapshot, local_files_only=True, torch_dtype=torch.float32, attn_implementation="sdpa")
        self.text = self.text.to(adapter.device).float().eval().requires_grad_(False)
        self.tokenizer = CLIPTokenizerFast.from_pretrained(snapshot, local_files_only=True)
        for key in ("hidden_size", "image_size", "patch_size", "num_hidden_layers"):
            if getattr(self.vision.config, key) != getattr(adapter.model.config.vision_config, key):
                raise ValueError(f"reference CLIP/native crop geometry differs: {key}")
        self.projection = self.vision.visual_projection
        self.adapter, self.captured = adapter, None
        self._versions = self._parameter_versions()

    def _parameter_versions(self):
        return tuple((id(p), p._version, p.requires_grad) for model in (self.vision, self.text) for p in model.parameters())

    def capture(self, module, args, output):
        import torch
        if self.captured is not None:
            raise RuntimeError("unexpected second native image forward in one CDPruner request")
        if len(args) != 1 or args[0].ndim != 4:
            raise ValueError("native vision hook must expose the exact stacked crop tensor")
        layer = self.adapter.model.config.vision_feature_layer
        with torch.no_grad():
            reference = self.vision(pixel_values=args[0].detach().float(), output_hidden_states=True)
            features = reference.hidden_states[layer][:, 1:]
            if features.shape != output.hidden_states[layer][:, 1:].shape:
                raise ValueError("reference CLIP features do not align with native crop geometry")
            self.captured = self.projection(self.vision.vision_model.post_layernorm(features)).detach()

    def __call__(self, prepared):
        import torch
        if self.captured is None or prepared.native_inputs["image_sizes"].shape[0] != 1:
            raise ValueError("expected one independently encoded reference CLIP image")
        model = self.adapter.model
        aligned = self.captured
        packed, _ = model.pack_image_features(
            [aligned], prepared.native_inputs["image_sizes"], vision_feature_select_strategy="default",
            image_newline=aligned.new_zeros(aligned.shape[-1]))
        all_visual = prepared.native_inputs["input_ids"][0].eq(model.config.image_token_index).nonzero().flatten()
        local = torch.searchsorted(all_visual, prepared.visual_positions)
        if len(packed) != len(all_visual) or not torch.equal(all_visual[local], prepared.visual_positions):
            raise ValueError("reference CLIP/native patch mapping failed")
        question = prepared.metadata["question"]
        if question.endswith(native.SUFFIX):
            question = question[:-len(native.SUFFIX)]
        inputs = self.tokenizer(text=question, return_tensors="pt", truncation=False)
        inputs = {key: value.to(self.adapter.device) for key, value in
                  native.segment_inputs(inputs, self.text.config.max_position_embeddings).items()}
        with torch.no_grad():
            text = self.text(**inputs).text_embeds
        if self._parameter_versions() != self._versions:
            raise RuntimeError("frozen reference CLIP parameters mutated")
        return native.clip_relevance(packed[local], text)


@contextmanager
def reference_implementation():
    """Scope dependency injection; all frozen old source files remain unchanged."""
    old_class, old_config = native.ClipRelevance, native.configuration
    native.ClipRelevance, native.configuration = ReferenceClipRelevance, configuration
    try:
        yield
    finally:
        native.ClipRelevance, native.configuration = old_class, old_config


def main(argv=None):
    with reference_implementation():
        native.main(argv)


if __name__ == "__main__":
    main()
