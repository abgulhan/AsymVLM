"""Phi-3.5-Vision adapter preserving the recovered trainer semantics."""

from typing import Any, Dict

import torch

from benchmark_cross_sim import greedy_generate_with_embeds
from learned_score.scorer import ScoreMaskCache
from learned_score.train_learned_score import (
    build_phi3_inputs,
    compute_image_text_boundary,
)

from .base import PreparedBatch, PrunedBatch, VLMAdapter


class Phi3VisionAdapter(VLMAdapter):
    name = "phi3_vision"
    version = "1"

    def __init__(self, dataset=None, device="cuda"):
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.hidden_size = 0

    def load_model_and_processor(self, config):
        from transformers import AutoProcessor
        from vlmeval.vlm.Phi3_5v.modeling_phi3_v import Phi3VForCausalLM

        self.processor = AutoProcessor.from_pretrained(
            config.model,
            trust_remote_code=True,
            num_crops=getattr(config, "num_crops", 16),
        )
        self.model = Phi3VForCausalLM.from_pretrained(
            config.model,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            _attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        return self

    def prepare_sample(self, dataset_row) -> PreparedBatch:
        if self.dataset is None:
            raise ValueError("Phi adapter requires its dataset prompt builder")
        inputs = build_phi3_inputs(
            self.processor, self.dataset.build_prompt(dataset_row)
        )
        inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        boundary = compute_image_text_boundary(inputs["input_ids"])
        with torch.no_grad():
            embeddings = self.model.model.vision_embed_tokens(
                inputs["input_ids"],
                pixel_values=inputs.get("pixel_values"),
                image_sizes=inputs.get("image_sizes"),
            ).detach()
        valid = inputs["attention_mask"][0].bool()
        visual = torch.arange(boundary, device=self.device)
        text = torch.arange(
            boundary, embeddings.shape[1], device=self.device
        )
        text = text[valid[text]]
        return PreparedBatch(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
            visual_positions=visual,
            semantic_text_positions=text,
            prompt_text_positions=text,
            native_inputs=inputs,
            metadata={"boundary": boundary},
        )

    def teacher_forward(self, prepared):
        with torch.no_grad():
            output = self.model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                use_cache=False,
                output_hidden_states=True,
            )
            hidden = output.hidden_states[-1].detach()
        return {"hidden": hidden}

    def student_forward_with_key_penalty(self, prepared, key_penalty):
        cache = ScoreMaskCache()
        cache.set_score_mask(
            key_penalty.to(prepared.inputs_embeds.dtype)[None, None, None, :]
        )
        output = self.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=prepared.attention_mask,
            past_key_values=cache,
            use_cache=True,
            output_hidden_states=True,
        )
        return {"hidden": output.hidden_states[-1]}

    def position_id_policy(
        self, prepared, retained_source_positions, policy
    ):
        if policy == "compact":
            values = torch.arange(
                len(retained_source_positions),
                device=retained_source_positions.device,
            )
        elif policy == "original":
            values = retained_source_positions
        else:
            raise ValueError(f"unknown position policy: {policy}")
        return values.unsqueeze(0)

    def physically_prune(
        self, prepared, selected_visual_indices, position_policy
    ):
        visual_source = prepared.visual_positions[selected_visual_indices]
        text_source = prepared.prompt_text_positions
        retained = torch.cat((visual_source, text_source)).sort().values
        embeddings = prepared.inputs_embeds[:, retained]
        mask = torch.ones(
            embeddings.shape[:2],
            device=embeddings.device,
            dtype=prepared.attention_mask.dtype,
        )
        return PrunedBatch(
            inputs_embeds=embeddings,
            attention_mask=mask,
            position_ids=self.position_id_policy(
                prepared, retained, position_policy
            ),
            retained_source_positions=retained,
            metadata={
                "position_policy": position_policy,
                "visual_tokens": int(len(visual_source)),
            },
        )

    def generate_from_pruned(self, pruned, generation_config):
        # The recovered Phi helper uses compact cache positions. Original
        # position IDs are evaluated only after a dedicated Phi parity gate.
        if pruned.metadata["position_policy"] != "compact":
            raise NotImplementedError(
                "Phi generation with original position IDs is not validated"
            )
        return greedy_generate_with_embeds(
            self.model,
            pruned.inputs_embeds,
            max_new_tokens=int(generation_config["max_new_tokens"]),
        )

