"""LLaVA-NeXT adapter with variable AnyRes visual spans and protected newlines."""

import torch
from PIL import Image

from benchmark_dual_kv_v2 import parse_answer_field
from learned_score.eval_llava_adapter import split_message
from learned_score.llava_objectives import expanded_token_positions_for_char_span

from .base import PreparedBatch
from .llava import LlavaAdapter


def build_llava_next_embeds(model, inputs):
    """Reproduce the native AnyRes packing and identify layout newlines."""

    input_ids = inputs["input_ids"]
    token_embeds = model.get_input_embeddings()(input_ids)
    per_crop = model.get_image_features(
        pixel_values=inputs["pixel_values"],
        image_sizes=inputs["image_sizes"],
        vision_feature_layer=model.config.vision_feature_layer,
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
    )
    packed, feature_lens = model.pack_image_features(
        per_crop,
        inputs["image_sizes"],
        vision_feature_select_strategy=model.config.vision_feature_select_strategy,
        image_newline=model.image_newline,
    )
    image_mask = input_ids.eq(model.config.image_token_index)
    if image_mask.sum().item() != packed.shape[0]:
        raise RuntimeError(
            "expanded LLaVA-NeXT image-token count does not match packed "
            f"features: {image_mask.sum().item()} vs {packed.shape[0]}"
        )
    newline = model.image_newline.to(packed.device, packed.dtype)
    newline_local = torch.isclose(
        packed, newline.unsqueeze(0), rtol=0.0, atol=0.0
    ).all(dim=-1)
    expanded_mask = image_mask.unsqueeze(-1).expand_as(token_embeds)
    merged = token_embeds.masked_scatter(
        expanded_mask, packed.to(token_embeds.dtype)
    )
    return merged, image_mask, newline_local, feature_lens


def split_prunable_and_protected(image_mask, newline_local):
    """Map packed feature-local newline markers to source sequence positions."""

    all_visual = image_mask[0].nonzero(as_tuple=False).squeeze(1)
    newline_local = torch.as_tensor(
        newline_local, dtype=torch.bool, device=all_visual.device
    )
    if newline_local.numel() != all_visual.numel():
        raise ValueError("newline mask and visual source span differ in length")
    return all_visual[~newline_local], all_visual[newline_local]


class LlavaNextAdapter(LlavaAdapter):
    """AsymVLM mechanics for LLaVA-1.6/LLaVA-NeXT Mistral checkpoints."""

    name = "llava_next"
    version = "1"

    def load_model_and_processor(self, config):
        from transformers import AutoProcessor, LlavaNextForConditionalGeneration

        revision = getattr(config, "model_revision", None)
        self.processor = AutoProcessor.from_pretrained(
            config.model,
            revision=(getattr(config, "processor_revision", None) or revision),
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model,
            revision=revision,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=getattr(config, "attn_implementation", "eager"),
        ).to(self.device)
        grid = getattr(config, "llava_next_grid_pinpoints", None)
        if grid is not None:
            grid = [list(map(int, pair)) for pair in grid]
            self.processor.image_processor.image_grid_pinpoints = grid
            self.model.config.image_grid_pinpoints = grid
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        return self

    @staticmethod
    def prompt_for(question, answer=None):
        prefix = f"[INST] <image>\n{question} [/INST]"
        return prefix if answer is None else f"{prefix} {answer}"

    def prepare_sample(self, dataset_row, include_answer=False):
        if self.dataset is None:
            raise ValueError("LLaVA-NeXT adapter requires its dataset prompt builder")
        image_path, question = split_message(self.dataset.build_prompt(dataset_row))
        image = Image.open(image_path).convert("RGB")
        answer = None
        if include_answer:
            references = parse_answer_field(dataset_row.get("answer", ""))
            if not references:
                raise ValueError("answer-aware training sample has no answer")
            answer = references[0]
        prompt = self.prompt_for(question, answer)
        inputs = self.processor(images=image, text=prompt, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.no_grad():
            embeddings, image_mask, newline_local, feature_lens = (
                build_llava_next_embeds(self.model, inputs)
            )
            embeddings = embeddings.detach()
        visual, protected = split_prunable_and_protected(
            image_mask, newline_local
        )
        ordinary_text = (
            ~image_mask[0] & inputs["attention_mask"][0].bool()
        ).nonzero(as_tuple=False).squeeze(1)
        all_nonprunable = torch.cat((ordinary_text, protected)).sort().values
        question_start = prompt.index(question)
        semantic = expanded_token_positions_for_char_span(
            self.processor.tokenizer,
            prompt,
            question_start,
            question_start + len(question),
            inputs["input_ids"][0].detach().cpu(),
            self.model.config.image_token_index,
        ).to(self.device)
        answer_positions = torch.empty(0, dtype=torch.long, device=self.device)
        prompt_nonprunable = all_nonprunable
        if answer is not None:
            answer_start = prompt.rindex(answer)
            answer_positions = expanded_token_positions_for_char_span(
                self.processor.tokenizer,
                prompt,
                answer_start,
                answer_start + len(answer),
                inputs["input_ids"][0].detach().cpu(),
                self.model.config.image_token_index,
            ).to(self.device)
            prompt_nonprunable = all_nonprunable[
                all_nonprunable < answer_positions.min()
            ]
        if not visual.numel():
            raise RuntimeError("LLaVA-NeXT packed image has no prunable visual tokens")
        return PreparedBatch(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
            visual_positions=visual,
            semantic_text_positions=semantic,
            prompt_text_positions=prompt_nonprunable,
            answer_positions=answer_positions,
            native_inputs=inputs,
            metadata={
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "image_path": image_path,
                "vision_feature_layer": self.model.config.vision_feature_layer,
                "vision_feature_select_strategy": (
                    self.model.config.vision_feature_select_strategy
                ),
                "packed_feature_lens": feature_lens.detach().cpu().tolist(),
                "protected_visual_positions": protected,
                "protected_visual_tokens": int(protected.numel()),
                "packed_visual_tokens": int(image_mask.sum()),
                "image_grid_pinpoints": self.model.config.image_grid_pinpoints,
            },
        )
