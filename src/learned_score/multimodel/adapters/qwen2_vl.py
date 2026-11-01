"""Qwen2-VL adapter preserving visual merger spans and multimodal RoPE."""

from typing import Sequence

import torch
from PIL import Image

from learned_score.eval_llava_adapter import split_message
from learned_score.llava_paper_mask import build_additive_causal_mask
from learned_score.qwen2_pruning import (
    build_pruned_inputs,
    extract_visual_features,
    greedy_generate,
)
from benchmark_dual_kv_v2 import parse_answer_field

from .base import PreparedBatch, PrunedBatch, VLMAdapter


def last_subsequence_positions(sequence: Sequence[int], query: Sequence[int]):
    """Return positions for the last exact occurrence of query in sequence."""
    if not query:
        raise ValueError("semantic question token sequence is empty")
    start = None
    stop = len(sequence) - len(query) + 1
    for index in range(max(0, stop)):
        if list(sequence[index : index + len(query)]) == list(query):
            start = index
    if start is None:
        raise ValueError("question tokens were not found in Qwen chat prompt")
    return torch.arange(start, start + len(query), dtype=torch.long)


class Qwen2VLAdapter(VLMAdapter):
    name = "qwen2_vl"
    version = "1"

    def __init__(self, dataset=None, device="cuda"):
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.hidden_size = 0

    def load_model_and_processor(self, config):
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            config.model,
            revision=(
                getattr(config, "processor_revision", None)
                or getattr(config, "model_revision", None)
            ),
            trust_remote_code=True,
            use_fast=False,
        )
        max_visual_patches = int(
            getattr(config, "max_visual_patches", 1280)
        )
        self.processor.image_processor.max_pixels = (
            max_visual_patches * 28 * 28
        )
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            config.model,
            revision=getattr(config, "model_revision", None),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation=getattr(
                config, "attn_implementation", "eager"
            ),
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.hidden_size)
        return self

    def extract_visual_features(
        self, pixel_values, image_grid_thw, with_stats=False
    ):
        """Architecture hook used by Qwen2 and Qwen2.5 adapters."""
        return extract_visual_features(
            self.model,
            pixel_values,
            image_grid_thw,
            with_stats=with_stats,
        )

    def prepare_sample(self, dataset_row, include_answer=False):
        if self.dataset is None:
            raise ValueError("Qwen adapter requires its dataset prompt builder")
        image_path, question = split_message(
            self.dataset.build_prompt(dataset_row)
        )
        image = Image.open(image_path).convert("RGB")
        user_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
                ],
            }
        ]
        messages = list(user_messages)
        answer = None
        if include_answer:
            references = parse_answer_field(dataset_row.get("answer", ""))
            if not references:
                raise ValueError("answer-aware training sample has no answer")
            answer = references[0]
            messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": answer}],
                }
            )
        prefix_prompt = self.processor.apply_chat_template(
            user_messages, tokenize=False, add_generation_prompt=True
        )
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=not include_answer,
        )
        inputs = self.processor(
            text=[prompt], images=[image], return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        with torch.no_grad():
            image_features, _, _ = self.extract_visual_features(
                inputs["pixel_values"],
                inputs["image_grid_thw"],
                with_stats=False,
            )
            embeddings, attention_mask, position_ids, image_mask = (
                build_pruned_inputs(
                    self.model, inputs, image_features, selected=None
                )
            )
            embeddings = embeddings.detach()
        visual = image_mask[0].nonzero(as_tuple=False).squeeze(1)
        valid = attention_mask[0].bool()
        prompt_text = ((~image_mask[0]) & valid).nonzero(
            as_tuple=False
        ).squeeze(1)
        question_ids = self.processor.tokenizer(
            question, add_special_tokens=False
        ).input_ids
        semantic = last_subsequence_positions(
            inputs["input_ids"][0].tolist(), question_ids
        ).to(self.device)
        if torch.isin(semantic, visual).any():
            raise RuntimeError("Qwen semantic question overlaps visual span")
        if visual.numel() != image_features.shape[0]:
            raise RuntimeError(
                "Qwen visual placeholder and merger counts do not match"
            )
        answer_positions = torch.empty(
            0, dtype=torch.long, device=self.device
        )
        scorer_prompt_text = prompt_text
        if answer is not None:
            prefix_inputs = self.processor(
                text=[prefix_prompt], images=[image], return_tensors="pt"
            )
            prefix_ids = prefix_inputs["input_ids"][0]
            full_ids = inputs["input_ids"][0].detach().cpu()
            prefix_length = int(prefix_ids.numel())
            if (
                full_ids.numel() <= prefix_length
                or not torch.equal(
                    full_ids[:prefix_length], prefix_ids
                )
            ):
                raise RuntimeError(
                    "Qwen assistant generation prefix is not an exact "
                    "prefix of the teacher-forced sequence"
                )
            special_ids = set(self.processor.tokenizer.all_special_ids)
            answer_stop = prefix_length
            while (
                answer_stop < full_ids.numel()
                and int(full_ids[answer_stop]) not in special_ids
            ):
                answer_stop += 1
            answer_positions = torch.arange(
                prefix_length,
                answer_stop,
                dtype=torch.long,
                device=self.device,
            )
            if not answer_positions.numel():
                raise RuntimeError("Qwen answer span is empty")
            if torch.isin(answer_positions, visual).any():
                raise RuntimeError("Qwen answer overlaps visual span")
            if answer_positions.min() <= semantic.max():
                raise RuntimeError(
                    "Qwen answer positions do not follow the question"
                )
            # Gold answer tokens are teacher targets, never scorer inputs.
            scorer_prompt_text = prompt_text[
                prompt_text < answer_positions.min()
            ]
        return PreparedBatch(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            visual_positions=visual,
            semantic_text_positions=semantic,
            prompt_text_positions=scorer_prompt_text,
            answer_positions=answer_positions,
            native_inputs=inputs,
            metadata={
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "image_path": image_path,
                "position_ids": position_ids,
                "image_grid_thw": inputs["image_grid_thw"],
                "max_visual_patches": (
                    self.processor.image_processor.max_pixels // (28 * 28)
                ),
            },
        )

    def teacher_forward(self, prepared):
        if prepared.answer_positions.numel():
            with torch.no_grad():
                output = self.model.model(
                    inputs_embeds=prepared.inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    position_ids=prepared.metadata["position_ids"],
                    use_cache=False,
                    return_dict=True,
                )
                hidden = output.last_hidden_state
                logits = self.model.lm_head(hidden)
            return {
                "hidden": hidden.detach(),
                "logits": logits.detach(),
            }
        with torch.no_grad():
            hidden = self.model.model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
                position_ids=prepared.metadata["position_ids"],
                use_cache=False,
                return_dict=True,
            ).last_hidden_state.detach()
        return {"hidden": hidden}

    def student_forward_with_key_penalty(self, prepared, key_penalty):
        additive = build_additive_causal_mask(
            prepared.attention_mask,
            key_penalty,
            prepared.inputs_embeds.dtype,
        )
        hidden = self.model.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=additive,
            position_ids=prepared.metadata["position_ids"],
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        result = {"hidden": hidden}
        if prepared.answer_positions.numel():
            result["logits"] = self.model.lm_head(hidden)
        return result

    def position_id_policy(
        self, prepared, retained_source_positions, policy
    ):
        if policy == "original":
            return prepared.metadata["position_ids"][
                :, :, retained_source_positions
            ]
        if policy == "compact":
            values = torch.arange(
                retained_source_positions.numel(),
                device=retained_source_positions.device,
            )
            return values[None, None, :].expand(3, 1, -1)
        raise ValueError(f"unknown position policy: {policy}")

    def physically_prune(
        self, prepared, selected_visual_indices, position_policy
    ):
        selected_visual_indices = selected_visual_indices.sort().values
        visual_source = prepared.visual_positions[selected_visual_indices]
        retained = torch.cat(
            (prepared.prompt_text_positions, visual_source)
        ).sort().values
        embeddings = prepared.inputs_embeds[:, retained]
        attention_mask = torch.ones(
            embeddings.shape[:2],
            device=embeddings.device,
            dtype=prepared.attention_mask.dtype,
        )
        return PrunedBatch(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=self.position_id_policy(
                prepared, retained, position_policy
            ),
            retained_source_positions=retained,
            metadata={
                "position_policy": position_policy,
                "visual_tokens_original": int(
                    prepared.visual_positions.numel()
                ),
                "visual_tokens": int(selected_visual_indices.numel()),
                "image_grid_thw": prepared.metadata[
                    "image_grid_thw"
                ].tolist(),
            },
        )

    def generate_from_pruned(self, pruned, generation_config):
        return greedy_generate(
            self.model,
            pruned.inputs_embeds,
            pruned.attention_mask,
            pruned.position_ids,
            int(generation_config["max_new_tokens"]),
        )

    def forward_pruned_with_answer(self, pruned):
        with torch.no_grad():
            output = self.model.model(
                inputs_embeds=pruned.inputs_embeds,
                attention_mask=pruned.attention_mask,
                position_ids=pruned.position_ids,
                use_cache=False,
                return_dict=True,
            )
            hidden = output.last_hidden_state
            logits = self.model.lm_head(hidden)
        return {"hidden": hidden.detach(), "logits": logits.detach()}
