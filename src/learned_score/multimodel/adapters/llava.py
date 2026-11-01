"""LLaVA-1.5 adapter with explicit patches, text spans and position IDs."""

from typing import Dict

import torch
from PIL import Image

from learned_score.eval_llava_adapter import (
    build_multimodal_embeds,
    split_message,
)
from learned_score.llava_objectives import (
    expanded_token_positions_for_char_span,
)
from learned_score.llava_paper_mask import build_additive_causal_mask
from benchmark_dual_kv_v2 import parse_answer_field

from .base import PreparedBatch, PrunedBatch, VLMAdapter


class LlavaAdapter(VLMAdapter):
    name = "llava_1_5"
    version = "1"

    def __init__(self, dataset=None, device="cuda"):
        self.dataset = dataset
        self.device = torch.device(device)
        self.model = None
        self.processor = None
        self.hidden_size = 0

    def load_model_and_processor(self, config):
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            config.model,
            revision=(
                getattr(config, "processor_revision", None)
                or getattr(config, "model_revision", None)
            ),
            trust_remote_code=True,
            use_fast=False,
        )
        self.model = LlavaForConditionalGeneration.from_pretrained(
            config.model,
            revision=getattr(config, "model_revision", None),
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            attn_implementation="eager",
        ).to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self.hidden_size = int(self.model.config.text_config.hidden_size)
        return self

    def prepare_sample(self, dataset_row, include_answer=False):
        if self.dataset is None:
            raise ValueError("LLaVA adapter requires its dataset prompt builder")
        image_path, question = split_message(
            self.dataset.build_prompt(dataset_row)
        )
        image = Image.open(image_path).convert("RGB")
        prefix = f"USER: <image>\n{question}\nASSISTANT:"
        answer = None
        if include_answer:
            references = parse_answer_field(dataset_row.get("answer", ""))
            if not references:
                raise ValueError("answer-aware training sample has no answer")
            answer = references[0]
            prompt = f"{prefix} {answer}"
        else:
            prompt = prefix
        inputs = self.processor(
            images=image, text=prompt, return_tensors="pt"
        )
        inputs = {
            key: value.to(self.device) for key, value in inputs.items()
        }
        with torch.no_grad():
            embeddings, image_mask = build_multimodal_embeds(
                self.model, inputs
            )
            embeddings = embeddings.detach()
        visual = image_mask[0].nonzero(as_tuple=False).squeeze(1)
        all_text = (
            ~image_mask[0] & inputs["attention_mask"][0].bool()
        ).nonzero(as_tuple=False).squeeze(1)
        question_start = prompt.index(question)
        semantic = expanded_token_positions_for_char_span(
            self.processor.tokenizer,
            prompt,
            question_start,
            question_start + len(question),
            inputs["input_ids"][0].detach().cpu(),
            self.model.config.image_token_index,
        ).to(self.device)
        answer_positions = torch.empty(
            0, dtype=torch.long, device=self.device
        )
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
            # The scorer may consume template/question text, never gold answers.
            prompt_text = all_text[
                all_text < answer_positions.min()
            ]
        else:
            prompt_text = all_text
        if visual.numel() != 576:
            raise RuntimeError(
                f"expected 576 projected CLIP patches, found {visual.numel()}"
            )
        return PreparedBatch(
            inputs_embeds=embeddings,
            attention_mask=inputs["attention_mask"],
            visual_positions=visual,
            semantic_text_positions=semantic,
            prompt_text_positions=prompt_text,
            answer_positions=answer_positions,
            native_inputs=inputs,
            metadata={
                "question": question,
                "answer": answer,
                "prompt": prompt,
                "image_path": image_path,
                "vision_feature_layer": (
                    self.model.config.vision_feature_layer
                ),
                "vision_feature_select_strategy": (
                    self.model.config.vision_feature_select_strategy
                ),
            },
        )

    def teacher_forward(self, prepared):
        if prepared.answer_positions.numel():
            with torch.no_grad():
                output = self.model.language_model(
                    inputs_embeds=prepared.inputs_embeds,
                    attention_mask=prepared.attention_mask,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
            return {
                "hidden": output.hidden_states[-1].detach(),
                "logits": output.logits.detach(),
            }
        with torch.no_grad():
            hidden = self.model.language_model.model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=prepared.attention_mask,
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
        if prepared.answer_positions.numel():
            output = self.model.language_model(
                inputs_embeds=prepared.inputs_embeds,
                attention_mask=additive,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            return {
                "hidden": output.hidden_states[-1],
                "logits": output.logits,
            }
        hidden = self.model.language_model.model(
            inputs_embeds=prepared.inputs_embeds,
            attention_mask=additive,
            use_cache=False,
            return_dict=True,
        ).last_hidden_state
        return {"hidden": hidden}

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
            },
        )

    def generate_from_pruned(self, pruned, generation_config):
        max_new_tokens = int(generation_config["max_new_tokens"])
        eos = self.model.generation_config.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        cache_position = torch.arange(
            pruned.inputs_embeds.shape[1],
            device=pruned.inputs_embeds.device,
        )
        with torch.no_grad():
            output = self.model(
                inputs_embeds=pruned.inputs_embeds,
                attention_mask=pruned.attention_mask,
                position_ids=pruned.position_ids,
                cache_position=cache_position,
                use_cache=True,
                logits_to_keep=1,
                return_dict=True,
            )
        cache = output.past_key_values
        token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated = [token]
        next_rope_position = pruned.position_ids[:, -1:] + 1
        mask = pruned.attention_mask
        for _ in range(max_new_tokens - 1):
            if token.item() in eos:
                break
            mask = torch.cat(
                (mask, torch.ones_like(mask[:, :1])), dim=1
            )
            cache_position = torch.tensor(
                [cache.get_seq_length()],
                device=pruned.inputs_embeds.device,
            )
            with torch.no_grad():
                output = self.model(
                    input_ids=token,
                    attention_mask=mask,
                    position_ids=next_rope_position,
                    cache_position=cache_position,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
            cache = output.past_key_values
            token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
            generated.append(token)
            next_rope_position = next_rope_position + 1
        return torch.cat(generated, dim=1)

    def forward_pruned_with_answer(self, pruned):
        with torch.no_grad():
            output = self.model.language_model(
                inputs_embeds=pruned.inputs_embeds,
                attention_mask=pruned.attention_mask,
                position_ids=pruned.position_ids,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        return {
            "hidden": output.hidden_states[-1].detach(),
            "logits": output.logits.detach(),
        }
