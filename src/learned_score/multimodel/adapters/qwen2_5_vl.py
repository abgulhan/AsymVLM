"""Qwen2.5-VL adapter preserving visual spans and multimodal RoPE."""

import torch

from learned_score.qwen25_pruning import extract_visual_features

from .qwen2_vl import Qwen2VLAdapter


class Qwen2_5_VLAdapter(Qwen2VLAdapter):
    name = "qwen2_5_vl"
    version = "1"

    def load_model_and_processor(self, config):
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        revision = getattr(config, "model_revision", None)
        self.processor = AutoProcessor.from_pretrained(
            config.model,
            revision=(
                getattr(config, "processor_revision", None) or revision
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
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.model,
            revision=revision,
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
        return extract_visual_features(
            self.model,
            pixel_values,
            image_grid_thw,
            with_stats=with_stats,
        )
