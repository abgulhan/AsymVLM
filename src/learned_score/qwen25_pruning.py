"""Qwen2.5-VL visual extraction with native window-order parity."""

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)


@torch.no_grad()
def extract_visual_features(model, pixel_values, grid_thw, with_stats=False):
    """Replicate Qwen2.5-VL's vision tower and expose optional final-block stats."""

    visual = model.visual
    hidden = visual.patch_embed(pixel_values.to(visual.dtype))
    rotary = visual.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = visual.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden.device,
        dtype=torch.int32,
    ).unique_consecutive()

    sequence_length = hidden.shape[0]
    merge_unit = visual.spatial_merge_unit
    hidden = hidden.reshape(
        sequence_length // merge_unit, merge_unit, -1
    )[window_index].reshape(sequence_length, -1)
    rotary = rotary.reshape(
        sequence_length // merge_unit, merge_unit, -1
    )[window_index].reshape(sequence_length, -1)
    repeated_rotary = torch.cat((rotary, rotary), dim=-1)
    position_embeddings = (
        repeated_rotary.cos(),
        repeated_rotary.sin(),
    )
    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
    ).cumsum(dim=0, dtype=torch.int32)
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    received_attention = None
    mean_keys = None
    for layer_index, block in enumerate(visual.blocks):
        current_cu = (
            cu_seqlens
            if layer_index in visual.fullatt_block_indexes
            else cu_window_seqlens
        )
        if with_stats and layer_index == len(visual.blocks) - 1:
            if layer_index not in visual.fullatt_block_indexes:
                raise RuntimeError(
                    "Qwen2.5 VisionZip statistics require a final full-attention block"
                )
            normalized = block.norm1(hidden)
            token_count = normalized.shape[0]
            q, k, _ = (
                block.attn.qkv(normalized)
                .reshape(token_count, 3, block.attn.num_heads, -1)
                .permute(1, 0, 2, 3)
                .unbind(0)
            )
            q, k = apply_rotary_pos_emb_vision(
                q,
                k,
                position_embeddings[0],
                position_embeddings[1],
            )
            q = q.transpose(0, 1).float()
            k = k.transpose(0, 1).float()
            weights = torch.softmax(
                q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]),
                dim=-1,
            )
            received_attention = weights.mean(dim=0).sum(dim=0)
            mean_keys = k.mean(dim=0)
        hidden = block(
            hidden,
            cu_seqlens=current_cu,
            position_embeddings=position_embeddings,
        )

    image_features = visual.merger(hidden)
    reverse_indices = torch.argsort(window_index)
    image_features = image_features[reverse_indices]
    if not with_stats:
        return image_features, None, None

    scores = received_attention.reshape(-1, merge_unit).mean(dim=-1)
    keys = mean_keys.reshape(
        -1, merge_unit, mean_keys.shape[-1]
    ).mean(dim=1)
    scores = scores[reverse_indices]
    keys = keys[reverse_indices]
    if len(scores) != len(image_features):
        raise ValueError("Qwen2.5 VisionZip statistics do not match visual features")
    return image_features, scores, keys
