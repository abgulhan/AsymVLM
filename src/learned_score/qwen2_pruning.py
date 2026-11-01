"""Qwen2-VL visual pruning utilities with multimodal-RoPE preservation."""

import math

import torch
import torch.nn.functional as F
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    apply_rotary_pos_emb_vision,
)


@torch.no_grad()
def extract_visual_features(model, pixel_values, grid_thw, with_stats=False):
    """Run Qwen2's visual tower and optionally expose VisionZip statistics."""
    visual = model.visual
    hidden = visual.patch_embed(pixel_values.to(visual.get_dtype()))
    rotary = visual.rot_pos_emb(grid_thw)
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
        if with_stats and layer_index == len(visual.blocks) - 1:
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
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
        )

    merge_unit = visual.spatial_merge_size**2
    image_features = visual.merger(hidden)
    if not with_stats:
        return image_features, None, None

    if hidden.shape[0] % merge_unit:
        raise ValueError(
            "Qwen2 visual patch count is not divisible by PatchMerger size"
        )
    scores = received_attention.reshape(-1, merge_unit).mean(dim=-1)
    keys = mean_keys.reshape(-1, merge_unit, mean_keys.shape[-1]).mean(dim=1)
    if len(scores) != len(image_features):
        raise ValueError("VisionZip statistics do not match visual features")
    return image_features, scores, keys


def query_selection(image_features, text_features, keep_count):
    image = F.normalize(image_features.float(), dim=-1)
    text = F.normalize(text_features.float(), dim=-1)
    scores = (image @ text.T).max(dim=-1).values
    return scores.topk(keep_count).indices.sort().values


def visionzip_selection(
    image_features,
    received_attention,
    keys,
    keep_count,
    contextual_fraction=0.05,
):
    """Apply the released Qwen VisionZip dominant/contextual construction."""
    token_count = len(image_features)
    keep_count = min(max(1, keep_count), token_count)
    contextual_count = min(
        max(1, int(token_count * contextual_fraction)),
        max(0, keep_count - 1),
    )
    dominant_count = keep_count - contextual_count
    dominant = received_attention.topk(dominant_count).indices
    dominant_mask = torch.zeros(
        token_count, dtype=torch.bool, device=image_features.device
    )
    dominant_mask[dominant] = True
    remaining = (~dominant_mask).nonzero(as_tuple=False).squeeze(1)

    if contextual_count == 0 or len(remaining) == 0:
        selected = dominant.sort().values
        return selected, image_features[selected], {
            "dominant_tokens": int(len(selected)),
            "contextual_tokens": 0,
        }

    step = max(1, len(remaining) // contextual_count)
    seed_relative = torch.arange(
        0, len(remaining), step, device=remaining.device
    )[:contextual_count]
    seed_indices = remaining[seed_relative]
    nonseed_mask = torch.ones(
        len(remaining), dtype=torch.bool, device=remaining.device
    )
    nonseed_mask[seed_relative] = False
    merge_indices = remaining[nonseed_mask]

    normalized_keys = F.normalize(keys.float(), dim=-1)
    contextual = image_features[seed_indices].float().clone()
    if len(merge_indices):
        assignments = (
            normalized_keys[merge_indices]
            @ normalized_keys[seed_indices].T
        ).argmax(dim=-1)
        sums = torch.zeros_like(contextual)
        sums.index_add_(0, assignments, image_features[merge_indices].float())
        counts = torch.zeros(
            contextual_count,
            device=image_features.device,
            dtype=torch.float32,
        )
        counts.index_add_(
            0,
            assignments,
            torch.ones_like(assignments, dtype=torch.float32),
        )
        contextual = contextual + sums / counts.clamp_min(1).unsqueeze(-1)

    replacement = {
        int(index): value.to(image_features.dtype)
        for index, value in zip(seed_indices.tolist(), contextual)
    }
    selected = torch.cat((dominant, seed_indices)).sort().values
    selected_features = torch.stack(
        [
            replacement.get(int(index), image_features[index])
            for index in selected.tolist()
        ]
    )
    return selected, selected_features, {
        "dominant_tokens": int(dominant_count),
        "contextual_tokens": int(contextual_count),
    }


def build_pruned_inputs(
    model,
    inputs,
    image_features,
    selected=None,
    selected_features=None,
):
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    image_mask = input_ids == model.config.image_token_id
    image_positions = image_mask[0].nonzero(as_tuple=False).squeeze(1)
    if len(image_positions) != len(image_features):
        raise ValueError(
            "Qwen2 image placeholders and visual features do not match"
        )

    embeddings = model.model.embed_tokens(input_ids)
    embeddings[0, image_positions] = image_features.to(embeddings.dtype)
    position_ids, _ = model.get_rope_index(
        input_ids,
        inputs.get("image_grid_thw"),
        inputs.get("video_grid_thw"),
        attention_mask,
    )
    if selected is None:
        return embeddings, attention_mask, position_ids, image_mask

    selected = selected.sort().values
    keep_mask = (~image_mask[0]) & attention_mask[0].bool()
    keep_mask[image_positions[selected]] = True
    positions = keep_mask.nonzero(as_tuple=False).squeeze(1)
    pruned_embeddings = embeddings[:, positions].clone()
    pruned_positions = position_ids[:, :, positions]
    pruned_mask = torch.ones(
        (1, len(positions)),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )

    selected_absolute = image_positions[selected]
    insertion = torch.searchsorted(positions, selected_absolute)
    if selected_features is None:
        selected_features = image_features[selected]
    pruned_embeddings[0, insertion] = selected_features.to(
        pruned_embeddings.dtype
    )
    return pruned_embeddings, pruned_mask, pruned_positions, image_mask


@torch.no_grad()
def greedy_generate(model, embeddings, attention_mask, position_ids, max_tokens):
    cache_position = torch.arange(
        embeddings.shape[1], device=embeddings.device
    )
    output = model.model(
        inputs_embeds=embeddings,
        attention_mask=attention_mask,
        position_ids=position_ids,
        cache_position=cache_position,
        use_cache=True,
        return_dict=True,
    )
    next_token = model.lm_head(output.last_hidden_state[:, -1]).argmax(
        dim=-1, keepdim=True
    )
    generated = [next_token]
    past = output.past_key_values
    next_position = position_ids[:, :, -1:] + 1

    for _ in range(max_tokens - 1):
        if next_token.item() == model.config.eos_token_id:
            break
        past_length = past.get_seq_length()
        step_mask = torch.ones(
            (1, past_length + 1),
            device=attention_mask.device,
            dtype=attention_mask.dtype,
        )
        output = model.model(
            input_ids=next_token,
            attention_mask=step_mask,
            position_ids=next_position,
            past_key_values=past,
            cache_position=torch.tensor(
                [past_length], device=embeddings.device
            ),
            use_cache=True,
            return_dict=True,
        )
        next_token = model.lm_head(output.last_hidden_state[:, -1]).argmax(
            dim=-1, keepdim=True
        )
        generated.append(next_token)
        past = output.past_key_values
        next_position = next_position + 1
    return torch.cat(generated, dim=1)
