"""Phi-3.5-Vision VisionZip adaptation (CLS-attention dominant groups + contextual merging).

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
import argparse

from contextlib import contextmanager

from dataclasses import replace

import json

import math

import os

from pathlib import Path

import signal

import socket

import time

import traceback

import torch

PROJECT = Path(__file__).resolve().parents[2]

METHOD = "visionzip_phi_native"

@contextmanager
def visionzip_capture(adapter):
    embed = adapter.model.model.vision_embed_tokens
    if (embed.layer_idx != -2 or embed.type_feature != "patch"
            or embed.hd_transform_order != "sub_glb"):
        raise ValueError("unvalidated Phi CLIP/HD configuration")
    layer = embed.img_processor.vision_model.encoder.layers[-2]
    result = {}

    def attention_hook(module, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        b, n, _ = hidden.shape
        q = module.q_proj(hidden[:, :1]).view(b, 1, module.num_heads, module.head_dim).transpose(1, 2)
        k = module.k_proj(hidden).view(b, n, module.num_heads, module.head_dim).transpose(1, 2)
        result["attention"] = ((q.float() @ k.float().transpose(-1, -2)) * module.scale).softmax(-1).mean(1)[:, 0, 1:].detach()
        result["keys"] = k.mean(1)[:, 1:].detach()

    def features_hook(module, args, output):
        result["features"] = output[0][:, 1:].detach()

    handles = [layer.self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True),
               layer.register_forward_hook(features_hook)]
    try:
        yield result
    finally:
        for handle in handles:
            handle.remove()


def group_2x2(tensor, *, concatenate):
    """Native TL,TR,BL,BR channel order; no spatial averaging for features."""
    if tensor.ndim != 3 or tensor.shape[1] != 576:
        raise ValueError("expected native [crops, 576, channels] CLIP patches")
    b, _, c = tensor.shape
    grouped = tensor.reshape(b, 12, 2, 12, 2, c).permute(0, 1, 3, 2, 4, 5)
    grouped = grouped.reshape(b, 12, 12, 4, c)
    return grouped.flatten(-2) if concatenate else grouped.float().mean(-2)


def pack_statistics(captured, image_sizes):
    sizes = torch.as_tensor(image_sizes).reshape(-1, 2).tolist()
    if len(sizes) != 1:
        raise ValueError("Phi VisionZip currently supports exactly one image")
    height, width = sizes[0]
    if min(height, width) < 336 or height % 336 or width % 336:
        raise ValueError("invalid Phi native crop dimensions")
    h, w = int(height) // 336, int(width) // 336
    crops = h * w
    packed = {}
    for key in ("features", "attention", "keys"):
        value = captured[key]
        if key == "attention":
            value = value.unsqueeze(-1)
        if len(value) < crops + 1:
            raise ValueError("captured statistics omit native crops")
        grouped = group_2x2(value[:crops + 1], concatenate=key == "features")
        channels = grouped.shape[-1]
        local = grouped[1:].reshape(h, w, 12, 12, channels).permute(0, 2, 1, 3, 4)
        packed[key] = torch.cat((local.reshape(-1, channels), grouped[0].reshape(-1, channels)))
    packed["attention"] = packed["attention"].flatten()
    return packed


def crop_budgets(groups, count):
    """Exact proportional largest-remainder allocation; deterministic crop ties."""
    normalized = [tuple(g) for g in groups]
    keys = sorted(set(normalized))
    sizes = [normalized.count(key) for key in keys]
    total = sum(sizes)
    if not 1 <= count <= total:
        raise ValueError("invalid exact token budget")
    if count < len(keys):
        raise ValueError("budget is infeasible: at least one token per native crop is required")
    raw = [count * size / total for size in sizes]
    result = [math.floor(value) for value in raw]
    remaining = count - sum(result)
    order = sorted(range(len(keys)), key=lambda i: (-(raw[i] - result[i]), keys[i]))
    for index in order[:remaining]:
        result[index] += 1
    if count >= len(keys) and any(value == 0 for value in result):
        raise ValueError("unexpected empty crop at a feasible crop-preserving budget")
    return dict(zip(keys, result))


@torch.no_grad()
def select_groups(packed, groups, count, contextual_fraction=0.05):
    from learned_score.qwen2_pruning import visionzip_selection
    n = len(packed["features"])
    if len(groups) != n or len(packed["keys"]) != n or len(packed["attention"]) != n:
        raise ValueError("native crop statistics/positions mismatch")
    if not 0 < contextual_fraction < 1:
        raise ValueError("contextual fraction must be between zero and one")
    normalized = [tuple(g) for g in groups]
    budgets = crop_budgets(normalized, count)
    selections, replacements, detail = [], [], []
    for key, keep in budgets.items():
        indices = torch.tensor([i for i, g in enumerate(normalized) if g == key],
                               dtype=torch.long, device=packed["features"].device)
        if not keep:
            detail.append(dict(crop=list(key), input_tokens=len(indices), kept_tokens=0,
                               dominant_tokens=0, contextual_tokens=0))
            continue
        local, merged, counts = visionzip_selection(packed["features"][indices],
            packed["attention"][indices], packed["keys"][indices], keep, contextual_fraction)
        selections.append(indices[local])
        replacements.append(merged)
        detail.append(dict(crop=list(key), input_tokens=len(indices), kept_tokens=keep, **counts))
    selected = torch.cat(selections)
    replacement = torch.cat(replacements)
    order = selected.argsort()
    selected, replacement = selected[order], replacement[order]
    if len(selected) != count or len(selected.unique()) != count:
        raise ValueError("VisionZip failed exact physical token conservation")
    return selected, replacement, dict(crops=detail,
        dominant_tokens=sum(x["dominant_tokens"] for x in detail),
        contextual_tokens=sum(x["contextual_tokens"] for x in detail),
        contextual_fraction_of_original_crop=contextual_fraction)


@torch.no_grad()
def visionzip_prune(adapter, prepared, captured, count, contextual_fraction=0.05):
    from learned_score.mlsys_sprint.phi_adaptive_v3 import phi_image_layout
    n = len(prepared.visual_positions)
    if not 1 <= count <= n or prepared.answer_positions.numel():
        raise ValueError("invalid count or answer leakage")
    visual, protected, groups = phi_image_layout(prepared.native_inputs["input_ids"],
                                               prepared.native_inputs["image_sizes"])
    if (not torch.equal(visual, prepared.visual_positions)
            or [tuple(g) for g in groups] != [tuple(g) for g in prepared.metadata["visual_crop_groups"]]):
        raise ValueError("Phi native crop/visual geometry changed")
    packed = pack_statistics(captured, prepared.native_inputs["image_sizes"])
    embed = adapter.model.model.vision_embed_tokens
    reconstructed = embed.img_projection(packed["features"].to(prepared.inputs_embeds.dtype))
    original = prepared.inputs_embeds[0, prepared.visual_positions]
    error = float((reconstructed.float() - original.float()).abs().max())
    if not torch.allclose(reconstructed, original, atol=.025, rtol=.025):
        raise ValueError(f"Phi native pre-projector feature alignment failed: {error}")
    selected, merged, detail = select_groups(packed, groups, count, contextual_fraction)
    if count == n:
        changed = prepared  # exact identity, not a numerically reprojected copy
    else:
        updated = prepared.inputs_embeds.clone()
        updated[0, prepared.visual_positions[selected]] = embed.img_projection(merged).to(updated.dtype)
        changed = replace(prepared, inputs_embeds=updated)
    pruned = adapter.physically_prune(changed, selected, "original")
    if not torch.equal(changed.inputs_embeds[:, protected], prepared.inputs_embeds[:, protected]):
        raise ValueError("protected Phi separator embeddings changed")
    if not torch.isin(protected, pruned.retained_source_positions).all():
        raise ValueError("protected Phi separator positions were removed")
    if pruned.metadata["visual_tokens"] != count:
        raise ValueError("physical retained count changed")
    return pruned, dict(**detail, selected_indices=selected.cpu().tolist(),
        reconstruction_max_abs_error=error, protected_visual_tokens=len(protected),
        merge_space="pre_projector_4096_dim_native_2x2_clip_groups",
        adaptation="per-native-crop; stitched local then global; original positions and separators",
        budget_includes_contextual_tokens=True, cross_crop_merging=False)


@contextmanager
def cache_compatibility():
    from unittest.mock import patch
    from learned_score.mlsys_runtime import dynamic
    class CompatibleCache(dynamic.GrowingDynamicCache):
        def get_usable_length(self, new_seq_length, layer_idx=0):
            return self.get_seq_length(layer_idx)
        def get_max_length(self):
            return None
    with patch.object(dynamic, "GrowingDynamicCache", CompatibleCache):
        yield


def generate(adapter, pruned, cap):
    from learned_score.mlsys_runtime.dynamic import generate_dynamic
    with cache_compatibility():
        return generate_dynamic(adapter.model, pruned, max_new_tokens=cap,
                                stop_on_eos=True, require_fused_attention=False)


def physical(runtime, entry, percent, cap=128, contextual_fraction=0.05):
    from learned_score.mlsys_sprint.manuscript_data_v35 import family
    from learned_score.mlsys_sprint.manuscript_execute_v35 import score
    from learned_score.mlsys_sprint.adaptive_experiment_v3 import benchmark
    from learned_score.mlsys_scorer.backbone import assert_frozen
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    with torch.inference_mode(), visionzip_capture(runtime.adapter) as captured:
        raw, prepared, _ = runtime.geometry(entry)
    n = len(prepared.visual_positions)
    count = (n * percent + 99) // 100
    with torch.inference_mode():
        pruned, detail = visionzip_prune(runtime.adapter, prepared, captured, count, contextual_fraction)
        result = generate(runtime.adapter, pruned, cap)
    prediction = runtime.adapter.processor.decode(result.ids[0], skip_special_tokens=True).strip()
    config = runtime.adapter.model.config
    layers = config.num_hidden_layers
    per_token = layers * 2 * config.num_key_value_heads * (config.hidden_size // config.num_attention_heads) * prepared.inputs_embeds.element_size()
    final = result.cache.memory_stats()["kv_active_bytes"]
    prefix = pruned.inputs_embeds.shape[1] * per_token
    if final != prefix + (result.ids.shape[1] - 1) * per_token:
        raise ValueError("physical KV-cache bytes differ from exact retained prefix")
    assert_frozen(runtime.adapter.model, runtime.versions)
    assert_frozen(runtime.scorer, runtime.scorer_versions)
    labeled = runtime.labels.row_for_entry(entry)
    if labeled["question"] != raw["question"]:
        raise ValueError("label/input identity mismatch")
    refs = benchmark.parse_references(labeled["answer"])
    return dict(status="ok", method=METHOD, model="phi35", dataset=family(entry["dataset"]),
        request_id=entry["row_hash"], group_id=entry["group_id"], panel_order=entry["panel_order"],
        percent=percent, input_visual_tokens=n, retained_visual_tokens=count,
        input_prefix_tokens=prepared.inputs_embeds.shape[1], retained_prefix_tokens=pruned.inputs_embeds.shape[1],
        prediction=prediction, quality_score=score(prediction, refs, entry["metric"]), metric=entry["metric"],
        generated_ids=result.ids[0].cpu().tolist(), max_new_tokens=cap, selection=detail,
        metrics={**result.metrics, "prefix_kv_bytes":prefix, "final_kv_bytes":final},
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), request_seconds=time.perf_counter()-started,
        baseline_fidelity="Phi architecture adaptation of released VisionZip; not upstream end-to-end reproduction",
        timing_isolated=False, physical_measurement_performed=True,
        source_sha256=source_hashes(), time=time.time(), host=None,
        attention_backend=runtime.adapter.model.config._attn_implementation,
        job_id=None)


def source_hashes():
    from learned_score.mlsys_sprint.state import file_digest
    from learned_score import qwen2_pruning
    return {Path(__file__).name:file_digest(__file__),
            "qwen2_pruning.py":file_digest(qwen2_pruning.__file__),
            "upstream_clip_encoder.py":file_digest(Path(__import__("os").environ.get("VISIONZIP_SRC", "external_baselines/VisionZip"))/"visionzip/clip_encoder.py")}


def canary(runtime, entry, root, contextual_fraction):
    from learned_score.mlsys_sprint.state import atomic_json
    with torch.inference_mode():
        _, prepared, _ = runtime.geometry(entry)
        full = runtime.adapter.physically_prune(prepared, torch.arange(len(prepared.visual_positions), device=prepared.inputs_embeds.device), "original")
        expected = generate(runtime.adapter, full, 32).ids[0].cpu().tolist()
    actual = physical(runtime, entry, 100, 32, contextual_fraction)
    if actual["generated_ids"] != expected:
        raise ValueError("Phi VisionZip full-token generated IDs differ")
    reduced = physical(runtime, entry, 30, 32, contextual_fraction)
    if reduced["metrics"]["prefix_kv_bytes"] >= actual["metrics"]["prefix_kv_bytes"]:
        raise ValueError("Phi VisionZip did not physically reduce cache")
    value = dict(passed=True, full_parity=True, unchanged_native_positions=True,
                 all_separators_protected=True, full=actual, reduced=reduced,
                 source_sha256=source_hashes())
    atomic_json(root/"canary.json", value)
    print(json.dumps(dict(event="canary_passed", request_id=entry["row_hash"])), flush=True)


