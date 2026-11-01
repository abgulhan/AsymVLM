"""Native FastV / VL-Cache ports for Qwen2.5-VL (mRoPE) and Phi-3.5-Vision (su-RoPE).

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
import argparse

from contextlib import contextmanager

import json

import math

import os

from pathlib import Path

import sys

import time

import traceback

import torch

from transformers.cache_utils import DynamicCache

from learned_score.mlsys_runtime.generation import decoder_forward_function, _synchronize

from evaluation.baselines.native_llava import layer_counts, top_with_recent

MODELS = ('qwen25', 'phi35')

METHODS = ('fastv_native', 'vlcache_native')

PROJECT = Path(__file__).resolve().parents[2]

BASE = PROJECT

class UnsupportedBudget(ValueError):
    """A declared budget below a baseline's physical minimum; never relaxed."""


def fastv_budget(visual_tokens, layers, retention):
    if visual_tokens < 1 or layers <= 2 or not 1 <= retention <= 100:
        raise ValueError('invalid FastV geometry or retention')
    nominal = (visual_tokens * retention + 99) // 100
    retained = min(visual_tokens, max(1, math.ceil(
        (layers * nominal - 2 * visual_tokens) / (layers - 2))))
    feasible = 2 * visual_tokens + layers - 2 <= layers * nominal
    return nominal, retained, feasible


def validate_existing(record, expected):
    """Never reuse a path merely because it exists, including unsupported rows."""
    for key, value in expected.items():
        if record.get(key) != value:
            raise ValueError(f'existing native port record identity mismatch: {key}')
    if record.get('status') not in ('ok', 'unsupported_budget'):
        raise ValueError('existing native port record has invalid status')
    if record['status'] == 'ok':
        if (not math.isfinite(record.get('score', float('nan')))
                or not record.get('generated_ids') or not record.get('runtime')):
            raise ValueError('existing native port result is incomplete')
    elif not record.get('reason') or record.get('method') != 'fastv_native':
        raise ValueError('invalid unsupported-budget marker')


class NativeCache(DynamicCache):
    """Compatibility with Phi's snapshot API; no output-space reservation."""
    def get_usable_length(self, new_seq_length, layer_idx=0):
        return self.get_seq_length(layer_idx)

    def get_max_length(self):
        return None

    @property
    def seen_tokens(self):
        return self.get_seq_length()


@contextmanager
def native_sdpa(model):
    """Select the snapshot's own SDPA class, without changing any weight."""
    if not hasattr(model, 'native'):
        if model.config._attn_implementation != 'sdpa':
            raise ValueError('Qwen port requires the pinned SDPA runtime')
        yield
        return
    classes = [type(layer.self_attn) for layer in model.native.model.layers]
    native_class = getattr(sys.modules[classes[0].__module__], 'Phi3SdpaAttention')
    old = model.config._attn_implementation, model.native.model._attn_implementation
    try:
        for layer in model.native.model.layers:
            layer.self_attn.__class__ = native_class
        model.config._attn_implementation = model.native.model._attn_implementation = 'sdpa'
        yield
    finally:
        for layer, previous in zip(model.native.model.layers, classes):
            layer.self_attn.__class__ = previous
        model.config._attn_implementation, model.native.model._attn_implementation = old


def rotated_qk(attn, kwargs, query_positions):
    """Native projections/rotations, with only the requested query rows."""
    hidden = kwargs['hidden_states']
    batch, length, _ = hidden.shape
    if batch != 1:
        raise ValueError('one unpadded request required')
    impl = sys.modules[type(attn).__module__]
    if hasattr(attn, 'qkv_proj'):
        # Phi's fused QKV is deliberately called once; selection never sees labels.
        qkv = attn.qkv_proj(hidden)
        qend = attn.num_heads * attn.head_dim
        kend = qend + attn.num_key_value_heads * attn.head_dim
        q = qkv[..., :qend].view(batch, length, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = qkv[..., qend:kend].view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        v = qkv[..., kend:].view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = attn.rotary_emb(v, kwargs['position_ids'], seq_len=length)
        q, k = impl.apply_rotary_pos_emb(q, k, cos, sin, kwargs['position_ids'])
        q = q[..., query_positions, :]
    else:
        q = attn.q_proj(hidden[:, query_positions]).view(batch, -1, attn.num_heads, attn.head_dim).transpose(1, 2)
        k = attn.k_proj(hidden).view(batch, length, attn.num_key_value_heads, attn.head_dim).transpose(1, 2)
        cos, sin = kwargs['position_embeddings']
        sections = attn.rope_scaling['mrope_section']
        q, _ = impl.apply_multimodal_rotary_pos_emb(q, q, cos[..., query_positions, :], sin[..., query_positions, :], sections)
        k, _ = impl.apply_multimodal_rotary_pos_emb(k, k, cos, sin, sections)
    return q, impl.repeat_kv(k, attn.num_key_value_groups)


def attention_stats(attn, kwargs, query_positions, chunk_size=16):
    q, k = rotated_qk(attn, kwargs, query_positions)
    length = k.shape[-2]
    sums = torch.zeros((1, q.shape[1], length), device=q.device, dtype=torch.float32)
    nonzero = torch.zeros((), device=q.device, dtype=torch.float64)
    denominator = 0
    columns = torch.arange(length, device=q.device)
    mask = kwargs.get('attention_mask')
    for start in range(0, q.shape[-2], chunk_size):
        positions = query_positions[start:start+chunk_size]
        logits = (q[..., start:start+chunk_size, :].float() @ k.float().transpose(-1, -2)) / math.sqrt(attn.head_dim)
        allowed = columns[None] <= positions[:, None]
        if mask is not None:
            if mask.ndim != 4:
                raise ValueError('attention-statistics mask must be explicit 4D or None')
            part = mask[..., positions, :length]
            logits += part.float()
            allowed &= (part[0, 0] > torch.finfo(part.dtype).min / 2)
        logits.masked_fill_(~allowed[None, None], torch.finfo(logits.dtype).min)
        probs = logits.softmax(-1)
        sums += probs.sum(-2)
        nonzero += ((probs >= .01 * probs.amax(-1, keepdim=True)) & allowed[None, None]).sum()
        denominator += int(allowed.sum()) * q.shape[1]
    heads = attn.num_key_value_heads
    scores = sums.view(1, heads, attn.num_key_value_groups, length).mean(2)
    return scores, float(nonzero / max(1, denominator))


def original_positions(model, prepared):
    if model.config.model_type in ('qwen2_vl', 'qwen2_5_vl'):
        values = prepared.metadata['position_ids']
        if values.shape != (3, 1, prepared.inputs_embeds.shape[1]):
            raise ValueError('native three-axis Qwen mRoPE positions required')
        return values
    return torch.arange(prepared.inputs_embeds.shape[1], device=prepared.inputs_embeds.device)[None]


class NativePruning:
    def __init__(self, model, prepared, method, percent):
        self.model, self.prepared, self.method, self.percent = model, prepared, method, percent
        self.decoder = model.model
        if model.config.model_type not in ('qwen2_vl', 'qwen2_5_vl', 'phi3_v'):
            raise ValueError('only pinned Qwen/Phi models are ported')
        self.is_phi = hasattr(model, 'native')
        self.layers = self.decoder.layers
        self.length = prepared.inputs_embeds.shape[1]
        self.visual = prepared.visual_positions
        self.nominal_visual, self.fastv_count, self.fastv_feasible = fastv_budget(len(self.visual), len(self.layers), percent)
        self.target_slots = len(self.layers) * (self.length - len(self.visual) + self.nominal_visual)
        # At tiny visual budgets, two full layers alone can exceed the target.
        self.keep = torch.arange(self.length, device=self.visual.device)
        self.scores, self.densities, self.sources, self.handles = {}, {}, {}, []
        self.prefill = True
        self.logical_length = self.length
        self.prefix_lengths = []

    def collect(self, index, module, args, kwargs):
        if not self.prefill or self.percent == 100:
            return
        if self.method == 'fastv_native':
            query = torch.tensor([self.length - 1], device=self.visual.device)
        else:
            last_image = int(self.visual.max())
            protected = self.prepared.metadata.get('protected_visual_positions')
            if protected is not None and len(protected):
                last_image = max(last_image, int(protected.max()))
            query = torch.arange(last_image + 1, self.length, device=self.visual.device)
            if not len(query):
                raise ValueError('VL-Cache needs post-image prompt tokens')
        scores, density = attention_stats(module, kwargs, query)
        if self.method == 'fastv_native':
            order = torch.argsort(scores.mean((0, 1))[self.visual], descending=True, stable=True)
            keep = torch.ones(self.length, dtype=torch.bool, device=self.visual.device)
            keep[self.visual] = False
            keep[self.visual[order[:self.fastv_count]]] = True
            self.keep = keep.nonzero().flatten()
        else:
            self.scores[index], self.densities[index] = scores, density

    def layer_hook(self, index, module, args, kwargs):
        if self.percent == 100:
            return
        if self.prefill and self.method == 'fastv_native' and index >= 2:
            if index == 2:
                if args:
                    args = (args[0][:, self.keep],) + args[1:]
                else:
                    kwargs['hidden_states'] = kwargs['hidden_states'][:, self.keep]
            for key in ('position_ids', 'cache_position'):
                if kwargs.get(key) is not None:
                    kwargs[key] = kwargs[key][..., self.keep]
            if kwargs.get('position_embeddings') is not None:
                kwargs['position_embeddings'] = tuple(x[..., self.keep, :] for x in kwargs['position_embeddings'])
            mask = kwargs.get('attention_mask')
            if mask is not None:
                kwargs['attention_mask'] = mask[..., self.keep, :][..., self.keep]
        elif not self.prefill:
            # mRoPE coordinates are NOT token sequence indices. Visibility uses
            # source indices, while native position_ids keep the spatial axes.
            sources = self.sources[index]
            position = self.logical_length - 1
            now = torch.full((*sources.shape[:-1], 1), position, device=sources.device, dtype=torch.long)
            keys = torch.cat((sources, now), -1)
            self.sources[index] = keys
            allowed = keys <= position
            window = getattr(module.self_attn, 'sliding_window', None)
            if self.is_phi:
                window = getattr(self.decoder.config, 'sliding_window', None)
            if window:
                allowed &= keys > position - window
            hidden = args[0] if args else kwargs['hidden_states']
            if bool(allowed.all()):
                # All compacted keys precede this single query, for every head.
                # Phi's native SDPA validates a shared 1-head mask shape.
                mask = torch.zeros((1, 1, 1, keys.shape[-1]), device=hidden.device, dtype=hidden.dtype)
            else:
                if self.is_phi and allowed.shape[1] > 1:
                    raise NotImplementedError('Phi head-varying sliding-window masks need a separately validated kernel')
                mask = torch.zeros((*keys.shape[:-1], 1, keys.shape[-1]), device=hidden.device, dtype=hidden.dtype)
                mask.masked_fill_(~allowed.unsqueeze(-2), torch.finfo(hidden.dtype).min)
                if mask.shape[1] != 1:
                    mask = mask.repeat_interleave(module.self_attn.num_key_value_groups, dim=1)
            kwargs['attention_mask'] = mask
        return args, kwargs

    def rope_hook(self, module, args, kwargs):
        # Phi chooses short/long RoPE factors using seq_len. Keep the original
        # logical length, not the number of entries surviving KV compaction.
        kwargs['seq_len'] = self.logical_length
        return args, kwargs

    def __enter__(self):
        if self.method == 'fastv_native' and not self.fastv_feasible:
            raise UnsupportedBudget('FastV first two full layers exceed this requested aggregate budget')
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda m, a, kw, i=index: self.layer_hook(i, m, a, kw), with_kwargs=True))
            if self.method == 'vlcache_native' or (self.method == 'fastv_native' and index == 1):
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    lambda m, a, kw, i=index: self.collect(i, m, a, kw), with_kwargs=True))
            if self.is_phi:
                self.handles.append(layer.self_attn.rotary_emb.register_forward_pre_hook(self.rope_hook, with_kwargs=True))
        return self

    def compact(self, cache):
        if self.method == 'vlcache_native' and self.percent != 100:
            counts = layer_counts([self.densities[i] for i in range(len(self.layers))], self.target_slots, self.length)
        else:
            counts = [cache.key_cache[i].shape[-2] for i in range(len(self.layers))]
        for index, count in enumerate(counts):
            if self.method == 'vlcache_native' and self.percent != 100:
                selected = top_with_recent(self.scores[index], count)
                for store in (cache.key_cache, cache.value_cache):
                    store[index] = store[index].gather(2, selected.unsqueeze(-1).expand(-1, -1, -1, store[index].shape[-1])).contiguous()
                self.sources[index] = selected
            else:
                source = self.keep if self.method == 'fastv_native' and index >= 2 else torch.arange(self.length, device=self.visual.device)
                self.sources[index] = source[None, None]
        self.prefix_lengths = counts
        self.prefill = False
        self.scores.clear()
        error = sum(counts) - self.target_slots
        if error and (self.method != 'fastv_native' or not 0 <= error < len(self.layers) - 2):
            raise AssertionError('physical cache budget not conserved')

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def generate_native(model, prepared, method, percent, max_new_tokens=128, stop_on_eos=True):
    if method not in ('all_tokens', *METHODS) or not 1 <= percent <= 100:
        raise ValueError('unsupported native port request')
    if prepared.inputs_embeds.shape[0] != 1 or not prepared.attention_mask.bool().all():
        raise ValueError('one unpadded request required')
    if max_new_tokens < 1:
        raise ValueError('positive output cap required')
    length = prepared.inputs_embeds.shape[1]
    device = prepared.inputs_embeds.device
    positions = original_positions(model, prepared)
    cache = NativeCache()
    forward = decoder_forward_function(model)
    eos = getattr(model.generation_config, 'eos_token_id', None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    _synchronize(device)
    started = time.perf_counter()
    with NativePruning(model, prepared, method, 100 if method == 'all_tokens' else percent) as policy:
        output = forward(inputs_embeds=prepared.inputs_embeds, attention_mask=prepared.attention_mask,
            position_ids=positions, cache_position=torch.arange(length, device=device),
            past_key_values=cache, use_cache=True, output_attentions=False)
        if output.past_key_values is not cache:
            raise AssertionError('native backbone replaced supplied cache')
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated = [token]
        policy.compact(cache)
        prefix_bytes = sum(x.numel() * x.element_size() for seq in (cache.key_cache, cache.value_cache) for x in seq)
        _synchronize(device)
        ttft = time.perf_counter() - started
        position_ids = positions[..., -1:] + 1
        for step in range(max_new_tokens - 1):
            if stop_on_eos and int(token.item()) in eos:
                break
            policy.logical_length = length + step + 1
            output = forward(input_ids=token, attention_mask=None,
                position_ids=position_ids, cache_position=torch.tensor([length + step], device=device),
                past_key_values=cache, use_cache=True, output_attentions=False)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
            position_ids = position_ids + 1
        _synchronize(device)
        elapsed = time.perf_counter() - started
        final_bytes = sum(x.numel() * x.element_size() for seq in (cache.key_cache, cache.value_cache) for x in seq)
        return torch.cat(generated, 1), dict(prefill_ttft_seconds=ttft, total_seconds=elapsed,
            decode_seconds=elapsed - ttft, prefix_kv_bytes=prefix_bytes, final_kv_bytes=final_bytes,
            prefix_tokens_per_layer=policy.prefix_lengths, target_prefix_slots=policy.target_slots,
            actual_prefix_slots=sum(policy.prefix_lengths), input_prefix_tokens=length,
            budget_error_slots=sum(policy.prefix_lengths) - policy.target_slots,
            visual_tokens=len(policy.visual), nominal_retained_visual_tokens=policy.nominal_visual,
            fastv_pruning_layer=2 if method == 'fastv_native' else None,
            fastv_retained_visual_tokens=policy.fastv_count if method == 'fastv_native' else None,
            layer_density=policy.densities, physical_cache_compaction=True,
            prompt_text_prefill_protected=True, prompt_text_cache_protected=method != 'vlcache_native',
            original_positions_preserved=True, attention_weights_requested=False,
            selection_statistics_chunk_size=16, backend='native_sdpa')


def first_panel(entries, per_dataset):
    grouped = {}
    for entry in entries:
        grouped.setdefault(entry['dataset'], []).append(entry)
    # Reuse the locked benchmark order, not a second independently sampled panel.
    return [entry for dataset in sorted(grouped) for entry in sorted(grouped[dataset], key=lambda e: (e['panel_order'], e['row_hash']))[:per_dataset]]


