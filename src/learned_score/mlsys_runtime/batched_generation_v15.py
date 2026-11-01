"""True shared-weight, left-padded DynamicCache decoding for short VLM batches.

No cache eviction, continuous batching, replica concurrency or capacity model is
implemented here. Original image/prompt positions and native local visibility
are preserved. Every padded K/V slot is counted in actual tensor storage.
"""
from dataclasses import dataclass
import time

import torch
from transformers.cache_utils import DynamicCache

from .generation import _synchronize, attention_backend, decoder_forward_function
from .masking import native_sliding_window


@dataclass
class PaddedBatch:
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    source_positions: torch.Tensor
    valid: torch.Tensor
    prefix_lengths: list[int]


@dataclass
class BatchedResult:
    tokens: list[list[int]]
    metrics: dict
    cache: DynamicCache


def left_pad(pruned, *, pad_to=None):
    if not pruned:
        raise ValueError('a nonempty batch is required')
    first = pruned[0].inputs_embeds
    if first.ndim != 3 or first.shape[0] != 1:
        raise ValueError('each prepared prefix must have batch size one')
    lengths = [int(p.inputs_embeds.shape[1]) for p in pruned]
    width = max(lengths) if pad_to is None else pad_to
    if type(width) is not int or width < max(lengths) or min(lengths) < 1:
        raise ValueError('invalid padded width')
    shape = pruned[0].position_ids.shape
    if len(shape) not in (2, 3) or shape[-2] != 1 or (len(shape) == 3 and shape[0] != 3):
        raise ValueError('expected Mistral [1,L] or Qwen mRoPE [3,1,L] positions')
    embeddings = first.new_zeros((len(pruned), width, first.shape[-1]))
    positions = torch.zeros(((3, len(pruned), width) if len(shape) == 3 else (len(pruned), width)),
                            device=first.device, dtype=torch.long)
    sources = torch.full((len(pruned), width), -1, dtype=torch.long, device=first.device)
    valid = torch.zeros((len(pruned), width), dtype=torch.bool, device=first.device)
    for index, (item, length) in enumerate(zip(pruned, lengths)):
        if (item.inputs_embeds.shape != (1, length, first.shape[-1])
                or item.inputs_embeds.dtype != first.dtype or item.inputs_embeds.device != first.device
                or item.position_ids.shape[:-1] != shape[:-1]
                or item.position_ids.shape[-1] != length
                or item.position_ids.dtype != torch.long or item.position_ids.device != first.device
                or item.attention_mask.shape != (1, length)
                or not bool(item.attention_mask.bool().all())):
            raise ValueError('incompatible or already-padded prepared prefix')
        source = item.retained_source_positions
        if (source.shape != (length,) or source.dtype != torch.long
                or bool((source < 0).any()) or bool((source[1:] <= source[:-1]).any())):
            raise ValueError('original retained source positions must increase')
        start = width - length
        embeddings[index, start:] = item.inputs_embeds[0]
        positions[..., index:index + 1, start:] = item.position_ids
        sources[index, start:] = source
        valid[index, start:] = True
    return PaddedBatch(embeddings, positions, sources, valid, lengths)


def causal_mask(query_sources, key_sources, key_valid, dtype, window=None, query_valid=None):
    """[B,1,Q,K] additive mask with per-request original-position locality."""
    if (query_sources.ndim != 2 or key_sources.ndim != 2 or key_valid.shape != key_sources.shape
            or query_sources.shape[0] != key_sources.shape[0]):
        raise ValueError('batched source/mask dimensions differ')
    if window is not None and window <= 0:
        raise ValueError('native sliding window must be positive')
    allowed = key_valid[:, None, :] & (key_sources[:, None, :] <= query_sources[:, :, None])
    if window is not None:
        allowed &= key_sources[:, None, :] > query_sources[:, :, None] - window
    if query_valid is not None:
        if query_valid.shape != query_sources.shape:
            raise ValueError('query validity dimensions differ')
        allowed &= query_valid[:, :, None]
        # Invalid padding/finished queries are never consumed. Give each one a
        # single finite key so SDPA does not encounter an entirely masked row.
        invalid = ~query_valid
        allowed[:, :, -1] |= invalid
    result = torch.zeros(allowed.shape, dtype=dtype, device=query_sources.device)
    result.masked_fill_(~allowed, torch.finfo(dtype).min)
    return result[:, None]


def cache_memory(cache):
    tensors = cache.key_cache + cache.value_cache
    storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                for t in tensors}
    return {'allocated_kv_bytes_including_padding': sum(storages.values()),
            'logical_tensor_bytes_including_padding': sum(t.numel() * t.element_size() for t in tensors),
            'physical_slots_per_request_per_layer': int(cache.get_seq_length()),
            'storage_implementation': 'transformers.DynamicCache_growing_cat',
            'padding_included': True, 'text_eviction_active': False}


@torch.no_grad()
def generate_batched(model, pruned, *, max_new_tokens=128, pad_to=None, stop_on_eos=True):
    if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
        raise ValueError('v15 batching is restricted to 1..128 generated tokens')
    if getattr(model, 'training', False):
        raise ValueError('batch measurement requires eval mode')
    if attention_backend(model) != 'sdpa':
        raise ValueError('original-position padded batches require audited SDPA masks')
    device = pruned[0].inputs_embeds.device if pruned else torch.device('cpu')
    _synchronize(device)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    batch = left_pad(pruned, pad_to=pad_to)
    count, width, _ = batch.inputs_embeds.shape
    cache, forward = DynamicCache(), decoder_forward_function(model)
    generation = getattr(model, 'generation_config', None)
    eos = getattr(generation, 'eos_token_id', None)
    eos = getattr(model.config, 'eos_token_id', None) if eos is None else eos
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    padding = getattr(generation, 'pad_token_id', None)
    padding = getattr(model.config, 'pad_token_id', None) if padding is None else padding
    padding = min(eos) if padding is None and eos else (0 if padding is None else padding)
    window = native_sliding_window(model)
    sources, valid = batch.source_positions, batch.valid
    output = forward(inputs_embeds=batch.inputs_embeds,
        attention_mask=causal_mask(sources, sources, valid, batch.inputs_embeds.dtype, window, valid),
        position_ids=batch.position_ids, cache_position=torch.arange(width, device=device),
        past_key_values=cache, use_cache=True, output_attentions=False)
    if output.past_key_values is not cache:
        raise RuntimeError('decoder replaced the supplied batch DynamicCache')
    token = output.logits[:, -1].argmax(dim=-1)
    del output
    first = token.cpu().tolist()
    tokens = [[int(value)] for value in first]
    finished = torch.tensor([stop_on_eos and value in eos for value in first], device=device)
    _synchronize(device)
    ttft = time.perf_counter() - started
    completed_at = [ttft if bool(done) else None for done in finished.cpu().tolist()]
    next_positions = batch.position_ids[..., -1:].clone() + 1
    next_sources = sources[:, -1:].clone() + 1
    valid_forward_slots = sum(batch.prefix_lengths)
    for _ in range(max_new_tokens - 1):
        if bool(finished.all()):
            break
        active = ~finished
        input_ids = torch.where(active, token, torch.full_like(token, padding))[:, None]
        valid = torch.cat((valid, active[:, None]), dim=1)
        sources = torch.cat((sources, next_sources), dim=1)
        physical = cache.get_seq_length()
        output = forward(input_ids=input_ids,
            attention_mask=causal_mask(next_sources, sources, valid, batch.inputs_embeds.dtype, window, active[:, None]),
            position_ids=next_positions, cache_position=torch.tensor([physical], device=device),
            past_key_values=cache, use_cache=True, output_attentions=False)
        if output.past_key_values is not cache:
            raise RuntimeError('decoder replaced the supplied batch DynamicCache')
        token = output.logits[:, -1].argmax(dim=-1)
        del output
        emitted = token.cpu().tolist()
        active_list = active.cpu().tolist()
        for index, value in enumerate(emitted):
            if active_list[index]:
                tokens[index].append(int(value))
        if stop_on_eos and eos:
            finished |= torch.tensor([value in eos for value in emitted], device=device)
            stopped = finished.cpu().tolist()
            step_elapsed = time.perf_counter() - started
            for index, done in enumerate(stopped):
                if done and completed_at[index] is None:
                    completed_at[index] = step_elapsed
        valid_forward_slots += sum(active_list)
        next_positions += 1
        next_sources += 1
    _synchronize(device)
    elapsed = time.perf_counter() - started
    physical_slots = count * cache.get_seq_length()
    stats = cache_memory(cache)
    metrics = {
        'batch_size': count, 'real_shared_weight_decoder_batch': True,
        'continuous_batching': False, 'model_replicas': 1,
        'timing_scope': 'padding_and_batched_prefill_to_last_active_request_token',
        'generation_seconds': elapsed, 'prefill_ttft_seconds': ttft,
        'completion_seconds_per_request': [elapsed if value is None else value for value in completed_at],
        'generated_tokens': sum(map(len, tokens)), 'generated_tokens_per_request': list(map(len, tokens)),
        'generated_tokens_per_second': sum(map(len, tokens)) / max(elapsed, 1e-12),
        'prefix_lengths': batch.prefix_lengths, 'padded_prefix_width': width,
        'unpadded_prefix_slots': sum(batch.prefix_lengths), 'padded_prefix_slots': count * width,
        'valid_forward_slots': valid_forward_slots,
        'padding_and_finished_row_waste_slots': physical_slots - valid_forward_slots,
        'padding_and_finished_row_waste_fraction': (physical_slots - valid_forward_slots) / physical_slots,
        'native_sliding_window': window, 'original_positions_preserved': True,
        'natural_eos': bool(stop_on_eos), 'attention_backend': 'sdpa',
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
        'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
        'cache': stats,
    }
    return BatchedResult(tokens, metrics, cache)


@torch.no_grad()
def validate_batch_parity(model, pruned, *, max_new_tokens=16):
    """Exact emitted-token parity; a mismatch blocks corresponding timing claims."""
    from .dynamic import generate_dynamic
    batch = generate_batched(model, pruned, max_new_tokens=max_new_tokens)
    tokens, width = batch.tokens, batch.metrics['padded_prefix_width']
    del batch
    records = []
    for index, item in enumerate(pruned):
        padded = generate_batched(model, [item], max_new_tokens=max_new_tokens, pad_to=width)
        padded_tokens = padded.tokens[0]
        del padded
        plain = generate_batched(model, [item], max_new_tokens=max_new_tokens)
        plain_tokens = plain.tokens[0]
        del plain
        native = generate_dynamic(model, item, max_new_tokens=max_new_tokens)
        native_tokens = native.ids[0].cpu().tolist()
        del native
        records.append({'row': index, 'batch_tokens': tokens[index],
            'single_same_padding_tokens': padded_tokens, 'single_unpadded_tokens': plain_tokens,
            'existing_single_dynamic_tokens': native_tokens,
            'batch_matches_single_same_padding': tokens[index] == padded_tokens,
            'padding_matches_unpadded': padded_tokens == plain_tokens,
            'unpadded_matches_existing_runtime': plain_tokens == native_tokens})
    return {'passed': all(r[k] for r in records for k in (
        'batch_matches_single_same_padding', 'padding_matches_unpadded', 'unpadded_matches_existing_runtime')),
        'batch_size': len(pruned), 'max_new_tokens': max_new_tokens, 'records': records,
        'failed_comparison_is_diagnostic_only': True, 'numeric_tolerance_cannot_override_token_mismatch': True}
