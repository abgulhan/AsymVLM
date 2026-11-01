"""Native FastV / VL-Cache / VisionZip ports for LLaVA-NeXT (frozen benchmark version).

Adapted from the code used to produce the paper results. Cluster scheduling
and campaign bookkeeping were removed; the numerical code is unchanged.
"""
from contextlib import contextmanager

from dataclasses import replace

import math

import time

import torch

import torch.nn.functional as F

from transformers.cache_utils import DynamicCache

from transformers.models.mistral.modeling_mistral import apply_rotary_pos_emb, repeat_kv

from learned_score.mlsys_runtime.generation import decoder_forward_function, _synchronize

METHODS = ('fastv_native', 'vlcache_native', 'visionzip_llava_native')

def layer_counts(density, total, length):
    """Bounded proportional allocation with deterministic exact integer sum."""
    values = torch.as_tensor(density, dtype=torch.float64, device='cpu')
    lower = max(1, math.ceil(length * .01))
    if (not torch.isfinite(values).all() or (values < 0).any()
            or not len(values)*lower <= total <= len(values)*length):
        raise ValueError('invalid density or infeasible total cache budget')
    values = values.clamp_min(1e-12)
    lo, hi = 0., length / float(values.min())
    for _ in range(100):
        mid = (lo+hi)/2
        if (values*mid).clamp(lower, length).sum() < total:
            lo = mid
        else:
            hi = mid
    raw = (values*((lo+hi)/2)).clamp(lower, length)
    counts = raw.floor().long()
    left = total-int(counts.sum())
    order = torch.argsort(raw-counts, descending=True, stable=True)
    for i in order.tolist():
        if left and counts[i] < length:
            counts[i] += 1
            left -= 1
    if left or int(counts.sum()) != total:
        raise AssertionError('cache budget reconciliation failed')
    return counts.tolist()


def top_with_recent(scores, count):
    """Independent KV-head selections, with paper's 10% recent window."""
    length = scores.shape[-1]
    if not 1 <= count <= length:
        raise ValueError('invalid count')
    recent = max(1, count//10)
    old = torch.argsort(scores[..., :length-recent], descending=True, stable=True)[..., :count-recent]
    tail = torch.arange(length-recent, length, device=scores.device).expand(*scores.shape[:-1], recent)
    return torch.cat((old, tail), -1).sort(-1).values


def postvision_stats(attn, kwargs, query_positions):
    """Recompute only needed attention rows; native attention remains SDPA."""
    hidden = kwargs['hidden_states']
    batch, length, _ = hidden.shape
    if batch != 1:
        raise ValueError('baseline runner supports one unpadded request')
    q = attn.q_proj(hidden[:, query_positions]).view(batch, -1, attn.config.num_attention_heads, attn.head_dim).transpose(1, 2)
    k = attn.k_proj(hidden).view(batch, length, attn.config.num_key_value_heads, attn.head_dim).transpose(1, 2)
    cos, sin = kwargs['position_embeddings']
    # Query and key have different sequence lengths, so rotate separately.
    q, _ = apply_rotary_pos_emb(q, q, cos[:, query_positions], sin[:, query_positions])
    k, _ = apply_rotary_pos_emb(k, k, cos, sin)
    k = repeat_kv(k, attn.num_key_value_groups)
    # Small chunks prevent long text prompts from materializing a large matrix.
    sums = torch.zeros((1, q.shape[1], length), device=q.device, dtype=torch.float32)
    nonzero = torch.zeros((), device=q.device, dtype=torch.float64)
    denominator = 0
    columns = torch.arange(length, device=q.device)
    mask = kwargs.get('attention_mask')
    for start in range(0, q.shape[-2], 16):
        positions = query_positions[start:start+16]
        logits = (q[:, :, start:start+16].float() @ k.float().transpose(-1, -2)) * attn.scaling
        allowed = columns[None, :] <= positions[:, None]
        window = getattr(attn.config, 'sliding_window', None)
        if window:
            allowed &= columns[None, :] > positions[:, None]-window
        logits.masked_fill_(~allowed[None, None], torch.finfo(logits.dtype).min)
        if mask is not None and mask.ndim == 4:
            logits += mask[:, :, positions, :length].float()
        probs = logits.softmax(-1)
        sums += probs.sum(-2)
        nonzero += ((probs >= .01*probs.amax(-1, keepdim=True)) & allowed[None, None]).sum()
        denominator += int(allowed.sum())*q.shape[1]
    density = float(nonzero / max(1, denominator))
    # GQA heads sharing a KV head also share its selected cache indices.
    scores = sums.view(1, attn.config.num_key_value_heads, attn.num_key_value_groups, length).mean(2)
    return scores, density


class NativePruning:
    def __init__(self, model, prepared, method, percent):
        self.model, self.prepared, self.method = model, prepared, method
        self.decoder = model.language_model.model if hasattr(model, 'language_model') else model.model
        if self.decoder.config.model_type != 'mistral':
            raise ValueError('only validated native Mistral/LLaVA-NeXT is supported')
        self.layers = self.decoder.layers
        self.length = prepared.inputs_embeds.shape[1]
        self.visual = prepared.visual_positions
        self.percent = percent
        self.nominal_visual = (len(self.visual)*percent+99)//100
        self.target_slots = len(self.layers)*(self.length-len(self.visual)+self.nominal_visual)
        # Match aggregate KV budget, accounting for FastV's two unpruned layers.
        self.fastv_count = min(len(self.visual), max(1, math.ceil(
            (len(self.layers)*self.nominal_visual-2*len(self.visual))/(len(self.layers)-2))))
        self.keep = torch.arange(self.length, device=self.visual.device)
        self.scores, self.densities, self.sources, self.handles = {}, {}, {}, []
        self.prefill = True
        self.prefix_lengths = []

    def collect(self, index, module, args, kwargs):
        if not self.prefill or self.percent == 100:
            return
        if self.method == 'fastv_native':
            query = torch.tensor([self.length-1], device=self.visual.device)
        else:
            last_image = int(self.visual.max())
            protected = self.prepared.metadata.get('protected_visual_positions')
            if protected is not None and len(protected):
                last_image = max(last_image, int(protected.max()))
            query = torch.arange(last_image+1, self.length, device=self.visual.device)
            if not len(query):
                raise ValueError('VL-Cache requires post-vision prompt text')
        scores, density = postvision_stats(module, kwargs, query)
        if self.method == 'fastv_native':
            order = torch.argsort(scores.mean((0, 1))[self.visual], descending=True, stable=True)
            mask = torch.ones(self.length, dtype=torch.bool, device=self.visual.device)
            mask[self.visual] = False
            mask[self.visual[order[:self.fastv_count]]] = True
            self.keep = mask.nonzero().flatten()
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
                kwargs['position_embeddings'] = tuple(x[:, self.keep] for x in kwargs['position_embeddings'])
            mask = kwargs.get('attention_mask')
            if mask is not None:
                kwargs['attention_mask'] = mask[..., self.keep, :][..., self.keep]
        elif not self.prefill:
            # Explicit source-position masks are necessary after physical compaction.
            sources = self.sources[index]
            position = kwargs['position_ids'][..., -1].reshape(-1)[0]
            now = torch.full((*sources.shape[:-1], 1), int(position), device=sources.device, dtype=torch.long)
            keys = torch.cat((sources, now), -1)
            self.sources[index] = keys
            allowed = keys <= position
            window = getattr(self.decoder.config, 'sliding_window', None)
            if window:
                allowed &= keys > position-window
            h = (args[0] if args else kwargs['hidden_states'])
            mask = torch.zeros((*keys.shape[:-1], 1, keys.shape[-1]), device=h.device, dtype=h.dtype)
            mask.masked_fill_(~allowed.unsqueeze(-2), torch.finfo(h.dtype).min)
            # Independent KV heads require matching query-head masks under GQA.
            if mask.shape[1] != 1:
                mask = mask.repeat_interleave(module.self_attn.num_key_value_groups, dim=1)
            kwargs['attention_mask'] = mask
        return args, kwargs

    def __enter__(self):
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.register_forward_pre_hook(
                lambda m, a, kw, i=index: self.layer_hook(i, m, a, kw), with_kwargs=True))
            if self.method == 'vlcache_native' or (self.method == 'fastv_native' and index == 1):
                self.handles.append(layer.self_attn.register_forward_pre_hook(
                    lambda m, a, kw, i=index: self.collect(i, m, a, kw), with_kwargs=True))
        return self

    def compact(self, cache):
        device = self.visual.device
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
                source = self.keep if self.method == 'fastv_native' and index >= 2 else torch.arange(self.length, device=device)
                self.sources[index] = source[None, None]
        self.prefix_lengths = counts
        self.prefill = False
        self.scores.clear()
        if self.method == 'vlcache_native' and sum(counts) != self.target_slots:
            raise AssertionError('VL-Cache physical allocation mismatch')

    def __exit__(self, *_):
        for handle in self.handles:
            handle.remove()


@torch.no_grad()
def generate_native(model, prepared, method, percent, max_new_tokens=128, stop_on_eos=True):
    if method not in ('all_tokens', 'fastv_native', 'vlcache_native') or not 1 <= percent <= 100:
        raise ValueError('unsupported method or retention')
    if prepared.inputs_embeds.shape[0] != 1 or not prepared.attention_mask.bool().all():
        raise ValueError('one unpadded request required')
    device = prepared.inputs_embeds.device
    length = prepared.inputs_embeds.shape[1]
    cache = DynamicCache()
    forward = decoder_forward_function(model)
    eos = getattr(getattr(model, 'generation_config', None), 'eos_token_id', None)
    eos = {eos} if isinstance(eos, int) else set(eos or [])
    _synchronize(device)
    started = time.perf_counter()
    with NativePruning(model, prepared, method, 100 if method == 'all_tokens' else percent) as policy:
        output = forward(inputs_embeds=prepared.inputs_embeds, attention_mask=prepared.attention_mask,
            position_ids=torch.arange(length, device=device)[None], cache_position=torch.arange(length, device=device),
            past_key_values=cache, use_cache=True, output_attentions=False)
        token = output.logits[:, -1].argmax(-1, keepdim=True)
        generated = [token]
        policy.compact(cache)
        prefix_bytes = sum(x.numel()*x.element_size() for seq in (cache.key_cache, cache.value_cache) for x in seq)
        _synchronize(device)
        ttft = time.perf_counter()-started
        for step in range(max_new_tokens-1):
            if stop_on_eos and int(token.item()) in eos:
                break
            position = length+step
            output = forward(input_ids=token, attention_mask=None,
                position_ids=torch.tensor([[position]], device=device), cache_position=torch.tensor([position], device=device),
                past_key_values=cache, use_cache=True, output_attentions=False)
            token = output.logits[:, -1].argmax(-1, keepdim=True)
            generated.append(token)
        _synchronize(device)
        elapsed = time.perf_counter()-started
        current_bytes = sum(x.numel()*x.element_size() for seq in (cache.key_cache, cache.value_cache) for x in seq)
        return torch.cat(generated, 1), dict(prefill_ttft_seconds=ttft, total_seconds=elapsed,
            decode_seconds=elapsed-ttft, prefix_kv_bytes=prefix_bytes, final_kv_bytes=current_bytes,
            prefix_tokens_per_layer=policy.prefix_lengths, target_prefix_slots=policy.target_slots,
            actual_prefix_slots=sum(policy.prefix_lengths), input_prefix_tokens=length,
            fastv_pruning_layer=2 if method == 'fastv_native' else None,
            fastv_retained_visual_tokens=policy.fastv_count if method == 'fastv_native' else None,
            layer_density=policy.densities, physical_cache_compaction=True)


@contextmanager
def visionzip_capture(model):
    """Capture one CLIP layer's raw features/keys and CLS attention in one pass."""
    if model.config.vision_feature_layer != -2 or model.config.vision_feature_select_strategy != 'default':
        raise ValueError('VisionZip requires validated CLIP penultimate/default features')
    result = {}
    layer = model.vision_tower.vision_model.encoder.layers[-2]
    def attention_hook(module, args, kwargs):
        hidden = kwargs.get('hidden_states', args[0] if args else None)
        b, n, _ = hidden.shape
        q = module.q_proj(hidden[:, :1]).view(b, 1, module.num_heads, module.head_dim).transpose(1, 2)
        k = module.k_proj(hidden).view(b, n, module.num_heads, module.head_dim).transpose(1, 2)
        result['attention'] = ((q.float() @ k.float().transpose(-1, -2))*module.scale).softmax(-1).mean(1)[:, 0, 1:]
        result['keys'] = k.mean(1)[:, 1:].detach()
    def feature_hook(module, args, output):
        result['features'] = output[0][:, 1:].detach()
    handles = [layer.self_attn.register_forward_pre_hook(attention_hook, with_kwargs=True),
               layer.register_forward_hook(feature_hook)]
    try:
        yield result
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def visionzip_prune(adapter, prepared, captured, count):
    """Global packed-image adaptation: pre-projector contextual token merging."""
    from learned_score.qwen2_pruning import visionzip_selection
    model = adapter.model
    n = len(prepared.visual_positions)
    if count == n:
        return adapter.physically_prune(prepared, torch.arange(n, device=prepared.inputs_embeds.device), 'original'), {}
    if not 1 <= count < n or len(prepared.native_inputs['image_sizes']) != 1:
        raise ValueError('single-image exact count required')
    packed = {}
    for key, tensor in captured.items():
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        value, _ = model.pack_image_features([tensor], prepared.native_inputs['image_sizes'],
            'default', image_newline=torch.zeros(tensor.shape[-1], device=tensor.device, dtype=tensor.dtype))
        image_mask = prepared.native_inputs['input_ids'][0].eq(model.config.image_token_index)
        image_sources = image_mask.nonzero().flatten()
        local = torch.searchsorted(image_sources, prepared.visual_positions)
        if not torch.equal(image_sources[local], prepared.visual_positions):
            raise ValueError('AnyRes source mapping mismatch')
        packed[key] = value[local]
    if packed['features'].shape[0] != n:
        raise ValueError('AnyRes statistics count mismatch')
    projected = model.multi_modal_projector(packed['features'])
    original = prepared.inputs_embeds[0, prepared.visual_positions]
    if not torch.allclose(projected, original, atol=.02, rtol=.02):
        raise ValueError('CLIP feature ordering does not match native AnyRes preparation')
    # Selection utility is representation independent; here the features are CLIP,
    # not projected decoder embeddings. Its merge is seed + mean assigned tokens.
    indices, merged, detail = visionzip_selection(packed['features'], packed['attention'].flatten(), packed['keys'], count)
    updated = prepared.inputs_embeds.clone()
    updated[0, prepared.visual_positions[indices]] = model.multi_modal_projector(merged).to(updated.dtype)
    changed = replace(prepared, inputs_embeds=updated)
    pruned = adapter.physically_prune(changed, indices, 'original')
    return pruned, dict(**detail, selected_indices=indices.cpu().tolist(), merge_space='pre_projector_clip',
        adaptation='global native AnyRes packed image, protected newlines, original positions')


