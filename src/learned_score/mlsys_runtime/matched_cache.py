"""Fixed-total-KV storage for explicitly adapted cache-selection baselines.

No image or prompt exemption is given to either competing selector. H2O scores
start at prefill, accumulate without decay, and are maintained independently
for every layer and KV head. Query heads sharing one KV head contribute their
SUM; this is an explicit GQA adaptation, not upstream H2O's broken GQA path.

Reference algorithms (not copied implementation):
https://github.com/FMInference/H2O/blob/main/h2o_hf/utils_real_drop/modify_llama.py
https://github.com/mit-han-lab/streaming-llm/blob/main/streaming_llm/kv_cache.py
"""

import torch
from transformers.cache_utils import Cache


METHODS = {"asymvlm", "sink_recent_native_positions", "streaming_llm_rolling", "h2o_gqa_native_positions"}


def attention_to_kv_mass(probabilities, num_kv_heads):
    """Sum all query positions and GQA query heads, retaining [B,Hkv,K]."""
    if probabilities.ndim != 4 or num_kv_heads < 1:
        raise ValueError("attention must be [batch, query_heads, queries, keys]")
    batch, heads, queries, keys = probabilities.shape
    if heads % num_kv_heads:
        raise ValueError("query heads must divide into complete KV groups")
    return probabilities.float().reshape(
        batch, num_kv_heads, heads // num_kv_heads, queries, keys,
    ).sum(dim=(2, 3))


def select_retained_indices(method, length, capacity_budget, prefix_length, scores=None, sink_tokens=4):
    """Select chronological slots; H2O returns independent [B,Hkv,C] indices."""
    if method not in METHODS or not 0 <= prefix_length <= capacity_budget <= length:
        raise ValueError("invalid method or prefix/budget/length relationship")
    device = scores.device if scores is not None else None
    if method == "asymvlm":
        suffix = capacity_budget - prefix_length
        return torch.cat((torch.arange(prefix_length, device=device),
                          torch.arange(length - suffix, length, device=device)))
    if method in {"sink_recent_native_positions", "streaming_llm_rolling"}:
        if sink_tokens < 0:
            raise ValueError("sink token count must be nonnegative")
        sinks = min(sink_tokens, capacity_budget)
        return torch.cat((torch.arange(sinks, device=device),
                          torch.arange(length - (capacity_budget - sinks), length, device=device)))
    if scores is None or scores.ndim != 3 or scores.shape[-1] != length:
        raise ValueError("H2O requires [batch, kv_heads, length] accumulated scores")
    heavy = capacity_budget // 2
    recent = capacity_budget - heavy
    # Stable ordering makes equal-score ties deterministic (earlier slot first).
    older = torch.argsort(scores[..., :length - recent], dim=-1, descending=True, stable=True)
    kept = older[..., :heavy].sort(dim=-1).values
    newest = torch.arange(length - recent, length, device=scores.device)
    newest = newest.expand(*scores.shape[:-1], recent)
    return torch.cat((kept, newest), dim=-1)


class MatchedKVCache(Cache):
    """All selectors have the SAME physical K/V capacity within a request.

    ``budget`` is prefix + generated_budget, INCLUDING multimodal/prompt tokens.
    ``capacity`` allows the same overflow for each method. Compaction happens
    before the next forward would exceed capacity. At slack=1 this has the same
    attention-visible C+1 sequence as post-forward single-token H2O eviction.
    Final cache may contain the last shared overflow slot; we report it, rather
    than quietly describing active C+1 storage as a C-token hard cap.
    """

    is_compileable = False
    is_sliding = False

    def __init__(self, prefix_length, generated_budget, method, slack=32,
                 prefix_source_positions=None, max_generated_tokens=None, total_retained_tokens=None):
        super().__init__()
        if prefix_length < 1 or generated_budget < 0 or slack < 1 or method not in METHODS:
            raise ValueError("invalid matched cache configuration")
        if max_generated_tokens is not None and max_generated_tokens < 0:
            raise ValueError("maximum generated tokens must be nonnegative")
        self.prefix_length = int(prefix_length)
        self.requested_generated_budget = int(generated_budget)
        self.method = method
        self.slack = int(slack)
        self.budget = (self.prefix_length + int(generated_budget) if total_retained_tokens is None
                       else int(total_retained_tokens))
        if self.budget < self.prefix_length:
            raise ValueError("total retained budget must accommodate the entire initial prefix")
        self.generated_budget = self.budget - self.prefix_length
        self.capacity = self.budget + self.slack
        if max_generated_tokens is not None:
            self.capacity = min(self.capacity, self.prefix_length + max_generated_tokens)
        if prefix_source_positions is not None and (
            prefix_source_positions.shape != (prefix_length,)
            or prefix_source_positions.dtype != torch.long
            or bool((prefix_source_positions < 0).any())
            or bool((prefix_source_positions[1:] <= prefix_source_positions[:-1]).any())
        ):
            raise ValueError("prefix source positions must be strictly increasing nonnegative longs")
        self._prefix_source = prefix_source_positions
        self._keys, self._values, self._sources, self._scores, self._lengths = [], [], [], [], []
        self._score_updates = []
        self._forward_updates = []
        self._scratch = None
        self._seen_tokens = 0
        self.current_query_sources = None
        self.current_query_position_ids = None
        self._rope_positions = None
        self.compaction_count = 0
        self.evicted_tokens = 0
        self.copied_kv_bytes = 0
        self.peak_active_tokens = 0
        self.attention_scratch_peak_bytes = 0
        self.rotary_scratch_peak_bytes = 0

    @property
    def key_cache(self):
        return [x[..., :n, :] for x, n in zip(self._keys, self._lengths)]

    @property
    def value_cache(self):
        return [x[..., :n, :] for x, n in zip(self._values, self._lengths)]

    def __len__(self):
        return len(self._keys)

    def __getitem__(self, layer_idx):
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx=0):
        return self._lengths[layer_idx] if layer_idx < len(self._lengths) else 0

    def get_max_cache_shape(self):
        return self.capacity

    def get_max_length(self):
        return self.capacity

    def get_usable_length(self, new_seq_length, layer_idx=0):
        if self.get_seq_length(layer_idx) + new_seq_length > self.capacity:
            raise RuntimeError("compact matched cache between forwards")
        return self.get_seq_length(layer_idx)

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("matched cache supports greedy, single-request inference only")

    def layer_source_positions(self, layer_idx):
        return self._sources[layer_idx][..., :self.get_seq_length(layer_idx)]

    @property
    def rope_positions(self):
        if self._rope_positions is None:
            raise RuntimeError("rolling cache requires original position IDs on each forward")
        positions = self._rope_positions[..., :self.get_seq_length()]
        if self.compaction_count:
            return torch.arange(self.get_seq_length(), device=positions.device).expand_as(positions)
        return positions

    def _check_complete(self):
        if self._lengths and any(n != self._lengths[0] for n in self._lengths):
            raise RuntimeError("all decoder layers must finish before compaction")

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.ndim != 4 or key_states.shape != value_states.shape or key_states.shape[0] != 1:
            raise ValueError("K/V must have identical [1, KV heads, tokens, head dim] shapes")
        if key_states.device != value_states.device or key_states.dtype != value_states.dtype:
            raise ValueError("K/V devices and dtypes must match")
        if not 0 <= layer_idx <= len(self._keys):
            raise ValueError("cache layers must initialize in contiguous order")
        if layer_idx == 0:
            self._check_complete()
        count = key_states.shape[-2]
        start = self.get_seq_length(layer_idx)
        end = start + count
        if end > self.capacity:
            raise RuntimeError("matched total KV capacity exceeded")
        if layer_idx == len(self._keys):
            if count != self.prefix_length:
                raise ValueError("first layer update must contain the complete prefix")
            if self._keys and (key_states.shape[:2] + key_states.shape[-1:] !=
                               self._keys[0].shape[:2] + self._keys[0].shape[-1:]):
                raise ValueError("all layers must have the same KV layout")
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self._keys.append(key_states.new_empty(shape))
            self._values.append(value_states.new_empty(shape))
            self._sources.append(torch.empty((*key_states.shape[:2], self.capacity), dtype=torch.long,
                                             device=key_states.device))
            self._scores.append(torch.zeros((*key_states.shape[:2], self.capacity), dtype=torch.float32,
                                            device=key_states.device) if self.method.startswith("h2o") else None)
            self._lengths.append(0)
            self._score_updates.append(0)
            self._forward_updates.append(0)
        elif (key_states.shape[:2] + key_states.shape[-1:] !=
              self._keys[layer_idx].shape[:2] + self._keys[layer_idx].shape[-1:]
              or key_states.dtype != self._keys[layer_idx].dtype
              or key_states.device != self._keys[layer_idx].device):
            raise ValueError("KV layout/dtype/device changed within a request")
        if layer_idx and end != self._lengths[0]:
            raise RuntimeError("all layers must append the same number of tokens")
        source = self.current_query_sources
        if source is None:
            if start == 0:
                source = self._prefix_source
                if source is None:
                    source = torch.arange(count, device=key_states.device)
            else:
                source = self._sources[layer_idx][0, 0, start - 1] + torch.arange(
                    1, count + 1, device=key_states.device)
        if source.shape != (count,) or source.dtype != torch.long:
            raise ValueError("current query sources must match appended token count")
        self._keys[layer_idx][..., start:end, :].copy_(key_states)
        self._values[layer_idx][..., start:end, :].copy_(value_states)
        self._sources[layer_idx][..., start:end].copy_(source.to(key_states.device))
        if self._scores[layer_idx] is not None:
            self._scores[layer_idx][..., start:end].zero_()
        self._lengths[layer_idx] = end
        self._forward_updates[layer_idx] += 1
        if layer_idx == 0:
            if self.method == "streaming_llm_rolling":
                positions = self.current_query_position_ids
                if positions is None or positions.shape[-1] != count:
                    raise ValueError("rolling cache must receive model-native query position IDs")
                if self._rope_positions is None:
                    self._rope_positions = torch.empty((*positions.shape[:-1], self.capacity),
                                                       device=positions.device, dtype=positions.dtype)
                self._rope_positions[..., start:end].copy_(positions)
            self._seen_tokens += count
            self.peak_active_tokens = max(self.peak_active_tokens, end)
        return self._keys[layer_idx][..., :end, :], self._values[layer_idx][..., :end, :]

    @torch.no_grad()
    def add_attention_mass(self, layer_idx, mass):
        if self.method != "h2o_gqa_native_positions":
            raise ValueError("only H2O records attention masses")
        target = self._scores[layer_idx][..., :self.get_seq_length(layer_idx)]
        if mass.shape != target.shape or mass.device != target.device:
            raise ValueError("attention mass must match active per-layer KV heads")
        if self._score_updates[layer_idx] + 1 != self._forward_updates[layer_idx]:
            raise RuntimeError("H2O must record exactly one full attention mass per layer forward")
        target.add_(mass.float())
        self._score_updates[layer_idx] += 1

    @torch.no_grad()
    def compact_if_needed(self, incoming_tokens=1):
        if incoming_tokens < 1:
            raise ValueError("incoming token count must be positive")
        self._check_complete()
        length = self.get_seq_length()
        if length + incoming_tokens <= self.capacity:
            return None
        if self.budget + incoming_tokens > self.capacity:
            raise ValueError("incoming tokens exceed shared overflow allowance")
        if self.method.startswith("h2o") and self._score_updates != self._forward_updates:
            raise RuntimeError("H2O must record full-prefill and every-decode attention before eviction")
        if self._scratch is None:
            first = self._keys[0]
            self._scratch = first.new_empty((*first.shape[:-2], self.budget, first.shape[-1]))
        copied = 0
        for layer_idx, (keys, values) in enumerate(zip(self._keys, self._values)):
            scores = self._scores[layer_idx]
            indices = select_retained_indices(
                self.method, length, self.budget, self.prefix_length,
                scores=scores[..., :length] if scores is not None else None,
            ).to(keys.device)
            if indices.ndim == 1:
                indices = indices.expand(*keys.shape[:2], self.budget)
            gather = indices[..., None].expand(*indices.shape, keys.shape[-1])
            for tensor in (keys, values):
                torch.gather(tensor, -2, gather, out=self._scratch)
                tensor[..., :self.budget, :].copy_(self._scratch)
                copied += self._scratch.numel() * self._scratch.element_size() * 2
            self._sources[layer_idx][..., :self.budget].copy_(
                self._sources[layer_idx].gather(-1, indices))
            if scores is not None:
                scores[..., :self.budget].copy_(scores.gather(-1, indices))
            self._lengths[layer_idx] = self.budget
        removed = length - self.budget
        if self.method == "streaming_llm_rolling":
            common_indices = indices[0, 0]
            self._rope_positions[..., :self.budget].copy_(
                self._rope_positions.index_select(-1, common_indices))
        self.compaction_count += 1
        self.evicted_tokens += removed
        self.copied_kv_bytes += copied
        return {"removed_tokens": removed, "active_cache_tokens": self.budget}

    def memory_stats(self):
        def nbytes(tensor):
            return tensor.numel() * tensor.element_size() if tensor is not None else 0
        capacity_bytes = sum(nbytes(x) for x in self._keys + self._values)
        bytes_per_slot = capacity_bytes // self.capacity if self.capacity else 0
        sources = sum(nbytes(x) for x in self._sources) + nbytes(self._rope_positions)
        scores = sum(nbytes(x) for x in self._scores)
        scratch = nbytes(self._scratch)
        return {
            "prefix_tokens": self.prefix_length,
            "generated_budget": self.generated_budget,
            "requested_generated_budget": self.requested_generated_budget,
            "total_steady_budget_tokens_per_layer": self.budget,
            "total_steady_budget_kv_bytes": bytes_per_slot * self.budget,
            "overflow_allowance": self.slack,
            "capacity_tokens_per_layer": self.capacity,
            "active_tokens": self.get_seq_length(),
            "logical_tokens_seen": self._seen_tokens,
            "kv_capacity_bytes": capacity_bytes,
            "kv_active_bytes": bytes_per_slot * self.get_seq_length(),
            "peak_active_kv_bytes": bytes_per_slot * self.peak_active_tokens,
            "peak_active_tokens": self.peak_active_tokens,
            "source_position_bytes": sources,
            "score_state_bytes": scores,
            "scratch_bytes": scratch,
            "auxiliary_persistent_bytes": sources + scores + scratch,
            "attention_statistics_scratch_peak_bytes_estimate": self.attention_scratch_peak_bytes,
            "rolling_rotary_scratch_peak_bytes_estimate": self.rotary_scratch_peak_bytes,
            "allocated_cache_and_auxiliary_bytes": capacity_bytes + sources + scores + scratch,
            "compaction_events": self.compaction_count,
            "evicted_tokens": self.evicted_tokens,
            "copied_kv_bytes": self.copied_kv_bytes,
            "attention_score_updates_per_layer": list(self._score_updates),
            "kv_budget_includes_entire_prefix": True,
            "protects_entire_prefix": self.method == "asymvlm",
        }
