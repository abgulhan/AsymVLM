"""Online, fixed-priority prefix-protected caches; frozen v3 is unchanged.

All selectors see only currently retained tokens plus newly appended tokens.
Priority is a deterministic function of the ORIGINAL generated-token index,
never recomputed from a compacted slot. Lower unsigned 64-bit priority wins.
One chronological source/selection vector is shared by every layer/KV head.
The physical payload ceiling remains M=original_full_prefix+B+1, with staging
included inside M. Priority lookup/index/scratch storage is reported separately.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import random

import torch

from .matched_cache import MatchedKVCache


RUNTIME_REVISION = "matched_fixed_priority_v4"
UINT64_MASK = (1 << 64) - 1
WEYL_INCREMENT = 0x9E3779B97F4A7C15
MAX_GENERATED_POSITIONS = 4096
RECENT_MODES = ("eight", "two_thirds")


@dataclass(frozen=True)
class PolicySpec:
    name: str
    protects_prefix: bool = True
    scoring: str = "fixed_priority"
    default_staging: int = 128
    selection_order: str = "chronological_shared_across_layers_and_heads"


POLICY_REGISTRY = {
    name: PolicySpec(name, scoring="recency" if name == "protected_recency" else "fixed_priority")
    for name in ("protected_recency", "protected_grid", "protected_weyl", "protected_priority_random")
}
POLICIES = tuple(POLICY_REGISTRY)


def _integer(value, name, lower=0, upper=None):
    if isinstance(value, bool) or not isinstance(value, int) or value < lower or (upper is not None and value > upper):
        raise ValueError(f"invalid {name}")
    return value


def reverse_bits64(value):
    value = _integer(value, "unsigned 64-bit index", upper=UINT64_MASK)
    result = 0
    for _ in range(64):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


@lru_cache(maxsize=32)
def priority_values(policy, max_positions=MAX_GENERATED_POSITIONS, seed=42, phase=0):
    """CPU-only exact unsigned priorities; no global RNG state is touched."""
    if policy not in POLICIES:
        raise ValueError("unknown fixed-priority policy")
    _integer(max_positions, "maximum generated positions", 1, MAX_GENERATED_POSITIONS)
    _integer(seed, "seed", upper=UINT64_MASK)
    _integer(phase, "phase", upper=UINT64_MASK)
    if policy in {"protected_recency", "protected_priority_random"} and phase:
        raise ValueError("phase applies only to grid and Weyl priorities")
    if policy == "protected_grid":
        return tuple(reverse_bits64(j ^ phase) for j in range(max_positions))
    if policy == "protected_weyl":
        return tuple(((j + 1) * WEYL_INCREMENT + phase) & UINT64_MASK for j in range(max_positions))
    if policy == "protected_priority_random":
        generator = random.Random(seed)
        return tuple(generator.getrandbits(64) for _ in range(max_positions))
    return tuple(range(max_positions))  # Recency does not use this lookup.


@lru_cache(maxsize=32)
def priority_ranks(policy, max_positions=MAX_GENERATED_POSITIONS, seed=42, phase=0):
    """Unique small signed ranks preserve (unsigned_priority, original_index)."""
    values = priority_values(policy, max_positions, seed, phase)
    order = sorted(range(len(values)), key=lambda j: (values[j], j))
    ranks = [0] * len(order)
    for rank, j in enumerate(order):
        ranks[j] = rank
    return tuple(ranks)


def policy_configuration(policy, staging=None, *, recent_mode="eight", phase=0, seed=42, storage="compact"):
    if policy not in POLICY_REGISTRY:
        raise ValueError("unknown v4 cache policy")
    if recent_mode not in RECENT_MODES:
        raise ValueError("recent_mode must be eight or two_thirds")
    if storage != "compact":
        raise NotImplementedError("only validated chronological compact storage is available")
    spec = POLICY_REGISTRY[policy]
    staging = spec.default_staging if staging is None else staging
    _integer(staging, "staging", 1)
    priority_values(policy, 1, seed, phase)  # Validate without materializing device state.
    return {**asdict(spec), "staging": staging, "recent_mode": recent_mode, "phase": phase,
            "seed": seed, "storage": storage, "priority_direction": "lower_unsigned64_is_better",
            "priority_tie_break": "earlier_original_generated_index",
            "random_priority_generator": "python_random_MT19937_getrandbits64" if policy == "protected_priority_random" else None,
            "online_selection": True, "evicted_tokens_never_resurrect": True}


def select_online_indices(sources, prefix_length, full_prefix_tokens, retained, recent, ranks=None, *, optimized=True):
    """Select from the live chronological cache, returning chronological slots.

    Prefix entries occupy its first prefix_length slots because ALL policies
    protect them and every previous selection was chronological. Rank lookup
    must be indexed by immutable original generated IDs, not cache slots.
    """
    if sources.ndim != 1 or sources.dtype != torch.long:
        raise ValueError("shared cache sources must be one-dimensional int64")
    length = sources.numel()
    if not 0 <= prefix_length <= retained <= length or not 0 <= recent <= retained - prefix_length:
        raise ValueError("invalid prefix, retained budget or recent allocation")
    prefix = torch.arange(prefix_length, device=sources.device)
    newest = torch.arange(length - recent, length, device=sources.device)
    anchors = retained - prefix_length - recent
    if anchors == 0:
        return torch.cat((prefix, newest))
    if ranks is None or ranks.ndim != 1 or ranks.dtype != torch.long:
        raise ValueError("nonrecent selection requires original-index priority ranks")
    older_slots = torch.arange(prefix_length, length - recent, device=sources.device)
    original_indices = sources[prefix_length:length - recent] - full_prefix_tokens
    priorities = ranks[original_indices]
    if optimized:
        selected = priorities.topk(anchors, largest=False, sorted=False).indices
    else:
        selected = priorities.argsort(stable=True)[:anchors]
    # All ranks are unique, so topk and stable reference sort have exact ties.
    selected = older_slots[selected].sort().values
    return torch.cat((prefix, selected, newest))


class MatchedKVCacheV4(MatchedKVCache):
    """Shared-mask fixed-priority selection with bounded physical K/V arrays."""

    def __init__(self, prefix_length, full_prefix_tokens, budget_profile, policy, *, staging=None,
                 prefix_source_positions=None, seed=42, optimized=True, trace=False,
                 visual_source_positions=None, recent_mode="eight", phase=0, storage="compact",
                 max_generated_positions=MAX_GENERATED_POSITIONS, recent_window=None):
        if recent_window is not None:
            raise ValueError("v4 recent allocation is specified by recent_mode, not an unregistered override")
        if isinstance(budget_profile, str) and budget_profile.startswith("base"):
            budget_profile = int(budget_profile[4:])
        _integer(prefix_length, "retained prefix", 1)
        _integer(full_prefix_tokens, "original prefix", prefix_length)
        _integer(budget_profile, "budget profile", 1)
        _integer(max_generated_positions, "maximum generated positions", 1, MAX_GENERATED_POSITIONS)
        self.configuration = policy_configuration(policy, staging, recent_mode=recent_mode, phase=phase,
                                                  seed=seed, storage=storage)
        self.spec, self.policy = POLICY_REGISTRY[policy], policy
        self.profile, self.full_prefix_tokens = budget_profile, full_prefix_tokens
        retained = full_prefix_tokens + budget_profile + 1 - self.configuration["staging"]
        if retained < prefix_length:
            raise ValueError("staging leaves insufficient capacity for the protected prefix")
        base_text = budget_profile + 1 - self.configuration["staging"]
        available_text = retained - prefix_length
        recent = available_text if policy == "protected_recency" else (8 if recent_mode == "eight" else 2 * base_text // 3)
        if recent < 0 or recent > available_text:
            raise ValueError("recent allocation does not fit inside the retained text budget")
        super().__init__(prefix_length, available_text, "asymvlm", slack=self.configuration["staging"],
                         prefix_source_positions=prefix_source_positions, total_retained_tokens=retained)
        self.recent, self.seed, self.phase = recent, seed, phase
        self.optimized, self.trace_enabled = bool(optimized), bool(trace)
        self.visual_source_positions = visual_source_positions
        self.max_generated_positions = max_generated_positions
        self._priority_tuple = (None if policy == "protected_recency" else
                               priority_ranks(policy, max_generated_positions, seed, phase))
        self._priorities = None
        self._shared_sources = None
        self._cached_recency_indices = {}
        self._source_scratch = None
        self._last_selection = None
        self.selection_scratch_peak_bytes = 0
        self.configuration.update({"recent_tokens": recent, "base_text_slots_after_compaction": base_text,
                                   "effective_text_slots_after_compaction": available_text,
                                   "maximum_generated_positions": max_generated_positions})

    @property
    def independently_selected_heads(self):
        return False

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if (key_states.ndim != 4 or key_states.shape != value_states.shape or key_states.shape[0] != 1
                or key_states.dtype != value_states.dtype or key_states.device != value_states.device):
            raise ValueError("K/V require matching [1,KV_heads,tokens,dim] tensors")
        if not 0 <= layer_idx <= len(self._keys):
            raise ValueError("decoder layers must initialize in contiguous order")
        if layer_idx == 0:
            self._check_complete()
        count, start = key_states.shape[-2], self.get_seq_length(layer_idx)
        end = start + count
        if end > self.capacity:
            raise RuntimeError("v4 physical KV ceiling exceeded; compact before the next forward")
        if layer_idx == len(self._keys):
            if count != self.prefix_length:
                raise ValueError("first update requires the complete protected prefix")
            if self._keys and (key_states.shape[:2] + key_states.shape[-1:] !=
                               self._keys[0].shape[:2] + self._keys[0].shape[-1:]):
                raise ValueError("KV layout changed between layers")
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self._keys.append(key_states.new_empty(shape))
            self._values.append(value_states.new_empty(shape))
            if self._shared_sources is None:
                self._shared_sources = torch.empty((1, 1, self.capacity), device=key_states.device, dtype=torch.long)
                if self._prefix_source is None:
                    self._prefix_source = torch.arange(self.prefix_length, device=key_states.device)
                if self._priority_tuple is not None:
                    self._priorities = torch.tensor(self._priority_tuple, device=key_states.device, dtype=torch.long)
            self._sources.append(self._shared_sources)  # Aliases; counted once in memory_stats.
            self._scores.append(None)
            self._lengths.append(0)
            self._score_updates.append(0)
            self._forward_updates.append(0)
        elif (key_states.shape[:2] + key_states.shape[-1:] != self._keys[layer_idx].shape[:2] + self._keys[layer_idx].shape[-1:]
              or key_states.device != self._keys[layer_idx].device or key_states.dtype != self._keys[layer_idx].dtype):
            raise ValueError("KV layout/dtype/device changed within a request")
        if layer_idx and end != self._lengths[0]:
            raise RuntimeError("all layers must append the same token count")
        sources = self.current_query_sources
        if sources is None:
            sources = self._prefix_source if not self._seen_tokens else torch.arange(
                self.full_prefix_tokens + self._seen_tokens - self.prefix_length,
                self.full_prefix_tokens + self._seen_tokens - self.prefix_length + count,
                device=key_states.device)
        if sources.shape != (count,) or sources.dtype != torch.long:
            raise ValueError("current original source IDs must match appended token count")
        if layer_idx == 0:
            if self._seen_tokens and count != 1:
                raise ValueError("v4 decode supports one token per forward")
            if self._seen_tokens + count - self.prefix_length > self.max_generated_positions:
                raise ValueError("generation exceeds the predeclared priority horizon")
            self._shared_sources[..., start:end].copy_(sources.to(key_states.device))
            self._seen_tokens += count
            self.peak_active_tokens = max(self.peak_active_tokens, end)
        self._keys[layer_idx][..., start:end, :].copy_(key_states)
        self._values[layer_idx][..., start:end, :].copy_(value_states)
        self._lengths[layer_idx] = end
        self._forward_updates[layer_idx] += 1
        return self._keys[layer_idx][..., :end, :], self._values[layer_idx][..., :end, :]

    def _select(self, length):
        if self.policy == "protected_recency" and self.optimized and length in self._cached_recency_indices:
            return self._cached_recency_indices[length]
        indices = select_online_indices(self._shared_sources[0, 0, :length], self.prefix_length,
                                        self.full_prefix_tokens, self.budget, self.recent, self._priorities,
                                        optimized=self.optimized)
        if self.policy == "protected_recency" and self.optimized:
            self._cached_recency_indices[length] = indices
        return indices

    @torch.no_grad()
    def compact_if_needed(self, incoming_tokens=1):
        _integer(incoming_tokens, "incoming tokens", 1)
        self._check_complete()
        length = self.get_seq_length()
        if length + incoming_tokens <= self.capacity:
            return None
        if self.budget + incoming_tokens > self.capacity:
            raise ValueError("incoming tokens exceed staging allowance")
        indices = self._select(length)
        suffix_only = self.optimized
        selected = indices[self.prefix_length:] if suffix_only else indices
        target_start = self.prefix_length if suffix_only else 0
        if self._scratch is None:
            first = self._keys[0]
            self._scratch = first.new_empty((*first.shape[:-2], selected.numel(), first.shape[-1]))
            self._source_scratch = torch.empty(self.budget, device=indices.device, dtype=torch.long)
        copied = 0
        gather = selected[None, None, :, None].expand(*self._keys[0].shape[:2], selected.numel(), self._keys[0].shape[-1])
        for layer, (keys, values) in enumerate(zip(self._keys, self._values)):
            for tensor in (keys, values):
                torch.gather(tensor, -2, gather, out=self._scratch)
                tensor[..., target_start:self.budget, :].copy_(self._scratch)
                copied += self._scratch.numel() * self._scratch.element_size() * 2
            self._lengths[layer] = self.budget
        torch.gather(self._shared_sources[0, 0, :length], 0, indices, out=self._source_scratch)
        self._shared_sources[0, 0, :self.budget].copy_(self._source_scratch)
        self._last_selection = indices
        # Conservative explicit-index tensor bound, not a claim to measure
        # topk/sort's implementation-dependent internal workspace. The runner
        # separately measures actual overall allocated/reserved GPU peaks.
        self.selection_scratch_peak_bytes = max(self.selection_scratch_peak_bytes,
                                                (10 * length + 4 * self.budget) * 8)
        removed = length - self.budget
        self.compaction_count += 1
        self.evicted_tokens += removed
        self.copied_kv_bytes += copied
        return {"removed_tokens": removed, "active_cache_tokens": self.budget,
                "copied_kv_bytes": copied, "staging_tokens": self.slack}

    def trace_snapshot(self):
        source = self._shared_sources[0, 0, :self.get_seq_length()].detach().cpu()
        prefix = torch.isin(source, self._prefix_source.detach().cpu())
        generated = source >= self.full_prefix_tokens
        layers = []
        for layer, keys in enumerate(self._keys):
            heads = keys.shape[1]
            row = {"layer": layer, "source_ids": [source.tolist() for _ in range(heads)],
                   "retained_prefix_per_head": [int(prefix.sum())] * heads,
                   "retained_generated_per_head": [int(generated.sum())] * heads}
            if self.visual_source_positions is not None:
                image = torch.isin(source, torch.as_tensor(self.visual_source_positions).detach().cpu())
                row.update({"retained_image_per_head": [int(image.sum())] * heads,
                            "retained_prompt_per_head": [int((prefix & ~image).sum())] * heads})
            layers.append(row)
        return {"policy": self.policy, "compaction_count": self.compaction_count,
                "logical_tokens_seen": self._seen_tokens, "active_tokens": self.get_seq_length(),
                "visual_identity_available": self.visual_source_positions is not None,
                "selection_shared_across_layers_and_heads": True, "layers": layers}

    def memory_stats(self):
        stats = super().memory_stats()
        nbytes = lambda x: 0 if x is None else x.numel() * x.element_size()
        sources = nbytes(self._shared_sources)
        extras = nbytes(self._priorities) + nbytes(self._source_scratch)
        index_tensors = list(self._cached_recency_indices.values()) + ([self._last_selection] if self._last_selection is not None else [])
        unique_indices = {x.data_ptr(): x for x in index_tensors}
        indices = sum(map(nbytes, unique_indices.values()))
        auxiliary = sources + nbytes(self._scratch) + extras + indices
        stats.update({"cache_policy": self.policy, "policy_configuration": dict(self.configuration),
                      "full_original_prefix_tokens": self.full_prefix_tokens, "budget_profile": self.profile,
                      "peak_budget_tokens": self.capacity, "staging_tokens": self.slack,
                      "retained_tokens_after_compaction": self.budget,
                      "effective_generated_retained_tokens": self.generated_budget, "recent_tokens": self.recent,
                      "protects_entire_prefix": True, "source_position_bytes": sources,
                      "source_positions_shared_across_layers_and_heads": True,
                      "selection_shared_across_layers_and_heads": True,
                      "priority_lookup_bytes": nbytes(self._priorities), "source_scratch_bytes": nbytes(self._source_scratch),
                      "selection_index_bytes": indices, "query_buffer_bytes": 0, "diagnostic_score_state_bytes": 0,
                      "selection_scratch_peak_bytes_estimate": self.selection_scratch_peak_bytes,
                      "selection_scratch_estimate_scope": "conservative explicit index tensor bound; excludes library-internal workspace",
                      "optimized_storage_and_selection": self.optimized,
                      "auxiliary_persistent_bytes": auxiliary,
                      "allocated_cache_and_auxiliary_bytes": stats["kv_capacity_bytes"] + auxiliary,
                      "original_generated_position_priority": True, "priority_horizon": self.max_generated_positions})
        return stats
