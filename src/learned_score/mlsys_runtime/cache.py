"""Bounded KV storage with an immutable prefix and compactable suffix.

This is a single-request, eager HF Cache implementation, not a paged serving
engine or a variable-length batched cache. Logical/source positions are kept
separate from physical storage indices. Attention receives active tensor views
and does not need attention-weight materialization.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers.cache_utils import Cache

from .policies import RecencyPolicy


@dataclass(frozen=True)
class Compaction:
    retained_indices: torch.Tensor
    removed_tokens: int
    copied_kv_bytes: int


class ProtectedPrefixCache(Cache):
    """Allocate P + min(B + slack, maximum generated inputs) slots per layer.

    The first update must contain the complete protected prefix. Subsequent
    model forwards append generated tokens. Call ``compact_if_needed`` between
    completed forwards, before preparing the next attention mask. ``update``
    intentionally never evicts implicitly: eviction inside one layer would
    invalidate masks already computed for other layers.

    ``prefix_source_positions`` identifies original sequence entries, not RoPE
    coordinates. The runner separately supplies unchanged model position IDs.
    """

    is_compileable = False
    is_sliding = False

    def __init__(
        self,
        prefix_length: int,
        generated_budget: int,
        slack: int = 32,
        prefix_source_positions: Optional[torch.Tensor] = None,
        max_generated_tokens: Optional[int] = None,
    ):
        super().__init__()
        if prefix_length < 0 or generated_budget < 0 or slack < 1:
            raise ValueError("prefix/budget must be nonnegative; slack must be positive")
        if max_generated_tokens is not None and max_generated_tokens < 0:
            raise ValueError("maximum generated inputs must be nonnegative")
        if prefix_source_positions is not None:
            if prefix_source_positions.ndim != 1 or prefix_source_positions.numel() != prefix_length:
                raise ValueError("prefix source positions must match the prefix")
            if prefix_source_positions.dtype != torch.long:
                raise ValueError("prefix source positions must have dtype torch.long")
            if prefix_length and (
                bool((prefix_source_positions < 0).any())
                or bool((prefix_source_positions[1:] <= prefix_source_positions[:-1]).any())
            ):
                raise ValueError("prefix source positions must be nonnegative and strictly increasing")
        self.prefix_length = int(prefix_length)
        self.generated_budget = int(generated_budget)
        self.slack = int(slack)
        suffix_capacity = generated_budget + slack
        if max_generated_tokens is not None:
            suffix_capacity = min(suffix_capacity, max_generated_tokens)
        self.capacity = prefix_length + suffix_capacity
        self._prefix_source = prefix_source_positions
        self._keys = []
        self._values = []
        self._lengths = []
        self._source_buffer = None
        self._next_source = None
        self._scratch = None
        self._seen_tokens = 0
        self.compaction_count = 0
        self.evicted_tokens = 0
        self.copied_kv_bytes = 0

    @property
    def key_cache(self):
        return [tensor[..., :length, :] for tensor, length in zip(self._keys, self._lengths)]

    @property
    def value_cache(self):
        return [tensor[..., :length, :] for tensor, length in zip(self._values, self._lengths)]

    @property
    def source_positions(self):
        if self._source_buffer is None:
            device = self._prefix_source.device if self._prefix_source is not None else "cpu"
            return torch.empty(0, dtype=torch.long, device=device)
        return self._source_buffer[:self.get_seq_length()]

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
        length = self.get_seq_length(layer_idx)
        if length + new_seq_length > self.capacity:
            raise RuntimeError("cache capacity exceeded; compact between forwards before appending")
        return length

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("beam search and batch reordering are not supported")

    def _initialize_sources(self, device):
        self._source_buffer = torch.empty(self.capacity, dtype=torch.long, device=device)
        prefix = self._prefix_source
        if prefix is None:
            prefix = torch.arange(self.prefix_length, device=device)
        else:
            prefix = prefix.to(device)
        self._source_buffer[:self.prefix_length].copy_(prefix)
        # One scalar read at initialization, never a per-cache-token transfer.
        self._next_source = int(prefix[-1]) + 1 if self.prefix_length else 0

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.ndim != 4 or key_states.shape != value_states.shape:
            raise ValueError("K/V must share [batch, kv_heads, tokens, head_dim] shape")
        if key_states.shape[0] != 1:
            raise ValueError("ProtectedPrefixCache currently supports one request, not batched generation")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("K/V dtype and device must match")
        if not 0 <= layer_idx <= len(self._keys):
            raise ValueError("initialize cache layers in contiguous increasing order")
        count = key_states.shape[-2]
        if layer_idx == 0:
            self._check_completed_forward()
        start = self.get_seq_length(layer_idx)
        end = start + count
        if end > self.capacity:
            raise RuntimeError("cache capacity exceeded; call compact_if_needed before the next forward")
        if layer_idx == len(self._keys):
            if count != self.prefix_length:
                raise ValueError("the first update of every layer must contain the complete protected prefix")
            if self._keys and (
                key_states.shape[:2] + key_states.shape[-1:] != self._keys[0].shape[:2] + self._keys[0].shape[-1:]
                or key_states.device != self._keys[0].device
                or key_states.dtype != self._keys[0].dtype
            ):
                raise ValueError("all layers must share KV shape, dtype and device for reusable scratch")
            shape = (*key_states.shape[:-2], self.capacity, key_states.shape[-1])
            self._keys.append(key_states.new_empty(shape))
            self._values.append(value_states.new_empty(shape))
            self._lengths.append(0)
            if layer_idx == 0:
                self._initialize_sources(key_states.device)
        elif key_states.shape[:2] + key_states.shape[-1:] != self._keys[layer_idx].shape[:2] + self._keys[layer_idx].shape[-1:]:
            raise ValueError("KV shape changed within a request")
        elif key_states.dtype != self._keys[layer_idx].dtype or key_states.device != self._keys[layer_idx].device:
            raise ValueError("KV dtype/device changed within a request")
        if layer_idx and end != self._lengths[0]:
            raise RuntimeError("each layer must append the same entries in a forward")
        self._keys[layer_idx][..., start:end, :].copy_(key_states)
        self._values[layer_idx][..., start:end, :].copy_(value_states)
        self._lengths[layer_idx] = end
        if layer_idx == 0:
            if start >= self.prefix_length and start != 0:
                self._source_buffer[start:end].copy_(torch.arange(
                    self._next_source, self._next_source + count,
                    device=key_states.device, dtype=torch.long,
                ))
                self._next_source += count
            elif self.prefix_length == 0 and count:
                self._source_buffer[start:end].copy_(torch.arange(
                    self._next_source, self._next_source + count,
                    device=key_states.device, dtype=torch.long,
                ))
                self._next_source += count
            self._seen_tokens += count
        return self._keys[layer_idx][..., :end, :], self._values[layer_idx][..., :end, :]

    def _check_completed_forward(self):
        if self._lengths and any(length != self._lengths[0] for length in self._lengths):
            raise RuntimeError("compaction must occur after all layers complete their forward")

    @torch.no_grad()
    def compact_suffix(self, suffix_indices):
        """Retain explicit suffix-relative indices; never rewrite prefix bytes.

        Explicit indices are validated here, which can synchronize the GPU.
        ``compact_if_needed`` trusts the built-in recency policy and avoids
        those reductions. A custom policy should use this checked entrypoint.
        """
        return self._compact_suffix(suffix_indices, validate=True)

    def _compact_suffix(self, suffix_indices, validate):
        self._check_completed_forward()
        before = self.get_seq_length()
        suffix_length = before - self.prefix_length
        if self._source_buffer is None:
            raise RuntimeError("cannot compact an uninitialized cache")
        if suffix_indices.ndim != 1 or suffix_indices.dtype != torch.long:
            raise ValueError("suffix indices must be a one-dimensional long tensor")
        if suffix_indices.device != self._source_buffer.device:
            raise ValueError("suffix indices and cache must share a device")
        keep = suffix_indices.numel()
        if keep > self.generated_budget:
            raise ValueError("retained suffix exceeds the generated budget")
        if validate and keep and (
            bool((suffix_indices < 0).any())
            or bool((suffix_indices >= suffix_length).any())
            or bool((suffix_indices[1:] <= suffix_indices[:-1]).any())
        ):
            raise ValueError("suffix indices must be in range and strictly increasing")
        absolute = suffix_indices + self.prefix_length
        retained = torch.cat((
            torch.arange(self.prefix_length, device=absolute.device), absolute,
        ))
        removed = suffix_length - keep
        if removed <= 0:
            return Compaction(retained, 0, 0)
        if keep and self._scratch is None:
            first = self._keys[0]
            self._scratch = first.new_empty((*first.shape[:-2], self.generated_budget, first.shape[-1]))
        copied = 0
        for keys, values in zip(self._keys, self._values):
            for tensor in (keys, values):
                if keep:
                    scratch = self._scratch[..., :keep, :]
                    # out= avoids transient allocations for every layer. The
                    # stream executes gather/copy serially before scratch reuse.
                    torch.index_select(tensor, -2, absolute, out=scratch)
                    tensor[..., self.prefix_length:self.prefix_length + keep, :].copy_(scratch)
                    copied += scratch.numel() * scratch.element_size() * 2
        selected_sources = self._source_buffer.index_select(0, absolute)
        self._source_buffer[self.prefix_length:self.prefix_length + keep].copy_(selected_sources)
        self._lengths = [self.prefix_length + keep] * len(self._lengths)
        self.compaction_count += 1
        self.evicted_tokens += removed
        self.copied_kv_bytes += copied
        return Compaction(retained, removed, copied)

    @torch.no_grad()
    def compact_if_needed(self, incoming_tokens=1, policy=None):
        if incoming_tokens < 1:
            raise ValueError("incoming_tokens must be positive")
        self._check_completed_forward()
        if self.get_seq_length() + incoming_tokens <= self.capacity:
            return None
        if self.prefix_length + self.generated_budget + incoming_tokens > self.capacity:
            raise ValueError("incoming chunk exceeds available overflow allowance")
        if policy is None:
            policy = RecencyPolicy()
        suffix = self.source_positions[self.prefix_length:]
        indices = policy.select(suffix, min(self.generated_budget, suffix.numel()))
        return self._compact_suffix(indices, validate=type(policy) is not RecencyPolicy)

    def memory_stats(self):
        def nbytes(tensor):
            return tensor.numel() * tensor.element_size() if tensor is not None else 0
        capacity_bytes = sum(nbytes(tensor) for tensor in self._keys + self._values)
        active_bytes = sum(nbytes(tensor) for tensor in self.key_cache + self.value_cache)
        scratch_bytes = nbytes(self._scratch)
        source_bytes = nbytes(self._source_buffer)
        return {
            "prefix_tokens": self.prefix_length,
            "generated_budget": self.generated_budget,
            "overflow_allowance": self.slack,
            "capacity_tokens_per_layer": self.capacity,
            "active_tokens": self.get_seq_length(),
            "logical_tokens_seen": self._seen_tokens,
            "kv_capacity_bytes": capacity_bytes,
            "kv_active_bytes": active_bytes,
            "scratch_bytes": scratch_bytes,
            "source_metadata_bytes": source_bytes,
            "total_owned_bytes": capacity_bytes + scratch_bytes + source_bytes,
            "compaction_events": self.compaction_count,
            "evicted_tokens": self.evicted_tokens,
            "compaction_copied_kv_bytes": self.copied_kv_bytes,
        }
