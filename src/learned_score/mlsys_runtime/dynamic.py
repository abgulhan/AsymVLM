"""Single-forward, growing HF DynamicCache baseline for the frozen runner.

The scoped factory substitution changes storage only. Model forwards, masks,
positions, EOS handling and measurement boundaries remain the same. It is not
the two-forward correctness reference and never preallocates the output cap.
This process-local wrapper is intentionally single-request/non-concurrent.
"""

from contextlib import contextmanager
import threading

import torch
from transformers.cache_utils import DynamicCache

from . import generation


_FACTORY_LOCK = threading.Lock()


class GrowingDynamicCache(DynamicCache):
    """Stock HF K/V concatenation plus original-position and byte accounting."""

    def __init__(self, prefix_length, generated_budget, slack=32,
                 prefix_source_positions=None, max_generated_tokens=None):
        super().__init__()
        if prefix_length < 1 or generated_budget < 1 or slack < 1:
            raise ValueError("dynamic baseline requires positive prefix/budget/slack")
        if max_generated_tokens is None or max_generated_tokens < 0:
            raise ValueError("dynamic baseline requires a finite nonnegative generation cap")
        if generated_budget < max_generated_tokens:
            raise ValueError("DynamicCache baseline does not implement eviction")
        if prefix_source_positions is not None and (
            prefix_source_positions.dtype != torch.long
            or prefix_source_positions.shape != (prefix_length,)
            or bool((prefix_source_positions < 0).any())
            or bool((prefix_source_positions[1:] <= prefix_source_positions[:-1]).any())
        ):
            raise ValueError("prefix source positions must be increasing nonnegative int64")
        self.prefix_length = prefix_length
        self.generated_budget = generated_budget
        self.slack = slack
        self.capacity = prefix_length + max_generated_tokens  # bound, NOT allocation
        self._prefix_sources = prefix_source_positions
        self._sources = None
        self._next_source = None
        self.compaction_count = self.evicted_tokens = self.copied_kv_bytes = 0

    @property
    def source_positions(self):
        if self._sources is None:
            raise RuntimeError("DynamicCache is not initialized")
        return self._sources

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if key_states.shape != value_states.shape or key_states.ndim != 4:
            raise ValueError("K/V must share [batch,heads,tokens,dimension] shape")
        if key_states.shape[0] != 1:
            raise ValueError("growing baseline supports one request")
        if key_states.dtype != value_states.dtype or key_states.device != value_states.device:
            raise ValueError("K/V dtype and device must match")
        before = self.get_seq_length(layer_idx)
        count = key_states.shape[-2]
        if before + count > self.capacity:
            raise RuntimeError("dynamic baseline exceeded its declared generation cap")
        if before == 0 and count != self.prefix_length:
            raise ValueError("first update must contain the complete prefix")
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if layer_idx == 0:
            if self._sources is None:
                source = self._prefix_sources
                self._sources = (torch.arange(self.prefix_length, device=key_states.device)
                                 if source is None else source.to(key_states.device).clone())
                self._next_source = int(self._sources[-1]) + 1
            else:
                new = torch.arange(self._next_source, self._next_source + count,
                                   device=key_states.device, dtype=torch.long)
                self._sources = torch.cat((self._sources, new))
                self._next_source += count
        return result

    def compact_if_needed(self, *args, **kwargs):
        raise RuntimeError("no-eviction DynamicCache must never compact")

    def reorder_cache(self, beam_idx):
        raise NotImplementedError("batch reordering/beam search are not benchmarked")

    def memory_stats(self):
        tensors = self.key_cache + self.value_cache
        # DynamicCache tensors are whole allocations, not views of a cap-sized
        # backing store. CUDA allocator peaks separately include cat transients.
        storage = sum(t.untyped_storage().nbytes() for t in tensors)
        active = sum(t.numel() * t.element_size() for t in tensors)
        metadata = 0 if self._sources is None else self._sources.untyped_storage().nbytes()
        length = self.get_seq_length()
        return {"prefix_tokens": self.prefix_length, "generated_budget": None,
                "overflow_allowance": 0, "capacity_tokens_per_layer": length,
                "declared_max_tokens_per_layer": self.capacity,
                "active_tokens": length, "logical_tokens_seen": self._seen_tokens,
                "kv_capacity_bytes": storage, "kv_active_bytes": active,
                "scratch_bytes": 0, "source_metadata_bytes": metadata,
                "total_owned_bytes": storage + metadata, "compaction_events": 0,
                "evicted_tokens": 0, "compaction_copied_kv_bytes": 0,
                "storage_implementation": "transformers.DynamicCache_growing_cat",
                "capacity_is_current_allocation_not_output_cap": True}


@contextmanager
def dynamic_storage():
    """Substitute only the runner's constructor; restore it even on failure.

Callers must not run another request in this process during the context. The
lock rejects nested/concurrent dynamic contexts rather than silently mixing
storage implementations. Experiment workers use separate processes/GPUs.
"""
    if not _FACTORY_LOCK.acquire(blocking=False):
        raise RuntimeError("dynamic baseline context is already active")
    original = generation.ProtectedPrefixCache
    try:
        generation.ProtectedPrefixCache = GrowingDynamicCache
        yield
    finally:
        generation.ProtectedPrefixCache = original
        _FACTORY_LOCK.release()


def generate_dynamic(model, pruned, max_new_tokens, generated_budget=None, **kwargs):
    if generated_budget not in (None, max_new_tokens):
        raise ValueError("growing DynamicCache is an eviction-disabled baseline")
    if kwargs.get("policy") is not None:
        raise ValueError("dynamic baseline cannot install an eviction policy")
    with dynamic_storage():
        result = generation.generate_bounded(model, pruned, max_new_tokens=max_new_tokens,
                                             generated_budget=max_new_tokens, **kwargs)
    result.metrics["storage_implementation"] = "transformers.DynamicCache_growing_cat"
    return result


def run_dynamic_request(adapter, row, select_visual, **kwargs):
    cap = kwargs["max_new_tokens"]
    if kwargs.get("generated_budget") not in (None, cap) or kwargs.get("policy") is not None:
        raise ValueError("dynamic request must disable text eviction")
    kwargs["generated_budget"] = cap
    with dynamic_storage():
        result = generation.run_request(adapter, row, select_visual, **kwargs)
    result.metrics["storage_implementation"] = "transformers.DynamicCache_growing_cat"
    return result
