"""Bounded text-KV cache: protected prefix + recency eviction of generated tokens."""

import pytest
import torch

runtime = pytest.importorskip("learned_score.mlsys_runtime")
ProtectedPrefixCache, RecencyPolicy = runtime.ProtectedPrefixCache, runtime.RecencyPolicy


def _states(indices, layer=0, heads=2, dim=3):
    values = torch.as_tensor(list(indices), dtype=torch.float32).reshape(1, 1, -1, 1)
    values = values.expand(1, heads, -1, dim).clone() + layer * 10000
    return values, values + 0.25


def _fill(prefix, budget, slack, steps, layers=2, source=None):
    cache = ProtectedPrefixCache(prefix, budget, slack, prefix_source_positions=source)
    for layer in range(layers):
        cache.update(*_states(range(prefix), layer), layer)
    for step in range(steps):
        cache.compact_if_needed()
        for layer in range(layers):
            cache.update(*_states([prefix + step], layer), layer)
    return cache


@pytest.mark.parametrize("prefix,budget,slack,steps", [(6, 4, 2, 50), (1, 0, 1, 10), (10, 16, 8, 200)])
def test_prefix_never_evicted_and_budget_respected(prefix, budget, slack, steps):
    cache = ProtectedPrefixCache(prefix, budget, slack)
    for layer in range(2):
        cache.update(*_states(range(prefix), layer), layer)
    for step in range(steps):
        cache.compact_if_needed()
        # Retained generated tokens after compaction <= budget (+slack before next compaction).
        assert cache.get_seq_length() - prefix <= budget + slack
        for layer in range(2):
            cache.update(*_states([prefix + step], layer), layer)
        for layer in range(2):
            assert cache.key_cache[layer][0, 0, :prefix, 0].tolist() == [float(i + layer * 10000) for i in range(prefix)]
        src = cache.source_positions.tolist()
        assert src[:prefix] == list(range(prefix))
        # Retained generated positions are the most recent, contiguous, in order.
        gen = src[prefix:]
        assert gen == list(range(prefix + step + 1 - len(gen), prefix + step + 1))
        # Stored values agree with their source positions (positions preserved).
        assert cache.key_cache[0][0, 0, :, 0].tolist() == [float(s) for s in src]
    assert cache._seen_tokens == prefix + steps


def test_compaction_keeps_exactly_budget_most_recent():
    cache = _fill(prefix=3, budget=4, slack=2, steps=6)
    assert cache.source_positions.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    event = cache.compact_if_needed()
    assert event is not None and event.removed_tokens == 2
    assert cache.source_positions.tolist() == [0, 1, 2, 5, 6, 7, 8]
    assert cache.get_seq_length() == 3 + 4


def test_noncontiguous_prefix_source_positions_preserved():
    source = torch.tensor([0, 4, 9, 30])  # pruned visual tokens keep original positions
    cache = _fill(prefix=4, budget=2, slack=1, steps=10, source=source)
    src = cache.source_positions.tolist()
    assert src[:4] == [0, 4, 9, 30]
    assert src[4:] == list(range(31 + 10 - len(src[4:]), 31 + 10))


def test_recency_policy():
    policy = RecencyPolicy()
    assert policy.select(torch.tensor([3, 5, 9, 12]), 2).tolist() == [2, 3]
    assert policy.select(torch.tensor([3, 5, 9]), 3).tolist() == [0, 1, 2]
    assert policy.select(torch.tensor([3, 5, 9]), 0).numel() == 0
    with pytest.raises(ValueError):
        policy.select(torch.tensor([1, 2]), 3)


def test_capacity_and_memory_accounting():
    cache = _fill(prefix=5, budget=8, slack=4, steps=0)
    assert cache.capacity == 5 + 8 + 4
    stats = cache.memory_stats()
    assert stats["kv_active_bytes"] <= stats["kv_capacity_bytes"]


@pytest.mark.parametrize("kwargs", [dict(prefix_length=-1, generated_budget=1),
                                    dict(prefix_length=1, generated_budget=-1),
                                    dict(prefix_length=1, generated_budget=1, slack=0)])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        ProtectedPrefixCache(**kwargs)


def test_overflow_requires_explicit_compaction():
    cache = _fill(prefix=2, budget=1, slack=1, steps=2)
    with pytest.raises(RuntimeError):
        cache.update(*_states([4]), 0)


# --------------------------------------------------------------------------- #
# Public asymvlm.text_eviction API (optional)
# --------------------------------------------------------------------------- #

def test_asymvlm_retained_capacity_basis_is_scoped():
    text_eviction = pytest.importorskip("asymvlm.text_eviction")
    generation = pytest.importorskip("learned_score.mlsys_runtime.matched_generation_v3")
    original = generation.MatchedKVCacheV3
    with text_eviction.retained_capacity_basis():
        assert generation.MatchedKVCacheV3 is not original
        assert issubclass(generation.MatchedKVCacheV3, original)
    assert generation.MatchedKVCacheV3 is original
    with pytest.raises(RuntimeError):
        with text_eviction.retained_capacity_basis():
            raise RuntimeError("boom")
    assert generation.MatchedKVCacheV3 is original
