"""Bounded-cache greedy generation smoke test on a tiny random Llama (CPU)."""

from types import SimpleNamespace

import pytest
import torch

runtime = pytest.importorskip("learned_score.mlsys_runtime")
transformers = pytest.importorskip("transformers")

VOCAB, HIDDEN = 96, 32


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    config = transformers.LlamaConfig(
        vocab_size=VOCAB, hidden_size=HIDDEN, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        attn_implementation="sdpa", eos_token_id=None, bos_token_id=None, pad_token_id=0)
    m = transformers.LlamaForCausalLM(config).eval()
    m.generation_config.eos_token_id = None
    return m


def _pruned(model, keep, seed=1):
    """A fake 'pruned' multimodal prefix: embeddings of retained tokens with their
    original (non-contiguous) positions, as produced by physical visual pruning."""
    torch.manual_seed(seed)
    ids = torch.randint(1, VOCAB, (1, 40))
    positions = torch.tensor(sorted(keep))
    embeds = model.get_input_embeddings()(ids)[:, positions]
    return SimpleNamespace(
        inputs_embeds=embeds,
        attention_mask=torch.ones(1, len(positions), dtype=torch.long),
        position_ids=positions[None],
        retained_source_positions=positions,
    )


KEEP = [0, 1, 2, 5, 9, 13, 20, 21, 30, 38, 39]


def test_greedy_is_deterministic(model):
    a = runtime.generate_bounded(model, _pruned(model, KEEP), max_new_tokens=12, generated_budget=4, slack=2)
    b = runtime.generate_bounded(model, _pruned(model, KEEP), max_new_tokens=12, generated_budget=4, slack=2)
    assert a.ids.shape == (1, 12)
    assert torch.equal(a.ids, b.ids)
    assert a.metrics["max_active_cache_tokens"] <= len(KEEP) + 4 + 2
    assert a.cache.source_positions[: len(KEEP)].tolist() == KEEP
    assert a.cache.compaction_count >= 1


def test_unbounded_budget_matches_manual_full_cache_greedy(model):
    pruned = _pruned(model, KEEP)
    result = runtime.generate_bounded(model, pruned, max_new_tokens=8, generated_budget=64)
    assert result.cache.compaction_count == 0
    # Reference: plain HF forward with a DynamicCache and the same positions.
    cache = transformers.DynamicCache()
    with torch.no_grad():
        out = model(inputs_embeds=pruned.inputs_embeds, position_ids=pruned.position_ids,
                    past_key_values=cache, use_cache=True)
        token = out.logits[:, -1].argmax(-1, keepdim=True)
        ref, pos = [token], pruned.position_ids[:, -1:] + 1
        for _ in range(7):
            out = model(input_ids=token, position_ids=pos, past_key_values=cache, use_cache=True)
            token = out.logits[:, -1].argmax(-1, keepdim=True)
            ref.append(token)
            pos = pos + 1
    assert result.ids.tolist() == torch.cat(ref, 1).tolist()


def test_eos_stops_generation(model):
    pruned = _pruned(model, KEEP)
    free = runtime.generate_bounded(model, pruned, max_new_tokens=10, generated_budget=64, stop_on_eos=False)
    eos = int(free.ids[0, 3])
    first = free.ids[0].tolist().index(eos)
    model.generation_config.eos_token_id = eos
    try:
        stopped = runtime.generate_bounded(model, pruned, max_new_tokens=10, generated_budget=64)
    finally:
        model.generation_config.eos_token_id = None
    assert stopped.ids[0].tolist() == free.ids[0, : first + 1].tolist()


def test_rejects_batched_or_padded_inputs(model):
    pruned = _pruned(model, KEEP)
    pruned.attention_mask = pruned.attention_mask.clone()
    pruned.attention_mask[0, 0] = 0
    with pytest.raises(ValueError):
        runtime.generate_bounded(model, pruned, max_new_tokens=2, generated_budget=2)
