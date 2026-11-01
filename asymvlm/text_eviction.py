"""Generated-text KV eviction: protected-prefix recency.

The image + prompt prefix (after visual pruning) is never evicted; of the
generated tokens only the most recent ``budget`` are kept in the KV cache.
Wraps ``learned_score.mlsys_runtime.matched_generation_v3.generate_matched_v3``
with policy ``'protected_recency'``. KV capacity per layer is
``retained_prefix + budget + 1`` (capacity computed on the retained prefix,
i.e. no budget donation from pruned visual tokens), as in the paper's joint
vision+text panel.
"""
from contextlib import contextmanager

from .config import TEXT_EVICTION_BUDGET


@contextmanager
def retained_capacity_basis():
    """Size the matched cache on the *retained* prefix (no donation of pruned visual budget)."""
    from learned_score.mlsys_runtime import matched_generation_v3 as generation
    original = generation.MatchedKVCacheV3

    class Cache(original):
        def __init__(self, prefix_length, full_prefix_tokens, budget_profile, policy, **kwargs):
            super().__init__(prefix_length, prefix_length, budget_profile, policy, **kwargs)
            self.full_prefix_tokens = full_prefix_tokens
    generation.MatchedKVCacheV3 = Cache
    try:
        yield
    finally:
        generation.MatchedKVCacheV3 = original


def generate_with_text_eviction(runtime, prepared, pruned, max_new_tokens=4096,
                                budget=TEXT_EVICTION_BUDGET, stop_on_eos=True, **kwargs):
    """Decode ``pruned`` keeping the protected prefix and the last ``budget`` generated tokens.

    ``prepared`` is the unpruned adapter output (used for the full prefix length
    and original visual positions); ``pruned`` comes from ``asymvlm.pruning.prune``.
    Returns ``(text, result)``.
    """
    import torch
    from learned_score.mlsys_runtime.matched_generation_v3 import generate_matched_v3
    full_prefix = int(prepared.inputs_embeds.shape[1])
    with torch.no_grad(), retained_capacity_basis():
        result = generate_matched_v3(runtime.model, pruned, max_new_tokens, full_prefix_tokens=full_prefix,
                                     budget_profile=budget, policy="protected_recency", staging=1,
                                     visual_source_positions=prepared.visual_positions,
                                     stop_on_eos=stop_on_eos, **kwargs)
    return runtime.decode(result.ids.detach().cpu()), result


def prune_and_generate_long(runtime, prepared, keep_ratio=0.7, budget=TEXT_EVICTION_BUDGET, **kwargs):
    """Joint AsymVLM: visual pruning at ``keep_ratio`` + protected-recency text eviction."""
    from .pruning import prune, select_tokens
    indices = select_tokens(prepared, keep_ratio, runtime.scorer)
    return generate_with_text_eviction(runtime, prepared, prune(runtime, prepared, indices), budget=budget, **kwargs)
