"""Matched-budget visual token selectors for the LLaVA adapter."""

import torch
import torch.nn.functional as F


def query_similarity(vision, text, keep_count, scorer=None):
    if scorer is None:
        scores = (
            F.normalize(vision.float(), dim=-1)
            @ F.normalize(text.float(), dim=-1).T
        ).max(dim=-1).values
    else:
        scores = scorer(vision, text, "max")
    return scores.topk(keep_count).indices.sort().values


def divprune(vision, keep_count):
    """Official DivPrune max-min diversity rule."""
    return divprune_order(vision, keep_count).sort().values


def divprune_order(vision, keep_count):
    """Return DivPrune selections in greedy marginal-gain order."""
    normalized = F.normalize(vision.float(), dim=-1)
    distances = 1.0 - normalized @ normalized.T
    selected = torch.empty(
        keep_count, dtype=torch.long, device=vision.device
    )
    for index in range(keep_count):
        if index == 0:
            scores = distances.topk(2, dim=0, largest=False).values[1]
        else:
            scores = distances[selected[:index]].min(dim=0).values
        selected[index] = scores.argmax()
    return selected


def cdpruner(vision, relevance, keep_count):
    """Official CDPruner conditional-DPP MAP rule."""
    return cdpruner_order(vision, relevance, keep_count).sort().values


def cdpruner_kernel(vision, relevance):
    """Build the conditional-DPP kernel used by CDPruner."""
    normalized = F.normalize(vision.float(), dim=-1)
    similarity = normalized @ normalized.T
    relevance = relevance.float()
    relevance = (
        relevance - relevance.min() + 1e-6
    ) / (relevance.max() - relevance.min()).clamp_min(1e-6)
    return relevance[:, None] * similarity * relevance[None, :]


def cdpruner_order(vision, relevance, keep_count):
    """Return CDPruner selections in greedy DPP-MAP order."""
    kernel = cdpruner_kernel(vision, relevance)

    token_count = len(vision)
    cis = torch.zeros(
        (keep_count, token_count),
        device=vision.device,
        dtype=kernel.dtype,
    )
    di2s = torch.diagonal(kernel).clone()
    selected = torch.empty(
        keep_count, device=vision.device, dtype=torch.long
    )
    for index in range(keep_count):
        chosen = di2s.argmax()
        selected[index] = chosen
        if index:
            correction = (
                cis[:index, chosen, None] * cis[:index]
            ).sum(dim=0)
        else:
            correction = 0.0
        denominator = di2s[chosen].clamp_min(1e-12).sqrt()
        update = (kernel[chosen] - correction) / denominator
        cis[index] = update
        di2s -= update.square()
        di2s[chosen] = -float("inf")
    return selected


def keep_visual_positions(
    inputs_embeds, attention_mask, image_mask, selected_relative
):
    image_positions = image_mask[0].nonzero(as_tuple=False).squeeze(1)
    text_positions = (
        ~image_mask[0] & attention_mask[0].bool()
    ).nonzero(as_tuple=False).squeeze(1)
    selected_positions = image_positions[selected_relative]
    positions = torch.cat(
        [text_positions, selected_positions]
    ).sort().values
    pruned = inputs_embeds[:, positions, :]
    pruned_mask = torch.ones(
        pruned.shape[:2],
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return pruned, pruned_mask
