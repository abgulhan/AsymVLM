"""Versioned, physically bounded cache policies for quality/systems ablations.

The hard payload ceiling is M = full original prefix + profile + 1.  Staging
space is INSIDE M: after compression there are M-staging retained entries.
H2O sums all queries in each GQA group. R-KV takes the maximum query-head logit
in each GQA group, uses the last eight queries, and preserves score order.
These are different, explicit algorithms; no sampled-attention H2O label.

Independent port of the published algorithms; frozen v1/v2 are not edited.
R-KV reference: Zefan-Cai/R-KV, 90db663f0ebb7f656c9b55f12f524bfbcf6e8966,
HuggingFace/rkv/compression/{r1_kv,utils}.py. H2O reference:
FMInference/H2O, ac75c2a8a9e76832b2a4139b9363373b56336bfb.
"""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .matched_cache import MatchedKVCache


RUNTIME_REVISION = "matched_policy_quality_systems_v3"
REFERENCE_COMMITS = {
    "H2O": "ac75c2a8a9e76832b2a4139b9363373b56336bfb",
    "R-KV": "90db663f0ebb7f656c9b55f12f524bfbcf6e8966",
}


@dataclass(frozen=True)
class PolicySpec:
    name: str
    protects_prefix: bool
    scoring: str
    default_staging: int
    selection_order: str


POLICY_REGISTRY = {
    p.name: p for p in (
        PolicySpec("protected_recency", True, "recency", 1, "chronological"),
        PolicySpec("h2o", False, "h2o", 1, "chronological"),
        PolicySpec("h2o_recent_adjusted", False, "h2o", 1, "chronological"),
        PolicySpec("protected_h2o", True, "h2o", 1, "chronological"),
        PolicySpec("rkv", False, "rkv", 128, "score_then_recent"),
        PolicySpec("protected_rkv", True, "rkv", 128, "prefix_then_score_then_recent"),
        PolicySpec("protected_random", True, "random", 128, "prefix_then_random_score_then_recent"),
    )
}
POLICIES = tuple(POLICY_REGISTRY)


def policy_configuration(policy, staging=None):
    if policy not in POLICY_REGISTRY:
        raise ValueError(f"unknown cache policy: {policy!r}")
    spec = POLICY_REGISTRY[policy]
    staging = spec.default_staging if staging is None else staging
    if isinstance(staging, bool) or not isinstance(staging, int) or staging < 1:
        raise ValueError("staging must be a positive integer included in the peak cap")
    return {**asdict(spec), "staging": staging, "rkv_window": 8,
            "rkv_kernel": 7, "rkv_mix_lambda": .1, "rkv_similarity_threshold": .5,
            "rkv_retain_direction": "last", "reference_commits": dict(REFERENCE_COMMITS)}


def _chronological_topk(scores, keep, optimized=True):
    """Stable score-descending/earlier-index ties, followed by index ordering."""
    length = scores.shape[-1]
    if not 0 <= keep <= length:
        raise ValueError("invalid top-k retention")
    shape = scores.shape[:-1]
    if keep == 0:
        return torch.empty((*shape, 0), device=scores.device, dtype=torch.long)
    if keep == length:
        return torch.arange(length, device=scores.device).expand(*shape, length)
    if optimized and keep == length - 1:
        # Stable descending sort retains earlier equal minima: remove the last
        # minimum. Attention masses are nonnegative finite probabilities/sums.
        drop = length - 1 - scores.flip(-1).argmin(dim=-1, keepdim=True)
        base = torch.arange(keep, device=scores.device).expand(*shape, keep)
        return base + (base >= drop).long()
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[..., :keep].sort(-1).values


def _protected_slot_indices(sources, prefix_sources):
    """Locate immutable source identities, even after per-head reordering."""
    prefix_sources = prefix_sources.to(sources.device)
    # searchsorted over the immutable source set avoids a [heads,K,P] tensor.
    location = torch.searchsorted(prefix_sources, sources.contiguous())
    membership = (location < len(prefix_sources)) & (
        prefix_sources[location.clamp(max=len(prefix_sources) - 1)] == sources)
    # Protected policies guarantee exactly P surviving prefix entries per head.
    rank = torch.where(membership, sources, torch.iinfo(torch.long).max)
    slots = rank.argsort(dim=-1, stable=True)[..., :len(prefix_sources)]
    return slots, membership


def h2o_indices(scores, sources, retained, recent, *, prefix_sources=None, optimized=True):
    """H2O heavy/recent selection; protected mode reserves source identities."""
    if scores.shape != sources.shape or scores.ndim != 3:
        raise ValueError("scores and source IDs must be [batch,KV heads,length]")
    length = scores.shape[-1]
    if not 0 <= recent <= retained <= length:
        raise ValueError("invalid H2O retained/recent counts")
    newest = torch.arange(length - recent, length, device=scores.device).expand(*scores.shape[:-1], recent)
    if prefix_sources is None:
        heavy = _chronological_topk(scores[..., :length - recent], retained - recent, optimized)
        return torch.cat((heavy, newest), dim=-1)
    prefix, protected = _protected_slot_indices(sources, prefix_sources)
    # Protection is checked by immutable identity, not by physical cache offset.
    eligible = ~protected[..., :length - recent]
    slot = torch.arange(length - recent, device=scores.device).expand_as(eligible)
    candidates = slot.masked_select(eligible).reshape(*scores.shape[:-1], length - recent - len(prefix_sources))
    candidate_scores = scores.gather(-1, candidates)
    chosen = candidates.gather(-1, _chronological_topk(
        candidate_scores, retained - recent - len(prefix_sources), optimized))
    return torch.cat((prefix, chosen, newest), dim=-1).sort(dim=-1).values


@torch.no_grad()
def rkv_scores(queries, keys, *, window=8, kernel=7, mix_lambda=.1, scratch_observer=None):
    """R-KV importance minus redundancy, independently for each KV head.

    Recent entries are excluded BEFORE importance softmax but INCLUDED in the
    redundancy matrix and its normalization. This reproduces the pinned code's
    no-neighbour convention: column zero is exempted if no cosine exceeds .5.
    Return score order to the selector; do not chronologically sort top-k.
    """
    if queries.ndim != 4 or keys.ndim != 4 or queries.shape[0] != keys.shape[0]:
        raise ValueError("R-KV expects Q and K tensors [batch,heads,length,dim]")
    batch, kv_heads, length, dim = keys.shape
    if queries.shape[1] % kv_heads or queries.shape[-1] != dim or queries.shape[-2] < window:
        raise ValueError("R-KV requires eight recent queries and complete GQA groups")
    if length <= window or kernel < 1 or kernel % 2 != 1 or not 0 <= mix_lambda <= 1:
        raise ValueError("invalid R-KV window, pooling kernel, or mixture")
    groups = queries.shape[1] // kv_heads
    grouped = queries[..., -window:, :].reshape(batch, kv_heads, groups, window, dim)
    raw_logits = torch.matmul(grouped, keys[:, :, None].transpose(-1, -2)) * (dim ** -.5)
    logits = raw_logits.max(dim=2).values
    probabilities = logits[..., :-window].softmax(dim=-1, dtype=torch.float32)
    importance = probabilities.mean(dim=-2).to(queries.dtype)
    importance = F.max_pool1d(importance, kernel_size=kernel, stride=1, padding=kernel // 2)
    normalized = keys / (keys.norm(dim=-1, keepdim=True) + 1e-8)
    similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
    diagonal = torch.arange(length, device=keys.device)
    similarity[..., diagonal, diagonal] = 0
    above_threshold = similarity > .5
    locations = torch.where(above_threshold, diagonal, torch.zeros((), dtype=torch.long, device=keys.device))
    exempt = locations.max(dim=-1).values
    similarity.scatter_(-1, exempt.unsqueeze(-1), 0)
    redundancy = similarity.mean(dim=-2).softmax(dim=-1)[..., :-window]
    result = importance * mix_lambda - redundancy * (1 - mix_lambda)
    if scratch_observer is not None:
        scratch_observer(sum(x.numel() * x.element_size() for x in (
            raw_logits, logits, probabilities, importance, normalized, similarity,
            above_threshold, locations, exempt, redundancy, result)))
    return result


def rkv_indices(scores, sources, retained, *, prefix_sources=None, window=8):
    length = sources.shape[-1]
    if scores.shape != (*sources.shape[:-1], length - window) or not window < retained <= length:
        raise ValueError("R-KV scores must cover all but the recent eight entries")
    recent = torch.arange(length - window, length, device=scores.device).expand(*sources.shape[:-1], window)
    if prefix_sources is None:
        selected = scores.topk(retained - window, dim=-1).indices
        return torch.cat((selected, recent), dim=-1)
    prefix, protected = _protected_slot_indices(sources, prefix_sources)
    keep = retained - len(prefix_sources) - window
    if keep < 0:
        raise ValueError("protected prefix and observation window exceed retained budget")
    masked = scores.masked_fill(protected[..., :-window], -torch.inf)
    selected = masked.topk(keep, dim=-1).indices
    return torch.cat((prefix, selected, recent), dim=-1)


class MatchedKVCacheV3(MatchedKVCache):
    """One-request physical storage with identical hard peak slots per policy."""

    def __init__(self, prefix_length, full_prefix_tokens, budget_profile, policy, *, staging=None,
                 prefix_source_positions=None, seed=42, optimized=True, trace=False,
                 visual_source_positions=None, recent_window=None):
        self.configuration = policy_configuration(policy, staging)
        self.spec = POLICY_REGISTRY[policy]
        if isinstance(budget_profile, str) and budget_profile.startswith("base"):
            budget_profile = int(budget_profile[4:])
        if (isinstance(budget_profile, bool) or not isinstance(budget_profile, int) or budget_profile < 1
                or not isinstance(full_prefix_tokens, int) or full_prefix_tokens < prefix_length):
            raise ValueError("profile must be positive and full prefix cannot be shorter than retained prefix")
        self.full_prefix_tokens = full_prefix_tokens
        self.profile = budget_profile
        capacity = full_prefix_tokens + budget_profile + 1
        retained = capacity - self.configuration["staging"]
        if retained < prefix_length:
            raise ValueError("staging inside the peak capacity leaves insufficient prefix space")
        if self.spec.scoring in {"rkv", "random"} and (
            retained <= 8 or (self.spec.protects_prefix and retained - prefix_length < 8)
        ):
            raise ValueError("R-KV/random need space for their complete recent-eight window")
        legacy = "h2o_gqa_native_positions" if self.spec.scoring == "h2o" else "asymvlm"
        super().__init__(prefix_length, retained - prefix_length, legacy,
                         slack=self.configuration["staging"], prefix_source_positions=prefix_source_positions,
                         total_retained_tokens=retained)
        self.policy = policy
        self.optimized = bool(optimized)
        self.trace_enabled = bool(trace)
        self.visual_source_positions = visual_source_positions
        self.seed = int(seed)
        self._generator = None
        self._indices = {}
        self._queries, self._query_updates = [], []
        self._prefill_scores, self._decode_scores = [], []
        self.query_buffer_bytes = 0
        self.selection_scratch_peak_bytes = 0
        self.recent = (retained - retained // 2 if policy == "h2o" else
                       (retained - prefix_length) // 2 if self.spec.scoring == "h2o" else
                       8 if self.spec.scoring in {"rkv", "random"} else retained - prefix_length)
        if recent_window is not None:
            limit = retained - prefix_length if self.spec.protects_prefix else retained
            if (policy not in {"h2o_recent_adjusted", "protected_h2o"}
                    or isinstance(recent_window, bool) or not isinstance(recent_window, int)
                    or not 0 <= recent_window <= limit):
                raise ValueError("recent_window is a bounded diagnostic override for adjusted/protected H2O only")
            self.recent = recent_window
        self.configuration["recent_window_override"] = recent_window

    @property
    def independently_selected_heads(self):
        return self.spec.scoring != "recency" and self.compaction_count > 0

    @torch.no_grad()
    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        result = super().update(key_states, value_states, layer_idx, cache_kwargs)
        if self._prefix_source is None:
            self._prefix_source = torch.arange(self.prefix_length, device=key_states.device)
        if self.trace_enabled and self.spec.scoring == "h2o":
            if len(self._prefill_scores) == layer_idx:
                self._prefill_scores.append(torch.zeros_like(self._scores[layer_idx]))
                self._decode_scores.append(torch.zeros_like(self._scores[layer_idx]))
            count = key_states.shape[-2]
            end = self.get_seq_length(layer_idx)
            self._prefill_scores[layer_idx][..., end-count:end].zero_()
            self._decode_scores[layer_idx][..., end-count:end].zero_()
        return result

    @torch.no_grad()
    def add_attention_mass(self, layer_idx, mass):
        super().add_attention_mass(layer_idx, mass)
        if self.trace_enabled:
            target = self._prefill_scores if self._forward_updates[layer_idx] == 1 else self._decode_scores
            target[layer_idx][..., :mass.shape[-1]].add_(mass.float())

    @torch.no_grad()
    def observe_queries(self, layer_idx, queries):
        if self.spec.scoring != "rkv":
            return
        if layer_idx == len(self._queries):
            self._queries.append(queries.new_empty((*queries.shape[:-2], 8, queries.shape[-1])))
            self._query_updates.append(0)
        if self._query_updates[layer_idx] + 1 != self._forward_updates[layer_idx]:
            raise RuntimeError("R-KV needs exactly one query observation after every layer forward")
        count = min(queries.shape[-2], 8)
        if count < 8:
            # Eight queries are available before an eligible compression. The
            # initial short prefix's uninitialized entries are flushed first.
            self._queries[layer_idx][..., :-count, :].copy_(self._queries[layer_idx][..., count:, :].clone())
        self._queries[layer_idx][..., -count:, :].copy_(queries[..., -count:, :])
        self._query_updates[layer_idx] += 1

    def _observe_selection_scratch(self, size):
        self.selection_scratch_peak_bytes = max(self.selection_scratch_peak_bytes, size)

    def _select(self, layer_idx, length):
        sources = self.layer_source_positions(layer_idx)
        if self.spec.scoring == "recency":
            cache_key = (length, sources.device)
            if self.optimized and cache_key in self._indices:
                return self._indices[cache_key]
            indices = torch.cat((torch.arange(self.prefix_length, device=sources.device),
                                 torch.arange(length - self.generated_budget, length, device=sources.device)))
            if self.optimized:
                self._indices[cache_key] = indices
            return indices.expand(*sources.shape[:-1], self.budget)
        prefix = self._prefix_source if self.spec.protects_prefix else None
        if self.spec.scoring == "h2o":
            return h2o_indices(self._scores[layer_idx][..., :length], sources, self.budget,
                               self.recent, prefix_sources=prefix, optimized=self.optimized)
        if self.spec.scoring == "rkv":
            scores = rkv_scores(self._queries[layer_idx], self._keys[layer_idx][..., :length, :],
                                scratch_observer=self._observe_selection_scratch)
        else:
            if self._generator is None:
                self._generator = torch.Generator(device=sources.device).manual_seed(self.seed)
            scores = torch.rand((*sources.shape[:-1], length - 8), device=sources.device,
                                generator=self._generator)
        return rkv_indices(scores, sources, self.budget, prefix_sources=prefix)

    @torch.no_grad()
    def compact_if_needed(self, incoming_tokens=1):
        if incoming_tokens < 1:
            raise ValueError("incoming token count must be positive")
        self._check_complete()
        length = self.get_seq_length()
        if length + incoming_tokens <= self.capacity:
            return None
        if self.budget + incoming_tokens > self.capacity:
            raise ValueError("incoming tokens exceed staging space")
        if self.spec.scoring == "h2o" and self._score_updates != self._forward_updates:
            raise RuntimeError("missing H2O attention update before compaction")
        if self.spec.scoring == "rkv" and self._query_updates != self._forward_updates:
            raise RuntimeError("missing R-KV recent-query observation before compaction")
        suffix_only = self.optimized and self.spec.scoring == "recency"
        copied_slots = self.generated_budget if suffix_only else self.budget
        if self._scratch is None:
            first = self._keys[0]
            self._scratch = first.new_empty((*first.shape[:-2], copied_slots, first.shape[-1]))
        copied = 0
        for layer, (keys, values) in enumerate(zip(self._keys, self._values)):
            indices = self._select(layer, length)
            if indices.ndim == 1:
                indices = indices.expand(*keys.shape[:2], self.budget)
            selected = indices[..., self.prefix_length:] if suffix_only else indices
            gather = selected[..., None].expand(*selected.shape, keys.shape[-1])
            target_start = self.prefix_length if suffix_only else 0
            for tensor in (keys, values):
                torch.gather(tensor, -2, gather, out=self._scratch)
                tensor[..., target_start:self.budget, :].copy_(self._scratch)
                copied += self._scratch.numel() * self._scratch.element_size() * 2
            self._sources[layer][..., :self.budget].copy_(self._sources[layer].gather(-1, indices))
            if self._scores[layer] is not None:
                self._scores[layer][..., :self.budget].copy_(self._scores[layer].gather(-1, indices))
            if self.trace_enabled and self.spec.scoring == "h2o":
                for component in (self._prefill_scores, self._decode_scores):
                    component[layer][..., :self.budget].copy_(component[layer].gather(-1, indices))
            self._lengths[layer] = self.budget
        removed = length - self.budget
        self.compaction_count += 1
        self.evicted_tokens += removed
        self.copied_kv_bytes += copied
        return {"removed_tokens": removed, "active_cache_tokens": self.budget,
                "copied_kv_bytes": copied, "staging_tokens": self.slack}

    def trace_snapshot(self):
        """Diagnostic CPU transfer only; never called when tracing is disabled."""
        layers = []
        visual = self.visual_source_positions
        for layer in range(len(self._keys)):
            sources = self.layer_source_positions(layer).detach().cpu()
            prefix = torch.isin(sources, self._prefix_source.detach().cpu())
            generated = sources >= self.full_prefix_tokens
            row = {"layer": layer, "source_ids": sources[0].tolist(),
                   "retained_prefix_per_head": prefix[0].sum(-1).tolist(),
                   "retained_generated_per_head": generated[0].sum(-1).tolist()}
            if visual is not None:
                image = torch.isin(sources, torch.as_tensor(visual).detach().cpu())
                row.update({"retained_image_per_head": image[0].sum(-1).tolist(),
                            "retained_prompt_per_head": (prefix & ~image)[0].sum(-1).tolist()})
            if self.trace_enabled and self.spec.scoring == "h2o":
                length = self.get_seq_length(layer)
                row["prefill_attention_scores"] = self._prefill_scores[layer][0, :, :length].detach().cpu().tolist()
                row["decode_attention_scores"] = self._decode_scores[layer][0, :, :length].detach().cpu().tolist()
            layers.append(row)
        return {"policy": self.policy, "compaction_count": self.compaction_count,
                "logical_tokens_seen": self._seen_tokens, "active_tokens": self.get_seq_length(),
                "visual_identity_available": visual is not None, "layers": layers}

    def memory_stats(self):
        stats = super().memory_stats()
        nbytes = lambda x: x.numel() * x.element_size()
        queries = sum(map(nbytes, self._queries))
        indices = sum(map(nbytes, self._indices.values()))
        diagnostics = sum(map(nbytes, self._prefill_scores + self._decode_scores))
        extra = queries + indices + diagnostics
        stats.update({"cache_policy": self.policy, "policy_configuration": dict(self.configuration),
                      "full_original_prefix_tokens": self.full_prefix_tokens,
                      "budget_profile": self.profile, "peak_budget_tokens": self.capacity,
                      "staging_tokens": self.slack, "retained_tokens_after_compaction": self.budget,
                      "effective_generated_retained_tokens": self.budget - self.prefix_length,
                      "recent_tokens": self.recent, "protects_entire_prefix": self.spec.protects_prefix,
                      "query_buffer_bytes": queries, "selection_index_bytes": indices,
                      "diagnostic_score_state_bytes": diagnostics,
                      "selection_scratch_peak_bytes_estimate": self.selection_scratch_peak_bytes,
                      "optimized_storage_and_selection": self.optimized,
                      "attention_query_observations_per_layer": list(self._query_updates)})
        stats["auxiliary_persistent_bytes"] += extra
        stats["allocated_cache_and_auxiliary_bytes"] += extra
        return stats
