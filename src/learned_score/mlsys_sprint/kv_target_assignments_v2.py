"""Answer-free allocations targeting *allocated cache* memory, not GPU memory.

For homogeneous K/V layers in ProtectedPrefixCache, each reserved sequence slot
owns 2 * layers * KV_heads * head_dim * element_size K/V bytes and 8 bytes of
source-position metadata. With max_new_tokens=128 and a 512-token text allowance,
127 generated-input slots are reserved and no compaction can occur; scratch
allocation is therefore zero. Early EOS changes active cache, not this capacity.

Each dataset's target is floor((1-reduction) * sum(N + P + reserve)) slots,
where N is the original visual count and P is the protected nonvisual prefix.
The adaptive and uniform allocations have *exactly the same aggregate slots*.
Uniform learned selection, DivPrune and CDPruner must all reuse uniform counts.

The frozen entropy rule is transferred to lower memory budgets without fitting
new quality labels. This is exploratory transfer, not calibrated confirmation.
The final integer-token residuals are explicit; they are not disguised as exact
one-percent choices. Full fallback is preserved and infeasibility fails closed.
"""

from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import re

from .adaptive_vision_policy import AdaptiveVisionPolicy, DEVELOPMENT_SPLITS
from .state import digest


FORMAT = "mlsys_kv_target_assignments_v2"
CONTRACT_FORMAT = "protected_prefix_allocated_cache_contract_v1"
TARGET_REDUCTIONS = (0.4, 0.5, 0.6)
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_RAW_CONTRACT = {"num_hidden_layers", "num_key_value_heads", "head_dim", "element_size_bytes",
                 "max_new_tokens", "generated_budget", "slack"}
_ROW_FIELDS = {"row_hash", "group_id", "image_sha256", "dataset", "split", "normalized_entropy",
               "input_visual_tokens", "input_text_tokens"}


def _positive_integer(value, name, *, zero=False):
    if type(value) is not int or value < (0 if zero else 1):
        raise ValueError(f"{name} must be a {'nonnegative' if zero else 'positive'} integer")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty unpadded string")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _percent(value):
    return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def make_cache_contract(*, num_hidden_layers, num_key_value_heads, head_dim,
                        element_size_bytes, max_new_tokens=128, generated_budget=512, slack=1):
    """Return exact capacity accounting for a no-compaction bounded-cache run.

    The caller obtains geometry/dtype from the actual local model configuration
    and must verify measured cache.memory_stats() against this contract at run
    time. This pure helper does not assume a particular Qwen/LLaVA geometry.
    """
    fields = dict(num_hidden_layers=num_hidden_layers, num_key_value_heads=num_key_value_heads,
                  head_dim=head_dim, element_size_bytes=element_size_bytes,
                  max_new_tokens=max_new_tokens, generated_budget=generated_budget, slack=slack)
    for name, value in fields.items():
        _positive_integer(value, name, zero=name == "generated_budget")
    if element_size_bytes not in {1, 2, 4, 8}:
        raise ValueError("unsupported K/V element size")
    reserve = max_new_tokens - 1
    if generated_budget + slack < reserve:
        raise ValueError("contract requires no compaction; generated allowance plus slack is too small")
    kv_bytes = 2 * num_hidden_layers * num_key_value_heads * head_dim * element_size_bytes
    return {"format": CONTRACT_FORMAT, **fields, "batch_size": 1,
            "reserved_generated_slots": reserve, "kv_bytes_per_slot": kv_bytes,
            "source_metadata_bytes_per_slot": 8, "total_owned_bytes_per_slot": kv_bytes + 8,
            "scratch_bytes": 0, "compaction_possible": False,
            "memory_quantity": "cache.memory_stats.total_owned_bytes = kv_capacity_bytes + source_metadata_bytes; scratch zero",
            "scope": "allocated cache capacity only; excludes weights, other tensors, active-cache-only and total GPU memory",
            "layer_shape_assumption": "all layers share batch=1, KV-head count, head dimension and element size",
            "early_eos_changes_capacity": False}


def _contract(value):
    if not isinstance(value, dict) or not _RAW_CONTRACT <= value.keys():
        raise ValueError("cache contract is missing model/runtime geometry")
    expected = make_cache_contract(**{key: value[key] for key in _RAW_CONTRACT})
    if value != expected and set(value) != _RAW_CONTRACT:
        raise ValueError("cache contract derived accounting differs from runtime formula")
    return expected


def _validate_rows(policy, rows):
    if not isinstance(policy, AdaptiveVisionPolicy) or policy.statistic != "normalized_entropy" or policy.form != "continuous":
        raise ValueError("KV-budget transfer requires a frozen continuous entropy policy")
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("assignment rows must be nonempty")
    checked, seen, images = [], set(), {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
            raise ValueError(f"rows must contain only answer-free fields {sorted(_ROW_FIELDS)}")
        for name in ("row_hash", "image_sha256"):
            _hash(row[name], name)
        for name in ("group_id", "dataset", "split"):
            _string(row[name], name)
        if row["split"] not in DEVELOPMENT_SPLITS:
            raise ValueError("KV-budget transfer is development-only, not final evaluation")
        if row["row_hash"] in seen:
            raise ValueError("duplicate request identity")
        seen.add(row["row_hash"])
        previous = images.setdefault(row["image_sha256"], row["group_id"])
        if previous != row["group_id"]:
            raise ValueError("same image appears in different document groups")
        n = _positive_integer(row["input_visual_tokens"], "input_visual_tokens")
        p = _positive_integer(row["input_text_tokens"], "input_text_tokens", zero=True)
        entropy = _number(row["normalized_entropy"], "normalized_entropy")
        if not 0 <= entropy <= 1:
            raise ValueError("normalized entropy must lie in [0,1]")
        # Explicit nondevelopment split makes any calibration overlap an error,
        # while the actual saved inputs remain labeled development throughout.
        policy.validate_evaluation_identity(row_hash=row["row_hash"], group_id=row["group_id"],
                                            image_sha256=row["image_sha256"], split="scorer_selection")
        checked.append({**row, "input_visual_tokens": n, "input_text_tokens": p,
                        "normalized_entropy": entropy})
    return sorted(checked, key=lambda row: row["row_hash"])


def _add_residual(counts, maxima, goal, order):
    result = list(counts)
    residual = goal - sum(result)
    if residual < 0:
        raise ValueError("initial allocation exceeds its exact target")
    while residual:
        eligible = [index for index in order if result[index] < maxima[index]]
        if not eligible:
            raise ValueError("insufficient capacity for exact integer reconciliation")
        # Bulk complete rounds avoid one loop per token on large-image inputs;
        # the last partial round still follows the declared deterministic order.
        rounds = min(residual // len(eligible), min(maxima[i] - result[i] for i in eligible))
        if rounds:
            for index in eligible:
                result[index] += rounds
            residual -= rounds * len(eligible)
        else:
            for index in eligible[:residual]:
                result[index] += 1
            residual -= min(residual, len(eligible))
    return result


def _uniform(rows, goal, minima):
    total = sum(row["input_visual_tokens"] for row in rows)
    maxima = [row["input_visual_tokens"] for row in rows]
    initial = [max(low, n * goal // total) for low, n in zip(minima, maxima)]
    order = sorted(range(len(rows)), key=lambda i: (-(maxima[i] * goal % total), rows[i]["row_hash"]))
    final = _add_residual(initial, maxima, goal, order)
    return final, initial, goal / total


def _adaptive(policy, rows, goal, minimum_percent, uniform_fraction):
    decisions = [policy.decide_from_statistic(row["normalized_entropy"], row["input_visual_tokens"]) for row in rows]
    counts = [row["input_visual_tokens"] for row in rows]
    minima = [n if decision.fallback else max(1, n * minimum_percent // 100) for n, decision in zip(counts, decisions)]
    maxima = counts
    if not sum(minima) <= goal <= sum(maxima):
        raise ValueError("full-fallback/protected-prefix constraints make adaptive KV target infeasible")
    center = _percent(policy._continuous_retention(0.5))
    uniform_percent = _percent(uniform_fraction)
    candidates = []
    for shift in range(-200, 201):
        percentages = [100 if d.fallback else min(100, max(minimum_percent, uniform_percent + _percent(d.retention) - center + shift))
                       for d in decisions]
        allocated = [n if d.fallback else max(1, n * pct // 100)
                     for n, d, pct in zip(counts, decisions, percentages)]
        if sum(allocated) <= goal:
            candidates.append((sum(allocated), -abs(shift), shift, allocated, percentages))
    if not candidates:
        raise ValueError("no adaptive percentage shift satisfies KV target")
    _, _, shift, initial, percentages = max(candidates, key=lambda item: item[:3])
    order = sorted(range(len(rows)), key=lambda i: (-rows[i]["normalized_entropy"], rows[i]["row_hash"]))
    final = _add_residual(initial, maxima, goal, order)
    return final, initial, percentages, decisions, shift


def build_assignments(policy, rows, *, manifest_sha256, cache_contract,
                      reductions=TARGET_REDUCTIONS, minimum_retention=0.1, source_provenance=None):
    """Build immutable-content, answer-free assignments for explicit cache targets.

    Returned payload is indexed as targets['40']['rows'][request_id]. All three
    uniform selectors MUST use uniform_selected_count. The adaptive selector
    MUST use adaptive_selected_count. Neither allocation chooses patch identity.
    """
    _hash(manifest_sha256, "manifest_sha256")
    if manifest_sha256 != policy.calibration_manifest_sha256:
        raise ValueError("manifest differs from frozen policy provenance")
    contract = _contract(cache_contract)
    rows = _validate_rows(policy, rows)
    low = _number(minimum_retention, "minimum_retention")
    minimum_percent = _percent(low)
    if not 0 < low < 1 or not math.isclose(low * 100, minimum_percent, abs_tol=1e-9, rel_tol=0):
        raise ValueError("minimum retention must use an integer percentage in (0,100)")
    if not isinstance(reductions, (list, tuple)) or not reductions:
        raise ValueError("at least one cache reduction is required")
    requested = []
    for value in reductions:
        reduction = _number(value, "reduction")
        percent = _percent(reduction)
        if reduction not in TARGET_REDUCTIONS or not math.isclose(reduction * 100, percent, abs_tol=1e-9, rel_tol=0):
            raise ValueError("only preregistered 40%, 50%, and 60% cache reductions are supported")
        requested.append(percent)
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate target reduction")
    if source_provenance is None:
        source_provenance = {}
    if not isinstance(source_provenance, dict):
        raise ValueError("source_provenance must be a JSON object")
    provenance = json.loads(json.dumps(source_provenance, sort_keys=True, allow_nan=False))
    groups = defaultdict(list)
    for row in rows:
        groups[row["dataset"]].append(row)
    reserve, bytes_per_slot = contract["reserved_generated_slots"], contract["total_owned_bytes_per_slot"]
    targets = {}
    for reduction in sorted(requested):
        assigned_rows, datasets = {}, {}
        for dataset, entries in sorted(groups.items()):
            counts = [row["input_visual_tokens"] for row in entries]
            overhead = [row["input_text_tokens"] + reserve for row in entries]
            full_slots = sum(counts) + sum(overhead)
            target_slots = full_slots * (100 - reduction) // 100
            visual_goal = target_slots - sum(overhead)
            minima = [max(1, n * minimum_percent // 100) for n in counts]
            if not sum(minima) <= visual_goal <= sum(counts):
                raise ValueError(f"{dataset}: {reduction}% cache reduction infeasible with protected prefix, reserve and minimum vision")
            uniform, uniform_before, uniform_fraction = _uniform(entries, visual_goal, minima)
            adaptive, adaptive_before, percentages, decisions, shift = _adaptive(
                policy, entries, visual_goal, minimum_percent, uniform_fraction)
            if sum(uniform) != visual_goal or sum(adaptive) != visual_goal:
                raise AssertionError("exact cache target reconciliation failed")
            for index, (row, decision) in enumerate(zip(entries, decisions)):
                fixed = overhead[index]
                assigned_rows[row["row_hash"]] = {
                    **{key: row[key] for key in ("row_hash", "group_id", "image_sha256", "dataset", "split", "input_visual_tokens", "input_text_tokens")},
                    "statistic_value": row["normalized_entropy"], "difficulty_percentile": decision.percentile,
                    "fallback": decision.fallback, "policy_original_retention": decision.retention,
                    "uniform_selected_count": uniform[index], "adaptive_selected_count": adaptive[index],
                    "uniform_initial_selected_count": uniform_before[index],
                    "adaptive_initial_selected_count": adaptive_before[index],
                    "adaptive_shifted_retention_percent": percentages[index],
                    "uniform_integer_residual_tokens": uniform[index] - uniform_before[index],
                    "adaptive_integer_residual_tokens": adaptive[index] - adaptive_before[index],
                    "full_expected_cache_bytes": (counts[index] + fixed) * bytes_per_slot,
                    "uniform_expected_cache_bytes": (uniform[index] + fixed) * bytes_per_slot,
                    "adaptive_expected_cache_bytes": (adaptive[index] + fixed) * bytes_per_slot,
                }
            full_bytes, target_bytes = full_slots * bytes_per_slot, target_slots * bytes_per_slot
            datasets[dataset] = {
                "requests": len(entries), "full_cache_slots": full_slots, "target_cache_slots": target_slots,
                "protected_nonvisual_prefix_slots": sum(row["input_text_tokens"] for row in entries),
                "reserved_generated_slots": len(entries) * reserve, "full_visual_tokens": sum(counts),
                "target_visual_tokens": visual_goal, "uniform_visual_tokens": sum(uniform), "adaptive_visual_tokens": sum(adaptive),
                "uniform_retention_fraction": uniform_fraction, "adaptive_percentage_shift": shift,
                "full_expected_cache_bytes": full_bytes, "uniform_expected_cache_bytes": target_bytes,
                "adaptive_expected_cache_bytes": target_bytes,
                "aggregate_allocated_cache_reduction_fraction": 1 - target_bytes / full_bytes,
                "uniform_mean_per_request_reduction_fraction": sum(1 - (n + o) / (original + o) for n, original, o in zip(uniform, counts, overhead)) / len(entries),
                "adaptive_mean_per_request_reduction_fraction": sum(1 - (n + o) / (original + o) for n, original, o in zip(adaptive, counts, overhead)) / len(entries),
                "full_fallback_requests": sum(d.fallback for d in decisions),
                "uniform_integer_residual_tokens": sum(uniform) - sum(uniform_before),
                "adaptive_integer_residual_tokens": sum(adaptive) - sum(adaptive_before),
                "exact_adaptive_uniform_cache_match": True, "token_error": 0,
            }
        full_total = sum(item["full_expected_cache_bytes"] for item in datasets.values())
        target_total = sum(item["adaptive_expected_cache_bytes"] for item in datasets.values())
        targets[str(reduction)] = {"requested_reduction_fraction": reduction / 100, "rows": assigned_rows,
                                   "datasets": datasets,
                                   "aggregate": {"full_expected_cache_bytes": full_total,
                                                 "adaptive_expected_cache_bytes": target_total,
                                                 "uniform_expected_cache_bytes": target_total,
                                                 "allocated_cache_reduction_fraction": 1 - target_total / full_total,
                                                 "ratio_definition": "1 - sum(method allocated cache bytes) / sum(full-image allocated cache bytes)"}}
    payload = {"format": FORMAT, "policy_sha256": digest(policy.payload), "policy_identity": {
                   name: getattr(policy, name) for name in ("model", "adapter", "checkpoint_sha256", "scorer_variant", "selector")},
               "manifest_sha256": manifest_sha256, "cache_contract": contract,
               "minimum_retention": low, "maximum_retention": 1.0,
               "labels_used": False, "refitted_for_targets": False,
               "scope": "exploratory transfer of frozen entropy curve to allocated-cache budgets on previously examined development inputs",
               "target_definition": "per dataset floor((1-reduction)*sum(full_visual + nonvisual_prefix + generated_reserve)) slots",
               "uniform_methods": ["uniform_learned", "uniform_divprune", "uniform_cdpruner"],
               "uniform_rule": "one dataset-level visual retention fraction; floor counts; largest-remainder row-hash tie-break reconciliation",
               "adaptive_rule": "frozen entropy percentile retention curve centered at uniform target; global percentage shift; higher-entropy-first token residuals",
               "source_provenance": provenance, "inputs": {row["row_hash"]: row for row in rows}, "targets": targets}
    return {**payload, "sha256": digest(payload)}


def load_assignments(path, policy, manifest_sha256, entries):
    """Reject any identity/accounting change by reconstructing the whole artifact.

    Manifest entries establish document/image identity. N/P and model geometry
    additionally require runtime checks against prepared embeddings and measured
    cache counters; no manifest-only loader can infer those model-dependent sizes.
    """
    from learned_score.mlsys_eval.common import dataset_family
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict) or payload.get("sha256") != digest({key: value for key, value in payload.items() if key != "sha256"}):
        raise ValueError("KV assignment digest mismatch")
    expected = {entry["row_hash"]: entry for entry in entries}
    if (len(expected) != len(entries) or not expected or payload.get("format") != FORMAT
            or payload.get("labels_used") is not False or payload.get("refitted_for_targets") is not False
            or payload.get("policy_sha256") != digest(policy.payload)
            or payload.get("manifest_sha256") != manifest_sha256
            or set(payload.get("inputs", {})) != set(expected)):
        raise ValueError("KV assignments do not match frozen policy, manifest, or request set")
    for request, entry in expected.items():
        row = payload["inputs"][request]
        if any(row.get(key) != entry.get(key) for key in ("row_hash", "group_id", "image_sha256")):
            raise ValueError("KV assignment identity mismatch")
        if row.get("dataset") != dataset_family(entry["dataset"]):
            raise ValueError("KV assignment dataset mismatch")
        for key in ("input_visual_tokens", "input_text_tokens", "split"):
            if key in entry and row.get(key) != entry[key]:
                raise ValueError(f"KV assignment {key} differs from supplied entries")
    rebuilt = build_assignments(policy, list(payload["inputs"].values()), manifest_sha256=manifest_sha256,
                                cache_contract=payload["cache_contract"],
                                reductions=tuple(int(key) / 100 for key in payload["targets"]),
                                minimum_retention=payload["minimum_retention"],
                                source_provenance=payload["source_provenance"])
    if rebuilt != payload:
        raise ValueError("KV assignment does not equal deterministic answer-free reconstruction")
    return payload
