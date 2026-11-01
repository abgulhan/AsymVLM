"""Immutable development-only protocol for small cache-history experiments.

Selection uses existing page/source/category identities, never answer labels or
the outcomes of v3. The reserved final split cannot be executed by this module.
"""

from collections import Counter
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from .state import digest, file_digest


FORMAT = "mlsys_cache_history_protocol_v4"
STAGES = ("stage1", "stage2", "stage3")
MODELS = ("qwen25", "llava_next")
ACTIVE_MODELS = ("qwen25",)
SEEDS = (42, 43, 44)
PATTERNS = ("protected_grid", "protected_weyl")
POLICIES = ("protected_recency", "protected_priority_random", *PATTERNS,
            "h2o_recent_adjusted", "rkv")


@dataclass(frozen=True)
class CacheCell:
    policy: str
    budget: int = 512
    staging: int = 128
    recent_mode: str = "eight"
    seed: int | None = None
    phase: int = 0

    def __post_init__(self):
        if (self.policy not in POLICIES or type(self.budget) is not int or self.budget < 8
                or type(self.staging) is not int or not 1 <= self.staging <= self.budget
                or self.recent_mode not in ("eight", "two_thirds")
                or type(self.phase) is not int or self.phase != 0):
            raise ValueError("invalid v4 cache cell")
        if self.policy == "protected_priority_random":
            if type(self.seed) is not int or self.seed not in SEEDS:
                raise ValueError("priority-random controls require a declared seed")
        elif self.seed is not None:
            raise ValueError("only priority-random controls accept a seed")

    @property
    def key(self):
        seed = "" if self.seed is None else f"_seed{self.seed}"
        return (f"{self.policy}_b{self.budget}_s{self.staging}"
                f"_recent{self.recent_mode}_phase{self.phase}{seed}")


def selected_pattern(selection):
    if not isinstance(selection, dict):
        raise ValueError("later stages require an immutable development selection")
    policy = selection.get("policy")
    mode = selection.get("recent_mode", "eight")
    phase = selection.get("phase", 0)
    if policy not in PATTERNS or mode not in ("eight", "two_thirds") or type(phase) is not int or phase != 0:
        raise ValueError("selection must name one predeclared deterministic pattern")
    return policy, mode, phase


def cells_for_stage(stage, selection=None):
    if stage not in STAGES:
        raise ValueError("v4 permits development stages only, never final evaluation")
    if stage == "stage1":
        if selection is not None:
            raise ValueError("stage1 cannot consume a post-outcome selection")
        return (CacheCell("protected_recency"),
                *(CacheCell("protected_priority_random", seed=seed) for seed in SEEDS),
                *(CacheCell(policy) for policy in PATTERNS))
    policy, mode, phase = selected_pattern(selection)
    if stage == "stage2":
        return (CacheCell(policy, recent_mode="two_thirds", phase=phase),
                *(CacheCell("protected_priority_random", recent_mode="two_thirds", seed=seed)
                  for seed in SEEDS))
    result = []
    for budget in (128, 256):
        staging = min(128, budget // 4)
        result.extend((CacheCell(policy, budget, staging, mode, phase=phase),
                       CacheCell("protected_recency", budget, 1),
                       CacheCell("h2o_recent_adjusted", budget, 1),
                       CacheCell("rkv", budget, staging)))
    return tuple(result)


def seal(value):
    value = dict(value)
    value.pop("sha256", None)
    return {**value, "sha256": digest(value)}


def _read_sealed(path):
    value = json.loads(Path(path).read_text())
    if value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError(f"immutable protocol digest mismatch: {path}")
    return value


def select_development_entries(entries, count=24):
    """Greedy category balancing with one page per source and fixed SHA ties.

At each step choose the least represented category still possessing an unused
source, then its first hashed source/page. Multiple categories increment all
their counts. Rare categories exhaust naturally; no impossible equal quota is
claimed. All decisions depend only on identities and category metadata.
"""
    entries = list(entries)
    if type(count) is not int or count < 1:
        raise ValueError("selection count must be a positive integer")
    identities = [entry.get("row_hash") for entry in entries]
    if any(not isinstance(rid, str) or not rid for rid in identities) or len(set(identities)) != len(identities):
        raise ValueError("development identities must be nonempty and unique")
    for entry in entries:
        if (not isinstance(entry.get("group_id"), str) or not entry["group_id"]
                or not isinstance(entry.get("categories"), list) or not entry["categories"]
                or any(not isinstance(c, str) or not c for c in entry["categories"])
                or entry.get("split", "scorer_dev") != "scorer_dev"):
            raise ValueError("development rows require source/category metadata")
    if len({entry["group_id"] for entry in entries}) < count:
        raise ValueError("insufficient distinct development sources")
    chosen, used_sources, coverage = [], set(), Counter()
    while len(chosen) < count:
        available = [entry for entry in entries if entry["group_id"] not in used_sources]
        categories = {category for entry in available for category in entry["categories"]}
        category = min(categories, key=lambda c: (coverage[c], c))
        entry = min((e for e in available if category in e["categories"]),
                    key=lambda e: (digest(["cache-history-v4-scout", e["group_id"], e["row_hash"]]), e["row_hash"]))
        chosen.append(dict(entry))
        used_sources.add(entry["group_id"])
        coverage.update(set(entry["categories"]))
    return chosen


def validate_protocol(protocol):
    required = {"format": FORMAT, "split": "scorer_dev", "final_labels_opened": False,
                "final_pages_opened": False, "max_new_tokens": 4096,
                "visual_retention": 1.0, "random_seeds": list(SEEDS),
                "relative_quality_floor": .95, "all_results_development": True}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("unsupported v4 protocol or forbidden final-label access")
    if protocol.get("active_models") != list(ACTIVE_MODELS):
        raise ValueError("v4 development execution is initially limited to Qwen")
    entries = protocol.get("development_entries", [])
    if len(entries) != 24 or len({e["group_id"] for e in entries}) != 24:
        raise ValueError("v4 requires exactly 24 distinct development sources")
    if protocol.get("development_request_ids") != [entry["row_hash"] for entry in entries]:
        raise ValueError("v4 page identities differ from the original entries")
    if protocol.get("entries") != entries:
        raise ValueError("v4 entry aliases differ")
    native = protocol.get("native_request_ids", [])
    if len(native) != 8 or len(set(native)) != 8:
        raise ValueError("v4 requires the eight original native diagnostic identities")
    selection = protocol.get("stage_selection")
    expected = [asdict(cell) for cell in cells_for_stage(protocol.get("stage"), selection)]
    if protocol.get("cells") != expected:
        raise ValueError("v4 stage cell matrix changed")
    if protocol["stage"] != "stage1" and not protocol.get("selection_artifact"):
        raise ValueError("later stage requires a pinned selection artifact")
    return protocol


def load_protocol(path):
    protocol = validate_protocol(_read_sealed(path))
    parent_path = Path(protocol["parent_protocol"])
    if file_digest(parent_path) != protocol["parent_sha256"]:
        raise ValueError("frozen v3 parent changed")
    parent = _read_sealed(parent_path)
    if (parent.get("format") != "mlsys_matched_cache_protocol_v3"
            or parent.get("split") != "scorer_dev" or parent.get("final_pages_opened") is not False):
        raise ValueError("v4 parent must be the untouched v3 development protocol")
    if any(protocol["input_hashes"].get(filename) != expected
           for filename, expected in parent["input_hashes"].items()):
        raise ValueError("v4 discarded a frozen parent input")
    for filename, expected in protocol["input_hashes"].items():
        if file_digest(filename) != expected:
            raise ValueError(f"frozen v4 input changed: {filename}")
    if not set(protocol["development_request_ids"]).issubset(parent["development_request_ids"]):
        raise ValueError("v4 request is outside the existing development split")
    if protocol.get("models") != parent.get("models"):
        raise ValueError("v4 model/checkpoint identities changed")
    manifest = json.loads(Path(protocol["manifest"]).read_text())
    development = {entry["row_hash"]: entry for entry in manifest["splits"]["scorer_dev"]}
    if any(development.get(entry["row_hash"]) != entry for entry in protocol["development_entries"]):
        raise ValueError("v4 original development entries changed")
    return protocol


def build_stage_protocol(protocol, stage, selection, *, selection_path):
    """Derive a new immutable stage; never mutate the earlier stage artifact."""
    validate_protocol(protocol)
    if stage not in ("stage2", "stage3"):
        raise ValueError("only stage2/stage3 may follow a development selection")
    if (protocol["stage"], stage) not in (("stage1", "stage2"), ("stage2", "stage3")):
        raise ValueError("v4 stages must advance in their declared order")
    selection_path = Path(selection_path).resolve()
    artifact = json.loads(selection_path.read_text())
    # The complete decision (metrics, provenance and gates) is bound by hash.
    if artifact.get("selection") != selection or artifact.get("protocol_sha256") != protocol["sha256"]:
        raise ValueError("selection artifact does not bind this protocol and selection")
    if artifact.get("passed") is not True or artifact.get("final_labels_opened") is not False:
        raise ValueError("failed or final-informed selection cannot launch another stage")
    value = {**protocol, "stage": stage, "stage_selection": dict(selection),
             "previous_stage_sha256": protocol["sha256"],
             "selection_artifact": str(selection_path),
             "selection_artifact_sha256": file_digest(selection_path),
             "cells": [asdict(cell) for cell in cells_for_stage(stage, selection)],
             "input_hashes": {**protocol["input_hashes"], str(selection_path): file_digest(selection_path)}}
    if artifact.get("source_report"):
        report_path = Path(artifact["source_report"]).resolve()
        report_hash = file_digest(report_path)
        if artifact.get("source_report_sha256") != report_hash:
            raise ValueError("selection source report changed")
        value.update(previous_report=str(report_path), previous_report_sha256=report_hash)
        value["input_hashes"][str(report_path)] = report_hash
    return seal(validate_protocol(value))
