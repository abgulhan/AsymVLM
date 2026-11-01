"""Read-only, exact-identity splitting of new work and audited physical reuse.

The input index must already belong to the caller's pinned development model,
checkpoint, and manifest. Historical outputs do not all carry those identities
inline, so this helper never invents that audit. It revalidates every reused
original file, outcome, geometry and selected patch set, and checks every
available inline identity against the new schedule and feature curves.

No writes, GPU work, label-based count decisions, or interpolation occur here.
"""

import copy
from pathlib import Path

from .compression_matrix_v8 import seal
from .compression_policy_v10 import canonical_dataset
from .compression_report_v11 import _verify_physical_provenance
from .state import digest, file_digest

FORMAT = "mlsys_compression_reuse_v11"
IDENTITIES = ("checkpoint_sha256", "manifest_sha256", "model_revision", "generation_config_sha256",
              "inference_config_sha256", "runtime_contract_sha256", "max_new_tokens", "generated_budget",
              "query_control", "prompt_sha256", "image_sha256")


def _checked_schedule(schedule):
    if not isinstance(schedule, dict) or schedule.get("sha256") != digest({k: v for k, v in schedule.items() if k != "sha256"}):
        raise ValueError("schedule content hash is invalid")
    if schedule.get("split") not in ("fit", "screening") or schedule.get("answers_used_to_choose_counts") is not False:
        raise ValueError("reuse requires an answer-free development schedule")
    if not isinstance(schedule.get("rows"), dict) or not schedule["rows"]:
        raise ValueError("nonempty scheduled development rows required")
    if not schedule.get("model") or not schedule.get("checkpoint_sha256") or not schedule.get("manifest_sha256"):
        raise ValueError("frozen model/checkpoint/manifest identity required")


def _check_geometry(schedule, request, geometry, cell, source):
    if source.get("request_id") != request or geometry.get("row_hash", request) != request:
        raise ValueError("cached physical request identity differs")
    for field in ("group_id", "input_visual_tokens", "input_text_tokens"):
        if field not in source or source[field] != geometry.get(field):
            raise ValueError("cached physical geometry differs: " + field)
    if canonical_dataset(source["dataset"]) != canonical_dataset(geometry["dataset"]):
        raise ValueError("cached physical dataset differs")
    if source.get("model_tag") != schedule["model"]:
        raise ValueError("cached physical model tag differs")
    if source.get("split") != schedule["split"]:
        raise ValueError("cached physical development split differs")
    if source.get("selector") != cell["selector"] or source.get("selected_count") != cell["selected_count"]:
        raise ValueError("cached physical selector/count differs")
    if source.get("retained_visual_tokens") != cell["selected_count"]:
        raise ValueError("cached physical retained count differs")
    if source.get("status") != "ok":
        raise ValueError("cached physical output was not successful")
    for owner in (schedule, geometry, cell):
        for field in IDENTITIES:
            if field in owner and field in source and source[field] != owner[field]:
                raise ValueError("cached physical identity differs: " + field)
    for field in ("quality_metric", "category"):
        if field in geometry and geometry.get(field) != source.get(field):
            raise ValueError("cached physical native metric identity differs: " + field)


def _index_hash(source, n, count):
    indices = source.get("selected_patch_indices")
    if (not isinstance(indices, list) or len(indices) != count or
            any(type(k) is not int or not 0 <= k < n for k in indices) or len(set(indices)) != count):
        raise ValueError("cached selected patch indices must be unique, valid, and exact-count")
    if indices != sorted(indices):
        raise ValueError("cached selected patches did not preserve original positional order")
    value = digest(indices)
    if source.get("selected_patch_indices_sha256", value) != value:
        raise ValueError("cached selected patch index hash differs")
    return value


def _feature_hash(schedule, request, geometry, selector, count, feature_curves):
    if feature_curves is None or request not in feature_curves:
        return None
    feature = feature_curves[request]
    if feature.get("request_id", request) != request or feature.get("input_visual_tokens") != geometry["input_visual_tokens"]:
        raise ValueError("feature curve request or visual geometry differs")
    for field, expected in (("model_tag", schedule["model"]), ("split", schedule["split"]),
                            ("checkpoint_sha256", schedule["checkpoint_sha256"]),
                            ("group_id", geometry["group_id"]), ("input_text_tokens", geometry["input_text_tokens"])):
        if field in feature and feature[field] != expected:
            raise ValueError("feature curve identity differs: " + field)
    if "dataset" in feature and canonical_dataset(feature["dataset"]) != canonical_dataset(geometry["dataset"]):
        raise ValueError("feature curve dataset differs")
    if feature.get("answers_loaded", False) is not False or feature.get("answers_consumed_by_feature_extractor", False) is not False:
        raise ValueError("feature curve consumed answers")
    if selector == "full":
        return digest(list(range(geometry["input_visual_tokens"])))
    curve = feature.get("curves", {}).get(selector)
    if curve is None:
        return None
    expected = curve.get("selected_indices_sha256", {}).get(str(count))
    if expected is None:
        return None  # Arbitrary integer counts need not be on the feature grid.
    if curve.get("input_visual_tokens") != geometry["input_visual_tokens"] or curve.get("selector", selector) != selector:
        raise ValueError("selector curve geometry differs")
    if curve.get("physical_prefix_parity") is not True or curve.get("valid") is not True:
        raise ValueError("selector curve lacks the physical-prefix audit")
    return expected


def split_missing(schedule, index, feature_curves=None):
    """Return sealed (missing_schedule, receipt), without changing any input.

    index maps (request, selector, exact_integer_count) to an audited physical
    row. feature_curves optionally maps request IDs to v11 feature records.
    Receipt cells are addressed by ``request + '.' + cell_key``. A cell is
    ``cached_physical`` (source path/hash), ``execute``, or ``alias`` of another
    cell in that same request. Missing exact counts are sent to GPU inference.
    An existing but invalid cache entry raises; it is never silently reused.
    """
    _checked_schedule(schedule)
    missing_rows, receipts, source_pins = {}, {}, {}
    totals = {"requested_cells": 0, "unique_physical_cells": 0, "cached_unique_cells": 0,
              "missing_unique_cells": 0, "alias_cells": 0}
    for request, geometry in sorted(schedule["rows"].items()):
        n, t = geometry.get("input_visual_tokens"), geometry.get("input_text_tokens")
        if (type(n) is not int or n < 1 or type(t) is not int or t < 0 or
                not geometry.get("group_id") or not isinstance(geometry.get("cells"), list)):
            raise ValueError("invalid scheduled native geometry")
        keys, seen, missing = set(), {}, []
        for cell in sorted(geometry["cells"], key=lambda c: c.get("key", "")):
            if any(field in owner for owner in (geometry, cell)
                   for field in ("answer", "answers", "quality_score", "quality_by_count", "prediction", "references")):
                raise ValueError("answer labels are forbidden in an execution schedule")
            key, count, selector = cell.get("key"), cell.get("selected_count"), cell.get("selector")
            if not isinstance(key, str) or not key or key in keys:
                raise ValueError("unique nonempty scheduled cell keys required")
            keys.add(key)
            if type(count) is not int or not 1 <= count <= n or selector not in ("full", "scorer", "divprune", "cdpruner"):
                raise ValueError("invalid exact physical cell")
            if (selector == "full") != (count == n):
                raise ValueError("full-count cells must use the structural full selector")
            if selector == "cdpruner" and schedule["model"] != "llava_next":
                raise ValueError("Qwen CDPruner has no audited implementation")
            identity, address = (request, selector, count), request + "." + key
            if address in receipts:
                raise ValueError("ambiguous receipt cell address")
            base = {"request_id": request, "cell_key": key, "selector": selector, "selected_count": count}
            totals["requested_cells"] += 1
            if identity in seen:
                primary, primary_cell = seen[identity]
                for field in IDENTITIES:
                    if cell.get(field) != primary_cell.get(field):
                        raise ValueError("alias cell changed runtime identity: " + field)
                receipts[address] = {**base, "kind": "alias", "alias_of": primary}
                totals["alias_cells"] += 1
                continue
            seen[identity] = address, cell
            totals["unique_physical_cells"] += 1
            source = index.get(identity)
            if source is None:
                missing.append(copy.deepcopy(cell))
                receipts[address] = {**base, "kind": "execute"}
                totals["missing_unique_cells"] += 1
                continue
            _check_geometry(schedule, request, geometry, cell, source)
            source_path, sha = source.get("physical_source_path"), source.get("physical_source_sha256")
            if not isinstance(source_path, str) or not isinstance(sha, str):
                raise ValueError("cached source path/hash unavailable; cannot reuse")
            # For a measured row with a source pointer, validate the file too;
            # the report helper otherwise trusts True rows as caller-audited.
            _verify_physical_provenance({**source, "physical_measurement_performed": False})
            actual_hash = _index_hash(source, n, count)
            expected_hash = _feature_hash(schedule, request, geometry, selector, count, feature_curves)
            if expected_hash is not None and actual_hash != expected_hash:
                raise ValueError("cached selected patches differ from the frozen selector curve")
            source_path = str(Path(source_path).resolve())
            if source_path in source_pins and source_pins[source_path] != sha:
                raise ValueError("one original source has conflicting content hashes")
            source_pins[source_path] = sha
            receipts[address] = {**base, "kind": "cached_physical", "physical_source_path": source_path,
                "physical_source_sha256": sha, "selected_patch_indices_sha256": actual_hash,
                "feature_prefix_checked": expected_hash is not None,
                "source_was_reuse_view": source.get("physical_measurement_performed") is not True,
                "inline_identity_fields_checked": sorted(set(source) & set(IDENTITIES))}
            totals["cached_unique_cells"] += 1
        if missing:
            missing_rows[request] = {**copy.deepcopy(geometry), "cells": missing}
    missing = seal({**{k: copy.deepcopy(v) for k, v in schedule.items() if k != "sha256"},
        "rows": missing_rows, "original_schedule_sha256": schedule["sha256"],
        "physical_reuse_audited": True, "deduplicated_same_request_cells": True})
    receipt = seal({"format": FORMAT, "original_schedule_sha256": schedule["sha256"],
        "missing_schedule_sha256": missing["sha256"], "model": schedule["model"], "split": schedule["split"],
        "checkpoint_sha256": schedule["checkpoint_sha256"], "manifest_sha256": schedule["manifest_sha256"],
        "cells": receipts, "counts": totals, "source_pins": source_pins,
        "feature_curves_sha256": digest(feature_curves) if feature_curves is not None else None,
        "input_index_contract": "caller supplied a previously audited pinned development model/checkpoint/manifest index",
        "source_hashes": {str(Path(__file__).resolve()): file_digest(__file__)},
        "new_inference_performed": False, "interpolation_used": False})
    return missing, receipt
