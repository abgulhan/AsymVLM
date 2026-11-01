"""CPU publication of local measurements, policies and physical schedules.

Fit opens fit features and physically measured local fit responses only.
Screening/evaluation schedules open answer-free features and frozen policies,
never screening/evaluation answer qualities. All publications are immutable.
"""

from collections import defaultdict
import json
import math
from pathlib import Path

from . import adaptive_allocation_v4 as allocation
from .adaptive_data_v4 import load_manifest
from .followup_common import write_immutable
from .state import digest, file_digest


FORMAT = "mlsys_adaptive_planning_v4"
SEEDS = (42, 43, 44)


class IncompleteStudyError(RuntimeError):
    """A complete required physical stage has not arrived yet."""


def _read(path):
    path = Path(path)
    if not path.is_file():
        raise IncompleteStudyError(f"missing required artifact: {path}")
    return json.loads(path.read_text())


def _seal(value):
    return {**value, "sha256": digest(value)}


def _checked(value):
    if not isinstance(value, dict) or value.get("sha256") != digest({k: v for k, v in value.items() if k != "sha256"}):
        raise ValueError("publication content hash mismatch")
    return value


def _sources():
    return {str(Path(__file__).resolve()): file_digest(__file__),
            str(Path(allocation.__file__).resolve()): file_digest(allocation.__file__)}


def _context(study, model):
    _checked(study)
    if model not in study["models"]:
        raise ValueError("unknown study model")
    manifest = load_manifest(study["manifest"])
    if manifest.file_sha256 != study["manifest_sha256"]:
        raise ValueError("study manifest has changed")
    output = Path(study["output_root"]) / model
    checkpoint = file_digest(study["models"][model]["checkpoint"])
    return manifest, output, checkpoint


def _metadata(directory, manifest, entries, split, checkpoint, *, schedule=None):
    directory = Path(directory)
    metadata = _read(directory / "rank0.metadata.json")
    expected = {"format": "mlsys_adaptive_experiment_v4", "manifest_sha256": manifest.file_sha256,
                "checkpoint_sha256": checkpoint, "split": split, "limit": None,
                "phase": "physical" if schedule else "features", "max_new_tokens": 128, "generated_budget": 512}
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise ValueError("feature/physical metadata differs from the new v4 study")
    if schedule and metadata.get("schedule_content_sha256") != schedule["sha256"]:
        raise ValueError("physical stage used a different schedule")
    world = metadata.get("world_size")
    if type(world) is not int or world < 1:
        raise ValueError("invalid physical shard count")
    if len(list(directory.glob("rank*.metadata.json"))) != world:
        raise IncompleteStudyError("metadata does not yet cover all shards")
    ordered = sorted(entries, key=lambda row: (row["dataset"], row["row_hash"]))
    if schedule:
        ordered = [entry for entry in ordered if entry["row_hash"] in schedule["rows"]]
    for rank in range(world):
        if _read(directory / f"rank{rank}.metadata.json") != metadata:
            raise ValueError("mixed runtime metadata across shards")
        marker = _read(directory / f"rank{rank}.complete.json")
        assigned = ordered[rank::world]
        results = sum(len(schedule["rows"][e["row_hash"]]["cells"]) for e in assigned) if schedule else len(assigned)
        if marker.get("identity_sha256") != digest(metadata) or marker.get("requests") != len(assigned) or marker.get("results") != results:
            raise ValueError("completion marker does not cover its exact assigned work")
    return metadata


def _features(study, model, split):
    manifest, output, checkpoint = _context(study, model)
    directory = output / f"{split}_features"
    entries = manifest.splits[split]
    metadata = _metadata(directory, manifest, entries, split, checkpoint)
    contract = metadata["cache_contract"]
    if (contract.get("reserved_generated_slots") != 127 or contract.get("generated_budget") != 512
            or contract.get("scratch_bytes") != 0 or contract.get("compaction_possible") is not False
            or type(contract.get("total_owned_bytes_per_slot")) is not int or contract["total_owned_bytes_per_slot"] < 1):
        raise ValueError("unsupported protected-prefix allocated-cache contract")
    expected_ids = {entry["row_hash"] for entry in entries}
    found = {path.stem for path in (directory / "features").glob("*.json")}
    if found - expected_ids:
        raise ValueError("unknown feature identities")
    if found != expected_ids:
        raise IncompleteStudyError("not every manifest request has features")
    fields = {"format", "measurement_version", "request_id", "group_id", "image_sha256", "dataset", "split",
              "identity_sha256", "checkpoint_sha256", "cache_contract", "answers_consumed_by_feature_extractor",
              "input_visual_tokens", "input_text_tokens", "normalized_entropy", "nearest_neighbor_diversity", "top_mass_by_percent"}
    features = {}
    for entry in entries:
        row = _read(directory / "features" / f"{entry['row_hash']}.json")
        expected = {"format": "mlsys_answer_free_features_v3", "measurement_version": "v4",
                    "request_id": entry["row_hash"], "group_id": entry["group_id"], "image_sha256": entry["image_sha256"],
                    "dataset": entry["dataset"], "split": split, "identity_sha256": digest(metadata),
                    "checkpoint_sha256": checkpoint, "cache_contract": contract, "answers_consumed_by_feature_extractor": False}
        if set(row) != fields or any(row.get(key) != value for key, value in expected.items()):
            raise ValueError("feature identity differs or contains unexpected/label-bearing fields")
        for key, minimum in (("input_visual_tokens", 1), ("input_text_tokens", 0)):
            if type(row[key]) is not int or row[key] < minimum:
                raise ValueError("invalid measured token geometry")
        for statistic in allocation.STATISTICS:
            allocation.statistic_from_features(row, statistic, max(1, row["input_visual_tokens"] // 2))
        features[entry["row_hash"]] = row
    return manifest, output, checkpoint, metadata, features


def _geometry(feature):
    return {"row_hash": feature["request_id"], **{key: feature[key] for key in
            ("group_id", "dataset", "input_visual_tokens", "input_text_tokens")}}


def _budgets(features, targets, contract):
    groups = defaultdict(list)
    for feature in features.values():
        groups[feature["dataset"]].append(_geometry(feature))
    budgets, infeasible = {}, []
    for target in targets:
        budgets[str(target)] = {}
        for dataset, rows in sorted(groups.items()):
            plan = allocation.uniform_budget(rows, target, reserved_generated_slots=contract["reserved_generated_slots"])
            budgets[str(target)][dataset] = plan
            if not plan["feasible"]:
                infeasible.append({"target": target, "dataset": dataset, "reason": plan["reason"],
                                   "target_visual_tokens": plan["target_visual_tokens"], "requests": len(rows)})
    return budgets, infeasible


def _schedule_base(study, model, split, metadata, features, budgets, infeasible, *, stage):
    return {"format": FORMAT + "_schedule", "stage": stage, "model": model, "split": split,
            "manifest_sha256": study["manifest_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
            "cache_contract": metadata["cache_contract"], "source_sha256": _sources(), "study_sha256": study["sha256"],
            "features_identity_sha256": digest(metadata), "features_sha256": digest(features),
            "budgets": budgets, "infeasible_panels": infeasible, "rows": {},
            "evidence_status": "development; physical inference required; no evaluation quality consumed"}


def build_local_schedule(study, model):
    _, output, _, metadata, features = _features(study, model, "fit")
    budgets, infeasible = _budgets(features, study["kv_reduction_percent"], metadata["cache_contract"])
    schedule = _schedule_base(study, model, "fit", metadata, features, budgets, infeasible, stage="local_calibration")
    for request, feature in sorted(features.items()):
        by_count, probes = defaultdict(list), {}
        for target in study["kv_reduction_percent"]:
            budget = budgets[str(target)][feature["dataset"]]
            if not budget["feasible"]:
                continue
            uniform = budget["counts"][request]
            counts = allocation.local_probe_counts(feature["input_visual_tokens"], uniform)
            probes[str(target)] = {"uniform_selected_count": uniform, "counts_by_offset": {str(k): v for k, v in counts.items()}}
            for offset, count in counts.items():
                by_count[count].append({"target": target, "offset_percent": offset})
        if by_count:
            schedule["rows"][request] = {**_geometry(feature), "local_probes": probes,
                "cells": [{"key": f"local_k{count:06d}", "selector": "scorer", "selected_count": count,
                           "target": None, "policy": "local_physical_response", "mode": "local_calibration", "uses": uses}
                          for count, uses in sorted(by_count.items())]}
    path = output / "schedules/local_calibration.json"
    write_immutable(path, _seal(schedule))
    return path


def _physical_fit(study, model):
    manifest, output, checkpoint, metadata, features = _features(study, model, "fit")
    schedule = _checked(_read(output / "schedules/local_calibration.json"))
    if (schedule["features_sha256"] != digest(features) or schedule["manifest_sha256"] != manifest.file_sha256
            or schedule["checkpoint_sha256"] != checkpoint):
        raise ValueError("local schedule provenance changed")
    directory = output / "local_calibration"
    runtime = _metadata(directory, manifest, manifest.splits["fit"], "fit", checkpoint, schedule=schedule)
    if runtime["cache_contract"] != metadata["cache_contract"]:
        raise ValueError("physical and feature extraction cache contracts differ")
    expected_files = {f"{request}.{cell['key']}" for request, row in schedule["rows"].items() for cell in row["cells"]}
    actual_files = {path.stem for path in (directory / "rows").glob("*.json")}
    if actual_files - expected_files:
        raise ValueError("unexpected local physical result identities")
    if actual_files != expected_files:
        raise IncompleteStudyError("local physical response matrix is incomplete")
    qualities, record_hashes = {}, {}
    for request, planned in schedule["rows"].items():
        feature = features[request]
        quality = {}
        for cell in planned["cells"]:
            path = directory / "rows" / f"{request}.{cell['key']}.json"
            row = _read(path)
            expected = {"format": "mlsys_adaptive_experiment_v4", "identity_sha256": digest(runtime),
                        "request_id": request, "group_id": feature["group_id"], "image_sha256": feature["image_sha256"],
                        "dataset": feature["dataset"], "split": "fit", "phase": "physical", "status": "ok",
                        "selector": "scorer", "selected_count": cell["selected_count"], "schedule_cell": cell,
                        "schedule_content_sha256": schedule["sha256"], "max_new_tokens": 128, "generated_budget": 512,
                        "expected_cache_bytes": (cell["selected_count"] + feature["input_text_tokens"] + 127) * metadata["cache_contract"]["total_owned_bytes_per_slot"]}
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError("physical local response provenance/count differs")
            score = row.get("quality_score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("invalid local physical quality score")
            if row.get("stop_reason") not in ("eos_token", "max_new_tokens") or not row.get("generated_token_ids"):
                raise ValueError("local quality lacks corrected natural stopping evidence")
            quality[str(cell["selected_count"])] = score
            record_hashes[path.name] = digest(row)
        qualities[request] = quality
    return output, metadata, features, schedule, qualities, digest(record_hashes)


def _fit_rows(features, schedule, qualities, target, statistic):
    rows = []
    for request, feature in sorted(features.items()):
        probes = schedule["rows"].get(request, {}).get("local_probes", {}).get(str(target))
        if probes is None:
            continue
        uniform = probes["uniform_selected_count"]
        row = {**_geometry(feature), "uniform_selected_count": uniform,
               "statistic_value": allocation.statistic_from_features(feature, statistic, uniform),
               "quality_by_count": {str(count): qualities[request][str(count)] for count in sorted(set(probes["counts_by_offset"].values()))}}
        rows.append(row)
    return rows


def fit_candidates(study, model, learned=False):
    output = Path(study["output_root"]) / model
    index_path = output / "policies" / ("learned_fit_index.json" if learned else "fit_index.json")
    if index_path.exists():
        existing = _checked(_read(index_path))
        _, _, current_checkpoint = _context(study, model)
        if (existing["study_sha256"] != study["sha256"] or existing["source_sha256"] != _sources()
                or existing["checkpoint_sha256"] != current_checkpoint):
            raise ValueError("existing fitted policies have changed source/study provenance")
        for record in existing["policies"]:
            if file_digest(record["path"]) != record["file_sha256"]:
                raise ValueError("fitted policy file changed")
            if record.get("benefit_model_path") and file_digest(record["benefit_model_path"]) != record["benefit_model_file_sha256"]:
                raise ValueError("learned scalar file changed")
        return index_path
    output, metadata, features, schedule, qualities, physical_digest = _physical_fit(study, model)
    candidates = ("learned_benefit_scalar",) if learned else tuple(study["statistic_candidates"])
    datasets = sorted({feature["dataset"] for feature in features.values()})
    index = {"format": FORMAT + "_fit_index", "study_sha256": study["sha256"], "source_sha256": _sources(),
             "model": model, "manifest_sha256": study["manifest_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
             "local_schedule_sha256": schedule["sha256"], "physical_rows_sha256": physical_digest,
             "policies": [], "unavailable_policies": [], "learned": learned, "labels_consumed": "fit physical responses only"}
    for target in study["kv_reduction_percent"]:
        for candidate in candidates:
            source_statistic = "normalized_entropy" if learned else candidate
            base_rows = _fit_rows(features, schedule, qualities, target, source_statistic)
            for held_out in (None, *datasets):
                rows = [row for row in base_rows if row["dataset"] != held_out]
                mode = "all_fit" if held_out is None else "lobo"
                tag = "all_fit" if held_out is None else f"lobo_{digest(held_out)[:12]}"
                identity = {"model": model, "target": target, "candidate": candidate, "held_out_benchmark": held_out,
                            "manifest_sha256": study["manifest_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                            "physical_rows_sha256": physical_digest}
                if len({r["group_id"] for r in rows}) < 25:
                    index["unavailable_policies"].append({**identity, "reason": "fewer_than25_independent_fit_groups"})
                    continue
                benefit_path, benefit_file_sha = None, None
                if learned:
                    expanded = []
                    for row in rows:
                        feature = features[row["row_hash"]]
                        expanded.append({**row, "benefit_features": {stat: allocation.statistic_from_features(feature, stat, row["uniform_selected_count"])
                                                                    for stat in allocation.STATISTICS}})
                    nested = allocation.fit_crossfitted_benefit_policy(expanded, identity=identity, min_groups=20, shrinkage=30.)
                    crossfit, policy = nested["crossfit"], nested["policy"]
                    benefit_path = output / "policies" / candidate / f"target{target}_{tag}_benefit.json"
                    write_immutable(benefit_path, crossfit)
                    benefit_file_sha = file_digest(benefit_path)
                else:
                    policy = allocation.fit_policy(rows, statistic=candidate, identity=identity, min_groups=20, shrinkage=30.)
                path = output / "policies" / candidate / f"target{target}_{tag}.json"
                write_immutable(path, policy)
                index["policies"].append({"candidate": candidate, "target": target, "mode": mode, "held_out_benchmark": held_out,
                                          "path": str(path), "file_sha256": file_digest(path), "policy_sha256": policy["policy_sha256"],
                                          "benefit_model_path": str(benefit_path) if benefit_path else None,
                                          "benefit_model_file_sha256": benefit_file_sha})
    write_immutable(index_path, _seal(index))
    return index_path


def _load_policy(index, candidate, target, held_out=None):
    matches = [record for record in index["policies"] if record["candidate"] == candidate and record["target"] == target
               and record["held_out_benchmark"] == held_out]
    if len(matches) != 1:
        return None, None
    record = matches[0]
    if file_digest(record["path"]) != record["file_sha256"]:
        raise ValueError("frozen policy file changed")
    policy = allocation.load_policy(_read(record["path"]))
    benefit = None
    if record["benefit_model_path"]:
        if file_digest(record["benefit_model_path"]) != record["benefit_model_file_sha256"]:
            raise ValueError("frozen benefit scalar changed")
        benefit = _checked(_read(record["benefit_model_path"]))["model"]
    return policy, benefit


def _assign(policy, benefit, features, budget):
    rows = []
    for feature in features:
        uniform = budget["counts"][feature["request_id"]]
        if benefit:
            statistic = allocation.predict_benefit_scalar(benefit, {stat: allocation.statistic_from_features(feature, stat, uniform)
                                                                   for stat in allocation.STATISTICS})
        else:
            statistic = allocation.statistic_from_features(feature, policy["statistic"], uniform)
        rows.append({**_geometry(feature), "statistic_value": statistic})
    return allocation.allocate(policy, rows, uniform_counts=budget["counts"])


def _append(schedule, features, target, candidate, mode, assignment, *, seed=None, policy_sha256=None):
    suffix = f"shuffled{seed}" if mode == "shuffled" else mode
    key = f"t{target:03d}_{candidate}_{suffix}"
    for request, row in assignment["rows"].items():
        planned = schedule["rows"].setdefault(request, {**_geometry(features[request]), "cells": []})
        cell = {"key": key, "selector": "scorer", "selected_count": row["adaptive_selected_count"], "target": target,
                "policy": candidate, "mode": mode, "uniform_selected_count": row["uniform_selected_count"],
                "assignments_sha256": assignment["assignments_sha256"], "policy_sha256": policy_sha256,
                "support_failures": row.get("support_failures", []), "out_of_support": row.get("out_of_support", False),
                "fallback": row.get("fallback", False), "fallback_kind": row.get("fallback_kind"),
                "changed_from_uniform": row["changed_from_uniform"]}
        if mode == "shuffled":
            cell.update(shuffle_seed=seed, shuffle_movable_fraction=assignment["movable_fraction"],
                        shuffle_changed_fraction=assignment["changed_fraction"],
                        shuffle_informative=assignment["informative_control"] and assignment["movable_fraction"] >= .2,
                        source_adaptive_selected_count=row["source_adaptive_selected_count"])
        if mode == "lobo":
            cell.update(held_out_dataset=features[request]["dataset"],
                        transfer_scope=("predeclared transfer ablation; no heldout quality selection" if candidate in allocation.STATISTICS
                                        else "exploratory learned-scalar transfer; candidate branch selected using all-benchmark screening"))
        planned["cells"].append(cell)


def _build_policy_schedule(study, model, *, split, candidates, learned=False, evaluate=False):
    _, output, checkpoint, metadata, features = _features(study, model, split)
    index_name = "learned_fit_index.json" if learned else "fit_index.json"
    index = _checked(_read(output / "policies" / index_name))
    if (index["study_sha256"] != study["sha256"] or index["source_sha256"] != _sources()
            or index["checkpoint_sha256"] != checkpoint):
        raise ValueError("fitted policy index provenance differs")
    transfer_index = index
    if evaluate and learned:
        transfer_index = _checked(_read(output / "policies/fit_index.json"))
        if (transfer_index["study_sha256"] != study["sha256"] or transfer_index["source_sha256"] != _sources()
                or transfer_index["checkpoint_sha256"] != checkpoint):
            raise ValueError("predeclared transfer policy index provenance differs")
    budgets, infeasible = _budgets(features, study["kv_reduction_percent"], metadata["cache_contract"])
    stage = "evaluation" if evaluate else "benefit_screening" if learned else "screening"
    schedule = _schedule_base(study, model, split, metadata, features, budgets, infeasible, stage=stage)
    schedule["fit_index_sha256"] = index["sha256"]
    schedule["unavailable_panels"] = []
    by_dataset = defaultdict(list)
    for feature in features.values():
        by_dataset[feature["dataset"]].append(feature)
    if evaluate:
        for request, feature in sorted(features.items()):
            schedule["rows"][request] = {**_geometry(feature), "cells": [
                {"key": "full_all_tokens", "selector": "full", "selected_count": feature["input_visual_tokens"],
                 "target": 0, "policy": "full", "mode": "full"}]}
    for target in study["kv_reduction_percent"]:
        for dataset, cohort in sorted(by_dataset.items()):
            budget = budgets[str(target)][dataset]
            if not budget["feasible"]:
                continue
            for feature in cohort:
                request = feature["request_id"]
                planned = schedule["rows"].setdefault(request, {**_geometry(feature), "cells": []})
                uniform = budget["counts"][request]
                planned["cells"].append({"key": f"t{target:03d}_uniform", "selector": "scorer", "selected_count": uniform,
                                         "target": target, "policy": "uniform", "mode": "uniform"})
                if evaluate:
                    for selector in ("divprune", "cdpruner"):
                        planned["cells"].append({"key": f"t{target:03d}_{selector}_uniform", "selector": selector,
                                                 "selected_count": uniform, "target": target, "policy": f"{selector}_uniform", "mode": "uniform"})
            for candidate in candidates:
                policy, benefit = _load_policy(index, candidate, target)
                if policy is None:
                    schedule["unavailable_panels"].append({"target": target, "dataset": dataset, "candidate": candidate,
                                                            "mode": "raw", "reason": "no_fitted_policy"})
                    continue
                assignment = _assign(policy, benefit, cohort, budget)
                _append(schedule, features, target, candidate, "raw", assignment, policy_sha256=policy["policy_sha256"])
                for seed in study.get("shuffle_seeds", SEEDS):
                    shuffled = allocation.shuffle_assignments(assignment, seed=seed)
                    _append(schedule, features, target, candidate, "shuffled", shuffled, seed=seed, policy_sha256=policy["policy_sha256"])
            if evaluate:
                # All three rules were declared before inspecting any screening
                # answers. Selecting one rule globally using all benchmarks and
                # showing only its LOBO result would not establish completely
                # unseen-benchmark rule selection.
                transfer_candidates = (*allocation.STATISTICS, *(("learned_benefit_scalar",) if learned else ()))
                for candidate in transfer_candidates:
                    candidate_index = index if candidate == "learned_benefit_scalar" else transfer_index
                    held_policy, held_benefit = _load_policy(candidate_index, candidate, target, dataset)
                    if held_policy is None:
                        schedule["unavailable_panels"].append({"target": target, "dataset": dataset, "candidate": candidate,
                                                               "mode": "lobo", "reason": "no_benchmark_held_out_policy"})
                    else:
                        held_assignment = _assign(held_policy, held_benefit, cohort, budget)
                        _append(schedule, features, target, candidate, "lobo", held_assignment, policy_sha256=held_policy["policy_sha256"])
    path = output / "schedules" / f"{stage}.json"
    write_immutable(path, _seal(schedule))
    return path


def build_screening_schedule(study, model, learned=False):
    candidates = ("learned_benefit_scalar",) if learned else tuple(study["statistic_candidates"])
    return _build_policy_schedule(study, model, split="screening", candidates=candidates, learned=learned)


def build_evaluation_schedule(study, model, selected_candidate):
    if selected_candidate not in (*allocation.STATISTICS, "learned_benefit_scalar"):
        raise ValueError("unknown physically selected candidate")
    return _build_policy_schedule(study, model, split="evaluation", candidates=(selected_candidate,),
                                  learned=selected_candidate == "learned_benefit_scalar", evaluate=True)
