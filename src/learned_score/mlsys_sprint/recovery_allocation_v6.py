"""Fit-only removal-harm recovery, without changing frozen v4/v5 modules.

Three existing answer-free features produce one ridge-regression scalar. Its
supervision is positive removal harm, not the old lower-to-upper endpoint
slope. The downstream signed physical-response curves and runtime wire formats
remain v4-compatible. Version/source metadata belong to the outer v6 wrapper.

The CPU admission diagnostic measures held-out scalar-target MSE, not measured
allocation quality. A passing diagnostic still requires exact-budget physical
inference and informative shuffled controls before any allocation claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
import json
from pathlib import Path

import numpy as np

from . import adaptive_allocation_v4 as original
from . import adaptive_utility_v3 as utility
from .state import file_digest


FORMAT = "mlsys_recovery_allocation_v6"
TARGET = "mean positive removal quality loss per actual1percentagepoint at distinct-5,-10probes"
REMOVAL_OFFSETS = (-5, -10)
REQUIRED_MODELS = ("llava_next", "qwen25")
DEFAULT_TARGETS = (60, 70)
_EXPANDED_FIELDS = original._FIT | {"benefit_features"}


def sources():
    """Pin every module implementing scalar fitting, curves, and group folds."""
    return {str(Path(module.__file__).resolve()): file_digest(module.__file__)
            for module in (original, utility)} | {str(Path(__file__).resolve()): file_digest(__file__)}


def _identity(identity):
    if identity is None:
        return {}
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be an object")
    value = json.loads(original._canonical(dict(identity)))
    if value.get("split", "fit") != "fit":
        raise ValueError("recovery fitting accepts fit-only labels")
    return value


def _clean_rows(rows):
    if not isinstance(rows, (list, tuple)) or not rows:
        raise ValueError("removal-harm fitting requires nonempty rows")
    if any(not isinstance(row, Mapping) or set(row) != _EXPANDED_FIELDS for row in rows):
        raise ValueError("fit rows require physical fields and exactly three declared benefit features")
    physical = original._fit_rows([{key: row[key] for key in original._FIT} for row in rows])
    feature_index = {}
    for row in rows:
        features = row["benefit_features"]
        if not isinstance(features, Mapping) or set(features) != set(original.STATISTICS):
            raise ValueError("benefit_features must name exactly the three frozen statistics")
        feature_index[row["row_hash"]] = {
            key: original._finite(features[key], key) for key in original.STATISTICS}
    return [{**{key: row[key] for key in original._FIT},
             "benefit_features": feature_index[row["row_hash"]]} for row in physical]


def _removal_observation(row):
    n, uniform = row["input_visual_tokens"], row["uniform_selected_count"]
    probes = original.local_probe_counts(n, uniform)
    counts = sorted({probes[offset] for offset in REMOVAL_OFFSETS if probes[offset] < uniform})
    baseline = row["quality_by_count"][str(uniform)]
    measured = []
    for count in counts:
        removed_pp = 100.0 * (uniform - count) / n
        loss = max(0.0, baseline - row["quality_by_count"][str(count)])
        measured.append({"selected_count": count, "removed_tokens": uniform - count,
                         "actual_removed_pp": removed_pp, "positive_quality_loss": loss,
                         "loss_per_actual_pp": loss / removed_pp})
    return {"target": float(np.mean([item["loss_per_actual_pp"] for item in measured])) if measured else None,
            "measured_removals": measured,
            "exclusion_reason": None if measured else "no_distinct_physically_measured_lower_count"}


def removal_harm_target(row):
    """Mean positive loss/actual removed pp; return None, not zero, if undefined."""
    if not isinstance(row, Mapping) or set(row) not in (original._FIT, _EXPANDED_FIELDS):
        raise ValueError("target requires only declared physical fit fields")
    physical = original._fit_rows([{key: row[key] for key in original._FIT}])[0]
    return _removal_observation(physical)["target"]


def _weights(rows):
    """Equal benchmark weight and equal image-group weight within benchmark."""
    if not rows:
        return np.empty(0, dtype=float)
    group_counts = Counter(row["group_id"] for row in rows)
    domain_groups = {dataset: len({row["group_id"] for row in rows if row["dataset"] == dataset})
                     for dataset in {row["dataset"] for row in rows}}
    return np.asarray([1.0 / (len(domain_groups) * domain_groups[row["dataset"]]
                             * group_counts[row["group_id"]]) for row in rows])


def fit_removal_harm_scalar(rows, *, identity=None, ridge=10.0):
    """Three features -> one unclamped scalar, using the v4 scalar wire format.

    Requests with no removable measured count are omitted from regression and
    its normalization/weights, but cross-fitting still predicts their scalar
    for the downstream signed-response calibration.
    """
    checked = _clean_rows(rows)
    identity = _identity(identity)
    ridge = original._finite(ridge, "ridge")
    if ridge <= 0:
        raise ValueError("ridge must be positive")
    labeled = [(row, _removal_observation(row)["target"]) for row in checked]
    labeled = [(row, target) for row, target in labeled if target is not None]
    if not labeled:
        raise ValueError("no physically measured lower counts are available for scalar regression")
    eligible = [row for row, _ in labeled]
    x = np.asarray([[row["benefit_features"][key] for key in original.STATISTICS] for row in eligible])
    y = np.asarray([target for _, target in labeled])
    weights = _weights(eligible)
    mean = np.average(x, axis=0, weights=weights)
    scale = np.sqrt(np.average((x - mean) ** 2, axis=0, weights=weights)).clip(1e-8)
    z = (x - mean) / scale
    intercept = float(np.average(y, weights=weights))
    fitting_weights = weights * len(eligible)
    coefficients = np.linalg.solve(z.T @ (fitting_weights[:, None] * z) + ridge * np.eye(3),
                                   z.T @ (fitting_weights * (y - intercept)))
    return original._seal({
        "format": original.FORMAT + "_benefit_scalar", "identity": identity,
        "feature_names": list(original.STATISTICS), "mean": mean.tolist(), "scale": scale.tolist(),
        "coefficients": coefficients.tolist(), "intercept": intercept, "ridge": ridge, "target": TARGET,
        "runtime_inputs": "three frozen statistics; one scalar output, not a one-statistic rule",
        "calibration_rows": [row["row_hash"] for row in eligible],
        "calibration_groups": sorted({row["group_id"] for row in eligible}),
        "calibration_data_sha256": original._digest({"rows": eligible, "targets": y.tolist()})})


def predict_recovery(model, features):
    """Validate the supervision marker, then use unchanged v4 inference."""
    if not isinstance(model, Mapping) or model.get("target") != TARGET:
        raise ValueError("not a removal-harm scalar")
    return original.predict_benefit_scalar(model, features)


def crossfit_removal_harm_scalar(rows, *, identity=None, ridge=10.0):
    checked, identity = _clean_rows(rows), _identity(identity)
    folds = utility.grouped_folds(checked)
    predictions, members = {}, []
    for fold in range(5):
        training = [row for row in checked if folds[row["group_id"]] != fold]
        model = fit_removal_harm_scalar(training, identity={**identity, "excluded_fold": fold}, ridge=ridge)
        members.append(model)
        for row in checked:
            if folds[row["group_id"]] == fold:
                predictions[row["row_hash"]] = predict_recovery(model, row["benefit_features"])
    model = fit_removal_harm_scalar(checked, identity={**identity, "excluded_fold": None}, ridge=ridge)
    return original._seal({
        "format": original.FORMAT + "_benefit_crossfit", "model": model, "members": members,
        "out_of_fold_values": predictions, "group_folds": folds, "training_only": True,
        "policy_training_instruction": "replace statistic_value with out_of_fold_values before fit_policy"})


def _summary(records):
    eligible = [row for row in records if row["target"] is not None]
    base = {"rows": len(records), "eligible_rows": len(eligible), "excluded_no_lower_count_rows": len(records) - len(eligible),
            "groups": len({row["group_id"] for row in eligible}),
            "positive_harm_rows": sum(row["target"] > 0 for row in eligible),
            "positive_harm_groups": len({row["group_id"] for row in eligible if row["target"] > 0})}
    if not eligible:
        return {**base, "mse": None, "constant_mse": None, "mse_reduction": None,
                "relative_mse_reduction": None, "improves_constant": False}
    weights = _weights(eligible)
    targets = np.asarray([row["target"] for row in eligible])
    predicted = np.asarray([row["prediction"] for row in eligible])
    constant = np.asarray([row["constant_prediction"] for row in eligible])
    mse = float(np.average((targets - predicted) ** 2, weights=weights))
    constant_mse = float(np.average((targets - constant) ** 2, weights=weights))
    reduction = constant_mse - mse
    return {**base, "mse": mse, "constant_mse": constant_mse, "mse_reduction": reduction,
            "relative_mse_reduction": reduction / constant_mse if constant_mse > 0 else None,
            "improves_constant": reduction > max(1e-15, constant_mse * 1e-12)}


def scalar_diagnostics(rows, crossfit):
    """Evaluate held-out removal targets against training-fold constants only."""
    checked = _clean_rows(rows)
    crossfit = original._verify(crossfit)
    if crossfit.get("format") != original.FORMAT + "_benefit_crossfit":
        raise ValueError("wrong cross-fit format")
    folds = utility.grouped_folds(checked)
    if crossfit["group_folds"] != folds or set(crossfit["out_of_fold_values"]) != {row["row_hash"] for row in checked}:
        raise ValueError("cross-fit request/group identities differ")
    records = []
    for row in checked:
        fold = folds[row["group_id"]]
        model = original._verify(crossfit["members"][fold])
        if model.get("target") != TARGET or row["group_id"] in model["calibration_groups"]:
            raise ValueError("diagnostic prediction is not held out from scalar training")
        prediction = predict_recovery(model, row["benefit_features"])
        if prediction != crossfit["out_of_fold_values"][row["row_hash"]]:
            raise ValueError("stored out-of-fold prediction differs from its member")
        records.append({"row_hash": row["row_hash"], "group_id": row["group_id"], "dataset": row["dataset"],
                        "fold": fold, "input_visual_tokens": row["input_visual_tokens"],
                        "uniform_selected_count": row["uniform_selected_count"],
                        **_removal_observation(row), "prediction": prediction,
                        "constant_prediction": model["intercept"]})
    return original._seal({
        "format": FORMAT + "_scalar_diagnostics", "target": TARGET,
        "scope": "fit-only grouped out-of-fold scalar MSE; not physical allocation quality",
        "weighting": "equal benchmark weight; equal image-group weight within benchmark",
        "constant_baseline": "eligible training-fold weighted mean target; no held-out labels",
        "overall": _summary(records),
        "by_fold": {str(fold): _summary([row for row in records if row["fold"] == fold]) for fold in range(5)},
        "by_benchmark": {dataset: _summary([row for row in records if row["dataset"] == dataset])
                         for dataset in sorted({row["dataset"] for row in records})},
        "by_fold_and_benchmark": {str(fold): {
            dataset: _summary([row for row in records if row["fold"] == fold and row["dataset"] == dataset])
            for dataset in sorted({row["dataset"] for row in records})} for fold in range(5)},
        "rows": records})


def fit_recovery(rows, *, identity=None, ridge=10.0, min_groups=20, shrinkage=30.0):
    """Fit a fully nested v4-compatible scalar and signed response policy.

    Source/version tags are sealed outside the unchanged v4 model/policy wire
    schemas. Both the all-fit model and each outer utility member are trained
    without using their prediction groups' labels through any feature path.
    """
    checked, identity = _clean_rows(rows), _identity(identity)
    crossfit = crossfit_removal_harm_scalar(checked, identity=identity, ridge=ridge)

    def scalar_rows(source, fitted):
        return [{**{key: row[key] for key in original._FIT},
                 "statistic_value": fitted["out_of_fold_values"][row["row_hash"]]} for row in source]

    policy = original.fit_policy(scalar_rows(checked, crossfit), statistic="learned_benefit_scalar",
                                 identity=identity, min_groups=min_groups, shrinkage=shrinkage)
    members = []
    for fold in range(5):
        source = [row for row in checked if crossfit["group_folds"][row["group_id"]] != fold]
        nested = crossfit_removal_harm_scalar(source, identity={**identity, "utility_excluded_fold": fold}, ridge=ridge)
        member = original._fit_member(original._fit_rows(scalar_rows(source, nested)), min_groups, shrinkage)
        member.update(excluded_fold=fold, scalar_training_groups=sorted({row["group_id"] for row in source}),
                      scalar_crossfit_sha256=nested["sha256"])
        members.append(member)
    policy = {key: value for key, value in policy.items() if key != "policy_sha256"}
    policy.update(members=members, learned_scalar_crossfit="nested within each excluded-group utility member",
                  all_fit_scalar_crossfit_sha256=crossfit["sha256"])
    policy = original._seal(policy, "policy_sha256")
    original.load_policy(policy)
    return original._seal({"format": FORMAT + "_fit", "identity": identity, "source_sha256": sources(),
                           "target": TARGET, "crossfit": crossfit, "policy": policy,
                           "diagnostics": scalar_diagnostics(checked, crossfit)})


def admission_report(fits_by_model, *, required_models=REQUIRED_MODELS, required_targets=DEFAULT_TARGETS):
    """Require strictly better OOF MSE than constants for each required model.

    Each model is averaged equally over its declared targets. Target coverage
    must agree across models; missing models/targets never count as passing.
    This operational CPU gate is not a statistical or physical-quality claim.
    """
    if not isinstance(fits_by_model, Mapping):
        raise ValueError("admission requires model-to-fit-results mapping")
    if not required_models or len(set(required_models)) != len(required_models):
        raise ValueError("required models must be distinct and nonempty")
    if (not required_targets or len(set(required_targets)) != len(required_targets)
            or any(type(target) is not int or target not in original.TARGETS for target in required_targets)):
        raise ValueError("required targets must be distinct declared compression targets")
    summaries, coverage = {}, {}
    for model in required_models:
        results = fits_by_model.get(model, [])
        if not isinstance(results, (list, tuple)):
            raise ValueError("each model must contain a list of target fit results")
        per_target = {}
        for result in results:
            result = original._verify(result)
            if result.get("format") != FORMAT + "_fit" or result.get("target") != TARGET:
                raise ValueError("admission received a non-recovery fit")
            if result["identity"].get("model") != model:
                raise ValueError("admission model identity differs")
            target = original._integer(result["identity"].get("target"), "target", 1)
            if target in per_target:
                raise ValueError("duplicate admission target")
            diagnostic = original._verify(result["diagnostics"])
            per_target[target] = diagnostic["overall"]
        coverage[model] = sorted(per_target)
        valid = bool(per_target) and all(value["mse"] is not None for value in per_target.values())
        mse = float(np.mean([value["mse"] for value in per_target.values()])) if valid else None
        baseline = float(np.mean([value["constant_mse"] for value in per_target.values()])) if valid else None
        reduction = baseline - mse if valid else None
        summaries[model] = {"targets": per_target, "mean_mse": mse, "mean_constant_mse": baseline,
                            "mean_mse_reduction": reduction,
                            "improves_constant": valid and reduction > max(1e-15, baseline * 1e-12)}
    matched = all(values == sorted(required_targets) for values in coverage.values())
    passed = matched and all(value["improves_constant"] for value in summaries.values())
    return original._seal({"format": FORMAT + "_cpu_admission", "target": TARGET,
                           "required_models": list(required_models), "required_targets": list(required_targets),
                           "matched_target_coverage": matched,
                           "models": summaries, "passed": passed,
                           "decision": "admit_exact_budget_physical_screening" if passed else "do_not_expand_gpu_allocation",
                           "criterion": "each required model improves grouped/domain-weighted OOF scalar MSE over training-fold constants; equal target weight",
                           "limitations": ["fit-only operational gate, not fresh confirmation",
                                           "does not estimate exact-budget allocation quality",
                                           "no interpolation is represented as measured physical quality"]})


def main(argv=None):
    """CPU-only CLI: consume validated v4 fit data; write one immutable bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=REQUIRED_MODELS, default=list(REQUIRED_MODELS))
    parser.add_argument("--targets", nargs="+", type=int, choices=original.TARGETS, default=list(DEFAULT_TARGETS))
    args = parser.parse_args(argv)
    if len(set(args.models)) != len(args.models) or len(set(args.targets)) != len(args.targets):
        parser.error("models and targets must not contain duplicates")
    from . import adaptive_planning_v4 as planning
    from .followup_common import write_immutable

    study = planning._checked(planning._read(args.study))
    fits = {}
    for model in args.models:
        _, metadata, features, schedule, qualities, physical_sha = planning._physical_fit(study, model)
        fits[model] = []
        for target in args.targets:
            rows = planning._fit_rows(features, schedule, qualities, target, "normalized_entropy")
            expanded = [{**row, "benefit_features": {
                statistic: original.statistic_from_features(features[row["row_hash"]], statistic, row["uniform_selected_count"])
                for statistic in original.STATISTICS}} for row in rows]
            identity = {"model": model, "target": target, "split": "fit", "study_sha256": study["sha256"],
                        "manifest_sha256": study["manifest_sha256"], "checkpoint_sha256": metadata["checkpoint_sha256"],
                        "physical_rows_sha256": physical_sha}
            fits[model].append(fit_recovery(expanded, identity=identity))
    admission = admission_report(fits)
    bundle = original._seal({"format": FORMAT + "_bundle", "source_sha256": sources(),
                             "study_path": str(args.study.resolve()), "study_file_sha256": file_digest(args.study),
                             "fit_loader_source_sha256": {str(Path(planning.__file__).resolve()): file_digest(planning.__file__)},
                             "fits": fits, "admission": admission})
    write_immutable(args.output, bundle)
    print(json.dumps({"output": str(args.output), "passed": admission["passed"], "decision": admission["decision"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
