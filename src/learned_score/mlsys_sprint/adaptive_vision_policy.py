"""Frozen, development-only rules for choosing a per-input image-token count.

This module does not fit a policy or inspect evaluation answers.  A caller must
provide an explicitly calibrated artifact bound to the scorer and model being
used.  Difficulty is oriented so larger values always retain at least as many
tokens.  Out-of-support inputs keep every image token and are labeled as such;
there is no silent fixed-budget replacement.

``normalized_entropy`` uses the existing MLSys population-standardized score
formula, not raw-score softmax entropy.  ``raw_score_iqr`` is retained only for a
clearly marked legacy ablation.  Neither rule is a claim of improved quality.
"""

from bisect import bisect_left, bisect_right
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path
import re
from typing import Mapping

from learned_score.mlsys_next.entropy import FORMULA as ENTROPY_FORMULA


FORMAT = "mlsys_adaptive_vision_policy_v1"
FORMULAS = {
    "raw_score_iqr": "quantile(scores,0.75)-quantile(scores,0.25)",
    "normalized_entropy": ENTROPY_FORMULA,
}
DIRECTIONS = {
    "raw_score_iqr": "lower_is_harder",
    "normalized_entropy": "higher_is_harder",
}
DEVELOPMENT_SPLITS = frozenset({"scorer_dev", "scorer_screening", "development"})
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMON_FIELDS = frozenset({
    "format", "statistic", "statistic_formula", "form", "direction",
    "checkpoint_sha256", "scorer_variant", "selector", "model", "adapter",
    "calibration_manifest_sha256", "calibration_split", "calibration_row_hashes",
    "calibration_group_ids", "calibration_image_sha256", "calibration_data_sha256",
    "calibration_status", "support", "retentions",
})


def _string(value, name):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"{name} must be a nonempty, unpadded string")
    return value


def _hash(value, name):
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _numbers(value, name, *, minimum=1):
    if not isinstance(value, (list, tuple)) or len(value) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} values")
    return tuple(_number(item, name) for item in value)


def _identities(value, name, *, hashed):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{name} must contain calibration identities")
    validator = _hash if hashed else _string
    result = tuple(validator(item, name) for item in value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate identities")
    return result


def _strictly_increasing(values):
    return all(left < right for left, right in zip(values, values[1:]))


def _percent(value):
    """Avoid losing a token to binary float multiplication, e.g. 100 * .29."""
    return int((Decimal(str(value)) * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


def calculate_statistic(scores, statistic):
    """Return one answer-free statistic from a finite, one-dimensional vector.

    Inputs are never mutated and gradients are not retained.  Single-token
    inputs have IQR and entropy zero; they always retain their sole token.
    """
    import torch

    if statistic not in FORMULAS:
        raise ValueError(f"unsupported statistic: {statistic}")
    values = torch.as_tensor(scores).detach()
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("scores must be a nonempty one-dimensional vector")
    if values.is_complex() or values.dtype == torch.bool or not torch.isfinite(values).all():
        raise ValueError("scores must contain finite real numbers")
    values = values.float()
    if not torch.isfinite(values).all():
        raise ValueError("scores are outside the supported FP32 range")
    if values.numel() == 1:
        return 0.0
    if statistic == "raw_score_iqr":
        quantiles = torch.quantile(values, values.new_tensor([0.25, 0.75]))
        result = float(quantiles[1] - quantiles[0])
    else:
        from learned_score.mlsys_next.entropy import normalized_entropy

        # The historical helper clips its output.  Reject numeric overflow
        # before clipping can conceal a non-finite standardized distribution.
        mean, deviation = values.mean(), values.std(unbiased=False)
        standardized = (values - mean) / deviation.clamp_min(1e-8)
        if not torch.isfinite(mean) or not torch.isfinite(deviation) or not torch.isfinite(standardized).all():
            raise ValueError("non-finite standardized score distribution")
        result = normalized_entropy(values)
    if not math.isfinite(result):
        raise ValueError("statistic computation produced a non-finite value")
    return result


@dataclass(frozen=True)
class AdaptiveVisionDecision:
    input_count: int
    selected_count: int
    retention: float
    statistic_value: float
    oriented_value: float
    percentile: float | None
    fallback: bool
    fallback_reason: str | None
    legacy_only: bool

    def asdict(self):
        return asdict(self)


@dataclass(frozen=True, init=False)
class AdaptiveVisionPolicy:
    """Validated immutable policy; construct only with ``from_payload/load``."""

    statistic: str
    form: str
    direction: str
    checkpoint_sha256: str
    scorer_variant: str
    selector: str
    model: str
    adapter: str
    calibration_manifest_sha256: str
    calibration_split: str
    calibration_row_hashes: tuple[str, ...]
    calibration_group_ids: tuple[str, ...]
    calibration_image_sha256: tuple[str, ...]
    calibration_data_sha256: str
    support: tuple[float, float]
    retentions: tuple[float, ...]
    thresholds: tuple[float, ...]
    percentile_scores: tuple[float, ...]
    percentile_knots: tuple[float, ...]
    _payload_json: str = field(repr=False)

    @classmethod
    def from_payload(cls, payload):
        if not isinstance(payload, Mapping):
            raise ValueError("adaptive policy must be a JSON object")
        form = payload.get("form")
        if not isinstance(form, str) or form not in {"ordered_thresholds", "continuous"}:
            raise ValueError("unsupported adaptive policy form")
        form_fields = {"thresholds"} if form == "ordered_thresholds" else {
            "percentile_scores", "percentile_knots"}
        expected = _COMMON_FIELDS | form_fields
        missing, unexpected = expected - payload.keys(), payload.keys() - expected
        if missing or unexpected:
            raise ValueError(f"adaptive policy fields differ: missing={sorted(missing)}, unexpected={sorted(unexpected)}")
        if payload["format"] != FORMAT:
            raise ValueError("unsupported adaptive policy format")
        statistic = payload["statistic"]
        if not isinstance(statistic, str) or statistic not in FORMULAS or payload["statistic_formula"] != FORMULAS[statistic]:
            raise ValueError("unsupported statistic or mismatched statistic formula")
        if payload["direction"] != DIRECTIONS[statistic]:
            raise ValueError("statistic direction is fixed and cannot be reversed")
        if payload["calibration_status"] != "frozen":
            raise ValueError("adaptive calibration must be explicitly frozen")
        if not isinstance(payload["calibration_split"], str) or payload["calibration_split"] not in DEVELOPMENT_SPLITS:
            raise ValueError("calibration must use development data, never final evaluation")

        values = {"statistic": statistic, "form": form, "direction": payload["direction"],
                  "calibration_split": payload["calibration_split"]}
        for key in ("checkpoint_sha256", "calibration_manifest_sha256", "calibration_data_sha256"):
            values[key] = _hash(payload[key], key)
        for key in ("scorer_variant", "model", "adapter"):
            values[key] = _string(payload[key], key)
        selector = payload["selector"]
        if not isinstance(selector, str) or selector not in {"scorer", "scorer_coverage"}:
            raise ValueError("adaptive policy requires an explicitly bound learned selector")
        if selector == "scorer_coverage" and values["scorer_variant"] == "original":
            raise ValueError("coverage requires a rank32 scorer, not the original scorer")
        values["selector"] = selector
        for key in ("calibration_row_hashes", "calibration_group_ids", "calibration_image_sha256"):
            values[key] = _identities(payload[key], key, hashed=key != "calibration_group_ids")

        support = _numbers(payload["support"], "support", minimum=2)
        if len(support) != 2 or not support[0] < support[1]:
            raise ValueError("support must be an increasing [minimum, maximum] pair in oriented units")
        values["support"] = support
        retentions = _numbers(payload["retentions"], "retentions", minimum=2)
        if any(not 0 < retention <= 1 for retention in retentions):
            raise ValueError("retentions must lie in (0, 1]")
        if any(not math.isclose(100 * retention, _percent(retention), rel_tol=0, abs_tol=1e-9)
               for retention in retentions):
            raise ValueError("retentions must be integer-percent increments")
        if any(left > right for left, right in zip(retentions, retentions[1:])):
            raise ValueError("retentions must be nondecreasing with oriented difficulty")
        percentages = tuple(_percent(value) for value in retentions)
        if any(not 1 <= value <= 100 for value in percentages):
            raise ValueError("retentions must retain at least one percent")
        if len(set(percentages)) < 2:
            raise ValueError("a constant retention is a fixed baseline, not an adaptive policy")
        values["retentions"] = tuple(value / 100 for value in percentages)
        values.update(thresholds=(), percentile_scores=(), percentile_knots=())

        if form == "ordered_thresholds":
            thresholds = _numbers(payload["thresholds"], "thresholds")
            if not _strictly_increasing(thresholds):
                raise ValueError("thresholds must be strictly increasing")
            if any(not support[0] < threshold < support[1] for threshold in thresholds):
                raise ValueError("thresholds must be strictly inside oriented calibration support")
            if len(retentions) != len(thresholds) + 1:
                raise ValueError("ordered thresholds require one more retention than thresholds")
            values["thresholds"] = thresholds
        else:
            scores = _numbers(payload["percentile_scores"], "percentile_scores", minimum=2)
            if any(left > right for left, right in zip(scores, scores[1:])) or scores[0] == scores[-1]:
                raise ValueError("percentile scores must be sorted with at least two distinct values")
            if len(scores) != len(values["calibration_row_hashes"]):
                raise ValueError("percentile scores must cover exactly the recorded calibration rows")
            if scores[0] != support[0] or scores[-1] != support[1]:
                raise ValueError("continuous support must equal the observed oriented score range")
            knots = _numbers(payload["percentile_knots"], "percentile_knots", minimum=2)
            if not _strictly_increasing(knots) or knots[0] != 0 or knots[-1] != 1:
                raise ValueError("percentile knots must increase strictly from 0 to 1")
            if len(retentions) != len(knots):
                raise ValueError("each percentile knot requires a retention")
            values.update(percentile_scores=scores, percentile_knots=knots)

        # Keep an immutable serialized provenance copy; property callers get a
        # fresh object and cannot change the policy by mutating their input.
        values["_payload_json"] = json.dumps(dict(payload), sort_keys=True, allow_nan=False)
        result = object.__new__(cls)
        for key, value in values.items():
            object.__setattr__(result, key, value)
        return result

    @classmethod
    def load(cls, path):
        return cls.from_payload(json.loads(Path(path).read_text()))

    @property
    def payload(self):
        return json.loads(self._payload_json)

    @property
    def legacy_only(self):
        return self.statistic == "raw_score_iqr"

    def validate_identity(self, *, checkpoint_sha256, scorer_variant, model, adapter, selector=None):
        identity = {"checkpoint_sha256": _hash(checkpoint_sha256, "checkpoint_sha256"),
                    "scorer_variant": _string(scorer_variant, "scorer_variant"),
                    "model": _string(model, "model"), "adapter": _string(adapter, "adapter")}
        if selector is not None:
            identity["selector"] = _string(selector, "selector")
        for key, value in identity.items():
            if getattr(self, key) != value:
                raise ValueError(f"adaptive policy {key} differs from the active scorer/model")

    def validate_evaluation_identity(self, *, row_hash, group_id, image_sha256, split):
        """Reject calibration leakage in nondevelopment data; return reuse flag.

        The caller must record this returned flag as ``calibration_reuse`` so a
        development rerun cannot be presented as independent confirmation.
        """
        row_hash = _hash(row_hash, "row_hash")
        group_id = _string(group_id, "group_id")
        image_sha256 = _hash(image_sha256, "image_sha256")
        split = _string(split, "split")
        overlap = (row_hash in self.calibration_row_hashes or group_id in self.calibration_group_ids
                   or image_sha256 in self.calibration_image_sha256)
        if overlap and split not in DEVELOPMENT_SPLITS:
            raise ValueError("evaluation identity overlaps adaptive-policy calibration data")
        return overlap

    def _percentile(self, value):
        scores = self.percentile_scores
        left, right = bisect_left(scores, value), bisect_right(scores, value)
        if left != right:
            return ((left + right - 1) / 2) / (len(scores) - 1)
        # Called only inside support, which equals the calibration range.
        lower, upper = scores[left - 1], scores[left]
        lower_rank = (bisect_left(scores, lower) + left - 1) / 2
        upper_rank = (left + bisect_right(scores, upper) - 1) / 2
        weight = (value - lower) / (upper - lower)
        return (lower_rank + weight * (upper_rank - lower_rank)) / (len(scores) - 1)

    def _continuous_retention(self, percentile):
        knots = self.percentile_knots
        right = min(bisect_right(knots, percentile), len(knots) - 1)
        left = max(0, right - 1)
        lower, upper = Decimal(str(knots[left])), Decimal(str(knots[right]))
        weight = (Decimal(str(percentile)) - lower) / (upper - lower)
        retention = Decimal(str(self.retentions[left])) + weight * (
            Decimal(str(self.retentions[right])) - Decimal(str(self.retentions[left])))
        return int((retention * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)) / 100

    def decide(self, scores):
        statistic_value = calculate_statistic(scores, self.statistic)
        return self.decide_from_statistic(statistic_value, len(scores))

    def decide_from_statistic(self, statistic_value, count):
        """Apply the same frozen rule to a verified, answer-free cached scalar."""
        statistic_value = _number(statistic_value, "statistic_value")
        if type(count) is not int or count < 1:
            raise ValueError("input patch count must be a positive integer")
        if self.statistic == "normalized_entropy" and not 0 <= statistic_value <= 1:
            raise ValueError("normalized entropy must lie in [0,1]")
        if self.statistic == "raw_score_iqr" and statistic_value < 0:
            raise ValueError("IQR cannot be negative")
        oriented = -statistic_value if self.direction == "lower_is_harder" else statistic_value
        fallback = not self.support[0] <= oriented <= self.support[1]
        percentile = None
        if fallback:
            retention = 1.0
        elif self.form == "ordered_thresholds":
            retention = self.retentions[bisect_right(self.thresholds, oriented)]
        else:
            percentile = self._percentile(oriented)
            retention = self._continuous_retention(percentile)
        selected = min(count, max(1, count * _percent(retention) // 100))
        return AdaptiveVisionDecision(
            input_count=count, selected_count=selected, retention=retention,
            statistic_value=statistic_value, oriented_value=oriented, percentile=percentile,
            fallback=fallback, fallback_reason="out_of_support" if fallback else None,
            legacy_only=self.legacy_only)


# Short aliases are convenient for downstream runners without changing the
# descriptive names used in saved metadata and tests.
Policy = AdaptiveVisionPolicy
Decision = AdaptiveVisionDecision
