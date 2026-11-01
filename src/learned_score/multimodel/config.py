"""Typed configuration for auditable multimodel scorer runs."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple


Aggregation = Literal["max", "mean", "normalized_logsumexp"]
ObjectiveMode = Literal[
    "paper_prompt_mse",
    "answer_distillation",
    "hard_forward_ste",
    "relevance_diversity",
]
TrainingVariant = Literal[
    "legacy",
    "v1_balanced",
    "v2_paired_budgets",
    "v3_saliency_ranking",
    "v4_calibrated_confidence",
]
PositionPolicy = Literal["compact", "original"]


@dataclass(frozen=True)
class MaskingConfig:
    keep_ratio: float = 0.75
    training_keep_ratios: Tuple[float, ...] = ()
    ratio_schedule: Literal[
        "fixed",
        "uniform",
        "curriculum",
        "multi_loss",
        "stratified",
        "paired_hash",
    ] = "fixed"
    C: float = 5.0
    tau: float = 1.0

    def validate(self) -> None:
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1]")
        ratios = self.training_keep_ratios or (self.keep_ratio,)
        if any(not 0.0 < ratio <= 1.0 for ratio in ratios):
            raise ValueError("training_keep_ratios must be in (0, 1]")
        if self.ratio_schedule not in {
            "fixed",
            "uniform",
            "curriculum",
            "multi_loss",
            "stratified",
            "paired_hash",
        }:
            raise ValueError("unsupported ratio_schedule")
        if self.C < 0.0:
            raise ValueError("C must be non-negative")
        if self.tau <= 0.0:
            raise ValueError("tau must be positive")


@dataclass(frozen=True)
class ScorerConfig:
    aggregation: Aggregation = "max"
    scorer_text: Literal["all-text", "question"] = "all-text"
    regularization: float = 0.001
    weight_min: float = 0.0
    weight_max: float = 3.0

    def validate(self) -> None:
        if self.regularization < 0.0:
            raise ValueError("regularization must be non-negative")
        if self.weight_min > self.weight_max:
            raise ValueError("weight_min cannot exceed weight_max")


@dataclass(frozen=True)
class ObjectiveConfig:
    mode: ObjectiveMode = "paper_prompt_mse"
    kl_temperature: float = 1.0
    kl_weight: float = 1.0
    hidden_weight: float = 1.0
    cross_entropy_weight: float = 0.0
    diversity_weight: float = 0.0
    saliency_ranking_weight: float = 0.2
    confidence_ranking_weight: float = 0.05
    confidence_target_keep_ratio: float = 0.70


@dataclass(frozen=True)
class TrainerConfig:
    adapter: str
    model: str
    dataset: str
    split_manifest: str
    output_dir: str
    epochs: int = 3
    learning_rate: float = 1e-2
    weight_decay: float = 1e-2
    seed: int = 42
    position_policy: PositionPolicy = "compact"
    masking: MaskingConfig = field(default_factory=MaskingConfig)
    scorer: ScorerConfig = field(default_factory=ScorerConfig)
    objective: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    model_revision: Optional[str] = None
    processor_revision: Optional[str] = None
    num_crops: int = 16
    max_visual_patches: int = 1280
    llava_next_grid_pinpoints: Optional[Tuple[Tuple[int, int], ...]] = None
    attn_implementation: Literal["eager", "sdpa"] = "eager"
    max_steps_per_rank: Optional[int] = None
    checkpoint_every_steps: Optional[int] = None
    checkpoint_every_examples: Optional[int] = None
    resume_from_latest: bool = False
    gradient_accumulation_steps: int = 1
    training_variant: TrainingVariant = "legacy"
    saliency_cache_dir: Optional[str] = None
    saliency_cache_examples: int = 2000
    initial_scorer_checkpoint: Optional[str] = None

    def validate(self) -> None:
        self.masking.validate()
        self.scorer.validate()
        if self.epochs < 0:
            raise ValueError("epochs must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.position_policy not in {"compact", "original"}:
            raise ValueError("unsupported position policy")
        if self.num_crops <= 0:
            raise ValueError("num_crops must be positive")
        if self.max_visual_patches <= 0:
            raise ValueError("max_visual_patches must be positive")
        if self.llava_next_grid_pinpoints is not None:
            if self.adapter != "llava_next":
                raise ValueError(
                    "llava_next_grid_pinpoints requires the LLaVA-NeXT adapter"
                )
            if not self.llava_next_grid_pinpoints or any(
                len(pair) != 2 or any(int(value) <= 0 for value in pair)
                for pair in self.llava_next_grid_pinpoints
            ):
                raise ValueError("invalid LLaVA-NeXT image grid pinpoints")
        if self.attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("unsupported attention implementation")
        if (
            self.max_steps_per_rank is not None
            and self.max_steps_per_rank <= 0
        ):
            raise ValueError("max_steps_per_rank must be positive")
        if (
            self.checkpoint_every_steps is not None
            and self.checkpoint_every_steps <= 0
        ):
            raise ValueError("checkpoint_every_steps must be positive")
        if (
            self.checkpoint_every_examples is not None
            and self.checkpoint_every_examples <= 0
        ):
            raise ValueError("checkpoint_every_examples must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.training_variant not in {
            "legacy",
            "v1_balanced",
            "v2_paired_budgets",
            "v3_saliency_ranking",
            "v4_calibrated_confidence",
        }:
            raise ValueError("unsupported native scorer training variant")
        if self.saliency_cache_examples < 0:
            raise ValueError("saliency_cache_examples must be non-negative")
        if (
            self.initial_scorer_checkpoint is not None
            and not self.initial_scorer_checkpoint
        ):
            raise ValueError("initial_scorer_checkpoint cannot be empty")
        if not isinstance(self.resume_from_latest, bool):
            raise ValueError("resume_from_latest must be boolean")
        if self.objective.mode not in {
            "paper_prompt_mse",
            "answer_distillation",
            "hard_forward_ste",
        }:
            raise ValueError(
                "the shared trainer implements paper_prompt_mse, "
                "answer_distillation, and hard_forward_ste"
            )
        if (
            self.objective.mode != "paper_prompt_mse"
            and self.adapter not in {
                "llava_1_5", "llava_next", "qwen2_vl", "qwen2_5_vl"
            }
        ):
            raise ValueError(
                "answer-aware objectives require the LLaVA or Qwen2-VL "
                "adapter"
            )
        if self.objective.kl_temperature <= 0.0:
            raise ValueError("kl_temperature must be positive")
        for name in (
            "kl_weight",
            "hidden_weight",
            "cross_entropy_weight",
            "diversity_weight",
            "saliency_ranking_weight",
            "confidence_ranking_weight",
        ):
            if getattr(self.objective, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.objective.diversity_weight != 0.0:
            raise ValueError(
                "diversity regularization is not enabled in the corrected "
                "answer-aware objective"
            )
        if not 0.0 < self.objective.confidence_target_keep_ratio <= 1.0:
            raise ValueError(
                "confidence_target_keep_ratio must be in (0, 1]"
            )
        native = self.training_variant != "legacy"
        if native:
            if self.objective.mode != "answer_distillation":
                raise ValueError(
                    "native scorer variants require answer_distillation"
                )
            expected = {0.5, 0.6, 0.7, 0.8}
            actual = set(
                self.masking.training_keep_ratios
                or (self.masking.keep_ratio,)
            )
            if actual != expected:
                raise ValueError(
                    "native scorer variants require budgets "
                    "{0.50, 0.60, 0.70, 0.80}"
                )
            schedule = self.masking.ratio_schedule
            if (
                self.training_variant == "v2_paired_budgets"
                and schedule != "paired_hash"
            ):
                raise ValueError("V2 requires ratio_schedule=paired_hash")
            if (
                self.training_variant != "v2_paired_budgets"
                and schedule != "stratified"
            ):
                raise ValueError(
                    "V1/V3/V4 require ratio_schedule=stratified"
                )
            if self.scorer.scorer_text != "question":
                raise ValueError("native variants require question-only scoring")
            if self.position_policy != "original":
                raise ValueError(
                    "native variants require original physical positions"
                )
            if self.gradient_accumulation_steps != 8:
                raise ValueError(
                    "native variants require gradient_accumulation_steps=8"
                )
            if self.checkpoint_every_examples != 250:
                raise ValueError(
                    "native variants require checkpoints every 250 examples"
                )
            if self.training_variant in {
                "v3_saliency_ranking",
                "v4_calibrated_confidence",
            } and not self.saliency_cache_dir:
                raise ValueError("V3/V4 require saliency_cache_dir")

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TrainerConfig":
        values = dict(payload)
        if values.get("llava_next_grid_pinpoints") is not None:
            values["llava_next_grid_pinpoints"] = tuple(
                tuple(int(item) for item in pair)
                for pair in values["llava_next_grid_pinpoints"]
            )
        masking = dict(values.get("masking", {}))
        if "training_keep_ratios" in masking:
            masking["training_keep_ratios"] = tuple(
                masking["training_keep_ratios"]
            )
        values["masking"] = MaskingConfig(**masking)
        values["scorer"] = ScorerConfig(**values.get("scorer", {}))
        values["objective"] = ObjectiveConfig(**values.get("objective", {}))
        config = cls(**values)
        config.validate()
        return config
