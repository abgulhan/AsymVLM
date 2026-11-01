"""Model registry, checkpoint locations and paper defaults for AsymVLM.

All paths are resolved relative to ``ASYMVLM_ROOT`` (default: the repository
root that contains this package). Hugging Face snapshots are resolved through
the standard ``HF_HOME`` cache.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def asymvlm_root():
    return Path(os.environ.get("ASYMVLM_ROOT", str(REPO_ROOT)))


def checkpoint_dir():
    return Path(os.environ.get("ASYMVLM_CHECKPOINTS", str(asymvlm_root() / "checkpoints")))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    adapter: str
    hf_id: str
    revision: str
    attention_implementation: str
    scorer_checkpoint: str              # relative to checkpoint_dir()
    reference_checkpoint: str = None    # backbone-integration identity (Qwen only)
    scorer_format: str = "rank32"       # "rank32" or "query_selector_v24"
    max_visual_patches: int = None
    num_crops: int = None
    gate_threshold: float = None        # calibrated self-consistency threshold (cap 40%)
    extra: dict = field(default_factory=dict)

    def scorer_path(self):
        return checkpoint_dir() / self.scorer_checkpoint

    def reference_path(self):
        return None if self.reference_checkpoint is None else checkpoint_dir() / self.reference_checkpoint


MODELS = {
    "qwen25": ModelSpec(
        key="qwen25", adapter="qwen2_5_vl", hf_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="66285546d2b821cf421d4f5eb2576359d3770cd3", attention_implementation="sdpa",
        scorer_checkpoint="qwen25/scorer.pt", reference_checkpoint="qwen25/reference_scorer.pt",
        scorer_format="query_selector_v24", max_visual_patches=1280, gate_threshold=0.68),
    "llava_next": ModelSpec(
        key="llava_next", adapter="llava_next", hf_id="llava-hf/llava-v1.6-mistral-7b-hf",
        revision="2424fdd47412fccc66d91719126b420e9fbd7065", attention_implementation="sdpa",
        scorer_checkpoint="llava_next/scorer.pt", gate_threshold=0.53),
    "phi35": ModelSpec(
        key="phi35", adapter="phi3_vision", hf_id="microsoft/Phi-3.5-vision-instruct",
        revision="12b77fb40b63a2c73c68243d3f767aab688a1b2a", attention_implementation="eager",
        scorer_checkpoint="phi35/scorer.pt", num_crops=16, gate_threshold=0.65),
}

ALIASES = {
    "qwen": "qwen25", "qwen2.5-vl": "qwen25", "Qwen/Qwen2.5-VL-3B-Instruct": "qwen25",
    "llava": "llava_next", "llava-next": "llava_next", "llava-hf/llava-v1.6-mistral-7b-hf": "llava_next",
    "phi": "phi35", "phi3.5": "phi35", "microsoft/Phi-3.5-vision-instruct": "phi35",
}


def get_spec(model):
    if isinstance(model, ModelSpec):
        return model
    key = ALIASES.get(model, model)
    if key not in MODELS:
        raise KeyError(f"unknown model {model!r}; choose from {sorted(MODELS)}")
    return MODELS[key]


# Paper defaults (greedy decoding; bounded generated-KV storage).
GENERATION_DEFAULTS = dict(max_new_tokens=128, generated_budget=512, greedy=True, slack=32)
# Long-generation text eviction: protected image+prompt prefix, most recent N generated tokens.
TEXT_EVICTION_BUDGET = 256
# Adaptive self-consistency gate: probe levels and default cap (percent of visual tokens kept).
GATE_PROBES = (10, 20)
GATE_CAP = 40
GATE_THRESHOLDS = {k: v.gate_threshold for k, v in MODELS.items()}
