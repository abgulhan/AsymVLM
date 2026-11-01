"""AsymVLM: asymmetric visual-token pruning + generated-text KV eviction for VLMs.

Public API (import-safe on CPU; torch/transformers are imported lazily):

    from asymvlm import load_model_and_scorer, prepare, prune_and_generate
    rt = load_model_and_scorer("qwen25")
    prepared = prepare(rt, "image.png", "What is written on the sign?")
    text, result = prune_and_generate(rt, prepared, keep_ratio=0.4)
"""
from .config import (GATE_CAP, GATE_PROBES, GATE_THRESHOLDS, GENERATION_DEFAULTS, MODELS,
                     TEXT_EVICTION_BUDGET, ModelSpec, get_spec)
from .gate import (calibrate_threshold, entropy_rule, gate_decision, normalize_answer,
                   scorer_entropy, self_consistency_gate)
from .pruning import (PromptDataset, Runtime, answer, generate, keep_count, load_model_and_scorer,
                      load_scorer, prepare, prune, prune_and_generate, scorer_scores, select_tokens)
from .text_eviction import generate_with_text_eviction, prune_and_generate_long

__version__ = "0.1.0"
