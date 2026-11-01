"""Shared helpers: locate the repo and make `learned_score` importable."""
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("ASYMVLM_ROOT", Path(__file__).resolve().parents[2]))
CONFIGS = Path(__file__).resolve().parents[1] / "configs"
MODELS = ("qwen25", "llava_next", "phi35")


def setup_path():
    src = REPO / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))


def load_config(model):
    import json
    return json.loads((CONFIGS / f"{model}.json").read_text())


def hf_model(config, override=None):
    """Model id (resolved through HF_HOME by transformers) unless a local snapshot is given."""
    return override or os.environ.get("ASYMVLM_MODEL_PATH") or config["hf_model_id"]
