"""Versioned Phi loader recovery: force eager in the loaded native config.

The pinned snapshot records ``_attn_implementation=flash_attention_2``. Loading
that config implicitly via AutoModel can override the requested public eager
keyword. This wrapper supplies a separately loaded, explicitly eager config.
No frozen source, checkpoint, historical state, or attention formula is edited.
"""

from contextlib import contextmanager
from pathlib import Path
import sys

import torch

from learned_score.mlsys_scorer.protocol import digest_file
from . import phi_adaptive_v3 as original


class PhiEagerAdapterV3(original.PhiAdaptiveAdapterV3):
    version = "mlsys_adaptive_v3_explicit_eager_config"

    def load_model_and_processor(self, config):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor
        path = Path(config.model).resolve()
        if not path.is_dir():
            raise ValueError("Phi v3 requires an existing pinned local snapshot")
        dtype_name = getattr(config, "backbone_dtype", "bfloat16")
        if dtype_name not in {"float16", "bfloat16"}:
            raise ValueError("Phi backbone dtype must be float16 or bfloat16")
        num_crops = int(getattr(config, "num_crops", 16))
        if num_crops != 16:
            raise ValueError("Phi v3 requires the frozen 16-crop preprocessing")
        self.processor = AutoProcessor.from_pretrained(
            str(path), trust_remote_code=True, local_files_only=True, num_crops=num_crops)
        native_config = AutoConfig.from_pretrained(str(path), trust_remote_code=True,
                                                   local_files_only=True)
        native_config._attn_implementation = "eager"
        # Let HF validate the explicitly chosen backend, never reuse a stale
        # autoset decision from the snapshot's FlashAttention configuration.
        native_config._attn_implementation_autoset = False
        native = AutoModelForCausalLM.from_pretrained(
            str(path), config=native_config, trust_remote_code=True, local_files_only=True,
            torch_dtype=getattr(torch, dtype_name), attn_implementation="eager",
            low_cpu_mem_usage=True).to(self.device)
        if (native.config._attn_implementation != "eager"
                or native.model._attn_implementation != "eager"
                or any(type(layer.self_attn).__name__ != "Phi3Attention" for layer in native.model.layers)):
            raise RuntimeError("Phi explicit eager loader selected an unexpected decoder backend")
        self.model = original.PhiRuntimeModel(native).eval().requires_grad_(False)
        self.hidden_size = int(native.config.hidden_size)
        self.snapshot_path, self.dtype_name, self.num_crops = path, dtype_name, num_crops
        return self


@contextmanager
def installed():
    """Process-local CLI substitution, restored even on failure."""
    previous_adapter = original.PhiAdaptiveAdapterV3
    previous_factory = original.load_phi_backbone
    original.PhiAdaptiveAdapterV3 = PhiEagerAdapterV3

    def factory(args, manifest):
        adapter, dataset, identity = previous_factory(args, manifest)
        identity = {**identity, "loader_compatibility": "explicit_native_eager_config_v1",
                    "source_sha256": {**identity["source_sha256"], Path(__file__).name: digest_file(__file__)}}
        return adapter, dataset, identity

    original.load_phi_backbone = factory
    try:
        yield
    finally:
        original.load_phi_backbone = previous_factory
        original.PhiAdaptiveAdapterV3 = previous_adapter


def load_phi_backbone(args, manifest):
    with installed():
        return original.load_phi_backbone(args, manifest)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    experiment = "--experiment" in argv
    if experiment:
        if argv.count("--experiment") != 1:
            raise ValueError("--experiment must occur exactly once")
        argv.remove("--experiment")
    with installed():
        if experiment:
            from . import adaptive_experiment_v3
            previous_hashes = adaptive_experiment_v3.source_hashes
            def source_hashes(adapter):
                result = previous_hashes(adapter)
                if adapter == "phi3_vision":
                    result["mlsys_sprint/phi_adaptive_v3_compat.py"] = digest_file(__file__)
                return result
            adaptive_experiment_v3.source_hashes = source_hashes
            try:
                return adaptive_experiment_v3.main(argv)
            finally:
                adaptive_experiment_v3.source_hashes = previous_hashes
        return original.main(argv)


if __name__ == "__main__":
    main()
