"""Native-resolution LLaVA training with reduced saved activations.

Use the existing scorer CLI unchanged after the launcher option::

    python -m learned_score.mlsys_scorer.memory_training \
      --memory-strategy checkpoint extract --manifest ... --adapter llava_next ...

The backbone and dropout remain frozen. Only MistralModel's root ``training``
flag is enabled to activate its non-reentrant layer-checkpoint branch. No
recursive ``train()`` call, image resizing change, or attention-backend change
is performed. ``save_on_cpu`` is an explicit, slower alternative, never an
automatic change on OOM. Both strategies are recorded in artifact identity.
"""

import argparse
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path

from .protocol import digest_file


STRATEGIES = ("checkpoint", "save_on_cpu")


def memory_strategy_identity(strategy):
    if strategy not in STRATEGIES:
        raise ValueError("unknown training-memory strategy")
    return {
        "format": "mlsys_training_memory_v1",
        "strategy": strategy,
        "decoder_class": "MistralModel",
        "checkpoint_use_reentrant": False if strategy == "checkpoint" else None,
        "dropout_modules_training": False,
        "backbone_parameters_frozen": True,
        "native_resolution_unchanged": True,
        "attention_backend_unchanged": True,
        "launcher_sha256": digest_file(Path(__file__)),
    }


def enable_decoder_checkpointing(model):
    """Enable recomputation without changing attention/dropout child modes."""
    from torch.utils.checkpoint import checkpoint
    language = getattr(model, "language_model", None)
    decoder = getattr(language, "model", None)
    if decoder is None or type(decoder).__name__ != "MistralModel":
        raise ValueError("memory launcher is validated only for LLaVA-NeXT Mistral")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("memory optimization requires a frozen backbone")
    if any(module.training for module in model.modules()):
        raise ValueError("all model modules must start in evaluation mode")
    if getattr(decoder, "gradient_checkpointing", False):
        raise ValueError("decoder checkpointing is already enabled")
    decoder._gradient_checkpointing_func = partial(
        checkpoint, use_reentrant=False, preserve_rng_state=True)
    decoder.gradient_checkpointing = True
    # Assign directly. decoder.train() would turn on attention dropout.
    decoder.training = True
    if any(module.training for module in decoder.modules() if module is not decoder):
        raise RuntimeError("a decoder child unexpectedly left evaluation mode")
    return decoder


@contextmanager
def patched_backbone_loader(strategy):
    """Scope integration to this process; existing hashed source files stay intact."""
    from . import backbone
    original = backbone.load_backbone

    def load(args, manifest):
        if args.adapter != "llava_next":
            raise ValueError("memory-training launcher is restricted to llava_next")
        adapter, dataset, identity = original(args, manifest)
        if strategy == "checkpoint" and args.command in {"extract", "train-original"}:
            enable_decoder_checkpointing(adapter.model)
        # Evaluation reads the identical training identity but runs normal eval
        # mode: checkpointing must not disable inference KV caching.
        identity = {**identity, "training_memory_strategy": memory_strategy_identity(strategy)}
        return adapter, dataset, identity

    backbone.load_backbone = load
    try:
        yield
    finally:
        backbone.load_backbone = original


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--memory-strategy", choices=STRATEGIES, default="checkpoint")
    options, remaining = parser.parse_known_args(argv)
    if not remaining or remaining[0] not in {"extract", "train-original", "evaluate"}:
        parser.error("expected extract, train-original, or evaluate after launcher options")
    from .__main__ import main as scorer_main
    if options.memory_strategy == "save_on_cpu":
        from torch.autograd.graph import save_on_cpu
        context = save_on_cpu(pin_memory=True)
    else:
        context = nullcontext()
    with patched_backbone_loader(options.memory_strategy), context:
        scorer_main(remaining)


if __name__ == "__main__":
    main()
