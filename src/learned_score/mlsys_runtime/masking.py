"""Causal/sliding masks based on original positions, not compacted slots."""

import torch


def native_sliding_window(model):
    """Resolve the window consumed by the pinned HF text decoder's mask builder.

    Transformers 4.51 Mistral and Qwen2.5 SDPA mask builders consult the text
    config's ``sliding_window`` directly. Preserve that behavior rather than
    silently changing the pretrained model's configuration.
    """
    config = getattr(model.config, "text_config", model.config)
    window = getattr(config, "sliding_window", None)
    if window is not None and window <= 0:
        raise ValueError("native sliding window must be positive or None")
    return window


def logical_attention_mask(query_sources, key_sources, dtype, window):
    """Additive [1,1,Q,K] SDPA mask preserving causal and local visibility.

    Protected prefix entries remain in storage but receive no special
    exemption from the model's native sliding-attention window.
    """
    if query_sources.ndim != 1 or key_sources.ndim != 1:
        raise ValueError("query/key sources must be one-dimensional")
    if window is not None and window <= 0:
        raise ValueError("window must be positive or None")
    allowed = key_sources[None, :] <= query_sources[:, None]
    if window is not None:
        allowed &= key_sources[None, :] > query_sources[:, None] - window
    mask = torch.zeros(allowed.shape, dtype=dtype, device=query_sources.device)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    return mask[None, None]
