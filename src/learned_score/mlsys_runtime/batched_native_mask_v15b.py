"""Native 2D mask fastpath, preserving original-position local visibility.

Reuse the frozen batch loop unchanged. Like dynamic_storage, this process-local
factory substitution is serialized and never permits concurrent/nested calls.
No kernel is forced and no token-parity tolerance is changed.
"""
from contextlib import contextmanager
import threading

import torch

from . import batched_generation_v15 as original

_LOCK = threading.Lock()
_EXPLICIT = original.causal_mask


def native_or_logical_mask(query_sources, key_sources, key_valid, dtype, window=None, query_valid=None):
    if (query_sources.ndim != 2 or key_sources.ndim != 2 or key_valid.shape != key_sources.shape
            or query_sources.shape[0] != key_sources.shape[0]):
        raise ValueError('batched source/mask dimensions differ')
    if window is not None and window <= 0:
        raise ValueError('native sliding window must be positive')
    if not bool(key_valid.any(dim=-1).all()):
        raise ValueError('every request needs a valid protected prefix')
    if query_valid is None:
        query_valid = torch.ones_like(query_sources, dtype=torch.bool)
    if query_valid.shape != query_sources.shape:
        raise ValueError('query validity dimensions differ')
    low = key_sources.masked_fill(~key_valid, torch.iinfo(torch.long).max).amin(dim=-1)
    high = query_sources.masked_fill(~query_valid, torch.iinfo(torch.long).min).amax(dim=-1)
    high = torch.where(query_valid.any(dim=-1), high, low)
    # Native compact-slot local attention is equivalent only while no valid
    # original-position pair can cross the native window. Padding never becomes
    # a valid key. Once locality matters, retain the audited explicit mask.
    if window is None or bool((high - low < window).all()):
        return key_valid.to(dtype=torch.long)
    return _EXPLICIT(query_sources, key_sources, key_valid, dtype, window, query_valid)


@contextmanager
def native_masks():
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('native batch mask substitution is already active')
    previous = original.causal_mask
    try:
        if previous is not _EXPLICIT:
            raise RuntimeError('unexpected preexisting batch mask substitution')
        original.causal_mask = native_or_logical_mask
        yield
    finally:
        original.causal_mask = previous
        _LOCK.release()


def generate_batched(*args, **kwargs):
    with native_masks():
        result = original.generate_batched(*args, **kwargs)
    result.metrics['attention_mask_policy'] = 'native_2d_when_original_position_window_inactive_else_logical_4d'
    result.metrics['sdpa_kernel_forced'] = False
    return result


def validate_batch_parity(*args, **kwargs):
    with native_masks():
        return original.validate_batch_parity(*args, **kwargs)
