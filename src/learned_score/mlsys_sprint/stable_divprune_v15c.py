"""DivPrune max-min selection with already selected indices excluded.

The distance matrix, first choice, floating-point precision, and tie ordering
are identical to the historical helper. Only an illegal repeated selection is
prevented. Thus every historically valid unique prefix is unchanged exactly:
inductively, excluding earlier choices cannot change an unselected argmax.
No distances are clamped, jittered, interpolated, or compared with a tolerance.
"""
import torch
from torch.nn import functional as F


@torch.no_grad()
def stable_divprune_order(vision, keep_count):
    if (vision.ndim != 2 or vision.shape[0] < 2 or vision.shape[1] < 1
            or type(keep_count) is not int or not 1 <= keep_count <= vision.shape[0]
            or not bool(torch.isfinite(vision).all())):
        raise ValueError('finite feature matrix with at least two rows and an in-range integer count required')
    normalized = F.normalize(vision.float(), dim=-1)
    distances = 1.0 - normalized @ normalized.T
    selected = torch.empty(keep_count, dtype=torch.long, device=vision.device)
    for index in range(keep_count):
        if index == 0:
            scores = distances.topk(2, dim=0, largest=False).values[1]
        else:
            scores = distances[selected[:index]].min(dim=0).values
            scores[selected[:index]] = -float('inf')
        selected[index] = scores.argmax()
    return selected


def audit_legacy_prefix(vision, keep_count):
    """Read-only exact comparison; a valid old result must never change."""
    from learned_score.llava_selectors import divprune_order
    old = divprune_order(vision, keep_count)
    new = stable_divprune_order(vision, keep_count)
    first_repeat = next((i for i in range(len(old)) if bool((old[:i] == old[i]).any())), None)
    prefix = len(old) if first_repeat is None else first_repeat
    if not torch.equal(old[:prefix], new[:prefix]):
        raise AssertionError('no-repeat repair changed a valid historical greedy prefix')
    if len(new.unique()) != keep_count:
        raise AssertionError('no-repeat selector did not select exact unique count')
    return {'historical_valid': first_repeat is None,
            'historical_unique_count': len(old.unique()), 'requested_count': keep_count,
            'first_illegal_reselection': first_repeat, 'unchanged_valid_prefix_length': prefix,
            'whole_order_unchanged': torch.equal(old, new),
            'corrected_unique_count': len(new.unique())}


def audit_record_indices(record):
    """Validate saved selector-v5 evidence; never invent missing Phi hashes."""
    from .state import digest
    if record.get('status') != 'ok' or record.get('selector') != 'divprune':
        raise ValueError('successful DivPrune result required')
    values = record.get('selected_patch_indices')
    expected = record.get('selected_patch_indices_sha256')
    if values is None and expected is None:
        return {'stored_indices_available': False,
                'reuse_basis': 'historical runner exact-count uniqueness guard; no empirical prefix hash stored'}
    count, n = record['selected_count'], record['input_visual_tokens']
    if (not isinstance(values, list) or len(values) != count or len(set(values)) != count
            or any(type(i) is not int or not 0 <= i < n for i in values)
            or values != sorted(values) or digest(values) != expected):
        raise ValueError('stored DivPrune selected indices or hash are invalid')
    return {'stored_indices_available': True, 'selected_patch_indices_sha256': expected,
            'reuse_basis': 'stored exact unique patch indices plus unchanged-valid-prefix proof'}
