"""Exact-budget 4x4 regional reserve for validated single-image Qwen features.

This is a new development selector, not an unchanged-paper implementation.
Coordinates come from original mRoPE positions and are checked against the
merged image grid. No embedding or positional ID is modified.
"""

import math

import torch


def qwen_patch_coordinates(prepared, spatial_merge_size=2):
    metadata = prepared.metadata
    grid = metadata.get("image_grid_thw")
    positions = metadata.get("position_ids")
    if not isinstance(grid, torch.Tensor) or grid.shape != (1, 3):
        raise ValueError("spatial selector supports exactly one Qwen image, not video/multiple images")
    temporal, height, width = map(int, grid[0].tolist())
    if (temporal != 1 or spatial_merge_size < 1 or height % spatial_merge_size
            or width % spatial_merge_size):
        raise ValueError("unsupported temporal or merged image grid")
    height, width = height // spatial_merge_size, width // spatial_merge_size
    if height < 1 or width < 1 or height * width != prepared.visual_positions.numel():
        raise ValueError("Qwen grid size does not match visual feature count")
    if (not isinstance(positions, torch.Tensor) or positions.ndim != 3
            or positions.shape[:2] != (3, 1) or positions.dtype != torch.long):
        raise ValueError("original three-axis Qwen int64 positions are required")
    spatial = positions[:, 0, prepared.visual_positions]
    y = spatial[1] - spatial[1].min()
    x = spatial[2] - spatial[2].min()
    # Native Qwen2.5 restores window order before returning features. Require
    # exactly that row-major merged grid, not a guessed sequence partition.
    expected = torch.arange(height * width, device=x.device)
    if not (bool((spatial[0] == spatial[0, 0]).all())
            and torch.equal(y, expected // width) and torch.equal(x, expected % width)):
        raise ValueError("mRoPE coordinates do not align with row-major merged features")
    return torch.stack((y, x), dim=1), (height, width)


def balanced_quotas(capacities, reserve):
    """Deterministic capped equal allocation; row-major ties/redistribution."""
    if reserve < 0 or any(type(n) is not int or n < 0 for n in capacities) or reserve > sum(capacities):
        raise ValueError("invalid regional reserve/capacities")
    quotas = [0] * len(capacities)
    remaining = reserve
    while remaining:
        for index, capacity in enumerate(capacities):
            if quotas[index] < capacity:
                quotas[index] += 1
                remaining -= 1
                if not remaining:
                    break
    return quotas


@torch.no_grad()
def select_spatial(scores, coordinates, grid_shape, retention=.7, fraction=.2, regions=4):
    if (scores.ndim != 1 or not scores.is_floating_point() or not scores.numel()
            or not bool(torch.isfinite(scores).all())):
        raise ValueError("finite, nonempty one-dimensional patch scores required")
    if (coordinates.shape != (len(scores), 2) or coordinates.dtype != torch.long
            or coordinates.device != scores.device):
        raise ValueError("coordinates must be aligned device-local [patch,2] int64")
    height, width = grid_shape
    if (height < 1 or width < 1 or height * width != len(scores)
            or not 0 < retention <= 1 or not 0 <= fraction <= 1
            or not math.isfinite(retention) or not math.isfinite(fraction)
            or type(regions) is not int or regions < 1):
        raise ValueError("invalid spatial selection geometry or budget")
    if bool((coordinates < 0).any()) or bool((coordinates[:, 0] >= height).any()) or bool((coordinates[:, 1] >= width).any()):
        raise ValueError("coordinate outside the validated grid")
    if len(torch.unique(coordinates[:, 0] * width + coordinates[:, 1])) != len(scores):
        raise ValueError("duplicate spatial coordinates")
    keep = max(1, int(len(scores) * retention))
    reserve = min(keep, int(math.ceil(keep * fraction)))
    tile = (coordinates[:, 0] * regions // height) * regions + coordinates[:, 1] * regions // width
    capacities = torch.bincount(tile, minlength=regions * regions).tolist()
    quotas = balanced_quotas(capacities, reserve)
    order = torch.argsort(scores.float(), descending=True, stable=True)
    selected = torch.zeros(len(scores), dtype=torch.bool, device=scores.device)
    for region, quota in enumerate(quotas):
        if quota:
            ranked = order[tile[order] == region]
            selected[ranked[:quota]] = True
    chosen = int(selected.sum())
    if chosen != reserve:
        raise RuntimeError("regional allocation violated its exact reserve")
    selected[order[~selected[order]][:keep - reserve]] = True
    indices = selected.nonzero(as_tuple=False).flatten()
    if len(indices) != keep:
        raise RuntimeError("spatial selection violated the exact retained count")
    return indices, {"grid_h": height, "grid_w": width, "regions": regions,
                     "regional_fraction": fraction, "regional_reserved": reserve,
                     "quota_counts": quotas, "region_capacities": capacities,
                     "retained_count": keep, "original_count": len(scores),
                     "coordinates": "verified_original_qwen_mrope_row_major"}
