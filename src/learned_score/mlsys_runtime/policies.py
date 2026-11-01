"""GPU selection policies independent of cache storage and model adapters."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RecencyPolicy:
    """Keep the newest generated entries, preserving their source order."""

    name: str = "generated_recency"

    def select(self, source_positions: torch.Tensor, keep_count: int) -> torch.Tensor:
        if source_positions.ndim != 1:
            raise ValueError("source_positions must be one-dimensional")
        if not 0 <= keep_count <= source_positions.numel():
            raise ValueError("keep_count must fit the generated suffix")
        return torch.arange(
            source_positions.numel() - keep_count,
            source_positions.numel(),
            device=source_positions.device,
            dtype=torch.long,
        )
