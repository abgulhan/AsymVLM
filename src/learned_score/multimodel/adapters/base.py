"""Adapter contract separating model mechanics from training semantics."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch


@dataclass
class PreparedBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    visual_positions: torch.Tensor
    semantic_text_positions: torch.Tensor
    prompt_text_positions: torch.Tensor
    answer_positions: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long)
    )
    native_inputs: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PrunedBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    retained_source_positions: torch.Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


class VLMAdapter(ABC):
    name: str
    version: str
    hidden_size: int

    @abstractmethod
    def load_model_and_processor(self, config): ...

    @abstractmethod
    def prepare_sample(self, dataset_row) -> PreparedBatch: ...

    def visual_positions(self, prepared: PreparedBatch) -> torch.Tensor:
        return prepared.visual_positions

    def semantic_text_positions(self, prepared: PreparedBatch) -> torch.Tensor:
        return prepared.semantic_text_positions

    def answer_positions(self, prepared: PreparedBatch) -> torch.Tensor:
        return prepared.answer_positions

    @abstractmethod
    def teacher_forward(self, prepared: PreparedBatch) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def student_forward_with_key_penalty(
        self, prepared: PreparedBatch, key_penalty: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        ...

    @abstractmethod
    def physically_prune(
        self,
        prepared: PreparedBatch,
        selected_visual_indices: torch.Tensor,
        position_policy: str,
    ) -> PrunedBatch:
        ...

    def physically_prune_with_answer(
        self,
        prepared: PreparedBatch,
        selected_visual_indices: torch.Tensor,
        position_policy: str = "original",
    ) -> PrunedBatch:
        """Delete visual tokens while retaining prompt and teacher-forced answer."""

        if prepared.answer_positions.numel() == 0:
            raise ValueError("answer-aware physical pruning requires answer tokens")
        selected = selected_visual_indices.sort().values
        visual_source = prepared.visual_positions[selected]
        retained = torch.cat((
            prepared.prompt_text_positions,
            visual_source,
            prepared.answer_positions,
        )).unique(sorted=True)
        embeddings = prepared.inputs_embeds[:, retained]
        attention_mask = torch.ones(
            embeddings.shape[:2],
            device=embeddings.device,
            dtype=prepared.attention_mask.dtype,
        )
        source_to_pruned = {
            int(source): index for index, source in enumerate(retained.tolist())
        }
        answer_pruned = torch.tensor(
            [source_to_pruned[int(source)] for source in prepared.answer_positions],
            device=retained.device,
            dtype=torch.long,
        )
        return PrunedBatch(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
            position_ids=self.position_id_policy(
                prepared, retained, position_policy
            ),
            retained_source_positions=retained,
            metadata={
                "position_policy": position_policy,
                "visual_tokens_original": int(prepared.visual_positions.numel()),
                "visual_tokens": int(selected.numel()),
                "answer_positions": answer_pruned,
                "answer_source_positions": prepared.answer_positions,
            },
        )

    def forward_pruned_with_answer(
        self, pruned: PrunedBatch
    ) -> Dict[str, torch.Tensor]:
        """Teacher-forced forward on a physically pruned prompt and answer."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement answer supervision"
        )

    @abstractmethod
    def generate_from_pruned(
        self, pruned: PrunedBatch, generation_config: Dict[str, Any]
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def position_id_policy(
        self,
        prepared: PreparedBatch,
        retained_source_positions: torch.Tensor,
        policy: str,
    ) -> torch.Tensor:
        ...

    def metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "hidden_size": self.hidden_size,
        }
