"""GRU detector adapter for embedding-sequence batches."""

from __future__ import annotations

from typing import Mapping

from torch import Tensor

from ..config.schema import ModelConfig
from ..models.gru import GruSpoofDetector
from .base import Detector, DetectorOutput
from .registry import register_detector


class GruDetector(Detector):
    """Runs the GRU detector on a B1 batch with ``features`` ``[B, T, D]``."""

    def __init__(self, model: GruSpoofDetector) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GruDetector":
        return cls(GruSpoofDetector(**config.parameters))

    def forward(self, batch: Mapping[str, Tensor]) -> DetectorOutput:
        if "features" not in batch:
            raise KeyError("GRU detector batches must contain a 'features' tensor with shape [batch, time, dim].")
        return DetectorOutput(
            logits=self.model(
                batch["features"],
                batch.get("valid_lengths"),
                batch.get("padding_mask"),
            )
        )


register_detector("gru", GruDetector.from_config)
