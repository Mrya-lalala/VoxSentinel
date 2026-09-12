"""Stable data contracts emitted by the ML backbone layer."""

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class ChunkMetadata:
    """Provenance supplied by the audio pipeline for one normalized chunk."""

    source_id: str
    start_ms: float
    duration_ms: float
    sample_rate: int = 16_000
    language: str | None = None


@dataclass(frozen=True)
class EmbeddingSequence:
    """Contextual acoustic features for one chunk, in time-major order."""

    features: np.ndarray
    frame_hop_ms: float
    backbone_id: str
    route: str
    chunk_metadata: ChunkMetadata
    checkpoint_version: str | None = None

    def __post_init__(self) -> None:
        features = np.asarray(self.features)
        if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
            raise ValueError("features must have non-empty shape [frames, hidden_dim]")
        if features.dtype != np.float32:
            features = features.astype(np.float32, copy=False)
        if self.frame_hop_ms <= 0:
            raise ValueError("frame_hop_ms must be positive")
        if not self.backbone_id or not self.route:
            raise ValueError("backbone_id and route are required")
        object.__setattr__(self, "features", features)
