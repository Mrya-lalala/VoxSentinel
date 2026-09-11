"""Common interface implemented by all acoustic feature extractors."""

from abc import ABC, abstractmethod

import numpy as np

from .errors import BackboneInputError
from .schemas import ChunkMetadata, EmbeddingSequence


class BackboneExtractor(ABC):
    """Converts one normalized mono waveform into time-major embeddings."""

    backbone_id: str

    @abstractmethod
    def extract(self, samples: np.ndarray, metadata: ChunkMetadata) -> EmbeddingSequence:
        """Return features with shape ``[frames, hidden_dim]`` for one 16 kHz chunk."""

    @staticmethod
    def validate_samples(samples: np.ndarray, metadata: ChunkMetadata) -> np.ndarray:
        waveform = np.asarray(samples, dtype=np.float32)
        if metadata.sample_rate != 16_000:
            raise BackboneInputError("backbones require normalized 16 kHz audio")
        if waveform.ndim != 1:
            raise BackboneInputError("samples must be a mono waveform with shape [samples]")
        if waveform.size == 0:
            raise BackboneInputError("samples must not be empty")
        if not np.isfinite(waveform).all():
            raise BackboneInputError("samples contain non-finite values")
        return waveform
