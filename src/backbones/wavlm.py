"""Transformers-backed WavLM global-language fallback extractor."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import BackboneExtractor
from .errors import BackboneLoadError
from .schemas import ChunkMetadata, EmbeddingSequence


@dataclass(frozen=True)
class WavLMConfig:
    model_name_or_path: str
    device: str = "cpu"
    backbone_id: str = "wavlm"
    checkpoint_version: str | None = None


class WavLMExtractor(BackboneExtractor):
    """Lazy-load WavLM and expose its last hidden states in the common contract."""

    def __init__(self, config: WavLMConfig, model: Any | None = None) -> None:
        self.config = config
        self.backbone_id = config.backbone_id
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            from transformers import AutoModel
            self._model = AutoModel.from_pretrained(self.config.model_name_or_path).to(self.config.device).eval()
        except ImportError as exc:
            raise BackboneLoadError("WavLM requires the configured Transformers runtime") from exc
        except Exception as exc:
            raise BackboneLoadError(f"Unable to load WavLM model: {self.config.model_name_or_path}") from exc
        return self._model

    def extract(self, samples: np.ndarray, metadata: ChunkMetadata) -> EmbeddingSequence:
        waveform = self.validate_samples(samples, metadata)
        try:
            import torch
        except ImportError as exc:
            raise BackboneLoadError("WavLM extraction requires PyTorch") from exc
        with torch.inference_mode():
            features = self._load_model()(input_values=torch.from_numpy(waveform).unsqueeze(0).to(self.config.device)).last_hidden_state
        array = features.squeeze(0).detach().float().cpu().numpy()
        return EmbeddingSequence(array, metadata.duration_ms / array.shape[0], self.backbone_id, "global_fallback", metadata, self.config.checkpoint_version)
