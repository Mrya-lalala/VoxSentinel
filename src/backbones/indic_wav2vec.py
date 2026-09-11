"""Fairseq-backed IndicWav2Vec encoder integration.

The ASR decoder/CTC head is intentionally never invoked: this module emits only
contextual acoustic representations for the spoof-detection adapters.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .base import BackboneExtractor
from .errors import BackboneLoadError
from .schemas import ChunkMetadata, EmbeddingSequence


@dataclass(frozen=True)
class IndicWav2VecConfig:
    checkpoint_path: str
    device: str = "cpu"
    output_layer: int | None = None
    backbone_id: str = "indicwav2vec"
    checkpoint_version: str | None = None


class IndicWav2VecExtractor(BackboneExtractor):
    """Lazy-load an IndicWav2Vec Fairseq checkpoint and extract encoder states."""

    def __init__(self, config: IndicWav2VecConfig, model: Any | None = None) -> None:
        self.config = config
        self.backbone_id = config.backbone_id
        self._model = model

    def _load_model(self) -> Any:
        if self._model is not None:
            return self._model
        checkpoint = Path(self.config.checkpoint_path)
        if not checkpoint.is_file():
            raise BackboneLoadError(f"IndicWav2Vec checkpoint not found: {checkpoint}")
        try:
            from fairseq import checkpoint_utils
            import torch
        except ImportError as exc:
            raise BackboneLoadError("IndicWav2Vec requires the configured Fairseq runtime") from exc
        try:
            models, _, _ = checkpoint_utils.load_model_ensemble_and_task([str(checkpoint)])
            self._model = models[0].to(torch.device(self.config.device)).eval()
        except Exception as exc:  # Preserve upstream detail while presenting one contract error.
            raise BackboneLoadError(f"Unable to load IndicWav2Vec checkpoint: {checkpoint}") from exc
        return self._model

    def extract(self, samples: np.ndarray, metadata: ChunkMetadata) -> EmbeddingSequence:
        waveform = self.validate_samples(samples, metadata)
        try:
            import torch
        except ImportError as exc:
            raise BackboneLoadError("IndicWav2Vec extraction requires PyTorch") from exc
        model = self._load_model()
        source = torch.from_numpy(waveform).unsqueeze(0).to(self.config.device)
        kwargs: dict[str, Any] = {"source": source, "padding_mask": None, "mask": False}
        if self.config.output_layer is not None:
            kwargs["output_layer"] = self.config.output_layer
        with torch.inference_mode():
            output = model.extract_features(**kwargs)
        features = output[0] if isinstance(output, tuple) else output
        if isinstance(features, dict):
            features = features.get("x")
        if features is None or getattr(features, "ndim", 0) != 3:
            raise BackboneLoadError("IndicWav2Vec returned an unsupported feature tensor")
        array = features.squeeze(0).detach().float().cpu().numpy()
        return EmbeddingSequence(
            features=array,
            frame_hop_ms=(metadata.duration_ms / array.shape[0]),
            backbone_id=self.backbone_id,
            route="indic",
            chunk_metadata=metadata,
            checkpoint_version=self.config.checkpoint_version,
        )
