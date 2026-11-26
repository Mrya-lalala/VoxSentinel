"""Training adapter for the hash-pinned IndicWav2Vec Large checkpoint."""

from dataclasses import replace

import torch
from torch import nn

from .errors import BackboneInputError, BackboneLoadError
from .indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from .tensor_interface import EncoderBatch


class TrainableIndicWav2Vec(nn.Module):
    """Prepared PCM in, final-layer frames out. Only model parameters receive gradients."""

    def __init__(self, config: IndicWav2VecConfig, *, mode: str = "frozen") -> None:
        super().__init__()
        if mode not in ("frozen", "last_two", "all_blocks"):
            raise ValueError("mode must be 'frozen', 'last_two', or 'all_blocks'")
        if config.checkpoint_sha256 is None:
            raise ValueError("checkpoint_sha256 is required")
        if config.output_layer is not None:
            raise ValueError("output_layer must be None")
        if config.expected_embedding_dim not in (None, 1024):
            raise ValueError("expected_embedding_dim must be 1024 when specified")

        self._mode = mode
        self._extractor = IndicWav2VecExtractor(replace(config, expected_embedding_dim=1024))
        with torch.inference_mode(False), torch.no_grad():
            self.model = self._extractor._load_model()
        layers = self.model.encoder.layers
        if not isinstance(layers, nn.ModuleList) or len(layers) != 24:
            raise BackboneLoadError("IndicWav2Vec Large must have 24 encoder blocks")
        for name in self.trainable_block_names:
            self.get_submodule(name).requires_grad_(True)
        self.train()

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def embedding_dim(self) -> int:
        return self._extractor.embedding_dim

    @property
    def checkpoint_sha256(self) -> str:
        return self._extractor.checkpoint_sha256

    @property
    def trainable_block_names(self) -> tuple[str, ...]:
        if self.mode == "frozen":
            return ()
        if self.mode == "all_blocks":
            return tuple(f"model.encoder.layers.{index}" for index in range(24))
        return ("model.encoder.layers.22", "model.encoder.layers.23")

    def train(self, mode: bool = True) -> "TrainableIndicWav2Vec":
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        # Encoder.eval() also disables layer dropping in Fairseq's block loop.
        self.model.eval()
        for name in self.trainable_block_names:
            self.get_submodule(name).train(mode)
        return self

    def forward(
        self, samples: torch.Tensor, valid_lengths: torch.Tensor, *, sample_rate: int = 16_000
    ) -> EncoderBatch:
        if not isinstance(samples, torch.Tensor) or samples.dtype != torch.float32:
            raise BackboneInputError("samples must be float32 PCM")
        return self._extractor._extract_batch(
            samples, valid_lengths, sample_rate=sample_rate, frozen=self.mode == "frozen"
        )

    def get_extra_state(self) -> dict:
        return {
            "version": 1,
            "checkpoint_sha256": self.checkpoint_sha256,
            "mode": self.mode,
            "trainable_blocks": self.trainable_block_names,
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise RuntimeError("IndicWav2Vec state metadata must match the base SHA-256 and block configuration")

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs,
    ):
        # Require provenance even in a non-strict load or inside a larger module.
        if prefix + "_extra_state" not in state_dict:
            raise RuntimeError("IndicWav2Vec state metadata is missing")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
