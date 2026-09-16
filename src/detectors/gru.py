"""Custom GRU spoof head over frozen IndicWav2Vec embedding sequences.

Architecture:

    features [B, T, D]  (D = 1024 for the current encoder)
      -> LayerNorm(D)
      -> Linear(D, hidden_size)
      -> one-layer unidirectional GRU
      -> valid-frame mean pooling
      -> Dropout
      -> Linear(hidden_size, 2)   # 0 = genuine, 1 = spoof

This is a custom head inspired by the RawNet2 idea of recurrent modelling over
frame features; it is not the published RawNet2 model and has no waveform or
sinc front end.  Padded frames never enter the recurrence (packed sequences)
and never contribute to pooling (masked mean).
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..config import ModelConfig
from .base import Detector, DetectorOutput
from .registry import register_detector


class GruSpoofDetector(nn.Module):
    """Trainable genuine/spoof head for embedding sequences ``[B, T, D]``."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.0,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if num_classes != 2:
            raise ValueError("The GRU head uses exactly two classes (0=genuine, 1=spoof).")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.input_norm = nn.LayerNorm(input_dim)
        self.projection = nn.Linear(input_dim, hidden_size)
        self.gru = nn.GRU(
            hidden_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _valid_mask(
        self,
        features: Tensor,
        valid_lengths: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        """Resolve a contiguous-prefix valid-frame mask or raise."""
        batch, frames, _ = features.shape
        positions = torch.arange(frames, device=features.device).unsqueeze(0)
        valid = torch.ones((batch, frames), dtype=torch.bool, device=features.device)

        if valid_lengths is not None:
            lengths = valid_lengths
            if lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"valid_lengths must be an integer tensor; got dtype {lengths.dtype}.")
            lengths = lengths.to(device=features.device, dtype=torch.long)
            if lengths.ndim != 1 or lengths.shape[0] != batch:
                raise ValueError(f"valid_lengths must have shape [batch]; got {tuple(valid_lengths.shape)}.")
            if bool(((lengths < 0) | (lengths > frames)).any()):
                raise ValueError(f"valid_lengths must be within [0, {frames}] (output frames, not samples).")
            valid = valid & (positions < lengths.unsqueeze(1))

        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise ValueError(f"padding_mask must be boolean with True = padding; got dtype {padding_mask.dtype}.")
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != (batch, frames):
                raise ValueError(f"padding_mask must have shape [batch, time] = [{batch}, {frames}].")
            valid = valid & ~padding_mask.to(device=features.device)

        expected = positions < valid.sum(dim=1, keepdim=True)
        if not torch.equal(valid, expected):
            raise ValueError(
                "valid_lengths/padding_mask must describe right padding only; "
                "left padding or masks with holes are not supported."
            )
        return valid

    def forward(
        self,
        features: Tensor,
        valid_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return genuine/spoof logits ``[batch, 2]``."""
        if features.ndim != 3:
            raise ValueError(f"GRU head expects features [batch, time, dim]; got {tuple(features.shape)}.")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected embedding dim {self.input_dim}; got {features.shape[-1]}.")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain non-finite values.")

        valid = self._valid_mask(features, valid_lengths, padding_mask)
        lengths = valid.sum(dim=1)
        if bool((lengths == 0).any()):
            raise ValueError("Every sequence must contain at least one valid frame.")

        projected = self.projection(self.input_norm(features))
        packed = pack_padded_sequence(
            projected,
            lengths.detach().to(device="cpu", dtype=torch.long),
            batch_first=True,
            enforce_sorted=False,
        )
        sequence, _ = self.gru(packed)
        sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=features.shape[1])

        weights = valid.unsqueeze(-1).to(dtype=sequence.dtype)
        pooled = (sequence * weights).sum(dim=1) / lengths.unsqueeze(-1).to(dtype=sequence.dtype)
        return self.classifier(self.dropout(pooled))


class GruDetector(Detector):
    """Batch adapter: ``features`` plus optional ``valid_lengths``/``padding_mask``."""

    def __init__(self, model: GruSpoofDetector) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_config(cls, config: ModelConfig) -> "GruDetector":
        return cls(GruSpoofDetector(**config.parameters))

    def forward(self, batch: Mapping[str, Tensor]) -> DetectorOutput:
        if "features" not in batch:
            raise KeyError("GRU detector batches must contain a 'features' tensor [batch, time, dim].")
        return DetectorOutput(
            logits=self.model(
                batch["features"],
                batch.get("valid_lengths"),
                batch.get("padding_mask"),
            )
        )


register_detector("gru", GruDetector.from_config)
