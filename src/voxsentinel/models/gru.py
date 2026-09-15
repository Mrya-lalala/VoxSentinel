"""Custom GRU spoof detector over B1 embedding sequences.

The detector follows the RawNet2 idea of recurrent modelling over frame-level
features, but it consumes B1 encoder embeddings ``[B, T, D]`` instead of raw
waveforms.  RawNet2's front-end (sinc filterbank, residual/FMS blocks) is not
part of this model.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .base import Backbone
from .registry import register_backbone


class GruSpoofDetector(Backbone):
    """LayerNorm -> Linear -> unidirectional GRU -> masked pooling -> logits.

    Input is an embedding sequence ``[batch, time, input_dim]``.  Padded frames
    are excluded from both the recurrent computation and the pooling by using
    ``valid_lengths`` and/or ``padding_mask``.  Padding is assumed to be
    trailing (right-padded), matching the B1 encoder batch contract.

    Logits follow the repository convention: index 0 is genuine, index 1 is
    spoof.  ``DetectorOutput.scores`` therefore reports P(spoof).
    """

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
            raise ValueError("The GRU detector uses exactly two classes (0=genuine, 1=spoof).")
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

    @staticmethod
    def _resolve_valid_mask(
        features: Tensor,
        valid_lengths: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        """Return a bool ``[B, T]`` mask where True marks a valid frame."""
        batch, frames, _ = features.shape
        if valid_lengths is None and padding_mask is None:
            return torch.ones((batch, frames), dtype=torch.bool, device=features.device)

        if valid_lengths is not None:
            lengths = valid_lengths.to(device=features.device, dtype=torch.long)
            if lengths.ndim != 1 or lengths.shape[0] != batch:
                raise ValueError(f"valid_lengths must have shape [batch]; got {tuple(valid_lengths.shape)}.")
            if bool(((lengths < 0) | (lengths > frames)).any()):
                raise ValueError(f"valid_lengths must be within [0, {frames}].")
            positions = torch.arange(frames, device=features.device).unsqueeze(0)
            valid = positions < lengths.unsqueeze(1)
        else:
            valid = torch.ones((batch, frames), dtype=torch.bool, device=features.device)

        if padding_mask is not None:
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != (batch, frames):
                raise ValueError(f"padding_mask must have shape [batch, time] = [{batch}, {frames}].")
            valid = valid & ~padding_mask.to(device=features.device, dtype=torch.bool)
        return valid

    def forward(
        self,
        features: Tensor,
        valid_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return genuine/spoof logits with shape ``[batch, 2]``."""
        if features.ndim != 3:
            raise ValueError(f"GRU detector expects features with shape [batch, time, dim]; got {tuple(features.shape)}.")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected embedding dim {self.input_dim}; got {features.shape[-1]}.")

        valid = self._resolve_valid_mask(features, valid_lengths, padding_mask)
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


def _create_gru(config):
    return GruSpoofDetector(**config.parameters)


register_backbone("gru", _create_gru)
