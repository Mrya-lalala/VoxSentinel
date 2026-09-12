"""Tensor-only encoder contract, independent of a detector or routing policy."""

from dataclasses import dataclass
from typing import Protocol

import torch


@dataclass(frozen=True)
class EncoderBatch:
    """Detached float32 features [B,T,D]; padded frames are zero.

    valid_lengths: int64 [B], measured in output frames.
    padding_mask: bool [B,T], True means padding (ignore), False means valid.
    D is encoder-specific; shared shapes do not make head weights transferable.
    """

    features: torch.Tensor
    valid_lengths: torch.Tensor
    padding_mask: torch.Tensor
    frame_hop_ms: float
    backbone_id: str
    checkpoint_version: str
    output_layer: int | None

    @property
    def embedding_dim(self) -> int:
        return self.features.shape[-1]


class TensorEncoder(Protocol):
    """16 kHz mono floating PCM [B,S] and integer sample lengths [B]."""

    @property
    def embedding_dim(self) -> int: ...

    def extract_batch(
        self, samples: torch.Tensor, valid_lengths: torch.Tensor, *, sample_rate: int = 16_000
    ) -> EncoderBatch: ...
