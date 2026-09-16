"""Shared detector contract for B2 training and evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..data.labels import validate_label_tensor


@dataclass
class DetectorOutput:
    """Raw detector output; class 0 = genuine, class 1 = spoof."""

    logits: Tensor

    @property
    def scores(self) -> Tensor:
        """Spoof score ``P(spoof)`` from ``softmax(logits)[..., 1]``.

        These are uncalibrated model scores, not verified probabilities of
        fraud.
        """
        return torch.softmax(self.logits, dim=-1)[..., 1]


class Detector(nn.Module, ABC):
    """Consumes prepared tensors; audio loading and feature extraction stay outside."""

    @abstractmethod
    def forward(self, batch: Mapping[str, Tensor]) -> DetectorOutput:
        raise NotImplementedError

    def loss(self, output: DetectorOutput, batch: Mapping[str, Tensor]) -> Tensor:
        if "labels" not in batch:
            raise KeyError("Detector batches must contain a 'labels' tensor.")
        if output.logits.ndim != 2 or output.logits.shape[-1] != 2:
            raise ValueError("Detector logits must have shape [batch, 2].")
        labels = validate_label_tensor(batch["labels"]).flatten()
        if labels.numel() != output.logits.shape[0]:
            raise ValueError(
                f"Expected {output.logits.shape[0]} labels for the logits but got {labels.numel()}."
            )
        return F.cross_entropy(output.logits, labels)
