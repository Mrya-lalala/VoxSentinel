"""Shared adapter boundary for detector training and evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class DetectorOutput:
    logits: Tensor

    @property
    def scores(self) -> Tensor:
        """Spoof-class probabilities, where label 1 denotes spoof."""
        return torch.softmax(self.logits, dim=-1)[..., 1]


class Detector(nn.Module, ABC):
    """Consumes already-prepared tensors; audio loading stays outside B2."""

    @abstractmethod
    def forward(self, batch: Mapping[str, Tensor]) -> DetectorOutput:
        raise NotImplementedError

    def loss(self, output: DetectorOutput, batch: Mapping[str, Tensor]) -> Tensor:
        if "labels" not in batch:
            raise KeyError("Detector batches must contain a 'labels' tensor.")
        if output.logits.ndim != 2 or output.logits.shape[-1] != 2:
            raise ValueError("Detector logits must have shape [batch, 2].")
        return F.cross_entropy(output.logits, batch["labels"].long())
