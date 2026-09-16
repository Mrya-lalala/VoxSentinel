"""Shared B2 fixtures.  Packages import as ``src.<area>`` from the repo root,
matching the B1 test layout; no sys.path shims are used."""

import pytest
import torch
from torch import nn

from src.detectors.base import Detector, DetectorOutput


class DummyDetector(Detector):
    """Mean-pooled linear stub used by contract tests."""

    def __init__(self, input_dim: int = 4) -> None:
        super().__init__()
        self.layer = nn.Linear(input_dim, 2)

    def forward(self, batch):
        pooled = batch["features"].mean(dim=1)
        return DetectorOutput(self.layer(pooled.float()))


@pytest.fixture
def dummy_detector():
    torch.manual_seed(7)
    return DummyDetector()


@pytest.fixture
def dummy_batches():
    torch.manual_seed(0)
    features = torch.randn(4, 3, 4)
    labels = torch.tensor([0, 1, 0, 1])
    return [
        {"features": features[:2], "labels": labels[:2]},
        {"features": features[2:], "labels": labels[2:]},
    ]
