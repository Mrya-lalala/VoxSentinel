import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from voxsentinel.detectors.base import Detector, DetectorOutput


class DummyDetector(Detector):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 2)

    def forward(self, batch):
        return DetectorOutput(self.layer(batch["waveforms"].float()))


@pytest.fixture
def detector():
    torch.manual_seed(7)
    return DummyDetector()


@pytest.fixture
def batches():
    return [
        {"waveforms": torch.tensor([[0.0], [1.0]]), "labels": torch.tensor([0, 1])},
        {"waveforms": torch.tensor([[1.0], [0.0]]), "labels": torch.tensor([1, 0])},
    ]
