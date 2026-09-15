import pytest
import torch

from voxsentinel.evaluation import binary_metrics


def test_binary_metrics_perfect_predictions():
    metrics = binary_metrics(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    assert metrics.accuracy == metrics.precision == metrics.recall == metrics.f1 == 1.0
    assert metrics.eer == 0.0


def test_binary_metrics_rejects_non_binary_labels():
    with pytest.raises(ValueError, match="binary"):
        binary_metrics(torch.tensor([0.5]), torch.tensor([2]))
