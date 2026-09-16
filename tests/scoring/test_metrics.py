import pytest
import torch

from src.scoring import binary_metrics, error_rates


def test_perfect_predictions_report_counts_and_rates():
    metrics = binary_metrics(torch.tensor([0.1, 0.9]), torch.tensor([0, 1]))
    assert (metrics.accuracy, metrics.precision, metrics.recall, metrics.f1, metrics.eer) == (1.0, 1.0, 1.0, 1.0, 0.0)
    assert (metrics.true_negative, metrics.false_positive, metrics.false_negative, metrics.true_positive) == (1, 0, 0, 1)
    assert metrics.genuine_false_alarm_rate == 0.0 and metrics.spoof_miss_rate == 0.0


def test_reversed_ranking_has_eer_one():
    metrics = binary_metrics(torch.tensor([0.9, 0.1]), torch.tensor([0, 1]))
    assert metrics.eer == pytest.approx(1.0)


def test_all_scores_tied_has_eer_half():
    metrics = binary_metrics(torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.tensor([0, 1, 0, 1]))
    assert metrics.eer == pytest.approx(0.5)


def test_eer_crossing_with_partial_ties():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1, 0, 1, 0])
    assert binary_metrics(scores, labels).eer == pytest.approx(0.5)


def test_eer_is_none_for_a_single_class():
    metrics = binary_metrics(torch.tensor([0.2, 0.8]), torch.tensor([0, 0]))
    assert metrics.eer is None
    assert metrics.genuine_false_alarm_rate == pytest.approx(0.5)
    assert metrics.spoof_miss_rate == 0.0


def test_error_rates_at_a_threshold():
    far, frr = error_rates(torch.tensor([0.1, 0.6, 0.7, 0.9]), torch.tensor([0, 0, 1, 1]), threshold=0.6)
    assert far == pytest.approx(0.5)
    assert frr == pytest.approx(0.0)


def test_metrics_reject_invalid_inputs():
    with pytest.raises(ValueError, match="binary"):
        binary_metrics(torch.tensor([0.5]), torch.tensor([0.5]))
    with pytest.raises(ValueError, match="binary"):
        binary_metrics(torch.tensor([0.5]), torch.tensor([True]))
    with pytest.raises(ValueError, match="non-finite"):
        binary_metrics(torch.tensor([float("nan")]), torch.tensor([0]))
    with pytest.raises(ValueError, match="same number"):
        binary_metrics(torch.tensor([0.5, 0.5]), torch.tensor([0]))
    with pytest.raises(ValueError, match="at least one"):
        binary_metrics(torch.tensor([]), torch.tensor([]))
    with pytest.raises(ValueError, match="threshold"):
        binary_metrics(torch.tensor([0.5]), torch.tensor([0]), threshold=float("inf"))
