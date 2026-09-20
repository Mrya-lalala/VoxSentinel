import pytest
import torch

from src.scoring import select_threshold


def test_threshold_selection_picks_the_empirical_crossing():
    scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    labels = torch.tensor([1, 0, 1, 0])
    selection = select_threshold(scores, labels)
    assert selection.threshold == pytest.approx(0.8)
    assert selection.genuine_false_alarm_rate == pytest.approx(0.5)
    assert selection.spoof_miss_rate == pytest.approx(0.5)
    assert selection.rule == "closest_empirical_crossing"


def test_threshold_selection_on_separable_scores():
    selection = select_threshold(torch.tensor([0.9, 0.1]), torch.tensor([1, 0]))
    assert selection.threshold == pytest.approx(0.9)
    assert selection.genuine_false_alarm_rate == 0.0
    assert selection.spoof_miss_rate == 0.0


def test_threshold_selection_requires_two_classes_and_finite_scores():
    with pytest.raises(ValueError, match="both genuine and spoof"):
        select_threshold(torch.tensor([0.2, 0.8]), torch.tensor([1, 1]))
    with pytest.raises(ValueError, match="non-finite"):
        select_threshold(torch.tensor([0.2, float("nan")]), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="equally sized"):
        select_threshold(torch.tensor([0.2]), torch.tensor([0, 1]))


def test_selected_threshold_can_be_frozen_for_held_out_evaluation():
    dev_scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
    dev_labels = torch.tensor([1, 0, 1, 0])

    selection = select_threshold(dev_scores, dev_labels)

    test_scores = torch.tensor([0.95, 0.85, 0.15, 0.05])
    test_labels = torch.tensor([1, 0, 0, 1])

    from src.scoring import binary_metrics

    metrics = binary_metrics(
        test_scores,
        test_labels,
        threshold=selection.threshold,
    )

    assert metrics.threshold == pytest.approx(selection.threshold)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.auroc <= 1.0