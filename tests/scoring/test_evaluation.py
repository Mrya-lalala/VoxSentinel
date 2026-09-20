import pytest
import torch

from src.scoring import evaluate


def test_evaluate_aggregates_scores_labels_and_weighted_loss(dummy_detector, dummy_batches):
    result = evaluate(dummy_detector, dummy_batches, device="cpu")
    assert result.scores.shape == result.labels.shape == (4,)
    assert result.batches == 2
    assert result.loss is not None and result.loss >= 0
    assert 0.0 <= result.metrics.accuracy <= 1.0


def test_evaluate_returns_all_binary_metrics(dummy_detector, dummy_batches):
    result = evaluate(dummy_detector, dummy_batches, device="cpu")

    metrics = result.metrics

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.eer <= 1.0
    assert 0.0 <= metrics.auroc <= 1.0

    assert metrics.true_negative >= 0
    assert metrics.false_positive >= 0
    assert metrics.false_negative >= 0
    assert metrics.true_positive >= 0

    assert 0.0 <= metrics.genuine_false_alarm_rate <= 1.0
    assert 0.0 <= metrics.spoof_miss_rate <= 1.0
    assert 0.0 <= metrics.threshold <= 1.0


def test_evaluate_uses_the_provided_loss_fn(dummy_detector, dummy_batches):
    constant = lambda logits, labels: logits.sum() * 0.0 + 2.5
    result = evaluate(dummy_detector, dummy_batches, device="cpu", loss_fn=constant)
    assert result.loss == pytest.approx(2.5)


def test_evaluate_rejects_missing_labels_and_device_mismatch(dummy_detector, dummy_batches):
    with pytest.raises(KeyError, match="labels"):
        evaluate(dummy_detector, [{"features": torch.randn(1, 2, 4)}], device="cpu")

    with pytest.raises(ValueError, match="move the detector"):
        evaluate(dummy_detector, dummy_batches, device="cuda")


def test_evaluate_rejects_empty_batches(dummy_detector):
    with pytest.raises(ValueError, match="at least one batch"):
        evaluate(dummy_detector, [], device="cpu")