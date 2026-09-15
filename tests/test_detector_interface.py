import torch


def test_detector_returns_scores_and_loss(detector):
    batch = {"waveforms": torch.tensor([[0.0], [1.0]]), "labels": torch.tensor([0, 1])}
    output = detector(batch)
    assert output.logits.shape == (2, 2)
    assert output.scores.shape == (2,)
    assert detector.loss(output, batch).item() > 0
