import pytest
import torch

from src.config import ModelConfig
from src.data import collate_examples, synthetic_examples
from src.detectors import GruDetector, GruSpoofDetector, create_detector


def _small_config() -> ModelConfig:
    return ModelConfig(
        name="gru",
        parameters={"input_dim": 8, "hidden_size": 6, "num_layers": 1, "dropout": 0.0},
    )


def _batch() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    features = torch.randn(3, 5, 8)
    valid_lengths = torch.tensor([5, 3, 1])
    padding_mask = torch.arange(5).unsqueeze(0) >= valid_lengths.unsqueeze(1)
    return {
        "features": features,
        "valid_lengths": valid_lengths,
        "padding_mask": padding_mask,
        "labels": torch.tensor([0, 1, 1]),
    }


def test_gru_defaults_match_the_embedding_contract():
    model = GruSpoofDetector()
    assert model.input_dim == 1024
    assert model.hidden_size == 256
    assert model.num_layers == 1
    assert model.gru.bidirectional is False
    assert model.classifier.out_features == 2
    assert sum(parameter.numel() for parameter in model.parameters()) == 659_714


def test_gru_forward_returns_two_logits_for_variable_lengths():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    with torch.no_grad():
        logits = model(batch["features"], batch["valid_lengths"], batch["padding_mask"])
    assert logits.shape == (3, 2)


def test_gru_ignores_padded_frames():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    clean = batch["features"].clone()
    noisy = batch["features"].clone()
    noisy[batch["padding_mask"]] = 1_000.0
    with torch.no_grad():
        clean_logits = model(clean, batch["valid_lengths"], batch["padding_mask"])
        noisy_logits = model(noisy, batch["valid_lengths"], batch["padding_mask"])
    assert torch.allclose(clean_logits, noisy_logits, atol=1e-5)


def test_gru_accepts_lengths_or_mask_and_allows_a_stricter_mask():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    stricter = batch["padding_mask"] | (torch.arange(5).unsqueeze(0) >= 2)
    with torch.no_grad():
        from_lengths = model(batch["features"], valid_lengths=batch["valid_lengths"])
        from_mask = model(batch["features"], padding_mask=batch["padding_mask"])
        strict = model(batch["features"], batch["valid_lengths"], stricter)
        unpadded = model(batch["features"][:, :2], valid_lengths=torch.tensor([2, 2, 1]))
    assert torch.allclose(from_lengths, from_mask, atol=1e-5)
    assert strict.shape == (3, 2)
    assert unpadded.shape == (3, 2)


def test_gru_rejects_left_padding_and_mask_holes():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    features = torch.randn(1, 5, 8)
    left_padded = torch.tensor([[True, False, False, False, False]])
    hole = torch.tensor([[False, False, True, False, False]])
    with pytest.raises(ValueError, match="right padding"):
        model(features, padding_mask=left_padded)
    with pytest.raises(ValueError, match="right padding"):
        model(features, torch.tensor([4]), hole)


def test_gru_rejects_invalid_lengths_masks_and_features():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    features = torch.randn(1, 4, 8)
    with pytest.raises(ValueError, match="integer"):
        model(features, valid_lengths=torch.tensor([2.0]))
    with pytest.raises(ValueError, match="integer|shape"):
        model(features, valid_lengths=torch.tensor([[2]]))
    with pytest.raises(ValueError, match="boolean"):
        model(features, padding_mask=torch.zeros(1, 4, dtype=torch.long))
    with pytest.raises(ValueError, match="within"):
        model(features, valid_lengths=torch.tensor([9]))
    with pytest.raises(ValueError, match="valid frame"):
        model(features, valid_lengths=torch.tensor([0]))
    with pytest.raises(ValueError, match="non-finite"):
        model(torch.full((1, 4, 8), float("inf")))
    with pytest.raises(ValueError, match="embedding dim"):
        model(torch.randn(1, 4, 3))
    with pytest.raises(ValueError, match="batch, time, dim"):
        model(torch.randn(4, 8))


def test_gru_detector_adapter_scores_and_gradients():
    detector = create_detector(_small_config())
    assert isinstance(detector, GruDetector)
    batch = _batch()
    output = detector(batch)
    assert output.logits.shape == (3, 2)
    assert torch.allclose(output.scores, output.logits.softmax(dim=-1)[..., 1])
    loss = detector.loss(output, {**batch, "labels": batch["labels"]})
    loss.backward()
    assert detector.model.projection.weight.grad is not None
    assert detector.model.gru.weight_ih_l0.grad is not None
    assert detector.model.classifier.weight.grad is not None


def test_gru_adapter_missing_features_message():
    detector = create_detector(_small_config())
    with pytest.raises(KeyError, match="features"):
        detector({"labels": torch.tensor([0])})


def test_gru_batch_from_collated_examples_ignores_padding():
    detector = create_detector(ModelConfig(name="gru", parameters={"input_dim": 16, "hidden_size": 8})).eval()
    batch = collate_examples(synthetic_examples(6, input_dim=16, min_frames=4, max_frames=9, seed=0))
    with torch.no_grad():
        clean = detector(batch).logits
        noisy_features = batch["features"].clone()
        noisy_features[batch["padding_mask"]] = 999.0
        noisy = detector({**batch, "features": noisy_features}).logits
    assert torch.allclose(clean, noisy, atol=1e-5)
