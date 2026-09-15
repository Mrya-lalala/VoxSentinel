import torch

from voxsentinel.checkpoints import load_checkpoint, restore_checkpoint, save_checkpoint
from voxsentinel.config import ModelConfig
from voxsentinel.detectors import GruDetector, create_detector
from voxsentinel.models import GruSpoofDetector


def _small_config() -> ModelConfig:
    # Narrow test dimensions keep the CPU tests quick while preserving the
    # LayerNorm -> Linear -> GRU -> masked pooling -> classifier ordering.
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


def test_gru_forward_returns_two_logits_for_variable_lengths():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    with torch.no_grad():
        logits = model(batch["features"], batch["valid_lengths"], batch["padding_mask"])
    assert logits.shape == (3, 2)
    assert logits.device.type == "cpu"


def test_gru_ignores_padded_frames():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    padding_mask = batch["padding_mask"]
    clean = batch["features"].clone()
    noisy = batch["features"].clone()
    noisy[padding_mask] = 1_000.0
    with torch.no_grad():
        logits_clean = model(clean, batch["valid_lengths"], padding_mask)
        logits_noisy = model(noisy, batch["valid_lengths"], padding_mask)
    assert torch.allclose(logits_clean, logits_noisy, atol=1e-5)


def test_gru_accepts_valid_lengths_or_padding_mask_alone():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    with torch.no_grad():
        from_lengths = model(batch["features"], valid_lengths=batch["valid_lengths"])
        from_mask = model(batch["features"], padding_mask=batch["padding_mask"])
    assert torch.allclose(from_lengths, from_mask, atol=1e-5)


def test_gru_honors_both_lengths_and_mask_when_they_disagree():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    batch = _batch()
    stricter_mask = batch["padding_mask"] | (torch.arange(5).unsqueeze(0) >= 2)
    clean = batch["features"].clone()
    noisy = clean.clone()
    noisy[stricter_mask] = 500.0
    with torch.no_grad():
        logits_clean = model(clean, batch["valid_lengths"], stricter_mask)
        logits_noisy = model(noisy, batch["valid_lengths"], stricter_mask)
    assert torch.allclose(logits_clean, logits_noisy, atol=1e-5)


def test_gru_rejects_sequences_without_valid_frames():
    model = GruSpoofDetector(**_small_config().parameters).eval()
    features = torch.randn(1, 4, 8)
    with torch.no_grad():
        try:
            model(features, valid_lengths=torch.tensor([0]))
        except ValueError as error:
            assert "valid frame" in str(error)
        else:
            raise AssertionError("A zero-length sequence should be rejected.")


def test_gru_detector_registry_and_adapter():
    detector = create_detector(_small_config()).eval()
    assert isinstance(detector, GruDetector)
    batch = _batch()
    with torch.no_grad():
        output = detector(batch)
    assert output.logits.shape == (3, 2)
    assert output.scores.shape == (3,)
    assert torch.allclose(output.scores, output.logits.softmax(dim=-1)[..., 1])


def test_gru_detector_is_trainable():
    detector = create_detector(_small_config()).train()
    batch = _batch()
    output = detector(batch)
    loss = detector.loss(output, batch)
    loss.backward()
    assert loss.item() > 0
    assert detector.model.projection.weight.grad is not None
    assert detector.model.gru.weight_ih_l0.grad is not None
    assert detector.model.classifier.weight.grad is not None


def test_gru_checkpoint_round_trip(tmp_path):
    config = _small_config()
    detector = create_detector(config).eval()
    path = tmp_path / "gru.pt"
    save_checkpoint(path, detector, "gru", config.parameters, epoch=2, global_step=8)
    restored = create_detector(config).eval()
    payload = restore_checkpoint(path, restored, model_name="gru")
    assert load_checkpoint(path, model_name="gru")["model_config"] == config.parameters
    assert payload["epoch"] == 2
    for before, after in zip(detector.parameters(), restored.parameters()):
        assert torch.equal(before, after)
