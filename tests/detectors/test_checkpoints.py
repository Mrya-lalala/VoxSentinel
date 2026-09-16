import pytest
import torch

from src.config import ModelConfig
from src.detectors import (
    build_checkpoint,
    create_detector,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)


def _config() -> ModelConfig:
    return ModelConfig(name="gru", parameters={"input_dim": 6, "hidden_size": 4})


def test_checkpoint_round_trip_with_optimizer_and_metadata(tmp_path):
    detector = create_detector(_config())
    optimizer = torch.optim.SGD(detector.parameters(), lr=0.1)
    path = tmp_path / "gru_best.pt"
    save_checkpoint(
        path,
        detector,
        "gru",
        _config().parameters,
        optimizer=optimizer,
        epoch=3,
        global_step=12,
        metrics={"eer": 0.25},
        metadata={"encoder": "indicwav2vec-large", "threshold": 0.4},
    )
    loaded = load_checkpoint(path, model_name="gru")
    restored = create_detector(_config())
    restore_checkpoint(path, restored, model_name="gru")
    assert loaded["epoch"] == 3
    assert loaded["model_config"] == _config().parameters
    assert "optimizer_state_dict" in loaded
    assert loaded["metadata"]["threshold"] == 0.4
    for before, after in zip(detector.parameters(), restored.parameters()):
        assert torch.equal(before, after)


def test_checkpoint_rejects_wrong_model_name_and_format(tmp_path):
    detector = create_detector(_config())
    path = tmp_path / "gru.pt"
    save_checkpoint(path, detector, "gru", _config().parameters)
    with pytest.raises(ValueError, match="not aasist"):
        load_checkpoint(path, model_name="aasist")

    payload = build_checkpoint(detector, "gru", _config().parameters)
    payload["format_version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Unsupported checkpoint format"):
        load_checkpoint(path)
