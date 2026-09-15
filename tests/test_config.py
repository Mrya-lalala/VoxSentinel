from pathlib import Path

import pytest

from voxsentinel.config import load_config


def test_config_overlay_loads_gru():
    root = Path(__file__).resolve().parents[1]
    config = load_config([root / "configs/base.yaml", root / "configs/gru.yaml"])
    assert config.model.name == "gru"
    assert config.model.parameters["input_dim"] == 1024
    assert config.model.parameters["hidden_size"] == 256
    assert config.optimizer.name == "adamw"
    assert config.optimizer.learning_rate == 0.001
    assert config.training.batch_size == 8
    assert config.training.epochs == 10
    assert config.training.loss == "cross_entropy"
    assert config.checkpoint.best_metric == "eer"
    assert config.checkpoint.best_mode == "min"


def test_config_requires_model_name(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text("training:\n  epochs: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="model.name"):
        load_config(path)
