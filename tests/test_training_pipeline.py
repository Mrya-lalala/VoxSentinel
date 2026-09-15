from pathlib import Path

import pytest
import torch

from voxsentinel.checkpoints import load_checkpoint, restore_checkpoint
from voxsentinel.config import (
    AppConfig,
    CheckpointConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from voxsentinel.data import batch_iterator, collate_examples, synthetic_examples
from voxsentinel.detectors import create_detector
from voxsentinel.evaluation import evaluate
from voxsentinel.training import build_loss, build_optimizer, fit, run_training

INPUT_DIM = 32


def _config(tmp_path: Path, *, epochs: int = 4) -> AppConfig:
    return AppConfig(
        model=ModelConfig(
            name="gru",
            parameters={"input_dim": INPUT_DIM, "hidden_size": 16, "num_layers": 1, "dropout": 0.0},
        ),
        optimizer=OptimizerConfig(name="adamw", learning_rate=0.01, weight_decay=0.0),
        training=TrainingConfig(
            batch_size=8,
            epochs=epochs,
            device="cpu",
            use_amp=False,
            shuffle=True,
            seed=0,
            loss="cross_entropy",
        ),
        evaluation=EvaluationConfig(threshold=0.5),
        checkpoint=CheckpointConfig(directory=str(tmp_path), best_metric="eer", best_mode="min"),
    )


def _examples(seed: int, count: int = 32) -> list:
    return synthetic_examples(count, input_dim=INPUT_DIM, min_frames=6, max_frames=12, seed=seed)


def test_fit_updates_gru_weights_and_runs_validation(tmp_path):
    config = _config(tmp_path, epochs=3)
    detector = create_detector(config.model)
    before = {name: value.detach().clone() for name, value in detector.state_dict().items()}
    optimizer = build_optimizer(detector, config.optimizer)
    checkpoint_path = tmp_path / "gru_best.pt"

    result = fit(
        detector,
        batch_iterator(_examples(0), config.training.batch_size, shuffle=True),
        batch_iterator(_examples(1, count=16), config.training.batch_size),
        optimizer,
        model_name="gru",
        model_config=config.model.parameters,
        epochs=config.training.epochs,
        loss_fn=build_loss(config.training.loss),
        checkpoint_path=checkpoint_path,
        monitor="eer",
        mode="min",
        threshold=config.evaluation.threshold,
    )

    after = detector.state_dict()
    for name in ("model.input_norm.weight", "model.projection.weight", "model.gru.weight_ih_l0", "model.classifier.weight"):
        assert not torch.equal(before[name], after[name]), f"{name} did not change during training"

    assert len(result.epochs) == config.training.epochs
    assert result.best_epoch in range(1, config.training.epochs + 1)
    for summary in result.epochs:
        assert summary.train_steps > 0 and summary.train_examples > 0
        assert summary.train_loss >= 0 and summary.val_loss >= 0
        assert 0.0 <= summary.metrics.accuracy <= 1.0
    assert checkpoint_path.exists()


def test_run_training_learns_synthetic_split_and_saves_best_checkpoint(tmp_path):
    config = _config(tmp_path)
    result = run_training(config, _examples(0), _examples(1, count=16))

    assert result.monitor == "eer" and result.mode == "min"
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()
    assert result.best_epoch is not None
    assert result.best_metric == pytest.approx(min(summary.metrics.eer for summary in result.epochs))

    first, last = result.epochs[0], result.epochs[-1]
    assert last.train_loss < first.train_loss
    assert last.metrics.accuracy >= 0.75
    assert result.epochs[0].metrics.eer is not None

    payload = load_checkpoint(result.best_checkpoint, model_name="gru")
    assert payload["epoch"] == result.best_epoch
    assert payload["model_config"] == config.model.parameters
    assert payload["optimizer_state_dict"] is not None
    assert {"accuracy", "precision", "recall", "f1", "eer"}.issubset(payload["metrics"])


def test_restored_best_checkpoint_reproduces_validation_metrics(tmp_path):
    config = _config(tmp_path)
    val_examples = _examples(1, count=16)
    result = run_training(config, _examples(0), val_examples)
    assert result.best_checkpoint is not None

    payload = load_checkpoint(result.best_checkpoint, model_name="gru")
    restored = create_detector(config.model)
    restore_checkpoint(result.best_checkpoint, restored, model_name="gru")
    evaluation = evaluate(restored, [collate_examples(val_examples)], threshold=config.evaluation.threshold)

    assert evaluation.loss is not None and evaluation.loss >= 0
    assert evaluation.metrics.accuracy == pytest.approx(payload["metrics"]["accuracy"])
    assert evaluation.metrics.eer == pytest.approx(payload["metrics"]["eer"])


def test_padded_frames_do_not_change_batch_logits(tmp_path):
    config = _config(tmp_path)
    detector = create_detector(config.model).eval()
    batch = collate_examples(_examples(0, count=6))

    with torch.no_grad():
        clean = detector(batch).logits
        noisy_features = batch["features"].clone()
        noisy_features[batch["padding_mask"]] = 1_000.0
        noisy = detector({**batch, "features": noisy_features}).logits

    assert torch.allclose(clean, noisy, atol=1e-5)


def test_builders_reject_unknown_optimizer_and_loss(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="optimizer"):
        build_optimizer(create_detector(config.model), OptimizerConfig(name="rmsprop"))
    with pytest.raises(ValueError, match="loss"):
        build_loss("focal")
    with pytest.raises(NotImplementedError, match="use_amp"):
        run_training(
            AppConfig(
                model=config.model,
                optimizer=config.optimizer,
                training=TrainingConfig(epochs=1, use_amp=True),
                evaluation=config.evaluation,
                checkpoint=config.checkpoint,
            ),
            _examples(0),
            _examples(1, count=16),
        )
