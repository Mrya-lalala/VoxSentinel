from pathlib import Path

import pytest
import torch

from src.config import (
    AppConfig,
    CheckpointConfig,
    EvaluationConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.data import batch_iterator, collate_examples, synthetic_examples
from src.detectors import (
    Detector,
    DetectorOutput,
    build_loss,
    build_optimizer,
    create_detector,
    fit,
    load_checkpoint,
    restore_checkpoint,
    run_training,
    train_epoch,
)
from src.scoring import evaluate

INPUT_DIM = 32


class FixedLossDetector(Detector):
    """Stub whose loss is the mean of a per-example loss supplied in the batch."""

    def __init__(self) -> None:
        super().__init__()
        self.layer = torch.nn.Linear(INPUT_DIM, 2)

    def forward(self, batch):
        return DetectorOutput(self.layer(batch["features"].mean(dim=1)))

    def loss(self, output, batch):
        return batch["per_example_loss"].float().mean() + self.layer.weight.sum() * 0.0


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


def test_epoch_loss_weights_a_short_final_batch_by_examples():
    detector = FixedLossDetector()
    optimizer = torch.optim.SGD(detector.parameters(), lr=0.1)
    batches = [
        {
            "features": torch.randn(3, 2, INPUT_DIM),
            "labels": torch.tensor([0, 1, 0]),
            "per_example_loss": torch.tensor([1.0, 1.0, 1.0]),
        },
        {
            "features": torch.randn(1, 2, INPUT_DIM),
            "labels": torch.tensor([1]),
            "per_example_loss": torch.tensor([0.0]),
        },
    ]
    result = train_epoch(detector, batches, optimizer, "cpu")
    assert result.examples == 4
    assert result.loss == pytest.approx(0.75)


def test_train_epoch_rejects_an_empty_epoch():
    detector = FixedLossDetector()
    optimizer = torch.optim.SGD(detector.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="no batches"):
        train_epoch(detector, [], optimizer, "cpu")


def test_fit_updates_weights_validates_and_saves_the_best_checkpoint(tmp_path):
    config = _config(tmp_path, epochs=3)
    detector = create_detector(config.model)
    before = {name: value.detach().clone() for name, value in detector.state_dict().items()}
    optimizer = build_optimizer(detector, config.optimizer)
    checkpoint = tmp_path / "gru_best.pt"

    result = fit(
        detector,
        lambda epoch: batch_iterator(_examples(0), 8, shuffle=True, seed=epoch),
        lambda epoch: batch_iterator(_examples(1, count=16), 8),
        optimizer,
        model_name="gru",
        model_config=config.model.parameters,
        epochs=3,
        loss_fn=build_loss("cross_entropy"),
        checkpoint_path=checkpoint,
        monitor="eer",
        mode="min",
        threshold=0.5,
        metadata={"encoder": "indicwav2vec-large"},
    )

    after = detector.state_dict()
    for name in ("model.projection.weight", "model.gru.weight_ih_l0", "model.classifier.weight"):
        assert not torch.equal(before[name], after[name]), f"{name} did not change"
    assert len(result.epochs) == 3
    assert result.best_epoch in (1, 2, 3)
    for summary in result.epochs:
        assert summary.train_steps > 0 and summary.val_loss >= 0
        assert 0.0 <= summary.metrics.accuracy <= 1.0
        assert summary.metrics.true_positive is not None
    payload = load_checkpoint(checkpoint, model_name="gru")
    assert payload["epoch"] == result.best_epoch
    assert payload["metadata"]["encoder"] == "indicwav2vec-large"
    assert {"accuracy", "precision", "recall", "f1", "eer"}.issubset(payload["metrics"])
    assert "genuine_false_alarm_rate" in payload["metrics"] and "spoof_miss_rate" in payload["metrics"]


def test_fit_rejects_device_mismatch_and_one_shot_generators(tmp_path):
    config = _config(tmp_path, epochs=1)
    detector = create_detector(config.model)
    optimizer = build_optimizer(detector, config.optimizer)
    batches = batch_iterator(_examples(0), 8)
    with pytest.raises(TypeError, match="callable"):
        fit(
            detector,
            batches,
            batches,
            optimizer,
            model_name="gru",
            model_config=config.model.parameters,
            epochs=1,
        )
    detector.to("cpu")
    with pytest.raises(ValueError, match="move the detector"):
        fit(
            detector,
            lambda epoch: batch_iterator(_examples(0), 8),
            lambda epoch: batch_iterator(_examples(1, count=8), 8),
            optimizer,
            model_name="gru",
            model_config=config.model.parameters,
            epochs=1,
            device="cuda",
        )


def test_run_training_end_to_end_with_synthetic_fixture(tmp_path):
    config = _config(tmp_path)
    result = run_training(config, _examples(0), _examples(1, count=16))
    assert result.best_checkpoint is not None and result.best_checkpoint.exists()
    assert result.best_epoch is not None
    assert result.best_metric == pytest.approx(min(summary.metrics.eer for summary in result.epochs))
    first, last = result.epochs[0], result.epochs[-1]
    assert last.train_loss < first.train_loss
    assert last.metrics.accuracy >= 0.75

    restored = create_detector(config.model)
    payload = restore_checkpoint(result.best_checkpoint, restored, model_name="gru")
    evaluation = evaluate(restored, [collate_examples(_examples(1, count=16))], threshold=0.5)
    assert evaluation.metrics.accuracy == pytest.approx(payload["metrics"]["accuracy"])
    assert evaluation.metrics.eer == pytest.approx(payload["metrics"]["eer"])


def test_builders_and_runner_reject_unknown_or_unsupported_options(tmp_path):
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
            _examples(1, count=8),
        )
