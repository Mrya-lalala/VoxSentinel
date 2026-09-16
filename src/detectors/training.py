"""B2 head training: epoch loop, optimizer/loss factories, best-checkpoint fit."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import OptimizerConfig
from ..data.labels import validate_label_tensor
from ..scoring import BinaryMetrics, evaluate
from .base import Detector
from .checkpoints import save_checkpoint

LossFn = Callable[[Tensor, Tensor], Tensor]
BatchFactory = Callable[[int], Iterable[Mapping[str, Tensor]]]
BatchSource = Union[Sequence[Mapping[str, Tensor]], BatchFactory]


@dataclass(frozen=True)
class EpochResult:
    loss: float
    steps: int
    examples: int


@dataclass(frozen=True)
class EpochSummary:
    epoch: int
    train_loss: float
    val_loss: float
    metrics: BinaryMetrics
    train_steps: int
    train_examples: int


@dataclass(frozen=True)
class TrainingResult:
    epochs: tuple[EpochSummary, ...]
    best_epoch: int | None
    best_metric: float | None
    best_checkpoint: Path | None
    monitor: str
    mode: str


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build one of the optimizers the B2 config allows."""
    if config.learning_rate <= 0:
        raise ValueError("optimizer.learning_rate must be positive.")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("Model has no trainable parameters.")

    name = config.name.strip().lower()
    if name == "adamw":
        return torch.optim.AdamW(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "adam":
        return torch.optim.Adam(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    raise ValueError(f"Unknown optimizer '{config.name}'. Supported: adamw, adam, sgd.")


def build_loss(name: str) -> LossFn:
    """Return the objective over ``[B, 2]`` logits (class 1 = spoof)."""
    key = name.strip().lower()
    if key in ("cross_entropy", "ce"):
        return F.cross_entropy
    raise ValueError(f"Unknown loss '{name}'. Supported: cross_entropy.")


def _move(batch: Mapping[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _assert_model_device(model: torch.nn.Module, device: torch.device) -> None:
    """Reject a model/device mismatch instead of moving it after optimizer creation."""
    for parameter in model.parameters():
        if parameter.device != device:
            raise ValueError(
                f"Model parameter is on {parameter.device} but training requested {device}; "
                "move the detector before constructing the optimizer."
            )


def train_epoch(
    detector: Detector,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    loss_fn: LossFn | None = None,
) -> EpochResult:
    """Train one epoch; the returned loss is the example-weighted mean."""
    target_device = torch.device(device)
    _assert_model_device(detector, target_device)
    detector.train()
    total_loss, steps, examples = 0.0, 0, 0
    for step, batch in enumerate(batches, start=1):
        moved = _move(batch, target_device)
        if "labels" not in moved:
            raise KeyError("Training batches must contain a 'labels' tensor.")
        moved = {**moved, "labels": validate_label_tensor(moved["labels"]).flatten()}
        optimizer.zero_grad(set_to_none=True)
        output = detector(moved)
        loss = loss_fn(output.logits, moved["labels"]) if loss_fn is not None else detector.loss(output, moved)
        if not bool(torch.isfinite(loss)):
            raise ValueError(f"Training loss is not finite at step {step} (batch size {int(moved['labels'].shape[0])}).")
        loss.backward()
        optimizer.step()
        count = int(moved["labels"].shape[0])
        total_loss += float(loss.detach()) * count
        examples += count
        steps += 1
    if steps == 0:
        raise ValueError("Training epoch received no batches.")
    return EpochResult(loss=total_loss / examples, steps=steps, examples=examples)


def _resolve_batches(source: BatchSource, name: str) -> BatchFactory:
    """Accept a re-iterable sequence or a per-epoch factory; reject one-shot iterators."""
    if callable(source):
        return source
    if isinstance(source, Sequence):
        return lambda epoch: source
    raise TypeError(
        f"{name} must be a reusable sequence of batches or a callable(epoch) factory; "
        f"got {type(source).__name__}. Wrap one-shot generators in a lambda that builds them per epoch."
    )


def _metrics_dict(metrics: BinaryMetrics) -> dict[str, float]:
    values = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "eer": metrics.eer,
        "genuine_false_alarm_rate": metrics.genuine_false_alarm_rate,
        "spoof_miss_rate": metrics.spoof_miss_rate,
        "threshold": metrics.threshold,
        "true_negative": metrics.true_negative,
        "false_positive": metrics.false_positive,
        "false_negative": metrics.false_negative,
        "true_positive": metrics.true_positive,
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def _monitored_value(metrics: BinaryMetrics, monitor: str) -> float:
    if not hasattr(metrics, monitor):
        raise ValueError(
            f"Unknown monitor metric '{monitor}'. Use one of: accuracy, precision, recall, f1, eer."
        )
    value = getattr(metrics, monitor)
    if value is None:
        raise ValueError(
            f"Cannot monitor '{monitor}': the validation split must contain both genuine and spoof examples."
        )
    return float(value)


def fit(
    detector: Detector,
    train_batches: BatchSource,
    val_batches: BatchSource,
    optimizer: torch.optim.Optimizer,
    *,
    model_name: str,
    model_config: Mapping[str, Any],
    epochs: int,
    device: torch.device | str = "cpu",
    loss_fn: LossFn | None = None,
    checkpoint_path: str | Path | None = None,
    monitor: str = "eer",
    mode: str = "min",
    threshold: float = 0.5,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train for ``epochs`` and keep the best-validation checkpoint.

    Training and validation use the same ``loss_fn`` (default: the detector's
    cross-entropy).  The detector must already be on ``device``.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'.")
    target_device = torch.device(device)
    _assert_model_device(detector, target_device)
    train_source = _resolve_batches(train_batches, "train_batches")
    val_source = _resolve_batches(val_batches, "val_batches")

    path = Path(checkpoint_path) if checkpoint_path is not None else None
    history: list[EpochSummary] = []
    best_value: float | None = None
    best_epoch: int | None = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        trained = train_epoch(detector, train_source(epoch), optimizer, target_device, loss_fn=loss_fn)
        global_step += trained.steps
        evaluation = evaluate(
            detector,
            val_source(epoch),
            target_device,
            threshold=threshold,
            loss_fn=loss_fn,
        )
        metrics = evaluation.metrics
        value = _monitored_value(metrics, monitor)
        history.append(
            EpochSummary(
                epoch=epoch,
                train_loss=trained.loss,
                val_loss=float(evaluation.loss or 0.0),
                metrics=metrics,
                train_steps=trained.steps,
                train_examples=trained.examples,
            )
        )

        improved = best_value is None or (value < best_value if mode == "min" else value > best_value)
        if improved:
            best_value, best_epoch = value, epoch
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                save_checkpoint(
                    path,
                    detector,
                    model_name,
                    dict(model_config),
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    metrics=_metrics_dict(metrics),
                    metadata=dict(metadata) if metadata is not None else None,
                )

    return TrainingResult(
        epochs=tuple(history),
        best_epoch=best_epoch,
        best_metric=best_value,
        best_checkpoint=path,
        monitor=monitor,
        mode=mode,
    )


__all__ = [
    "BatchFactory",
    "BatchSource",
    "EpochResult",
    "EpochSummary",
    "LossFn",
    "TrainingResult",
    "build_loss",
    "build_optimizer",
    "fit",
    "train_epoch",
]
