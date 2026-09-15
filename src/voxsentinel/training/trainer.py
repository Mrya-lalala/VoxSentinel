"""Epoch driver: train, validate with B2 metrics, keep the best checkpoint."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Union

import torch
from torch import Tensor

from ..checkpoints import save_checkpoint
from ..detectors.base import Detector
from ..evaluation import BinaryMetrics, evaluate
from .builders import LossFn
from .engine import train_epoch

# Either a fixed batch stream or a factory that rebuilds batches per epoch
# (the latter allows a fresh shuffle every epoch).
BatchSource = Union[Iterable[Mapping[str, Tensor]], Callable[[int], Iterable[Mapping[str, Tensor]]]]


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


def _batches_for(source: BatchSource, epoch: int) -> Iterable[Mapping[str, Tensor]]:
    return source(epoch) if callable(source) else source


def _metrics_dict(metrics: BinaryMetrics) -> dict[str, float]:
    values = {
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "eer": metrics.eer,
    }
    return {key: float(value) for key, value in values.items() if value is not None}


def _monitored_value(metrics: BinaryMetrics, monitor: str) -> float:
    if not hasattr(metrics, monitor):
        raise ValueError(f"Unknown monitor metric '{monitor}'. Use one of: accuracy, precision, recall, f1, eer.")
    value = getattr(metrics, monitor)
    if value is None:
        raise ValueError(f"Cannot monitor '{monitor}': the validation split must contain both genuine and spoof examples.")
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
) -> TrainingResult:
    """Run ``epochs`` of training and save the best-validation checkpoint."""
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if mode not in ("min", "max"):
        raise ValueError("mode must be 'min' or 'max'.")

    # A plain iterable may be a one-shot generator; snapshot it once so every
    # epoch can iterate the same batches.  Callables rebuild per epoch instead.
    if not callable(train_batches):
        train_batches = tuple(train_batches)
    if not callable(val_batches):
        val_batches = tuple(val_batches)

    path = Path(checkpoint_path) if checkpoint_path is not None else None
    history: list[EpochSummary] = []
    best_value: float | None = None
    best_epoch: int | None = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        trained = train_epoch(detector, _batches_for(train_batches, epoch), optimizer, device, loss_fn=loss_fn)
        global_step += trained.steps
        evaluation = evaluate(detector, _batches_for(val_batches, epoch), device, threshold)
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
                )

    return TrainingResult(
        epochs=tuple(history),
        best_epoch=best_epoch,
        best_metric=best_value,
        best_checkpoint=path,
        monitor=monitor,
        mode=mode,
    )
