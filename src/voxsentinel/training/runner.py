"""Config-driven entry point that wires B2 examples to the GRU trainer."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch

from ..config.schema import AppConfig
from ..data import EmbeddingExample, batch_iterator
from ..detectors import create_detector
from .builders import build_loss, build_optimizer
from .trainer import TrainingResult, fit


def run_training(
    config: AppConfig,
    train_examples: Sequence[EmbeddingExample],
    val_examples: Sequence[EmbeddingExample],
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
) -> TrainingResult:
    """Train the configured detector on in-memory embedding examples."""
    if config.training.use_amp:
        raise NotImplementedError("Mixed precision is not implemented; set training.use_amp to false.")
    if len(train_examples) == 0 or len(val_examples) == 0:
        raise ValueError("run_training requires non-empty train and validation example lists.")

    torch.manual_seed(config.training.seed)
    detector = create_detector(config.model)
    optimizer = build_optimizer(detector, config.optimizer)
    loss_fn = build_loss(config.training.loss)
    batch_size = config.training.batch_size
    seed = config.training.seed

    def train_batches(epoch: int):
        return batch_iterator(train_examples, batch_size, shuffle=config.training.shuffle, seed=seed + epoch)

    def val_batches(epoch: int):
        return batch_iterator(val_examples, batch_size, shuffle=False)

    if checkpoint_path is None:
        checkpoint_path = Path(config.checkpoint.directory) / f"{config.model.name}_best.pt"

    return fit(
        detector,
        train_batches,
        val_batches,
        optimizer,
        model_name=config.model.name,
        model_config=config.model.parameters,
        epochs=config.training.epochs,
        device=device if device is not None else config.training.device,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        monitor=config.checkpoint.best_metric,
        mode=config.checkpoint.best_mode,
        threshold=config.evaluation.threshold,
    )
