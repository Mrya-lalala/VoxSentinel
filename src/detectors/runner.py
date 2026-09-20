"""Config-driven training entry point for B2 heads.

Wires an :class:`AppConfig`, a detector from the registry, an optimizer, a
loss, and in-memory embedding examples into :func:`fit`.  Real-corpus loading
(manifest + cached frozen features) is a separate data-layer task; this runs a
supplied example list or the synthetic fixture.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from ..config import AppConfig
from ..data import EmbeddingExample, batch_iterator
from .registry import create_detector
from .training import TrainingResult, build_loss, build_optimizer, fit


def run_training(
    config: AppConfig,
    train_examples: Sequence[EmbeddingExample],
    val_examples: Sequence[EmbeddingExample],
    *,
    checkpoint_path: str | Path | None = None,
    device: str | torch.device | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> TrainingResult:
    """Train the configured detector on in-memory labelled embedding examples.

    The device is resolved once, the detector is moved there *before* the
    optimizer is constructed, and batches follow the same device.  The frozen
    B1 encoder has its own configuration and is not moved by this function.
    """
    if config.training.use_amp:
        raise NotImplementedError("Mixed precision is not implemented; set training.use_amp to false.")
    if len(train_examples) == 0 or len(val_examples) == 0:
        raise ValueError("run_training requires non-empty train and validation example lists.")

    resolved_device = torch.device(device if device is not None else config.training.device)
    torch.manual_seed(config.training.seed)
    detector = create_detector(config.model)
    detector.to(resolved_device)
    from .standardization import fit_detector_standardizer
    transform_metadata = fit_detector_standardizer(detector, train_examples)
    if transform_metadata is not None:
        metadata = {**dict(metadata or {}), "feature_standardization": transform_metadata}
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
        device=resolved_device,
        loss_fn=loss_fn,
        checkpoint_path=checkpoint_path,
        monitor=config.checkpoint.best_metric,
        mode=config.checkpoint.best_mode,
        threshold=config.evaluation.threshold,
        metadata=metadata,
    )


__all__ = ["run_training"]
