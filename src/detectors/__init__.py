"""B2 detector heads, adapters, checkpoints and training."""

from .base import Detector, DetectorOutput
from .checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    build_checkpoint,
    load_checkpoint,
    restore_checkpoint,
    save_checkpoint,
)
from .gru import GruDetector, GruSpoofDetector
from .registry import available_detectors, create_detector, register_detector
from .runner import run_training
from .training import (
    BatchFactory,
    BatchSource,
    EpochResult,
    EpochSummary,
    LossFn,
    TrainingResult,
    build_loss,
    build_optimizer,
    fit,
    train_epoch,
)

__all__ = [
    "BatchFactory",
    "BatchSource",
    "CHECKPOINT_FORMAT_VERSION",
    "Detector",
    "DetectorOutput",
    "EpochResult",
    "EpochSummary",
    "GruDetector",
    "GruSpoofDetector",
    "LossFn",
    "TrainingResult",
    "available_detectors",
    "build_checkpoint",
    "build_loss",
    "build_optimizer",
    "create_detector",
    "fit",
    "load_checkpoint",
    "register_detector",
    "restore_checkpoint",
    "run_training",
    "save_checkpoint",
    "train_epoch",
]
