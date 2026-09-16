"""YAML configuration for B2 head training and evaluation.

The project keeps run settings in ``configs/*.yaml``.  Later files override
earlier ones, so ``load_config(["configs/base.yaml", "configs/gru.yaml"])``
produces the GRU run configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class ModelConfig:
    name: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 1
    device: str = "cpu"
    use_amp: bool = False
    shuffle: bool = True
    seed: int = 0
    loss: str = "cross_entropy"


@dataclass(frozen=True)
class EvaluationConfig:
    threshold: float = 0.5


@dataclass(frozen=True)
class CheckpointConfig:
    format_version: int = 1
    strict_load: bool = True
    directory: str = "artifacts/gru"
    best_metric: str = "eer"
    best_mode: str = "min"


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _read(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Configuration {path} must contain a mapping.")
    return value


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping.")
    return value


def load_config(paths: str | Path | Iterable[str | Path]) -> AppConfig:
    """Load one or more YAML files; later files override earlier values."""
    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    merged: dict[str, Any] = {}
    for path in path_list:
        merged = _merge(merged, _read(path))

    model = _section(merged, "model")
    if not model.get("name"):
        raise ValueError("Configuration requires model.name.")
    config = AppConfig(
        model=ModelConfig(name=str(model["name"]), parameters=dict(model.get("parameters", {}))),
        optimizer=OptimizerConfig(**_section(merged, "optimizer")),
        training=TrainingConfig(**_section(merged, "training")),
        evaluation=EvaluationConfig(**_section(merged, "evaluation")),
        checkpoint=CheckpointConfig(**_section(merged, "checkpoint")),
    )
    if config.training.batch_size <= 0:
        raise ValueError("training.batch_size must be positive.")
    if config.training.epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if config.checkpoint.best_mode not in ("min", "max"):
        raise ValueError("checkpoint.best_mode must be 'min' or 'max'.")
    return config


__all__ = [
    "AppConfig",
    "CheckpointConfig",
    "EvaluationConfig",
    "ModelConfig",
    "OptimizerConfig",
    "TrainingConfig",
    "load_config",
]
