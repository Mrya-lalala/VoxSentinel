"""YAML configuration loading with a simple, predictable overlay rule."""

from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import AppConfig, CheckpointConfig, EvaluationConfig, ModelConfig, OptimizerConfig, TrainingConfig


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
    return AppConfig(
        model=ModelConfig(name=str(model["name"]), parameters=dict(model.get("parameters", {}))),
        optimizer=OptimizerConfig(**_section(merged, "optimizer")),
        training=TrainingConfig(**_section(merged, "training")),
        evaluation=EvaluationConfig(**_section(merged, "evaluation")),
        checkpoint=CheckpointConfig(**_section(merged, "checkpoint")),
    )
