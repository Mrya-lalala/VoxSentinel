"""Small typed representation of the supported B2 configuration."""

from dataclasses import dataclass, field
from typing import Any


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
