"""Config-driven factories for the optimizer and loss used by B2 training."""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config.schema import OptimizerConfig

LossFn = Callable[[Tensor, Tensor], Tensor]


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> torch.optim.Optimizer:
    """Build one of the small set of optimizers the B2 config allows."""
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
    """Return the loss over detector logits, where class index 1 is spoof."""
    key = name.strip().lower()
    if key in ("cross_entropy", "ce"):
        return F.cross_entropy
    raise ValueError(f"Unknown loss '{name}'. Supported: cross_entropy.")
