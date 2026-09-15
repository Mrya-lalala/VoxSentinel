"""Stable checkpoint payload used by all B2 detectors."""

from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1


def build_checkpoint(
    model: nn.Module,
    model_name: str,
    model_config: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict[str, float] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model_name": model_name,
        "model_config": dict(model_config),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "metrics": dict(metrics or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    return payload
