"""Save, load, and validate versioned B2 checkpoints."""

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .schema import CHECKPOINT_FORMAT_VERSION, build_checkpoint

_REQUIRED = {"format_version", "model_name", "model_config", "model_state_dict", "epoch", "global_step", "metrics"}


def save_checkpoint(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = build_checkpoint(*args, **kwargs)
    torch.save(payload, Path(path))
    return payload


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu", model_name: str | None = None) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # Torch versions before weights_only was introduced.
        payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or not _REQUIRED.issubset(payload):
        raise ValueError("Not a compatible VoxSentinel checkpoint payload.")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format {payload['format_version']}.")
    if model_name is not None and payload["model_name"].lower() != model_name.lower():
        raise ValueError(f"Checkpoint is for {payload['model_name']}, not {model_name}.")
    return payload


def restore_checkpoint(
    path: str | Path,
    model: nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
    model_name: str | None = None,
) -> dict[str, Any]:
    payload = load_checkpoint(path, map_location=map_location, model_name=model_name)
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload
