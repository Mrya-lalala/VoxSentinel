"""Versioned checkpoint payloads for B2 detector heads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

CHECKPOINT_FORMAT_VERSION = 1

_REQUIRED = {"format_version", "model_name", "model_config", "model_state_dict", "epoch", "global_step", "metrics"}


def build_checkpoint(
    model: nn.Module,
    model_name: str,
    model_config: dict[str, Any],
    *,
    optimizer: torch.optim.Optimizer | None = None,
    epoch: int = 0,
    global_step: int = 0,
    metrics: dict[str, float] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the checkpoint payload.

    ``metadata`` is optional and reserved for experiment identity (encoder
    checkpoint/layer, manifest hashes, preprocessing version, commit, device,
    selected threshold).  It is additive, so format version 1 stays readable.
    """
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
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    return payload


def save_checkpoint(path: str | Path, *args: Any, **kwargs: Any) -> dict[str, Any]:
    payload = build_checkpoint(*args, **kwargs)
    torch.save(payload, Path(path))
    return payload


def load_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    model_name: str | None = None,
) -> dict[str, Any]:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:  # Torch versions before weights_only was introduced.
        payload = torch.load(Path(path), map_location=map_location)
    if not isinstance(payload, dict) or not _REQUIRED.issubset(payload):
        raise ValueError("Not a compatible VoxSentinel checkpoint payload.")
    if payload["format_version"] != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported checkpoint format {payload['format_version']}.")
    if model_name is not None and str(payload["model_name"]).lower() != model_name.lower():
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
    requires_transform = bool(payload["model_config"].get("feature_standardization", False))
    has_transform = getattr(getattr(model, "model", model), "standardizer", None) is not None
    if requires_transform != has_transform:
        raise ValueError("Checkpoint and detector disagree about required feature standardization")
    model.load_state_dict(payload["model_state_dict"], strict=strict)
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


__all__ = [
    "CHECKPOINT_FORMAT_VERSION",
    "build_checkpoint",
    "load_checkpoint",
    "restore_checkpoint",
    "save_checkpoint",
]
