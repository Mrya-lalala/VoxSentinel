"""Detector registry: one name maps to one detector factory."""

from __future__ import annotations

from collections.abc import Callable

from ..config import ModelConfig
from .base import Detector

DetectorFactory = Callable[[ModelConfig], Detector]
_REGISTRY: dict[str, DetectorFactory] = {}
_RESERVED = ("gru", "aasist")


def available_detectors() -> tuple[str, ...]:
    return tuple(sorted(set(_RESERVED) | set(_REGISTRY)))


def register_detector(name: str, factory: DetectorFactory) -> None:
    if not name:
        raise ValueError("Detector name cannot be empty.")
    _REGISTRY[name.lower()] = factory


def create_detector(config: ModelConfig) -> Detector:
    name = config.name.lower()
    if name not in _REGISTRY:
        if name in _RESERVED:
            raise NotImplementedError(f"The {name} detector has not been implemented yet.")
        raise KeyError(f"Unknown detector '{config.name}'. Available names: {available_detectors()}")
    return _REGISTRY[name](config)
