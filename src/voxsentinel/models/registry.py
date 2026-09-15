"""Backbone registry for B2 models."""

from collections.abc import Callable

from .base import Backbone
from ..config.schema import ModelConfig

BackboneFactory = Callable[[ModelConfig], Backbone]
_REGISTRY: dict[str, BackboneFactory] = {}
_RESERVED = ("gru", "aasist")


def available_models() -> tuple[str, ...]:
    return tuple(sorted(set(_RESERVED) | set(_REGISTRY)))


def register_backbone(name: str, factory: BackboneFactory) -> None:
    if not name:
        raise ValueError("Backbone name cannot be empty.")
    _REGISTRY[name.lower()] = factory


def create_backbone(config: ModelConfig) -> Backbone:
    name = config.name.lower()
    if name not in _REGISTRY:
        if name in _RESERVED:
            raise NotImplementedError(f"The {name} backbone has not been implemented yet.")
        raise KeyError(f"Unknown backbone '{config.name}'. Available names: {available_models()}")
    return _REGISTRY[name](config)
