"""Backbone routing and acoustic embedding extraction for VoxSentinel."""

from .base import BackboneExtractor
from .router import BackboneRouter, RouteDecision, RouterConfig
from .schemas import ChunkMetadata, EmbeddingSequence

__all__ = [
    "BackboneExtractor",
    "BackboneRouter",
    "ChunkMetadata",
    "EmbeddingSequence",
    "RouteDecision",
    "RouterConfig",
]
