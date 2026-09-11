"""Deterministic, configuration-first selection of an acoustic backbone."""

from dataclasses import dataclass, field

import numpy as np

from .base import BackboneExtractor
from .schemas import ChunkMetadata, EmbeddingSequence


@dataclass(frozen=True)
class RouterConfig:
    indic_languages: frozenset[str] = field(
        default_factory=lambda: frozenset({"as", "bn", "gu", "hi", "kn", "ml", "mr", "ne", "or", "pa", "sa", "si", "ta", "te", "ur"})
    )
    unknown_language_route: str = "global_fallback"


@dataclass(frozen=True)
class RouteDecision:
    route: str
    reason: str


class BackboneRouter:
    """Routes known Indic languages to IndicWav2Vec and all others to global."""

    def __init__(
        self,
        indic_extractor: BackboneExtractor,
        global_extractor: BackboneExtractor,
        config: RouterConfig | None = None,
    ) -> None:
        self.indic_extractor = indic_extractor
        self.global_extractor = global_extractor
        self.config = config or RouterConfig()

    def decide(self, language: str | None) -> RouteDecision:
        code = (language or "").strip().lower().split("-")[0]
        if code in self.config.indic_languages:
            return RouteDecision("indic", "language is in configured Indic language set")
        return RouteDecision(self.config.unknown_language_route, "language is missing or outside configured Indic language set")

    def extract(self, samples: np.ndarray, metadata: ChunkMetadata) -> EmbeddingSequence:
        decision = self.decide(metadata.language)
        extractor = self.indic_extractor if decision.route == "indic" else self.global_extractor
        result = extractor.extract(samples, metadata)
        if result.route != decision.route:
            # Route is router-owned metadata; prevent a backend from mislabelling it.
            return EmbeddingSequence(
                features=result.features,
                frame_hop_ms=result.frame_hop_ms,
                backbone_id=result.backbone_id,
                route=decision.route,
                chunk_metadata=result.chunk_metadata,
                checkpoint_version=result.checkpoint_version,
            )
        return result
