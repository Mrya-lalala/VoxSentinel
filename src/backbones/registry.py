"""Explicit construction of configured B1 backbone implementations."""

from .indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from .router import BackboneRouter, RouterConfig
from .wavlm import WavLMConfig, WavLMExtractor


def build_router(config: dict) -> BackboneRouter:
    """Build a router from parsed ``configs/backbones.yaml`` data."""
    indic = IndicWav2VecExtractor(IndicWav2VecConfig(**config["indic_wav2vec"]))
    global_model = WavLMExtractor(WavLMConfig(**config["wavlm"]))
    routing = config.get("routing", {})
    return BackboneRouter(
        indic,
        global_model,
        RouterConfig(
            indic_languages=frozenset(routing.get("indic_languages", RouterConfig().indic_languages)),
            unknown_language_route=routing.get("unknown_language_route", "global_fallback"),
        ),
    )
