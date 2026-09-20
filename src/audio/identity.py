"""Truthful, reproducible preprocessing identity.

The identity separates the algorithm/schema version from the effective runtime
configuration.  Different effective chunk policies (window size, resampler,
range policy, ...) must never collide, so the identity string embeds a SHA-256
digest of a canonical serialization of the effective settings.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from .config import PreprocessingConfig

SCHEMA_VERSION = "audio-v1"


@dataclass(frozen=True)
class PreprocessingIdentity:
    """Effective preprocessing settings plus a stable schema version."""

    schema_version: str = SCHEMA_VERSION
    target_sample_rate: int = 16_000
    mono_policy: str = "mean_channel_downmix"
    scaling_policy: str = "signed-pcm-to-float;uint8-midpoint-subtract"
    resampling_method: str = "scipy_polyphase"
    window_samples: int = 64_000
    hop_samples: int = 64_000
    tail_policy: str = "retain-final-windows-at-least-400-samples-drop-shorter"
    silence_policy: str = "exact-zero-per-chunk"
    range_policy: str = "reject-malformed-input;bounded-overshoot-tolerance"
    overshoot_tolerance: float = 1e-3

    def to_dict(self) -> dict[str, object]:
        # Effective configuration only; the schema version is a separate
        # namespace prefix and must not be part of the configuration digest.
        data = asdict(self)
        data.pop("schema_version", None)
        return data

    def canonical_key(self) -> str:
        """Deterministic, order-independent serialization of effective settings."""
        payload = self.to_dict()
        return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256(self.canonical_key().encode("utf-8")).hexdigest()
        return digest[:16]

    @property
    def identity_string(self) -> str:
        return f"{self.schema_version}:sha256:{self.fingerprint}"

    @classmethod
    def from_config(cls, config: PreprocessingConfig) -> "PreprocessingIdentity":
        config.validate_chunking()
        return cls(
            target_sample_rate=config.target_sample_rate,
            resampling_method=config.resampling_method,
            window_samples=config.window_samples,
            hop_samples=config.hop_samples,
            overshoot_tolerance=config.overshoot_tolerance,
        )


# Identity of prepare.decode_mono_16k + dataset_prep.windows.select_window.
# Legacy PreprocessingIdentity above describes a DIFFERENT array/chunk API.
TRAINED_PREPROCESSING_VERSION = "voxsentinel-prep-2"


def assert_trained_preprocessing(version: object, *, allow_missing: bool = False) -> None:
    """Reject incompatible model contracts; warn for unidentified legacy heads."""
    if version is None and allow_missing:
        import warnings
        warnings.warn("Checkpoint has no preprocessing_version; compatibility is unverified. "
                      "Use INFERENCE_RELEASE for verified baseline inference.", RuntimeWarning, stacklevel=2)
        return
    if version != TRAINED_PREPROCESSING_VERSION:
        raise ValueError(f"Checkpoint preprocessing_version {version!r} differs from "
                         f"service preprocessing {TRAINED_PREPROCESSING_VERSION!r}")
