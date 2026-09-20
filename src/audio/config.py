"""Effective preprocessing configuration for the A2 audio boundary.

The values here are *effective runtime settings*, kept separate from the
algorithm/schema version carried by :class:`PreprocessingIdentity`.  Two runs
with different window sizes, resampling methods, or range policies must never
share the same complete preprocessing identity.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

# A2 owns its own error types; importing them keeps validation messages local.
from .errors import AudioValidationError

DEFAULT_TARGET_SAMPLE_RATE = 16_000
DEFAULT_WINDOW_SECONDS = 4.0
DEFAULT_HOP_SECONDS = 4.0
# Technical convolution minimum of the frozen IndicWav2Vec Large checkpoint.
DEFAULT_MIN_VALID_SAMPLES = 400


@dataclass(frozen=True)
class PreprocessingConfig:
    """Serializable, validated A2 preprocessing settings."""

    target_sample_rate: int = DEFAULT_TARGET_SAMPLE_RATE
    window_seconds: float = DEFAULT_WINDOW_SECONDS
    hop_seconds: float = DEFAULT_HOP_SECONDS
    min_valid_samples: int = DEFAULT_MIN_VALID_SAMPLES
    resampling_method: str = "scipy_polyphase"
    # Only resampler ringing within this bound is corrected (clamped); anything
    # larger is rejected.  Malformed float input outside [-1,1] is always
    # rejected with no correction.
    overshoot_tolerance: float = 1e-3
    # Absorbs float representation error only, not resampling overshoot.
    input_range_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        if (
            type(self.target_sample_rate) is not int
            or self.target_sample_rate <= 0
        ):
            raise AudioValidationError("target_sample_rate must be a positive integer")
        if not math.isfinite(self.window_seconds) or self.window_seconds <= 0:
            raise AudioValidationError("window_seconds must be finite and positive")
        if not math.isfinite(self.hop_seconds) or self.hop_seconds <= 0:
            raise AudioValidationError("hop_seconds must be finite and positive")
        if type(self.min_valid_samples) is not int or self.min_valid_samples <= 0:
            raise AudioValidationError("min_valid_samples must be a positive integer")
        if self.overshoot_tolerance < 0 or not math.isfinite(self.overshoot_tolerance):
            raise AudioValidationError("overshoot_tolerance must be finite and non-negative")
        if self.input_range_tolerance < 0 or not math.isfinite(self.input_range_tolerance):
            raise AudioValidationError("input_range_tolerance must be finite and non-negative")

    @property
    def window_samples(self) -> int:
        samples = round(self.window_seconds * self.target_sample_rate)
        if samples <= 0:
            raise AudioValidationError("window_seconds is too small for the target sample rate")
        return samples

    @property
    def hop_samples(self) -> int:
        samples = round(self.hop_seconds * self.target_sample_rate)
        if samples <= 0:
            raise AudioValidationError("hop_seconds is too small for the target sample rate")
        return samples

    def validate_chunking(self) -> None:
        """Reject chunk policies the A2 chunker does not implement yet."""
        if self.window_samples != self.hop_samples:
            raise AudioValidationError(
                "this milestone only implements contiguous non-overlapping windows; "
                "window_seconds and hop_seconds must be equal"
            )
        if self.min_valid_samples > self.window_samples:
            raise AudioValidationError(
                "min_valid_samples must not exceed window_samples"
            )


def load_preprocessing_config(
    path: str | Path = "configs/audio.yaml",
) -> PreprocessingConfig:
    """Load ``audio`` settings from a YAML file; missing fields use defaults."""
    data: dict[str, Any] = {}
    if Path(path).is_file():
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(value, dict):
            raise AudioValidationError(f"audio config {path} must contain a mapping")
        section = value.get("audio", {})
        if not isinstance(section, dict):
            raise AudioValidationError("audio config section 'audio' must be a mapping")
        data = dict(section)
    known = {field for field in PreprocessingConfig.__dataclass_fields__}  # type: ignore[attr-defined]
    unknown = set(data) - known
    if unknown:
        raise AudioValidationError(f"unknown audio config keys: {sorted(unknown)}")
    return PreprocessingConfig(**data)
