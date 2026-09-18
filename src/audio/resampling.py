"""A single real resampling path used by every A2 entry point.

The default backend is ``scipy.signal.resample_poly`` (a real Kaiser-windowed
polyphase FIR resampler), not an interpolation mock.  The backend and its
settings are recorded in the preprocessing identity.  Only this module
resamples; file loading and array entry points both delegate here.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import AudioValidationError

RESAMPLING_METHODS = ("scipy_polyphase",)


def resample_polyphase(
    samples: np.ndarray,
    source_rate: int,
    target_rate: int,
    *,
    method: str = "scipy_polyphase",
) -> np.ndarray:
    """Resample a 1-D float waveform to ``target_rate``.

    ``samples`` must already be mono floating amplitudes.  Output length is
    ``ceil(len(samples) * target_rate / source_rate)`` for this backend.
    """
    if method not in RESAMPLING_METHODS:
        raise AudioValidationError(
            f"unsupported resampling method {method!r}; expected one of {RESAMPLING_METHODS}"
        )
    source_rate = _positive_int(source_rate, "source_rate")
    target_rate = _positive_int(target_rate, "target_rate")
    if samples.ndim != 1:
        raise AudioValidationError("resampling expects a mono 1-D waveform")
    if samples.size == 0:
        raise AudioValidationError("cannot resample an empty waveform")

    if source_rate == target_rate:
        return samples.astype(np.float32, copy=True)

    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd

    from scipy.signal import resample_poly

    # scipy's default window is a Kaiser-windowed sinc; a real, tested
    # polyphase resampler.  Float64 intermediate avoids accumulating error.
    result = resample_poly(samples.astype(np.float64), up, down)
    return np.asarray(result, dtype=np.float32)


def expected_output_samples(
    input_samples: int,
    source_rate: int,
    target_rate: int,
    *,
    method: str = "scipy_polyphase",
) -> int:
    """Deterministic output length for the configured backend."""
    if method not in RESAMPLING_METHODS:
        raise AudioValidationError(
            f"unsupported resampling method {method!r}; expected one of {RESAMPLING_METHODS}"
        )
    source_rate = _positive_int(source_rate, "source_rate")
    target_rate = _positive_int(target_rate, "target_rate")
    if input_samples < 0:
        raise AudioValidationError("input_samples must be non-negative")
    gcd = math.gcd(source_rate, target_rate)
    up = target_rate // gcd
    down = source_rate // gcd
    return int(math.ceil(input_samples * up / down))


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or type(value) is not int or value <= 0:
        raise AudioValidationError(f"{name} must be a positive integer")
    return value
