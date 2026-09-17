"""PCM validation, scaling, and channel downmix for the A2 boundary.

The channel axis is always the *last* axis: mono ``[S]`` or samples-first
multichannel ``[S, C]``.  Channels-first ``[C, S]`` is rejected explicitly;
A2 never guesses the layout.
"""

from __future__ import annotations

import math

import numpy as np

from .errors import AudioValidationError

# Float dtypes are amplitudes already in [-1, 1] and are never integer-scaled.
_FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))

# Signed PCM mapping: value / 2**(bits-1), so the most negative code maps to
# exactly -1.0 and the most positive code maps to (2**(bits-1) - 1) / 2**(bits-1).
_SIGNED_PCM_DIVISORS = {
    np.dtype(np.int16): 2 ** 15,
    np.dtype(np.int32): 2 ** 31,
}

# Conventional uint8 PCM: midpoint 128 is silence.  (x - 128) / 128 maps
# 0 -> -1.0, 128 -> 0.0, 255 -> 127/128.
_UINT8_MIDPOINT = 128.0
_UINT8_SCALE = 128.0


def validate_waveform_array(samples: object) -> np.ndarray:
    """Return ``samples`` as an ndarray after rejecting unsupported shapes/dtypes."""
    if not isinstance(samples, np.ndarray):
        raise AudioValidationError("samples must be a NumPy array")
    if samples.ndim not in (1, 2):
        raise AudioValidationError(
            f"samples must have shape [S] (mono) or [S, C] (samples-first multichannel); "
            f"got rank {samples.ndim}"
        )
    if samples.shape[0] == 0:
        raise AudioValidationError("samples must contain at least one sample")
    if samples.ndim == 2 and samples.shape[1] == 0:
        raise AudioValidationError("multichannel audio must have at least one channel")
    if samples.dtype == np.dtype(object) or samples.dtype.kind in ("c", "b", "m", "S", "U", "V"):
        raise AudioValidationError(
            f"unsupported sample dtype {samples.dtype}; expected signed PCM integers, "
            f"uint8 PCM, or float32/float64 amplitudes"
        )
    if samples.dtype.kind == "f" and samples.dtype not in _FLOAT_DTYPES:
        raise AudioValidationError(
            f"unsupported float dtype {samples.dtype}; expected float32 or float64"
        )
    return samples


def validate_sample_rate(sample_rate: object) -> int:
    """Reject non-finite, fractional, or non-positive sample rates."""
    if isinstance(sample_rate, bool):
        raise AudioValidationError("sample_rate must be a positive integer")
    if isinstance(sample_rate, float):
        if not math.isfinite(sample_rate) or not sample_rate.is_integer():
            raise AudioValidationError("sample_rate must be a positive integer")
        sample_rate = int(sample_rate)
    if type(sample_rate) is not int or sample_rate <= 0:
        raise AudioValidationError("sample_rate must be a positive integer")
    return sample_rate


def pcm_to_float(samples: np.ndarray) -> np.ndarray:
    """Scale supported integer PCM to float amplitudes BEFORE downmixing.

    Float input is passed through untouched (already an amplitude, never
    integer-scaled again).  Unsupported integer/unsigned dtypes are rejected
    rather than silently coerced.
    """
    samples = validate_waveform_array(samples)
    if samples.dtype.kind == "f":
        return samples.astype(np.float64, copy=False)

    divisor = _SIGNED_PCM_DIVISORS.get(samples.dtype)
    if divisor is not None:
        return samples.astype(np.float64) / float(divisor)

    if samples.dtype == np.dtype(np.uint8):
        return (samples.astype(np.float64) - _UINT8_MIDPOINT) / _UINT8_SCALE

    raise AudioValidationError(
        f"unsupported integer PCM dtype {samples.dtype}; supported integer types "
        f"are int16, int32, and uint8"
    )


def validate_float_amplitudes(
    samples: np.ndarray,
    *,
    tolerance: float,
    context: str,
) -> np.ndarray:
    """Require finite values within [-1 - tol, 1 + tol]; never clamp here."""
    if not np.isfinite(samples).all():
        raise AudioValidationError(f"{context}: samples must be finite")
    lower = -1.0 - tolerance
    upper = 1.0 + tolerance
    if (samples < lower).any() or (samples > upper).any():
        raise AudioValidationError(
            f"{context}: samples must be within [-1, 1] "
            f"(tolerance {tolerance:g}); decode integer PCM upstream"
        )
    return samples


def downmix_to_mono(samples: np.ndarray) -> np.ndarray:
    """Average channels along the last axis; mono passes through unchanged."""
    if samples.ndim == 1:
        return samples
    return samples.mean(axis=-1)
