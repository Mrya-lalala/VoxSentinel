"""Minimal audio preparation adapter (decode / mono / resample) for the pilot.

The full A2 subsystem was absent in this checkout; see :mod:`src.audio.prepare`
for the documented integration gap and the exact policy applied.
"""

from .prepare import (
    AudioDecodeError,
    DecodedAudio,
    TARGET_SAMPLE_RATE,
    canonicalize_wav_peak_timestamp,
    decode_mono_16k,
    read_prepared_wav,
    resampler_identity,
    write_prepared_wav,
)

__all__ = [
    "AudioDecodeError",
    "DecodedAudio",
    "TARGET_SAMPLE_RATE",
    "canonicalize_wav_peak_timestamp",
    "decode_mono_16k",
    "read_prepared_wav",
    "resampler_identity",
    "write_prepared_wav",
]
