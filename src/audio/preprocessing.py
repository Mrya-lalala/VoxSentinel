"""One shared A2 conversion path: decode -> scale -> downmix -> resample.

Every entry point (already-decoded arrays and soundfile-loaded files) funnels
through :func:`preprocess_array`.  There is exactly one copy of the PCM
scaling, mono, resampling, and validation logic.  SoundFile decodes into float
amplitudes and those floats are never integer-scaled again.

The outgoing contract is finite floating mono PCM at ``target_sample_rate``
(16 kHz) with valid amplitudes in ``[-1, 1]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import PreprocessingConfig
from .conversion import (
    downmix_to_mono,
    pcm_to_float,
    validate_float_amplitudes,
    validate_sample_rate,
    validate_waveform_array,
)
from .errors import AudioLoadError, AudioValidationError
from .identity import PreprocessingIdentity
from .resampling import resample_polyphase


@dataclass(frozen=True)
class PreprocessedAudio:
    """Mono float32 waveform at the target rate, with source provenance."""

    samples: np.ndarray
    sample_rate: int
    source_id: str
    original_sample_rate: int
    original_channels: int
    original_format: str | None
    identity: PreprocessingIdentity

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError("PreprocessedAudio.samples must be a 1-D float32 array")
        object.__setattr__(self, "samples", samples)

    @property
    def identity_string(self) -> str:
        return self.identity.identity_string

    @property
    def num_samples(self) -> int:
        return int(self.samples.size)


def _enforce_outgoing_range(samples: np.ndarray, config: PreprocessingConfig) -> np.ndarray:
    """Validate post-resampling amplitudes and apply the documented overshoot policy.

    Malformed input is rejected earlier with no correction.  Here, resampler
    ringing within ``overshoot_tolerance`` is clamped to [-1, 1]; anything
    larger is rejected as material overshoot.  No peak/RMS normalization or
    per-recording rescaling is applied.
    """
    if not np.isfinite(samples).all():
        raise AudioValidationError("resampled samples must be finite")
    lower = -1.0 - config.overshoot_tolerance
    upper = 1.0 + config.overshoot_tolerance
    if (samples < lower).any() or (samples > upper).any():
        raise AudioValidationError(
            "resampling overshoot exceeds the documented tolerance; "
            f"values outside [-{1.0 + config.overshoot_tolerance:g}, "
            f"{1.0 + config.overshoot_tolerance:g}] were produced"
        )
    return np.clip(samples, -1.0, 1.0)


def preprocess_array(
    samples: object,
    sample_rate: object,
    *,
    source_id: str = "array",
    original_format: str | None = None,
    config: PreprocessingConfig | None = None,
) -> PreprocessedAudio:
    """Convert, downmix, resample and validate an already-decoded array.

    Accepts mono ``[S]`` or samples-first multichannel ``[S, C]``.  Integer PCM
    is scaled to float *before* channel averaging; float input is already an
    amplitude and is validated in ``[-1, 1]``.
    """
    config = config or PreprocessingConfig()
    sample_rate = validate_sample_rate(sample_rate)
    array = validate_waveform_array(samples)
    original_channels = 1 if array.ndim == 1 else int(array.shape[1])

    # 1. Scale integer PCM before downmixing (preserves dtype semantics).
    as_float = pcm_to_float(array)

    # 2. Malformed out-of-range float input fails explicitly at A2.
    as_float = validate_float_amplitudes(
        as_float,
        tolerance=config.input_range_tolerance,
        context="input",
    )

    # 3. One mono policy, then one real resampling path.
    mono = downmix_to_mono(as_float)
    resampled = resample_polyphase(
        mono,
        sample_rate,
        config.target_sample_rate,
        method=config.resampling_method,
    )

    # 4. Outgoing contract: finite, in-range, with documented overshoot policy.
    resampled = _enforce_outgoing_range(resampled, config)

    identity = PreprocessingIdentity.from_config(config)
    return PreprocessedAudio(
        samples=np.asarray(resampled, dtype=np.float32),
        sample_rate=config.target_sample_rate,
        source_id=source_id,
        original_sample_rate=sample_rate,
        original_channels=original_channels,
        original_format=original_format,
        identity=identity,
    )


def preprocess_file(
    path: str | Path,
    *,
    source_id: str | None = None,
    config: PreprocessingConfig | None = None,
) -> PreprocessedAudio:
    """Decode a file with SoundFile and delegate to the shared array path.

    SoundFile decodes into float amplitudes; those floats are validated and
    resampled by the same code used for already-decoded arrays.  Integer PCM is
    never scaled a second time.
    """
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"audio file not found: {path}")
    try:
        import soundfile as sf

        with sf.SoundFile(str(path)) as handle:
            sample_rate = handle.samplerate
            channels = handle.channels
            format_info = f"{handle.format}/{handle.subtype}"
            frames = handle.read(dtype="float64", always_2d=False)
    except AudioLoadError:
        raise
    except Exception as exc:  # decoder or container errors
        raise AudioLoadError(f"unable to decode {path}: {exc}") from exc

    source = source_id if source_id is not None else str(path)
    return preprocess_array(
        frames,
        sample_rate,
        source_id=source,
        original_format=format_info,
        config=config,
    )
