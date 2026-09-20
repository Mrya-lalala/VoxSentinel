"""Audio APIs for A2 preprocessing and the trained pilot preparation path.

Keep the pilot's soxr-based preparation contract separate from A2's
polyphase resampling/chunking API; exporting both does not migrate models.
"""

from .config import PreprocessingConfig, load_preprocessing_config
from .conversion import downmix_to_mono, pcm_to_float, validate_sample_rate
from .errors import AudioError, AudioLoadError, AudioValidationError
from .identity import PreprocessingIdentity
from .preprocessing import PreprocessedAudio, preprocess_array, preprocess_file
from .resampling import resample_polyphase
from .chunking import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_TOO_SHORT,
    AudioChunk,
    ChunkResult,
    SkippedSpan,
    chunk_audio,
    collate_chunks,
)
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
    "AudioChunk",
    "AudioError",
    "AudioLoadError",
    "AudioValidationError",
    "ChunkResult",
    "PreprocessedAudio",
    "PreprocessingConfig",
    "PreprocessingIdentity",
    "SkippedSpan",
    "STATUS_EMPTY",
    "STATUS_OK",
    "STATUS_TOO_SHORT",
    "chunk_audio",
    "collate_chunks",
    "downmix_to_mono",
    "load_preprocessing_config",
    "pcm_to_float",
    "preprocess_array",
    "preprocess_file",
    "resample_polyphase",
    "validate_sample_rate",
    "AudioDecodeError",
    "DecodedAudio",
    "TARGET_SAMPLE_RATE",
    "canonicalize_wav_peak_timestamp",
    "decode_mono_16k",
    "read_prepared_wav",
    "resampler_identity",
    "write_prepared_wav",
]
