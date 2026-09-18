"""A2 audio preprocessing: shared conversion, resampling, and chunking.

This package owns the audio boundary only: file/array decoding, integer PCM
scaling, mono downmix, resampling to 16 kHz, chunking, and truthful metadata.
It adds no denoising, VAD, silence trimming, AGC, or amplitude augmentation.
Checkpoint-specific waveform normalization remains exclusively inside B1.
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
]
