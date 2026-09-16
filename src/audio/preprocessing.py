"""Shared file-to-chunks audio path for encoder and inference callers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as ta_functional


TARGET_SAMPLE_RATE = 16_000
PREPROCESSING_VERSION = "audio-v1-16khz-mono-chunk4s"


class AudioStatus(str, Enum):
    """Outcome of decoding and basic usable-audio checks."""

    USABLE = "usable"
    SILENT = "silent"
    EMPTY = "empty"


@dataclass(frozen=True)
class PreprocessedAudio:
    """A complete decoded waveform after the shared preprocessing policy."""

    source_id: str
    samples: np.ndarray
    original_sample_rate: int
    sample_rate: int
    original_channels: int
    status: AudioStatus
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError("samples must be a one-dimensional float32 waveform")
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("preprocessed audio must be sampled at 16 kHz")
        if self.original_channels < 1:
            raise ValueError("original_channels must be positive")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True)
class AudioChunk:
    """One non-overlapping chunk with source-level sample provenance."""

    source_id: str
    samples: np.ndarray
    start_sample: int
    end_sample: int
    valid_samples: int
    sample_rate: int = TARGET_SAMPLE_RATE
    status: AudioStatus = AudioStatus.USABLE
    preprocessing_version: str = PREPROCESSING_VERSION

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError("chunk samples must be a one-dimensional float32 waveform")
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("chunk sample offsets must be ordered and non-empty")
        if self.valid_samples != samples.size or self.end_sample - self.start_sample != samples.size:
            raise ValueError("chunk offsets and valid sample count must match samples")
        if self.sample_rate != TARGET_SAMPLE_RATE:
            raise ValueError("chunks must be sampled at 16 kHz")
        object.__setattr__(self, "samples", samples)

    @property
    def start_ms(self) -> float:
        return self.start_sample * 1000.0 / self.sample_rate

    @property
    def end_ms(self) -> float:
        return self.end_sample * 1000.0 / self.sample_rate


def load_audio(path: str | Path, *, source_id: str | None = None) -> PreprocessedAudio:
    """Decode a soundfile-supported file and convert it to mono 16 kHz PCM."""

    file_path = Path(path)
    source = source_id or str(file_path)
    samples, sample_rate = sf.read(file_path, dtype="float32", always_2d=True)
    if samples.size == 0:
        mono = np.empty(0, dtype=np.float32)
        channels = samples.shape[1]
    else:
        mono = np.asarray(samples, dtype=np.float32).mean(axis=1, dtype=np.float32)
        channels = samples.shape[1]
    if not np.isfinite(mono).all():
        raise ValueError(f"{file_path}: decoded audio contains non-finite samples")
    if sample_rate != TARGET_SAMPLE_RATE and mono.size:
        waveform = torch.from_numpy(mono)
        resampled = ta_functional.resample(waveform, sample_rate, TARGET_SAMPLE_RATE)
        mono = resampled.numpy().astype(np.float32, copy=False)
    status = _audio_status(mono)
    return PreprocessedAudio(source, mono, sample_rate, TARGET_SAMPLE_RATE, channels, status)


def preprocess_audio(
    samples: np.ndarray,
    sample_rate: int,
    *,
    source_id: str = "audio",
) -> PreprocessedAudio:
    """Apply the same mono, scaling, and resampling policy to decoded samples."""

    array = np.asarray(samples)
    if array.ndim == 1:
        channels = 1
        mono = array
    elif array.ndim == 2:
        channels = array.shape[1]
        mono = array.mean(axis=1, dtype=np.float32)
    else:
        raise ValueError("samples must have shape [samples] or [samples, channels]")
    mono = _to_float32_pcm(mono)
    if not np.isfinite(mono).all():
        raise ValueError("audio contains non-finite samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if sample_rate != TARGET_SAMPLE_RATE and mono.size:
        mono = ta_functional.resample(torch.from_numpy(mono), sample_rate, TARGET_SAMPLE_RATE).numpy()
        mono = mono.astype(np.float32, copy=False)
    return PreprocessedAudio(
        source_id, mono, sample_rate, TARGET_SAMPLE_RATE, channels, _audio_status(mono)
    )


def chunk_audio(audio: PreprocessedAudio, *, chunk_seconds: float = 4.0) -> list[AudioChunk]:
    """Split audio into contiguous chunks, retaining a final short chunk."""

    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    chunk_samples = round(chunk_seconds * audio.sample_rate)
    if chunk_samples < 400:
        raise ValueError("chunk_seconds must provide at least 400 samples")
    if audio.samples.size == 0:
        return []
    chunks = []
    for start in range(0, audio.samples.size, chunk_samples):
        end = min(start + chunk_samples, audio.samples.size)
        chunks.append(
            AudioChunk(
                audio.source_id,
                audio.samples[start:end].copy(),
                start,
                end,
                end - start,
                audio.sample_rate,
                audio.status,
                audio.preprocessing_version,
            )
        )
    return chunks


def _audio_status(samples: np.ndarray) -> AudioStatus:
    if samples.size == 0:
        return AudioStatus.EMPTY
    return AudioStatus.SILENT if np.all(samples == 0) else AudioStatus.USABLE


def _to_float32_pcm(samples: np.ndarray) -> np.ndarray:
    if np.issubdtype(samples.dtype, np.integer):
        info = np.iinfo(samples.dtype)
        scale = max(abs(info.min), info.max)
        return samples.astype(np.float32) / scale
    return np.asarray(samples, dtype=np.float32)