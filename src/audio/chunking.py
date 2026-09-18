"""B1-safe chunking with truthful per-chunk status and provenance.

Default policy: contiguous non-overlapping 4-second windows at 16 kHz.  Final
short windows with at least ``min_valid_samples`` (400) samples are retained at
their actual length; tails of 1-399 samples are omitted and recorded.  A whole
recording shorter than 400 samples is explicitly "too short".

Offsets index the *resampled* 16 kHz waveform, never the original-rate array.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import PreprocessingConfig
from .errors import AudioValidationError
from .preprocessing import PreprocessedAudio

STATUS_OK = "ok"
STATUS_EMPTY = "empty"
STATUS_TOO_SHORT = "too_short"
SKIP_REASON_TOO_SHORT = "recording_too_short"
SKIP_REASON_TAIL = "tail_below_min_valid_samples"


@dataclass(frozen=True)
class SkippedSpan:
    """A span of samples that was not emitted, with the reason."""

    source_id: str
    start_sample: int
    end_sample: int
    sample_count: int
    reason: str


@dataclass(frozen=True)
class AudioChunk:
    """One encodable window plus its resampled-waveform provenance."""

    source_id: str
    samples: np.ndarray
    sample_rate: int
    start_sample: int
    end_sample: int
    is_silent: bool
    preprocessing_identity: str

    def __post_init__(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or samples.dtype != np.float32:
            raise ValueError("AudioChunk.samples must be a 1-D float32 array")
        object.__setattr__(self, "samples", samples)

    @property
    def valid_samples(self) -> int:
        return int(self.samples.size)

    @property
    def start_ms(self) -> float:
        return self.start_sample * 1000.0 / self.sample_rate

    @property
    def duration_ms(self) -> float:
        return self.valid_samples * 1000.0 / self.sample_rate


@dataclass
class ChunkResult:
    """Chunks, omitted spans, and a status that never hides why audio is empty."""

    source_id: str
    chunks: list[AudioChunk] = field(default_factory=list)
    skipped: list[SkippedSpan] = field(default_factory=list)
    status: str = STATUS_OK
    message: str = ""

    @property
    def all_silent(self) -> bool:
        return bool(self.chunks) and all(chunk.is_silent for chunk in self.chunks)


def _is_exact_silence(samples: np.ndarray) -> bool:
    return bool((samples == 0).all())


def chunk_audio(
    audio: PreprocessedAudio,
    *,
    config: PreprocessingConfig | None = None,
) -> ChunkResult:
    """Split preprocessed audio into B1-safe contiguous windows."""
    config = config or PreprocessingConfig()
    config.validate_chunking()

    total = audio.num_samples
    identity_string = audio.identity_string
    result = ChunkResult(source_id=audio.source_id)

    if total == 0:
        result.status = STATUS_EMPTY
        result.message = "decoded waveform contains zero samples"
        return result

    if total < config.min_valid_samples:
        result.status = STATUS_TOO_SHORT
        result.message = (
            f"recording has {total} samples, below the {config.min_valid_samples}-sample minimum"
        )
        result.skipped.append(
            SkippedSpan(audio.source_id, 0, total, total, SKIP_REASON_TOO_SHORT)
        )
        return result

    window = config.window_samples
    hop = config.hop_samples
    start = 0
    while start < total:
        end = min(start + window, total)
        length = end - start
        if length >= config.min_valid_samples:
            samples = audio.samples[start:end].astype(np.float32, copy=True)
            result.chunks.append(
                AudioChunk(
                    source_id=audio.source_id,
                    samples=samples,
                    sample_rate=audio.sample_rate,
                    start_sample=start,
                    end_sample=end,
                    is_silent=_is_exact_silence(samples),
                    preprocessing_identity=identity_string,
                )
            )
        else:
            result.skipped.append(
                SkippedSpan(
                    audio.source_id,
                    start,
                    end,
                    length,
                    SKIP_REASON_TAIL,
                )
            )
        start += hop

    if not result.chunks:
        # Unreachable for the contiguous non-overlapping policy because the
        # first window always meets the minimum, but kept for robustness.
        result.status = STATUS_TOO_SHORT
        result.message = "no window met the minimum valid sample count"
    return result


def collate_chunks(chunks: list[AudioChunk]) -> tuple[np.ndarray, np.ndarray]:
    """Right-pad chunks into the B1 batch contract.

    Returns ``(samples [B, S] float32, valid_lengths [B] int64)`` where
    ``valid_lengths`` are *sample* counts (not frames) and padding is finite
    zero.  An empty list raises; callers must handle the no-eligible-chunks
    case explicitly before calling this.
    """
    if not chunks:
        raise AudioValidationError("cannot collate an empty chunk list")
    max_len = max(chunk.valid_samples for chunk in chunks)
    batch = np.zeros((len(chunks), max_len), dtype=np.float32)
    lengths = np.empty(len(chunks), dtype=np.int64)
    for index, chunk in enumerate(chunks):
        batch[index, : chunk.valid_samples] = chunk.samples
        lengths[index] = chunk.valid_samples
    return batch, lengths
