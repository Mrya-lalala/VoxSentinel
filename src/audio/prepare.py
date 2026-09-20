"""Shared training, file prediction and C1 decode/mono/resample contract.

This is the ``voxsentinel-prep-2`` path. The separate legacy A2 array/chunk
API retains its own ``audio-v1`` identity and must not be substituted for
this path when scoring the trained research baseline:

* the caller (this module) decodes the container, scales integer PCM to
  floating amplitudes, converts channels to mono and resamples once to
  16,000 Hz;
* no denoising, loudness normalization, speed or pitch processing, and no
  codec augmentation;
* the frozen B1 encoder remains the only component that performs its
  checkpoint-specific waveform normalization;
* M4A/AAC containers that libsndfile cannot read are decoded through the
  local ``ffmpeg`` binary to float32 PCM at the original rate before the
  standard mono/resample path (documented fallback; decoder is recorded);

Gain policy (explicit and recorded — **not** "no gain applied"): no gain is
applied except the bounded overflow attenuation below.  If resampling leaves
amplitudes above 1.0 (soxr overshoot on near-full-scale input), the whole clip
is attenuated by exactly ``1/peak`` (<= 0 dB, never clipped, never upscaled);
``overflow_gain``, ``peak_before_overflow_gain`` and the affected counts are
recorded per window so the report can quantify them by class/source.  Any
audio-affecting change here invalidates dependent feature caches.

Channel policy: arithmetic mean of channels (single documented policy for
every source).  Resampling: ``soxr`` at ``HQ`` quality; its version is recorded
in the prepared manifests.  Never silently clip or boost quiet clips.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

TARGET_SAMPLE_RATE = 16000


class AudioDecodeError(RuntimeError):
    """Raised when audio cannot be decoded under the documented policy."""


@dataclass(frozen=True)
class DecodedAudio:
    """Decoded, mono, 16 kHz float32 waveform plus original-format facts."""

    samples: np.ndarray
    sample_rate: int
    original_format: str
    original_subtype: str
    original_rate: int
    original_channels: int
    original_frames: int
    resampled: bool
    channel_mixed: bool
    overflow_gain: float = 1.0
    peak_before_overflow_gain: float | None = None
    decoder: str = "libsndfile"

    @property
    def duration_seconds(self) -> float:
        return float(self.samples.shape[0]) / float(self.sample_rate)

    def original_facts(self) -> dict[str, Any]:
        return {
            "format": self.original_format,
            "subtype": self.original_subtype,
            "sample_rate": self.original_rate,
            "channels": self.original_channels,
            "frames": self.original_frames,
        }

    def processing_facts(self) -> dict[str, Any]:
        """Recorded decode/mono/resample policy facts (no hidden processing)."""
        import math

        return {
            "decoder": self.decoder,
            "resampled": self.resampled,
            "channel_policy": "mean" if self.channel_mixed else "single_channel_passthrough",
            "resampler": resampler_identity() if self.resampled else None,
            "overflow_gain": self.overflow_gain,
            "overflow_gain_db": round(20.0 * math.log10(self.overflow_gain), 6),
            "peak_before_overflow_gain": self.peak_before_overflow_gain,
            "denoising": False,
            "loudness_normalization": False,
        }


def resampler_identity() -> str:
    try:
        return f"soxr {package_version('soxr')}"
    except Exception:  # pragma: no cover - metadata rarely missing
        return "soxr (version unknown)"


def decode_mono_16k(path: str | Path, *, target_rate: int = TARGET_SAMPLE_RATE) -> DecodedAudio:
    """Decode ``path`` into validated mono float32 samples at ``target_rate``.

    Uses libsndfile first; M4A/AAC containers that libsndfile cannot read go
    through the local ``ffmpeg`` fallback (float32 PCM at the original rate,
    then the standard mono/resample path).  Raises :class:`AudioDecodeError`
    for unreadable, empty, non-finite input.  Integer PCM is scaled by the
    decoder; no additional gain is applied, except the explicit, recorded
    overflow policy: if resampling leaves amplitudes above 1.0, the whole clip
    is attenuated by exactly ``1/peak`` (never clipped, never upscaled,
    ``overflow_gain`` recorded in the manifests).
    """
    file_path = Path(path)
    decoder = "libsndfile"
    try:
        info = sf.info(file_path)
        if info.frames <= 0:
            raise AudioDecodeError("empty audio file (no frames)")
        data, rate = sf.read(file_path, dtype="float32", always_2d=True)
        original_format = info.format or "unknown"
        original_subtype = info.subtype or "unknown"
    except AudioDecodeError:
        raise
    except Exception as soundfile_error:
        data, rate, original_format, original_subtype = _decode_via_ffmpeg(file_path, soundfile_error)
        decoder = f"ffmpeg {_ffmpeg_version()}"

    if rate <= 0:
        raise AudioDecodeError(f"invalid declared sample rate {rate}")
    if data.shape[0] == 0:
        raise AudioDecodeError("decoded zero samples")
    if not np.isfinite(data).all():
        raise AudioDecodeError("decoded audio contains non-finite values")

    channel_mixed = data.shape[1] > 1
    mono = data.mean(axis=1) if channel_mixed else data[:, 0]
    mono = np.ascontiguousarray(mono, dtype=np.float32)

    resampled = rate != target_rate
    if resampled:
        try:
            import soxr
        except ImportError as error:  # pragma: no cover - environment issue
            raise AudioDecodeError("soxr is required for resampling but is not installed") from error
        mono = np.asarray(soxr.resample(mono, rate, target_rate, quality="HQ"), dtype=np.float32)

    if mono.size == 0:
        raise AudioDecodeError("decoded zero samples after resampling")
    if not np.isfinite(mono).all():
        raise AudioDecodeError("resampled audio contains non-finite values")

    # Explicit, class-independent overflow policy: resampler interpolation can
    # overshoot the true peak by a fraction of a percent for near-full-scale
    # input.  Never clip and never encode out-of-range amplitudes; attenuate by
    # exactly 1/peak (<= 0 dB, recorded in the manifests) when, and only when,
    # the resampled signal leaves [-1, 1].  No upscaling and no normalization.
    overflow_gain = 1.0
    peak_before: float | None = None
    peak = float(np.max(np.abs(mono)))
    if peak > 1.0:
        peak_before = peak
        overflow_gain = 1.0 / peak
        mono = (mono * overflow_gain).astype(np.float32)
        peak = float(np.max(np.abs(mono)))
    if peak > 1.0 + 1e-6:
        raise AudioDecodeError(f"decoded peak amplitude {peak:.6f} exceeds the valid [-1, 1] range")

    return DecodedAudio(
        samples=mono,
        sample_rate=target_rate,
        original_format=original_format,
        original_subtype=original_subtype,
        original_rate=int(rate),
        original_channels=int(data.shape[1]),
        original_frames=int(data.shape[0]),
        resampled=resampled,
        channel_mixed=channel_mixed,
        overflow_gain=overflow_gain,
        peak_before_overflow_gain=peak_before,
        decoder=decoder,
    )


def _ffmpeg_version() -> str:
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, check=True)
        first = out.stdout.splitlines()[0]
        return first.split("version", 1)[1].strip().split()[0] if "version" in first else "unknown"
    except Exception:  # pragma: no cover - environment issue
        return "unknown"


def _decode_via_ffmpeg(path: Path, soundfile_error: Exception) -> tuple[np.ndarray, int, str, str]:
    """Decode via the local ffmpeg binary; float32 PCM at the original rate."""
    if shutil.which("ffmpeg") is None:
        raise AudioDecodeError(f"libsndfile cannot read the container and ffmpeg is absent: {soundfile_error}")
    try:
        probe = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "a:0",
                "-show_entries", "stream=codec_name,sample_rate,channels",
                "-of", "json", str(path),
            ],
            capture_output=True, text=True, check=True,
        )
        streams = json.loads(probe.stdout).get("streams", [])
        if not streams:
            raise AudioDecodeError("ffprobe found no audio stream")
        stream = streams[0]
        with tempfile.TemporaryDirectory() as tmp:
            decoded = Path(tmp) / "decoded.wav"
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y", "-i", str(path),
                    "-map", "0:a:0", "-c:a", "pcm_f32le", str(decoded),
                ],
                capture_output=True, text=True, check=True,
            )
            data, rate = sf.read(decoded, dtype="float32", always_2d=True)
    except subprocess.CalledProcessError as error:
        raise AudioDecodeError(f"ffmpeg decode failed: {error.stderr.strip()[:200]}") from error
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise AudioDecodeError(f"ffprobe output unreadable: {error}") from error
    codec = str(stream.get("codec_name") or "unknown")
    return data, int(rate), f"MPEG-4 ({codec})", codec


def write_prepared_wav(path: str | Path, samples: np.ndarray, sample_rate: int = TARGET_SAMPLE_RATE) -> Path:
    """Write a mono 16 kHz lossless WAV with the FLOAT subtype (no gain applied).

    libsndfile adds a ``PEAK`` chunk whose timestamp field records the write
    moment; the timestamp is zeroed afterwards so prepared WAV bytes are fully
    deterministic (the audio payload and peak value are untouched).  This keeps
    manifest hashes reproducible across reruns.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("prepared audio must be a 1-D mono waveform")
    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(f"prepared audio must be {TARGET_SAMPLE_RATE} Hz, got {sample_rate}")
    if not np.isfinite(data).all():
        raise ValueError("refusing to write non-finite samples")
    if data.size and float(np.max(np.abs(data))) > 1.0:
        raise ValueError("refusing to write amplitudes outside [-1, 1] (no silent clipping)")
    sf.write(target, data, sample_rate, subtype="FLOAT", format="WAV")
    canonicalize_wav_peak_timestamp(target)
    return target


def canonicalize_wav_peak_timestamp(path: str | Path) -> bool:
    """Zero the PEAK-chunk timestamp; returns True when a change was applied."""
    target = Path(path)
    head = target.read_bytes()[:1024]
    index = head.find(b"PEAK")
    if index < 0 or index + 16 > len(head):
        return False
    payload = head[index + 8 : index + 16]
    if len(payload) != 8 or payload[:4] != b"\x01\x00\x00\x00":
        return False
    timestamp = head[index + 12 : index + 16]
    if timestamp == b"\x00\x00\x00\x00":
        return False
    data = bytearray(target.read_bytes())
    data[index + 12 : index + 16] = b"\x00\x00\x00\x00"
    target.write_bytes(bytes(data))
    return True


def read_prepared_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a prepared WAV; validate the frozen contract used by feature caching."""
    data, rate = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim != 1:
        raise AudioDecodeError(f"prepared audio must be mono; got shape {data.shape}")
    if rate != TARGET_SAMPLE_RATE:
        raise AudioDecodeError(f"prepared audio must be {TARGET_SAMPLE_RATE} Hz; got {rate}")
    if not np.isfinite(data).all():
        raise AudioDecodeError("prepared audio contains non-finite values")
    return np.ascontiguousarray(data, dtype=np.float32), int(rate)


__all__ = [
    "AudioDecodeError",
    "DecodedAudio",
    "TARGET_SAMPLE_RATE",
    "decode_mono_16k",
    "read_prepared_wav",
    "resampler_identity",
    "write_prepared_wav",
]
