"""Shared materialization: store originals, decode, select, hash, write WAV.

Adapters hand over the *original bytes* of each selected recording (downloaded
or extracted from an official archive); this module preserves them under
``raw/``, decodes under the documented adapter policy, selects one
class-independent window, writes the prepared mono 16 kHz FLOAT WAV under
``prepared/`` and computes the three hashes recorded in the manifests:

* ``original_sha256``  - hash of the untouched source bytes;
* ``decoded_sha256``   - hash of the full decoded mono 16 kHz float32 stream
  (canonical form for exact-duplicate detection);
* ``prepared_sha256``  - hash of the written prepared WAV file.

Exact hashing cannot establish absence of near-duplicates or unidentified
shared speakers; this limitation is recorded in the preparation report.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from ..audio.prepare import DecodedAudio, decode_mono_16k, write_prepared_wav
from .config import WindowPolicy
from .windows import WindowSelection, select_window


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_float32(samples: np.ndarray) -> str:
    array = np.ascontiguousarray(samples, dtype="<f4")
    return hashlib.sha256(array.tobytes()).hexdigest()


@dataclass(frozen=True)
class MaterializedWindow:
    """Result of decoding one recording and writing one prepared window."""

    raw_path: Path
    prepared_path: Path
    original_sha256: str
    decoded_sha256: str
    prepared_sha256: str
    selection: WindowSelection
    decoded: DecodedAudio

    def window_facts(self, *, original_rate: int) -> dict[str, Any]:
        selection = self.selection
        start_seconds = selection.start_sample / self.decoded.sample_rate
        end_seconds = selection.end_sample / self.decoded.sample_rate
        return {
            "start_seconds": round(start_seconds, 6),
            "end_seconds": round(end_seconds, 6),
            "start_sample": int(selection.start_sample),
            "end_sample": int(selection.end_sample),
            "sample_rate": int(self.decoded.sample_rate),
            "num_samples": int(selection.end_sample - selection.start_sample),
            "duration_seconds": round(end_seconds - start_seconds, 6),
            "start_sample_original_rate": int(round(start_seconds * original_rate)),
            "end_sample_original_rate": int(round(end_seconds * original_rate)),
            "original_sample_rate": int(original_rate),
            "rms": round(selection.rms, 8),
            "peak": round(selection.peak, 8),
            "silence_flag": bool(selection.silence_flag),
            "active_fraction": round(selection.active_fraction, 6),
        }

    def prepared_facts(self, dataset_root: Path) -> dict[str, Any]:
        return {
            "path": str(self.prepared_path.resolve().relative_to(Path(dataset_root).resolve())),
            "format": "WAV",
            "subtype": "FLOAT",
            "sample_rate": int(self.decoded.sample_rate),
            "channels": 1,
            "sha256": self.prepared_sha256,
            "processing": self.decoded.processing_facts(),
        }

    def decoded_facts(self) -> dict[str, Any]:
        return {
            "sample_rate": self.decoded.sample_rate,
            "num_samples": int(self.decoded.samples.shape[0]),
            "duration_seconds": round(self.decoded.duration_seconds, 6),
            "sha256": self.decoded_sha256,
            "processing": self.decoded.processing_facts(),
        }


def store_original_bytes(destination: str | Path, data: bytes) -> Path:
    """Atomically store untouched source bytes; skip rewrite when already present."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size == len(data):
        if sha256_file(target) == sha256_bytes(data):
            return target
    handle = tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=f".{target.name}.", delete=False)
    try:
        with handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, target)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return target


def materialize_window(
    raw_path: Path,
    prepared_path: Path,
    *,
    policy: WindowPolicy,
) -> MaterializedWindow:
    """Decode the stored original and write exactly one prepared window."""
    decoded = decode_mono_16k(raw_path)
    selection = select_window(decoded.samples, decoded.sample_rate, policy)
    if not selection.usable:
        raise ValueError(f"unusable recording: {selection.reason}")
    window_samples = decoded.samples[selection.start_sample : selection.end_sample]
    write_prepared_wav(prepared_path, window_samples, decoded.sample_rate)
    return MaterializedWindow(
        raw_path=raw_path,
        prepared_path=prepared_path,
        original_sha256=sha256_file(raw_path),
        decoded_sha256=sha256_float32(decoded.samples),
        prepared_sha256=sha256_file(prepared_path),
        selection=selection,
        decoded=decoded,
    )


class HashRegistry:
    """Exact-duplicate registry over decoded and prepared hashes already recorded."""

    def __init__(self, known_decoded: Iterable[str] = (), known_prepared: Iterable[str] = ()) -> None:
        self.decoded: set[str] = set(known_decoded)
        self.prepared: set[str] = set(known_prepared)

    @classmethod
    def from_records(cls, records: Iterable[Mapping[str, Any]]) -> "HashRegistry":
        decoded: set[str] = set()
        prepared: set[str] = set()
        for record in records:
            value = record.get("decoded_sha256")
            if value:
                decoded.add(str(value))
            value = (record.get("prepared_audio") or {}).get("sha256")
            if value:
                prepared.add(str(value))
        return cls(decoded, prepared)

    def check(self, materialized: MaterializedWindow) -> str | None:
        """Return an exclusion reason when the recording/window is an exact duplicate."""
        if materialized.decoded_sha256 in self.decoded:
            return "exact_duplicate_decoded_audio"
        if materialized.prepared_sha256 in self.prepared:
            return "exact_duplicate_prepared_window"
        return None

    def register(self, materialized: MaterializedWindow) -> None:
        self.decoded.add(materialized.decoded_sha256)
        self.prepared.add(materialized.prepared_sha256)


__all__ = [
    "HashRegistry",
    "MaterializedWindow",
    "materialize_window",
    "sha256_bytes",
    "sha256_file",
    "sha256_float32",
    "store_original_bytes",
]
