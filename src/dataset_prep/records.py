"""Unified record schema and atomic JSONL I/O for the preparation pilot.

Two record kinds are written to the dataset root:

* recording inventory rows: one per original source recording that was decoded
  and materialized (``manifests/recordings.jsonl``);
* selected-window rows: one per prepared training/evaluation window, grouped
  into pool-specific manifests (``manifests/windows.<pool>.jsonl``).

Both use plain JSON objects with a stable key set.  ``null`` means unknown.
No tokens, signed URLs or personal metadata are stored.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping

# Pool / status vocabulary.  A record's ``status`` is exactly one of these.
STATUS_AUDIO_READY = "audio_ready"
STATUS_FEATURES_READY = "features_ready"
STATUS_UNPAIRED_CANDIDATE = "unpaired_candidate"
STATUS_ACCESS_REQUIRED = "access_required"
STATUS_REQUIRES_LARGE_DOWNLOAD = "requires_large_download"
STATUS_MISSING_SOURCE = "missing_source"
STATUS_EXCLUDED = "excluded"

WINDOW_POOLS = (
    "core",
    "supplementary",
    "external_eval",
    "fallback_baseline",
    "unpaired_candidate",
    "missing_coverage",
    "excluded",
)

_LABEL_NAMES = {0: "genuine", 1: "spoof", None: "unknown"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file; missing file yields an empty list."""
    file_path = Path(path)
    if not file_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{file_path}:{line_number}: invalid JSON row") from error
            if not isinstance(value, dict):
                raise ValueError(f"{file_path}:{line_number}: JSONL rows must be objects")
            rows.append(value)
    return rows


def write_jsonl_atomic(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> Path:
    """Atomically replace a JSONL file with ``rows`` (write to temp, then rename)."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    _atomic_write_text(file_path, payload)
    return file_path


def write_json_atomic(path: str | Path, value: Mapping[str, Any] | list[Any]) -> Path:
    """Atomically replace a JSON file."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(file_path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return file_path


def read_json(path: str | Path, default: Any = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    """Append one JSON row; used for resume-friendly streaming writes."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _atomic_write_text(path: Path, payload: str) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=directory, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def label_name(label: int | None) -> str:
    return _LABEL_NAMES.get(label, "unknown")


def build_window_record(
    *,
    window_id: str,
    recording_id: str,
    dataset_id: str,
    source_url: str,
    source_file: str,
    original_split: str | None,
    pool: str,
    split: str | None,
    label: int | None,
    label_source: str,
    spoken_language: str | None,
    native_language: str | None,
    speaker_ids: Mapping[str, Any],
    generator: str | None,
    generator_version: str | None,
    parent_refs: Mapping[str, Any],
    original_audio: Mapping[str, Any],
    prepared_audio: Mapping[str, Any],
    window: Mapping[str, Any],
    preprocessing_version: str,
    license_note: str | None,
    access_status: str,
    status: str,
    exclusion_reason: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build one selected-window record with the stable unified schema."""
    if pool not in WINDOW_POOLS:
        raise ValueError(f"Unknown pool '{pool}'; expected one of {WINDOW_POOLS}.")
    return {
        "window_id": window_id,
        "recording_id": recording_id,
        "dataset_id": dataset_id,
        "source_url": source_url,
        "source_file": source_file,
        "original_split": original_split,
        "pool": pool,
        "split": split,
        "label": label,
        "label_name": label_name(label),
        "label_source": label_source,
        "spoken_language": spoken_language,
        "native_language": native_language,
        "speaker_ids": dict(speaker_ids),
        "generator": generator,
        "generator_version": generator_version,
        "parent_refs": dict(parent_refs),
        "original_audio": dict(original_audio),
        "prepared_audio": dict(prepared_audio),
        "window": dict(window),
        "preprocessing_version": preprocessing_version,
        "license": license_note,
        "access_status": access_status,
        "status": status,
        "exclusion_reason": exclusion_reason,
        "notes": notes,
    }


def build_recording_record(
    *,
    recording_id: str,
    dataset_id: str,
    source_url: str,
    source_file: str,
    original_split: str | None,
    label: int | None,
    spoken_language: str | None,
    native_language: str | None,
    speaker_ids: Mapping[str, Any],
    generator: str | None,
    generator_version: str | None,
    parent_refs: Mapping[str, Any],
    original_audio: Mapping[str, Any],
    decoded: Mapping[str, Any],
    license_note: str | None,
    notes: str | None = None,
) -> dict[str, Any]:
    """Build one recording-inventory row."""
    return {
        "recording_id": recording_id,
        "dataset_id": dataset_id,
        "source_url": source_url,
        "source_file": source_file,
        "original_split": original_split,
        "label": label,
        "label_name": label_name(label),
        "spoken_language": spoken_language,
        "native_language": native_language,
        "speaker_ids": dict(speaker_ids),
        "generator": generator,
        "generator_version": generator_version,
        "parent_refs": dict(parent_refs),
        "original_audio": dict(original_audio),
        "decoded": dict(decoded),
        "license": license_note,
        "notes": notes,
    }


__all__ = [
    "STATUS_ACCESS_REQUIRED",
    "STATUS_AUDIO_READY",
    "STATUS_EXCLUDED",
    "STATUS_FEATURES_READY",
    "STATUS_MISSING_SOURCE",
    "STATUS_REQUIRES_LARGE_DOWNLOAD",
    "STATUS_UNPAIRED_CANDIDATE",
    "WINDOW_POOLS",
    "append_jsonl",
    "build_recording_record",
    "build_window_record",
    "label_name",
    "read_json",
    "read_jsonl",
    "write_json_atomic",
    "write_jsonl_atomic",
]
