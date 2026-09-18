"""Shared helpers for source adapters (run accounting, resume, exclusions)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..materialize import HashRegistry
from ..records import read_jsonl


@dataclass
class SourceRun:
    """Per-source fetch accounting returned by every adapter."""

    source_id: str
    windows: list[dict[str, Any]] = field(default_factory=list)
    recordings: list[dict[str, Any]] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    metadata_requests: int = 0
    download_bytes: int = 0
    notes: list[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        if message not in self.notes:
            self.notes.append(message)


@dataclass
class ResumeState:
    """Existing manifest state for one source (resume-safe reruns)."""

    recordings: dict[str, dict[str, Any]]
    windows: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    registry: HashRegistry


def load_resume_state(
    *,
    recordings_path: Path,
    windows_path: Path,
    exclusions_path: Path,
    force: bool = False,
) -> ResumeState:
    """Read a source's previous manifests, honoring ``force``."""
    if force:
        return ResumeState(recordings={}, windows=[], exclusions=[], registry=HashRegistry())
    recordings = {str(row.get("recording_id")): row for row in read_jsonl(recordings_path)}
    windows = read_jsonl(windows_path)
    exclusions = read_jsonl(exclusions_path)
    return ResumeState(
        recordings=recordings,
        windows=windows,
        exclusions=exclusions,
        registry=HashRegistry.from_records(recordings.values()),
    )


def dedupe_exclusions(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Drop repeated (recording_id, reason) exclusion rows from reruns."""
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("recording_id")), str(row.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        result.append(dict(row))
    return result


def prune_exclusions(rows: Sequence[Mapping[str, Any]], materialized_ids: set[str]) -> list[dict[str, Any]]:
    """Drop exclusion rows for recordings that have since been materialized."""
    return [dict(row) for row in rows if str(row.get("recording_id")) not in materialized_ids]


def exclusion_row(
    *,
    recording_id: str,
    dataset_id: str,
    source_file: str,
    original_split: str | None,
    reason: str,
    status: str = "excluded",
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "recording_id": recording_id,
        "dataset_id": dataset_id,
        "source_file": source_file,
        "original_split": original_split,
        "reason": reason,
        "status": status,
    }
    row.update(extra)
    return row


__all__ = [
    "ResumeState",
    "SourceRun",
    "dedupe_exclusions",
    "exclusion_row",
    "load_resume_state",
    "prune_exclusions",
]
