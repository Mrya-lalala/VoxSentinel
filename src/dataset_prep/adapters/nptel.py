"""NPTEL2020 adapter: bounded import of the official published pure set.

The full NPTEL2020 corpus is terabyte-scale and is never downloaded.  Only the
official 189 MB ``nptel-pure-set.tar.gz`` release asset (v0.1) is used, and only
the first usable chunks are materialized (default ceiling: 10).

Lecturer identity is recorded when the pure set exposes it; otherwise the clips
stay supplementary/inventory-only and are explicitly *not* claimed to be
speaker-disjoint.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Mapping

from .. import httpio
from ..budget import BudgetExceeded, DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ...audio.prepare import AudioDecodeError
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "nptel"
PURE_SET_URL = "https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset/releases/download/v0.1/nptel-pure-set.tar.gz"
SOURCE_PAGE = "https://github.com/AI4Bharat/NPTEL2020-Indian-English-Speech-Dataset"

_SPEAKER_KEYS = ("speaker", "speaker_id", "lecturer", "user_id", "artist", "voice")


def _lecturer_from_metadata(metadata: Mapping[str, Any] | None) -> str | None:
    if not metadata:
        return None
    for key in _SPEAKER_KEYS:
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def fetch_nptel(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    force: bool = False,
) -> SourceRun:
    """Download the pure set and materialize up to ``max_windows`` chunks."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    release = cfg.release(DATASET_ID)
    max_windows = int(cfg.quota("nptel").get("max_windows", 10))
    recordings_path = cfg.manifests_dir / f"source_recordings.{DATASET_ID}.jsonl"
    windows_path = cfg.manifests_dir / f"source_windows.{DATASET_ID}.jsonl"
    exclusions_path = cfg.manifests_dir / f"source_exclusions.{DATASET_ID}.jsonl"
    state = load_resume_state(
        recordings_path=recordings_path,
        windows_path=windows_path,
        exclusions_path=exclusions_path,
        force=force,
    )
    registry = state.registry
    run.recordings = list(state.recordings.values())
    run.windows = list(state.windows)
    run.exclusions = list(state.exclusions)
    existing_ids = set(state.recordings)

    archive_path = cfg.staging_dir / DATASET_ID / "nptel-pure-set.tar.gz"
    try:
        result = httpio.download_resumable(
            PURE_SET_URL,
            archive_path,
            ledger=ledger,
            source_id=DATASET_ID,
            timeout=cfg.budget.request_timeout_seconds,
        )
    except BudgetExceeded as error:
        run.note(f"budget reached before the pure-set download: {error}")
        write_jsonl_atomic(recordings_path, run.recordings)
        write_jsonl_atomic(windows_path, run.windows)
        write_jsonl_atomic(exclusions_path, dedupe_exclusions(run.exclusions))
        return run
    run.download_bytes = result.transferred_bytes
    run.note(f"pure-set archive at {archive_path.name}: {result.transferred_bytes} new bytes")

    accepted = sum(1 for row in state.windows if row.get("dataset_id") == DATASET_ID)
    metadata_by_stem: dict[str, dict[str, Any]] = {}
    lecturer_resolved = False
    try:
        # Pass 1: collect chunk metadata without extracting audio.
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile() or not member.name.lower().endswith(".json"):
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                try:
                    metadata = json.loads(handle.read().decode("utf-8", errors="replace"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(metadata, dict):
                    metadata_by_stem[Path(member.name).stem] = metadata

        # Pass 2: materialize the first usable chunks until the ceiling is met.
        with tarfile.open(archive_path, mode="r:gz") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                name = member.name
                if not name.lower().endswith(".wav"):
                    continue
                window_id = f"{DATASET_ID}-{Path(name).stem}"
                if window_id in existing_ids:
                    continue
                if accepted >= max_windows:
                    break
                if member.size > 10 * 1024 * 1024:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_file=name,
                            original_split=None,
                            reason=f"oversized_member_{member.size}_bytes",
                        )
                    )
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                payload = handle.read()
                raw_path = cfg.raw_dir / DATASET_ID / Path(name).name
                store_original_bytes(raw_path, payload)
                prepared_path = cfg.prepared_dir / DATASET_ID / f"{window_id}.wav"
                try:
                    materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
                except (AudioDecodeError, ValueError) as error:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_file=name,
                            original_split=None,
                            reason=f"decode_or_window_failed: {error}",
                        )
                    )
                    continue
                duplicate = registry.check(materialized)
                if duplicate:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_file=name,
                            original_split=None,
                            reason=duplicate,
                        )
                    )
                    prepared_path.unlink(missing_ok=True)
                    continue
                registry.register(materialized)

                metadata = metadata_by_stem.get(Path(name).stem, {})
                lecturer = _lecturer_from_metadata(metadata)
                if lecturer is not None:
                    lecturer_resolved = True
                speaker_ids = {"lecturer": lecturer}
                parent_refs = {
                    "archive_path_in_tar": name,
                    "metadata_present": bool(metadata),
                    "video_id": metadata.get("video_id") or metadata.get("parent_video"),
                    "timestamp_start": metadata.get("start_time") or metadata.get("start"),
                    "timestamp_end": metadata.get("end_time") or metadata.get("end"),
                    "dataset_revision": release.get("revision"),
                }
                recording_record = build_recording_record(
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_url=SOURCE_PAGE,
                    source_file=name,
                    original_split=None,
                    label=0,
                    spoken_language="en",
                    native_language=None,
                    speaker_ids=speaker_ids,
                    generator=None,
                    generator_version=None,
                    parent_refs=parent_refs,
                    original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                    decoded=materialized.decoded_facts(),
                    license_note=str(release.get("license")),
                    notes="published pure set sample; identity resolution recorded in parent_refs",
                )
                window_record = build_window_record(
                    window_id=window_id,
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_url=SOURCE_PAGE,
                    source_file=name,
                    original_split=None,
                    pool="supplementary",
                    split=None,
                    label=0,
                    label_source="official NPTEL2020 pure set (genuine lecture speech)",
                    spoken_language="en",
                    native_language=None,
                    speaker_ids=speaker_ids,
                    generator=None,
                    generator_version=None,
                    parent_refs=parent_refs,
                    original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                    prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                    window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                    preprocessing_version=str(cfg.preprocessing.get("config_version")),
                    license_note=str(release.get("license")),
                    access_status="public",
                    status="audio_ready",
                    notes=(
                        "supplementary/inventory-only; lecturer identity resolved"
                        if lecturer is not None
                        else "supplementary/inventory-only; lecturer identity unresolved so splits are NOT speaker-disjoint"
                    ),
                )
                run.recordings.append(recording_record)
                run.windows.append(window_record)
                existing_ids.add(window_id)
                accepted += 1
    except (tarfile.TarError, OSError) as error:
        run.note(f"pure-set extraction stopped: {error}")

    run.note(
        f"accepted {accepted}/{max_windows} chunks; lecturer identity "
        + ("resolved for at least one chunk" if lecturer_resolved else "UNRESOLVED (inventory-only)")
    )
    write_jsonl_atomic(recordings_path, run.recordings)
    write_jsonl_atomic(windows_path, run.windows)
    write_jsonl_atomic(exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, existing_ids)))
    return run


__all__ = ["DATASET_ID", "fetch_nptel"]
