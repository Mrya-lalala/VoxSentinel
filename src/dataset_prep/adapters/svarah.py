"""Svarah adapter: genuine Indian-English read speech for the external evaluation pool.

Svarah (AI4Bharat) distributes one ``test`` split as three parquet files whose
rows carry the audio bytes, a transcript and speaker metadata (gender, age
group, primary language, native state/district).  There is no spoof material in
the dataset, so every selected window is genuine (``label: 0``) and the pool is
``external_eval``: it must never be merged into train/validation.

Selection rules applied here:

* rows are read row-group by row-group from a fixed, documented group plan that
  spreads reads across all three files; the plan is deterministic and capped by
  ``audio_byte_cap`` so the acquisition ledger stays bounded;
* every window is one whole upstream recording (one chunk), duration recorded
  and windowed with the shared strict 4 s policy;
* speakers are used at most once while a fresh speaker is available, states are
  preferred when not yet covered, and at most two windows per speaker overall;
* the upstream revision SHA is resolved at acquisition and stored in every
  parent_refs block; no token is ever written to the artifacts.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from ..budget import DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ...audio.prepare import AudioDecodeError
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "svarah"
HF_DATASET = "ai4bharat/Svarah"
DATA_DIR = "data"
FILES = [
    "test-00000-of-00003.parquet",
    "test-00001-of-00003.parquet",
    "test-00002-of-00003.parquet",
]
# (file index, row group) in read order; group 0 of each file is skipped because
# it was used for the reconnaissance sketch recorded in the build report.
GROUP_PLAN: list[tuple[int, int]] = [
    (0, 1),
    (1, 4),
    (2, 2),
    (0, 8),
    (1, 12),
    (2, 10),
    (0, 15),
    (1, 20),
    (2, 17),
    (0, 22),
]
COLUMNS = [
    "audio_filepath",
    "duration",
    "text",
    "gender",
    "age-group",
    "primary_language",
    "native_place_state",
    "native_place_district",
]


class SvarahError(RuntimeError):
    pass


class _CountingFile:
    """Wrap an HfFileSystem file and count delivered bytes (charged by caller)."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.bytes = 0

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        data = self._inner.read(size)
        self.bytes += len(data)
        return data

    def readinto(self, buffer) -> int:  # type: ignore[override]
        view = memoryview(buffer)
        data = self._inner.read(len(view))
        if data:
            view[: len(data)] = data
            self.bytes += len(data)
        return len(data)

    def seek(self, *args: Any) -> int:
        return self._inner.seek(*args)

    def tell(self) -> int:
        return self._inner.tell()

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._inner.close()

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - delegate
        return getattr(self._inner, name)


def dataset_revision() -> str:
    """Resolve the Hub commit SHA for the current acquisition (no token stored)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(HF_DATASET)
    if not info.sha:
        raise SvarahError("could not resolve dataset revision SHA")
    return info.sha


def parse_speaker(audio_path: str) -> dict[str, Any]:
    """Parse ``<record>_<code>_chunk_<n>.wav`` style names."""
    stem = Path(str(audio_path)).stem
    parts = stem.split("_")
    record = parts[0] if parts else stem
    code = parts[1] if len(parts) > 1 else stem
    chunk = ""
    if "chunk" in parts:
        index = parts.index("chunk")
        if index + 1 < len(parts):
            chunk = parts[index + 1]
    return {"record": record, "speaker": code, "chunk": chunk}


def fetch_svarah(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    force: bool = False,
) -> SourceRun:
    """Select and materialize up to ``target_windows`` Svarah recordings."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    quota = cfg.quota("svarah")
    release = cfg.release(DATASET_ID)
    target = int(quota.get("target_windows", 20))
    min_seconds = float(quota.get("min_duration_seconds", 1.0))
    max_seconds = float(quota.get("max_duration_seconds", 30.0))
    byte_cap = int(quota.get("audio_byte_cap", 200_000_000))
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

    def finalize() -> SourceRun:
        write_jsonl_atomic(recordings_path, run.recordings)
        write_jsonl_atomic(windows_path, run.windows)
        write_jsonl_atomic(
            exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, existing_ids))
        )
        return run

    if len(run.windows) >= target:
        run.note(f"already satisfied: {len(run.windows)}/{target} windows on disk")
        return finalize()

    revision = dataset_revision()
    destination = cfg.raw_dir / DATASET_ID
    destination.mkdir(parents=True, exist_ok=True)
    prepared_dir = cfg.prepared_dir / DATASET_ID
    release_url = str(release.get("source_url"))
    license_note = str(release.get("license"))

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    used_speakers: dict[str, int] = {}
    used_states: set[str] = set()
    total = len(run.windows)
    groups_read = 0
    bytes_read = 0

    plan = list(GROUP_PLAN)
    for position, (file_index, group) in enumerate(plan):
        if total >= target:
            break
        if bytes_read >= byte_cap:
            run.note(f"audio byte cap reached before group plan completed ({bytes_read} bytes)")
            break
        remaining_plan = len(plan) - position
        remaining = target - total
        per_group = max(1, min(6, math.ceil(remaining / max(1, remaining_plan))))
        source_path = f"datasets/{HF_DATASET}/{DATA_DIR}/{FILES[file_index]}"
        counted = _CountingFile(fs.open(source_path, "rb"))
        before = counted.bytes
        rows: list[dict[str, Any]] = []
        try:
            parquet = pq.ParquetFile(counted)
            if group >= parquet.metadata.num_row_groups:
                run.note(f"group {group} missing in {FILES[file_index]}; skipped")
                continue
            table = parquet.read_row_group(group, columns=COLUMNS).to_pydict()
            for index, cell in enumerate(table["audio_filepath"]):
                duration = float(table["duration"][index])
                if not (min_seconds <= duration <= max_seconds):
                    continue
                payload = cell.get("bytes") if isinstance(cell, Mapping) else None
                if not payload:
                    continue
                audio_path = str(cell.get("path") or "")
                parsed = parse_speaker(audio_path)
                rows.append(
                    {
                        "audio_path": audio_path,
                        "payload": bytes(payload),
                        "duration": duration,
                        "gender": table["gender"][index],
                        "age_group": table["age-group"][index],
                        "primary_language": table["primary_language"][index],
                        "state": table["native_place_state"][index],
                        "district": table["native_place_district"][index],
                        "text": table["text"][index],
                        **parsed,
                    }
                )
        finally:
            delta = counted.bytes - before
            counted.close()
        if delta:
            ledger.charge(DATASET_ID, delta)
            bytes_read += delta
            run.download_bytes += delta
        groups_read += 1

        rows.sort(
            key=lambda row: (
                str(row["state"]) in used_states,
                used_speakers.get(str(row["speaker"]), 0),
                str(row["state"]),
                str(row["speaker"]),
                str(row["audio_path"]),
            )
        )
        picked = 0
        for row in rows:
            if picked >= per_group or total >= target:
                break
            speaker = str(row["speaker"])
            if used_speakers.get(speaker, 0) >= 1 and any(
                used_speakers.get(str(other["speaker"]), 0) == 0 for other in rows
            ):
                continue
            if used_speakers.get(speaker, 0) >= 2:
                continue
            window_id = f"svarah-{Path(str(row['audio_path'])).stem}"
            if window_id in existing_ids:
                continue
            raw_path = destination / Path(str(row["audio_path"])).name
            store_original_bytes(raw_path, row["payload"])
            prepared_path = prepared_dir / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
            except (AudioDecodeError, ValueError) as error:
                run.exclusions.append(
                    exclusion_row(
                        recording_id=window_id,
                        dataset_id=DATASET_ID,
                        source_file=str(row["audio_path"]),
                        original_split="test",
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
                        source_file=str(row["audio_path"]),
                        original_split="test",
                        reason=duplicate,
                    )
                )
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)

            speaker_ids = {"speaker": speaker, "record": str(row["record"])}
            if row.get("gender"):
                speaker_ids["gender"] = str(row["gender"])
            parent_refs = {
                "dataset_revision": revision,
                "audio_filepath": str(row["audio_path"]),
                "speaker_code": speaker,
                "source_record": str(row["record"]),
                "chunk": str(row["chunk"]),
                "upstream_duration_seconds": float(row["duration"]),
                "native_place_state": row["state"],
                "native_place_district": row["district"],
                "primary_language": row["primary_language"],
                "age_group": row["age_group"],
                "transcript": row["text"],
            }
            notes = "external evaluation pool; genuine human read speech (no spoof material upstream)"
            recording_record = build_recording_record(
                recording_id=window_id,
                dataset_id=DATASET_ID,
                source_url=release_url,
                source_file=str(row["audio_path"]),
                original_split="test",
                label=0,
                spoken_language="en",
                native_language=str(row["primary_language"]) if row.get("primary_language") else None,
                speaker_ids=speaker_ids,
                generator=None,
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                decoded=materialized.decoded_facts(),
                license_note=license_note,
                notes=notes,
            )
            window_record = build_window_record(
                window_id=window_id,
                recording_id=window_id,
                dataset_id=DATASET_ID,
                source_url=release_url,
                source_file=str(row["audio_path"]),
                original_split="test",
                pool="external_eval",
                split=None,
                label=0,
                label_source="official Svarah distribution (genuine read speech; no spoof labels exist)",
                spoken_language="en",
                native_language=str(row["primary_language"]) if row.get("primary_language") else None,
                speaker_ids=speaker_ids,
                generator=None,
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                preprocessing_version=str(cfg.preprocessing.get("config_version")),
                license_note=license_note,
                access_status="public",
                status="audio_ready",
                notes=notes,
            )
            run.recordings.append(recording_record)
            run.windows.append(window_record)
            existing_ids.add(window_id)
            used_speakers[speaker] = used_speakers.get(speaker, 0) + 1
            used_states.add(str(row["state"]))
            total += 1
            picked += 1

    states = len({str(row.get("parent_refs", {}).get("native_place_state")) for row in run.windows})
    speakers = len({str(row.get("parent_refs", {}).get("speaker_code")) for row in run.windows})
    run.note(
        f"selected {len(run.windows)}/{target} windows; speakers={speakers} states={states} "
        f"row_groups_read={groups_read} bytes_read={bytes_read} revision={revision[:12]}"
    )
    if len(run.windows) < target:
        run.note(f"shortfall: {target - len(run.windows)} windows (group plan or byte cap exhausted)")
    return finalize()


__all__ = ["DATASET_ID", "SvarahError", "dataset_revision", "fetch_svarah", "parse_speaker"]
