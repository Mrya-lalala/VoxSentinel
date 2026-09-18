"""Schema-compatible importer for user-supplied synthetic Indian-English audio.

No traceable synthetic Indian-English source was found during preparation, so
this module ships as the ready import path described by the master plan: point
it at a local directory plus a metadata JSONL following
``configs/synthetic_english_template.json`` and it materializes windows into the
same manifests, keeping the class pairing with NISP genuine English.

Rules enforced on import:

* every row must declare the generator and its version; a missing generator is
  an error - never guessed;
* ``accent_verified`` defaults to ``false``; a TTS ``en`` setting alone does not
  establish an Indian accent;
* reference voices must not be evaluation speakers - the caller declares
  ``reference_speaker_ids`` and the importer only records them;
* no paid services, no uploading voices, no TTS training: this importer only
  ingests assets the user is authorized to use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..budget import DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ...audio.prepare import AudioDecodeError
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "synthetic_english"

REQUIRED_FIELDS = ("file", "generator", "generator_version")
OPTIONAL_FIELDS = (
    "split",
    "spoken_language",
    "accent",
    "accent_verified",
    "reference_speaker_ids",
    "reference_audio",
    "license",
    "notes",
)


def _read_metadata(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: metadata rows must be objects")
            missing = [field for field in REQUIRED_FIELDS if not value.get(field)]
            if missing:
                raise ValueError(
                    f"{path}:{line_number}: missing required fields {missing}; "
                    "a generator must never be guessed."
                )
            rows.append(dict(value))
    return rows


def import_synthetic_english(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    import_dir: str | Path | None = None,
    metadata_file: str | Path | None = None,
    force: bool = False,
) -> SourceRun:
    """Import user-supplied synthetic Indian-English assets (if any are provided)."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    quota = cfg.quota("synthetic_english")
    import_root_raw = import_dir or quota.get("import_dir")
    metadata_raw = metadata_file or quota.get("metadata_file")
    if not import_root_raw:
        run.note(
            "no user-supplied synthetic Indian-English assets were provided "
            "(quotas.synthetic_english.import_dir is empty); coverage remains missing_source"
        )
        return run
    import_root = Path(str(import_root_raw))
    if not import_root.exists() or not import_root.is_dir():
        run.note(f"import directory does not exist: {import_root}; nothing imported")
        return run
    if not metadata_raw:
        run.note("no metadata JSONL provided; nothing imported")
        return run
    metadata_path = Path(str(metadata_raw))
    if not metadata_path.is_file():
        run.note(f"metadata file not found under {metadata_path}; nothing imported")
        return run

    rows = _read_metadata(metadata_path)
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

    for row in rows:
        relative = Path(str(row["file"]))
        source_path = (import_root / relative).resolve()
        if not str(source_path).startswith(str(import_root.resolve())):
            raise ValueError(f"metadata file path escapes the import directory: {relative}")
        window_id = f"{DATASET_ID}-{relative.stem}"
        if window_id in existing_ids:
            continue
        if not source_path.exists():
            run.exclusions.append(
                exclusion_row(
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_file=str(relative),
                    original_split=None,
                    reason="declared file missing from import directory",
                )
            )
            continue
        raw_path = cfg.raw_dir / DATASET_ID / source_path.name
        store_original_bytes(raw_path, source_path.read_bytes())
        prepared_path = cfg.prepared_dir / DATASET_ID / f"{window_id}.wav"
        try:
            materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
        except (AudioDecodeError, ValueError) as error:
            run.exclusions.append(
                exclusion_row(
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_file=str(relative),
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
                    source_file=str(relative),
                    original_split=None,
                    reason=duplicate,
                )
            )
            prepared_path.unlink(missing_ok=True)
            continue
        registry.register(materialized)

        split = row.get("split")
        if split not in ("train", "val"):
            split = None
        speaker_ids = {"reference": row.get("reference_speaker_ids")}
        parent_refs = {
            "reference_audio": row.get("reference_audio"),
            "accent_declared": row.get("accent"),
            "accent_verified": bool(row.get("accent_verified", False)),
            "user_notes": row.get("notes"),
        }
        recording_record = build_recording_record(
            recording_id=window_id,
            dataset_id=DATASET_ID,
            source_url="user-supplied local assets",
            source_file=str(relative),
            original_split=None,
            label=1,
            spoken_language=str(row.get("spoken_language") or "en"),
            native_language=None,
            speaker_ids=speaker_ids,
            generator=str(row["generator"]),
            generator_version=str(row["generator_version"]),
            parent_refs=parent_refs,
            original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
            decoded=materialized.decoded_facts(),
            license_note=str(row.get("license") or "user-declared"),
            notes="user-supplied synthetic asset; accent verification "
            + ("declared true" if row.get("accent_verified") else "NOT reviewed"),
        )
        window_record = build_window_record(
            window_id=window_id,
            recording_id=window_id,
            dataset_id=DATASET_ID,
            source_url="user-supplied local assets",
            source_file=str(relative),
            original_split=None,
            pool="core" if split else "unpaired_candidate",
            split=split,
            label=1,
            label_source="user-supplied synthetic generation metadata",
            spoken_language=str(row.get("spoken_language") or "en"),
            native_language=None,
            speaker_ids=speaker_ids,
            generator=str(row["generator"]),
            generator_version=str(row["generator_version"]),
            parent_refs=parent_refs,
            original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
            prepared_audio=materialized.prepared_facts(cfg.dataset_root),
            window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
            preprocessing_version=str(cfg.preprocessing.get("config_version")),
            license_note=str(row.get("license") or "user-declared"),
            access_status="user_supplied",
            status="audio_ready",
            notes="schema-compatible synthetic English import",
        )
        run.recordings.append(recording_record)
        run.windows.append(window_record)
        existing_ids.add(window_id)

    run.note(f"imported {len(run.windows)} synthetic English windows from {import_root}")
    write_jsonl_atomic(recordings_path, run.recordings)
    write_jsonl_atomic(windows_path, run.windows)
    write_jsonl_atomic(exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, existing_ids)))
    return run


__all__ = ["DATASET_ID", "import_synthetic_english"]
