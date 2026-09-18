"""NISP adapter: genuine Indian-accented English via streamed multipart archives.

The English folders of the five native-language groups are distributed as one
gzip stream split into ~95 MiB ``RECS.tar.gz.a*`` parts that must be
concatenated *in exact part order*.  Parts are streamed on demand with a
per-language byte ceiling; extraction stops as soon as the group's quota is met.

Rules applied here:

* only speakers in the official ``train_spkrID`` list may enter our
  train/validation pool; official ``test_spkrID`` speakers are never selected;
* folder native language is the *accent background*; recorded language is
  always English (``spoken_language: "en"``);
* one window per original recording; at most
  ``windows_per_train_speaker``/``windows_per_val_speaker`` windows per speaker.

Because no traceable synthetic Indian-English source is available, these windows
are delivered in the ``unpaired_candidate`` pool and excluded from the default
balanced core manifest.
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import httpio
from ..budget import BudgetExceeded, DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ...audio.prepare import AudioDecodeError
from .archive_stream import SegmentedGzipReader
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "nisp"
REPO = "iiscleap/NISP-Dataset"
BRANCH = "master"
RAW_BASE = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}"
API_BASE = f"https://api.github.com/repos/{REPO}"
ASCII_BASE = f"https://github.com/{REPO}/releases/download"

GROUP_CODES = {
    "Hindi": "Hin",
    "Kannada": "Kan",
    "Malayalam": "Mal",
    "Tamil": "Tam",
    "Telugu": "Tel",
}

_WAV_PATTERN = re.compile(r"^(?P<code>[A-Za-z]{3})_(?P<speaker>\d+)_Eng_(?P<gender>[mf])_(?P<utt>\d+)\.wav$")


@dataclass(frozen=True)
class GroupOutcome:
    group: str
    accepted: int
    train_windows: int
    val_windows: int
    train_speakers: int
    val_speakers: int
    parts_used: int
    bytes_downloaded: int
    stopped_early: bool


def _canonical_speaker(token: str) -> str | None:
    parts = token.strip().split("_")
    if len(parts) >= 2 and parts[0] and parts[1]:
        return f"{parts[0]}_{parts[1]}"
    return None


def _fetch_speaker_lists(cfg: PrepConfig, run: SourceRun) -> tuple[set[str], set[str]]:
    metadata_dir = cfg.raw_dir / DATASET_ID / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    train_url = f"{RAW_BASE}/train_spkrID"
    test_url = f"{RAW_BASE}/test_spkrID"
    train_path = metadata_dir / "train_spkrID.txt"
    test_path = metadata_dir / "test_spkrID.txt"
    for url, path in ((train_url, train_path), (test_url, test_path)):
        payload = httpio.get_bytes(url, timeout=cfg.budget.request_timeout_seconds)
        store_original_bytes(path, payload)
        run.metadata_requests += 1
    train_ids = {_canonical_speaker(line) for line in train_path.read_text().splitlines() if line.strip()}
    test_ids = {_canonical_speaker(line) for line in test_path.read_text().splitlines() if line.strip()}
    train_ids.discard(None)
    test_ids.discard(None)
    if not train_ids:
        raise httpio.HttpError("NISP train_spkrID parsed empty; refusing to select speakers blindly.")
    overlap = train_ids & test_ids
    if overlap:
        raise httpio.HttpError(f"NISP train/test speaker lists overlap ({sorted(overlap)[:5]}); aborting.")
    return train_ids, test_ids


def _list_parts(cfg: PrepConfig, group: str, run: SourceRun) -> list[tuple[str, str, int]]:
    url = f"{API_BASE}/contents/{group}_master/English_{group}"
    payload = httpio.get_json(url, timeout=cfg.budget.request_timeout_seconds)
    run.metadata_requests += 1
    if not isinstance(payload, list):
        raise httpio.HttpError(f"unexpected GitHub contents response for {group}: {type(payload).__name__}")
    parts: list[tuple[str, str, int]] = []
    for entry in payload:
        name = str(entry.get("name", ""))
        if not name.startswith("RECS.tar.gz."):
            continue
        suffix = name.rsplit(".", 1)[-1]
        parts.append(
            (
                name,
                f"{RAW_BASE}/{group}_master/English_{group}/{name}",
                int(entry.get("size", 0)),
            )
        )
    parts.sort(key=lambda item: item[0].rsplit(".", 1)[-1])
    if not parts:
        raise httpio.HttpError(f"no RECS.tar.gz.* parts found for {group}")
    return parts


def _quota(cfg: PrepConfig) -> dict[str, int]:
    quota = cfg.quota("nisp")
    return {
        "train_speakers": int(quota.get("train_speakers_per_group", 3)),
        "val_speakers": int(quota.get("val_speakers_per_group", 2)),
        "train_windows_per_speaker": int(quota.get("windows_per_train_speaker", 2)),
        "val_windows_per_speaker": int(quota.get("windows_per_val_speaker", 1)),
        "max_archive_bytes": int(quota.get("max_archive_bytes_per_group", 400 * 1024 * 1024)),
    }


def fetch_nisp(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    groups: Sequence[str] | None = None,
    force: bool = False,
) -> SourceRun:
    """Run the bounded NISP English materialization."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    quota_cfg = cfg.quota("nisp")
    selected_groups = list(groups or quota_cfg.get("native_groups", list(GROUP_CODES)))
    limits = _quota(cfg)
    release = cfg.release(DATASET_ID)

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

    train_ids, test_ids = _fetch_speaker_lists(cfg, run)

    for group in selected_groups:
        if group not in GROUP_CODES:
            run.note(f"{group}: unknown native-language group; skipped")
            continue
        outcome = _stream_group(
            cfg,
            ledger,
            run,
            state,
            registry,
            group,
            train_ids,
            test_ids,
            limits,
            release,
        )
        run.note(
            f"{group}: windows train={outcome.train_windows} val={outcome.val_windows} "
            f"(speakers {outcome.train_speakers}+{outcome.val_speakers}), "
            f"parts={outcome.parts_used}, bytes={outcome.bytes_downloaded}"
            + (", stopped early (budget/end of archive)" if outcome.stopped_early else "")
        )

    successful_ids = {str(row.get("recording_id")) for row in run.recordings}
    write_jsonl_atomic(recordings_path, run.recordings)
    write_jsonl_atomic(windows_path, run.windows)
    write_jsonl_atomic(exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, successful_ids)))
    return run


def _stream_group(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    run: SourceRun,
    state,
    registry,
    group: str,
    train_ids: set[str],
    test_ids: set[str],
    limits: Mapping[str, int],
    release: Mapping[str, Any],
) -> GroupOutcome:
    slug = group.lower()
    parts = _list_parts(cfg, group, run)
    staging = cfg.staging_dir / DATASET_ID / slug

    def _download(url: str, destination: Path) -> httpio.DownloadResult:
        result = httpio.download_resumable(
            url,
            destination,
            ledger=ledger,
            source_id=DATASET_ID,
            timeout=cfg.budget.request_timeout_seconds,
        )
        run.download_bytes += result.transferred_bytes
        return result

    reader = SegmentedGzipReader(
        parts,
        download=_download,
        staging_dir=staging,
        group_budget_bytes=limits["max_archive_bytes"],
    )

    role_of_speaker: dict[str, str] = {}
    speaker_windows: dict[str, int] = {}
    train_windows = val_windows = 0
    existing_ids = set(state.recordings)
    accepted = 0
    error_note: str | None = None

    # Resume: seed counters and speaker roles from windows already materialized.
    for row in state.windows:
        if row.get("dataset_id") != DATASET_ID:
            continue
        if f"-{slug}-" not in str(row.get("window_id")):
            continue
        split = str(row.get("split"))
        speaker = str((row.get("speaker_ids") or {}).get("speaker"))
        if speaker and speaker not in role_of_speaker:
            role_of_speaker[speaker] = split
        speaker_windows[speaker] = speaker_windows.get(speaker, 0) + 1
        if split == "train":
            train_windows += 1
        elif split == "val":
            val_windows += 1

    def _targets_met() -> bool:
        return train_windows >= limits["train_speakers"] * limits["train_windows_per_speaker"] and val_windows >= limits["val_speakers"] * limits["val_windows_per_speaker"]

    try:
        with tarfile.open(fileobj=reader, mode="r|") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                basename = member.name.rsplit("/", 1)[-1]
                match = _WAV_PATTERN.match(basename)
                if not match:
                    continue
                code = match.group("code")
                if code.lower() != GROUP_CODES[group].lower():
                    continue
                speaker = f"{GROUP_CODES[group]}_{match.group('speaker')}"
                if speaker not in train_ids:
                    continue  # unknown or official test speaker: never selected
                if speaker in test_ids:
                    continue
                if member.size > 5 * 1024 * 1024:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=f"{DATASET_ID}-{slug}-{basename}",
                            dataset_id=DATASET_ID,
                            source_file=f"{group}/{basename}",
                            original_split="official_train",
                            reason=f"oversized_member_{member.size}_bytes",
                        )
                    )
                    continue
                role = role_of_speaker.get(speaker)
                if role is None:
                    train_speakers = sum(1 for value in role_of_speaker.values() if value == "train")
                    val_speakers = sum(1 for value in role_of_speaker.values() if value == "val")
                    if train_speakers < limits["train_speakers"]:
                        role = "train"
                    elif val_speakers < limits["val_speakers"]:
                        role = "val"
                    else:
                        continue  # quota complete; ignore further speakers
                    role_of_speaker[speaker] = role
                cap = limits["train_windows_per_speaker"] if role == "train" else limits["val_windows_per_speaker"]
                if speaker_windows.get(speaker, 0) >= cap:
                    continue

                window_id = f"{DATASET_ID}-{slug}-{Path(basename).stem.lower()}"
                if window_id in existing_ids:
                    continue
                handle = archive.extractfile(member)
                if handle is None:
                    continue
                payload = handle.read()
                raw_path = cfg.raw_dir / DATASET_ID / slug / basename
                store_original_bytes(raw_path, payload)
                prepared_path = cfg.prepared_dir / DATASET_ID / slug / f"{window_id}.wav"
                try:
                    materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
                except (AudioDecodeError, ValueError) as error:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_file=f"{group}/{basename}",
                            original_split="official_train",
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
                            source_file=f"{group}/{basename}",
                            original_split="official_train",
                            reason=duplicate,
                        )
                    )
                    prepared_path.unlink(missing_ok=True)
                    continue
                registry.register(materialized)

                speaker_ids = {"speaker": speaker, "gender": match.group("gender")}
                recording_record = build_recording_record(
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_url=str(release.get("source_url")),
                    source_file=f"{group}/English_{group}/{basename}",
                    original_split="official_train",
                    label=0,
                    spoken_language="en",
                    native_language=group,
                    speaker_ids=speaker_ids,
                    generator=None,
                    generator_version=None,
                    parent_refs={
                        "archive_parts": [part[0] for part in parts],
                        "utterance_id": match.group("utt"),
                        "dataset_revision": release.get("revision"),
                    },
                    original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                    decoded=materialized.decoded_facts(),
                    license_note=str(release.get("license")),
                    notes="genuine speech; folder native language is accent background, spoken language is English",
                )
                window_record = build_window_record(
                    window_id=window_id,
                    recording_id=window_id,
                    dataset_id=DATASET_ID,
                    source_url=str(release.get("source_url")),
                    source_file=f"{group}/English_{group}/{basename}",
                    original_split="official_train",
                    pool="unpaired_candidate",
                    split=role,
                    label=0,
                    label_source="official NISP dataset (genuine speech)",
                    spoken_language="en",
                    native_language=group,
                    speaker_ids=speaker_ids,
                    generator=None,
                    generator_version=None,
                    parent_refs=recording_record["parent_refs"],
                    original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                    prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                    window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                    preprocessing_version=str(cfg.preprocessing.get("config_version")),
                    license_note=str(release.get("license")),
                    access_status="public",
                    status="unpaired_candidate",
                    notes="single window per official train recording",
                )
                run.recordings.append(recording_record)
                run.windows.append(window_record)
                existing_ids.add(window_id)
                speaker_windows[speaker] = speaker_windows.get(speaker, 0) + 1
                accepted += 1
                if role == "train":
                    train_windows += 1
                else:
                    val_windows += 1

                if _targets_met():
                    break
    except (tarfile.TarError, BudgetExceeded, OSError) as error:
        error_note = f"{group}: archive stream stopped: {error}"

    if error_note:
        run.note(error_note)
    if reader.stopped_early:
        run.note(f"{group}: byte ceiling/ledger reached after {reader.bytes_downloaded} bytes")
    if not _targets_met():
        run.note(
            f"{group}: quota not fully met (train={train_windows}/"
            f"{limits['train_speakers'] * limits['train_windows_per_speaker']}, val={val_windows}/"
            f"{limits['val_speakers'] * limits['val_windows_per_speaker']})"
        )

    return GroupOutcome(
        group=group,
        accepted=accepted,
        train_windows=train_windows,
        val_windows=val_windows,
        train_speakers=sum(1 for value in role_of_speaker.values() if value == "train"),
        val_speakers=sum(1 for value in role_of_speaker.values() if value == "val"),
        parts_used=reader.parts_used,
        bytes_downloaded=reader.bytes_downloaded,
        stopped_early=reader.stopped_early,
    )


__all__ = ["DATASET_ID", "GROUP_CODES", "fetch_nisp"]
