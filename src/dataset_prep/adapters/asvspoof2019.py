"""ASVspoof 2019 LA adapter: selective remote-ZIP extraction.

The official DataShare archive ``LA.zip`` is 7.64 GB and is never downloaded
whole.  A seekable HTTP file is wrapped in a byte-accounting reader; only the
protocol files and the selected FLAC members are transferred (HTTP range
requests), then the archive parts relevant to each member.

Official train/dev membership, protocol labels, speaker IDs and attack IDs are
preserved.  The pool stays separate from the main Indic pilot and is never
routed through the Indic encoder automatically.
"""

from __future__ import annotations

import io
import re
import random
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence

from .. import httpio
from ..budget import BudgetExceeded, DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ...audio.prepare import AudioDecodeError
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "asvspoof2019"
ARCHIVE_URL = "https://datashare.ed.ac.uk/server/api/core/bitstreams/a9f87c35-f055-4015-80e2-2fdff0d46269/content"
ARCHIVE_PAGE = "https://datashare.ed.ac.uk/handle/10283/3336"

TRACKS = {
    "train": {
        "protocol": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
        "flac_dir": "LA/ASVspoof2019_LA_train/flac/",
    },
    "dev": {
        "protocol": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
        "flac_dir": "LA/ASVspoof2019_LA_dev/flac/",
    },
}

_ROW = re.compile(r"^(?P<speaker>\S+)\s+(?P<utterance>LA_[DTE]_[A-Z0-9]+)\s+(?P<dash>\S+)\s+(?P<attack>\S+)\s+(?P<key>\S+)\s*$")


class _CountingReader(io.RawIOBase):
    """Read-only wrapper that charges delivered bytes to the ledger in batches.

    fsspec's readahead cache can serve repeated ranges without a new network
    request; charging the bytes *delivered* is therefore a conservative
    (never lower than actual) account of transferred data for this adapter.
    Charges are flushed in ~4 MiB batches to keep ledger writes rare.
    """

    _FLUSH_THRESHOLD = 4 * 1024 * 1024

    def __init__(self, inner: Any, ledger: DownloadLedger, source_id: str) -> None:
        self._inner = inner
        self._ledger = ledger
        self._source_id = source_id
        self._pending = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        return self._inner.seek(offset, whence)

    def tell(self) -> int:
        return self._inner.tell()

    def read(self, size: int = -1) -> bytes:  # type: ignore[override]
        data = self._inner.read(size)
        if data:
            self._charge(len(data))
        return data

    def readinto(self, buffer) -> int:  # type: ignore[override]
        view = memoryview(buffer)
        data = self._inner.read(len(view))
        if data:
            view[: len(data)] = data
            self._charge(len(data))
        return len(data)

    def _charge(self, count: int) -> None:
        self._pending += count
        if self._pending >= self._FLUSH_THRESHOLD:
            self.flush()

    def flush(self) -> None:
        if self._pending <= 0:
            return
        pending, self._pending = self._pending, 0
        self._ledger.charge(self._source_id, pending)

    def close(self) -> None:
        try:
            self.flush()
        finally:
            try:
                self._inner.close()
            finally:
                super().close()


def _parse_protocol(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = _ROW.match(line)
        if not match:
            continue
        key = match.group("key").lower()
        label = 0 if key.startswith("bonafide") or key.startswith("bona-fide") else 1
        rows.append(
            {
                "speaker": match.group("speaker"),
                "utterance": match.group("utterance"),
                "attack": match.group("attack"),
                "key": key,
                "label": label,
            }
        )
    return rows


def _select(rows: Sequence[Mapping[str, Any]], count: int, label: int, seed: str) -> list[dict[str, Any]]:
    pool = [dict(row) for row in rows if int(row["label"]) == label]
    rng = random.Random(seed)
    rng.shuffle(pool)
    if label == 0:
        # Genuine: spread speakers first.
        selected: list[dict[str, Any]] = []
        seen_speakers: set[str] = set()
        for row in pool:
            if row["speaker"] not in seen_speakers:
                selected.append(row)
                seen_speakers.add(row["speaker"])
            if len(selected) >= count:
                break
        for row in pool:
            if len(selected) >= count:
                break
            if row not in selected:
                selected.append(row)
        return selected[:count]
    # Spoof: spread attacks first, then speakers within each attack.
    by_attack: dict[str, list[dict[str, Any]]] = {}
    for row in pool:
        by_attack.setdefault(str(row["attack"]), []).append(row)
    attacks = sorted(by_attack)
    selected = []
    seen_speakers = set()
    cursor = {attack: 0 for attack in attacks}
    progress = True
    while len(selected) < count and progress:
        progress = False
        for attack in attacks:
            if len(selected) >= count:
                break
            bucket = by_attack[attack]
            pick = None
            while cursor[attack] < len(bucket):
                candidate = bucket[cursor[attack]]
                cursor[attack] += 1
                if candidate["speaker"] not in seen_speakers:
                    pick = candidate
                    break
            if pick is None and cursor[attack] >= len(bucket):
                continue
            if pick is not None:
                selected.append(pick)
                seen_speakers.add(pick["speaker"])
                progress = True

    # If the attack round-robin exhausted buckets, fill the remainder.
    for row in pool:
        if len(selected) >= count:
            break
        if row not in selected:
            selected.append(row)
    return selected[:count]


def fetch_asvspoof2019(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    force: bool = False,
) -> SourceRun:
    """Select and materialize the bounded LA train/dev subset from the remote ZIP."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    release = cfg.release(DATASET_ID)
    quota = cfg.quota("asvspoof2019").get("label_quotas", {})
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
    protocol_dir = cfg.raw_dir / DATASET_ID / "protocols"

    import fsspec  # Local import: heavy dependency only needed by this adapter.

    fs = fsspec.filesystem("http")
    try:
        with fs.open(ARCHIVE_URL, "rb", block_size=4 * 1024 * 1024) as raw_file:
            reader = _CountingReader(raw_file, ledger, DATASET_ID)
            with zipfile.ZipFile(reader) as archive:
                names = set(archive.namelist())
                for track, spec in TRACKS.items():
                    protocol_name = spec["protocol"]
                    if protocol_name not in names:
                        run.note(f"{track}: protocol {protocol_name} missing from archive")
                        continue
                    protocol_bytes = archive.read(protocol_name)
                    store_original_bytes(protocol_dir / Path(protocol_name).name, protocol_bytes)
                    rows = _parse_protocol(protocol_bytes.decode("utf-8", errors="replace"))
                    targets = quota.get(track, {})
                    genuine_target = int(targets.get("genuine", 0))
                    spoof_target = int(targets.get("spoof", 0))
                    genuine = _select(rows, genuine_target, 0, f"{cfg.seed}:{track}:genuine")
                    spoof = _select(rows, spoof_target, 1, f"{cfg.seed}:{track}:spoof")
                    run.note(
                        f"{track}: protocol rows={len(rows)} selected genuine={len(genuine)} spoof={len(spoof)}"
                    )
                    for row in genuine + spoof:
                        window_id = f"{DATASET_ID}-{track}-{row['utterance'].lower()}"
                        if window_id in existing_ids:
                            continue
                        member = f"{spec['flac_dir']}{row['utterance']}.flac"
                        if member not in names:
                            run.exclusions.append(
                                exclusion_row(
                                    recording_id=window_id,
                                    dataset_id=DATASET_ID,
                                    source_file=member,
                                    original_split=track,
                                    reason="utterance_missing_from_archive",
                                )
                            )
                            continue
                        try:
                            payload = archive.read(member)
                        except BudgetExceeded as error:
                            run.note(f"{track}: budget reached while reading {member}: {error}")
                            break
                        raw_path = cfg.raw_dir / DATASET_ID / track / f"{row['utterance']}.flac"
                        store_original_bytes(raw_path, payload)
                        prepared_path = cfg.prepared_dir / DATASET_ID / track / f"{window_id}.wav"
                        try:
                            materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
                        except (AudioDecodeError, ValueError) as error:
                            run.exclusions.append(
                                exclusion_row(
                                    recording_id=window_id,
                                    dataset_id=DATASET_ID,
                                    source_file=member,
                                    original_split=track,
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
                                    source_file=member,
                                    original_split=track,
                                    reason=duplicate,
                                )
                            )
                            prepared_path.unlink(missing_ok=True)
                            continue
                        registry.register(materialized)

                        speaker_ids = {"speaker": row["speaker"]}
                        recording_record = build_recording_record(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_url=ARCHIVE_PAGE,
                            source_file=member,
                            original_split=track,
                            label=int(row["label"]),
                            spoken_language="en",
                            native_language=None,
                            speaker_ids=speaker_ids,
                            generator=None,
                            generator_version=None,
                            parent_refs={
                                "protocol_file": Path(protocol_name).name,
                                "utterance_id": row["utterance"],
                                "attack_id": None if row["attack"] == "-" else row["attack"],
                                "protocol_key": row["key"],
                                "dataset_revision": release.get("revision"),
                            },
                            original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                            decoded=materialized.decoded_facts(),
                            license_note=str(release.get("license")),
                            notes="fallback/baseline pool; not part of the Indic pilot",
                        )
                        window_record = build_window_record(
                            window_id=window_id,
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_url=ARCHIVE_PAGE,
                            source_file=member,
                            original_split=track,
                            pool="fallback_baseline",
                            split=track,
                            label=int(row["label"]),
                            label_source=f"official protocol ({Path(protocol_name).name})",
                            spoken_language="en",
                            native_language=None,
                            speaker_ids=speaker_ids,
                            generator=None,
                            generator_version=None,
                            parent_refs=recording_record["parent_refs"],
                            original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                            prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                            window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                            preprocessing_version=str(cfg.preprocessing.get("config_version")),
                            license_note=str(release.get("license")),
                            access_status="public_with_terms",
                            status="audio_ready",
                            notes="one window per official utterance; official split preserved; no Indic encoder features",
                        )
                        run.recordings.append(recording_record)
                        run.windows.append(window_record)
                        existing_ids.add(window_id)
    except BudgetExceeded as error:
        run.note(f"download budget reached: {error}")
    except (zipfile.BadZipFile, IOError, OSError) as error:
        run.note(f"remote archive access failed: {error}")

    write_jsonl_atomic(recordings_path, run.recordings)
    write_jsonl_atomic(windows_path, run.windows)
    write_jsonl_atomic(exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, existing_ids)))
    return run


__all__ = ["ARCHIVE_PAGE", "ARCHIVE_URL", "DATASET_ID", "fetch_asvspoof2019"]
