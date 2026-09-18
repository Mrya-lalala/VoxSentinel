"""Kathbath (IndicSUPERB) adapter: genuine Indic speech over the HF Hub.

Access route (verified 2026-09-18 with the user's local HF login): the dataset
is gated (auto terms) and the authenticated account is authorized.  Data is
organized as ``<language>/<split>-NNNNN-of-MMMMM.parquet``; every row carries
``fname`` (``<recording_id>-<speaker_id>-<gender>.m4a``), ``speaker_id``,
``duration``, ``gender`` and embedded M4A audio bytes.

Selective reads:  the parquet footer and the small metadata columns are read
through ``HfFileSystem`` range requests (a few hundred KiB per shard), and only
the row groups that contain selected recordings are read for audio
(12-23 MB each).  Every delivered byte is charged to the shared download
ledger through a counting wrapper.  The user's token is never written anywhere;
only Hub commit SHAs, file names and hashes are recorded.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..budget import BudgetExceeded, DownloadLedger
from ..config import PrepConfig
from ..materialize import store_original_bytes
from ..records import read_jsonl, write_json_atomic

DATASET_ID = "kathbath"
HF_DATASET = "ai4bharat/Kathbath"

_FNAME = re.compile(r"^(?P<record>\d+)-(?P<speaker>\d+)-(?P<gender>[mf])\.(?P<ext>m4a|wav|mp3|flac|ogg)$")

INVENTORY_COLUMNS = ["fname", "speaker_id", "duration", "gender"]


class KathbathError(RuntimeError):
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


def parse_fname(fname: str) -> dict[str, Any] | None:
    """Parse ``<record>-<speaker>-<gender>.<ext>``; returns None when unmatched."""
    match = _FNAME.match(str(fname))
    if not match:
        return None
    return {
        "record_id": match.group("record"),
        "speaker": match.group("speaker"),
        "gender": match.group("gender"),
        "extension": match.group("ext"),
    }


def dataset_revision() -> str:
    """Resolve the Hub commit SHA for the current acquisition (no token stored)."""
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(HF_DATASET)
    if not info.sha:
        raise KathbathError("could not resolve dataset revision SHA")
    return info.sha


def list_split_files(cfg: PrepConfig, folder: str, *, revision: str) -> dict[str, list[str]]:
    """List shard file names per upstream split, cached in staging (no secrets)."""
    cache = cfg.staging_dir / DATASET_ID / f"files.{folder}.json"
    if cache.exists():
        cached = json.loads(cache.read_text())
        if cached.get("revision") == revision:
            return {str(k): list(v) for k, v in cached["splits"].items()}
    from huggingface_hub import HfApi

    info = HfApi().dataset_info(HF_DATASET, files_metadata=True, revision=revision)
    splits: dict[str, list[str]] = {"train": [], "valid": [], "test": []}
    for sibling in info.siblings or []:
        name = sibling.rfilename
        if not name.endswith(".parquet") or not name.startswith(folder + "/"):
            continue
        stem = Path(name).name
        for split in splits:
            if stem.startswith(split + "-"):
                splits[split].append(name)
    splits = {k: sorted(v) for k, v in splits.items() if v}
    write_json_atomic(cache, {"revision": revision, "splits": splits})
    return splits


def _row_group_index_by_row(num_rows_per_group: Sequence[int]) -> list[int]:
    mapping: list[int] = []
    for group, count in enumerate(num_rows_per_group):
        mapping.extend([group] * int(count))
    return mapping


def scan_shards(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    folder: str,
    shard_names: Sequence[str],
    workers: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read metadata columns of the given shards in parallel.

    Returns ``(rows, errors)`` where each row is one recording:
    ``fname, record_id, speaker, gender, duration, shard, row_index, row_group``.
    Bytes are charged to the ledger (tiny: ~0.1-1 MB per shard).
    """
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    def _scan_one(shard: str) -> tuple[list[dict[str, Any]], int]:
        counted = _CountingFile(fs.open(f"datasets/{HF_DATASET}/{shard}", "rb"))
        try:
            parquet = pq.ParquetFile(counted)
            table = parquet.read(columns=INVENTORY_COLUMNS).to_pydict()
            group_of_row = _row_group_index_by_row(
                [parquet.metadata.row_group(g).num_rows for g in range(parquet.metadata.num_row_groups)]
            )
            out: list[dict[str, Any]] = []
            for index, fname in enumerate(table["fname"]):
                parsed = parse_fname(fname)
                if parsed is None:
                    continue
                out.append(
                    {
                        "fname": fname,
                        "record_id": parsed["record_id"],
                        "speaker": parsed["speaker"],
                        "gender": parsed["gender"],
                        "extension": parsed["extension"],
                        "duration": float(table["duration"][index]),
                        "upstream_speaker_id": int(table["speaker_id"][index]),
                        "shard": shard,
                        "row_index": index,
                        "row_group": group_of_row[index] if index < len(group_of_row) else -1,
                    }
                )
            return out, counted.bytes
        finally:
            counted.close()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_scan_one, shard): shard for shard in shard_names}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                shard_rows, bytes_read = future.result()
            except Exception as error:  # noqa: BLE001 - per-shard error isolation
                errors.append({"shard": shard, "error": f"{type(error).__name__}: {error}"})
                continue
            try:
                ledger.charge(DATASET_ID, bytes_read)
            except BudgetExceeded as error:
                errors.append({"shard": shard, "error": f"budget: {error}"})
                break
            rows.extend(shard_rows)
    rows.sort(key=lambda r: (r["shard"], r["row_index"]))
    return rows, errors


def scan_folder(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    folder: str,
    max_shards: int,
    stop: Callable[[list[dict[str, Any]]], bool] | None = None,
    workers: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Scan train shards progressively until ``stop`` or the shard budget is hit.

    Returns ``(rows, errors, scanned_shards)``.
    """
    revision = dataset_revision()
    splits = list_split_files(cfg, folder, revision=revision)
    train = list(splits.get("train", []))
    if not train:
        raise KathbathError(f"no train shards found for {folder}")
    collected: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    scanned: list[str] = []
    batch = max(1, min(int(workers), 6))
    position = 0
    while position < len(train) and len(scanned) < max_shards:
        window = train[position : position + batch]
        position += batch
        rows, shard_errors = scan_shards(cfg, ledger, folder=folder, shard_names=window, workers=len(window))
        collected.extend(rows)
        scanned.extend(window)
        errors.extend(shard_errors)
        if stop is not None and stop(collected):
            break
    return collected, errors, scanned


def valid_recording_ids(cfg: PrepConfig, ledger: DownloadLedger, *, folder: str, workers: int = 6) -> set[str]:
    """Record ids that appear in the upstream ``valid``/``test`` shards (held out)."""
    revision = dataset_revision()
    splits = list_split_files(cfg, folder, revision=revision)
    held_out = list(splits.get("valid", [])) + list(splits.get("test", []))
    if not held_out:
        return set()
    rows, errors = scan_shards(cfg, ledger, folder=folder, shard_names=held_out, workers=workers)
    if errors:
        # Do not silently continue as if held-out membership were verified.
        raise KathbathError(f"failed to scan held-out shards for {folder}: {errors[:2]}")
    return {row["record_id"] for row in rows}


@dataclass
class FetchResult:
    stored: dict[str, Path]
    bytes_read: int
    row_groups_read: int


def fetch_audio(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    folder: str,
    rows: Sequence[Mapping[str, Any]],
    byte_cap: int,
    workers: int = 4,
) -> FetchResult:
    """Fetch the audio bytes for selected recordings, row group by row group."""
    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    fs = HfFileSystem()
    destination = cfg.raw_dir / DATASET_ID / folder
    destination.mkdir(parents=True, exist_ok=True)
    stored: dict[str, Path] = {}
    total_bytes = 0
    groups_read = 0

    by_shard: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        by_shard.setdefault(str(row["shard"]), []).append(row)

    def _fetch_shard(shard: str, wanted: list[Mapping[str, Any]]) -> tuple[dict[str, bytes], int, int]:
        counted = _CountingFile(fs.open(f"datasets/{HF_DATASET}/{shard}", "rb"))
        try:
            parquet = pq.ParquetFile(counted)
            group_offsets: list[int] = []
            running = 0
            for group in range(parquet.metadata.num_row_groups):
                group_offsets.append(running)
                running += parquet.metadata.row_group(group).num_rows
            by_group: dict[int, list[Mapping[str, Any]]] = {}
            for row in wanted:
                by_group.setdefault(int(row["row_group"]), []).append(row)
            payloads: dict[str, bytes] = {}
            groups = 0
            for group in sorted(by_group):
                table = parquet.read_row_group(group, columns=["audio_filepath"]).to_pydict()
                column = table["audio_filepath"]
                offset = group_offsets[group] if group < len(group_offsets) else 0
                for row in by_group[group]:
                    # ``row_index`` is the file-global index; convert it to the
                    # index inside this single row group.
                    local = int(row["row_index"]) - offset
                    if not 0 <= local < len(column):
                        raise KathbathError(
                            f"row {row['fname']} (index {row['row_index']}) outside row group {group}"
                        )
                    cell = column[local]
                    data = cell.get("bytes") if isinstance(cell, Mapping) else None
                    if data:
                        payloads[str(row["fname"])] = bytes(data)
                groups += 1
            return payloads, counted.bytes, groups
        finally:
            counted.close()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_fetch_shard, shard, wanted): shard for shard, wanted in by_shard.items()}
        for future in as_completed(futures):
            payloads, bytes_read, groups = future.result()
            if total_bytes + bytes_read > byte_cap:
                raise KathbathError(
                    f"audio byte cap reached for {folder}: {total_bytes + bytes_read} > {byte_cap}"
                )
            ledger.charge(DATASET_ID, bytes_read)
            total_bytes += bytes_read
            groups_read += groups
            for fname, payload in payloads.items():
                stored[fname] = store_original_bytes(destination / fname, payload)

    return FetchResult(stored=stored, bytes_read=total_bytes, row_groups_read=groups_read)


def load_inventory(path: str | Path) -> list[dict[str, Any]]:
    return read_jsonl(path)


__all__ = [
    "DATASET_ID",
    "HF_DATASET",
    "FetchResult",
    "KathbathError",
    "dataset_revision",
    "fetch_audio",
    "list_split_files",
    "load_inventory",
    "parse_fname",
    "scan_folder",
    "scan_shards",
    "valid_recording_ids",
]
