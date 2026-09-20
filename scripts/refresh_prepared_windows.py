"""Re-materialize already-selected windows from local raw audio.

The preprocessing contract (``voxsentinel-prep-2``) introduced a strict
4-second cap, speech-activity facts and explicit decode/overflow records.  This
maintenance script refreshes the non-core pools that were built under the
earlier contract **without any download**: it re-reads the raw files already on
disk, re-runs the shared window materializer, and rewrites the recording/window
facts (``original_audio``, ``decoded``, ``prepared_audio``, ``window``,
``preprocessing_version``) atomically.

Usage (from the repository root, data environment active):

    python -m scripts.refresh_prepared_windows --sources nisp nptel asvspoof2019
    python -m scripts.refresh_prepared_windows --sources asvspoof2019 --dry-run

Windows whose raw audio cannot be found locally are reported and left untouched;
no network access and no ledger charge happen here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from src.audio.prepare import AudioDecodeError
from src.dataset_prep.config import DEFAULT_CONFIG_PATH, PrepConfig, load_prep_config
from src.dataset_prep.materialize import materialize_window
from src.dataset_prep.records import read_jsonl, write_jsonl_atomic

RawLocator = Callable[[PrepConfig, dict[str, Any]], Path | None]


def _locator_nisp(cfg: PrepConfig, row: dict[str, Any]) -> Path | None:
    name = Path(str(row.get("source_file", ""))).name
    matches = sorted((cfg.raw_dir / "nisp").glob(f"*/{name}"))
    return matches[0] if matches else None


def _locator_nptel(cfg: PrepConfig, row: dict[str, Any]) -> Path | None:
    name = Path(str(row.get("source_file", ""))).name
    candidate = cfg.raw_dir / "nptel" / name
    return candidate if candidate.exists() else None


def _locator_asvspoof(cfg: PrepConfig, row: dict[str, Any]) -> Path | None:
    name = Path(str(row.get("source_file", ""))).name
    split = "train" if "train" in str(row.get("source_file", "")).lower() else "dev"
    candidate = cfg.raw_dir / "asvspoof2019" / split / name
    if candidate.exists():
        return candidate
    matches = sorted((cfg.raw_dir / "asvspoof2019").glob(f"*/{name}"))
    return matches[0] if matches else None


LOCATORS: dict[str, RawLocator] = {
    "nisp": _locator_nisp,
    "nptel": _locator_nptel,
    "asvspoof2019": _locator_asvspoof,
}


def refresh_source(cfg: PrepConfig, source_id: str, *, dry_run: bool = False) -> dict[str, Any]:
    locator = LOCATORS[source_id]
    recordings_path = cfg.manifests_dir / f"source_recordings.{source_id}.jsonl"
    windows_path = cfg.manifests_dir / f"source_windows.{source_id}.jsonl"
    recordings = read_jsonl(recordings_path)
    windows = read_jsonl(windows_path)
    if not windows:
        return {"source": source_id, "windows": 0, "refreshed": 0, "missing_raw": 0, "failed": 0}

    recordings_by_id = {str(row.get("recording_id")): row for row in recordings}
    refreshed = 0
    missing_raw: list[str] = []
    failed: list[str] = []
    over_length_before = sum(
        1 for row in windows if int((row.get("window") or {}).get("num_samples", 0)) > 64000
    )
    over_length_after = 0
    for row in windows:
        raw_path = locator(cfg, row)
        if raw_path is None:
            missing_raw.append(str(row.get("window_id")))
            continue
        prepared_rel = str((row.get("prepared_audio") or {}).get("path") or "")
        prepared_path = (
            cfg.dataset_root / prepared_rel
            if prepared_rel
            else cfg.prepared_dir / source_id / f"{row['window_id']}.wav"
        )
        try:
            materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
        except (AudioDecodeError, ValueError) as error:
            failed.append(f"{row.get('window_id')}: {error}")
            continue
        original_facts = {**materialized.decoded.original_facts(), "sha256": materialized.original_sha256}
        decoded_facts = materialized.decoded_facts()
        row["original_audio"] = original_facts
        row["prepared_audio"] = materialized.prepared_facts(cfg.dataset_root)
        row["window"] = materialized.window_facts(original_rate=materialized.decoded.original_rate)
        row["preprocessing_version"] = str(cfg.preprocessing.get("config_version"))
        if int(row["window"].get("num_samples", 0)) > 64000:
            over_length_after += 1
        recording = recordings_by_id.get(str(row.get("recording_id")))
        if recording is not None:
            recording["original_audio"] = original_facts
            recording["decoded"] = decoded_facts
        refreshed += 1

    if not dry_run and refreshed:
        write_jsonl_atomic(windows_path, windows)
        write_jsonl_atomic(recordings_path, recordings)
    return {
        "source": source_id,
        "windows": len(windows),
        "refreshed": refreshed,
        "missing_raw": len(missing_raw),
        "missing_raw_ids": missing_raw[:5],
        "failed": len(failed),
        "failed_examples": failed[:5],
        "over_length_before": over_length_before,
        "over_length_after": over_length_after,
        "dry_run": dry_run,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--sources", nargs="*", default=sorted(LOCATORS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_prep_config(args.config, dataset_root=args.dataset_root)
    for source_id in args.sources:
        if source_id not in LOCATORS:
            print(f"unknown source '{source_id}'; expected one of {sorted(LOCATORS)}")
            return 2
        summary = refresh_source(cfg, source_id, dry_run=args.dry_run)
        print(
            f"{source_id}: refreshed {summary['refreshed']}/{summary['windows']} windows "
            f"(missing raw: {summary['missing_raw']}, failed: {summary['failed']}, "
            f"over-length before: {summary['over_length_before']} -> after: {summary['over_length_after']})"
        )
        for item in summary["missing_raw_ids"]:
            print(f"  missing raw for {item}")
        for item in summary["failed_examples"]:
            print(f"  failed {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
