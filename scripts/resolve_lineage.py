"""Evidence-based lineage resolution for core IndicSynth reference recordings.

The core synthetic windows reference upstream Kathbath recordings (voice-conversion
sources and targets, TTS conditioning/reference recordings).  Initial preparation
could only confirm references whose recording happened to fall inside the
partially scanned inventory shards; everything else was marked ``parsed_only``.

This tool resolves those references against the **complete** authenticated
Kathbath train shards using parquet footer statistics as a cheap exact index:

1. read every train shard's footer (≈ 64 KiB) and collect each row group's
   ``fname`` min/max statistics;
2. a reference id can only occur in a row group whose statistics bracket it
   (fname = ``<record>-<speaker>-<gender>.m4a``), so prune with the string
   bounds ``record + "-"`` .. ``record + "~"``;
3. read the ``fname`` column of the (few) bracketing row groups and confirm
   exact membership, producing evidence: shard, row group, global row index and
   the exact stored fname;
4. references found in the upstream ``valid``/``test`` shards (already fully
   enumerated) are reported as ``upstream_eval_parent``; references that exist
   in no train row group are reported as ``not_found`` (proof of absence:
   every row group that could contain them was examined).

Results are cached per language and charged to the normal download ledger.
``--apply`` writes the verification status plus evidence and the preserved raw
row metadata into the staged core records, and records rows that must be
replaced in ``staging/joint/lineage_excluded.<Language>.json`` for the planner.

Usage (data environment):

    python -m scripts.resolve_lineage scan
    python -m scripts.resolve_lineage apply
"""

from __future__ import annotations

import argparse
import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.dataset_prep.budget import BudgetExceeded, DownloadLedger
from src.dataset_prep.config import DEFAULT_CONFIG_PATH, load_prep_config
from src.dataset_prep.records import read_json, write_json_atomic, write_jsonl_atomic

JOINT = "joint"
CORE = Path("staging") / "core"
HF_DATASET = "ai4bharat/Kathbath"
FOLDER_BY_LANGUAGE = {
    "Bengali": "bengali",
    "Gujarati": "gujarati",
    "Hindi": "hindi",
    "Kannada": "kannada",
    "Malayalam": "malayalam",
    "Marathi": "marathi",
    "Odia": "odia",
    "Punjabi": "punjabi",
    "Sanskrit": "sanskrit",
    "Tamil": "tamil",
    "Telugu": "telugu",
    "Urdu": "urdu",
}

VALID_STATUSES = ("verified_train", "upstream_eval_parent", "not_found")


def _record_id(reference: str) -> str | None:
    value = str(reference or "").strip()
    return value.split("-", 1)[0] or None


def _parse_reference(reference: str) -> dict[str, Any] | None:
    value = str(reference or "").strip()
    parts = value.split("-")
    if len(parts) < 3:
        return None
    return {"record": parts[0], "speaker": parts[1], "gender": parts[2].split(".")[0]}


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


def _core_synthetic_rows(cfg) -> list[dict[str, Any]]:
    path = cfg.manifests_dir / "source_windows.indicsynth.jsonl"
    return _read_jsonl(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _outstanding_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """Per language: outstanding parsed_only references by role."""
    outstanding: dict[str, dict[str, set[str]]] = {}
    for row in rows:
        parent = row.get("parent_refs") or {}
        language = str(row.get("spoken_language"))
        entry = outstanding.setdefault(language, {"source": set(), "target": set()})
        if parent.get("source_kind") != "tts_target_only":
            reference = parent.get("source_reference")
            if reference and parent.get("source_parent_verification") != "verified_train":
                record = _record_id(str(reference))
                if record:
                    entry["source"].add(record)
        reference = parent.get("target_reference")
        if reference and parent.get("target_parent_verification") != "verified_train":
            record = _record_id(str(reference))
            if record:
                entry["target"].add(record)
    return outstanding


def _resolve_language(
    cfg,
    ledger: DownloadLedger,
    *,
    language: str,
    record_ids: Iterable[str],
    workers: int = 6,
) -> dict[str, dict[str, Any]]:
    """Resolve record ids against the complete Kathbath train shards."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    folder = FOLDER_BY_LANGUAGE[language]
    cache_path = cfg.staging_dir / JOINT / f"lineage_resolution.{language}.json"
    cache: dict[str, dict[str, Any]] = read_json(cache_path, default={}) or {}

    files = read_json(cfg.staging_dir / "kathbath" / f"files.{folder}.json", default={}) or {}
    train_shards = list((files.get("splits") or {}).get("train", []))
    if not train_shards:
        raise RuntimeError(f"missing Kathbath train shard list for {language}")

    held_out = set(read_json(cfg.staging_dir / JOINT / f"kathbath_heldout.{language}.json", default=[]) or [])
    inventory = _read_jsonl(cfg.staging_dir / JOINT / f"kathbath_inventory.{language}.jsonl")
    inventory_by_record = {
        str(row["fname"]).split("-", 1)[0]: row for row in inventory
    }

    pending: list[str] = []
    for record in sorted(set(record_ids)):
        if record in cache:
            continue
        if record in held_out:
            cache[record] = {"status": "upstream_eval_parent", "evidence": {"partition": "valid/test"}}
            continue
        row = inventory_by_record.get(record)
        if row is not None:
            cache[record] = {
                "status": "verified_train",
                "evidence": {
                    "method": "scanned_inventory",
                    "shard": row["shard"],
                    "row_group": int(row["row_group"]),
                    "row_index": int(row["row_index"]),
                    "fname": row["fname"],
                },
            }
            continue
        pending.append(record)

    if not pending:
        write_json_atomic(cache_path, cache)
        return cache

    print(
        f"  [resolve {language}] {len(pending)} reference(s) across {len(train_shards)} train shards"
        f" (workers={workers}); ledger={ledger.total_bytes}",
        flush=True,
    )
    fs = HfFileSystem()
    found: dict[str, dict[str, Any]] = {}
    unresolved = set(pending)
    lock = threading.Lock()

    def _scan_shard(shard: str) -> dict[str, Any]:
        counted = _CountingFile(fs.open(f"datasets/{HF_DATASET}/{shard}", "rb"))
        before = counted.bytes
        local_found: dict[str, dict[str, Any]] = {}
        try:
            parquet = pq.ParquetFile(counted)
            metadata = parquet.metadata
            fname_index = metadata.schema.names.index("fname")
            row_counts = [metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)]
            offsets = [0]
            for count in row_counts:
                offsets.append(offsets[-1] + count)
            group_bounds: list[tuple[int, str | None, str | None]] = []
            for g in range(metadata.num_row_groups):
                stats = metadata.row_group(g).column(fname_index).statistics
                group_bounds.append((g, stats.min if stats else None, stats.max if stats else None))
            with lock:
                records = list(unresolved)
            shard_min = min((lo for _, lo, _ in group_bounds if lo is not None), default=None)
            shard_max = max((hi for _, _, hi in group_bounds if hi is not None), default=None)
            candidates = {
                record
                for record in records
                if shard_min is None
                or shard_max is None
                or (shard_min <= record + "~" and shard_max >= record + "-")
            }
            if not candidates:
                return {"found": {}, "bytes": counted.bytes - before, "shard": shard}
            group_rows: dict[int, list[str]] = defaultdict(list)
            for g, lo, hi in group_bounds:
                for record in candidates:
                    if lo is None or hi is None or (lo <= record + "~" and hi >= record + "-"):
                        group_rows[g].append(record)
            for g in sorted(group_rows):
                table = parquet.read_row_group(g, columns=["fname"]).to_pydict()
                by_record: dict[str, tuple[int, str]] = {}
                for index, fname in enumerate(table["fname"]):
                    rid = str(fname).split("-", 1)[0]
                    by_record.setdefault(rid, (index, str(fname)))
                for record in group_rows[g]:
                    hit = by_record.get(record)
                    if hit is None:
                        continue
                    local_index, fname = hit
                    local_found[record] = {
                        "status": "verified_train",
                        "evidence": {
                            "method": "full_train_shard_scan",
                            "shard": shard,
                            "row_group": g,
                            "row_index": offsets[g] + local_index,
                            "fname": fname,
                        },
                    }
            return {"found": local_found, "bytes": counted.bytes - before, "shard": shard}
        finally:
            counted.close()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_scan_shard, shard): shard for shard in train_shards}
        done = 0
        for future in as_completed(futures):
            result = future.result()
            done += 1
            if result["bytes"]:
                try:
                    ledger.charge("kathbath", result["bytes"])
                except BudgetExceeded as error:
                    print(f"  [resolve {language}] budget reached: {error}", flush=True)
                    cache.update(found)
                    for record in unresolved:
                        cache.setdefault(record, {"status": "budget_stopped"})
                    write_json_atomic(cache_path, cache)
                    raise
            for record, entry in result["found"].items():
                if record not in found:
                    found[record] = entry
            unresolved -= set(result["found"])
            if done % 4 == 0 or not unresolved:
                print(
                    f"  [resolve {language}] shards {done}/{len(train_shards)};"
                    f" resolved {len(pending) - len(unresolved)}/{len(pending)}; ledger={ledger.total_bytes}",
                    flush=True,
                )

    cache.update(found)
    for record in sorted(unresolved):
        # Absence is proven once every shard has been examined: each row group
        # whose fname statistics bracket the id was read completely.
        cache[record] = {"status": "not_found", "evidence": {"method": "full_train_shard_scan"}}
    write_json_atomic(cache_path, cache)
    return cache


def command_scan(cfg, ledger: DownloadLedger, languages: Sequence[str] | None, *, workers: int = 6) -> int:
    rows = _core_synthetic_rows(cfg)
    if not rows:
        print("no staged synthetic rows; run the core stage first")
        return 1
    outstanding = _outstanding_rows(rows)
    selected = languages or sorted(outstanding)
    total = 0
    for language in selected:
        entry = outstanding.get(language)
        if not entry:
            print(f"  [resolve {language}] nothing outstanding")
            continue
        record_ids = sorted(entry["source"] | entry["target"])
        if not record_ids:
            print(f"  [resolve {language}] nothing outstanding")
            continue
        total += len(record_ids)
        cache = _resolve_language(cfg, ledger, language=language, record_ids=record_ids, workers=workers)
        counts: dict[str, int] = defaultdict(int)
        for record in record_ids:
            counts[str(cache.get(record, {}).get("status"))] += 1
        print(f"  [resolve {language}] {dict(counts)}")
    print(f"resolved {total} outstanding reference(s); ledger total now {ledger.total_bytes}")
    return 0


def _apply_language(cfg, language: str, cache: Mapping[str, Any]) -> dict[str, Any]:
    windows_path = cfg.manifests_dir / "source_windows.indicsynth.jsonl"
    recordings_path = cfg.manifests_dir / "source_recordings.indicsynth.jsonl"
    windows = _read_jsonl(windows_path)
    recordings = _read_jsonl(recordings_path)
    recordings_by_id = {str(row.get("recording_id")): row for row in recordings}
    inventory_rows = _read_jsonl(cfg.staging_dir / JOINT / f"kathbath_inventory.{language}.jsonl")
    inventory_by_record = {
        str(row["fname"]).split("-", 1)[0]: row for row in inventory_rows
    }

    def _entry_for(record: str | None, role: str) -> dict[str, Any] | None:
        if not record:
            return None
        entry = cache.get(record)
        if entry:
            return entry
        row = inventory_by_record.get(record)
        if row is not None:
            return {
                "status": "verified_train",
                "evidence": {
                    "method": "scanned_inventory",
                    "shard": row["shard"],
                    "row_group": int(row["row_group"]),
                    "row_index": int(row["row_index"]),
                    "fname": row["fname"],
                },
            }
        return None

    bad_rows: list[dict[str, Any]] = []
    updated = 0
    for row in windows:
        if str(row.get("spoken_language")) != language:
            continue
        parent = dict(row.get("parent_refs") or {})
        source_kind = str(parent.get("source_kind"))
        evidence: dict[str, Any] = dict(parent.get("reference_evidence") or {})
        changed = "upstream" not in parent

        if source_kind != "tts_target_only":
            reference = parent.get("source_reference")
            record = _record_id(str(reference)) if reference else None
            entry = _entry_for(record, "source")
            if entry:
                parent["source_parent_verification"] = entry["status"]
                evidence["source"] = {"reference": reference, **(entry.get("evidence") or {})}
                changed = True
        else:
            evidence["source"] = {
                "reference": None,
                "note": "not applicable: TTS generation has no voice-conversion source step",
            }
            changed = True

        reference = parent.get("target_reference")
        record = _record_id(str(reference)) if reference else None
        entry = _entry_for(record, "target")
        if entry:
            parent["target_parent_verification"] = entry["status"]
            evidence["target"] = {"reference": reference, **(entry.get("evidence") or {})}
            changed = True

        if changed:
            index = int(str(row["window_id"]).rsplit("-", 1)[-1])
            parent["reference_evidence"] = evidence
            # Preserve the raw upstream row metadata next to the record so the
            # manifests stay self-describing without the staging caches.
            speakers = row.get("speaker_ids") or {}
            parent["upstream"] = {
                "dataset": "IndicSynth",
                "language": language,
                "upstream_split": "train",
                "row_index": index,
                "generator": row.get("generator"),
                "transcript": parent.get("transcript"),
                "source": {
                    "record": parent.get("source_record"),
                    "reference": parent.get("source_reference"),
                    "declared_speaker": speakers.get("source"),
                    "kind": parent.get("source_kind"),
                },
                "target": {
                    "record": parent.get("target_record"),
                    "reference": parent.get("target_reference"),
                    "declared_speaker": speakers.get("target"),
                },
            }
            row["parent_refs"] = parent
            recording = recordings_by_id.get(str(row.get("recording_id")))
            if recording is not None:
                recording["parent_refs"] = parent
            updated += 1

        statuses = {str(parent.get("source_parent_verification")), str(parent.get("target_parent_verification"))}
        if statuses & {"upstream_eval_parent", "not_found", "budget_stopped"}:
            reasons = sorted(statuses & {"upstream_eval_parent", "not_found", "budget_stopped"})
            index = int(str(row["window_id"]).rsplit("-", 1)[-1])
            bad_rows.append({"row_index": index, "window_id": row["window_id"], "reasons": reasons})

    write_jsonl_atomic(windows_path, windows)
    write_jsonl_atomic(recordings_path, recordings)
    exclusion_path = cfg.staging_dir / JOINT / f"lineage_excluded.{language}.json"
    write_json_atomic(exclusion_path, bad_rows)
    return {"language": language, "updated": updated, "excluded": len(bad_rows)}


def command_apply(cfg, languages: Sequence[str] | None) -> int:
    rows = _core_synthetic_rows(cfg)
    languages_present = sorted({str(row.get("spoken_language")) for row in rows})
    selected = languages or languages_present
    needs_replacement: dict[str, int] = {}
    for language in selected:
        cache_path = cfg.staging_dir / JOINT / f"lineage_resolution.{language}.json"
        cache = read_json(cache_path, default={}) or {}
        if not cache:
            print(f"  [apply {language}] no resolution cache; run scan first")
            continue
        # Reclassify early scans that lacked the held-out id list: a reference
        # that exists in the upstream valid/test partitions is not "not_found".
        held_out = set(
            read_json(cfg.staging_dir / JOINT / f"kathbath_heldout.{language}.json", default=[]) or []
        )
        reclassified = 0
        for record, entry in cache.items():
            if entry.get("status") == "not_found" and record in held_out:
                cache[record] = {"status": "upstream_eval_parent", "evidence": {"partition": "valid/test"}}
                reclassified += 1
        if reclassified:
            write_json_atomic(cache_path, cache)
            print(f"  [apply {language}] reclassified {reclassified} reference(s) as upstream_eval_parent")
        result = _apply_language(cfg, language, cache)
        if result["excluded"]:
            needs_replacement[str(result["language"])] = int(result["excluded"])
        print(f"  [apply {language}] updated={result['updated']} rows requiring replacement={result['excluded']}")
    if needs_replacement:
        print("replacement needed for:", needs_replacement)
        print("next: re-run core for those languages, then scan+apply again")
    else:
        print("all staged references carry verified evidence")
    return 0


def command_crosscheck(cfg, ledger: DownloadLedger, *, workers: int = 6, max_bytes: int | None = None) -> int:
    """Confirm not_found ids against every *other* language folder's train shards.

    Wide row-group ranges make footer statistics a cheap index but give many
    bracket matches; membership therefore requires reading the bracketing
    ``fname`` column chunks, exactly like the native-folder pass.  ``max_bytes``
    bounds the total spend of this optional confirmation; ids not confirmed
    within the cap stay unresolved and are replaced instead.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    from huggingface_hub import HfFileSystem
    import pyarrow.parquet as pq

    caches: dict[str, dict[str, Any]] = {}
    missing: dict[str, set[str]] = {}
    for path in sorted((cfg.staging_dir / JOINT).glob("lineage_resolution.*.json")):
        language = path.stem.split(".", 1)[1]
        cache = read_json(path, default={}) or {}
        caches[language] = cache
        ids = {record for record, entry in cache.items() if entry.get("status") == "not_found"}
        if ids:
            missing[language] = ids
    pending = set().union(*missing.values()) if missing else set()
    if not pending:
        print("nothing to cross-check")
        return 0
    print(f"cross-checking {len(pending)} id(s) across all language folders", flush=True)

    fs = HfFileSystem()
    found: dict[str, dict[str, Any]] = {}
    start_total = ledger.total_bytes

    def scan_shard(folder: str, shard: str, ids: set[str]) -> dict[str, Any]:
        counted = None
        before = 0
        try:
            counted = _CountingFile(fs.open(f"datasets/{HF_DATASET}/{shard}", "rb"))
            before = counted.bytes
            parquet = pq.ParquetFile(counted)
            metadata = parquet.metadata
            fname_index = metadata.schema.names.index("fname")
            row_counts = [metadata.row_group(g).num_rows for g in range(metadata.num_row_groups)]
            offsets = [0]
            for count in row_counts:
                offsets.append(offsets[-1] + count)
            group_rows: dict[int, list[str]] = defaultdict(list)
            for g in range(metadata.num_row_groups):
                stats = metadata.row_group(g).column(fname_index).statistics
                lo = stats.min if stats else None
                hi = stats.max if stats else None
                if lo is None or hi is None:
                    group_rows[g].extend(ids)
                    continue
                for record in ids:
                    if lo <= record + "~" and hi >= record + "-":
                        group_rows[g].append(record)
            local: dict[str, dict[str, Any]] = {}
            for g in sorted(group_rows):
                table = parquet.read_row_group(g, columns=["fname"]).to_pydict()
                by_record: dict[str, tuple[int, str]] = {}
                for index, fname in enumerate(table["fname"]):
                    rid = str(fname).split("-", 1)[0]
                    by_record.setdefault(rid, (index, str(fname)))
                for record in group_rows[g]:
                    hit = by_record.get(record)
                    if hit is None:
                        continue
                    local_index, fname = hit
                    local[record] = {
                        "status": "verified_train",
                        "evidence": {
                            "method": "cross_folder_scan",
                            "folder": folder,
                            "shard": shard,
                            "row_group": g,
                            "row_index": offsets[g] + local_index,
                            "fname": fname,
                        },
                    }
            return {"found": local, "bytes": counted.bytes - before if counted else 0, "error": None}
        except Exception as error:  # network flakiness: retried at the end
            return {"found": {}, "bytes": counted.bytes - before if counted else 0, "error": f"{shard}: {error}"}
        finally:
            if counted is not None:
                counted.close()

    for folder in sorted(FOLDER_BY_LANGUAGE.values()):
        if max_bytes is not None and (ledger.total_bytes - start_total) >= int(max_bytes):
            print(
                f"  byte cap reached ({ledger.total_bytes - start_total} >= {max_bytes});"
                " stopping cross-folder search",
                flush=True,
            )
            break
        unresolved_here = {
            record
            for language, ids in missing.items()
            if FOLDER_BY_LANGUAGE[language] != folder
            for record in ids
        } - set(found)
        if not unresolved_here:
            continue
        files = read_json(cfg.staging_dir / "kathbath" / f"files.{folder}.json", default={}) or {}
        shards = list((files.get("splits") or {}).get("train", []))
        errors: list[str] = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = [pool.submit(scan_shard, folder, shard, unresolved_here) for shard in shards]
            for future in as_completed(futures):
                result = future.result()
                if result["bytes"]:
                    ledger.charge("kathbath", result["bytes"])
                if result.get("error"):
                    errors.append(str(result["error"]))
                for record, entry in result["found"].items():
                    found.setdefault(record, entry)
        # one retry pass for shards that failed on transient network errors
        retry_shards = [shard for shard in shards if any(shard in message for message in errors)]
        for shard in retry_shards:
            result = scan_shard(folder, shard, unresolved_here - set(found))
            if result["bytes"]:
                ledger.charge("kathbath", result["bytes"])
            for record, entry in result["found"].items():
                found.setdefault(record, entry)
        if errors:
            print(f"  [{folder}] transient shard errors: {len(errors)} (retried {len(retry_shards)})", flush=True)
        print(
            f"  [{folder}] searched {len(shards)} shards; found {len(found)}/{len(pending)} so far;"
            f" ledger={ledger.total_bytes}",
            flush=True,
        )

    for language, cache in caches.items():
        updated = False
        for record in missing.get(language, set()):
            if record in found:
                cache[record] = found[record]
                updated = True
        if updated:
            write_json_atomic(cfg.staging_dir / JOINT / f"lineage_resolution.{language}.json", cache)
    remaining = pending - set(found)
    print(f"cross-folder result: found {len(found)}/{len(pending)}; still missing {len(remaining)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["scan", "apply", "crosscheck"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--languages", nargs="*", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--max-bytes", type=int, default=None, help="optional cap for the cross-folder pass")
    args = parser.parse_args(argv)
    cfg = load_prep_config(args.config, dataset_root=args.dataset_root)
    if args.action == "scan":
        ledger = DownloadLedger.load(cfg.manifests_dir / "downloads.json", cfg.budget.max_download_bytes)
        return command_scan(cfg, ledger, args.languages, workers=args.workers)
    if args.action == "crosscheck":
        ledger = DownloadLedger.load(cfg.manifests_dir / "downloads.json", cfg.budget.max_download_bytes)
        return command_crosscheck(cfg, ledger, workers=args.workers, max_bytes=args.max_bytes)
    return command_apply(cfg, args.languages)


if __name__ == "__main__":
    raise SystemExit(main())
