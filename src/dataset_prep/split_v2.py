"""VoxSentinel v2 Indic split preparation: discovery, exposure closure, assignment.

Scope (per ``docs/DATA_SPLIT_AGENT_MASTER_PROMPT.md``): prepare an audit-ready
Indic-only v2 split (Kathbath genuine + IndicSynth synthetic) whose new test
components are disjoint from all recorded prior exposure.  This module only
reads upstream metadata/acquisition with revision-pinned, ledger-charged,
resumable requests; it never trains, extracts features or evaluates a detector.

Design rules implemented here:

* Every ``hf`` read is pinned to the recorded Kathbath revision via
  ``HfFileSystem`` ``@revision`` paths; a revision mismatch aborts rather than
  silently reading a different snapshot.
* All transferred bytes pass through the shared acquisition ledger
  (``artifacts/datasets/manifests/downloads.json``); metadata scans and audio
  row-group reads are both charged.  The ledger is never reset.
* Discovery is progressive and resumable: per-language fresh inventories are
  appended only after a shard scan succeeds, and failed shards are recorded as
  errors (never as successfully scanned).
* Prior-exposure registry extension scans local run/prediction artifacts for
  window ids and asserts every found id maps into the frozen v1 core; anything
  else is reported as unknown instead of being assumed unexposed.
"""
from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .adapters import indicsynth as isynth
from .adapters.common import SourceRun
from .adapters.kathbath import (
    HF_DATASET,
    INVENTORY_COLUMNS,
    _CountingFile,
    parse_fname,
)
from .budget import BudgetExceeded, DownloadLedger
from .config import PrepConfig, load_prep_config
from .materialize import sha256_file
from .records import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from . import httpio

V1_ROOT = Path("artifacts/datasets")
V2_ROOT = Path("artifacts/datasets-v2")
CONFIG_PATH = "configs/datasets.yaml"
KATHBATH_REVISION = "5b9e92849222026d9141acba4e8434fe816396bf"
INDICSYNTH_REVISION = "c0a10386b723717aff682f757bd67f72983f269f"
SCHEMA_REGISTRY = "voxsentinel.exposure_registry.v2"
SCHEMA_FRESH = "voxsentinel.kathbath_fresh_inventory.v1"
SCHEMA_ISYNTH = "voxsentinel.indicsynth_fresh_candidates.v1"

CORE_LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
    "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
]
LANGUAGE_SLUGS = {lang: lang.lower() for lang in CORE_LANGUAGES}

WINDOW_ID_RE = re.compile(r"(?:kathbath|indicsynth|svarah|nisp|nptel|asvspoof2019)-[a-z_]+-[\w-]+")


# --------------------------------------------------------------------------- #
# Config / ledger helpers
# --------------------------------------------------------------------------- #

def v1_cfg() -> PrepConfig:
    return load_prep_config(CONFIG_PATH, dataset_root=V1_ROOT)


def v2_cfg() -> PrepConfig:
    cfg = load_prep_config(CONFIG_PATH, dataset_root=V2_ROOT)
    cfg.ensure_dirs()
    return cfg


def v2_ledger(cfg: PrepConfig) -> DownloadLedger:
    """The single shared acquisition ledger (never reset, never duplicated)."""
    return DownloadLedger.load(V1_ROOT / "manifests" / "downloads.json", cfg.budget.max_download_bytes)


def ledger_facts(ledger: DownloadLedger) -> dict[str, Any]:
    return {
        "cap_bytes": ledger.cap_bytes,
        "total_bytes": ledger.total_bytes,
        "remaining_bytes": ledger.remaining_bytes,
        "events": ledger.events,
        "per_source_bytes": dict(sorted(ledger.per_source.items())),
    }


# --------------------------------------------------------------------------- #
# Phase 1: extended prior-exposure registry
# --------------------------------------------------------------------------- #

def _load_v1_core() -> dict[str, list[dict[str, Any]]]:
    rows = {}
    for split in ("train", "val"):
        path = V1_ROOT / "manifests" / f"windows.core_{split}.jsonl"
        rows[split] = read_jsonl(path)
    return rows


def _scan_artifact_window_ids(root: Path) -> dict[str, dict[str, Any]]:
    """Find window ids in local run/prediction artifacts (bounded text scan)."""
    found: dict[str, dict[str, Any]] = {}
    suffixes = {".json", ".jsonl", ".csv", ".txt", ".md"}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stat().st_size > 40 * 1024 * 1024:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for match in WINDOW_ID_RE.finditer(text):
            wid = match.group(0)
            entry = found.setdefault(wid, {"artifacts": []})
            rel = str(path)
            if rel not in entry["artifacts"]:
                entry["artifacts"].append(rel)
    return found


def command_registry(out_dir: Path | None = None) -> dict[str, Any]:
    """Extend the initial prior-exposure registry with all local artifact use.

    Writes ``planning/exposure_registry.v2.json``; the earlier
    ``prior_exposure_exclusions.json`` stays untouched.
    """
    out = out_dir or (V2_ROOT / "planning")
    out.mkdir(parents=True, exist_ok=True)
    core = _load_v1_core()

    speaker_keys, reference_keys, audio_sha, prepared_sha = set(), set(), set(), set()
    window_ids = set()
    for split, rows in core.items():
        for row in rows:
            window_ids.add(row["window_id"])
            lang = row["spoken_language"].lower()
            values = ([row["source_file"]] if row["dataset_id"] == "kathbath" else
                      [row.get("parent_refs", {}).get(f"{role}_reference") for role in ("source", "target")])
            from scripts.gru_identity_audit import reference  # canonical parser shared with prior audits

            for value in values:
                parsed = reference(value)
                if parsed:
                    speaker_keys.add(("kathbath", lang, parsed["speaker"]))
                    reference_keys.add(("kathbath", lang, parsed["canonical"]))
            for kind in ("original_audio", "prepared_audio"):
                digest = (row.get(kind) or {}).get("sha256")
                if digest:
                    (audio_sha if kind == "original_audio" else prepared_sha).add(digest)

    # find extra local usage
    artifact_roots = [Path("artifacts/runs"), Path("artifacts/prediction-path"), Path("artifacts/releases")]
    usage: dict[str, dict[str, Any]] = {}
    for root in artifact_roots:
        if root.exists():
            for wid, entry in _scan_artifact_window_ids(root).items():
                usage.setdefault(wid, {"artifacts": []})
                usage[wid]["artifacts"].extend(a for a in entry["artifacts"] if a not in usage[wid]["artifacts"])

    unknown = sorted(wid for wid in usage if wid not in window_ids)
    provenance = [
        {"artifact_root": str(root), "artifacts_seen": sum(1 for w in usage.values() if any(str(root) in a for a in w["artifacts"]))}
        for root in artifact_roots if root.exists()
    ]

    prior = read_json(V2_ROOT / "planning" / "prior_exposure_exclusions.json", default={}) or {}
    registry = {
        "schema": SCHEMA_REGISTRY,
        "scope": (
            "All identities and recordings of the frozen v1 core train+development windows, extended with every "
            "locally scanned run/prediction artifact; language-scoped keys; used to exclude the v2 test set."
        ),
        "counts": {
            "window_ids": len(window_ids),
            "speaker_keys": len(speaker_keys),
            "reference_keys": len(reference_keys),
            "original_audio_sha256": len(audio_sha),
            "prepared_audio_sha256": len(prepared_sha),
            "artifact_window_ids_seen": len(usage),
            "artifact_window_ids_unknown": len(unknown),
        },
        "window_ids": sorted(window_ids),
        "speaker_keys": [list(k) for k in sorted(speaker_keys)],
        "reference_keys": [list(k) for k in sorted(reference_keys)],
        "original_audio_sha256": sorted(audio_sha),
        "prepared_audio_sha256": sorted(prepared_sha),
        "artifact_usage": {wid: sorted(entry["artifacts"]) for wid, entry in sorted(usage.items())},
        "artifact_usage_unknown_ids": unknown,
        "prior_registry_reference": {
            "path": "artifacts/datasets-v2/planning/prior_exposure_exclusions.json",
            "counts": {k: len(v) if isinstance(v, (list, dict)) else v for k, v in prior.items() if k != "scope"},
        },
        "source_manifest_sha256": {
            str(V1_ROOT / "manifests" / f"windows.core_{split}.jsonl"):
                sha256_file(V1_ROOT / "manifests" / f"windows.core_{split}.jsonl")
            for split in ("train", "val")
        },
        "limits": [
            "Language-scoped identity; cross-language person independence not established.",
            "Unknown artifacts that cannot be mapped to the frozen core are listed, not assumed unexposed.",
            "Feature caches and model runs reused these exact windows; no other core windows are known-exposed.",
        ],
    }
    (out / "exposure_registry.v2.json").write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    return registry


def load_registry() -> dict[str, Any]:
    path = V2_ROOT / "planning" / "exposure_registry.v2.json"
    if not path.exists():
        raise FileNotFoundError("run `prepare_dataset_v2 registry` first")
    return read_json(path) or {}


def speakers_by_language(registry: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for dataset, lang, speaker in registry.get("speaker_keys", []):
        result[str(lang)].add(str(speaker))
    return result


def references_by_language(registry: Mapping[str, Any]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for dataset, lang, canonical in registry.get("reference_keys", []):
        result[str(lang)].add(str(canonical))
        # also index bare record ids for membership tests
        result[str(lang)].add(str(canonical).split("-")[0])
    return result


def closure_speakers_for(language: str, registry: Mapping[str, Any]) -> set[str]:
    return set(speakers_by_language(registry).get(language.lower(), set()))


def closure_references_for(language: str, registry: Mapping[str, Any]) -> set[str]:
    return set(references_by_language(registry).get(language.lower(), set()))


# --------------------------------------------------------------------------- #
# Phase 2a: Kathbath unscanned-shard discovery (revision-pinned, charged)
# --------------------------------------------------------------------------- #

def kathbath_shard_state() -> dict[str, dict[str, Any]]:
    """Return per-language shard bookkeeping from the v1 scan state + shard lists."""
    state: dict[str, dict[str, Any]] = {}
    for language in CORE_LANGUAGES:
        folder = LANGUAGE_SLUGS[language]
        scanned_path = V1_ROOT / "staging" / "joint" / f"kathbath_scanned.{language}.json"
        files_path = V1_ROOT / "staging" / "kathbath" / f"files.{folder}.json"
        if not files_path.exists():
            raise FileNotFoundError(f"missing shard list cache {files_path}; refusing to guess shard names")
        files = read_json(files_path) or {}
        revision = files.get("revision")
        if revision != KATHBATH_REVISION:
            raise RuntimeError(
                f"{language}: cached shard list revision {revision} != pinned {KATHBATH_REVISION}; "
                "stop and re-resolve rather than scanning a different snapshot"
            )
        scanned = read_json(scanned_path, default=[]) or []
        train = list(files["splits"].get("train", []))
        valid = list(files["splits"].get("valid", []))
        test = list(files["splits"].get("test", []))
        remaining = [s for s in train if s not in scanned]
        state[language] = {
            "folder": folder, "revision": revision,
            "train": train, "valid": valid, "test": test,
            "scanned_v1": scanned, "remaining": remaining,
        }
    return state


def _pinned_open(shard: str):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return _CountingFile(fs.open(f"datasets/{HF_DATASET}@{KATHBATH_REVISION}/{shard}", "rb"))


def scan_kathbath_shards_pinned(
    ledger: DownloadLedger,
    shard_names: Sequence[str],
    *,
    workers: int = 6,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Metadata-scan shards pinned to :data:`KATHBATH_REVISION`, charging the ledger.

    Mirrors ``adapters.kathbath.scan_shards`` with an explicit revision pin.
    Returns ``(rows, errors, bytes_per_shard)``.
    """
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    bytes_per_shard: dict[str, int] = {}

    def _scan_one(shard: str) -> tuple[list[dict[str, Any]], int]:
        counted = _pinned_open(shard)
        try:
            parquet = pq.ParquetFile(counted)
            table = parquet.read(columns=INVENTORY_COLUMNS).to_pydict()
            group_of_row: list[int] = []
            for group in range(parquet.metadata.num_row_groups):
                group_of_row.extend([group] * parquet.metadata.row_group(group).num_rows)
            out = []
            for index, fname in enumerate(table["fname"]):
                parsed = parse_fname(fname)
                if parsed is None:
                    continue
                out.append({
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
                    "revision": KATHBATH_REVISION,
                })
            return out, counted.bytes
        finally:
            counted.close()

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {pool.submit(_scan_one, shard): shard for shard in shard_names}
        for future in as_completed(futures):
            shard = futures[future]
            try:
                shard_rows, bytes_read = future.result()
            except Exception as error:  # noqa: BLE001 - isolate per-shard failures
                errors.append({"shard": shard, "error": f"{type(error).__name__}: {error}"})
                continue
            try:
                ledger.charge("kathbath", bytes_read)
            except BudgetExceeded as error:
                errors.append({"shard": shard, "error": f"budget: {error}"})
                break
            rows.extend(shard_rows)
            bytes_per_shard[shard] = bytes_read
    rows.sort(key=lambda r: (r["shard"], r["row_index"]))
    return rows, errors, bytes_per_shard


def _load_fresh_state(language: str) -> list[dict[str, Any]]:
    path = V2_ROOT / "staging" / f"kathbath_fresh.{language}.jsonl"
    return read_jsonl(path) if path.exists() else []


def command_discover_kathbath(
    *,
    languages: Sequence[str],
    max_shards_per_language: int,
    min_male_speakers: int,
    min_fresh_speakers: int,
    batch: int = 4,
    workers: int = 6,
) -> dict[str, Any]:
    """Progressively scan unscanned Kathbath shards per language.

    Stops for a language once it holds at least ``min_male_speakers`` male
    speakers *and* ``min_fresh_speakers`` distinct speakers overall among the
    newly scanned rows, or after ``max_shards_per_language`` new shards.
    """
    ledger = v2_ledger(v2_cfg())
    state = kathbath_shard_state()
    summary: dict[str, Any] = {}
    for language in languages:
        info = state[language]
        fresh = _load_fresh_state(language)
        fresh_shards = sorted({row["shard"] for row in fresh})
        remaining = [s for s in info["remaining"] if s not in fresh_shards]
        errors_all: list[dict[str, Any]] = []
        bytes_total = 0
        while remaining and len(fresh_shards) < max_shards_per_language:
            speakers = {row["speaker"] for row in fresh}
            male = {row["speaker"] for row in fresh if row["gender"] == "m"}
            if len(male) >= min_male_speakers and len(speakers) >= min_fresh_speakers:
                break
            window = remaining[:batch]
            rows, errors, byte_map = scan_kathbath_shards_pinned(ledger, window, workers=min(workers, len(window)))
            errors_all.extend(errors)
            failed = {e["shard"] for e in errors}
            ok_rows = [r for r in rows if r["shard"] not in failed]
            # resume-safety: persist rows from successful shards before continuing
            merged = {row["fname"]: row for row in fresh}
            for row in ok_rows:
                merged[row["fname"]] = row
            fresh = sorted(merged.values(), key=lambda r: (r["shard"], r["row_index"]))
            write_jsonl_atomic(V2_ROOT / "staging" / f"kathbath_fresh.{language}.jsonl", fresh)
            fresh_shards = sorted({row["shard"] for row in fresh})
            bytes_total += sum(v for k, v in byte_map.items() if k not in failed)
            remaining = [s for s in remaining if s not in fresh_shards and s not in failed]
            summary.setdefault(language, {})["batches"] = summary.get(language, {}).get("batches", 0) + 1
        speakers = {row["speaker"] for row in fresh}
        males = {row["speaker"] for row in fresh if row["gender"] == "m"}
        females = {row["speaker"] for row in fresh if row["gender"] == "f"}
        summary[language] = {
            **summary.get(language, {}),
            "scanned_v1": len(info["scanned_v1"]),
            "scanned_v2_new": len(fresh_shards),
            "remaining_unscanned": len([s for s in info["remaining"] if s not in fresh_shards]),
            "rows_fresh": len(fresh),
            "speakers_fresh": len(speakers),
            "male_speakers_fresh": len(males),
            "female_speakers_fresh": len(females),
            "rows_male": sum(1 for row in fresh if row["gender"] == "m"),
            "rows_female": sum(1 for row in fresh if row["gender"] == "f"),
            "errors": errors_all,
            "criteria_met": len(males) >= min_male_speakers and len(speakers) >= min_fresh_speakers,
        }
        write_json_atomic(V2_ROOT / "staging" / f"kathbath_fresh_summary.{language}.json", summary[language])
        print(json.dumps({"language": language, **{k: v for k, v in summary[language].items() if k != "errors"}}))
        print(f"  ledger now: {ledger.total_bytes} / {ledger.cap_bytes} ({ledger.remaining_bytes} left)")
        write_json_atomic(V2_ROOT / "staging" / "kathbath_discovery_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Phase 2b: upstream held-out (valid/test) record ids
# --------------------------------------------------------------------------- #

def command_discover_heldout(*, languages: Sequence[str], workers: int = 6) -> dict[str, Any]:
    """Scan upstream ``valid``/``test`` shards and record their recording ids.

    A language whose held-out shards fail to scan is recorded as failed — the
    discovery must not claim held-out membership was verified.
    """
    cfg = v2_cfg()
    ledger = v2_ledger(cfg)
    state = kathbath_shard_state()
    result: dict[str, Any] = {}
    out_path = V2_ROOT / "staging" / "kathbath_heldout.json"
    existing = read_json(out_path, default={}) or {}
    for language in languages:
        info = state[language]
        held = list(info["valid"]) + list(info["test"])
        if not held:
            result[language] = {"status": "none_upstream", "shards": [], "record_ids": []}
            continue
        if existing.get(language, {}).get("status") == "verified" and set(existing[language].get("shards", [])) == set(held):
            result[language] = existing[language]
            continue
        rows, errors, _ = scan_kathbath_shards_pinned(ledger, held, workers=min(workers, len(held)))
        if errors:
            result[language] = {"status": "failed", "shards": held, "errors": errors, "record_ids": []}
        else:
            result[language] = {
                "status": "verified", "shards": held,
                "record_ids": sorted({row["record_id"] for row in rows}),
                "revision": KATHBATH_REVISION,
            }
        print(json.dumps({"language": language, "status": result[language]["status"],
                          "heldout_records": len(result[language]["record_ids"])}))
    (V2_ROOT / "staging").mkdir(parents=True, exist_ok=True)
    merged = {**existing, **result}
    out_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    print(f"ledger now: {ledger.total_bytes} / {ledger.cap_bytes} ({ledger.remaining_bytes} left)")
    return result


# --------------------------------------------------------------------------- #
# Phase 2c: IndicSynth fresh synthetic candidates
# --------------------------------------------------------------------------- #

def _server_json(url: str, run: SourceRun, cfg: PrepConfig, *, retries: int = 4) -> dict:
    if run.metadata_requests >= cfg.budget.max_metadata_requests_per_source:
        raise httpio.HttpError(
            f"metadata request cap reached ({cfg.budget.max_metadata_requests_per_source})"
        )
    time.sleep(0.12)
    payload = httpio.get_json(url, timeout=cfg.budget.request_timeout_seconds, retries=retries)
    run.metadata_requests += 1
    return payload


def command_discover_indicsynth(
    *,
    languages: Sequence[str],
    strata: int = 24,
    page: int = 100,
) -> dict[str, Any]:
    """Scan IndicSynth rows and keep candidates whose source/target speakers are outside the closure."""
    cfg = v2_cfg()
    registry = load_registry()
    closure = speakers_by_language(registry)
    out_dir = V2_ROOT / "staging"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {}
    server = isynth.SERVER
    dataset_q = isynth.HF_DATASET.replace("/", "%2F")
    for language in languages:
        run = SourceRun(source_id="indicsynth")
        lang_closure = {str(s) for s in closure.get(language.lower(), set())}
        fresh: list[dict[str, Any]] = []
        seen: set[int] = set()
        total_rows = 0
        try:
            size = _server_json(f"{server}/size?dataset={dataset_q}", run, cfg)
            for entry in (size.get("size", {}).get("configs", []) or []):
                if str(entry.get("config")) == language:
                    total_rows = int(entry.get("num_rows", 0))
        except httpio.HttpError as error:
            summary[language] = {"status": "failed", "error": str(error)}
            continue
        rng = random.Random(f"{cfg.seed}:{language}:v2-scan")
        offsets: list[int] = []
        width = max(page, total_rows // max(strata, 1))
        for stratum in range(max(strata, 1)):
            start = stratum * width
            if start >= total_rows:
                break
            end = min(total_rows - page, start + width - page) if total_rows > page else start
            offset = rng.randrange(start, max(start, end) + 1) if end > start else start
            offsets.append(min(offset, max(0, total_rows - page)))
        errors: list[str] = []
        for offset in offsets:
            url = (f"{server}/rows?dataset={dataset_q}&config={language}"
                   f"&split=train&offset={offset}&length={page}")
            try:
                payload = _server_json(url, run, cfg)
            except httpio.HttpError as error:
                errors.append(str(error))
                continue
            for item in payload.get("rows", []) or []:
                index = int(item.get("row_idx", -1))
                if index < 0 or index in seen:
                    continue
                seen.add(index)
                facts = isynth._row_facts(item.get("row", {}) or {}, index)
                if not facts.get("audio_src"):
                    continue
                src, tgt = facts.get("source_speaker"), facts.get("target_speaker")
                if src is not None and str(src) in lang_closure:
                    continue
                if tgt is not None and str(tgt) in lang_closure:
                    continue
                facts.update({"language": language, "scan_offset": offset, "scan_stratum_page": True})
                fresh.append(facts)
        fresh.sort(key=lambda row: row["row_index"])
        write_jsonl_atomic(out_dir / f"indicsynth_fresh.{language}.jsonl", fresh)
        speakers_src = {row["source_speaker"] for row in fresh if row.get("source_speaker")}
        speakers_tgt = {row["target_speaker"] for row in fresh if row.get("target_speaker")}
        generators = Counter(str(row.get("generator")) for row in fresh)
        summary[language] = {
            "status": "ok" if not errors else "partial",
            "total_rows_reported": total_rows,
            "scanned_rows_seen": len(seen),
            "fresh_candidates": len(fresh),
            "fresh_source_speakers": len(speakers_src),
            "fresh_target_speakers": len(speakers_tgt),
            "generators": dict(sorted(generators.items())),
            "errors": errors[:5],
            "metadata_requests": run.metadata_requests,
        }
        print(json.dumps({k: v for k, v in summary[language].items() if k != "errors"}))
    write_json_atomic(out_dir / "indicsynth_discovery_summary.json", summary)
    return summary


# --------------------------------------------------------------------------- #
# Phase 3: deterministic test selection (component graph over fresh candidates)
# --------------------------------------------------------------------------- #

TEST_TARGET_PER_CLASS = 4
MIN_RECORDING_DURATION = 2.5
SEED_DEFAULT = 42


def _canonical(record_id: Any, speaker: Any, gender: Any) -> str | None:
    try:
        return f"{int(record_id)}-{int(speaker)}-{str(gender)}"
    except (TypeError, ValueError):
        return None


def _old_inventory_rows(language: str) -> list[dict[str, Any]]:
    path = V1_ROOT / "staging" / "joint" / f"kathbath_inventory.{language}.jsonl"
    return read_jsonl(path) if path.exists() else []


def _old_indicsynth_rows(language: str) -> list[dict[str, Any]]:
    path = V1_ROOT / "staging" / "joint" / f"indicsynth_candidates.{language}.jsonl"
    return read_jsonl(path) if path.exists() else []


def _closure_node(kind: str, language: str, value: Any) -> tuple[str, str, str]:
    return (kind, str(language).lower(), str(value))


def _closure_edges(language: str) -> list[tuple[tuple[str, str, str], tuple[str, str, str]]]:
    """Admissible identity/derivation edges for one language.

    A scanned candidate row claims person<->parent-recording links on each of its
    sides (source, target).  It does NOT claim that the two persons of one
    conversion are the same identity, so the two sides are not cross-unioned
    (verified against the reviewer's 67-window co-participation chains; those
    carry no shared person/recording/hash between tests and exposure — see
    ``reports/closure_audit.json``).  Components merge across rows only through
    shared exact keys (same speaker id, canonical reference, or record id).
    Kathbath inventory rows add recording<->speaker mappings, which also route
    around upstream label conflicts.
    """
    from scripts.gru_identity_audit import reference as parse_reference

    lang = language.lower()
    edges: list[tuple[tuple[str, str, str], tuple[str, str, str]]] = []

    def keys_for(speaker: Any, extra_speaker: Any, ref: Any) -> list[tuple[str, str, str]]:
        keys: list[tuple[str, str, str]] = []
        for value in (speaker, extra_speaker):
            if value is None or str(value).strip() in ("", "None"):
                continue
            node = _closure_node("s", lang, int(float(str(value))))
            if node not in keys:
                keys.append(node)
        parsed = parse_reference(ref)
        if parsed:
            for node in (_closure_node("r", lang, parsed["canonical"]),
                         _closure_node("j", lang, parsed["record"])):
                if node not in keys:
                    keys.append(node)
        return keys

    def link(keys: list[tuple[str, str, str]]) -> None:
        for other in keys[1:]:
            edges.append((keys[0], other))

    for row in _old_indicsynth_rows(language):
        link(keys_for(row.get("declared_source_speaker"), row.get("source_recording_speaker"),
                      row.get("source_reference")))
        link(keys_for(row.get("declared_target_speaker"), row.get("target_recording_speaker"),
                      row.get("target_reference")))
    for row in _load_fresh_state(language):
        link(keys_for(row.get("source_speaker"), None, row.get("source_reference")))
        link(keys_for(row.get("target_speaker"), None, row.get("target_reference")))
    for row in [*_old_inventory_rows(language), *_load_fresh_state(language)]:
        record = row.get("record_id")
        speaker = row.get("speaker")
        if record is None or speaker is None:
            continue
        j = _closure_node("j", lang, str(record))
        edges.append((j, _closure_node("s", lang, int(float(str(speaker))))))
        parsed = parse_reference(row.get("fname"))
        if parsed:
            edges.append((j, _closure_node("r", lang, parsed["canonical"])))
    return edges


def build_closure_state(registry: Mapping[str, Any], heldout: Mapping[str, Any]) -> dict[str, Any]:
    """Transitive test-exclusion closure over admissible relationships vs anchors.

    Anchors: every v1 exposure identity/recording (registry) and every verified
    upstream held-out record id.  Determined via union-find over the admissible
    edges of all core languages (language-scoped node tuples).
    """
    from .splitting import UnionFind

    finder = UnionFind()
    edge_count = 0
    for language in CORE_LANGUAGES:
        for a, b in _closure_edges(language):
            finder.union(a, b)
            edge_count += 1
    anchors: set[tuple[str, str, str]] = set()
    for _dataset, lang, speaker in registry.get("speaker_keys", []):
        anchors.add(_closure_node("s", str(lang), int(float(str(speaker)))))
    for _dataset, lang, canonical in registry.get("reference_keys", []):
        canonical = str(canonical)
        anchors.add(_closure_node("r", str(lang), canonical))
        anchors.add(_closure_node("j", str(lang), canonical.split("-")[0]))
    for language, entry in (heldout or {}).items():
        if not isinstance(entry, Mapping) or entry.get("status") != "verified":
            continue
        for record in entry.get("record_ids", []):
            anchors.add(_closure_node("j", str(language), str(record)))
    poisoned_roots = {finder.find(node) for node in anchors}
    memo: dict[tuple[str, str, str], bool] = {}

    def is_poisoned(node: tuple[str, str, str]) -> bool:
        hit = memo.get(node)
        if hit is None:
            hit = finder.find(node) in poisoned_roots
            memo[node] = hit
        return hit

    return {"finder": finder, "is_poisoned": is_poisoned, "anchors": len(anchors),
            "edges": edge_count, "poisoned_components": len(poisoned_roots)}


def _genuine_closure_nodes(language: str, row: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    lang = language.lower()
    nodes: list[tuple[str, str, str]] = []
    speaker = row.get("speaker")
    record = row.get("record_id")
    gender = str(row.get("gender") or "")
    if speaker is not None:
        nodes.append(_closure_node("s", lang, int(float(str(speaker)))))
    if record is not None:
        nodes.append(_closure_node("j", lang, str(record)))
        if speaker is not None and gender in ("m", "f"):
            nodes.append(_closure_node("r", lang,
                                       f"{int(float(str(record)))}-{int(float(str(speaker)))}-{gender}"))
    return nodes


def _synth_closure_nodes(language: str, source_speaker: Any, target_speaker: Any,
                         source_reference: Any, target_reference: Any) -> list[tuple[str, str, str]]:
    from scripts.gru_identity_audit import reference as parse_reference

    lang = language.lower()
    nodes: list[tuple[str, str, str]] = []
    for speaker in (source_speaker, target_speaker):
        if speaker is not None and str(speaker).strip() not in ("", "None"):
            nodes.append(_closure_node("s", lang, int(float(str(speaker)))))
    for ref in (source_reference, target_reference):
        parsed = parse_reference(ref)
        if parsed:
            nodes.append(_closure_node("r", lang, parsed["canonical"]))
            nodes.append(_closure_node("j", lang, parsed["record"]))
    return nodes


def _genuine_shortlist(
    rows: Sequence[Mapping[str, Any]],
    closure_speakers: set[str],
    closure_references: set[str],
    heldout_ids: set[str] | None,
    cached_record_ids: set[str],
    *,
    language: str,
    closure_poisoned: Callable[[tuple], bool] | None = None,
) -> tuple[list[dict[str, Any]], Counter, int]:
    short: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in rows:
        canonical = _canonical(row["record_id"], row["speaker"], row["gender"])
        if str(row["speaker"]) in closure_speakers:
            counts["speaker_in_prior_exposure"] += 1
            continue
        if canonical is None or canonical in closure_references or str(row["record_id"]) in closure_references:
            counts["reference_in_prior_exposure"] += 1
            continue
        if closure_poisoned is not None and any(
                closure_poisoned(node) for node in _genuine_closure_nodes(language, row)):
            counts["closure_component_in_prior_exposure"] += 1
            continue
        if heldout_ids is not None and str(row["record_id"]) in heldout_ids:
            counts["upstream_heldout_record"] += 1
            continue
        if float(row["duration"]) < MIN_RECORDING_DURATION:
            counts["recording_too_short"] += 1
            continue
        short.append({**row, "canonical": canonical, "cached_inventory_hit": str(row["record_id"]) in cached_record_ids})
    return short, counts, len(rows)


def estimate_group_bytes(ledger: DownloadLedger, language: str, shards: Sequence[str]) -> dict[str, dict[int, int]]:
    """Per-row-group compressed audio-column sizes (footer reads, cached, charged)."""
    import pyarrow.parquet as pq

    cache_path = V2_ROOT / "planning" / f"group_sizes.{language}.json"
    # JSON round-trips int row-group keys to strings; normalize on read so a
    # cached lookup behaves identically to a fresh fetch (int keys).
    raw_cached = read_json(cache_path, default={}) or {}
    cached: dict[str, dict[int, int]] = {
        str(shard): {int(group): int(size) for group, size in (sizes or {}).items()}
        for shard, sizes in raw_cached.items()
    }
    missing = [shard for shard in shards if shard not in cached]
    for shard in missing:
        counted = _pinned_open(shard)
        try:
            parquet = pq.ParquetFile(counted)
            metadata = parquet.metadata
            names = list(parquet.schema_arrow.names)
            column_index = names.index("audio_filepath") if "audio_filepath" in names else None
            sizes: dict[int, int] = {}
            for group in range(metadata.num_row_groups):
                row_group = metadata.row_group(group)
                if column_index is not None and column_index < row_group.num_columns:
                    sizes[group] = int(row_group.column(column_index).total_compressed_size)
            ledger.charge("kathbath", counted.bytes)
            cached[shard] = sizes
        finally:
            counted.close()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cached, indent=2) + "\n", encoding="utf-8")
    return {shard: cached[shard] for shard in shards if shard in cached}


def _choose_genuine(short: Sequence[dict[str, Any]], target: int, ledger: DownloadLedger,
                    language: str) -> list[dict[str, Any]]:
    """Deterministic, cost-aware genuine selection.

    Chooses row groups first (capacity for an even male/female split, then the
    smallest estimated audio-column bytes from parquet footers), then picks the
    best recording per chosen speaker inside those groups.
    """
    group_rows: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in short:
        group_rows[(row["shard"], int(row["row_group"]))].append(row)

    def capacity(group: tuple[str, int]) -> tuple[int, int]:
        rows = group_rows[group]
        males = {str(r["speaker"]) for r in rows if r["gender"] == "m"}
        females = {str(r["speaker"]) for r in rows if r["gender"] == "f"}
        score = min(len(males), max(1, target // 2)) + min(len(females), max(1, target // 2))
        return score, len(males) + len(females)

    candidates = sorted(group_rows, key=lambda g: (-capacity(g)[0], -capacity(g)[1], g[0], g[1]))
    probe = candidates[:8]
    sizes = estimate_group_bytes(ledger, language, sorted({s for s, _ in probe}))

    def estimated_bytes(group: tuple[str, int]) -> int:
        return int(sizes.get(group[0], {}).get(group[1], 0)) or 10 ** 12

    ordered = sorted(candidates, key=lambda g: (-capacity(g)[0], estimated_bytes(g), g[0], g[1]))

    def speaker_order(speaker: str) -> tuple[int, int]:
        rows = group_rows_of[speaker]
        return (all(r["cached_inventory_hit"] for r in rows), int(speaker))

    chosen: list[str] = []
    chosen_rows: list[dict[str, Any]] = []
    per_gender = {"m": 0, "f": 0}
    target_per_gender = max(1, target // 2)
    group_rows_of: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rows in group_rows.values():
        for row in rows:
            group_rows_of[str(row["speaker"])].append(row)

    for group in ordered:
        if len(chosen) >= target:
            break
        rows = group_rows[group]
        for gender in ("m", "f"):
            need = target_per_gender - per_gender[gender]
            if need <= 0 or len(chosen) >= target:
                continue
            pool = sorted({str(r["speaker"]) for r in rows if r["gender"] == gender and str(r["speaker"]) not in chosen},
                          key=speaker_order)
            for speaker in pool[:need]:
                in_group = [r for r in rows if str(r["speaker"]) == speaker]
                pick = max(in_group, key=lambda r: (float(r["duration"]), -int(r["row_index"])))
                chosen.append(speaker)
                chosen_rows.append(pick)
                per_gender[gender] += 1
    if len(chosen) < target:
        for group in ordered:
            for row in sorted(group_rows[group], key=lambda r: (str(r["speaker"]), -float(r["duration"]))):
                if len(chosen) >= target:
                    break
                if str(row["speaker"]) not in chosen:
                    chosen.append(str(row["speaker"]))
                    chosen_rows.append(row)
    return chosen_rows


def _isynth_shortlist(
    rows: Sequence[Mapping[str, Any]],
    closure_speakers: set[str],
    closure_references: set[str],
    heldout_ids: set[str] | None,
    verified_record_ids: set[str],
    *,
    language: str,
    closure_poisoned: Callable[[tuple], bool] | None = None,
) -> tuple[list[dict[str, Any]], Counter, int]:
    from scripts.gru_identity_audit import reference as parse_reference

    short: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in rows:
        src, tgt = row.get("source_speaker"), row.get("target_speaker")
        if src is not None and str(src) in closure_speakers:
            counts["source_speaker_in_prior_exposure"] += 1
            continue
        if tgt is not None and str(tgt) in closure_speakers:
            counts["target_speaker_in_prior_exposure"] += 1
            continue
        src_ref = parse_reference(row.get("source_reference"))
        tgt_ref = parse_reference(row.get("target_reference"))
        if tgt_ref is None:
            counts["missing_target_reference"] += 1
            continue
        source_kind = ("conversion_source" if row.get("generator") == "freevc24" else "declared_source") \
            if src_ref is not None else "tts_target_only"
        if source_kind != "tts_target_only" and src_ref is None:
            counts["missing_source_reference"] += 1
            continue
        bad_ref = False
        for ref in (src_ref, tgt_ref):
            if ref is None:
                continue
            if ref["canonical"] in closure_references or ref["record"] in closure_references:
                bad_ref = True
        if bad_ref:
            counts["reference_in_prior_exposure"] += 1
            continue
        if closure_poisoned is not None and any(
                closure_poisoned(node) for node in _synth_closure_nodes(
                    language, row.get("source_speaker"), row.get("target_speaker"),
                    row.get("source_reference"), row.get("target_reference"))):
            counts["closure_component_in_prior_exposure"] += 1
            continue
        if heldout_ids is not None:
            ids = [r["record"] for r in (src_ref, tgt_ref) if r is not None]
            if any(rid in heldout_ids for rid in ids):
                counts["upstream_heldout_parent"] += 1
                continue
        target_verified = tgt_ref["record"] in verified_record_ids
        source_verified = (src_ref["record"] in verified_record_ids) if src_ref is not None else None
        if not target_verified:
            counts["target_parent_unverified"] += 1
            continue
        if source_kind != "tts_target_only" and not source_verified:
            counts["source_parent_unverified"] += 1
            continue
        short.append({
            **row,
            "source_kind": source_kind,
            "source_reference_canonical": src_ref["canonical"] if src_ref else None,
            "target_reference_canonical": tgt_ref["canonical"],
            "source_record": src_ref["record"] if src_ref else None,
            "target_record": tgt_ref["record"],
            "target_verified": target_verified,
            "source_verified": source_verified,
        })
    return short, counts, len(rows)


def _choose_isynth(short: Sequence[dict[str, Any]], target: int, rng: random.Random) -> list[dict[str, Any]]:
    usable = [row for row in short if row.get("audio_src")]
    if not usable:
        return []
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in usable:
        by_generator[str(row.get("generator") or "unknown")].append(row)
    ordered: list[dict[str, Any]] = []
    blocks: list[list[dict[str, Any]]] = []
    for generator in sorted(by_generator):
        tiers: dict[tuple[bool, bool], list[dict[str, Any]]] = defaultdict(list)
        for row in by_generator[generator]:
            tiers[(not bool(row["target_verified"]), row.get("transcript") is None)].append(row)
        block: list[dict[str, Any]] = []
        for tier_key in sorted(tiers):
            group = tiers[tier_key]
            rng.shuffle(group)  # seeded shuffle within each preference tier
            block.extend(group)
        blocks.append(block)
    # Interleave generators so the first pass draws one row per generator before
    # filling from any single generator again (round-robin diversity).
    for position in range(max((len(b) for b in blocks), default=0)):
        for block in blocks:
            if position < len(block):
                ordered.append(block[position])
    selected: list[dict[str, Any]] = []
    used_targets: set[str] = set()
    for row in ordered:
        if len(selected) >= target:
            break
        target_speaker = str(row.get("target_speaker"))
        if target_speaker in used_targets and any(str(r.get("target_speaker")) not in used_targets for r in ordered):
            continue
        selected.append(row)
        used_targets.add(target_speaker)
    if len(selected) < target:
        for row in ordered:
            if len(selected) >= target:
                break
            if row not in selected:
                selected.append(row)
    return selected


def _component_summary(language: str, genuine: Sequence[Mapping[str, Any]], synthetic: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Deterministic union-find over selected items' relationship keys.

    The validator re-derives these relationships independently from provenance
    fields; this summary is planning evidence, not the audit.
    """
    from .splitting import UnionFind

    finder = UnionFind()
    item_keys: dict[str, list[tuple[str, str, str]]] = {}

    for row in genuine:
        wid = f"kathbath-{LANGUAGE_SLUGS[language]}-{row['record_id']}-{row['speaker']}-{row['gender']}"
        item_keys[wid] = [("s", language, str(row["speaker"])), ("r", language, str(row["canonical"])),
                          ("j", language, str(row["record_id"]))]
    for row in synthetic:
        wid = f"indicsynth-{LANGUAGE_SLUGS[language]}-{int(row['row_index']):06d}"
        keys: list[tuple[str, str, str]] = []
        if row.get("source_speaker") is not None:
            keys.append(("s", language, str(row["source_speaker"])))
        if row.get("target_speaker") is not None:
            keys.append(("s", language, str(row["target_speaker"])))
        if row.get("source_reference_canonical"):
            keys.append(("r", language, str(row["source_reference_canonical"])))
            keys.append(("j", language, str(row["source_record"])))
        keys.append(("r", language, str(row["target_reference_canonical"])))
        keys.append(("j", language, str(row["target_record"])))
        item_keys[wid] = keys

    for keys in item_keys.values():
        finder.union(keys[0], keys[0])
        for key in keys[1:]:
            finder.union(keys[0], key)

    groups: dict[Any, list[str]] = defaultdict(list)
    for wid, keys in item_keys.items():
        groups[finder.find(keys[0])].append(wid)
    components = [
        {"root": str(root), "members": sorted(members)}
        for root, members in sorted(groups.items(), key=lambda kv: str(kv[0]))
    ]
    return {
        "language": language,
        "n_components": len(components),
        "components": components,
        "item_keys": {wid: [list(k) for k in keys] for wid, keys in sorted(item_keys.items())},
    }


def _candidate_id(language: str, kind: str, row: Mapping[str, Any]) -> str:
    slug = LANGUAGE_SLUGS[language]
    if kind == "genuine":
        return f"kathbath-{slug}-{row['record_id']}-{row['speaker']}-{row['gender']}"
    return f"indicsynth-{slug}-{int(row['row_index']):06d}"


def load_quarantine() -> dict[str, str]:
    return read_json(V2_ROOT / "planning" / "quarantine.json", default={}) or {}


def _quarantine_entries(reasons: Mapping[str, str]) -> None:
    """Append candidate failures to the quarantine file (id -> reason)."""
    path = V2_ROOT / "planning" / "quarantine.json"
    current = read_json(path, default={}) or {}
    current.update(reasons)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(current.items())), indent=2) + "\n", encoding="utf-8")


def _plan(languages: Sequence[str], target: int, seed: int) -> dict[str, Any]:
    registry = load_registry()
    closure_speakers = speakers_by_language(registry)
    closure_references = references_by_language(registry)
    heldout = read_json(V2_ROOT / "staging" / "kathbath_heldout.json", default={}) or {}
    closure = build_closure_state(registry, heldout)
    plan: dict[str, Any] = {
        "schema": "voxsentinel.test_plan.v2", "seed": seed, "target_per_class": target,
        "languages": {}, "input_hashes": {},
        "closure": {"anchors": closure["anchors"], "edges": closure["edges"],
                    "poisoned_components": closure["poisoned_components"]},
    }
    for language in languages:
        slug = LANGUAGE_SLUGS[language]
        cs = {str(s) for s in closure_speakers.get(language.lower(), set())}
        cr = {str(s) for s in closure_references.get(language.lower(), set())}
        held = heldout.get(language) or {}
        heldout_ids = set(str(x) for x in held.get("record_ids", [])) if held.get("status") == "verified" else None

        fresh_path = V2_ROOT / "staging" / f"kathbath_fresh.{language}.jsonl"
        isynth_path = V2_ROOT / "staging" / f"indicsynth_fresh.{language}.jsonl"
        fresh_rows = read_jsonl(fresh_path)
        isynth_rows = read_jsonl(isynth_path)
        plan["input_hashes"][f"kathbath_fresh.{language}"] = sha256_file(fresh_path) if fresh_path.exists() else None
        plan["input_hashes"][f"indicsynth_fresh.{language}"] = sha256_file(isynth_path) if isynth_path.exists() else None

        old_rows = _old_inventory_rows(language)
        cached_ids = {str(r["record_id"]) for r in old_rows}
        verified_records = cached_ids | {str(r["record_id"]) for r in fresh_rows}

        entry: dict[str, Any] = {
            "status": "ok",
            "closure_speakers": len(cs), "closure_references": len(cr),
            "heldout_status": held.get("status"), "heldout_records": len(heldout_ids or ()),
        }
        if heldout_ids is None:
            entry["status"] = "blocked_heldout_unverified"
            plan["languages"][language] = entry
            continue

        genuine_short, genuine_counts, genuine_raw = _genuine_shortlist(
            fresh_rows, cs, cr, heldout_ids, cached_ids,
            language=language, closure_poisoned=closure["is_poisoned"])
        synth_short, synth_counts, synth_raw = _isynth_shortlist(
            isynth_rows, cs, cr, heldout_ids, verified_records,
            language=language, closure_poisoned=closure["is_poisoned"])

        quarantine = load_quarantine()
        if quarantine:
            kept = []
            for row in genuine_short:
                if _candidate_id(language, "genuine", row) in quarantine:
                    genuine_counts["quarantined_previous_failure"] += 1
                else:
                    kept.append(row)
            genuine_short = kept
            kept = []
            for row in synth_short:
                if _candidate_id(language, "synthetic", row) in quarantine:
                    synth_counts["quarantined_previous_failure"] += 1
                else:
                    kept.append(row)
            synth_short = kept

        rng = random.Random(f"{seed}:{language}:selection")
        genuine_sel = _choose_genuine(genuine_short, target, v2_ledger(v2_cfg()), language)
        synth_sel = _choose_isynth(synth_short, target, rng)

        components = _component_summary(language, genuine_sel, synth_sel)

        def _item(row: Mapping[str, Any], kind: str) -> dict[str, Any]:
            return {"language": language, "kind": kind,
                    "window_id": _candidate_id(language, kind, row), "row": dict(row)}

        selected_items = [_item(r, "genuine") for r in genuine_sel] + [_item(r, "synthetic") for r in synth_sel]
        groups = Counter((r["shard"], r["row_group"]) for r in genuine_sel)

        entry.update({
            "genuine": {
                "raw_fresh_rows": genuine_raw, "shortlist": len(genuine_short), "selected": len(genuine_sel),
                "excluded_by_reason": dict(genuine_counts),
                "selected_speakers": [r["speaker"] for r in genuine_sel],
                "selected_genders": Counter(r["gender"] for r in genuine_sel),
                "row_groups_needed": [list(g) for g in sorted(groups)],
            },
            "synthetic": {
                "raw_fresh_candidates": synth_raw, "shortlist": len(synth_short), "selected": len(synth_sel),
                "excluded_by_reason": dict(synth_counts),
                "selected_generators": dict(Counter(str(r.get("generator")) for r in synth_sel)),
                "selected_targets": [str(r.get("target_speaker")) for r in synth_sel],
            },
            "components": components,
            "selected": selected_items,
        })
        plan["languages"][language] = entry

        # per-language accounting: every shortlist candidate disposition
        accounting_dir = V2_ROOT / "planning"
        selected_ids = {item["window_id"] for item in selected_items}
        lines = []
        for kind, rows_ in (("genuine", genuine_short), ("synthetic", synth_short)):
            for row in rows_:
                wid = _candidate_id(language, kind, row)
                lines.append({
                    "candidate_id": wid, "language": language, "kind": kind,
                    "disposition": "selected" if wid in selected_ids else "eligible_unused",
                    "reason": None if wid in selected_ids else "not_needed_for_target_or_diversity",
                    "component_hint": row.get("canonical") or row.get("target_reference_canonical"),
                })
        write_jsonl_atomic(accounting_dir / f"accounting.{language}.jsonl", lines)
    return plan


def command_plan(*, languages: Sequence[str], target_per_class: int = TEST_TARGET_PER_CLASS,
                 seed: int = SEED_DEFAULT) -> dict[str, Any]:
    plan = _plan(languages, target_per_class, seed)
    planning = V2_ROOT / "planning"
    planning.mkdir(parents=True, exist_ok=True)
    (planning / "test_plan.json").write_text(json.dumps(plan, indent=2, default=str) + "\n", encoding="utf-8")
    selection_rows = []
    for language, entry in plan["languages"].items():
        selection_rows.extend(entry.get("selected", []))
    write_jsonl_atomic(planning / "test_selection.v2.jsonl", selection_rows)
    summary = {
        language: {
            "status": entry["status"],
            "genuine_selected": entry.get("genuine", {}).get("selected"),
            "synthetic_selected": entry.get("synthetic", {}).get("selected"),
            "genders": dict(entry.get("genuine", {}).get("selected_genders", {})),
            "row_groups": len(entry.get("genuine", {}).get("row_groups_needed", [])),
        }
        for language, entry in plan["languages"].items()
    }
    (planning / "test_plan_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))
    return plan


# --------------------------------------------------------------------------- #
# Phase 4: materialization (revision-pinned fetch + prep-2 windows)
# --------------------------------------------------------------------------- #

def _language_for_shard(shard: str) -> str | None:
    slug = str(shard).split("/")[0]
    for language, candidate in LANGUAGE_SLUGS.items():
        if candidate == slug:
            return language
    return None


def _cached_group_estimate(language: str | None, shard: str, groups: Sequence[int],
                           cache_root: Path | None = None) -> int | None:
    """Pre-read byte estimate for the needed row groups from the cached footer sizes."""
    if language is None:
        return None
    cache = read_json((cache_root or V2_ROOT) / "planning" / f"group_sizes.{language}.json", default={}) or {}
    sizes = cache.get(shard)
    if not sizes:
        return None
    total = 0
    for group in groups:
        size = sizes.get(str(group), sizes.get(group))
        if size is None:
            return None
        total += int(size)
    return total + 65536  # footer + request overhead slack


def fetch_kathbath_audio_pinned(
    ledger: DownloadLedger,
    rows: Sequence[Mapping[str, Any]],
    *,
    byte_cap: int,
    workers: int = 4,
    cache_root: Path | None = None,
) -> tuple[dict[str, bytes], dict[str, int], int, list[str], list[dict[str, Any]]]:
    """Read selected shards serially with pre-charged, no-read-ahead I/O.

    Charges are conservative requested payload bytes, retained on exceptions.
    Run one acquisition job at a time against the shared ledger. ``workers`` is
    retained for caller compatibility; audio shard reads are deliberately serial.
    ``cache_root`` selects the planning cache used for pre-read estimates
    (defaults to the v2 tree; the v3 coverage pipeline passes its own root).
    """
    import pyarrow.parquet as pq

    by_shard: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_shard[str(row["shard"])].append(row)
    payloads: dict[str, bytes] = {}
    bytes_by_shard: dict[str, int] = {}
    groups_read = 0
    total = 0
    skipped: list[str] = []
    skip_events: list[dict[str, Any]] = []
    estimated_total = 0
    to_fetch: dict[str, list[Mapping[str, Any]]] = {}
    for shard, wanted in sorted(by_shard.items()):
        groups_needed = sorted({int(row["row_group"]) for row in wanted})
        estimate = _cached_group_estimate(_language_for_shard(shard), shard, groups_needed, cache_root)
        if estimate is not None and (estimated_total + estimate > byte_cap or not ledger.affordable(estimate)):
            skipped.append(shard)
            skip_events.append({"shard": shard, "reason": "pre_read_cap_skip",
                                "estimated_bytes": estimate, "charged_bytes": 0})
            continue
        if estimate is not None:
            estimated_total += estimate
        to_fetch[shard] = wanted

    allowance = {"remaining": byte_cap}

    def _fetch(shard: str, wanted: list[Mapping[str, Any]]) -> tuple[dict[str, bytes], int, int]:
        from huggingface_hub import HfFileSystem
        from .bounded_reader import SeekableRangeReader

        fs = HfFileSystem()
        counted = SeekableRangeReader(
            fs, f"datasets/{HF_DATASET}@{KATHBATH_REVISION}/{shard}", ledger, "kathbath", allowance)
        try:
            parquet = pq.ParquetFile(counted)
            offsets: list[int] = []
            running = 0
            for group in range(parquet.metadata.num_row_groups):
                offsets.append(running)
                running += parquet.metadata.row_group(group).num_rows
            by_group: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for row in wanted:
                by_group[int(row["row_group"])].append(row)
            out: dict[str, bytes] = {}
            groups = 0
            for group in sorted(by_group):
                table = parquet.read_row_group(group, columns=["audio_filepath"]).to_pydict()
                column = table["audio_filepath"]
                offset = offsets[group] if group < len(offsets) else 0
                for row in by_group[group]:
                    local = int(row["row_index"]) - offset
                    if not 0 <= local < len(column):
                        raise RuntimeError(f"row {row['fname']} outside row group {group}")
                    cell = column[local]
                    data = cell.get("bytes") if isinstance(cell, Mapping) else None
                    if data:
                        out[str(row["fname"])] = bytes(data)
                groups += 1
            return out, counted.bytes, groups
        finally:
            counted.close()

    for shard, wanted in to_fetch.items():
        before = allowance["remaining"]
        try:
            got, bytes_read, groups = _fetch(shard, wanted)
        except Exception as error:
            charged = before - allowance["remaining"]
            skipped.append(shard)
            skip_events.append({"shard": shard, "reason": "read_failed_or_capped",
                                "charged_bytes": charged, "error_type": type(error).__name__,
                                "error": str(error)[:300]})
            continue
        total += bytes_read
        groups_read += groups
        bytes_by_shard[shard] = bytes_read
        payloads.update(got)
    return payloads, bytes_by_shard, groups_read, skipped, skip_events


def command_materialize(*, languages: Sequence[str] | None = None, byte_cap_per_language: int = 64_000_000,
                        seed: int = SEED_DEFAULT) -> dict[str, Any]:
    from .materialize import HashRegistry, materialize_window, store_original_bytes
    from .records import build_recording_record, build_window_record
    from . import joint_core  # `_fetch_synth_asset` is joint_core's charged asset fetcher

    plan = read_json(V2_ROOT / "planning" / "test_plan.json", default=None)
    if not plan:
        raise FileNotFoundError("run `prepare_dataset_v2 plan` first")
    v2 = v2_cfg()
    ledger = v2_ledger(v2)
    windows: list[dict[str, Any]] = []
    recordings: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    quarantine_map: dict[str, str] = {}
    run = SourceRun(source_id="indic_synth_v2")

    # duplicate protection seeded from every frozen v1 record
    v1_windows = read_jsonl(V1_ROOT / "manifests" / "windows.core_train.jsonl") + \
        read_jsonl(V1_ROOT / "manifests" / "windows.core_val.jsonl")
    registry = HashRegistry.from_records(v1_windows)

    release_kb = v2.release("kathbath")
    release_is = v2.release("indicsynth")
    pre_version = str(v2.preprocessing.get("config_version"))
    languages = languages or [lang for lang in plan["languages"] if plan["languages"][lang]["status"] == "ok"]
    summary: dict[str, Any] = {}
    for language in languages:
        entry = plan["languages"][language]
        slug = LANGUAGE_SLUGS[language]
        if entry["status"] != "ok":
            summary[language] = {"status": entry["status"]}
            continue
        genuine_rows = [item["row"] for item in entry["selected"] if item["kind"] == "genuine"]
        synth_rows = [item["row"] for item in entry["selected"] if item["kind"] == "synthetic"]
        added_windows, added_recordings, exc = [], [], []
        language_inventory_records = {str(r["record_id"]) for r in _old_inventory_rows(language)}
        language_fresh_records = {str(r["record_id"]) for r in _load_fresh_state(language)}

        def _parent_evidence(reference: Any, record: Any, *, tts: bool = False) -> dict[str, Any]:
            """Actual lookup evidence for a parent recording, not a restated reference."""
            if tts:
                return {"method": "not_applicable_tts", "reference": None, "verified": True}
            found = []
            if record is not None:
                if str(record) in language_inventory_records:
                    found.append("v1_scan_inventory")
                if str(record) in language_fresh_records:
                    found.append("v2_shard_scan")
            return {"reference": reference, "record": (str(record) if record is not None else None),
                    "method": "record_id_lookup_in_scanned_kathbath_inventories",
                    "found_in": sorted(found), "verified": bool(found)}

        # ---- Kathbath genuine
        raw_dir = v2.raw_dir / "kathbath" / slug
        to_fetch = [row for row in genuine_rows if not (raw_dir / str(row["fname"])).exists()]
        bytes_used, groups_read, skipped_shards = 0, 0, []
        if to_fetch:
            try:
                payloads, _, groups_read, skipped_shards, skip_events = fetch_kathbath_audio_pinned(
                    ledger, to_fetch, byte_cap=byte_cap_per_language)
            except BudgetExceeded as error:
                payloads = {}
                skip_events = []
                exc.append({"dataset_id": "kathbath", "language": language, "reason": f"budget: {error}"})
            for fname, payload in payloads.items():
                store_original_bytes(raw_dir / fname, payload)
                bytes_used += len(payload)
            for event in skip_events:
                exc.append({"dataset_id": "kathbath", "language": language,
                            "reason": f"fetch_skip[{event.get('reason')}]",
                            "shard": event.get("shard"), "charged_bytes": event.get("charged_bytes"),
                            "estimated_bytes": event.get("estimated_bytes")})
        for row in genuine_rows:
            window_id = _candidate_id(language, "genuine", row)
            raw_path = raw_dir / str(row["fname"])
            if not raw_path.exists():
                exc.append({"dataset_id": "kathbath", "language": language, "candidate": window_id,
                            "reason": "raw_audio_missing_after_fetch"})
                quarantine_map[window_id] = "raw_audio_missing_after_fetch"
                continue
            prepared_path = v2.prepared_dir / "kathbath" / slug / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=v2.window)
            except Exception as error:  # decode / window failures are recorded, never papered over
                exc.append({"dataset_id": "kathbath", "language": language, "candidate": window_id,
                            "reason": f"decode_or_window_failed: {type(error).__name__}: {error}"})
                quarantine_map[window_id] = f"decode_or_window_failed: {type(error).__name__}"
                continue
            duplicate = registry.check(materialized)
            if duplicate:
                exc.append({"dataset_id": "kathbath", "language": language, "candidate": window_id,
                            "reason": duplicate})
                quarantine_map[window_id] = duplicate
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)
            parent_refs = {
                "dataset_revision": KATHBATH_REVISION,
                "shard": row["shard"], "row_group": int(row["row_group"]), "row_index": int(row["row_index"]),
                "upstream_duration_seconds": float(row["duration"]), "upstream_split": "train",
            }
            speaker_ids = {"speaker": str(row["speaker"]), "gender": str(row["gender"])}
            original = {**materialized.decoded.original_facts(), "sha256": materialized.original_sha256}
            prepared = materialized.prepared_facts(v2.dataset_root)
            window_facts = materialized.window_facts(original_rate=materialized.decoded.original_rate)
            added_recordings.append(build_recording_record(
                recording_id=window_id, dataset_id="kathbath", source_url=str(release_kb.get("source_url")),
                source_file=f"{slug}/{row['fname']}", original_split="train", label=0,
                spoken_language=language, native_language=None, speaker_ids=speaker_ids,
                generator=None, generator_version=None, parent_refs=parent_refs,
                original_audio=original, decoded=materialized.decoded_facts(),
                license_note=str(release_kb.get("license")), notes="v2 test selection; previously unscanned shard",
            ))
            added_windows.append(build_window_record(
                window_id=window_id, recording_id=window_id, dataset_id="kathbath",
                source_url=str(release_kb.get("source_url")), source_file=f"{slug}/{row['fname']}",
                original_split="train", pool="core", split="test", label=0,
                label_source="official Kathbath corpus (genuine speech)", spoken_language=language,
                native_language=None, speaker_ids=speaker_ids, generator=None, generator_version=None,
                parent_refs=parent_refs, original_audio=original, prepared_audio=prepared,
                window=window_facts, preprocessing_version=pre_version,
                license_note=str(release_kb.get("license")), access_status="public_with_terms",
                status="audio_ready", notes="v2 test selection; previously unscanned shard; gender from upstream filename metadata",
            ))

        # ---- IndicSynth synthetic
        for row in synth_rows:
            index = int(row["row_index"])
            window_id = _candidate_id(language, "synthetic", row)
            raw_path = v2.raw_dir / "indicsynth" / slug / f"{index:06d}.wav"
            if not raw_path.exists():
                try:
                    payload = joint_core._fetch_synth_asset(v2, ledger, run, dict(row), language, index)
                except BudgetExceeded as error:
                    exc.append({"dataset_id": "indicsynth", "language": language, "candidate": window_id,
                                "reason": f"budget: {error}"})
                    continue
                if payload is None:
                    exc.append({"dataset_id": "indicsynth", "language": language, "candidate": window_id,
                                "reason": "audio_asset_fetch_failed"})
                    quarantine_map[window_id] = "audio_asset_fetch_failed"
                    continue
                store_original_bytes(raw_path, payload)
            prepared_path = v2.prepared_dir / "indicsynth" / slug / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=v2.window)
            except Exception as error:
                exc.append({"dataset_id": "indicsynth", "language": language, "candidate": window_id,
                            "reason": f"decode_or_window_failed: {type(error).__name__}: {error}"})
                quarantine_map[window_id] = f"decode_or_window_failed: {type(error).__name__}"
                continue
            duplicate = registry.check(materialized)
            if duplicate:
                exc.append({"dataset_id": "indicsynth", "language": language, "candidate": window_id,
                            "reason": duplicate})
                quarantine_map[window_id] = duplicate
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)
            speaker_ids = {"source": row.get("source_speaker"), "target": row.get("target_speaker")}
            evidence = {
                "source": _parent_evidence(row.get("source_reference"), row.get("source_record"),
                                           tts=row.get("source_kind") == "tts_target_only"),
                "target": _parent_evidence(row.get("target_reference"), row.get("target_record")),
            }
            parent_refs = {
                "dataset_revision": INDICSYNTH_REVISION,
                "source_reference": row.get("source_reference"), "target_reference": row.get("target_reference"),
                "transcript": row.get("transcript"), "source_kind": row["source_kind"],
                "source_parent_verification": (
                    "not_applicable_tts" if row["source_kind"] == "tts_target_only"
                    else ("verified_train" if row.get("source_verified") else "parsed_only")),
                "target_parent_verification": "verified_train" if row.get("target_verified") else "parsed_only",
                "reference_evidence": evidence,
            }
            original = {**materialized.decoded.original_facts(), "sha256": materialized.original_sha256}
            prepared = materialized.prepared_facts(v2.dataset_root)
            window_facts = materialized.window_facts(original_rate=materialized.decoded.original_rate)
            added_recordings.append(build_recording_record(
                recording_id=window_id, dataset_id="indicsynth", source_url=str(release_is.get("source_url")),
                source_file=f"{language}/train/row-{index}", original_split="train", label=1,
                spoken_language=language, native_language=None, speaker_ids=speaker_ids,
                generator=row.get("generator"), generator_version=None, parent_refs=parent_refs,
                original_audio=original, decoded=materialized.decoded_facts(),
                license_note=str(release_is.get("license")), notes="v2 test selection; relationship closure checked",
            ))
            added_windows.append(build_window_record(
                window_id=window_id, recording_id=window_id, dataset_id="indicsynth",
                source_url=str(release_is.get("source_url")), source_file=f"{language}/train/row-{index}",
                original_split="train", pool="core", split="test", label=1,
                label_source="generated_synthetic (IndicSynth generator field)", spoken_language=language,
                native_language=None, speaker_ids=speaker_ids, generator=row.get("generator"),
                generator_version=None, parent_refs=parent_refs, original_audio=original,
                prepared_audio=prepared, window=window_facts, preprocessing_version=pre_version,
                license_note=str(release_is.get("license")), access_status="public", status="audio_ready",
                notes="v2 test selection; relationship closure checked",
            ))
        windows.extend(added_windows)
        recordings.extend(added_recordings)
        exclusions.extend(exc)
        summary[language] = {
            "genuine_windows": sum(1 for w in added_windows if w["dataset_id"] == "kathbath"),
            "synthetic_windows": sum(1 for w in added_windows if w["dataset_id"] == "indicsynth"),
            "kathbath_fetch_bytes": bytes_used, "kathbath_row_groups_read": groups_read,
            "kathbath_skipped_shards": skipped_shards,
            "exclusions": len(exc),
        }
        print(json.dumps({"language": language, **summary[language]}))

    write_jsonl_atomic(v2.manifests_dir / "windows.test.jsonl", windows)
    write_jsonl_atomic(v2.manifests_dir / "recordings.test.jsonl", recordings)
    write_jsonl_atomic(v2.manifests_dir / "windows.test.excluded.jsonl", exclusions)
    if quarantine_map:
        _quarantine_entries(quarantine_map)
        print(json.dumps({"quarantined": quarantine_map}, indent=2))
    (V2_ROOT / "staging" / "materialization_summary.json").write_text(
        json.dumps({"languages": summary, "ledger": ledger_facts(ledger)}, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"total_windows": len(windows), "ledger": ledger_facts(ledger)}, indent=2))
    return {"windows": len(windows), "summary": summary}


def command_finalize() -> dict[str, Any]:
    """Copy the frozen v1 train/dev manifests into the v2 tree and write dataset-version metadata."""
    import shutil

    v2 = v2_cfg()
    v1_train = V1_ROOT / "manifests" / "windows.core_train.jsonl"
    v1_val = V1_ROOT / "manifests" / "windows.core_val.jsonl"
    train_copy = v2.manifests_dir / "windows.train.jsonl"
    dev_copy = v2.manifests_dir / "windows.dev.jsonl"
    shutil.copyfile(v1_train, train_copy)
    shutil.copyfile(v1_val, dev_copy)
    ledger = v2_ledger(v2)

    def _hash(path: Path) -> str | None:
        return sha256_file(path) if path.exists() else None

    version = {
        "schema": "voxsentinel.dataset_version.v2",
        "seed": SEED_DEFAULT,
        "split_policy": "speaker_recording_disjoint_v2: v1 train/dev preserved; test excludes exposed identities/recordings across all roles; unused conversion pair chains are not exclusion edges",
        "strict_conversion_family_compliant": False,
        "preprocessing_version": str(v2.preprocessing.get("config_version")),
        "revisions": {"kathbath": KATHBATH_REVISION, "indicsynth": INDICSYNTH_REVISION},
        "manifests": {
            "train": {"path": "artifacts/datasets-v2/manifests/windows.train.jsonl", "sha256": _hash(train_copy),
                       "copy_of": "artifacts/datasets/manifests/windows.core_train.jsonl",
                       "source_sha256": _hash(v1_train),
                       "path_base": "artifacts/datasets", "split_field": "train"},
            "dev": {"path": "artifacts/datasets-v2/manifests/windows.dev.jsonl", "sha256": _hash(dev_copy),
                     "copy_of": "artifacts/datasets/manifests/windows.core_val.jsonl",
                     "source_sha256": _hash(v1_val),
                     "path_base": "artifacts/datasets", "split_field": "val"},
            "test": {"path": "artifacts/datasets-v2/manifests/windows.test.jsonl", "sha256": _hash(v2.manifests_dir / "windows.test.jsonl"),
                      "path_base": "artifacts/datasets-v2", "split_field": "test"},
            "test_excluded": {"path": "artifacts/datasets-v2/manifests/windows.test.excluded.jsonl",
                               "sha256": _hash(v2.manifests_dir / "windows.test.excluded.jsonl")},
        },
        "supporting": {
            "exposure_registry": "artifacts/datasets-v2/planning/exposure_registry.v2.json",
            "test_plan": "artifacts/datasets-v2/planning/test_plan.json",
            "test_selection": "artifacts/datasets-v2/planning/test_selection.v2.jsonl",
            "config": {"path": "configs/datasets.yaml", "sha256": _hash(Path("configs/datasets.yaml"))},
            "plan_config": {"path": "configs/datasets_v2_plan.yaml", "sha256": _hash(Path("configs/datasets_v2_plan.yaml"))},
        },
        "ledger": ledger_facts(ledger),
    }
    (v2.manifests_dir / "dataset-version.v2.json").write_text(json.dumps(version, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v["sha256"] for k, v in version["manifests"].items()}, indent=2))
    return version


def command_replay(*, languages: Sequence[str], target_per_class: int = TEST_TARGET_PER_CLASS,
                   seed: int = SEED_DEFAULT) -> dict[str, Any]:
    """Recompute the selection from frozen inputs and compare with the stored plan."""
    stored = read_json(V2_ROOT / "planning" / "test_plan.json", default=None)
    if not stored:
        raise FileNotFoundError("no stored test plan to compare against")
    recomputed = _plan(languages, target_per_class, seed)

    def _strip(plan: Mapping[str, Any]) -> dict[str, Any]:
        return {lang: [
            {"window_id": item["window_id"], "kind": item["kind"]}
            for item in entry.get("selected", [])
        ] for lang, entry in plan["languages"].items()}

    a, b = _strip(stored), _strip(recomputed)
    mismatches = {lang: {"stored": a.get(lang), "recomputed": b.get(lang)}
                  for lang in set(a) | set(b) if a.get(lang) != b.get(lang)}
    input_match = {
        key: stored.get("input_hashes", {}).get(key) == recomputed.get("input_hashes", {}).get(key)
        for key in set(stored.get("input_hashes", {})) | set(recomputed.get("input_hashes", {}))
    }
    report = {
        "schema": "voxsentinel.replay_check.v1",
        "selection_match": not mismatches,
        "input_hash_match": all(input_match.values()),
        "mismatch_languages": sorted(mismatches),
        "input_hash_detail": input_match,
        "n_selected": sum(len(v) for v in b.values()),
    }
    (V2_ROOT / "reports").mkdir(parents=True, exist_ok=True)
    (V2_ROOT / "reports" / "replay_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def command_scan_counts(*, languages: Sequence[str], workers: int = 6) -> dict[str, Any]:
    """Footer-only num_rows per scanned shard, for exact unparsed-row accounting.

    Reads only parquet footers (charged to the ledger) so we can reconcile
    ``data_rows == parsed_inventory_rows + unparsed_fname_rows`` per shard.
    """
    import pyarrow.parquet as pq

    ledger = v2_ledger(v2_cfg())
    state = kathbath_shard_state()
    out: dict[str, Any] = {}
    for language in languages:
        info = state[language]
        fresh = _load_fresh_state(language)
        fresh_shards = sorted({row["shard"] for row in fresh})
        parsed_counts = Counter(row["shard"] for row in fresh)
        for row in _old_inventory_rows(language):
            parsed_counts[row["shard"]] += 1
        shards = sorted(set(info["scanned_v1"]) | set(fresh_shards))
        results: dict[str, dict[str, int]] = {}
        errors: list[dict[str, Any]] = []

        def _count(shard: str) -> tuple[str, int, int]:
            counted = _pinned_open(shard)
            try:
                parquet = pq.ParquetFile(counted)
                return shard, int(parquet.metadata.num_rows), counted.bytes
            finally:
                counted.close()

        with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
            futures = {pool.submit(_count, shard): shard for shard in shards}
            for future in as_completed(futures):
                shard = futures[future]
                try:
                    shard_name, data_rows, bytes_read = future.result()
                except Exception as error:  # noqa: BLE001
                    errors.append({"shard": shard, "error": f"{type(error).__name__}: {error}"})
                    continue
                try:
                    ledger.charge("kathbath", bytes_read)
                except BudgetExceeded as error:
                    errors.append({"shard": shard, "error": f"budget: {error}"})
                    break
                results[shard_name] = {"data_rows": data_rows, "parsed_rows": parsed_counts.get(shard_name, 0),
                                        "unparsed_rows": data_rows - parsed_counts.get(shard_name, 0)}
        entry = {
            "language": language, "shards_counted": len(results), "errors": errors,
            "total_data_rows": sum(v["data_rows"] for v in results.values()),
            "total_parsed_rows": sum(v["parsed_rows"] for v in results.values()),
            "total_unparsed_rows": sum(v["unparsed_rows"] for v in results.values()),
            "per_shard": results,
        }
        out[language] = entry
        write_json_atomic(V2_ROOT / "staging" / f"kathbath_shard_counts.{language}.json", entry)
        print(json.dumps({k: entry[k] for k in ("language", "shards_counted", "total_data_rows",
                                                 "total_parsed_rows", "total_unparsed_rows")}))
    write_json_atomic(V2_ROOT / "staging" / "kathbath_shard_counts_summary.json", out)
    print(f"ledger now: {ledger.total_bytes} / {ledger.cap_bytes} ({ledger.remaining_bytes} left)")
    return out


__all__ = [
    "CORE_LANGUAGES", "INDICSYNTH_REVISION", "KATHBATH_REVISION", "V2_ROOT", "V1_ROOT",
    "closure_references_for", "closure_speakers_for", "command_discover_heldout",
    "command_discover_indicsynth", "command_discover_kathbath", "command_finalize",
    "command_materialize", "command_plan", "command_registry", "command_replay", "command_scan_counts",
    "fetch_kathbath_audio_pinned", "kathbath_shard_state", "ledger_facts", "load_registry",
    "scan_kathbath_shards_pinned", "v2_cfg", "v2_ledger", "v1_cfg",
]
