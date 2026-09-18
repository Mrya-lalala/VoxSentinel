"""Assemble pool manifests from per-source records.

The ``split`` stage groups every source's selected windows into the pool files
the handoff promises (core train/validation, supplementary, external evaluation,
fallback/baseline, unpaired candidate) and emits the missing-coverage inventory
for gated sources.  It also performs the final processing safeguards:

* prepared files must exist for every non-excluded window;
* window IDs must be unique across sources;
* exact-duplicate prepared windows inside the core pool are removed (keeping the
  earliest window ID) so the same audio can never appear in train and validation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .config import PrepConfig
from .records import read_jsonl, write_jsonl_atomic

CORE_LANGS = [
    "Bengali",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Tamil",
    "Telugu",
    "Urdu",
]

GATED_GAPS: list[dict[str, Any]] = [
    {
        "source_id": "spire_sies",
        "class": "genuine",
        "intended_pool": "supplementary",
        "status": "awaiting_user_links",
        "reason": "SPIRE portal is request-based; the user reported the archive as unavailable for this session.",
        "user_action": "Provide a local SPIRE-SIES archive path in a follow-up run (portal: https://spiredatasets.ee.iisc.ac.in/).",
    },
    {
        "source_id": "indic_timit",
        "class": "genuine",
        "intended_pool": "supplementary",
        "status": "awaiting_user_links",
        "reason": "SPIRE portal is request-based; the user reported the archive as unavailable for this session.",
        "user_action": "Provide a local Indic TIMIT release path in a follow-up run (portal: https://spiredatasets.ee.iisc.ac.in/indictimitcorpus).",
    },
    {
        "source_id": "indicvoices",
        "class": "genuine",
        "intended_pool": "external_eval",
        "status": "accessible_not_processed",
        "reason": "The HF gate was accepted during this run and file reads succeed, but the bounded scope selected Svarah for new external-evaluation material.",
        "user_action": "Optional: extend the external evaluation pool with IndicVoices shards in a follow-up run.",
    },
    {
        "source_id": "synthetic_english",
        "class": "spoof",
        "intended_pool": "core",
        "status": "missing_source",
        "reason": "No traceable synthetic Indian-English source exists and no user assets were supplied.",
        "user_action": "Provide user-authorized synthetic assets following configs/synthetic_english_template.json.",
    },
]


def _source_windows(cfg: PrepConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cfg.manifests_dir.glob("source_windows.*.jsonl")):
        rows.extend(read_jsonl(path))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        window_id = str(row.get("window_id"))
        if window_id in deduped and deduped[window_id] != row:
            raise ValueError(f"conflicting duplicate window_id {window_id} across source manifests")
        deduped[window_id] = row
    return [deduped[key] for key in sorted(deduped)]


def _source_exclusions(cfg: PrepConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(cfg.manifests_dir.glob("source_exclusions.*.jsonl")):
        rows.extend(read_jsonl(path))
    seen: set[tuple[str, str]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        key = (str(row.get("recording_id")), str(row.get("reason")))
        if key in seen:
            continue
        seen.add(key)
        result.append(row)
    return result


def _missing_coverage(cfg: PrepConfig, existing_windows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    kathbath_we_have = any(row.get("dataset_id") == "kathbath" for row in existing_windows)
    if not kathbath_we_have:
        for language in CORE_LANGS:
            gaps.append(
                {
                    "gap_id": f"kathbath-genuine-{language.lower()}",
                    "dataset_id": "kathbath",
                    "class": "genuine",
                    "spoken_language": language,
                    "intended_pool": "core",
                    "intended_windows": 20,
                    "status": "access_required",
                    "reason": "core genuine counterpart missing; the balanced core manifest must not conceal it",
                    "user_action": "https://huggingface.co/datasets/ai4bharat/Kathbath (accept terms as the account owner)",
                }
            )
    for gap in GATED_GAPS:
        gaps.append(
            {
                "gap_id": f"{gap['source_id']}-{gap['class']}",
                "dataset_id": gap["source_id"],
                "class": gap["class"],
                "spoken_language": None,
                "intended_pool": gap["intended_pool"],
                "intended_windows": None,
                "status": gap["status"],
                "reason": gap["reason"],
                "user_action": gap["user_action"],
            }
        )
    return gaps


def assemble_pools(cfg: PrepConfig) -> dict[str, Any]:
    """Build every pool manifest and return counters for the report."""
    cfg.ensure_dirs()
    windows = _source_windows(cfg)
    exclusions = _source_exclusions(cfg)

    # Safeguard 1: prepared files must exist; otherwise exclude explicitly.
    verified: list[dict[str, Any]] = []
    for row in windows:
        if row.get("pool") == "excluded":
            exclusions.append(
                {
                    "recording_id": row.get("record_id") or row.get("window_id"),
                    "dataset_id": row.get("dataset_id"),
                    "source_file": row.get("source_file"),
                    "reason": row.get("exclusion_reason") or "excluded_by_source_adapter",
                    "status": "excluded",
                }
            )
            continue
        prepared = cfg.dataset_root / str((row.get("prepared_audio") or {}).get("path", ""))
        if not prepared.exists():
            exclusions.append(
                {
                    "recording_id": row.get("recording_id"),
                    "dataset_id": row.get("dataset_id"),
                    "source_file": row.get("source_file"),
                    "reason": "prepared_file_missing",
                    "status": "excluded",
                }
            )
            continue
        verified.append(row)

    # Safeguard 2: exact-duplicate prepared windows cannot sit in the core pool
    # (or across pools); keep the first by sorted window_id.
    seen_prepared: set[str] = set()
    final: list[dict[str, Any]] = []
    for row in sorted(verified, key=lambda item: str(item.get("window_id"))):
        digest = str((row.get("prepared_audio") or {}).get("sha256"))
        if digest in seen_prepared:
            exclusions.append(
                {
                    "recording_id": row.get("recording_id"),
                    "dataset_id": row.get("dataset_id"),
                    "source_file": row.get("source_file"),
                    "reason": "exact_duplicate_prepared_window_after_assembly",
                    "status": "excluded",
                }
            )
            continue
        seen_prepared.add(digest)
        final.append(row)

    # Safeguard 3: prune exclusions whose recording was later materialized
    # (e.g. a retry succeeded) so the excluded list cannot contradict manifests.
    materialized_ids = {str(row.get("recording_id")) for row in final}
    exclusions = [
        row for row in exclusions if str(row.get("recording_id")) not in materialized_ids
    ]

    pool_files = {
        "core_train": [row for row in final if row.get("pool") == "core" and row.get("split") == "train"],
        "core_val": [row for row in final if row.get("pool") == "core" and row.get("split") == "val"],
        "supplementary": [row for row in final if row.get("pool") == "supplementary"],
        "external_eval": [row for row in final if row.get("pool") == "external_eval"],
        "fallback_baseline": [row for row in final if row.get("pool") == "fallback_baseline"],
        "unpaired_candidate": [row for row in final if row.get("pool") == "unpaired_candidate"],
    }
    written: dict[str, int] = {}
    for name, rows in pool_files.items():
        write_jsonl_atomic(cfg.manifests_dir / f"windows.{name}.jsonl", rows)
        written[name] = len(rows)

    gaps = _missing_coverage(cfg, final)
    write_jsonl_atomic(cfg.manifests_dir / "windows.missing_coverage.jsonl", gaps)
    written["missing_coverage"] = len(gaps)

    # Merged recording inventory (convenience; source files remain authoritative).
    recordings: dict[str, dict[str, Any]] = {}
    for path in sorted(cfg.manifests_dir.glob("source_recordings.*.jsonl")):
        for row in read_jsonl(path):
            recordings[str(row.get("recording_id"))] = row
    write_jsonl_atomic(cfg.manifests_dir / "recordings.jsonl", [recordings[key] for key in sorted(recordings)])

    write_jsonl_atomic(cfg.manifests_dir / "windows.excluded.jsonl", exclusions)

    counters: dict[str, Any] = {
        "pools": written,
        "exclusions": len(exclusions),
        "recordings": len(recordings),
        "windows_by_dataset_pool": _counts(final, "dataset_id", "pool"),
        "windows_by_dataset_split": _counts(
            [row for row in final if row.get("pool") == "core"], "dataset_id", "split"
        ),
        "speakers_by_dataset_split": _speaker_counts(final),
    }
    return counters


def _counts(rows: Sequence[Mapping[str, Any]], *keys: str) -> dict[str, int]:
    counter: Counter[tuple[Any, ...]] = Counter(tuple(row.get(key) for key in keys) for row in rows)
    return {"|".join(str(part) for part in key): count for key, count in sorted(counter.items(), key=lambda kv: str(kv[0]))}


def _speaker_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    # ``speaker_ids`` may carry non-identity background fields (e.g. gender);
    # count only identity keys for the speaker totals.
    non_identity = {"gender", "accent", "background", "record"}
    grouped: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    for row in rows:
        ids = row.get("speaker_ids") or {}
        for key, value in ids.items():
            if str(key).lower() in non_identity:
                continue
            if isinstance(value, str) and value:
                grouped[(row.get("dataset_id"), row.get("pool"), row.get("split"))].add(value)
            elif isinstance(value, list):
                for item in value:
                    if item:
                        grouped[(row.get("dataset_id"), row.get("pool"), row.get("split"))].add(str(item))
    return {
        "|".join(str(part) for part in key): len(values)
        for key, values in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    }


__all__ = ["CORE_LANGS", "assemble_pools"]
