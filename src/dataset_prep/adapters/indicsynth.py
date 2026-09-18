"""IndicSynth adapter: synthetic Indic speech via the HF datasets-server.

Access route (verified 2026-09-17): the dataset is public (no gate) under
CC BY-NC 4.0.  Rows are selected metadata-first through the public
datasets-server ``/rows`` API; only the chosen rows' audio assets are
downloaded.  Relationship components over source speaker, target speaker and
reference-recording identifiers are split train/validation *before* any window
is created; cross-partition conversions are impossible by construction.

Every stored window records the actual generator field, canonicalized speaker
identifiers, reference recordings, the supplied transcript and the upstream
``train`` partition.  Identifiers are never invented: a missing source speaker
is stored as ``null``, not as a shared fake identity.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .. import httpio
from ..budget import BudgetExceeded, DownloadLedger
from ..config import PrepConfig
from ..materialize import materialize_window, store_original_bytes
from ..records import build_recording_record, build_window_record, write_jsonl_atomic
from ..splitting import component_keys, relationship_components, split_components
from ...audio.prepare import AudioDecodeError
from .common import SourceRun, dedupe_exclusions, exclusion_row, load_resume_state, prune_exclusions

DATASET_ID = "indicsynth"
HF_DATASET = "vdivyasharma/IndicSynth"
SERVER = "https://datasets-server.huggingface.co"

LANGUAGE_SLUGS = {
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


def _server_json(url: str, run: SourceRun, cfg: PrepConfig, *, retries: int = 4) -> dict:
    if run.metadata_requests >= cfg.budget.max_metadata_requests_per_source:
        raise httpio.HttpError(
            f"metadata request cap reached ({cfg.budget.max_metadata_requests_per_source}); "
            "increase budget.max_metadata_requests_per_source to scan further."
        )
    time.sleep(0.15)
    payload = httpio.get_json(url, timeout=cfg.budget.request_timeout_seconds, retries=retries)
    run.metadata_requests += 1
    return payload


def _canonical_identifier(value: Any) -> str | None:
    """Canonicalize numeric identifiers; never turn a missing ID into a fake one."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        if not float(value).is_integer():
            return None
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return text or None
        if math.isnan(number) or math.isinf(number):
            return None
        return str(int(number)) if number.is_integer() else text
    return None


def _extension_for(audio_type: str | None, url: str) -> str:
    kind = (audio_type or "").lower()
    if "flac" in kind:
        return ".flac"
    if "mpeg" in kind or "mp3" in kind:
        return ".mp3"
    if "wav" in kind:
        return ".wav"
    suffix = Path(url.split("?", 1)[0]).suffix.lower()
    return suffix if suffix in {".wav", ".flac", ".mp3", ".ogg", ".m4a"} else ".wav"


def _row_facts(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    """Normalize one datasets-server row into the fields this pilot records."""
    audio_cells = row.get("audio") or []
    audio_src, audio_type = None, None
    if isinstance(audio_cells, Sequence) and audio_cells:
        cell = audio_cells[0]
        if isinstance(cell, Mapping):
            audio_src = cell.get("src")
            audio_type = cell.get("type")
    return {
        "row_index": row_index,
        "audio_src": audio_src,
        "audio_type": audio_type,
        "generator": row.get("Generative Model"),
        "source_speaker": _canonical_identifier(row.get("Source Speaker_ID")),
        "target_speaker": _canonical_identifier(row.get("Target Speaker ID")),
        "gender": row.get("Gender"),
        "source_reference": row.get("Source Reference Audio"),
        "target_reference": row.get("Target Reference Audio"),
        "transcript": row.get("TTS Transcript"),
    }


def _scan_language(cfg: PrepConfig, language: str, run: SourceRun, *, total_rows: int | None) -> list[dict[str, Any]]:
    quota = cfg.quota("indicsynth")
    strata = int(quota.get("scan_strata", 10))
    page = int(quota.get("page_length", 100))
    rng = random.Random(f"{cfg.seed}:{language}:scan")

    if total_rows is None or total_rows <= 0:
        size = _server_json(f"{SERVER}/size?dataset={HF_DATASET.replace('/', '%2F')}", run, cfg)
        total_rows = 0
        for entry in (size.get("size", {}).get("configs", []) or []):
            if str(entry.get("config")) == language:
                total_rows = int(entry.get("num_rows", 0))
                break
    total_rows = total_rows or 0
    if total_rows <= 0:
        run.note(f"{language}: dataset-server reported no rows")
        return []

    offsets: list[int] = []
    stratum_width = max(page, total_rows // max(strata, 1))
    for stratum in range(max(strata, 1)):
        start = stratum * stratum_width
        if start >= total_rows:
            break
        end = min(total_rows - page, start + stratum_width - page) if total_rows > page else 0
        offset = rng.randrange(start, max(start, end) + 1) if end > start else start
        offsets.append(min(offset, max(0, total_rows - page)))

    rows: list[dict[str, Any]] = []
    seen_indices: set[int] = set()
    for offset in offsets:
        url = (
            f"{SERVER}/rows?dataset={HF_DATASET.replace('/', '%2F')}&config={language}"
            f"&split=train&offset={offset}&length={page}"
        )
        payload = _server_json(url, run, cfg)
        for item in payload.get("rows", []) or []:
            index = int(item.get("row_idx", -1))
            if index < 0 or index in seen_indices:
                continue
            seen_indices.add(index)
            rows.append(_row_facts(item.get("row", {}) or {}, index))
    return rows


def _select_candidates(rows: Sequence[Mapping[str, Any]], count: int, seed: str) -> list[dict[str, Any]]:
    """Pick spread candidates: round-robin over generators, distinct pairs first."""
    usable = [dict(row) for row in rows if row.get("audio_src")]
    if not usable:
        return []
    rng = random.Random(seed)
    order = list(range(len(usable)))
    rng.shuffle(order)

    def pair(row: Mapping[str, Any]) -> tuple[Any, Any]:
        return (row.get("source_speaker"), row.get("target_speaker"))

    by_generator: dict[str, list[int]] = {}
    for index in order:
        key = str(usable[index].get("generator") or "unknown")
        by_generator.setdefault(key, []).append(index)
    generators = sorted(by_generator)

    selected: list[int] = []
    chosen: set[int] = set()
    seen_pairs: set[tuple[Any, Any]] = set()
    progress = True
    while len(selected) < count and progress:
        progress = False
        for generator in generators:
            if len(selected) >= count:
                break
            pick: int | None = None
            for index in by_generator[generator]:
                if index in chosen:
                    continue
                if pair(usable[index]) not in seen_pairs:
                    pick = index
                    break
            if pick is None:
                for index in by_generator[generator]:
                    if index not in chosen:
                        pick = index
                        break
            if pick is not None:
                chosen.add(pick)
                selected.append(pick)
                seen_pairs.add(pair(usable[pick]))
                progress = True
    return [usable[index] for index in selected]


def _relationship_keys(row: Mapping[str, Any]) -> Iterable[tuple[str, str]]:
    keys: list[tuple[str, str]] = []
    if row.get("source_speaker"):
        keys.append(("speaker", str(row["source_speaker"])))
    if row.get("target_speaker"):
        keys.append(("speaker", str(row["target_speaker"])))
    if row.get("source_reference"):
        keys.append(("recording", str(row["source_reference"])))
    if row.get("target_reference"):
        keys.append(("recording", str(row["target_reference"])))
    return keys


def plan_language_split(
    candidates: Sequence[Mapping[str, Any]], cfg: PrepConfig, language: str
) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    """Assign candidate rows to train/val via relationship components.

    Returns ``(kept_rows_with_split, dropped_rows)`` where dropped rows are
    documented shortfall/exclusion material.  Window quotas are applied only
    after splitting, trimming round-robin across generators.
    """
    quota = cfg.quota("indicsynth")
    windows_per_language = int(quota.get("windows_per_language", 20))
    rows = [dict(row) for row in candidates]
    if not rows:
        return [], []

    components = relationship_components(rows, _relationship_keys)
    counts = [len(component) for component in components]
    assignment = split_components(components, counts, val_fraction=0.2, seed=cfg.seed)
    split_of_row: dict[int, str] = {}
    for component_index, component in enumerate(components):
        for row_index in component:
            split_of_row[row_index] = assignment[component_index]

    # Integrity check: every row's relationship keys must live in one partition.
    keys_by_component = component_keys(rows, components, _relationship_keys)
    owner: dict[tuple[str, str], str] = {}
    for component_index, keys in enumerate(keys_by_component):
        for key in keys:
            owner[key] = assignment[component_index]
    for index, row in enumerate(rows):
        partitions = {owner[key] for key in _relationship_keys(row)}
        if len(partitions) > 1:
            # Should not occur after component-level splitting; drop defensively.
            split_of_row[index] = "reject"

    target_train = max(1, round(windows_per_language * 0.8))
    target_val = max(0, windows_per_language - target_train)
    kept: list[tuple[dict[str, Any], str]] = []
    dropped: list[dict[str, Any]] = []

    def trim(pool_name: str, budget: int) -> None:
        indices = [i for i in range(len(rows)) if split_of_row.get(i) == pool_name]
        rng = random.Random(f"{cfg.seed}:{language}:{pool_name}:trim")
        by_generator: dict[str, list[int]] = {}
        for index in indices:
            key = str(rows[index].get("generator") or "unknown")
            by_generator.setdefault(key, []).append(index)
        for bucket in by_generator.values():
            rng.shuffle(bucket)
        generators = sorted(by_generator)
        cursor = {key: 0 for key in generators}
        taken = 0
        progress = True
        while taken < budget and progress:
            progress = False
            for generator in generators:
                if taken >= budget:
                    break
                bucket = by_generator[generator]
                while cursor[generator] < len(bucket):
                    index = bucket[cursor[generator]]
                    cursor[generator] += 1
                    kept.append((rows[index], pool_name))
                    taken += 1
                    progress = True
                    break
        for generator in generators:
            for index in by_generator[generator][cursor[generator]:]:
                dropped.append({**rows[index], "reason": f"quota_trim_{pool_name}", "planned_split": pool_name})

    trim("train", target_train)
    trim("val", target_val)
    for index in range(len(rows)):
        split = split_of_row.get(index)
        if split == "reject" or split is None:
            dropped.append({**rows[index], "reason": "cross_partition_conversion", "planned_split": split})
    return kept, dropped


def fetch_indicsynth(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    languages: Sequence[str] | None = None,
    force: bool = False,
) -> SourceRun:
    """Run the full bounded IndicSynth materialization."""
    cfg.ensure_dirs()
    run = SourceRun(source_id=DATASET_ID)
    quota = cfg.quota("indicsynth")
    selected_languages = list(languages or quota.get("languages", list(LANGUAGE_SLUGS)))
    candidates_per_language = int(quota.get("candidates_per_language", 28))
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
    existing_recordings = state.recordings
    registry = state.registry
    run.recordings = list(existing_recordings.values())
    run.windows = list(state.windows)
    run.exclusions = list(state.exclusions)

    size = httpio.get_json(f"{SERVER}/size?dataset={HF_DATASET.replace('/', '%2F')}", timeout=cfg.budget.request_timeout_seconds)
    run.metadata_requests += 1
    rows_per_config = {
        str(entry.get("config")): int(entry.get("num_rows", 0))
        for entry in (size.get("size", {}).get("configs", []) or [])
    }

    for language in selected_languages:
        slug = LANGUAGE_SLUGS.get(language, language.lower())
        scanned = _scan_language(cfg, language, run, total_rows=rows_per_config.get(language))
        candidates = _select_candidates(scanned, candidates_per_language, f"{cfg.seed}:{language}:candidates")
        kept, dropped = plan_language_split(candidates, cfg, language)
        for row in dropped:
            run.exclusions.append(
                exclusion_row(
                    recording_id=f"{DATASET_ID}-{slug}-{row.get('row_index')}",
                    dataset_id=DATASET_ID,
                    source_file=f"{language}/train/row-{row.get('row_index')}",
                    original_split="train",
                    reason=str(row.get("reason")),
                    planned_split=row.get("planned_split"),
                    generator=row.get("generator"),
                )
            )

        pending = [pair for pair in kept if f"{DATASET_ID}-{slug}-{pair[0].get('row_index'):06d}" not in existing_recordings]
        run.note(f"{language}: scanned={len(scanned)} candidates={len(candidates)} kept={len(kept)} new={len(pending)}")

        for row, split in pending:
            row_index = int(row["row_index"])
            window_id = f"{DATASET_ID}-{slug}-{row_index:06d}"
            source_file = f"{language}/train/row-{row_index}"
            extension = _extension_for(row.get("audio_type"), str(row.get("audio_src") or ""))
            raw_path = cfg.raw_dir / DATASET_ID / slug / f"{row_index:06d}{extension}"
            if not (raw_path.exists() and raw_path.stat().st_size > 0):
                try:
                    payload = httpio.get_bytes(
                        str(row["audio_src"]), timeout=cfg.budget.request_timeout_seconds, max_bytes=20 * 1024 * 1024
                    )
                    ledger.charge(DATASET_ID, len(payload))
                    run.download_bytes += len(payload)
                except BudgetExceeded as error:
                    run.note(f"{language}: stopped at row {row_index}: {error}")
                    break
                except httpio.HttpError as error:
                    run.exclusions.append(
                        exclusion_row(
                            recording_id=window_id,
                            dataset_id=DATASET_ID,
                            source_file=source_file,
                            original_split="train",
                            reason=f"audio_asset_fetch_failed: {error}",
                        )
                    )
                    continue
                store_original_bytes(raw_path, payload)
            prepared_path = cfg.prepared_dir / DATASET_ID / slug / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
            except (AudioDecodeError, ValueError) as error:
                run.exclusions.append(
                    exclusion_row(
                        recording_id=window_id,
                        dataset_id=DATASET_ID,
                        source_file=source_file,
                        original_split="train",
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
                        source_file=source_file,
                        original_split="train",
                        reason=duplicate,
                    )
                )
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)

            speaker_ids = {"source": row.get("source_speaker"), "target": row.get("target_speaker")}
            recording_record = build_recording_record(
                recording_id=window_id,
                dataset_id=DATASET_ID,
                source_url=str(release.get("source_url")),
                source_file=source_file,
                original_split="train",
                label=1,
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=row.get("generator"),
                generator_version=None,
                parent_refs={
                    "source_reference": row.get("source_reference"),
                    "target_reference": row.get("target_reference"),
                    "transcript": row.get("transcript"),
                    "gender": row.get("gender"),
                    "dataset_revision": release.get("revision"),
                },
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                decoded=materialized.decoded_facts(),
                license_note=str(release.get("license")),
                notes="synthetic (generated) example; not genuine speech",
            )
            window_record = build_window_record(
                window_id=window_id,
                recording_id=window_id,
                dataset_id=DATASET_ID,
                source_url=str(release.get("source_url")),
                source_file=source_file,
                original_split="train",
                pool="core",
                split=split,
                label=1,
                label_source="generated_synthetic (IndicSynth generator field)",
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=row.get("generator"),
                generator_version=None,
                parent_refs=recording_record["parent_refs"],
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                preprocessing_version=str(cfg.preprocessing.get("config_version")),
                license_note=str(release.get("license")),
                access_status="public",
                status="audio_ready",
                notes="one window per generated recording; upstream train partition",
            )
            existing_recordings[window_id] = recording_record
            run.recordings.append(recording_record)
            run.windows.append(window_record)

    write_jsonl_atomic(recordings_path, run.recordings)
    write_jsonl_atomic(windows_path, [record for record in run.windows])
    write_jsonl_atomic(
        exclusions_path, dedupe_exclusions(prune_exclusions(run.exclusions, set(existing_recordings)))
    )
    return run


__all__ = ["DATASET_ID", "LANGUAGE_SLUGS", "SourceRun", "fetch_indicsynth", "plan_language_split"]
