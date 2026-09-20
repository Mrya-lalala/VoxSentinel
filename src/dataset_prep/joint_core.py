"""Joint genuine(Kathbath) + synthetic(IndicSynth) core-pilot builder.

One relationship graph per language is built BEFORE any window is created:

* every IndicSynth candidate contributes its source speaker, target speaker,
  source reference recording and target reference recording as one connected
  component;
* every Kathbath genuine recording contributes its recording + speaker;
* components are split 80/20 between train and validation, then windows are
  selected inside each split with per-individual-speaker exposure caps.

Reference strings (`844424932377082-1005-f.wav`) are resolved against Kathbath
originals by recording identity (extension differences are expected); declared
speaker fields must agree with the parsed reference.  Candidates whose parents
appear in upstream held-out shards, or whose lineage does not parse, are
excluded from the strict core pool and itemized in the exclusions inventory.
"""

from __future__ import annotations

import json
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import httpio
from .budget import BudgetExceeded, DownloadLedger
from .config import PrepConfig
from .materialize import HashRegistry, materialize_window, store_original_bytes
from .records import (
    build_recording_record,
    build_window_record,
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from .splitting import UnionFind
from ..audio.prepare import AudioDecodeError
from .adapters import kathbath as kb
from .adapters.common import SourceRun, dedupe_exclusions, exclusion_row

SERVER = "https://datasets-server.huggingface.co"
HF_DATASET = "vdivyasharma/IndicSynth"

JOINT_DIRNAME = "joint"


def _canonical_int(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not float(number).is_integer() or abs(number) == float("inf"):
        return None
    return str(int(number))


def _parse_reference(reference: Any) -> dict[str, Any] | None:
    """Parse an IndicSynth reference name into Kathbath recording identity."""
    if not isinstance(reference, str) or not reference.strip():
        return None
    return kb.parse_fname(reference.strip())


# ---------------------------------------------------------------------------
# IndicSynth candidate scan (public datasets-server; cached in staging)
# ---------------------------------------------------------------------------


def _row_facts(row: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    source = _parse_reference(row.get("Source Reference Audio"))
    target = _parse_reference(row.get("Target Reference Audio"))
    declared_source = _canonical_int(row.get("Source Speaker_ID"))
    declared_target = _canonical_int(row.get("Target Speaker ID"))
    audio_cells = row.get("audio") or []
    audio_src = None
    if isinstance(audio_cells, Sequence) and audio_cells:
        cell = audio_cells[0]
        if isinstance(cell, Mapping):
            audio_src = cell.get("src")
    return {
        "row_index": row_index,
        "generator": row.get("Generative Model"),
        "declared_source_speaker": declared_source,
        "declared_target_speaker": declared_target,
        "source_reference": row.get("Source Reference Audio"),
        "target_reference": row.get("Target Reference Audio"),
        "source_record": source["record_id"] if source else None,
        "source_recording_speaker": source["speaker"] if source else None,
        "target_record": target["record_id"] if target else None,
        "target_recording_speaker": target["speaker"] if target else None,
        "transcript": row.get("TTS Transcript"),
        "audio_src": audio_src,
    }


def scan_indicsynth_candidates(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    language: str,
    *,
    force: bool = False,
) -> list[dict[str, Any]]:
    """Scanned candidate rows for one language, cached in staging (resume-safe)."""
    cache = cfg.staging_dir / JOINT_DIRNAME / f"indicsynth_candidates.{language}.jsonl"
    if cache.exists() and not force:
        rows = read_jsonl(cache)
        if rows:
            return rows
    quota = cfg.quota("indicsynth")
    strata = int(quota.get("scan_strata", 12))
    page = int(quota.get("page_length", 100))
    rng = random.Random(f"{cfg.seed}:{language}:scan")

    size = httpio.get_json(f"{SERVER}/size?dataset={HF_DATASET.replace('/', '%2F')}", timeout=cfg.budget.request_timeout_seconds)
    total_rows = 0
    for entry in (size.get("size", {}).get("configs", []) or []):
        if str(entry.get("config")) == language:
            total_rows = int(entry.get("num_rows", 0))
            break
    if total_rows <= 0:
        return []

    stratum_width = max(page, total_rows // max(strata, 1))
    offsets: list[int] = []
    for stratum in range(max(strata, 1)):
        start = stratum * stratum_width
        if start >= total_rows:
            break
        end = min(max(0, total_rows - page), start + stratum_width - page)
        offset = rng.randrange(start, max(start, end) + 1) if end > start else start
        offsets.append(min(offset, max(0, total_rows - page)))

    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for offset in offsets:
        url = (
            f"{SERVER}/rows?dataset={HF_DATASET.replace('/', '%2F')}&config={language}"
            f"&split=train&offset={offset}&length={page}"
        )
        time.sleep(0.12)
        payload = httpio.get_json(url, timeout=cfg.budget.request_timeout_seconds, retries=5)
        for item in payload.get("rows", []) or []:
            index = int(item.get("row_idx", -1))
            if index < 0 or index in seen:
                continue
            seen.add(index)
            rows.append(_row_facts(item.get("row", {}) or {}, index))
    write_jsonl_atomic(cache, rows)
    return rows


def refresh_asset_urls(
    cfg: PrepConfig,
    candidates: dict[int, dict[str, Any]],
    row_indices: Iterable[int],
    language: str,
) -> None:
    """Fetch fresh signed asset URLs for specific rows (one small query each)."""
    for index in row_indices:
        row = candidates[index]
        url = (
            f"{SERVER}/rows?dataset={HF_DATASET.replace('/', '%2F')}&config={language}"
            f"&split=train&offset={index}&length=1"
        )
        time.sleep(0.1)
        payload = httpio.get_json(url, timeout=cfg.budget.request_timeout_seconds, retries=4)
        items = payload.get("rows", []) or []
        if items:
            fresh = _row_facts(items[0].get("row", {}) or {}, index)
            row["audio_src"] = fresh.get("audio_src")


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


@dataclass
class PlanResult:
    language: str
    plan: dict[str, Any]
    exclusions: list[dict[str, Any]] = field(default_factory=list)


def _pair_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    source = row.get("declared_source_speaker")
    source_key = "" if source in (None, "None") else str(source)
    return (source_key, str(row["declared_target_speaker"]), str(row["generator"]))


def _choose_pairs(
    rows: Sequence[Mapping[str, Any]],
    *,
    n_pairs: int,
    max_rows_per_pair: int,
    seed: str,
) -> tuple[list[tuple[tuple[str, str, str], list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Pick up to ``n_pairs`` (source,target,generator) triples, disjoint-first."""
    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[_pair_key(row)].append(dict(row))
    keys = sorted(grouped)
    rng.shuffle(keys)
    # interleave generators so no single generator dominates the candidate list
    by_generator: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in keys:
        by_generator[key[2]].append(key)
    for bucket in by_generator.values():
        bucket.sort(key=lambda k: (k[0], k[1]))
    generators = sorted(by_generator)
    ordered: list[tuple[str, str, str]] = []
    position = 0
    while True:
        added = False
        for generator in generators:
            bucket = by_generator[generator]
            if position < len(bucket):
                ordered.append(bucket[position])
                added = True
        position += 1
        if not any(position < len(by_generator[g]) for g in generators):
            break
        if not added:  # pragma: no cover - defensive
            break

    selected: list[tuple[tuple[str, str, str], list[dict[str, Any]]]] = []
    used: set[str] = set()
    for pass_index in range(2):  # pass 0: disjoint pairs; pass 1: allow reuse
        for key in ordered:
            if len(selected) >= n_pairs:
                break
            if any(key == existing[0] for existing in selected):
                continue
            source, target, _generator = key
            if pass_index == 0:
                if (source and source in used) or target in used:
                    continue
            bucket = sorted(grouped[key], key=lambda r: int(r["row_index"]))[:max_rows_per_pair]
            selected.append((key, bucket))
            if source:
                used.add(source)
            used.add(target)
        if len(selected) >= n_pairs:
            break
    dropped = [
        row
        for key, bucket in grouped.items()
        if key not in {existing[0] for existing in selected}
        for row in bucket
    ]
    return selected, dropped


def _genuine_candidates_by_speaker(
    inventory: Sequence[Mapping[str, Any]],
    speakers: Iterable[str],
    *,
    minimum_seconds: float = 1.0,
    maximum_seconds: float = 30.0,
) -> dict[str, list[dict[str, Any]]]:
    wanted = set(str(speaker) for speaker in speakers)
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in inventory:
        if str(row["speaker"]) in wanted and minimum_seconds <= float(row["duration"]) <= maximum_seconds:
            result[str(row["speaker"])].append(dict(row))
    for bucket in result.values():
        bucket.sort(key=lambda r: (str(r["shard"]), int(r["row_group"]), int(r["row_index"])))
    return result


def _select_genuine(
    available: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    target: int,
    per_speaker_cap: int,
    used_shards_inventory: dict[tuple[str, int], int],
    seed: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Pick genuine recordings, preferring speakers with more material and reusing row groups."""
    rng = random.Random(seed)
    speakers = sorted(available)
    rng.shuffle(speakers)
    speakers.sort(key=lambda s: (-len(available[s]), s))
    chosen: list[dict[str, Any]] = []
    per_speaker: Counter[str] = Counter()
    group_use: dict[tuple[str, int], list[str]] = defaultdict(list)

    def score(row: Mapping[str, Any]) -> tuple[int, float]:
        group = (str(row["shard"]), int(row["row_group"]))
        reuse = 0 if group in used_shards_inventory else 1
        duration_penalty = abs(float(row["duration"]) - 5.0)
        return (reuse, duration_penalty)

    pending = [speaker for speaker in speakers]
    while pending and len(chosen) < target:
        progressed = False
        for speaker in list(pending):
            if len(chosen) >= target:
                break
            if per_speaker[speaker] >= per_speaker_cap:
                pending.remove(speaker)
                continue
            bucket = sorted(
                (row for row in available[speaker] if row["fname"] not in {c["fname"] for c in chosen}),
                key=score,
            )
            if not bucket:
                pending.remove(speaker)
                continue
            pick = bucket[0]
            chosen.append(pick)
            per_speaker[speaker] += 1
            group = (str(pick["shard"]), int(pick["row_group"]))
            group_use[group].append(speaker)
            used_shards_inventory[group] = used_shards_inventory.get(group, 0) + 1
            progressed = True
        if not progressed:
            break
    diagnostics = {
        "windows": len(chosen),
        "speakers": len({row["speaker"] for row in chosen}),
        "row_groups": len({(row["shard"], row["row_group"]) for row in chosen}),
        "per_speaker": {speaker: int(count) for speaker, count in sorted(per_speaker.items())},
    }
    return chosen, diagnostics


def _synthetic_diagnostics(chosen: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "windows": len(chosen),
        "generators": dict(Counter(str(row["generator"]) for row in chosen)),
        "source_speakers": len(
            {str(row["declared_source_speaker"]) for row in chosen if row.get("declared_source_speaker")}
        ),
        "target_speakers": len({str(row["declared_target_speaker"]) for row in chosen}),
        "components": len({str(row.get("_component")) for row in chosen if row.get("_component")}),
    }


def _select_synthetic(
    rows: Sequence[Mapping[str, Any]],
    *,
    target: int,
    source_cap: int,
    target_cap: int,
    seed: str,
    per_component_cap: int | None = None,
    initial_source_use: Mapping[str, int] | None = None,
    initial_target_use: Mapping[str, int] | None = None,
    initial_component_use: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    ordered = sorted(rows, key=lambda r: int(r["row_index"]))
    rng.shuffle(ordered)
    by_generator: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ordered:
        by_generator[str(row["generator"])].append(row)
    generators = sorted(by_generator)
    for bucket in by_generator.values():
        bucket.sort(key=lambda r: int(r["row_index"]))
    source_use: Counter[str] = Counter(initial_source_use or {})
    target_use: Counter[str] = Counter(initial_target_use or {})
    component_use: Counter[str] = Counter(initial_component_use or {})
    chosen: list[dict[str, Any]] = []
    cursor = {generator: 0 for generator in generators}

    def admissible(row: Mapping[str, Any]) -> bool:
        source = str(row.get("declared_source_speaker") or "")
        target_speaker = str(row["declared_target_speaker"])
        if target_use[target_speaker] >= target_cap:
            return False
        if source and source_use[source] >= source_cap:
            return False
        if per_component_cap is not None:
            component = str(row.get("_component") or "")
            if component and component_use[component] >= per_component_cap:
                return False
        return True

    while len(chosen) < target:
        progressed = False
        for generator in generators:
            if len(chosen) >= target:
                break
            bucket = by_generator[generator]
            while cursor[generator] < len(bucket):
                row = bucket[cursor[generator]]
                cursor[generator] += 1
                if admissible(row):
                    chosen.append(row)
                    source = str(row.get("declared_source_speaker") or "")
                    if source:
                        source_use[source] += 1
                    target_use[str(row["declared_target_speaker"])] += 1
                    component = str(row.get("_component") or "")
                    if component:
                        component_use[component] += 1
                    progressed = True
                    break
        if not progressed:
            break
    diagnostics = _synthetic_diagnostics(chosen)
    return chosen, diagnostics


def filter_eligible(
    candidates: Sequence[Mapping[str, Any]],
    language: str,
    held_out_recordings: set[str],
    *,
    excluded_rows: Mapping[int, Sequence[str]] | None = None,
    resolution_cache: Mapping[str, Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Lineage/identity eligibility filter; returns (eligible rows, exclusions).

    ``excluded_rows`` carries planner-level exclusions (row index -> reasons)
    produced by the lineage resolution pass, so replacement rounds never
    re-select rows whose references could not be verified against the full
    upstream train shards.

    Expected reference shapes (verified against the released metadata):

    * ``freevc24`` (voice conversion): source speaker + source recording +
      target speaker + target recording are all required;
    * ``xtts_v2`` / ``vits`` (text-to-speech): only the target speaker and
      target reference recording exist - a missing source step is *absent by
      design*, not unresolved lineage, and is recorded as such.
    """
    exclusions: list[dict[str, Any]] = []
    eligible: list[dict[str, Any]] = []
    blocked = {int(key): list(value) for key, value in (excluded_rows or {}).items()}
    for row in candidates:
        index = int(row["row_index"])
        record_id = f"indicsynth-{language.lower()}-{index:06d}"
        generator = str(row.get("generator"))
        if index in blocked:
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="lineage_replacement_required: " + ", ".join(blocked[index]),
                generator=row.get("generator")))
            continue
        if resolution_cache:
            unresolved_roles: list[str] = []
            for role, reference in (
                ("source", row.get("source_reference")),
                ("target", row.get("target_reference")),
            ):
                if not reference:
                    continue
                record = str(reference).split("-", 1)[0]
                status = str((resolution_cache.get(record) or {}).get("status"))
                if status in ("not_found", "upstream_eval_parent", "budget_stopped"):
                    unresolved_roles.append(f"{role}:{status}")
            if unresolved_roles:
                exclusions.append(exclusion_row(
                    recording_id=record_id, dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{index}", original_split="train",
                    reason="reference_unresolvable_precheck: " + ", ".join(unresolved_roles),
                    generator=row.get("generator")))
                continue
        if generator not in ("freevc24", "xtts_v2", "vits"):
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="unknown_generator", generator=row.get("generator")))
            continue
        target_speaker = _canonical_int(row.get("target_recording_speaker"))
        declared_target = _canonical_int(row.get("declared_target_speaker"))
        if not row.get("target_record") or target_speaker is None or declared_target is None:
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="unresolved_target_lineage"))
            continue
        if target_speaker != declared_target:
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="target_speaker_mismatch",
                declared_target=declared_target, parsed_target=target_speaker))
            continue
        normalized = dict(row)
        if generator == "freevc24":
            source_speaker = _canonical_int(row.get("source_recording_speaker"))
            declared_source = _canonical_int(row.get("declared_source_speaker"))
            if not row.get("source_record") or source_speaker is None or declared_source is None:
                exclusions.append(exclusion_row(
                    recording_id=record_id, dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{index}", original_split="train",
                    reason="unresolved_source_lineage"))
                continue
            if source_speaker != declared_source:
                exclusions.append(exclusion_row(
                    recording_id=record_id, dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{index}", original_split="train",
                    reason="source_speaker_mismatch",
                    declared_source=declared_source, parsed_source=source_speaker))
                continue
            normalized["source_kind"] = "conversion_source"
        else:
            # TTS: no source voice step exists; keep any extra source data if
            # present and consistent, otherwise record the absence explicitly.
            source_speaker = _canonical_int(row.get("source_recording_speaker"))
            declared_source = _canonical_int(row.get("declared_source_speaker"))
            if (row.get("source_record") and source_speaker is not None and declared_source is not None
                    and source_speaker == declared_source):
                normalized["source_kind"] = "declared_source"
            else:
                normalized["source_kind"] = "tts_target_only"
                normalized["source_record"] = None
                normalized["source_recording_speaker"] = None
                normalized["declared_source_speaker"] = None
                normalized["source_reference"] = None
        if row["target_record"] in held_out_recordings:
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="upstream_held_out_parent"))
            continue
        if row.get("source_record") and row["source_record"] in held_out_recordings:
            exclusions.append(exclusion_row(
                recording_id=record_id, dataset_id="indicsynth",
                source_file=f"{language}/train/row-{index}", original_split="train",
                reason="upstream_held_out_parent"))
            continue
        eligible.append(normalized)
    return eligible, exclusions


def select_final_pairs(
    eligible: Sequence[Mapping[str, Any]],
    inventory_speakers: set[str],
    *,
    n_pairs: int,
    max_rows_per_pair: int,
    seed: str,
) -> tuple[list[tuple[tuple[str, str, str], list[dict[str, Any]]]], list[dict[str, Any]]]:
    """Strictly speaker-disjoint, generator-balanced pair selection.

    Disjointness is enforced across every identifiable participant (source
    speaker, target speaker, and therefore their reference recordings): two
    selected pairs may never share a person, so no two pair-components can merge
    and no cross-component bridges exist inside the selection.

    Pairs are drawn round-robin over generators; rows whose participants include
    a speaker with genuine recordings (``inventory_speakers``) are preferred so
    the two classes can co-locate.  A candidate that cannot be placed without
    overlapping an already selected participant is dropped (reported), never
    merged.
    """
    rng = random.Random(seed)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[_pair_key(row)].append(dict(row))
    keys = sorted(grouped)
    rng.shuffle(keys)

    def participants(key: tuple[str, str, str]) -> set[str]:
        source, target, _generator = key
        return {value for value in (source, target) if value}

    def anchored(key: tuple[str, str, str]) -> bool:
        return bool(participants(key) & inventory_speakers)

    by_generator: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
    for key in keys:
        by_generator[key[2]].append(key)
    for bucket in by_generator.values():
        # anchored keys first, then stable by key value
        bucket.sort(key=lambda k: (0 if anchored(k) else 1, k[0], k[1]))
    generators = sorted(by_generator)

    selected: list[tuple[tuple[str, str, str], list[dict[str, Any]]]] = []
    used: set[str] = set()
    cursor = {generator: 0 for generator in generators}
    while len(selected) < n_pairs:
        progressed = False
        for generator in generators:
            if len(selected) >= n_pairs:
                break
            bucket = by_generator[generator]
            while cursor[generator] < len(bucket):
                key = bucket[cursor[generator]]
                cursor[generator] += 1
                if participants(key) & used:
                    continue
                rows = sorted(grouped[key], key=lambda r: int(r["row_index"]))[:max_rows_per_pair]
                selected.append((key, rows))
                used.update(participants(key))
                progressed = True
                break
        if not progressed:
            break

    selected_keys = {key for key, _ in selected}
    dropped = [row for key, bucket in grouped.items() if key not in selected_keys for row in bucket]
    return selected, dropped


def plan_language(
    cfg: PrepConfig,
    language: str,
    candidates: Sequence[Mapping[str, Any]],
    inventory: Sequence[Mapping[str, Any]],
    held_out_recordings: set[str],
    *,
    pairs: list[tuple[tuple[str, str, str], list[dict[str, Any]]]] | None = None,
    base_exclusions: list[dict[str, Any]] | None = None,
) -> PlanResult:
    """Build the joint graph for one language and select train/validation windows.

    ``pairs`` and ``base_exclusions`` may be supplied by the orchestrator (which
    computes them deterministically before deciding how many Kathbath shards to
    scan); otherwise they are derived here.
    """
    quota = cfg.quota("core")
    n_pairs = int(quota.get("candidate_pairs", 8))
    max_rows_per_pair = int(quota.get("max_rows_per_pair", 5))
    train_genuine_target = int(quota.get("train_genuine", 16))
    train_synth_target = int(quota.get("train_synthetic", 16))
    val_genuine_target = int(quota.get("val_genuine", 4))
    val_synth_target = int(quota.get("val_synthetic", 4))
    genuine_cap = int(quota.get("max_genuine_windows_per_speaker", 4))
    synth_source_cap = int(quota.get("max_synth_rows_per_source_speaker", 4))
    synth_target_cap = int(quota.get("max_synth_rows_per_target_speaker", 4))
    min_train_speakers = int(quota.get("min_train_speakers", 3))
    min_val_speakers = int(quota.get("min_val_speakers", 2))

    if pairs is None:
        eligible, exclusions = filter_eligible(candidates, language, held_out_recordings)
        pairs, dropped_rows = _choose_pairs(
            eligible, n_pairs=n_pairs, max_rows_per_pair=max_rows_per_pair,
            seed=f"{cfg.seed}:{language}:pairs")
        for row in dropped_rows:
            exclusions.append(exclusion_row(
                recording_id=f"indicsynth-{language.lower()}-{int(row['row_index']):06d}",
                dataset_id="indicsynth", source_file=f"{language}/train/row-{int(row['row_index'])}",
                original_split="train", reason="candidate_pool_not_selected",
                generator=row.get("generator")))
    else:
        exclusions = list(base_exclusions or [])

    # ---- relationship graph --------------------------------------------------
    finder = UnionFind()
    pair_rows: list[dict[str, Any]] = []
    for _, bucket in pairs:
        for row in bucket:
            pair_rows.append(row)
            target_node = ("s", str(row["declared_target_speaker"]))
            target_rec_node = ("r", str(row["target_record"]))
            finder.union(target_node, target_rec_node)
            source = row.get("declared_source_speaker")
            if source and row.get("source_record"):
                source_node = ("s", str(source))
                source_rec_node = ("r", str(row["source_record"]))
                # A single generated row links its source side and target side:
                # all identifiable participants form one component.
                finder.union(target_node, source_node)
                finder.union(target_node, source_rec_node)

    source_speakers = {
        str(row["declared_source_speaker"]) for row in pair_rows if row.get("declared_source_speaker")
    }
    target_speakers = {str(row["declared_target_speaker"]) for row in pair_rows}
    participant_speakers = sorted(source_speakers | target_speakers)
    inventory_speakers = sorted({str(row["speaker"]) for row in inventory})
    # Every scanned speaker is a genuine candidate.  Participant speakers are
    # unioned into their pair's component; the others stay standalone genuine
    # components ("free agents") that can be placed on either side later.
    available_genuine = _genuine_candidates_by_speaker(inventory, inventory_speakers)
    for speaker, bucket in available_genuine.items():
        for row in bucket:
            finder.union(("s", speaker), ("r", str(row["record_id"])))

    components: dict[Any, dict[str, Any]] = {}
    for row in pair_rows:
        root = finder.find(("s", str(row["declared_target_speaker"])))
        components.setdefault(root, {"synthetic": [], "genuine": []})["synthetic"].append(row)
    for speaker, bucket in available_genuine.items():
        root = finder.find(("s", speaker))
        components.setdefault(root, {"synthetic": [], "genuine": []})["genuine"].extend(bucket)

    ordered_components = [
        {**component, "key": str(root)}
        for root, component in sorted(
            components.items(),
            key=lambda kv: (-(len(kv[1]["synthetic"]) + len(kv[1]["genuine"])), str(kv[0])),
        )
    ]

    # ---- component resources -------------------------------------------------
    import math

    for component in ordered_components:
        for row in component["synthetic"]:
            row["_component"] = component["key"]
        component["synth_rows"] = sorted(component["synthetic"], key=lambda r: int(r["row_index"]))
        component["generators"] = sorted({str(row["generator"]) for row in component["synthetic"]})
        buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in component["genuine"]:
            buckets[str(row["speaker"])].append(row)
        component["genuine_by_speaker"] = buckets

    pair_components = [component for component in ordered_components if component["synthetic"]]
    free_components = [component for component in ordered_components if not component["synthetic"]]
    available_generators = sorted({g for component in pair_components for g in component["generators"]})

    # ---- split assignment ----------------------------------------------------
    # Pairs are strictly speaker-disjoint (guaranteed by select_final_pairs), so
    # every pair is its own component and splits can be carved at pair level.
    # Validation takes two components, preferring attached genuine material and
    # generator diversity; the remaining pairs go to training.  Genuine-only
    # components are assigned afterwards wherever they are needed.
    val_components: list[dict[str, Any]] = []
    if pair_components:
        ranked = sorted(
            pair_components,
            key=lambda c: (0 if c["genuine_by_speaker"] else 1, len(c["synth_rows"]), c["key"]),
        )
        val_components.append(ranked[0])
        taken_generators = set(val_components[0]["generators"])
        second = next(
            (c for c in ranked[1:] if set(c["generators"]) - taken_generators),
            ranked[1] if len(ranked) > 1 else None,
        )
        if second is not None:
            val_components.append(second)
    val_keys = {component["key"] for component in val_components}
    for component in pair_components:
        component["side"] = "val" if component["key"] in val_keys else "train"

    # ---- synthetic window selection ------------------------------------------
    used_groups: dict[tuple[str, int], int] = {}
    used_free: dict[str, str] = {}
    selection: dict[str, dict[str, list[dict[str, Any]]]] = {"train": {}, "val": {}}
    diagnostics: dict[str, Any] = {
        "val_components": [component["key"] for component in val_components],
        "available_generators": available_generators,
    }

    def _speaker_set(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        return {str(row["speaker"]) for row in rows}

    def _top_up_genuine(
        rows: list[dict[str, Any]],
        side: str,
        *,
        target: int,
        min_speakers: int,
        per_speaker_cap: int,
    ) -> list[dict[str, Any]]:
        """Fill genuine deficits from free-agent speakers, spreading speakers."""
        current = _speaker_set(rows)
        for speaker in sorted(free_pool, key=lambda s: (-len(free_pool[s]), s)):
            if len(rows) >= target and len(current) >= min_speakers:
                break
            if speaker in used_free or speaker in current:
                continue
            budget = target - len(rows)
            if budget <= 0:
                continue
            take = min(per_speaker_cap, budget)
            if len(current) < min_speakers:
                take = min(take, max(1, math.ceil(budget / 2)))
            bucket = sorted(
                free_pool[speaker],
                key=lambda r: (
                    0 if (str(r["shard"]), int(r["row_group"])) in used_groups else 1,
                    abs(float(r["duration"]) - 5.0),
                ),
            )[:take]
            for row in bucket:
                group = (str(row["shard"]), int(row["row_group"]))
                used_groups[group] = used_groups.get(group, 0) + 1
            rows.extend(bucket)
            current.add(speaker)
            used_free[speaker] = side
        return rows

    free_pool: dict[str, list[dict[str, Any]]] = {}
    for component in free_components:
        for speaker, bucket in component["genuine_by_speaker"].items():
            free_pool[speaker] = list(bucket)

    # Validation genuine first (smaller, needs speaker spread), then training.
    val_attached: dict[str, list[dict[str, Any]]] = defaultdict(list)
    train_attached: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for component in pair_components:
        pool = val_attached if component["side"] == "val" else train_attached
        for speaker, bucket in component["genuine_by_speaker"].items():
            pool[speaker].extend(bucket)

    val_total_speakers = len(val_attached) + len(free_pool)
    val_genuine, _ = _select_genuine(
        val_attached, target=val_genuine_target,
        per_speaker_cap=min_val_speakers if val_total_speakers >= min_val_speakers else genuine_cap,
        used_shards_inventory=used_groups, seed=f"{cfg.seed}:{language}:val:genuine",
    )
    val_genuine = _top_up_genuine(
        val_genuine, "val", target=val_genuine_target, min_speakers=min_val_speakers,
        per_speaker_cap=max(1, math.ceil(val_genuine_target / min_val_speakers)),
    )
    train_genuine, _ = _select_genuine(
        train_attached, target=train_genuine_target, per_speaker_cap=genuine_cap,
        used_shards_inventory=used_groups, seed=f"{cfg.seed}:{language}:train:genuine",
    )
    train_genuine = _top_up_genuine(
        train_genuine, "train", target=train_genuine_target, min_speakers=min_train_speakers,
        per_speaker_cap=genuine_cap,
    )

    for side, synth_target in (("train", train_synth_target), ("val", val_synth_target)):
        side_components = [component for component in pair_components if component["side"] == side]
        synth_pool = [row for component in side_components for row in component["synth_rows"]]
        per_component_cap = (
            synth_target if len(side_components) <= 1
            else max(1, math.ceil(synth_target / len(side_components)))
        )
        synth_selected, _ = _select_synthetic(
            synth_pool, target=synth_target, source_cap=synth_source_cap,
            target_cap=synth_target_cap, seed=f"{cfg.seed}:{language}:{side}:synth",
            per_component_cap=per_component_cap)
        if len(synth_selected) < synth_target:
            # Soft top-up: per-component balance matters, but a side must still
            # reach its quota from rows that remain globally admissible.
            chosen_indices = {int(row["row_index"]) for row in synth_selected}
            remaining_pool = [row for row in synth_pool if int(row["row_index"]) not in chosen_indices]
            source_use = Counter(
                str(row["declared_source_speaker"])
                for row in synth_selected
                if row.get("declared_source_speaker")
            )
            target_use = Counter(str(row["declared_target_speaker"]) for row in synth_selected)
            component_use = Counter(
                str(row.get("_component")) for row in synth_selected if row.get("_component")
            )
            top_up, _ = _select_synthetic(
                remaining_pool,
                target=synth_target - len(synth_selected),
                source_cap=synth_source_cap,
                target_cap=synth_target_cap,
                seed=f"{cfg.seed}:{language}:{side}:synth:topup",
                per_component_cap=None,
                initial_source_use=source_use,
                initial_target_use=target_use,
                initial_component_use=component_use,
            )
            synth_selected = synth_selected + top_up
        synth_diag = _synthetic_diagnostics(synth_selected)
        side_genuine = val_genuine if side == "val" else train_genuine
        genuine_diag = {
            "windows": len(side_genuine),
            "speakers": len(_speaker_set(side_genuine)),
            "row_groups": len({(row["shard"], row["row_group"]) for row in side_genuine}),
            "attached_windows": sum(1 for row in side_genuine if str(row["speaker"]) in (val_attached if side == "val" else train_attached)),
            "free_windows": sum(1 for row in side_genuine if used_free.get(str(row["speaker"])) == side),
        }
        selection[side] = {"synthetic": synth_selected, "genuine": side_genuine}
        required = min_train_speakers if side == "train" else min_val_speakers
        side_generators = sorted({str(row["generator"]) for row in synth_selected})
        diagnostics[side] = {
            "synthetic": synth_diag,
            "genuine": genuine_diag,
            "generators": side_generators,
            "min_speakers_required": required,
            "checks": {
                "synth_speakers_ok": synth_diag["target_speakers"] >= required,
                "genuine_speakers_ok": genuine_diag["speakers"] >= required,
                "generator_mix_ok": (
                    len(available_generators) <= 1
                    or len(side_generators) >= min(2, len(available_generators))
                ),
            },
        }
    for component in free_components:
        speaker = next(iter(component["genuine_by_speaker"]), "")
        component["side"] = used_free.get(speaker, "unused")
    diagnostics["assignment"] = dict(Counter(
        component["side"] for component in ordered_components
    ))
    diagnostics["free_agents"] = dict(used_free)

    plan = {
        "language": language,
        "pairs": [
            {
                "key": list(key),
                "row_indices": [int(row["row_index"]) for row in bucket],
                "component": str(finder.find(("s", key[1]))),
            }
            for key, bucket in pairs
        ],
        "components": [
            {
                "key": component["key"],
                "side": component["side"],
                "generators": component["generators"],
                "synthetic_rows": [int(row["row_index"]) for row in component["synthetic"]],
                "genuine_speakers": sorted(component["genuine_by_speaker"]),
                "genuine_recordings": [
                    str(row["fname"]) for bucket in component["genuine_by_speaker"].values() for row in bucket
                ],
            }
            for component in ordered_components
        ],
        "selection": {
            side: {
                "synthetic": [int(row["row_index"]) for row in selection[side]["synthetic"]],
                "genuine": [str(row["fname"]) for row in selection[side]["genuine"]],
            }
            for side in ("train", "val")
        },
        "diagnostics": diagnostics,
    }
    return PlanResult(language=language, plan=plan, exclusions=exclusions)


# ---------------------------------------------------------------------------
# Materialization + orchestration
# ---------------------------------------------------------------------------


def _fetch_synth_asset(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    run: SourceRun,
    row: dict[str, Any],
    language: str,
    index: int,
) -> bytes | None:
    """Fetch one selected IndicSynth row's audio asset (refreshing stale URLs)."""
    url = row.get("audio_src")
    for _attempt in range(2):
        if not url:
            refresh_asset_urls(cfg, {index: row}, [index], language)
            url = row.get("audio_src")
            if not url:
                return None
        try:
            payload = httpio.get_bytes(url, timeout=cfg.budget.request_timeout_seconds, max_bytes=20 * 1024 * 1024)
        except httpio.HttpError:
            refresh_asset_urls(cfg, {index: row}, [index], language)
            url = row.get("audio_src")
            continue
        ledger.charge("indicsynth", len(payload))
        run.download_bytes += len(payload)
        return payload
    return None


def _materialize_language(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    run: SourceRun,
    language: str,
    folder: str,
    plan: Mapping[str, Any],
    candidates_by_index: Mapping[int, dict[str, Any]],
    inventory_by_fname: Mapping[str, dict[str, Any]],
    indicynth_revision: str,
    kathbath_revision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Fetch audio for one language's plan and build window/recording records."""
    slug = language.lower()
    registry = HashRegistry()
    windows: list[dict[str, Any]] = []
    recordings: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    observed_records = {row["record_id"] for row in inventory_by_fname.values()}
    release_kb = cfg.release("kathbath")
    release_is = cfg.release("indicsynth")
    preprocessing_version = str(cfg.preprocessing.get("config_version"))

    for side in ("train", "val"):
        for index in plan["selection"][side]["synthetic"]:
            row = candidates_by_index[int(index)]
            window_id = f"indicsynth-{slug}-{int(index):06d}"
            raw_path = cfg.raw_dir / "indicsynth" / slug / f"{int(index):06d}.wav"
            if not (raw_path.exists() and raw_path.stat().st_size > 0):
                try:
                    payload = _fetch_synth_asset(cfg, ledger, run, row, language, int(index))
                except BudgetExceeded as error:
                    exclusions.append(exclusion_row(
                        recording_id=window_id, dataset_id="indicsynth",
                        source_file=f"{language}/train/row-{int(index)}", original_split="train",
                        reason=f"budget_exceeded: {error}"))
                    continue
                if payload is None:
                    exclusions.append(exclusion_row(
                        recording_id=window_id, dataset_id="indicsynth",
                        source_file=f"{language}/train/row-{int(index)}", original_split="train",
                        reason="audio_asset_fetch_failed"))
                    continue
                store_original_bytes(raw_path, payload)
            prepared_path = cfg.prepared_dir / "indicsynth" / slug / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
            except (AudioDecodeError, ValueError) as error:
                exclusions.append(exclusion_row(
                    recording_id=window_id, dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{int(index)}", original_split="train",
                    reason=f"decode_or_window_failed: {error}"))
                continue
            duplicate = registry.check(materialized)
            if duplicate:
                exclusions.append(exclusion_row(
                    recording_id=window_id, dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{int(index)}", original_split="train",
                    reason=duplicate))
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)

            speaker_ids = {
                "source": (
                    str(row["declared_source_speaker"]) if row.get("declared_source_speaker") else None
                ),
                "target": str(row["declared_target_speaker"]),
            }
            source_kind = row.get("source_kind")
            if source_kind is None:
                # Fallback for rows materialized without the normalized view: a
                # row with no source reference is a pure TTS target row.
                if row.get("source_reference"):
                    source_kind = "conversion_source" if row.get("generator") == "freevc24" else "declared_source"
                else:
                    source_kind = "tts_target_only"
            parent_refs = {
                "source_reference": row.get("source_reference"),
                "target_reference": row.get("target_reference"),
                "transcript": row.get("transcript"),
                "source_kind": source_kind,
                "source_parent_verification": (
                    "not_applicable_tts"
                    if source_kind == "tts_target_only"
                    else ("verified_train" if row.get("source_record") in observed_records else "parsed_only")
                ),
                "target_parent_verification": (
                    "verified_train" if row.get("target_record") in observed_records else "parsed_only"
                ),
                "dataset_revision": indicynth_revision,
            }
            recording_record = build_recording_record(
                recording_id=window_id,
                dataset_id="indicsynth",
                source_url=str(release_is.get("source_url")),
                source_file=f"{language}/train/row-{int(index)}",
                original_split="train",
                label=1,
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=row.get("generator"),
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                decoded=materialized.decoded_facts(),
                license_note=str(release_is.get("license")),
                notes="synthetic (generated) example from the joint core selection",
            )
            window_record = build_window_record(
                window_id=window_id,
                recording_id=window_id,
                dataset_id="indicsynth",
                source_url=str(release_is.get("source_url")),
                source_file=f"{language}/train/row-{int(index)}",
                original_split="train",
                pool="core",
                split=side,
                label=1,
                label_source="generated_synthetic (IndicSynth generator field)",
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=row.get("generator"),
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                preprocessing_version=preprocessing_version,
                license_note=str(release_is.get("license")),
                access_status="public",
                status="audio_ready",
                notes="joint core selection; relationship component assigned before windowing",
            )
            recordings.append(recording_record)
            windows.append(window_record)

    genuine_rows = [
        inventory_by_fname[fname]
        for side in ("train", "val")
        for fname in plan["selection"][side]["genuine"]
    ]
    destination = cfg.raw_dir / "kathbath" / folder
    pending_rows = [
        row for row in genuine_rows
        if not ((destination / str(row["fname"])).exists() and (destination / str(row["fname"])).stat().st_size > 0)
    ]
    if pending_rows:
        try:
            fetched = kb.fetch_audio(
                cfg, ledger, folder=folder, rows=pending_rows,
                byte_cap=int(cfg.quota("core").get("kathbath_audio_byte_cap", 250_000_000)),
            )
        except (kb.KathbathError, BudgetExceeded) as error:
            run.note(f"{language}: kathbath audio fetch failed: {error}")
            fetched = kb.FetchResult(stored={}, bytes_read=0, row_groups_read=0)
    else:
        fetched = kb.FetchResult(stored={}, bytes_read=0, row_groups_read=0)
    run.download_bytes += fetched.bytes_read

    for side in ("train", "val"):
        for fname in plan["selection"][side]["genuine"]:
            info = inventory_by_fname[fname]
            window_id = f"kathbath-{slug}-{Path(fname).stem}"
            raw_path = fetched.stored.get(fname) or (cfg.raw_dir / "kathbath" / folder / fname)
            if not (raw_path.exists() and raw_path.stat().st_size > 0):
                exclusions.append(exclusion_row(
                    recording_id=window_id, dataset_id="kathbath",
                    source_file=f"{folder}/{fname}", original_split="train",
                    reason="audio_fetch_missing"))
                continue
            prepared_path = cfg.prepared_dir / "kathbath" / slug / f"{window_id}.wav"
            try:
                materialized = materialize_window(raw_path, prepared_path, policy=cfg.window)
            except (AudioDecodeError, ValueError) as error:
                exclusions.append(exclusion_row(
                    recording_id=window_id, dataset_id="kathbath",
                    source_file=f"{folder}/{fname}", original_split="train",
                    reason=f"decode_or_window_failed: {error}"))
                continue
            duplicate = registry.check(materialized)
            if duplicate:
                exclusions.append(exclusion_row(
                    recording_id=window_id, dataset_id="kathbath",
                    source_file=f"{folder}/{fname}", original_split="train",
                    reason=duplicate))
                prepared_path.unlink(missing_ok=True)
                continue
            registry.register(materialized)

            speaker_ids = {"speaker": str(info["speaker"]), "gender": str(info["gender"])}
            parent_refs = {
                "shard": info["shard"],
                "row_group": int(info["row_group"]),
                "row_index": int(info["row_index"]),
                "upstream_duration_seconds": float(info["duration"]),
                "upstream_split": "train",
                "dataset_revision": kathbath_revision,
            }
            recording_record = build_recording_record(
                recording_id=window_id,
                dataset_id="kathbath",
                source_url=str(release_kb.get("source_url")),
                source_file=f"{folder}/{fname}",
                original_split="train",
                label=0,
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=None,
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                decoded=materialized.decoded_facts(),
                license_note=str(release_kb.get("license")),
                notes="genuine speech; parents kept inside the split's relationship component",
            )
            window_record = build_window_record(
                window_id=window_id,
                recording_id=window_id,
                dataset_id="kathbath",
                source_url=str(release_kb.get("source_url")),
                source_file=f"{folder}/{fname}",
                original_split="train",
                pool="core",
                split=side,
                label=0,
                label_source="official Kathbath corpus (genuine speech)",
                spoken_language=language,
                native_language=None,
                speaker_ids=speaker_ids,
                generator=None,
                generator_version=None,
                parent_refs=parent_refs,
                original_audio={**materialized.decoded.original_facts(), "sha256": materialized.original_sha256},
                prepared_audio=materialized.prepared_facts(cfg.dataset_root),
                window=materialized.window_facts(original_rate=materialized.decoded.original_rate),
                preprocessing_version=preprocessing_version,
                license_note=str(release_kb.get("license")),
                access_status="public_with_terms",
                status="audio_ready",
                notes="joint core selection; upstream train shard only",
            )
            recordings.append(recording_record)
            windows.append(window_record)

    return windows, recordings, exclusions


def _ensure_inventory(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    language: str,
    folder: str,
    speakers_of_interest: set[str],
    *,
    min_speakers: int,
    min_recordings: int,
    min_shards: int,
    max_shards: int,
    workers: int,
) -> list[dict[str, Any]]:
    """Progressive, resumable Kathbath metadata scan for one language."""
    joint_dir = cfg.staging_dir / JOINT_DIRNAME
    inventory_path = joint_dir / f"kathbath_inventory.{language}.jsonl"
    shards_path = joint_dir / f"kathbath_scanned.{language}.json"
    rows = read_jsonl(inventory_path)
    by_fname = {str(row["fname"]): row for row in rows}
    scanned: list[str] = json.loads(shards_path.read_text()) if shards_path.exists() else []
    scanned_set = set(scanned)

    revision = kb.dataset_revision()
    splits = kb.list_split_files(cfg, folder, revision=revision)
    train_shards = [s for s in splits.get("train", []) if s not in scanned_set]

    def sufficient() -> bool:
        found = {str(r["speaker"]) for r in by_fname.values() if str(r["speaker"]) in speakers_of_interest}
        recordings_found = sum(1 for r in by_fname.values() if str(r["speaker"]) in speakers_of_interest)
        return len(found) >= min_speakers and recordings_found >= min_recordings

    while train_shards and len(scanned) < max_shards and (len(scanned) < min_shards or not sufficient()):
        batch = train_shards[: min(workers, len(train_shards))]
        train_shards = train_shards[len(batch) :]
        try:
            new_rows, errors = kb.scan_shards(cfg, ledger, folder=folder, shard_names=batch, workers=len(batch))
        except BudgetExceeded as error:
            print(f"  [{language}] kathbath scan stopped: {error}")
            break
        for row in new_rows:
            by_fname.setdefault(str(row["fname"]), row)
        scanned.extend(batch)
        write_jsonl_atomic(inventory_path, [by_fname[k] for k in sorted(by_fname)])
        write_json_atomic(shards_path, sorted(scanned))
        if errors:
            print(f"  [{language}] shard errors: {errors[:2]}")
    return [by_fname[k] for k in sorted(by_fname)]


def build_core(
    cfg: PrepConfig,
    ledger: DownloadLedger,
    *,
    languages: Sequence[str] | None = None,
    workers: int = 6,
    force_scan: bool = False,
    plan_only: bool = False,
) -> dict[str, Any]:
    """Build the joint core pool for the requested languages into staging."""
    cfg.ensure_dirs()
    joint_dir = cfg.staging_dir / JOINT_DIRNAME
    joint_dir.mkdir(parents=True, exist_ok=True)
    core_stage = cfg.staging_dir / "core"
    core_stage.mkdir(parents=True, exist_ok=True)
    quota = cfg.quota("core")
    folders = dict(cfg.quota("kathbath").get("folder_by_language", {}))
    defaults = [
        "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
        "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
    ]
    selected = list(languages or quota.get("languages", defaults))
    min_speakers = int(quota.get("kathbath_min_speakers", 7))
    min_recordings = int(quota.get("kathbath_min_recordings", 30))
    min_shards = int(quota.get("kathbath_min_shards", 4))
    max_shards = int(quota.get("kathbath_max_shards", 8))
    n_pairs = int(quota.get("candidate_pairs", 8))
    max_rows_per_pair = int(quota.get("max_rows_per_pair", 5))

    from datetime import datetime, timezone

    revisions = {
        "indicsynth": _hf_dataset_sha("vdivyasharma/IndicSynth"),
        "kathbath": kb.dataset_revision(),
        "resolved_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json_atomic(joint_dir / "revisions.json", revisions)

    run = SourceRun(source_id="core")
    all_windows: list[dict[str, Any]] = []
    all_recordings: list[dict[str, Any]] = []
    all_exclusions: list[dict[str, Any]] = []
    per_language: dict[str, Any] = {}

    for language in selected:
        folder = folders.get(language)
        if not folder:
            run.note(f"{language}: no Kathbath folder mapping; skipped")
            continue
        print(f"== core {language} ==", flush=True)
        try:
            candidates = scan_indicsynth_candidates(cfg, ledger, language, force=force_scan)
            heldout_path = joint_dir / f"kathbath_heldout.{language}.json"
            if heldout_path.exists():
                held_out = set(json.loads(heldout_path.read_text()))
            else:
                held_out = kb.valid_recording_ids(cfg, ledger, folder=folder, workers=workers)
                write_json_atomic(heldout_path, sorted(held_out))
            lineage_excluded_rows: dict[int, list[str]] = {}
            lineage_file = joint_dir / f"lineage_excluded.{language}.json"
            for item in read_json(lineage_file, default=[]) or []:
                lineage_excluded_rows[int(item["row_index"])] = [
                    str(reason) for reason in (item.get("reasons") or ["lineage_unresolved"])
                ]
            resolution_cache = read_json(
                joint_dir / f"lineage_resolution.{language}.json", default={}
            ) or {}
            eligible, base_exclusions = filter_eligible(
                candidates, language, held_out,
                excluded_rows=lineage_excluded_rows,
                resolution_cache=resolution_cache,
            )
            initial_pairs, _initial_dropped = select_final_pairs(
                eligible, set(),
                n_pairs=n_pairs, max_rows_per_pair=max_rows_per_pair,
                seed=f"{cfg.seed}:{language}:pairs",
            )
            speakers_of_interest = {
                str(row["declared_source_speaker"]) for _, bucket in initial_pairs for row in bucket
                if row.get("declared_source_speaker")
            } | {
                str(row["declared_target_speaker"]) for _, bucket in initial_pairs for row in bucket
            }
            inventory = _ensure_inventory(
                cfg, ledger, language, folder, speakers_of_interest,
                min_speakers=min_speakers, min_recordings=min_recordings,
                min_shards=min_shards, max_shards=max_shards, workers=workers,
            )
            inventory_speakers = {str(row["speaker"]) for row in inventory}
            final_pairs, dropped = select_final_pairs(
                eligible, inventory_speakers,
                n_pairs=n_pairs, max_rows_per_pair=max_rows_per_pair,
                seed=f"{cfg.seed}:{language}:pairs:final",
            )
            if not final_pairs:
                run.note(f"{language}: no co-located pairs found; falling back to initial pair set")
                final_pairs = initial_pairs
            for row in dropped:
                base_exclusions.append(exclusion_row(
                    recording_id=f"indicsynth-{language.lower()}-{int(row['row_index']):06d}",
                    dataset_id="indicsynth",
                    source_file=f"{language}/train/row-{int(row['row_index'])}",
                    original_split="train",
                    reason="candidate_pool_not_selected",
                    generator=row.get("generator"),
                ))
            inventory_by_fname = {str(row["fname"]): row for row in inventory}
            result = plan_language(
                cfg, language, candidates, inventory, held_out,
                pairs=final_pairs, base_exclusions=base_exclusions,
            )
            write_json_atomic(joint_dir / f"plan.{language}.json", result.plan)
            diagnostics = result.plan["diagnostics"]
            all_exclusions.extend(result.exclusions)
            if plan_only:
                per_language[language] = {"plan": diagnostics, "materialized": None}
                continue
            candidates_by_index = {int(row["row_index"]): dict(row) for row in candidates}
            # Prefer the normalized (lineage-checked) view of each candidate row so
            # provenance fields such as ``source_kind`` survive into materialization.
            candidates_by_index.update({int(row["row_index"]): dict(row) for row in eligible})
            windows, recordings, exclusions = _materialize_language(
                cfg, ledger, run, language, folder, result.plan,
                candidates_by_index, inventory_by_fname,
                revisions["indicsynth"], revisions["kathbath"],
            )
            all_windows.extend(windows)
            all_recordings.extend(recordings)
            all_exclusions.extend(exclusions)
            per_language[language] = {
                "plan": diagnostics,
                "materialized": {
                    "windows": len(windows),
                    "train_synthetic": sum(1 for w in windows if w["split"] == "train" and w["label"] == 1),
                    "train_genuine": sum(1 for w in windows if w["split"] == "train" and w["label"] == 0),
                    "val_synthetic": sum(1 for w in windows if w["split"] == "val" and w["label"] == 1),
                    "val_genuine": sum(1 for w in windows if w["split"] == "val" and w["label"] == 0),
                },
            }
            print(
                f"  materialized: {per_language[language]['materialized']}",
                flush=True,
            )
        except BudgetExceeded as error:
            run.note(f"{language}: stopped - {error}")
            per_language[language] = {"error": f"budget_exceeded: {error}"}
            break
        except kb.KathbathError as error:
            run.note(f"{language}: kathbath error - {error}")
            per_language[language] = {"error": f"kathbath: {error}"}

    staged = {
        "source_windows.indicsynth.jsonl": [w for w in all_windows if w["dataset_id"] == "indicsynth"],
        "source_recordings.indicsynth.jsonl": [r for r in all_recordings if r["dataset_id"] == "indicsynth"],
        "source_exclusions.indicsynth.jsonl": dedupe_exclusions(
            [e for e in all_exclusions if e.get("dataset_id") == "indicsynth"]
        ),
        "source_windows.kathbath.jsonl": [w for w in all_windows if w["dataset_id"] == "kathbath"],
        "source_recordings.kathbath.jsonl": [r for r in all_recordings if r["dataset_id"] == "kathbath"],
        "source_exclusions.kathbath.jsonl": dedupe_exclusions(
            [e for e in all_exclusions if e.get("dataset_id") == "kathbath"]
        ),
    }

    # Merge with any previously staged languages so an incremental ``core`` run
    # (e.g. processing one language later) never drops earlier results.  For the
    # languages processed in THIS invocation the new rows REPLACE the previous
    # ones (a re-planned language must not accumulate stale selections whose
    # row ids changed), while untouched languages are preserved exactly.
    processed_languages = {str(language) for language in per_language}
    merged_totals: dict[str, int] = {}
    for name, rows in staged.items():
        key = "window_id" if "windows" in name else "recording_id"
        combined: dict[str, dict[str, Any]] = {}
        if (core_stage / name).exists():
            for row in read_jsonl(core_stage / name):
                if str(row.get("spoken_language")) in processed_languages:
                    continue  # replaced by this run's fresh selection
                combined[str(row.get(key))] = row
        for row in rows:
            combined[str(row.get(key))] = row
        ordered = [combined[identifier] for identifier in sorted(combined)]
        staged[name] = ordered
        write_jsonl_atomic(core_stage / name, ordered)
        if "windows" in name:
            merged_totals[name] = len(ordered)

    existing_summary = read_json(core_stage / "build-summary.json", default={}) or {}
    existing_languages = dict(existing_summary.get("languages") or {})
    existing_languages.update(per_language)

    all_staged_windows = list(staged["source_windows.indicsynth.jsonl"]) + list(
        staged["source_windows.kathbath.jsonl"]
    )
    summary = {
        "languages": dict(sorted(existing_languages.items())),
        "totals": {
            "windows": len(all_staged_windows),
            "train_synthetic": sum(1 for w in all_staged_windows if w["split"] == "train" and w["label"] == 1),
            "train_genuine": sum(1 for w in all_staged_windows if w["split"] == "train" and w["label"] == 0),
            "val_synthetic": sum(1 for w in all_staged_windows if w["split"] == "val" and w["label"] == 1),
            "val_genuine": sum(1 for w in all_staged_windows if w["split"] == "val" and w["label"] == 0),
            "exclusions": len(staged["source_exclusions.indicsynth.jsonl"])
            + len(staged["source_exclusions.kathbath.jsonl"]),
            "download_bytes": int((existing_summary.get("totals") or {}).get("download_bytes", 0))
            + run.download_bytes,
        },
        "notes": run.notes,
        "revisions": revisions,
    }
    write_json_atomic(core_stage / "build-summary.json", summary)
    return summary


_HF_SHA_CACHE: dict[str, str] = {}


def _hf_dataset_sha(dataset: str) -> str:
    if dataset not in _HF_SHA_CACHE:
        from huggingface_hub import HfApi

        info = HfApi().dataset_info(dataset)
        _HF_SHA_CACHE[dataset] = str(info.sha or "unknown")
    return _HF_SHA_CACHE[dataset]


def promote_core(cfg: PrepConfig) -> dict[str, Any]:
    """Validate staged core artifacts and atomically swap them into manifests/."""
    core_stage = cfg.staging_dir / "core"
    summary_path = core_stage / "build-summary.json"
    if not summary_path.exists():
        raise kb.KathbathError("no staged core build found; run the core stage first")
    summary = json.loads(summary_path.read_text())

    problems: list[str] = []
    for name in (
        "source_windows.indicsynth.jsonl",
        "source_recordings.indicsynth.jsonl",
        "source_exclusions.indicsynth.jsonl",
        "source_windows.kathbath.jsonl",
        "source_recordings.kathbath.jsonl",
        "source_exclusions.kathbath.jsonl",
    ):
        if not (core_stage / name).exists():
            problems.append(f"missing staged file {name}")
    rows = read_jsonl(core_stage / "source_windows.indicsynth.jsonl") + read_jsonl(
        core_stage / "source_windows.kathbath.jsonl"
    )
    for row in rows:
        prepared = cfg.dataset_root / str(row["prepared_audio"]["path"])
        if not prepared.exists():
            problems.append(f"missing prepared file {prepared}")
            if len(problems) > 10:
                break
    if problems:
        raise kb.KathbathError("staged core build incomplete: " + "; ".join(problems[:10]))

    for name in (
        "source_windows.indicsynth.jsonl",
        "source_recordings.indicsynth.jsonl",
        "source_exclusions.indicsynth.jsonl",
        "source_windows.kathbath.jsonl",
        "source_recordings.kathbath.jsonl",
        "source_exclusions.kathbath.jsonl",
    ):
        (core_stage / name).replace(cfg.manifests_dir / name)

    # Prune orphaned prepared files from the previous IndicSynth core selection.
    referenced: set[str] = set()
    for path in sorted(cfg.manifests_dir.glob("source_windows.*.jsonl")):
        for row in read_jsonl(path):
            prepared = (row.get("prepared_audio") or {}).get("path")
            if prepared:
                referenced.add(str(cfg.dataset_root / prepared))
    pruned = 0
    for prepared in (cfg.prepared_dir / "indicsynth").rglob("*.wav"):
        if str(prepared) not in referenced:
            prepared.unlink()
            pruned += 1
    return {"promoted_windows": len(rows), "pruned_prepared_files": pruned}


__all__ = [
    "JOINT_DIRNAME",
    "PlanResult",
    "build_core",
    "filter_eligible",
    "plan_language",
    "promote_core",
    "refresh_asset_urls",
    "scan_indicsynth_candidates",
]
