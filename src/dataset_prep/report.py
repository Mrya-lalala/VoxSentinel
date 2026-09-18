"""Preparation report: machine-readable JSON plus a Markdown handoff.

The report is generated from the manifests that were actually written - no
numbers are recomputed from intentions.  It records dataset revisions, source
links/terms, branch/commit, commands actually executed, output paths, actual
counts and durations, download usage, preprocessing/split rules and unresolved
coverage.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .config import PrepConfig
from .records import read_json, read_jsonl, write_json_atomic

POOL_FILES = {
    "core_train": "windows.core_train.jsonl",
    "core_val": "windows.core_val.jsonl",
    "supplementary": "windows.supplementary.jsonl",
    "external_eval": "windows.external_eval.jsonl",
    "fallback_baseline": "windows.fallback_baseline.jsonl",
    "unpaired_candidate": "windows.unpaired_candidate.jsonl",
}


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        ).stdout.strip()
    except OSError:
        return ""


def _repo_state() -> dict[str, Any]:
    status = _git("status", "--porcelain")
    return {
        "branch": _git("branch", "--show-current") or "unknown",
        "commit": _git("rev-parse", "HEAD") or "unknown",
        "uncommitted_changes": [line for line in status.splitlines() if line.strip()],
    }


def _pool_rows(cfg: PrepConfig) -> dict[str, list[dict[str, Any]]]:
    return {
        name: read_jsonl(cfg.manifests_dir / filename)
        for name, filename in POOL_FILES.items()
    }


def _group_stats(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Iterable[str],
) -> dict[str, Any]:
    non_identity = {"gender", "accent", "background"}
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row.get(name) for name in keys)
        entry = grouped.setdefault(
            key,
            {
                "windows": 0,
                "durations_seconds": 0.0,
                "speakers": set(),
                "generators": Counter(),
                "labels": Counter(),
            },
        )
        entry["windows"] += 1
        entry["durations_seconds"] += float((row.get("window") or {}).get("duration_seconds") or 0.0)
        entry["labels"][str(row.get("label_name"))] += 1
        for speaker_key, value in (row.get("speaker_ids") or {}).items():
            if str(speaker_key).lower() in non_identity:
                continue
            values = value if isinstance(value, list) else [value]
            for item in values:
                if item:
                    entry["speakers"].add(str(item))
        if row.get("generator"):
            entry["generators"][str(row["generator"])] += 1
    result: dict[str, Any] = {}
    for key, entry in sorted(grouped.items(), key=lambda kv: str(kv[0])):
        name = "|".join(str(part) for part in key)
        result[name] = {
            "windows": entry["windows"],
            "durations_seconds": round(entry["durations_seconds"], 3),
            "durations_minutes": round(entry["durations_seconds"] / 60.0, 3),
            "speakers": len(entry["speakers"]),
            "generators": dict(entry["generators"]),
            "labels": dict(entry["labels"]),
        }
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_version(cfg: PrepConfig, report: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze a complete, consistent dataset/config version stamp.

    Written atomically next to the report; hashes every pool manifest, every
    per-source manifest and every feature bundle, together with the cache
    identities, preprocessing version and achieved counts, so the exact state
    used for a training run can be reconstructed and verified.
    """
    manifests_dir = cfg.manifests_dir
    manifest_hashes: dict[str, str] = {}
    for pattern in ("windows.*.jsonl", "source_windows.*.jsonl"):
        for path in sorted(manifests_dir.glob(pattern)):
            manifest_hashes[path.name] = _sha256(path)
    features: dict[str, Any] = {}
    for split in ("train", "val"):
        bundle = cfg.features_dir / f"{split}.pt"
        sidecar = cfg.features_dir / f"{split}.meta.json"
        if not bundle.exists():
            continue
        meta = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        features[split] = {
            "bundle_sha256": _sha256(bundle),
            "items": meta.get("items"),
            "identity": meta.get("identity"),
            "label_counts": meta.get("label_counts"),
        }
    return {
        "created": datetime.now(timezone.utc).isoformat(),
        "preprocessing_version": cfg.preprocessing.get("config_version"),
        "seed": cfg.seed,
        "pools": report.get("pools"),
        "manifest_hashes": manifest_hashes,
        "feature_caches": features,
        "lineage_status_counts": (report.get("lineage_resolution") or {}).get("status_counts"),
        "readiness": report.get("readiness"),
    }


def build_report(cfg: PrepConfig) -> dict[str, Any]:
    """Compute the full report dictionary from manifests on disk."""
    pools = _pool_rows(cfg)
    all_pool_rows = [row for rows in pools.values() for row in rows]
    exclusions = read_jsonl(cfg.manifests_dir / "windows.excluded.jsonl")
    gaps = read_jsonl(cfg.manifests_dir / "windows.missing_coverage.jsonl")
    recordings = read_jsonl(cfg.manifests_dir / "recordings.jsonl")
    ledger = read_json(cfg.manifests_dir / "downloads.json", default={}) or {}
    catalog = read_json(cfg.manifests_dir / "source-catalog.json", default={}) or {}
    run_log = read_jsonl(cfg.manifests_dir / "run-log.jsonl")
    core_build = read_json(cfg.staging_dir / "core" / "build-summary.json", default=None)
    features_meta: dict[str, Any] = {}
    for meta_path in sorted(cfg.features_dir.glob("*.meta.json")):
        try:
            features_meta[meta_path.stem.replace(".meta", "")] = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            features_meta[meta_path.stem] = {"status": "unreadable"}

    core_rows = pools["core_train"] + pools["core_val"]
    core_stats = _core_identity_stats(pools["core_train"], pools["core_val"])
    readiness = _readiness(pools, gaps, cfg)
    try:
        from .features import validate_core_training_cache

        cache_validation = validate_core_training_cache(
            cfg.features_dir,
            train_manifest=cfg.manifests_dir / "windows.core_train.jsonl",
            val_manifest=cfg.manifests_dir / "windows.core_val.jsonl",
        )
    except Exception as error:  # pragma: no cover - audit must never break the report
        cache_validation = {"ready": False, "error": f"{type(error).__name__}: {error}"}

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_root": str(cfg.dataset_root),
        "config_path": str(cfg.config_path),
        "seed": cfg.seed,
        "repo": _repo_state(),
        "preprocessing": cfg.preprocessing,
        "window_policy": {
            "max_seconds": cfg.window.max_seconds,
            "min_seconds": cfg.window.min_seconds,
            "scan_hop_seconds": cfg.window.scan_hop_seconds,
            "silence_rms_threshold": cfg.window.silence_rms_threshold,
            "activity_frame_ms": getattr(cfg.window, "activity_frame_ms", None),
            "activity_rms_threshold": getattr(cfg.window, "activity_rms_threshold", None),
            "min_active_fraction": getattr(cfg.window, "min_active_fraction", None),
        },
        "downloaded_bytes": ledger,
        "sources": catalog.get("sources", []),
        "pools": {name: len(rows) for name, rows in pools.items()},
        "recordings_total": len(recordings),
        "exclusions_total": len(exclusions),
        "exclusions_by_reason": dict(
            Counter(str(row.get("reason")) for row in exclusions).most_common()
        ),
        "gaps": gaps,
        "features": features_meta,
        "features_validation": cache_validation,
        "core_build": core_build,
        "core_identity": core_stats,
        "lineage_resolution": _lineage_summary(cfg),
        "preprocessing_audit": _preprocessing_audit(all_pool_rows),
        "dataset_revisions": _revisions_summary(all_pool_rows),
        "readiness": readiness,
        "run_scope_note": "No tests, detector training or evaluation were run.",
        "commands": [
            {"timestamp": row.get("timestamp"), "stage": row.get("stage"), "argv": row.get("argv")}
            for row in run_log
        ],
        "notes": [note for row in run_log for note in (row.get("notes") or [])],
    }

    report["core_by_language_split"] = _group_stats(
        core_rows, keys=("dataset_id", "spoken_language", "split")
    )
    report["core_by_generator"] = _group_stats(core_rows, keys=("dataset_id", "generator", "split"))
    report["unpaired_by_group_split"] = _group_stats(
        pools["unpaired_candidate"], keys=("dataset_id", "native_language", "split")
    )
    report["fallback_by_split_label"] = _group_stats(
        pools["fallback_baseline"], keys=("original_split", "label_name")
    )
    report["supplementary_summary"] = _group_stats(pools["supplementary"], keys=("dataset_id",))
    report["external_eval_summary"] = _group_stats(
        pools["external_eval"], keys=("dataset_id", "native_language")
    )
    report["all_by_pool"] = _group_stats(all_pool_rows, keys=("pool", "dataset_id"))
    return report


_NON_IDENTITY_SPEAKER_KEYS = {"gender", "accent", "background", "age-group", "age_group", "record"}


def _speaker_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key, value in (row.get("speaker_ids") or {}).items():
        if str(key).lower() in _NON_IDENTITY_SPEAKER_KEYS:
            continue
        if value not in (None, ""):
            tokens.add(str(value))
    return tokens


def _reference_tokens(row: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    for key, value in (row.get("parent_refs") or {}).items():
        if key.endswith("reference") and value:
            tokens.add(str(value))
    return tokens


def _core_identity_stats(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Speaker/reference intersections and unique upstream originals for the core pools.

    Speaker ids are only unique per (dataset, language) namespace (Kathbath and
    IndicSynth ids repeat across languages), so the identity key is scoped
    accordingly; upstream reference strings are globally unique as recorded.
    """

    def speakers(rows: Sequence[Mapping[str, Any]]) -> set[tuple[str, str, str]]:
        result: set[tuple[str, str, str]] = set()
        for row in rows:
            for token in _speaker_tokens(row):
                result.add(
                    (str(row.get("dataset_id")), str(row.get("spoken_language")), str(token))
                )
        return result

    def references(rows: Sequence[Mapping[str, Any]]) -> set[str]:
        result: set[str] = set()
        for row in rows:
            result.update(_reference_tokens(row))
        return result

    train_speakers, val_speakers = speakers(train_rows), speakers(val_rows)
    train_refs, val_refs = references(train_rows), references(val_rows)
    languages: dict[str, dict[str, Any]] = {}
    for split_name, rows in (("train", train_rows), ("val", val_rows)):
        for row in rows:
            language = str(row.get("spoken_language") or "unknown")
            bucket = languages.setdefault(language, {"train": Counter(), "val": Counter()})[split_name]
            bucket[str(row.get("label_name"))] += 1
            if row.get("generator"):
                bucket[f"generator:{row['generator']}"] += 1
    languages_view = {
        language: {split: dict(counts) for split, counts in splits.items()}
        for language, splits in sorted(languages.items())
    }
    return {
        "train_windows": len(train_rows),
        "val_windows": len(val_rows),
        "train_unique_speakers": len(train_speakers),
        "val_unique_speakers": len(val_speakers),
        "speaker_overlap": sorted(train_speakers & val_speakers),
        "reference_overlap": sorted(train_refs & val_refs),
        "train_unique_originals": len(train_refs),
        "val_unique_originals": len(val_refs),
        "languages": languages_view,
    }


def _lineage_summary(cfg: PrepConfig) -> dict[str, Any]:
    """Reference-verification status for core synthetic windows plus replacements."""
    rows = read_jsonl(cfg.manifests_dir / "source_windows.indicsynth.jsonl")
    status_counts: Counter[str] = Counter()
    evidence_methods: Counter[str] = Counter()
    for row in rows:
        parent = row.get("parent_refs") or {}
        for role in ("source", "target"):
            status_counts[str(parent.get(f"{role}_parent_verification"))] += 1
            evidence = (parent.get("reference_evidence") or {}).get(role) or {}
            evidence_methods[str(evidence.get("method") or "none")] += 1
    replacements: dict[str, Any] = {}
    for path in sorted(cfg.staging_dir.glob("joint/lineage_excluded.*.json")):
        items = read_json(path, default=[]) or []
        if items:
            replacements[path.stem.split(".", 1)[1]] = items
    exclusion_history: Counter[str] = Counter()
    for row in read_jsonl(cfg.manifests_dir / "windows.excluded.jsonl"):
        reason = str(row.get("reason") or "")
        prefix = reason.split(":", 1)[0].strip()
        if prefix in (
            "lineage_replacement_required",
            "reference_unresolvable_precheck",
            "upstream_held_out_parent",
        ):
            exclusion_history[prefix] += 1
    return {
        "windows": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "evidence_methods": dict(sorted(evidence_methods.items())),
        "replacement_rows": replacements,
        "reference_exclusions_by_reason": dict(sorted(exclusion_history.items())),
        "meaning": {
            "verified_train": "recording confirmed inside the upstream train shards (full scan evidence)",
            "not_applicable_tts": "no voice-conversion source step exists for TTS rows",
            "parsed_only": "pattern-valid only (should be empty for a frozen core)",
            "upstream_eval_parent": "parent lives in the upstream valid/test partition (excluded)",
            "not_found": "absent from every train row group that could contain it (excluded)",
        },
    }


def _preprocessing_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Strict-cap and activity facts across every selected window."""
    full_window = 0
    shorter_than_cap = 0
    active_fractions: list[float] = []
    decoders: Counter[str] = Counter()
    overflow_adjusted: Counter[str] = Counter()
    by_class: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        window = row.get("window") or {}
        samples = int(window.get("num_samples") or 0)
        if samples >= 64000:
            full_window += 1
        else:
            shorter_than_cap += 1
        fraction = window.get("active_fraction")
        if fraction is not None:
            active_fractions.append(float(fraction))
        processing = (row.get("prepared_audio") or {}).get("processing") or {}
        decoder = str((row.get("original_audio") or {}).get("decoder") or processing.get("decoder") or "unknown")
        decoders[decoder] += 1
        gain = processing.get("overflow_gain")
        if gain is not None and float(gain) != 1.0:
            overflow_adjusted[row.get("label_name") or "unknown"] += 1
        by_class[str(row.get("label_name"))]["windows"] += 1
    return {
        "windows_at_full_cap": full_window,
        "windows_shorter_than_cap": shorter_than_cap,
        "active_fraction_min": min(active_fractions) if active_fractions else None,
        "active_fraction_mean": (
            round(sum(active_fractions) / len(active_fractions), 4) if active_fractions else None
        ),
        "decoders": dict(decoders),
        "overflow_gain_adjusted": dict(overflow_adjusted),
        "by_class": {key: dict(value) for key, value in sorted(by_class.items())},
    }


def _revisions_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, int]]:
    """Distinct immutable upstream revisions observed per dataset."""
    result: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        revision = str((row.get("parent_refs") or {}).get("dataset_revision") or "unrecorded")
        result[str(row.get("dataset_id"))][revision] += 1
    return {dataset: dict(revisions) for dataset, revisions in sorted(result.items())}


def _readiness(
    pools: Mapping[str, Sequence[Mapping[str, Any]]],
    gaps: Sequence[Mapping[str, Any]],
    cfg: PrepConfig,
) -> dict[str, Any]:
    """Explicit readiness flags for the two training goals and external audio."""
    core_train = pools["core_train"]
    core_val = pools["core_val"]
    core_languages = {
        str(row.get("spoken_language")) for row in list(core_train) + list(core_val)
    }
    train_labels = Counter(str(row.get("label_name")) for row in core_train)
    val_labels = Counter(str(row.get("label_name")) for row in core_val)
    svarah_windows = [row for row in pools.get("external_eval", []) if row.get("dataset_id") == "svarah"]
    english_gap = next(
        (
            gap
            for gap in gaps
            if (gap.get("source_id") or gap.get("dataset_id")) == "synthetic_english"
        ),
        None,
    )
    core_ok = (
        len(core_train) >= 384
        and len(core_val) >= 96
        and train_labels.get("genuine", 0) > 0
        and train_labels.get("spoof", 0) > 0
        and val_labels.get("genuine", 0) > 0
        and val_labels.get("spoof", 0) > 0
        and len(core_languages) >= 12
    )
    return {
        "core_indic_training_ready": {
            "ready": bool(core_ok),
            "languages": len(core_languages),
            "train_windows": len(core_train),
            "val_windows": len(core_val),
            "train_labels": dict(train_labels),
            "val_labels": dict(val_labels),
        },
        "indian_english_training_ready": {
            "ready": False,
            "reason": (
                english_gap.get("reason")
                if english_gap
                else "no synthetic Indian-English source is paired with the genuine English pool"
            ),
            "user_action": english_gap.get("user_action") if english_gap else None,
        },
        "external_audio_ready": {
            "ready": bool(svarah_windows),
            "windows": len(svarah_windows),
            "sources": sorted({str(row.get("dataset_id")) for row in pools.get("external_eval", [])}),
        },
    }


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    size = float(value)
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    return f"{size:.2f} {units[index]}"


def report_markdown(report: Mapping[str, Any]) -> str:
    """Render the human handoff document (without the final wording pass)."""
    lines: list[str] = []
    lines.append("# VoxSentinel dataset-preparation pilot report")
    lines.append("")
    lines.append(f"Generated: {report['generated_at']}")
    repo = report["repo"]
    lines.append(
        f"Checkout: branch `{repo['branch']}`, commit `{repo['commit']}` "
        f"({len(repo['uncommitted_changes'])} uncommitted files preserved)"
    )
    lines.append(f"Dataset root: `{report['dataset_root']}` (config `{report['config_path']}`)")
    lines.append(f"Seed: {report['seed']}")
    lines.append("")
    lines.append("## Pools (actual window counts)")
    lines.append("")
    lines.append("| Pool | Windows | File |")
    lines.append("| --- | ---: | --- |")
    for name, filename in POOL_FILES.items():
        lines.append(f"| {name} | {report['pools'].get(name, 0)} | `manifests/{filename}` |")
    lines.append(f"| excluded | {report['exclusions_total']} | `manifests/windows.excluded.jsonl` |")
    lines.append(f"| missing_coverage | {len(report['gaps'])} | `manifests/windows.missing_coverage.jsonl` |")
    lines.append("")

    def table(title: str, data: Mapping[str, Mapping[str, Any]]) -> None:
        lines.append(f"## {title}")
        lines.append("")
        if not data:
            lines.append("_none_")
            lines.append("")
            return
        lines.append("| Group | Windows | Minutes | Speakers | Generators | Labels |")
        lines.append("| --- | ---: | ---: | ---: | --- | --- |")
        for name, stats in data.items():
            generators = ", ".join(f"{key}:{value}" for key, value in stats["generators"].items()) or "-"
            labels = ", ".join(f"{key}:{value}" for key, value in stats["labels"].items()) or "-"
            lines.append(
                f"| {name} | {stats['windows']} | {stats['durations_minutes']} | "
                f"{stats['speakers']} | {generators} | {labels} |"
            )
        lines.append("")

    table("Core windows by dataset / language / split", report["core_by_language_split"])
    table("Core windows by generator / split", report["core_by_generator"])
    table("Unpaired candidate windows by native-language group / split", report["unpaired_by_group_split"])
    table("Fallback/baseline windows by official split / label", report["fallback_by_split_label"])

    lines.append("## Download usage")
    lines.append("")
    ledger = report.get("downloaded_bytes") or {}
    lines.append(f"Total new downloads: {_fmt_bytes(ledger.get('total_bytes', 0))} "
                 f"of {_fmt_bytes(ledger.get('cap_bytes', 0))} cap")
    for source, size in sorted((ledger.get("per_source_bytes") or {}).items()):
        lines.append(f"- {source}: {_fmt_bytes(size)}")
    lines.append("")

    lines.append("## Source access states")
    lines.append("")
    lines.append("| Source | Status | Role |")
    lines.append("| --- | --- | --- |")
    for source in report.get("sources", []):
        lines.append(f"| {source.get('source_id')} | {source.get('status')} | {source.get('role')} |")
    lines.append("")

    lines.append("## Readiness")
    lines.append("")
    readiness = report.get("readiness") or {}
    for name, entry in readiness.items():
        detail = ", ".join(
            f"{key}={value}" for key, value in entry.items() if key not in ("reason", "user_action") and not isinstance(value, dict)
        )
        lines.append(f"- `{name}`: ready={entry.get('ready')} ({detail})")
        if entry.get("reason"):
            lines.append(f"  - reason: {entry['reason']}")
        if entry.get("user_action"):
            lines.append(f"  - user action: {entry['user_action']}")
    lines.append("")

    lines.append("## Core composition (identity & cross-split disjointness)")
    lines.append("")
    identity = report.get("core_identity") or {}
    lines.append(
        f"- windows: train {identity.get('train_windows', 0)}, val {identity.get('val_windows', 0)}"
    )
    lines.append(
        f"- unique speakers: train {identity.get('train_unique_speakers', 0)}, "
        f"val {identity.get('val_unique_speakers', 0)}"
    )
    lines.append(
        f"- unique upstream originals: train {identity.get('train_unique_originals', 0)}, "
        f"val {identity.get('val_unique_originals', 0)}"
    )
    lines.append(
        f"- cross-split speaker overlap: {len(identity.get('speaker_overlap', []))}; "
        f"reference overlap: {len(identity.get('reference_overlap', []))}"
    )
    for language, splits in (identity.get("languages") or {}).items():
        train = splits.get("train", {})
        val = splits.get("val", {})
        lines.append(
            f"  - {language}: train {train.get('genuine', 0)}G/{train.get('spoof', 0)}S, "
            f"val {val.get('genuine', 0)}G/{val.get('spoof', 0)}S"
        )
    lines.append("")

    lines.append("## Dataset revisions (immutable references)")
    lines.append("")
    revisions = report.get("dataset_revisions") or {}
    if not revisions:
        lines.append("_none recorded_")
    for dataset, values in sorted(revisions.items()):
        rendered = ", ".join(f"`{sha[:16]}...` x{count}" for sha, count in sorted(values.items()))
        lines.append(f"- {dataset}: {rendered}")
    lines.append("")

    lineage = report.get("lineage_resolution") or {}
    if lineage:
        lines.append("## Lineage resolution (full train-shard scan)")
        lines.append("")
        lines.append(f"- synthetic windows: {lineage.get('windows', 0)}")
        lines.append(f"- verification statuses: {json.dumps(lineage.get('status_counts') or {})}")
        lines.append(f"- evidence methods: {json.dumps(lineage.get('evidence_methods') or {})}")
        replacements = lineage.get("replacement_rows") or {}
        if replacements:
            lines.append(f"- rows replaced for verified alternatives: {json.dumps(replacements)}")
        lines.append("")

    lines.append("## Preprocessing audit (voxsentinel-prep-2)")
    lines.append("")
    audit = report.get("preprocessing_audit") or {}
    lines.append(
        f"- windows at the strict 4 s cap: {audit.get('windows_at_full_cap', 0)}; "
        f"shorter (whole recordings under the cap): {audit.get('windows_shorter_than_cap', 0)}"
    )
    lines.append(
        f"- active fraction: min {audit.get('active_fraction_min')}, mean {audit.get('active_fraction_mean')}"
    )
    decoders = ", ".join(f"{key}:{value}" for key, value in (audit.get("decoders") or {}).items())
    lines.append(f"- decoders: {decoders or '-'}")
    overflow = ", ".join(
        f"{key}:{value}" for key, value in (audit.get("overflow_gain_adjusted") or {}).items()
    )
    lines.append(f"- overflow attenation applied (bounded, ≤ 0 dB): {overflow or 'none'}")
    lines.append("")

    lines.append("## Feature cache")
    lines.append("")
    features = report.get("features") or {}
    if not features:
        lines.append("_No feature bundles were written._")
    for split, meta in sorted(features.items()):
        identity_meta = meta.get("identity", {})
        lines.append(
            f"- `{split}`: {meta.get('items')} items, status {meta.get('status')}, "
            f"encoder {identity_meta.get('encoder_checkpoint_sha256', '?')[:12]}..., "
            f"layer {identity_meta.get('output_layer')}, dim {identity_meta.get('embedding_dim')}"
        )
    validation = report.get("features_validation") or {}
    lines.append(
        f"- training-cache audit: ready={validation.get('ready')}"
        + (f"; failures: {', '.join(validation.get('failures', []))}" if validation.get("failures") else "")
        + (f"; skipped: {', '.join(validation.get('skipped', []))}" if validation.get("skipped") else "")
    )
    lines.append("")

    lines.append("## Missing coverage (not concealed)")
    lines.append("")
    if not report["gaps"]:
        lines.append("_none_")
    else:
        seen: set[tuple[Any, Any]] = set()
        for gap in report["gaps"]:
            key = (gap.get("dataset_id"), gap.get("status"))
            if key in seen:
                continue
            seen.add(key)
            scope = gap.get("spoken_language") or "all languages/classes"
            lines.append(f"- {gap.get('dataset_id')} ({scope}): {gap.get('status')} - {gap.get('reason')}")
            if gap.get("user_action"):
                lines.append(f"  - user action: {gap['user_action']}")
    lines.append("")

    lines.append("## Exclusions")
    lines.append("")
    for reason, count in (report.get("exclusions_by_reason") or {}).items():
        lines.append(f"- {reason}: {count}")
    if not report.get("exclusions_by_reason"):
        lines.append("_none_")
    lines.append("")

    lines.append("## Commands actually executed")
    lines.append("")
    lines.append("```")
    for row in report.get("commands", []):
        lines.append("python -m scripts.prepare_datasets " + " ".join(str(a) for a in row.get("argv", [])))
    lines.append("```")
    lines.append("")
    if report.get("run_scope_note"):
        lines.append(report["run_scope_note"])
        lines.append("")
    return "\n".join(lines)


def write_reports(cfg: PrepConfig) -> dict[str, Any]:
    """Write JSON + Markdown reports and return the JSON payload."""
    report = build_report(cfg)
    version = dataset_version(cfg, report)
    report["dataset_version"] = version
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(cfg.reports_dir / "preparation-report.json", report)
    (cfg.reports_dir / "preparation-report.md").write_text(report_markdown(report), encoding="utf-8")
    write_json_atomic(cfg.manifests_dir / "dataset-version.json", version)
    return report


__all__ = ["build_report", "report_markdown", "write_reports"]
