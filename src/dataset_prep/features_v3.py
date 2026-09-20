"""v3 feature-cache builder, auditor and loader for the expanded coverage dataset.

Runs under the pinned encoder environment (``.venv``, Python 3.10 + Torch
2.2.2 + fairseq) because it drives the frozen IndicWav2Vec extractor.  The
cache format extends the v1 embedding cache:

* one entry per window with **unpadded** float32 ``features`` ``[frames, D]``,
  integer ``label`` (0 = genuine, 1 = spoof) and the item metadata the v1
  format already carried (ids, language, generator, prepared hash, sample
  count, valid frame count);
* per-item ``split`` (normalized ``train``/``dev``), ``feature_source``
  (``reused_v1`` vs ``extracted_v3``) and ``reused_from`` provenance;
* a sidecar/completion marker that binds the bundle to the dataset-version
  record (schema + file hash + protocol), the exact split manifest (declared
  and file hashes), window membership and distributions, prepared-audio
  identities, the pinned encoder/preprocessing/extractor identity and
  runtime, the bundle hash and an explicit completeness status.

Historical caches are never touched.  Unchanged retained windows are reused
per example only after a full identity match (window id, label, language,
prepared waveform hash, preprocessing version, encoder checkpoint hash,
selected layer, embedding dim/dtype and the extraction contract); everything
else is freshly extracted with the same pinned encoder and exact-length
batching the v1 cache builder used.
"""
from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset_version import (
    DatasetVersion,
    DatasetVersionError,
    load_dataset_version,
    load_split_pair,
    load_split_rows,
    resolve_prepared_path,
    sha256_file,
)

CACHE_FORMAT_V3 = "voxsentinel.embedding_cache.v3"
PREPROCESSING_VERSION = "voxsentinel-prep-2"
IMPLEMENTATION = "src.backbones.indic_wav2vec:extract_batch"

# Pins this task is allowed to build with (config + policy, not just config).
ENCODER_PINS: dict[str, Any] = {
    "checkpoint_sha256": "26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59",
    "backbone_id": "indicwav2vec-large",
    "output_layer": None,
    "embedding_dim": 1024,
    "dtype": "float32",
    "preprocessing_version": PREPROCESSING_VERSION,
    "implementation": IMPLEMENTATION,
    "sample_rate": 16000,
}

FEATURE_SOURCES = ("reused_v1", "extracted_v3")
SPLIT_NAMES = ("train", "dev")


class FeatureCacheError(RuntimeError):
    """A hard contract violation while building or loading v3 feature caches."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _runtime_identity() -> dict[str, str]:
    import sys
    from importlib.metadata import PackageNotFoundError, version as package_version

    def _version(name: str) -> str:
        try:
            return package_version(name)
        except (PackageNotFoundError, ValueError):
            return "absent"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _version("torch"),
        "torchaudio": _version("torchaudio"),
        "numpy": _version("numpy"),
        "fairseq": _version("fairseq"),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
    )
    try:
        with handle:
            handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _write_torch_atomic(bundle: Mapping[str, Any], path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    handle.close()
    try:
        torch.save(dict(bundle), handle.name)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def load_pinned_encoder(config_path: str | Path = "configs/backbones.yaml", *, device: str = "cpu"):
    """Load the frozen encoder and enforce the task's pins (no downloads)."""
    import yaml

    from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor

    settings = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))["indic_wav2vec"]
    if settings.get("checkpoint_sha256") != ENCODER_PINS["checkpoint_sha256"]:
        raise FeatureCacheError("configs/backbones.yaml checkpoint_sha256 does not match the pinned encoder")
    encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**{**settings, "device": device})).load()
    actual = {
        "checkpoint_sha256": str(encoder.checkpoint_sha256),
        "backbone_id": str(encoder.backbone_id),
        "output_layer": encoder.config.output_layer,
        "embedding_dim": int(encoder.embedding_dim),
        "preprocessing_version": PREPROCESSING_VERSION,
        "implementation": IMPLEMENTATION,
    }
    for key, expected in ENCODER_PINS.items():
        if key in ("dtype", "sample_rate"):
            continue
        if actual[key] != expected:
            raise FeatureCacheError(f"encoder pin mismatch for {key}: {actual[key]!r} != {expected!r}")
    if bool(getattr(encoder, "normalize", None)) is not True:
        raise FeatureCacheError("encoder lost its checkpoint-declared waveform normalization")
    return encoder


def _encode_identity(encoder, version: DatasetVersion, split: str, *, counts: Mapping[str, Any], items_signature: str) -> dict[str, Any]:
    return {
        "format": CACHE_FORMAT_V3,
        "dataset_version": {
            "schema": version.spec.get("schema"),
            "path": str(version.path),
            "sha256": version.sha256,
            "protocol": version.protocol,
            "strict_conversion_family_compliant": version.strict_conversion_family_compliant,
        },
        "split": split,
        "manifest": {
            "path": str(version.split(split).manifest_path),
            "file_sha256": sha256_file(version.split(split).manifest_path),
            "declared_sha256": version.split(split).declared_sha256,
            "windows": version.split(split).windows,
            "retained": version.split(split).retained,
            "additions": version.split(split).additions,
        },
        "resolution": {
            "default_base": version.default_base,
            "bases": sorted(version.extra_bases),
        },
        "encoder": {
            "checkpoint_sha256": str(encoder.checkpoint_sha256),
            "backbone_id": str(encoder.backbone_id),
            "output_layer": encoder.config.output_layer,
            "embedding_dim": int(encoder.embedding_dim),
            "frame_hop_ms": float(encoder.frame_hop_ms),
            "dtype": ENCODER_PINS["dtype"],
            "preprocessing_version": PREPROCESSING_VERSION,
            "implementation": IMPLEMENTATION,
            "sample_rate": ENCODER_PINS["sample_rate"],
        },
        "runtime": _runtime_identity(),
        "counts": dict(counts),
        "items_signature": items_signature,
    }


def _items_signature(items: Sequence[Mapping[str, Any]]) -> str:
    import hashlib

    digest = hashlib.sha256()
    for item in items:
        digest.update(str(item.get("window_id")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(int(item["label"])).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(item.get("spoken_language")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(item.get("prepared_sha256")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(item.get("feature_source")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(int(item.get("num_frames", -1))).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# v1 reuse source
# --------------------------------------------------------------------------- #

def _reuse_index(reuse_root: Path, *, torch_version: str) -> tuple[dict[str, dict[str, dict]], dict[str, Any]]:
    """Per-example reuse candidates from the historical v1 caches.

    Returns an index keyed ``split -> window_id -> item`` plus an audit record
    describing whether reuse was enabled and why.  A cache whose identity does
    not match the current extraction contract is ignored entirely (reuse is a
    performance path, never a correctness shortcut).
    """
    from .features import load_feature_bundle

    record: dict[str, Any] = {"root": str(reuse_root), "enabled": False, "reason": None, "bundles": {}}
    index: dict[str, dict[str, dict]] = {}
    for split, source_name in (("train", "train"), ("dev", "val")):
        bundle_path = reuse_root / "features" / f"{source_name}.pt"
        sidecar_path = reuse_root / "features" / f"{source_name}.meta.json"
        if not bundle_path.is_file() or not sidecar_path.is_file():
            record["reason"] = f"missing reuse bundle {bundle_path}"
            return index, record
        try:
            bundle = load_feature_bundle(bundle_path)
        except Exception as error:  # noqa: BLE001 - record and disable reuse
            record["reason"] = f"{bundle_path}: {error}"
            return index, record
        identity = bundle.get("identity") or {}
        problems = []
        expected = {
            "encoder_checkpoint_sha256": ENCODER_PINS["checkpoint_sha256"],
            "output_layer": ENCODER_PINS["output_layer"],
            "embedding_dim": ENCODER_PINS["embedding_dim"],
            "dtype": ENCODER_PINS["dtype"],
            "preprocessing_version": ENCODER_PINS["preprocessing_version"],
            "implementation": ENCODER_PINS["implementation"],
        }
        for key, value in expected.items():
            if identity.get(key) != value:
                problems.append(f"{key}: {identity.get(key)!r} != {value!r}")
        runtime = identity.get("runtime") or {}
        if runtime.get("torch") != torch_version:
            problems.append(f"runtime torch {runtime.get('torch')!r} != {torch_version!r}")
        record["bundles"][split] = {
            "path": str(bundle_path),
            "identity_core": {key: identity.get(key) for key in expected},
            "torch": runtime.get("torch"),
            "items": len(bundle.get("items", [])),
            "problems": problems,
        }
        if problems:
            record["reason"] = f"{bundle_path}: " + "; ".join(problems)
            return index, record
        index[split] = {str(item.get("window_id")): item for item in bundle["items"]}
    record["enabled"] = True
    return index, record


def _prepared_hash(path: Path, expected: str, window_id: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise FeatureCacheError(f"{window_id}: prepared audio changed since manifest ({actual} != {expected})")


def _item_from_reuse(row: Mapping[str, Any], item: Mapping[str, Any], *, source: str, split: str) -> dict[str, Any]:
    import torch

    features = item["features"]
    frames = int(features.shape[0])
    return {
        "window_id": str(row["window_id"]),
        "recording_id": row.get("recording_id"),
        "dataset_id": row.get("dataset_id"),
        "generator": row.get("generator"),
        "spoken_language": row.get("spoken_language"),
        "label": int(row["label"]),
        "num_frames": frames,
        "sample_count": int(item.get("sample_count", -1)),
        "prepared_sha256": str(row["prepared_audio"]["sha256"]),
        "window": row.get("window"),
        "split": split,
        "feature_source": "reused_v1",
        "reused_from": source,
        "features": features.detach().to("cpu", dtype=torch.float32).clone(),
    }


def _reuse_features_ok(item: Mapping[str, Any]) -> bool:
    """Per-example reuse contract for a candidate v1 cache item."""
    import torch

    features = item.get("features")
    if features is None or not isinstance(features, torch.Tensor) or features.ndim != 2:
        return False
    frames = int(features.shape[0])
    return bool(
        features.dtype == torch.float32
        and frames > 0
        and int(features.shape[1]) == ENCODER_PINS["embedding_dim"]
        and int(item.get("num_frames", -1)) == frames
        and torch.isfinite(features).all()
        and not features.requires_grad
    )


def _extract_items(encoder, rows: Sequence[Mapping[str, Any]], *, split: str, batch_size: int, device: str) -> list[dict[str, Any]]:
    import numpy as np
    import torch

    from src.audio.prepare import read_prepared_wav

    items: list[dict[str, Any]] = []
    for start in range(0, len(rows), max(1, batch_size)):
        batch_rows = list(rows[start : start + max(1, batch_size)])
        waveforms = []
        lengths: list[int] = []
        for row in batch_rows:
            path = Path(row["resolved_path"])
            if not path.is_file():
                raise FeatureCacheError(f"{row['window_id']}: prepared audio missing: {path}")
            _prepared_hash(path, str(row["prepared_audio"]["sha256"]), str(row["window_id"]))
            samples, rate = read_prepared_wav(path)
            if rate != ENCODER_PINS["sample_rate"]:
                raise FeatureCacheError(f"{row['window_id']}: prepared audio must be 16 kHz")
            waveforms.append(torch.from_numpy(np.ascontiguousarray(samples, dtype=np.float32)))
            lengths.append(int(samples.size))
        lengths_tensor = torch.tensor(lengths, dtype=torch.int64)
        padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
        encoded = encoder.extract_batch(padded, lengths_tensor, sample_rate=ENCODER_PINS["sample_rate"])
        for index, row in enumerate(batch_rows):
            frames = int(encoded.valid_lengths[index])
            feature = encoded.features[index, :frames].detach().to("cpu", dtype=torch.float32).clone()
            if feature.shape != (frames, ENCODER_PINS["embedding_dim"]) or not bool(torch.isfinite(feature).all()):
                raise FeatureCacheError(f"{row['window_id']}: encoder returned invalid features")
            items.append(
                {
                    "window_id": str(row["window_id"]),
                    "recording_id": row.get("recording_id"),
                    "dataset_id": row.get("dataset_id"),
                    "generator": row.get("generator"),
                    "spoken_language": row.get("spoken_language"),
                    "label": int(row["label"]),
                    "num_frames": frames,
                    "sample_count": lengths[index],
                    "prepared_sha256": str(row["prepared_audio"]["sha256"]),
                    "window": row.get("window"),
                    "split": split,
                    "feature_source": "extracted_v3",
                    "reused_from": None,
                    "features": feature,
                }
            )
    return items


def _counts(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_language: dict[str, dict[str, int]] = {}
    genuine = synthetic = reused = extracted = frames = 0
    for item in items:
        label = int(item["label"])
        genuine += int(label == 0)
        synthetic += int(label == 1)
        reused += int(item.get("feature_source") == "reused_v1")
        extracted += int(item.get("feature_source") == "extracted_v3")
        frames += int(item["num_frames"])
        bucket = by_language.setdefault(str(item.get("spoken_language")), {"0": 0, "1": 0})
        bucket[str(label)] = bucket.get(str(label), 0) + 1
    return {
        "windows": len(items),
        "genuine": genuine,
        "synthetic": synthetic,
        "reused_v1": reused,
        "extracted_v3": extracted,
        "valid_frames_total": frames,
        "by_language": {key: dict(sorted(value.items())) for key, value in sorted(by_language.items())},
    }


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build_cache(
    version_path: str | Path,
    features_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    reuse_root: str | Path | None = None,
    backbone_config: str | Path = "configs/backbones.yaml",
    batch_size: int = 8,
    device: str = "cpu",
    threads: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or verify-as-current) the complete v3 train/dev feature bundles."""
    import torch

    started = time.perf_counter()
    if threads is not None:
        torch.set_num_threads(int(threads))
    features_dir = Path(features_dir)
    version = load_dataset_version(version_path, repo_root=repo_root)
    train_rows, dev_rows = load_split_pair(version, repo_root=repo_root)

    report: dict[str, Any] = {
        "schema": "voxsentinel.v3_feature_cache.build.v1",
        "created": _now(),
        "dataset_version": {
            "path": str(version.path),
            "sha256": version.sha256,
            "protocol": version.protocol,
            "strict_conversion_family_compliant": version.strict_conversion_family_compliant,
        },
        "features_dir": str(features_dir),
        "splits": {},
    }

    # Everything that does not need the encoder first: if both bundles are
    # already current, skip loading the model entirely.
    if not force:
        current = audit_v3_cache(version_path, features_dir, repo_root=repo_root, check_audio_hashes=True)
        if current["ready"]:
            report["status"] = "already_current"
            report["audit"] = current
            report["seconds"] = round(time.perf_counter() - started, 3)
            _write_json_atomic(features_dir / "reports" / "cache_build_report.json", report)
            return report

    encoder = load_pinned_encoder(backbone_config, device=device)
    report["encoder"] = {
        "checkpoint_sha256": str(encoder.checkpoint_sha256),
        "output_layer": encoder.config.output_layer,
        "embedding_dim": int(encoder.embedding_dim),
        "frame_hop_ms": float(encoder.frame_hop_ms),
        "preprocessing_version": PREPROCESSING_VERSION,
        "implementation": IMPLEMENTATION,
    }
    report["runtime"] = _runtime_identity()

    reuse_index: dict[str, dict[str, dict]] = {}
    reuse_record: dict[str, Any] = {"enabled": False, "reason": "reuse disabled by request"}
    if reuse_root is not None:
        reuse_index, reuse_record = _reuse_index(Path(reuse_root), torch_version=_runtime_identity()["torch"])
    report["reuse"] = reuse_record

    split_rows = {"train": train_rows, "dev": dev_rows}
    for split in SPLIT_NAMES:
        rows = split_rows[split]
        source_name = "train" if split == "train" else "val"
        source_path = f"{reuse_root}/features/{source_name}.pt" if reuse_root is not None else None
        items: list[dict[str, Any]] = []
        pending: list[Mapping[str, Any]] = []
        reuse_lookup = reuse_index.get(split, {}) if reuse_record.get("enabled") else {}
        reuse_mismatches: list[str] = []
        for row in rows:
            path = Path(row["resolved_path"])
            if not path.is_file():
                raise FeatureCacheError(f"{row['window_id']}: prepared audio missing: {path}")
            _prepared_hash(path, str(row["prepared_audio"]["sha256"]), str(row["window_id"]))
            candidate = reuse_lookup.get(str(row["window_id"])) if not row.get("coverage_addition") else None
            if candidate is not None:
                if (int(candidate.get("label", -1)) == int(row["label"])
                        and str(candidate.get("spoken_language")) == str(row.get("spoken_language"))
                        and str(candidate.get("prepared_sha256")) == str(row["prepared_audio"]["sha256"])
                        and _reuse_features_ok(candidate)):
                    items.append(_item_from_reuse(row, candidate, source=str(source_path), split=split))
                    continue
                reuse_mismatches.append(str(row["window_id"]))
            pending.append(row)
        extracted = _extract_items(encoder, pending, split=split, batch_size=batch_size, device=device) if pending else []
        by_id = {str(item["window_id"]): item for item in items + extracted}
        items = [by_id[str(row["window_id"])] for row in rows]
        counts = _counts(items)
        signature = _items_signature(items)
        identity = _encode_identity(encoder, version, split, counts=counts, items_signature=signature)
        bundle = {
            "format": CACHE_FORMAT_V3,
            "identity": identity,
            "created": _now(),
            "manifest": str(version.split(split).manifest_path),
            "items": items,
        }
        bundle_path = features_dir / f"{split}.pt"
        _write_torch_atomic(bundle, bundle_path)
        sidecar = {
            "format": CACHE_FORMAT_V3,
            "identity": identity,
            "counts": counts,
            "items_signature": signature,
            "items": len(items),
            "bundle": str(bundle_path),
            "bundle_sha256": sha256_file(bundle_path),
            "complete": True,
            "status": "built",
            "created": bundle["created"],
            "manifest": str(version.split(split).manifest_path),
            "per_item_order": "manifest order",
            "reuse_mismatches": reuse_mismatches,
            "notes": [
                "features are unpadded [frames, 1024] float32 tensors, detached and autograd-compatible",
                "retained rows reuse the historical v1 cache only after full per-example identity match",
                "additions are freshly extracted with the pinned encoder (exact-length batching)",
            ],
        }
        _write_json_atomic(features_dir / f"{split}.meta.json", sidecar)
        report["splits"][split] = {
            "status": "built",
            "bundle": str(bundle_path),
            "bundle_sha256": sidecar["bundle_sha256"],
            "counts": counts,
            "items_signature": signature,
            "reuse_mismatches": reuse_mismatches,
        }

    report["status"] = "built"
    report["seconds"] = round(time.perf_counter() - started, 3)
    _write_json_atomic(features_dir / "reports" / "cache_build_report.json", report)
    return report


# --------------------------------------------------------------------------- #
# audit
# --------------------------------------------------------------------------- #

def audit_v3_cache(
    version_path: str | Path,
    features_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    check_audio_hashes: bool = True,
    deep: bool = True,
    splits: Sequence[str] = SPLIT_NAMES,
) -> dict[str, Any]:
    """Validate the v3 feature caches against the version contract.

    Collects every failure instead of raising, so callers can fail closed with
    the complete reason list.  ``deep`` loads the bundles and checks per-item
    contracts; ``check_audio_hashes`` re-hashes every prepared waveform.
    """
    features_dir = Path(features_dir)
    failures: list[dict[str, Any]] = []

    def fail(check: str, detail: Any) -> None:
        failures.append({"check": check, "detail": detail})

    try:
        version = load_dataset_version(version_path, repo_root=repo_root)
        train_rows, dev_rows = load_split_pair(version, repo_root=repo_root)
    except (DatasetVersionError, OSError) as error:
        return {"ready": False, "failures": [{"check": "dataset_version_invalid", "detail": str(error)}]}

    rows_by_split = {"train": train_rows, "dev": dev_rows}
    summaries: dict[str, Any] = {}
    all_ids: dict[str, set[str]] = {}

    for split in splits:
        rows = rows_by_split[split]
        expected_ids = [str(row["window_id"]) for row in rows]
        sidecar_path = features_dir / f"{split}.meta.json"
        bundle_path = features_dir / f"{split}.pt"
        summary: dict[str, Any] = {"windows": len(rows)}
        summaries[split] = summary
        if not sidecar_path.is_file():
            fail(f"{split}_sidecar_missing", str(sidecar_path))
            continue
        if not bundle_path.is_file():
            fail(f"{split}_bundle_missing", str(bundle_path))
            continue
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            fail(f"{split}_sidecar_unreadable", str(error))
            continue
        if sidecar.get("format") != CACHE_FORMAT_V3:
            fail(f"{split}_sidecar_format", f"not a {CACHE_FORMAT_V3} sidecar (v2 cache presented as v3?)")
            continue
        if not sidecar.get("complete") or sidecar.get("status") not in ("built", "verified"):
            fail(f"{split}_incomplete", f"status={sidecar.get('status')!r} complete={sidecar.get('complete')!r}")
            continue
        version_block = (sidecar.get("identity") or {}).get("dataset_version") or {}
        if version_block.get("sha256") != version.sha256:
            fail(f"{split}_dataset_version_mismatch",
                 f"sidecar {version_block.get('sha256')} != current {version.sha256}")
        manifest_block = (sidecar.get("identity") or {}).get("manifest") or {}
        current_manifest_sha = sha256_file(version.split(split).manifest_path)
        if manifest_block.get("file_sha256") != current_manifest_sha:
            fail(f"{split}_manifest_mismatch",
                 f"sidecar {manifest_block.get('file_sha256')} != current {current_manifest_sha}")
        counts = sidecar.get("counts") or {}
        if int(counts.get("windows", -1)) != len(rows):
            fail(f"{split}_count_mismatch", f"{counts.get('windows')} != {len(rows)}")
        if int(sidecar.get("items", -1)) != len(rows):
            fail(f"{split}_item_count", f"{sidecar.get('items')} != {len(rows)}")
        if (sidecar.get("identity") or {}).get("items_signature") != sidecar.get("items_signature"):
            fail(f"{split}_item_signature", "sidecar identity and top-level item signatures differ")
        recorded_bundle_sha = str(sidecar.get("bundle_sha256", ""))
        if not recorded_bundle_sha or sha256_file(bundle_path) != recorded_bundle_sha:
            fail(f"{split}_bundle_hash", "bundle bytes do not match the sidecar hash")
        encoder_block = (sidecar.get("identity") or {}).get("encoder") or {}
        for key, value in (
            ("checkpoint_sha256", ENCODER_PINS["checkpoint_sha256"]),
            ("output_layer", ENCODER_PINS["output_layer"]),
            ("embedding_dim", ENCODER_PINS["embedding_dim"]),
            ("dtype", ENCODER_PINS["dtype"]),
            ("preprocessing_version", ENCODER_PINS["preprocessing_version"]),
            ("implementation", ENCODER_PINS["implementation"]),
        ):
            if encoder_block.get(key) != value:
                fail(f"{split}_encoder_{key}", f"{encoder_block.get(key)!r} != {value!r}")
        summary.update({
            "bundle": str(bundle_path),
            "bundle_sha256": recorded_bundle_sha,
            "counts": counts,
        })
        if check_audio_hashes:
            for row in rows:
                path = resolve_prepared_path(row, version, repo_root=repo_root)
                if not path.is_file():
                    fail(f"{split}_audio_missing", str(path))
                elif sha256_file(path) != str(row["prepared_audio"]["sha256"]):
                    fail(f"{split}_audio_changed", str(row["window_id"]))
        if not deep:
            continue
        try:
            import torch

            bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
        except Exception as error:  # noqa: BLE001
            fail(f"{split}_bundle_unreadable", str(error))
            continue
        if not isinstance(bundle, dict) or bundle.get("format") != CACHE_FORMAT_V3:
            fail(f"{split}_bundle_format", "not a v3 bundle")
            continue
        if bundle.get("identity") != sidecar.get("identity"):
            fail(f"{split}_bundle_identity", "bundle and sidecar identities differ")
        items = bundle.get("items") or []
        item_ids = [str(item.get("window_id")) for item in items]
        if len(set(item_ids)) != len(item_ids):
            fail(f"{split}_duplicate_ids", "bundle contains duplicate window ids")
        if item_ids != expected_ids:
            missing = sorted(set(expected_ids) - set(item_ids))[:5]
            extra = sorted(set(item_ids) - set(expected_ids))[:5]
            fail(f"{split}_membership_or_order",
                 {"missing": missing, "extra": extra, "order_ok": set(item_ids) == set(expected_ids)})
        by_id = {item_id: item for item_id, item in zip(item_ids, items)}
        for row in rows:
            window_id = str(row["window_id"])
            item = by_id.get(window_id)
            if item is None:
                continue
            if int(item.get("label", -1)) != int(row["label"]):
                fail(f"{split}_label_changed", window_id)
            if str(item.get("spoken_language")) != str(row.get("spoken_language")):
                fail(f"{split}_language_changed", window_id)
            if str(item.get("prepared_sha256")) != str(row["prepared_audio"]["sha256"]):
                fail(f"{split}_prepared_hash_changed", window_id)
            if str(item.get("split")) != split:
                fail(f"{split}_item_split", f"{window_id}: {item.get('split')!r}")
            if item.get("feature_source") not in FEATURE_SOURCES:
                fail(f"{split}_feature_source", f"{window_id}: {item.get('feature_source')!r}")
            features = item.get("features")
            if features is None:
                fail(f"{split}_features_missing", window_id)
                continue
            if features.dtype != torch.float32:
                fail(f"{split}_feature_dtype", f"{window_id}: {features.dtype}")
            if features.ndim != 2 or int(features.shape[0]) != int(item.get("num_frames", -1)) \
                    or int(features.shape[1]) != ENCODER_PINS["embedding_dim"]:
                fail(f"{split}_feature_shape", f"{window_id}: {tuple(features.shape)}")
                continue
            if int(item.get("num_frames", 0)) < 1 or not bool(torch.isfinite(features).all()):
                fail(f"{split}_feature_invalid", window_id)
            if bool(features.requires_grad):
                fail(f"{split}_feature_requires_grad", window_id)
        if _items_signature(items) != sidecar.get("items_signature"):
            fail(f"{split}_item_signature", "bundle items do not match the sidecar signature")
        del bundle
        all_ids[split] = set(item_ids)

    if len(splits) == 2:
        overlap = all_ids.get("train", set()) & all_ids.get("dev", set())
        if overlap:
            fail("train_dev_id_overlap", sorted(overlap)[:5])
        benchmark_path = version.benchmark_manifest_path
        benchmark_ids: set[str] = set()
        if benchmark_path.is_file():
            for line in benchmark_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    window_id = json.loads(line).get("window_id")
                    if isinstance(window_id, str):
                        benchmark_ids.add(window_id)
        leaked = (all_ids.get("train", set()) | all_ids.get("dev", set())) & benchmark_ids
        if leaked:
            fail("benchmark_row_in_train_or_dev", sorted(leaked)[:5])

    return {"ready": not failures, "failures": failures, "splits": summaries,
            "dataset_version": {"path": str(version.path), "sha256": version.sha256}}


def load_v3_examples(
    version_path: str | Path,
    features_dir: str | Path,
    split: str,
    *,
    repo_root: str | Path = ".",
) -> tuple[list[Any], list[dict[str, Any]]]:
    """Load one split's training-ready examples and records in manifest order.

    Fails closed: the split must be ``train``/``dev`` (benchmark rows are never
    training inputs), and the cache must pass the deep audit for that split.
    """
    import torch

    from ..data.batch import EmbeddingExample

    if split not in SPLIT_NAMES:
        raise FeatureCacheError(
            f"unknown split {split!r}: the v3 dataset exposes train/dev only; "
            "the evaluated benchmark is never loaded as a training input"
        )
    audit = audit_v3_cache(version_path, features_dir, repo_root=repo_root,
                           check_audio_hashes=False, splits=(split,))
    if not audit["ready"]:
        raise FeatureCacheError(f"{split}: v3 cache failed validation: {audit['failures'][:4]}")
    version = load_dataset_version(version_path, repo_root=repo_root)
    rows = load_split_rows(version, split, repo_root=repo_root)
    bundle = torch.load(Path(features_dir) / f"{split}.pt", map_location="cpu", weights_only=False)
    items = {str(item["window_id"]): item for item in bundle["items"]}
    examples: list[Any] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        window_id = str(row["window_id"])
        item = items.get(window_id)
        if item is None:
            raise FeatureCacheError(f"{split}: {window_id} missing from the validated bundle")
        if int(item["label"]) != int(row["label"]):
            raise FeatureCacheError(f"{split}: {window_id} label mismatch")
        examples.append(EmbeddingExample(features=item["features"], label=int(item["label"])))
        records.append({
            "window_id": window_id,
            "recording_id": row.get("recording_id"),
            "dataset_id": row.get("dataset_id"),
            "generator": row.get("generator"),
            "spoken_language": row.get("spoken_language"),
            "label": int(row["label"]),
            "split": split,
            "resolved_path": str(row["resolved_path"]),
            "prepared_sha256": str(row["prepared_audio"]["sha256"]),
            "num_frames": int(item["num_frames"]),
            "sample_count": int(item["sample_count"]),
            "feature_source": item.get("feature_source"),
            "reused_from": item.get("reused_from"),
        })
    return examples, records


def _parity_pick(split_rows, split_items, *, max_languages: int = 6) -> list[dict[str, Any]]:
    """Deterministic predeclared parity sample spanning kinds, splits and languages."""
    picks: list[dict[str, Any]] = []
    chosen: set[str] = set()

    def add(reason: str, split: str, row_id: str, language: str, frames: int) -> None:
        if row_id in chosen:
            return
        chosen.add(row_id)
        picks.append({"reason": reason, "split": split, "window_id": row_id,
                      "language": language, "num_frames": frames})

    for split in SPLIT_NAMES:
        rows, items = split_rows[split], split_items[split]
        for is_addition in (False, True):
            pairs = [(row, item) for row, item in zip(rows, items)
                     if bool(row.get("coverage_addition")) == is_addition]
            if not pairs:
                continue
            pairs.sort(key=lambda pair: (int(pair[1]["num_frames"]), str(pair[0]["window_id"])))
            kind = "addition" if is_addition else "retained"
            for label, pair in (("shortest", pairs[0]), ("longest", pairs[-1])):
                row, item = pair
                add(f"{split}_{kind}_{label}", split, str(row["window_id"]),
                    str(row.get("spoken_language")), int(item["num_frames"]))

    languages = {pick["language"] for pick in picks}
    if len(languages) < max_languages:
        for split in SPLIT_NAMES:
            rows, items = split_rows[split], split_items[split]
            for row, item in zip(rows, items):
                language = str(row.get("spoken_language"))
                if language in languages:
                    continue
                add(f"language_cover:{language}", split, str(row["window_id"]),
                    language, int(item["num_frames"]))
                languages.add(language)
                if len(languages) >= max_languages:
                    break
            if len(languages) >= max_languages:
                break
    return picks


def run_parity(
    version_path: str | Path,
    features_dir: str | Path,
    *,
    repo_root: str | Path = ".",
    backbone_config: str | Path = "configs/backbones.yaml",
    tolerance: float = 1e-5,
    device: str = "cpu",
) -> dict[str, Any]:
    """Encoder feature parity + autograd-compatibility check for cached tensors.

    A small predeclared sample (retained/addition x train/dev x languages,
    shortest/longest) is re-extracted directly from the prepared audio through
    the frozen encoder's exact-length contract and compared against the cache.

    Tolerance policy: freshly extracted items must reproduce **bit-exactly**
    (error == 0); items reused from the historical v1 cache may drift at unit
    -in-the-last-place level because the CPU reduction order depends on the
    thread count of the original extraction session (observed max 5.2e-06,
    tolerance 1e-5).  This is encoder parity evidence, not classifier
    evaluation.
    """
    import numpy as np
    import torch

    from src.audio.prepare import read_prepared_wav

    features_dir = Path(features_dir)
    audit = audit_v3_cache(version_path, features_dir, repo_root=repo_root, check_audio_hashes=True)
    if not audit["ready"]:
        raise FeatureCacheError(f"parity requires a valid cache: {audit['failures'][:4]}")
    version = load_dataset_version(version_path, repo_root=repo_root)
    train_rows, dev_rows = load_split_pair(version, repo_root=repo_root)
    rows_by_split = {"train": train_rows, "dev": dev_rows}
    items_by_split = {}
    for split in SPLIT_NAMES:
        bundle = torch.load(features_dir / f"{split}.pt", map_location="cpu", weights_only=False)
        items_by_split[split] = bundle["items"]
    encoder = load_pinned_encoder(backbone_config, device=device)

    picks = _parity_pick(rows_by_split, items_by_split)
    lookup = {split: {str(item["window_id"]): item for item in items_by_split[split]} for split in SPLIT_NAMES}
    samples: list[dict[str, Any]] = []
    overall_max = 0.0
    for pick in picks:
        split = pick["split"]
        cached = lookup[split][pick["window_id"]]
        row = next(row for row in rows_by_split[split] if str(row["window_id"]) == pick["window_id"])
        samples_path = Path(row["resolved_path"])
        waveform, rate = read_prepared_wav(samples_path)
        if rate != ENCODER_PINS["sample_rate"]:
            raise FeatureCacheError(f"{row['window_id']}: prepared audio must be 16 kHz")
        tensor = torch.from_numpy(np.ascontiguousarray(waveform, dtype=np.float32))[None]
        batch = encoder.extract_batch(tensor, torch.tensor([waveform.size]), sample_rate=ENCODER_PINS["sample_rate"])
        frames = int(batch.valid_lengths[0])
        direct = batch.features[0, :frames].detach().to("cpu", dtype=torch.float32)
        cache_tensor = cached["features"]
        frames_match = int(cache_tensor.shape[0]) == frames
        error = float(torch.abs(cache_tensor - direct).max()) if frames_match else float("inf")
        overall_max = max(overall_max, error)
        # Autograd compatibility of the *cached* tensor (ordinary detached tensor).
        x = cache_tensor.clone()
        requires_before = bool(x.requires_grad)
        leaf_before = bool(x.is_leaf)
        x.requires_grad_(True)
        loss = (x * x).mean()
        loss.backward()
        autograd_ok = bool(x.grad is not None and torch.isfinite(x.grad).all())
        samples.append({
            **pick,
            "prepared_sha256": cached.get("prepared_sha256"),
            "feature_source": cached.get("feature_source"),
            "cached_frames": int(cache_tensor.shape[0]),
            "direct_frames": frames,
            "frames_match": frames_match,
            "max_abs_error": error,
            "requires_grad_before": requires_before,
            "is_leaf_before": leaf_before,
            "autograd_ok": autograd_ok,
        })
        if requires_before or not leaf_before or not autograd_ok:
            raise FeatureCacheError(f"{row['window_id']}: cached tensor is not autograd-compatible")
        if not frames_match:
            raise FeatureCacheError(f"{row['window_id']}: frame count mismatch {cache_tensor.shape[0]} != {frames}")
        if cached.get("feature_source") == "extracted_v3" and error != 0.0:
            raise FeatureCacheError(
                f"{row['window_id']}: fresh extraction must be bit-exact, got {error}")
        if error > tolerance:
            raise FeatureCacheError(f"{row['window_id']}: parity error {error} > tolerance {tolerance}")

    reuse_errors = [sample["max_abs_error"] for sample in samples if sample["feature_source"] == "reused_v1"]
    fresh_errors = [sample["max_abs_error"] for sample in samples if sample["feature_source"] == "extracted_v3"]
    report = {
        "schema": "voxsentinel.v3_feature_cache.parity.v1",
        "created": _now(),
        "dataset_version": {"path": str(version.path), "sha256": version.sha256},
        "tolerance": tolerance,
        "max_abs_error": overall_max,
        "max_abs_error_reused_v1": max(reuse_errors) if reuse_errors else 0.0,
        "max_abs_error_extracted_v3": max(fresh_errors) if fresh_errors else 0.0,
        "fresh_extraction_bit_exact": all(error == 0.0 for error in fresh_errors),
        "passed": overall_max <= tolerance and all(sample["frames_match"] for sample in samples),
        "samples": samples,
        "languages_covered": sorted({sample["language"] for sample in samples}),
        "encoder": {
            "checkpoint_sha256": str(encoder.checkpoint_sha256),
            "output_layer": encoder.config.output_layer,
            "embedding_dim": int(encoder.embedding_dim),
        },
        "runtime": _runtime_identity(),
        "note": (
            "encoder feature parity for cached tensors; not classifier evaluation. "
            "Fresh (same-session) extractions are bit-exact; reused v1 items may drift at "
            "unit-in-the-last-place level from cross-session CPU reduction order."
        ),
    }
    _write_json_atomic(features_dir / "reports" / "parity.json", report)
    return report
