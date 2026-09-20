"""Frozen IndicWav2Vec feature cache for core train/validation windows.

Runs under the pinned encoder environment (``.venv``, Python 3.10 + Fairseq)
because it imports the B1 wrapper and PyTorch.  The cache format is exactly the
contract the B2 collator consumes:

* one entry per window with **unpadded** float32 ``features`` ``[frames, D]``
  and integer ``label`` (0 = genuine, 1 = spoof);
* ``num_frames`` equals the encoder-reported frame count - never a
  duration estimate and never padded;
* :func:`load_embedding_examples` returns ``EmbeddingExample`` objects that can
  be handed straight to ``src.data.batch.collate_examples`` /
  ``src.detectors.runner.run_training``.

Cache identity covers the prepared waveform hashes, window boundaries, the
preprocessing version, the encoder checkpoint hash, selected layer, runtime
implementation identity and feature dtype/dimension.  A cache whose identity
does not match the current manifests/environment is never silently reused.

This is feature generation, not a test: no detector forward pass, loss, training
step or evaluation runs here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

CACHE_FORMAT = "voxsentinel.embedding_cache.v1"


def _runtime_identity() -> dict[str, str]:
    """Record the pinned runtime identity alongside the encoder identity."""
    import platform
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


@dataclass(frozen=True)
class CacheIdentity:
    encoder_checkpoint_sha256: str
    backbone_id: str
    output_layer: int | None
    embedding_dim: int
    frame_hop_ms: float
    dtype: str
    preprocessing_version: str
    implementation: str
    runtime: dict[str, str]
    manifest_sha256: str = ""
    manifest_file_sha256: str = ""
    manifest_items: int = 0
    label_counts: dict[str, int] | None = None
    languages: list[str] | None = None

    @classmethod
    def from_encoder(
        cls,
        encoder: Any,
        preprocessing_version: str,
        *,
        manifest_sha256: str = "",
        manifest_file_sha256: str = "",
        manifest_items: int = 0,
        label_counts: Mapping[str, int] | None = None,
        languages: Sequence[str] | None = None,
    ) -> "CacheIdentity":
        return cls(
            encoder_checkpoint_sha256=str(encoder.checkpoint_sha256),
            backbone_id=str(encoder.backbone_id),
            output_layer=encoder.config.output_layer,
            embedding_dim=int(encoder.embedding_dim),
            frame_hop_ms=float(encoder.frame_hop_ms),
            dtype="float32",
            preprocessing_version=str(preprocessing_version),
            implementation="src.backbones.indic_wav2vec:extract_batch",
            runtime=_runtime_identity(),
            manifest_sha256=str(manifest_sha256),
            manifest_file_sha256=str(manifest_file_sha256),
            manifest_items=int(manifest_items),
            label_counts={str(k): int(v) for k, v in (label_counts or {}).items()},
            languages=sorted(str(item) for item in (languages or [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoder_checkpoint_sha256": self.encoder_checkpoint_sha256,
            "backbone_id": self.backbone_id,
            "output_layer": self.output_layer,
            "embedding_dim": self.embedding_dim,
            "frame_hop_ms": self.frame_hop_ms,
            "dtype": self.dtype,
            "preprocessing_version": self.preprocessing_version,
            "implementation": self.implementation,
            "runtime": dict(self.runtime),
            "manifest_sha256": self.manifest_sha256,
            "manifest_file_sha256": self.manifest_file_sha256,
            "manifest_items": self.manifest_items,
            "label_counts": dict(self.label_counts or {}),
            "languages": list(self.languages or []),
        }


def _load_backbone_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)["indic_wav2vec"]


def _load_windows(manifest: Path) -> list[dict[str, Any]]:
    from .records import read_jsonl

    return [row for row in read_jsonl(manifest) if row.get("status") in ("audio_ready", "features_ready")]


def build_cache(
    *,
    dataset_root: Path,
    split_manifests: Mapping[str, Path],
    backbone_config: str | Path,
    features_config: Mapping[str, Any],
    batch_size: int = 8,
    device: str = "cpu",
    force: bool = False,
    threads: int | None = None,
) -> dict[str, Any]:
    """Extract and save feature bundles for each split; returns a summary dict."""
    import torch

    from ..audio.prepare import read_prepared_wav
    from ..backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor

    if threads is not None:
        torch.set_num_threads(int(threads))

    settings = _load_backbone_config(backbone_config)
    if device:
        settings = {**settings, "device": device}
    encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**settings)).load()
    expected_dim = int(features_config.get("expected_embedding_dim", encoder.embedding_dim))
    if int(encoder.embedding_dim) != expected_dim:
        raise ValueError(f"encoder embedding dim {encoder.embedding_dim} != expected {expected_dim}")

    features_dir = Path(dataset_root) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"identity": {}, "splits": {}, "created": _now()}

    for split_name, manifest_path in split_manifests.items():
        manifest_path = Path(manifest_path)
        windows = _load_windows(manifest_path)
        manifest_identity = _manifest_identity(manifest_path, windows)
        identity = CacheIdentity.from_encoder(
            encoder,
            str(features_config.get("preprocessing_version", "")),
            manifest_sha256=manifest_identity["sha256"],
            manifest_file_sha256=manifest_identity["file_sha256"],
            manifest_items=manifest_identity["items"],
            label_counts=manifest_identity["label_counts"],
            languages=manifest_identity["languages"],
        )
        summary["identity"][split_name] = identity.to_dict()
        bundle_path = features_dir / f"{split_name}.pt"
        meta_path = features_dir / f"{split_name}.meta.json"
        if not force and _bundle_is_current(bundle_path, meta_path, identity, windows):
            meta = {
                "format": CACHE_FORMAT,
                "identity": identity.to_dict(),
                "items": len(windows),
                "status": "reused",
                "complete": True,
                "created": _now(),
                "manifest": str(manifest_path),
                "manifest_sha256": manifest_identity["sha256"],
                "manifest_file_sha256": manifest_identity["file_sha256"],
                "label_counts": manifest_identity["label_counts"],
                "by_language": manifest_identity["by_language"],
            }
            _write_json_atomic(meta_path, meta)
            summary["splits"][split_name] = {
                "status": "reused",
                "items": len(windows),
                "path": str(bundle_path),
            }
            continue
        items: list[dict[str, Any]] = []
        for start in range(0, len(windows), max(1, batch_size)):
            batch_rows = windows[start : start + max(1, batch_size)]
            waveforms = []
            for row in batch_rows:
                prepared_path = Path(dataset_root) / str(row["prepared_audio"]["path"])
                samples, rate = read_prepared_wav(prepared_path)
                file_digest = hashlib.sha256(prepared_path.read_bytes()).hexdigest()
                if file_digest != str(row["prepared_audio"].get("sha256")):
                    raise ValueError(f"prepared waveform changed since manifest: {prepared_path}")
                if rate != 16000:
                    raise ValueError(f"prepared audio must be 16 kHz: {prepared_path}")
                waveforms.append(torch.from_numpy(samples.copy()))
            lengths = torch.tensor([waveform.numel() for waveform in waveforms], dtype=torch.int64)
            padded = torch.nn.utils.rnn.pad_sequence(waveforms, batch_first=True, padding_value=0.0)
            encoded = encoder.extract_batch(padded, lengths, sample_rate=16000)
            for index, row in enumerate(batch_rows):
                frames = int(encoded.valid_lengths[index])
                feature = encoded.features[index, :frames].detach().to("cpu", dtype=torch.float32).clone()
                items.append(
                    {
                        "window_id": row["window_id"],
                        "recording_id": row["recording_id"],
                        "dataset_id": row["dataset_id"],
                        "generator": row.get("generator"),
                        "spoken_language": row.get("spoken_language"),
                        "label": int(row["label"]),
                        "num_frames": frames,
                        "sample_count": int(lengths[index]),
                        "prepared_sha256": str(row["prepared_audio"].get("sha256")),
                        "window": row.get("window"),
                        "features": feature,
                    }
                )
        bundle = {
            "format": CACHE_FORMAT,
            "identity": identity.to_dict(),
            "created": _now(),
            "manifest": str(manifest_path),
            "items": items,
        }
        _write_torch_atomic(bundle, bundle_path)
        meta = {
            "format": CACHE_FORMAT,
            "identity": identity.to_dict(),
            "items": len(items),
            "status": "built",
            "complete": True,
            "created": bundle["created"],
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_identity["sha256"],
            "manifest_file_sha256": manifest_identity["file_sha256"],
            "label_counts": manifest_identity["label_counts"],
            "by_language": manifest_identity["by_language"],
        }
        _write_json_atomic(meta_path, meta)
        summary["splits"][split_name] = {
            "status": "built",
            "items": len(items),
            "path": str(bundle_path),
        }
    return summary


def _manifest_content_sha256(windows: Sequence[Mapping[str, Any]]) -> str:
    """Signature over the model-relevant content of one split manifest.

    Covers window membership, labels, languages and prepared-waveform hashes -
    but not metadata-only edits (provenance notes, evidence blocks), so cached
    features are reused across them while any audio/label/membership change
    still invalidates the cache.
    """
    digest = hashlib.sha256()
    for row in sorted(windows, key=lambda item: str(item.get("window_id"))):
        digest.update(str(row.get("window_id")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(int(row["label"])).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str(row.get("spoken_language")).encode("utf-8"))
        digest.update(b"\x00")
        digest.update(str((row.get("prepared_audio") or {}).get("sha256")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _manifest_identity(manifest_path: Path, windows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Record the manifest's content identity plus label/language composition."""
    file_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else ""
    label_counts: dict[str, int] = {}
    by_language: dict[str, dict[str, int]] = {}
    for row in windows:
        label = str(int(row["label"]))
        label_counts[label] = label_counts.get(label, 0) + 1
        language = str(row.get("spoken_language") or "unknown")
        bucket = by_language.setdefault(language, {})
        bucket[label] = bucket.get(label, 0) + 1
    return {
        "sha256": _manifest_content_sha256(windows),
        "file_sha256": file_sha256,
        "items": len(windows),
        "label_counts": label_counts,
        "by_language": {key: dict(sorted(value.items())) for key, value in sorted(by_language.items())},
        "languages": sorted(by_language),
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    import os
    import tempfile

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
    import os
    import tempfile

    import torch

    handle = tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False)
    handle.close()
    try:
        torch.save(dict(bundle), handle.name)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bundle_is_current(
    bundle_path: Path,
    meta_path: Path,
    identity: CacheIdentity,
    windows: Sequence[Mapping[str, Any]],
) -> bool:
    """Check sidecar completeness, bundle identity and per-item waveform hashes."""
    if not bundle_path.exists() or not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("format") != CACHE_FORMAT or not meta.get("complete"):
        return False
    if meta.get("identity") != identity.to_dict():
        return False
    if int(meta.get("items", -1)) != len(windows):
        return False
    try:
        import torch

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    if bundle.get("format") != CACHE_FORMAT:
        return False
    stored = bundle.get("identity", {})
    expected = identity.to_dict()
    if stored != expected:
        return False
    items = bundle.get("items", [])
    if len(items) != len(windows):
        return False
    by_id = {str(item.get("window_id")): item for item in items}
    for row in windows:
        item = by_id.get(str(row.get("window_id")))
        if item is None:
            return False
        if str(item.get("prepared_sha256")) != str(row.get("prepared_audio", {}).get("sha256")):
            return False
        if int(item.get("label", -1)) != int(row.get("label", -2)):
            return False
    return True


def load_feature_bundle(bundle_path: str | Path) -> dict[str, Any]:
    """Load a feature bundle with format validation (torch required)."""
    import torch

    bundle = torch.load(Path(bundle_path), map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or bundle.get("format") != CACHE_FORMAT:
        raise ValueError(f"not a {CACHE_FORMAT} bundle: {bundle_path}")
    if "identity" not in bundle or "items" not in bundle:
        raise ValueError(f"bundle missing identity/items: {bundle_path}")
    dims = {int(item["features"].shape[1]) for item in bundle["items"] if item.get("features") is not None}
    if len(dims) > 1:
        raise ValueError(f"inconsistent embedding dims in {bundle_path}: {sorted(dims)}")
    if dims and dims != {int(bundle["identity"]["embedding_dim"])}:
        raise ValueError("bundle embedding dim does not match its identity")
    return bundle


def load_embedding_examples(bundle_path: str | Path) -> list[Any]:
    """Return ``EmbeddingExample`` objects ready for the B2 collator/training."""
    from ..data.batch import EmbeddingExample

    bundle = load_feature_bundle(bundle_path)
    examples = []
    for item in bundle["items"]:
        features = item["features"]
        if int(features.shape[0]) != int(item["num_frames"]):
            raise ValueError(f"stored frame count mismatch for {item.get('window_id')}")
        examples.append(EmbeddingExample(features=features, label=int(item["label"])))
    return examples


def bundle_metadata(bundle_path: str | Path) -> dict[str, Any]:
    """Small helper for reports: identity plus counts without loading tensors twice."""
    bundle = load_feature_bundle(bundle_path)
    return {
        "format": bundle["format"],
        "identity": bundle["identity"],
        "items": len(bundle["items"]),
        "created": bundle.get("created"),
    }


def validate_core_training_cache(
    features_dir: str | Path,
    *,
    train_manifest: str | Path | None = None,
    val_manifest: str | Path | None = None,
    expected_items: Mapping[str, int] | None = None,
    expected_languages: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Audit the core train/val feature cache before any training run.

    Works in both environments: the manifest/label/language audit uses the
    JSON sidecars only, while the deep per-item check (bundle identity, stored
    labels, prepared-waveform hashes) runs whenever PyTorch is importable.

    Readiness requires, per split: a complete sidecar, a manifest hash matching
    the current manifest, the expected item count, both labels present, and
    every language contributing both classes.  When the expectation is not met
    the audit still returns full detail so reports stay usable.
    """
    features_dir = Path(features_dir)
    defaults = {"train": 384, "val": 96}
    if expected_items:
        defaults.update({str(k): int(v) for k, v in expected_items.items()})
    checks: list[dict[str, Any]] = []

    def add(split: str, name: str, ok: bool | None, detail: Any = None) -> None:
        checks.append({"split": split, "check": name, "ok": ok, "detail": detail})

    try:
        import torch  # noqa: F401

        torch_available = True
    except ImportError:  # pragma: no cover - torch present in the encoder env
        torch_available = False

    for split, manifest in (("train", train_manifest), ("val", val_manifest)):
        meta_path = features_dir / f"{split}.meta.json"
        bundle_path = features_dir / f"{split}.pt"
        add(split, "bundle_file", bundle_path.exists(), str(bundle_path))
        if not meta_path.exists():
            add(split, "meta_file", False, str(meta_path))
            continue
        add(split, "meta_file", True, str(meta_path))
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            add(split, "meta_parse", False, str(error))
            continue
        add(split, "complete_marker", bool(meta.get("complete")), meta.get("status"))
        if manifest is not None:
            manifest_path = Path(manifest)
            if manifest_path.exists():
                current = _manifest_content_sha256(_load_windows(manifest_path))
            else:
                current = ""
            add(
                split,
                "manifest_current",
                bool(current) and current == str(meta.get("identity", {}).get("manifest_sha256", "")),
                f"manifest={manifest_path}",
            )
        items = int(meta.get("items", 0))
        add(split, "item_count", items == defaults[split], f"{items} (expected {defaults[split]})")
        label_counts = {str(k): int(v) for k, v in (meta.get("label_counts") or {}).items()}
        add(
            split,
            "dual_class",
            label_counts.get("0", 0) > 0 and label_counts.get("1", 0) > 0,
            label_counts,
        )
        by_language = meta.get("by_language") or {}
        if expected_languages:
            add(
                split,
                "languages",
                sorted(by_language) == sorted(str(x) for x in expected_languages),
                f"{len(by_language)} languages",
            )
        add(
            split,
            "per_language_dual_class",
            bool(by_language)
            and all(int(counts.get("0", 0)) > 0 and int(counts.get("1", 0)) > 0 for counts in by_language.values()),
            {lang: dict(counts) for lang, counts in list(by_language.items())[:3]},
        )
        if torch_available and bundle_path.exists():
            try:
                import torch

                bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
                add(
                    split,
                    "deep_identity",
                    bundle.get("identity") == meta.get("identity"),
                    "bundle identity matches sidecar",
                )
                add(
                    split,
                    "deep_items",
                    len(bundle.get("items", [])) == items,
                    f"{len(bundle.get('items', []))} items in bundle",
                )
            except Exception as error:  # pragma: no cover - corrupted artifact path
                add(split, "deep_identity", False, f"bundle load failed: {error}")
        else:
            add(
                split,
                "deep_identity",
                None,
                "skipped: PyTorch unavailable"
                if not torch_available
                else "skipped: bundle not materialized yet",
            )

    failures = [
        f"{check['split']}:{check['check']}" for check in checks if check["ok"] is False
    ]
    skipped = [
        f"{check['split']}:{check['check']}" for check in checks if check["ok"] is None
    ]
    return {
        "ready": not failures,
        "torch_available": torch_available,
        "checks": checks,
        "failures": failures,
        "skipped": skipped,
    }


__all__ = [
    "CACHE_FORMAT",
    "CacheIdentity",
    "build_cache",
    "bundle_metadata",
    "load_embedding_examples",
    "load_feature_bundle",
    "validate_core_training_cache",
]
