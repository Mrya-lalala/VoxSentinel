"""v3 feature-cache audit and loader behavior (synthetic bundles, no encoder).

Every case fabricates a structurally complete but tiny v3 cache under the
module's own identity builder (with a fake encoder object carrying the pinned
fields), then mutates exactly one thing and asserts the audit fails closed.
The real encoder is never loaded and no audio is decoded: checking prepared
audio here means SHA-256 comparison only, which needs no decoding.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from src.dataset_prep.dataset_version import load_dataset_version, sha256_file
from src.dataset_prep.features_v3 import (
    CACHE_FORMAT_V3,
    ENCODER_PINS,
    FeatureCacheError,
    _counts,
    _encode_identity,
    _items_signature,
    _parity_pick,
    audit_v3_cache,
    load_v3_examples,
)

EXTRA_BASE = "artifacts/datasets-v3-coverage"
DEFAULT_BASE = "artifacts/datasets"


class _FakeConfig:
    output_layer = ENCODER_PINS["output_layer"]


class _FakeEncoder:
    checkpoint_sha256 = ENCODER_PINS["checkpoint_sha256"]
    backbone_id = ENCODER_PINS["backbone_id"]
    embedding_dim = ENCODER_PINS["embedding_dim"]
    frame_hop_ms = 20.0
    normalize = True
    config = _FakeConfig()


def _row(window_id: str, label: int, language: str, prepared: str, split: str,
         *, path_base: str | None = None, addition: bool = False) -> dict:
    dataset_id = "kathbath" if label == 0 else "indicsynth"
    revision = "5b9e92849222026d9141acba4e8434fe816396bf" if label == 0 else "c0a10386b723717aff682f757bd67f72983f269f"
    row = {
        "window_id": window_id,
        "recording_id": f"rec-{window_id}",
        "dataset_id": dataset_id,
        "generator": None,
        "spoken_language": language,
        "label": label,
        "split": split,
        "preprocessing_version": "voxsentinel-prep-2",
        "parent_refs": {"dataset_revision": revision},
        "prepared_audio": {"path": prepared, "sha256": sha256_file(Path(__file__))[0:64]},
        "window": {"duration_seconds": 2.0, "sample_rate": 16000},
    }
    if path_base is not None:
        row["path_base"] = path_base
    if addition:
        row["coverage_addition"] = "v3-male-genuine"
    return row


def _write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


@pytest.fixture()
def synth(tmp_path: Path):
    root = tmp_path
    manifests = root / "artifacts/datasets-v3-coverage/manifests"
    train_rows = [
        _row("w1", 0, "Hindi", "prepared/hindi/w1.wav", "train"),
        _row("w2", 1, "Hindi", "prepared/hindi/w2.wav", "train"),
        _row("w3", 0, "Tamil", "prepared/kathbath/tamil/w3.wav", "train", path_base=EXTRA_BASE, addition=True),
    ]
    dev_rows = [
        _row("w4", 0, "Bengali", "prepared/bengali/w4.wav", "val"),
        _row("w5", 1, "Odia", "prepared/kathbath/odia/w5.wav", "val", path_base=EXTRA_BASE, addition=True),
    ]
    _write_manifest(manifests / "windows.train.jsonl", train_rows)
    _write_manifest(manifests / "windows.dev.jsonl", dev_rows)
    _write_manifest(root / "artifacts/datasets-v2/manifests/windows.train.jsonl", [{"window_id": "old"}])
    _write_manifest(root / "artifacts/datasets-v2/manifests/windows.dev.jsonl", [{"window_id": "old-dev"}])
    benchmark = root / "artifacts/datasets-v2/manifests/windows.test.jsonl"
    _write_manifest(benchmark, [{"window_id": "bench-1"}])
    exposure = root / "artifacts/evaluations/gru-v2-test-epoch5/exposure.json"
    exposure.parent.mkdir(parents=True, exist_ok=True)
    exposure.write_text(json.dumps({"speaker_keys": []}), encoding="utf-8")
    spec = {
        "schema": "voxsentinel.dataset_version.v3-coverage.v1",
        "protocol": "speaker_recording_disjoint_v2",
        "strict_conversion_family_compliant": False,
        "revisions": {
            "kathbath": "5b9e92849222026d9141acba4e8434fe816396bf",
            "indicsynth": "c0a10386b723717aff682f757bd67f72983f269f",
        },
        "manifests": {
            "train": {"path": "artifacts/datasets-v3-coverage/manifests/windows.train.jsonl",
                      "sha256": sha256_file(manifests / "windows.train.jsonl"),
                      "windows": 3, "retained": 2, "additions": 1},
            "dev": {"path": "artifacts/datasets-v3-coverage/manifests/windows.dev.jsonl",
                    "sha256": sha256_file(manifests / "windows.dev.jsonl"),
                    "windows": 2, "retained": 1, "additions": 1},
        },
        "retained_sources": {
            "train": {"path": "artifacts/datasets-v2/manifests/windows.train.jsonl",
                      "sha256": sha256_file(root / "artifacts/datasets-v2/manifests/windows.train.jsonl")},
            "dev": {"path": "artifacts/datasets-v2/manifests/windows.dev.jsonl",
                    "sha256": sha256_file(root / "artifacts/datasets-v2/manifests/windows.dev.jsonl")},
        },
        "path_resolution": {"default_base": DEFAULT_BASE, "row_field": "path_base",
                            "bases": {EXTRA_BASE: "v3 additions"},
                            "rule": "rows without path_base resolve under default_base"},
        "benchmark": {
            "manifest": "artifacts/datasets-v2/manifests/windows.test.jsonl",
            "manifest_sha256": sha256_file(benchmark),
            "exposure": "artifacts/evaluations/gru-v2-test-epoch5/exposure.json",
            "exposure_sha256": sha256_file(exposure),
        },
    }
    version_path = manifests / "dataset-version.v3.json"
    version_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    features_dir = root / "artifacts/datasets-v3-coverage/features"
    return root, version_path, features_dir, train_rows, dev_rows


def _make_item(row: dict, split: str, *, frames: int = 3, source: str = "extracted_v3") -> dict:
    return {
        "window_id": row["window_id"],
        "recording_id": row["recording_id"],
        "dataset_id": row["dataset_id"],
        "generator": row["generator"],
        "spoken_language": row["spoken_language"],
        "label": int(row["label"]),
        "num_frames": frames,
        "sample_count": 16000,
        "prepared_sha256": row["prepared_audio"]["sha256"],
        "window": row["window"],
        "split": split,
        "feature_source": source,
        "reused_from": None,
        "features": torch.zeros(frames, ENCODER_PINS["embedding_dim"]),
    }


def _write_cache(root: Path, version_path: Path, features_dir: Path, rows_by_split: dict[str, list[dict]],
                 *, mutate=None) -> None:
    version = load_dataset_version(version_path, repo_root=root)
    encoder = _FakeEncoder()
    features_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in rows_by_split.items():
        items = [_make_item(row, split) for row in rows]
        if mutate is not None:
            mutate(items)
        counts = _counts(items)
        signature = _items_signature(items)
        identity = _encode_identity(encoder, version, split, counts=counts, items_signature=signature)
        bundle = {"format": CACHE_FORMAT_V3, "identity": identity, "created": "t",
                  "manifest": str(version.split(split).manifest_path), "items": items}
        bundle_path = features_dir / f"{split}.pt"
        torch.save(bundle, bundle_path)
        sidecar = {"format": CACHE_FORMAT_V3, "identity": identity, "counts": counts,
                   "items_signature": signature, "items": len(items), "bundle": str(bundle_path),
                   "bundle_sha256": sha256_file(bundle_path), "complete": True, "status": "built",
                   "created": "t", "manifest": str(version.split(split).manifest_path)}
        (features_dir / f"{split}.meta.json").write_text(json.dumps(sidecar, sort_keys=True), encoding="utf-8")


def _audit(root, version_path, features_dir, **kwargs):
    return audit_v3_cache(version_path, features_dir, repo_root=root,
                          check_audio_hashes=False, **kwargs)


def _checks(report) -> set[str]:
    return {failure["check"] for failure in report["failures"]}


def test_valid_synthetic_cache_audits_ready(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    report = _audit(root, version_path, features_dir)
    assert report["ready"], report["failures"]
    assert report["splits"]["train"]["counts"]["windows"] == 3
    assert report["splits"]["dev"]["counts"]["windows"] == 2


def test_changed_label_is_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth

    def mutate(items):
        items[1]["label"] = 0

    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows}, mutate=mutate)
    report = _audit(root, version_path, features_dir)
    assert "train_label_changed" in _checks(report)


def test_membership_mutations_are_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth

    def drop(items):
        items.pop()

    def extra(items):
        items.append(_make_item(_row("w99", 0, "Hindi", "prepared/hindi/w99.wav", "train"), "train"))

    def duplicate(items):
        items[-1] = {**items[-1], "window_id": items[0]["window_id"]}

    def shuffled(items):
        items.reverse()

    for mutate, expected in ((drop, "train_membership_or_order"), (extra, "train_membership_or_order"),
                             (duplicate, "train_duplicate_ids"), (shuffled, "train_membership_or_order")):
        _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows}, mutate=mutate)
        report = _audit(root, version_path, features_dir)
        assert expected in _checks(report), (mutate.__name__, report["failures"])


def test_interrupted_bundle_is_rejected_without_crashing(synth):
    """A partial/corrupt bundle (e.g. interrupted write) must fail closed.

    Construction writes the bundle atomically and only then the completion
    marker, so an interrupted build can never present as valid; a stale bundle
    with a valid sidecar is rejected by hash and unreadability checks.
    """
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    (features_dir / "train.pt").write_bytes(b"partial-bundle")
    report = _audit(root, version_path, features_dir)
    checks = _checks(report)
    assert not report["ready"]
    assert "train_bundle_hash" in checks and "train_bundle_unreadable" in checks


def test_nonfinite_features_are_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth

    def mutate(items):
        items[0]["features"][0, 0] = float("nan")

    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows}, mutate=mutate)
    report = _audit(root, version_path, features_dir)
    assert "train_feature_invalid" in _checks(report)


def test_incomplete_marker_and_v1_sidecar_are_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    sidecar_path = features_dir / "train.meta.json"
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["complete"] = False
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert "train_incomplete" in _checks(_audit(root, version_path, features_dir))

    sidecar["complete"] = True
    sidecar["format"] = "voxsentinel.embedding_cache.v1"  # v1/v2 cache presented as v3
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert "train_sidecar_format" in _checks(_audit(root, version_path, features_dir))


def test_encoder_and_version_identity_mismatch_are_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    sidecar_path = features_dir / "train.meta.json"
    sidecar = json.loads(sidecar_path.read_text())

    sidecar["identity"]["encoder"]["checkpoint_sha256"] = "0" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert "train_encoder_checkpoint_sha256" in _checks(_audit(root, version_path, features_dir))

    sidecar["identity"]["encoder"]["checkpoint_sha256"] = ENCODER_PINS["checkpoint_sha256"]
    sidecar["identity"]["encoder"]["output_layer"] = 23  # layer-23 substitution attempt
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert "train_encoder_output_layer" in _checks(_audit(root, version_path, features_dir))

    sidecar["identity"]["encoder"]["output_layer"] = ENCODER_PINS["output_layer"]
    sidecar["identity"]["dataset_version"]["sha256"] = "f" * 64
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    assert "train_dataset_version_mismatch" in _checks(_audit(root, version_path, features_dir))


def test_bundle_hash_and_audio_change_are_rejected(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    bundle_path = features_dir / "train.pt"
    bundle_path.write_bytes(bundle_path.read_bytes() + b"\x00")
    assert "train_bundle_hash" in _checks(_audit(root, version_path, features_dir))

    # Rebuild cleanly, then drop files whose contents do not match the manifest hash.
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    for row in train_rows + dev_rows:
        base = row.get("path_base", DEFAULT_BASE)
        target = root / base / row["prepared_audio"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"not-the-prepared-audio")
    report = audit_v3_cache(version_path, features_dir, repo_root=root, check_audio_hashes=True)
    checks = _checks(report)
    assert "train_audio_changed" in checks and "dev_audio_changed" in checks

    # A matching SHA-256 clears the audio failure even with no real decoder involved.
    payload = b"matched-bytes"
    digest = __import__("hashlib").sha256(payload).hexdigest()
    rows_fixed = []
    for row in train_rows + dev_rows:
        fixed = {**row, "prepared_audio": {**row["prepared_audio"], "sha256": digest}}
        rows_fixed.append(fixed)
        base = fixed.get("path_base", DEFAULT_BASE)
        target = root / base / fixed["prepared_audio"]["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    # Manifest hashes changed, so regenerate the version record for this sub-case.
    from src.dataset_prep.dataset_version import sha256_file as _sha

    manifests = root / "artifacts/datasets-v3-coverage/manifests"
    _write_manifest(manifests / "windows.train.jsonl", rows_fixed[:3])
    _write_manifest(manifests / "windows.dev.jsonl", rows_fixed[3:])
    spec = json.loads(version_path.read_text())
    spec["manifests"]["train"]["sha256"] = _sha(manifests / "windows.train.jsonl")
    spec["manifests"]["dev"]["sha256"] = _sha(manifests / "windows.dev.jsonl")
    version_path.write_text(json.dumps(spec), encoding="utf-8")
    _write_cache(root, version_path, features_dir, {"train": rows_fixed[:3], "dev": rows_fixed[3:]})
    report = audit_v3_cache(version_path, features_dir, repo_root=root, check_audio_hashes=True)
    assert not {"train_audio_changed", "train_audio_missing"} & _checks(report)


def test_loader_enforces_roles_and_manifest_order(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    for bad_split in ("test", "val", "benchmark"):
        with pytest.raises(FeatureCacheError, match="benchmark|train/dev"):
            load_v3_examples(version_path, features_dir, bad_split, repo_root=root)
    examples, records = load_v3_examples(version_path, features_dir, "dev", repo_root=root)
    assert [record["window_id"] for record in records] == [row["window_id"] for row in dev_rows]
    assert [example.label for example in examples] == [row["label"] for row in dev_rows]
    assert all(record["split"] == "dev" for record in records)


def test_loader_rejects_stale_cache_before_use(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    _write_cache(root, version_path, features_dir, {"train": train_rows, "dev": dev_rows})
    (features_dir / "dev.meta.json").unlink()
    with pytest.raises(FeatureCacheError, match="failed validation"):
        load_v3_examples(version_path, features_dir, "dev", repo_root=root)


def test_parity_pick_is_deterministic_and_spans_languages(synth):
    root, version_path, features_dir, train_rows, dev_rows = synth
    items = {"train": [_make_item(row, "train", frames=2 + index) for index, row in enumerate(train_rows)],
             "dev": [_make_item(row, "dev", frames=5 + index) for index, row in enumerate(dev_rows)]}
    rows = {"train": train_rows, "dev": dev_rows}
    first = _parity_pick(rows, items)
    second = _parity_pick(rows, items)
    assert first == second
    reasons = {pick["reason"] for pick in first}
    # Two retained train rows yield both extremes; single-row buckets yield one pick.
    assert "train_retained_shortest" in reasons and "train_retained_longest" in reasons
    assert "train_addition_shortest" in reasons
    assert "dev_retained_shortest" in reasons and "dev_addition_shortest" in reasons
    assert len({pick["language"] for pick in first}) >= 3
