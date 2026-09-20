"""v3 dataset-version contract: resolution, normalization, fail-closed behavior.

The synthetic fixture builds a miniature but structurally complete v3 version
record (manifests, retained sources, benchmark provenance) so every rejection
path is exercised without touching real artifacts.  One integration test binds
the real ``dataset-version.v3.json`` to its declared counts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.dataset_prep.dataset_version import (
    DatasetVersionError,
    load_dataset_version,
    load_split_pair,
    load_split_rows,
    resolve_prepared_path,
    sha256_file,
)

REPO = Path(__file__).resolve().parents[2]
REAL_VERSION = REPO / "artifacts" / "datasets-v3-coverage" / "manifests" / "dataset-version.v3.json"
EXTRA_BASE = "artifacts/datasets-v3-coverage"
DEFAULT_BASE = "artifacts/datasets"


def _write(path: Path, rows_or_text) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(rows_or_text, str):
        path.write_text(rows_or_text, encoding="utf-8")
    else:
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows_or_text), encoding="utf-8")


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
        "prepared_audio": {"path": prepared, "sha256": f"{window_id:0<64}"[:64].replace(" ", "0")},
        "window": {"duration_seconds": 2.0, "sample_rate": 16000},
    }
    if path_base is not None:
        row["path_base"] = path_base
    if addition:
        row["coverage_addition"] = "v3-male-genuine"
    return row


@pytest.fixture()
def synthetic_v3(tmp_path: Path):
    """Miniature v3 dataset; returns (root, version_path, train_rows, dev_rows)."""
    root = tmp_path
    train_rows = [
        _row("w1", 0, "Hindi", "prepared/hindi/w1.wav", "train"),
        _row("w2", 1, "Hindi", "prepared/hindi/w2.wav", "train"),
        _row("w3", 0, "Tamil", "prepared/kathbath/tamil/w3.wav", "train", path_base=EXTRA_BASE, addition=True),
    ]
    dev_rows = [
        _row("w4", 0, "Hindi", "prepared/hindi/w4.wav", "val"),
        _row("w5", 1, "Odia", "prepared/kathbath/odia/w5.wav", "val", path_base=EXTRA_BASE, addition=True),
    ]
    manifests = root / "artifacts/datasets-v3-coverage/manifests"
    _write(manifests / "windows.train.jsonl", train_rows)
    _write(manifests / "windows.dev.jsonl", dev_rows)
    _write(root / "artifacts/datasets-v2/manifests/windows.train.jsonl", [{"window_id": "old"}])
    _write(root / "artifacts/datasets-v2/manifests/windows.dev.jsonl", [{"window_id": "old-dev"}])
    benchmark = root / "artifacts/datasets-v2/manifests/windows.test.jsonl"
    _write(benchmark, [{"window_id": "bench-1"}])
    exposure = root / "artifacts/evaluations/gru-v2-test-epoch5/exposure.json"
    _write(exposure, json.dumps({"speaker_keys": []}))
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
        "path_resolution": {
            "default_base": DEFAULT_BASE,
            "row_field": "path_base",
            "bases": {EXTRA_BASE: "v3 additions (added rows only)"},
            "rule": "rows without a path_base field resolve under default_base",
        },
        "benchmark": {
            "manifest": "artifacts/datasets-v2/manifests/windows.test.jsonl",
            "manifest_sha256": sha256_file(benchmark),
            "exposure": "artifacts/evaluations/gru-v2-test-epoch5/exposure.json",
            "exposure_sha256": sha256_file(exposure),
        },
    }
    version_path = manifests / "dataset-version.v3.json"
    version_path.write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")
    return root, version_path, train_rows, dev_rows


def test_mixed_bases_resolve_and_split_is_normalized(synthetic_v3):
    root, version_path, _, _ = synthetic_v3
    version = load_dataset_version(version_path, repo_root=root)
    train = load_split_rows(version, "train", repo_root=root)
    dev = load_split_rows(version, "dev", repo_root=root)

    # Retained rows resolve under the default base; additions under their declared base.
    assert train[0]["resolved_path"] == str((root / DEFAULT_BASE / "prepared/hindi/w1.wav").resolve())
    assert train[2]["resolved_path"] == str((root / EXTRA_BASE / "prepared/kathbath/tamil/w3.wav").resolve())
    # Dev rows carry raw split "val" and normalize to "dev".
    assert all(row["resolved_split"] == "dev" for row in dev)
    assert dev[1]["resolved_path"] == str((root / EXTRA_BASE / "prepared/kathbath/odia/w5.wav").resolve())


def test_undeclared_base_and_escape_are_rejected(synthetic_v3):
    root, version_path, _, _ = synthetic_v3
    version = load_dataset_version(version_path, repo_root=root)
    rogue = _row("w9", 0, "Hindi", "prepared/hindi/w9.wav", "train", path_base="artifacts/somewhere-else")
    with pytest.raises(DatasetVersionError, match="undeclared path_base"):
        resolve_prepared_path(rogue, version, repo_root=root)
    escaping = _row("w9", 0, "Hindi", "../outside/w9.wav", "train")
    with pytest.raises(DatasetVersionError, match="safe and relative"):
        resolve_prepared_path(escaping, version, repo_root=root)
    absolute = _row("w9", 0, "Hindi", "/etc/w9.wav", "train")
    with pytest.raises(DatasetVersionError, match="safe and relative"):
        resolve_prepared_path(absolute, version, repo_root=root)


def test_wrong_raw_split_value_is_rejected(synthetic_v3):
    root, version_path, train_rows, _ = synthetic_v3
    bad = list(train_rows)
    bad[0] = {**bad[0], "split": "val"}
    _write(root / "artifacts/datasets-v3-coverage/manifests/windows.train.jsonl", bad)
    spec = json.loads(version_path.read_text())
    spec["manifests"]["train"]["sha256"] = sha256_file(
        root / "artifacts/datasets-v3-coverage/manifests/windows.train.jsonl")
    version_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    with pytest.raises(DatasetVersionError, match="allowed"):
        load_dataset_version(version_path, repo_root=root)


def test_revision_and_preprocessing_mismatch_are_rejected(synthetic_v3):
    root, version_path, train_rows, _ = synthetic_v3
    manifests = root / "artifacts/datasets-v3-coverage/manifests"

    bad_pre = [dict(train_rows[0], preprocessing_version="voxsentinel-prep-1"), *train_rows[1:]]
    _write(manifests / "windows.train.jsonl", bad_pre)
    spec = json.loads(version_path.read_text())
    spec["manifests"]["train"]["sha256"] = sha256_file(manifests / "windows.train.jsonl")
    version_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    version = load_dataset_version(version_path, repo_root=root)
    with pytest.raises(DatasetVersionError, match="preprocessing_version"):
        load_split_rows(version, "train", repo_root=root)

    bad_rev = [dict(train_rows[0], parent_refs={"dataset_revision": "0" * 40}), *train_rows[1:]]
    _write(manifests / "windows.train.jsonl", bad_rev)
    spec["manifests"]["train"]["sha256"] = sha256_file(manifests / "windows.train.jsonl")
    version_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    version = load_dataset_version(version_path, repo_root=root)
    with pytest.raises(DatasetVersionError, match="source revision"):
        load_split_rows(version, "train", repo_root=root)


def test_declared_hash_mismatch_is_rejected(synthetic_v3):
    root, version_path, _, _ = synthetic_v3
    manifest = root / "artifacts/datasets-v3-coverage/manifests/windows.train.jsonl"
    manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
    with pytest.raises(DatasetVersionError, match="file sha256"):
        load_dataset_version(version_path, repo_root=root)


def test_cross_split_overlap_and_benchmark_leak_are_rejected(synthetic_v3):
    root, version_path, train_rows, dev_rows = synthetic_v3
    # Train/dev overlap.
    overlapping_dev = [*dev_rows, _row("w1", 1, "Hindi", "prepared/hindi/w1b.wav", "val")]
    manifests = root / "artifacts/datasets-v3-coverage/manifests"
    _write(manifests / "windows.dev.jsonl", overlapping_dev)
    spec = json.loads(version_path.read_text())
    spec["manifests"]["dev"]["sha256"] = sha256_file(manifests / "windows.dev.jsonl")
    spec["manifests"]["dev"]["windows"] = 3
    spec["manifests"]["dev"]["retained"] = 2
    spec["manifests"]["dev"]["additions"] = 1
    version_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    version = load_dataset_version(version_path, repo_root=root)
    with pytest.raises(DatasetVersionError, match="shared between train and dev"):
        load_split_pair(version, repo_root=root)

    # Benchmark leak.
    _write(manifests / "windows.dev.jsonl", dev_rows)
    spec = json.loads(version_path.read_text())
    spec["manifests"]["dev"]["sha256"] = sha256_file(manifests / "windows.dev.jsonl")
    spec["manifests"]["dev"]["windows"] = 2
    spec["manifests"]["dev"]["retained"] = 1
    spec["manifests"]["dev"]["additions"] = 1
    benchmark_leaking_train = [*train_rows, _row("bench-1", 0, "Hindi", "prepared/hindi/b1.wav", "train")]
    _write(manifests / "windows.train.jsonl", benchmark_leaking_train)
    spec["manifests"]["train"]["sha256"] = sha256_file(manifests / "windows.train.jsonl")
    spec["manifests"]["train"]["windows"] = 4
    spec["manifests"]["train"]["retained"] = 3
    version_path.write_text(json.dumps(spec) + "\n", encoding="utf-8")
    version = load_dataset_version(version_path, repo_root=root)
    with pytest.raises(DatasetVersionError, match="benchmark"):
        load_split_pair(version, repo_root=root)


@pytest.mark.skipif(not REAL_VERSION.exists(), reason="v3 dataset artifacts are not present in this checkout")
def test_real_v3_version_record_binds_declared_counts():
    version = load_dataset_version(REAL_VERSION, repo_root=REPO)
    assert version.protocol == "speaker_recording_disjoint_v2"
    assert version.strict_conversion_family_compliant is False
    assert version.split("train").windows == 458
    assert version.split("train").retained == 384
    assert version.split("train").additions == 74
    assert version.split("dev").windows == 135
    assert version.split("dev").retained == 96
    assert version.split("dev").additions == 39
    train = load_split_rows(version, "train", repo_root=REPO)
    dev = load_split_rows(version, "dev", repo_root=REPO)
    assert {row["resolved_split"] for row in dev} == {"dev"}
    assert len({row["window_id"] for row in train} | {row["window_id"] for row in dev}) == 458 + 135
    additions = [row for row in train + dev if row.get("coverage_addition")]
    assert len(additions) == 113
    assert all(row["path_base"] == "artifacts/datasets-v3-coverage" for row in additions)
    assert all("path_base" not in row for row in train + dev if not row.get("coverage_addition"))
