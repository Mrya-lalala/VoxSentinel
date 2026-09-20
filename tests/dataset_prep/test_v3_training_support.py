"""V3 training-support plumbing: counts derive from inputs, not old constants.

These tests bind the runner's dataset identity and loader to the real v3
version record and caches.  They are skipped when the (untracked) v3 artifacts
are not present in the checkout; the synthetic cache tests remain the hermetic
core of the suite.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.dataset_prep.dataset_version import sha256_file

REPO = Path(__file__).resolve().parents[2]
VERSION = REPO / "artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json"
FEATURES = REPO / "artifacts/datasets-v3-coverage/features"

pytestmark = pytest.mark.skipif(
    not VERSION.exists() or not (FEATURES / "train.meta.json").exists(),
    reason="v3 dataset artifacts are not present in this checkout",
)


def test_initial_comparison_settings_match_documented_defaults():
    from src.config import load_config

    from scripts.train_gru_v3_from_cache import (
        DEFAULT_CONFIGS,
        _initial_comparison_check,
        _resolved_settings,
    )

    settings = _resolved_settings(load_config(DEFAULT_CONFIGS), DEFAULT_CONFIGS)
    check = _initial_comparison_check(settings)
    assert check["matches_initial_comparison"], check["mismatches"]
    resolved = settings["resolved"]
    assert resolved["training"]["epochs"] == 10
    assert resolved["training"]["batch_size"] == 8
    assert resolved["training"]["seed"] == 0
    assert resolved["evaluation"]["threshold"] == 0.5
    assert resolved["checkpoint"]["best_metric"] == "eer"
    assert resolved["checkpoint"]["best_mode"] == "min"
    assert resolved["model"]["parameters"]["feature_standardization"] is True
    notes = " ".join(settings["notes"]).lower()
    assert "no class weighting" in notes
    assert "earliest epoch" in notes


def test_dataset_identity_derives_counts_from_version_record():
    from scripts.train_gru_v3_from_cache import _dataset_identity

    identity = _dataset_identity(VERSION, FEATURES, REPO)
    assert identity["dataset_version"]["protocol"] == "speaker_recording_disjoint_v2"
    assert identity["dataset_version"]["strict_conversion_family_compliant"] is False
    assert identity["splits"]["train"]["windows"] == 458
    assert identity["splits"]["train"]["additions"] == 74
    assert identity["splits"]["dev"]["windows"] == 135
    assert identity["splits"]["dev"]["retained"] == 96
    assert identity["splits"]["dev"]["additions"] == 39
    for split in ("train", "dev"):
        bundle = FEATURES / f"{split}.pt"
        assert identity["splits"][split]["bundle_sha256"] == sha256_file(bundle)
        counts = identity["splits"][split]["counts"]
        assert counts["windows"] == identity["splits"][split]["windows"]
        assert counts["genuine"] + counts["synthetic"] == counts["windows"]


def test_v3_dev_plumbing_is_not_the_old_96_window_cache():
    from scripts.train_gru_v3_from_cache import _dataset_identity

    identity = _dataset_identity(VERSION, FEATURES, REPO)
    assert identity["splits"]["dev"]["windows"] != 96
    for split in ("train", "dev"):
        # No v3 split ever resolves to the historical core caches.
        assert "datasets/features/val.pt" not in str(identity["splits"][split]["bundle"])
        assert "datasets/features/train.pt" not in str(identity["splits"][split]["bundle"])


def test_v3_loader_returns_full_splits_in_manifest_order():
    from src.dataset_prep.dataset_version import load_dataset_version, load_split_rows
    from src.dataset_prep.features_v3 import load_v3_examples

    train_examples, train_records = load_v3_examples(VERSION, FEATURES, "train", repo_root=REPO)
    dev_examples, dev_records = load_v3_examples(VERSION, FEATURES, "dev", repo_root=REPO)
    assert len(train_examples) == 458
    assert len(dev_examples) == 135
    assert sum(example.label == 0 for example in dev_examples) == 87
    assert sum(example.label == 1 for example in dev_examples) == 48

    version = load_dataset_version(VERSION, repo_root=REPO)
    expected_dev_order = [row["window_id"] for row in load_split_rows(version, "dev", repo_root=REPO)]
    assert [record["window_id"] for record in dev_records] == expected_dev_order

    assert all(example.features.dtype == torch.float32 for example in dev_examples)
    assert all(example.features.shape[1] == 1024 and example.features.shape[0] > 0
               for example in dev_examples)
    assert all(not example.features.requires_grad for example in dev_examples)
