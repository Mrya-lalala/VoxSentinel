"""AASIST v3 runner identity/loader checks against the real (untracked) assets.

Skipped when the v3 caches are not present in the checkout, exactly like
``tests/dataset_prep/test_v3_training_support.py``.  These bind the AASIST
runner to the same dataset contract the GRU runner uses; they do not train.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from src.dataset_prep.dataset_version import load_dataset_version, load_split_rows
from src.dataset_prep.features_v3 import audit_v3_cache, load_v3_examples

REPO = Path(__file__).resolve().parents[2]
VERSION = REPO / "artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json"
FEATURES = REPO / "artifacts/datasets-v3-coverage/features"

pytestmark = pytest.mark.skipif(
    not VERSION.exists() or not (FEATURES / "train.meta.json").exists(),
    reason="v3 dataset artifacts are not present in this checkout",
)


def test_aasist_cache_audit_is_ready_and_counts_match_the_version_record():
    audit = audit_v3_cache(VERSION, FEATURES, repo_root=REPO, check_audio_hashes=False)
    assert audit["ready"], audit["failures"][:5]
    version = load_dataset_version(VERSION, repo_root=REPO)
    for split in ("train", "dev"):
        counts = audit["splits"][split]["counts"]
        assert counts["windows"] == version.split(split).windows
        assert counts["genuine"] + counts["synthetic"] == counts["windows"]
        assert counts["reused_v1"] + counts["extracted_v3"] == counts["windows"]


def test_aasist_dataset_identity_derives_counts_from_version_record():
    from scripts.train_aasist_v3_from_cache import _dataset_identity

    identity = _dataset_identity(VERSION, FEATURES, REPO)
    assert identity["dataset_version"]["protocol"] == "speaker_recording_disjoint_v2"
    assert identity["splits"]["train"]["windows"] == 458
    assert identity["splits"]["train"]["additions"] == 74
    assert identity["splits"]["dev"]["windows"] == 135
    assert identity["splits"]["dev"]["retained"] == 96
    assert identity["splits"]["dev"]["additions"] == 39
    assert identity["splits"]["train"]["cache_identity"]["encoder"]["output_layer"] is None
    assert identity["splits"]["train"]["cache_identity"]["encoder"]["preprocessing_version"] == "voxsentinel-prep-2"


def test_aasist_loader_returns_detached_full_splits_in_manifest_order():
    train_examples, _ = load_v3_examples(VERSION, FEATURES, "train", repo_root=REPO)
    dev_examples, dev_records = load_v3_examples(VERSION, FEATURES, "dev", repo_root=REPO)
    assert len(train_examples) == 458
    assert len(dev_examples) == 135
    assert sum(example.label == 0 for example in dev_examples) == 87
    assert sum(example.label == 1 for example in dev_examples) == 48
    for example in dev_examples:
        assert example.features.dtype == torch.float32
        assert example.features.shape[1] == 1024
        assert example.features.shape[0] >= 32
        assert not example.features.requires_grad

    version = load_dataset_version(VERSION, repo_root=REPO)
    expected_dev_order = [row["window_id"] for row in load_split_rows(version, "dev", repo_root=REPO)]
    assert [record["window_id"] for record in dev_records] == expected_dev_order
