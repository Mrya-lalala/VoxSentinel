"""Hermetic safeguards for the AASIST v3 runner (no dataset assets needed).

Mirrors ``tests/dataset_prep/test_v3_runner_safeguards.py``: the controlled
experiment must accept exactly the agreed protocol and reject architecture or
execution changes before any asset or fit step is involved.
"""
import copy

from src.config import load_config
from scripts.train_aasist_v3_from_cache import (
    DEFAULT_CONFIGS,
    _initial_comparison_check,
    _resolved_settings,
    checkpoint_encoder_identity,
)


def test_aasist_default_configs_are_base_plus_aasist_overlay():
    assert DEFAULT_CONFIGS == ["configs/base.yaml", "configs/aasist.yaml"]
    settings = _resolved_settings(load_config(DEFAULT_CONFIGS), DEFAULT_CONFIGS)
    assert settings["resolved"]["model"]["name"] == "aasist"


def test_aasist_resolved_raw_config_uses_nested_merge():
    s = _resolved_settings(load_config(DEFAULT_CONFIGS), DEFAULT_CONFIGS)
    assert s["raw_merged_config"]["model"] == s["resolved"]["model"]
    assert s["raw_merged_config"]["training"] == s["resolved"]["training"]


def test_aasist_matches_initial_comparison_and_rejects_changes():
    s = _resolved_settings(load_config(DEFAULT_CONFIGS), DEFAULT_CONFIGS)
    assert _initial_comparison_check(s)["matches_initial_comparison"]
    for key, value in [("shuffle", False), ("use_amp", True), ("device", "cuda")]:
        changed = copy.deepcopy(s)
        changed["resolved"]["training"][key] = value
        assert not _initial_comparison_check(changed)["matches_initial_comparison"]
    for key, value in [("channels", 64), ("latent_nodes", 8), ("pooled_bins", 16)]:
        changed = copy.deepcopy(s)
        changed["resolved"]["model"]["parameters"][key] = value
        assert not _initial_comparison_check(changed)["matches_initial_comparison"]
    changed = copy.deepcopy(s)
    changed["resolved"]["model"]["name"] = "gru"
    assert not _initial_comparison_check(changed)["matches_initial_comparison"]


def test_aasist_checkpoint_identity_matches_predictor_contract():
    source = {"checkpoint_sha256": "digest", "output_layer": None, "embedding_dim": 1024,
              "preprocessing_version": "voxsentinel-prep-2", "dtype": "float32",
              "backbone_id": "extra-cache-field"}
    assert checkpoint_encoder_identity(source) == {"checkpoint_sha256": "digest", "selected_layer": None,
                                                   "embedding_dim": 1024,
                                                   "preprocessing_version": "voxsentinel-prep-2",
                                                   "dtype": "float32"}
