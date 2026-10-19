"""Packaging checks: a consistent run packages; inconsistent runs are refused.

Fixtures are tiny checkpoints plus JSON metadata written to ``tmp_path``.  The
GRU reference release is a fixture too, so the tests do not depend on the real
release being present in the checkout.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from src.config import ModelConfig
from src.dataset_prep.config import WindowPolicy
from src.dataset_prep.materialize import sha256_file
from src.detectors.checkpoints import load_checkpoint, save_checkpoint
from src.detectors.registry import create_detector
import scripts.package_aasist_release as packaging

REPO_ROOT = Path(__file__).resolve().parents[2]
ENCODER = yaml.safe_load((REPO_ROOT / "configs/backbones.yaml").read_text(encoding="utf-8"))["indic_wav2vec"]
PINNED = "voxsentinel-prep-2"
LANGUAGES = ["bengali", "gujarati", "hindi", "kannada", "malayalam", "marathi",
             "odia", "punjabi", "sanskrit", "tamil", "telugu", "urdu"]
BY_LANGUAGE = {language.title(): {"0": 2, "1": 1} for language in LANGUAGES}
CONFIG = dict(input_dim=8, pooled_bins=4, channels=4, latent_nodes=4,
              gat_dims=[4, 2], feature_standardization=True, num_classes=2)


def _metrics(eer: float) -> dict:
    return {
        "windows": 135, "threshold": 0.5, "accuracy": 0.9, "balanced_accuracy": 0.88,
        "precision": 0.91, "recall": 0.8, "f1": 0.85, "eer": eer,
        "true_negative": 100, "false_positive": 5, "false_negative": 10, "true_positive": 20,
        "genuine_false_alarm_rate": 0.05, "spoof_miss_rate": 0.2,
    }


def _epoch_row(epoch: int, eer: float, accuracy: float) -> dict:
    return {"epoch": epoch, "val_eer": eer, "val_accuracy": accuracy,
            "train_steps": 10, "train_examples": 80}


def _splits(frame_totals: dict, bundles: dict) -> dict:
    def split_entry(window_meta, bundle_key):
        return {**window_meta,
                "bundle": str(bundles[bundle_key]), "bundle_sha256": sha256_file(bundles[bundle_key]),
                "cache_identity": {
                    "encoder": {"checkpoint_sha256": ENCODER["checkpoint_sha256"],
                                "output_layer": ENCODER["output_layer"],
                                "embedding_dim": ENCODER["expected_embedding_dim"],
                                "preprocessing_version": PINNED, "dtype": "float32"},
                    "counts": {"valid_frames_total": frame_totals[bundle_key],
                               "by_language": BY_LANGUAGE}}}

    return {
        "train": split_entry({"windows": 458, "retained": 384, "additions": 74}, "train"),
        "dev": split_entry({"windows": 135, "retained": 96, "additions": 39}, "dev"),
    }


def build_release_fixture(tmp_path, *, selected_epoch: int = 2, eer: float = 0.08,
                          frame_totals: dict | None = None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    frame_totals = frame_totals or {"train": 1000, "dev": 500}
    run = tmp_path / "run"
    run.mkdir()
    bundles = {"train": tmp_path / "train.bin", "dev": tmp_path / "dev.bin"}
    bundles["train"].write_bytes(b"train-cache-bytes")
    bundles["dev"].write_bytes(b"dev-cache-bytes")

    with torch.random.fork_rng():
        torch.manual_seed(5)
        detector = create_detector(ModelConfig("aasist", CONFIG)).eval()
        detector.model.standardizer.fit([SimpleNamespace(features=torch.randn(12, 8), label=0)])
    metrics = _metrics(eer)
    checkpoint_metrics = {key: metrics[key] for key in (
        "eer", "accuracy", "precision", "recall", "f1", "threshold",
        "true_negative", "false_positive", "false_negative", "true_positive",
        "genuine_false_alarm_rate", "spoof_miss_rate")}
    save_checkpoint(run / "aasist_best.pt", detector, "aasist", CONFIG,
                    epoch=selected_epoch, global_step=30, metrics=checkpoint_metrics,
                    metadata={"encoder": {"checkpoint_sha256": ENCODER["checkpoint_sha256"],
                                          "selected_layer": ENCODER["output_layer"],
                                          "embedding_dim": ENCODER["expected_embedding_dim"],
                                          "preprocessing_version": PINNED, "dtype": "float32"}})

    rows = [_epoch_row(1, 0.30, 0.70), _epoch_row(2, 0.20, 0.75), _epoch_row(3, 0.25, 0.72)]
    best = next(row for row in rows if row["epoch"] == selected_epoch)
    for key in ("val_accuracy", "val_precision", "val_recall", "val_f1", "val_threshold",
                "true_negative", "false_positive", "false_negative", "true_positive",
                "genuine_false_alarm_rate", "spoof_miss_rate"):
        best[key] = metrics[key.removeprefix("val_")]
    best["val_eer"] = eer

    (run / "history.json").write_text(json.dumps(rows))
    (run / "metrics.json").write_text(json.dumps(metrics))
    (run / "standardizer.json").write_text(json.dumps({"schema": "voxsentinel.frame_standardizer.v1"}))
    (run / "run_status.json").write_text(json.dumps({
        "status": "completed", "best_checkpoint_reload_max_logit_error": 0.0,
        "final_live_reload_max_logit_error": 0.0, "selected_epoch": selected_epoch,
        "selected_metric": {"name": "eer", "value": eer, "mode": "min", "tie_break": "earliest epoch"},
    }))
    (run / "settings.json").write_text(json.dumps({"resolved": {
        "model": {"name": "aasist", "parameters": CONFIG},
        "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
        "training": {"batch_size": 8, "epochs": 10, "device": "cpu", "use_amp": False,
                     "shuffle": True, "seed": 0, "loss": "cross_entropy"},
        "evaluation": {"threshold": 0.5},
        "checkpoint": {"format_version": 1, "strict_load": True, "directory": "artifacts/aasist",
                       "best_metric": "eer", "best_mode": "min"},
    }}))
    (run / "dataset_identity.json").write_text(json.dumps({
        "dataset_version": {"path": "dataset-version.v3.json", "sha256": "a" * 64,
                            "protocol": "speaker_recording_disjoint_v2",
                            "strict_conversion_family_compliant": False},
        "flag_note": "fixture",
        "splits": _splits(frame_totals, bundles),
        "label_mapping": {"0": "genuine", "1": "synthetic/spoof"},
    }, indent=2))

    gru = tmp_path / "gru-release.json"
    gru.write_text(json.dumps({
        "window": asdict(WindowPolicy()),
        "training_languages": LANGUAGES,
        "training_identity": {"splits": {
            "train": {"counts": {"valid_frames_total": frame_totals["train"]}},
            "dev": {"counts": {"valid_frames_total": frame_totals["dev"]}},
        }},
    }))
    return SimpleNamespace(run=run, gru=gru, metrics=metrics, rows=rows, frame_totals=frame_totals)


def package(monkeypatch, fixture, tmp_path, *, out_name: str = "release"):
    monkeypatch.setattr(packaging, "GRU_RELEASE", fixture.gru)
    out = tmp_path / out_name
    monkeypatch.setattr(packaging.sys, "argv", [
        "package_aasist_release", "--run-dir", str(fixture.run), "--out-dir", str(out),
    ])
    return packaging.main(), out


def test_consistent_run_packages_with_the_strengthened_checks(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    status, out = package(monkeypatch, fixture, tmp_path)

    assert status == 0
    spec = json.loads((out / "release.json").read_text(encoding="utf-8"))
    assert spec["head"]["sha256"] == sha256_file(fixture.run / "aasist_best.pt")
    assert spec["development_selection"]["selection_recomputed_from_history"] == {
        "epoch": 2, "eer": fixture.metrics["eer"], "matches_reported": True,
    }
    assert spec["cache_generation"]["frame_totals_match_gru_release_identity"] is True
    lines = (out / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    assert all(line.split("  ")[1] in ("release.json", "aasist_best.pt", "RESEARCH_USE.txt") for line in lines)


def test_wrong_reported_epoch_is_refused(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    # Make an earlier epoch the true minimum while everything else still claims
    # epoch 2 (checkpoint and run status agree with each other).
    rows = json.loads((fixture.run / "history.json").read_text())
    rows[0]["val_eer"] = 0.05
    (fixture.run / "history.json").write_text(json.dumps(rows))

    with pytest.raises(SystemExit, match="minimum-EER epoch"):
        package(monkeypatch, fixture, tmp_path)


def test_reported_eer_not_matching_history_is_refused(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    metrics = json.loads((fixture.run / "metrics.json").read_text())
    metrics["eer"] = 0.01  # "better" than anything in the history
    (fixture.run / "metrics.json").write_text(json.dumps(metrics))

    with pytest.raises(SystemExit, match="minimum development EER"):
        package(monkeypatch, fixture, tmp_path)


def test_metrics_disagreeing_with_the_checkpoint_are_refused(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    payload = load_checkpoint(fixture.run / "aasist_best.pt", model_name="aasist")
    payload["metrics"]["accuracy"] = 0.5
    torch.save(payload, fixture.run / "aasist_best.pt")

    with pytest.raises(SystemExit, match="checkpoint metrics disagree"):
        package(monkeypatch, fixture, tmp_path)


def test_frame_total_mismatch_with_the_gru_identity_is_refused(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    gru = json.loads(fixture.gru.read_text())
    gru["training_identity"]["splits"]["train"]["counts"]["valid_frames_total"] = 999
    fixture.gru.write_text(json.dumps(gru))

    with pytest.raises(SystemExit, match="frame totals do not match"):
        package(monkeypatch, fixture, tmp_path)


def test_history_row_disagreeing_with_metrics_is_refused(tmp_path, monkeypatch):
    fixture = build_release_fixture(tmp_path)
    rows = json.loads((fixture.run / "history.json").read_text())
    rows[1]["val_accuracy"] = 0.123
    (fixture.run / "history.json").write_text(json.dumps(rows))

    with pytest.raises(SystemExit, match="history.json disagree"):
        package(monkeypatch, fixture, tmp_path)
