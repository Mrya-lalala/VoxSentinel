"""Package the trained embedding-adapted AASIST run as a separate v3 release.

Writes a new release directory (default
``artifacts/releases/indic-aasist-v3-research``) containing the trained
checkpoint, a ``release.json`` manifest compatible with
``src.scoring.predict.FilePredictor`` (``head.model_name == "aasist"``), a
checksum file and a research-use/attribution notice.  The GRU release is never
touched or renamed.

Fail-closed checks performed before anything is copied:

* the run completed, with exact live/reload checkpoint parity;
* minimum-EER selection is **recomputed from the epoch history** and must match
  the reported selected epoch, the reported metrics and the checkpoint metrics
  (earliest epoch wins ties); ``metrics.json`` must also agree with the selected
  history row and with the checkpoint's stored metrics;
* the checkpoint is an ``aasist`` head whose config and epoch match the run's
  resolved settings and selection status;
* the checkpoint's training metadata encoder identity equals the pinned
  encoder/preprocessing identity from ``configs/backbones.yaml``;
* the run's dataset identity matches the v3 version record counts and the cache
  bundles exist with their recorded hashes;
* the frame totals are **compared against** the GRU release's recorded cache
  identity (the comparison result is recorded, never asserted blindly);
* the release window policy equals the shared ``WindowPolicy`` defaults, the
  threshold is the agreed fixed 0.5 and the training languages match the
  languages actually present in the training cache.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import yaml

from src.dataset_prep.config import WindowPolicy
from src.dataset_prep.materialize import sha256_file
from src.detectors.checkpoints import load_checkpoint

PINNED_PREPROCESSING = "voxsentinel-prep-2"
EXPECTED_COUNTS = {"train": 458, "dev": 135}
EXPECTED_THRESHOLD = 0.5
GRU_RELEASE = Path("artifacts/releases/indic-gru-v3-research/release.json")


def _fail(message: str) -> None:
    raise SystemExit(f"refusing to package: {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/runs/aasist-v3-expanded-standardized"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/releases/indic-aasist-v3-research"))
    parser.add_argument("--encoder-config", type=Path, default=Path("configs/backbones.yaml"))
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    for name in ("aasist_best.pt", "dataset_identity.json", "metrics.json", "standardizer.json",
                 "run_status.json", "settings.json", "history.json"):
        if not (run_dir / name).is_file():
            _fail(f"run directory is incomplete: missing {run_dir / name}")

    status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "completed":
        _fail(f"run did not complete: status={status.get('status')!r}")
    if status.get("best_checkpoint_reload_max_logit_error") != 0.0 or status.get("final_live_reload_max_logit_error") != 0.0:
        _fail("checkpoint reload parity is not exact")
    selected_epoch = status.get("selected_epoch")
    settings = json.loads((run_dir / "settings.json").read_text(encoding="utf-8"))
    resolved = settings["resolved"]
    threshold = float(resolved["evaluation"]["threshold"])
    if threshold != EXPECTED_THRESHOLD:
        _fail(f"threshold {threshold} is not the agreed fixed {EXPECTED_THRESHOLD}")
    expected_model_config = dict(resolved["model"]["parameters"])

    checkpoint = run_dir / "aasist_best.pt"
    payload = load_checkpoint(checkpoint, model_name="aasist")
    if payload["model_config"] != expected_model_config:
        _fail("checkpoint model_config differs from the run's resolved settings")
    if payload["model_config"].get("feature_standardization") is not True:
        _fail("AASIST releases require fitted feature standardization")
    if payload["epoch"] != selected_epoch:
        _fail(f"checkpoint epoch {payload['epoch']} != selected epoch {selected_epoch}")

    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    history = json.loads((run_dir / "history.json").read_text(encoding="utf-8"))

    # Minimum-EER selection (earliest epoch wins ties) is recomputed from the
    # epoch history and cross-checked against the reported metrics, the run
    # status and the checkpoint, so an inconsistent run cannot be packaged by
    # editing a single JSON file.
    eligible = [row for row in history if isinstance(row.get("val_eer"), (int, float))]
    if not eligible:
        _fail("history.json carries no development EER values")
    recomputed = min(eligible, key=lambda row: (row["val_eer"], row["epoch"]))
    recomputed_epoch = recomputed["epoch"]
    recomputed_eer = float(recomputed["val_eer"])
    if selected_epoch != recomputed_epoch:
        _fail("reported selected epoch is not the minimum-EER epoch (earliest wins ties) recomputed from history.json")
    if float(metrics["eer"]) != recomputed_eer:
        _fail("metrics.json EER does not match the minimum development EER recomputed from history.json")
    selected_metric = status.get("selected_metric") or {}
    if selected_metric.get("name") != "eer" or selected_metric.get("mode") != "min":
        _fail("run_status does not record minimum-EER selection")
    if selected_metric.get("value") != metrics["eer"]:
        _fail("run_status selected metric value does not match metrics.json")

    selected_history = next(row for row in history if row["epoch"] == selected_epoch)
    history_metric_map = {
        "accuracy": "val_accuracy", "precision": "val_precision", "recall": "val_recall",
        "f1": "val_f1", "eer": "val_eer", "threshold": "val_threshold",
        "true_negative": "true_negative", "false_positive": "false_positive",
        "false_negative": "false_negative", "true_positive": "true_positive",
        "genuine_false_alarm_rate": "genuine_false_alarm_rate",
        "spoof_miss_rate": "spoof_miss_rate",
    }
    for metric_key, row_key in history_metric_map.items():
        if metric_key in metrics and row_key in selected_history and metrics[metric_key] != selected_history[row_key]:
            _fail("metrics.json and history.json disagree at the selected epoch")

    checkpoint_metrics = payload.get("metrics") or {}
    for key in ("eer", "accuracy"):
        if key not in checkpoint_metrics or key not in metrics:
            _fail("checkpoint metrics and metrics.json must both carry the EER and accuracy")
    for key in ("eer", "accuracy", "precision", "recall", "f1", "threshold",
                "true_negative", "false_positive", "false_negative", "true_positive",
                "genuine_false_alarm_rate", "spoof_miss_rate"):
        if key in checkpoint_metrics and key in metrics and checkpoint_metrics[key] != metrics[key]:
            _fail("checkpoint metrics disagree with metrics.json")

    encoder = yaml.safe_load(args.encoder_config.read_text(encoding="utf-8"))["indic_wav2vec"]
    identity = json.loads((run_dir / "dataset_identity.json").read_text(encoding="utf-8"))
    cache_encoder = identity["splits"]["train"]["cache_identity"]["encoder"]
    if (cache_encoder["checkpoint_sha256"] != encoder["checkpoint_sha256"]
            or cache_encoder["output_layer"] != encoder["output_layer"]
            or cache_encoder["embedding_dim"] != encoder["expected_embedding_dim"]
            or cache_encoder["preprocessing_version"] != PINNED_PREPROCESSING):
        _fail("run cache identity differs from the pinned encoder/preprocessing contract")
    expected_head_identity = {
        "checkpoint_sha256": encoder["checkpoint_sha256"],
        "selected_layer": encoder["output_layer"],
        "embedding_dim": encoder["expected_embedding_dim"],
        "preprocessing_version": PINNED_PREPROCESSING,
        "dtype": "float32",
    }
    if payload.get("metadata", {}).get("encoder") != expected_head_identity:
        _fail("head training metadata is incompatible with the pinned inference identity")
    for split, expected in EXPECTED_COUNTS.items():
        entry = identity["splits"][split]
        if int(entry["windows"]) != expected:
            _fail(f"{split} window count {entry['windows']} != {expected}")
        bundle = Path(entry["bundle"])
        if not bundle.is_file() or sha256_file(bundle) != entry["bundle_sha256"]:
            _fail(f"{split} cache bundle is missing or changed: {bundle}")

    frame_totals = {
        split: int(identity["splits"][split]["cache_identity"]["counts"]["valid_frames_total"])
        for split in ("train", "dev")
    }
    gru_totals = None
    window = asdict(WindowPolicy())
    if GRU_RELEASE.is_file():
        gru = json.loads(GRU_RELEASE.read_text(encoding="utf-8"))
        if gru.get("window") != window:
            _fail("window policy differs from the existing GRU release contract")
        gru_totals = {
            split: int(gru["training_identity"]["splits"][split]["counts"]["valid_frames_total"])
            for split in ("train", "dev")
        }
        if gru_totals != frame_totals:
            _fail("run frame totals do not match the GRU release's recorded cache identity")

    languages = sorted({language.lower() for language in identity["splits"]["train"]["cache_identity"]["counts"]["by_language"]})
    if GRU_RELEASE.is_file() and sorted(gru.get("training_languages", [])) != languages:
        _fail("training languages differ from the existing GRU release contract")

    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        _fail(f"output directory already exists and is not empty: {args.out_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    import shutil

    release_head_name = "aasist_best.pt"
    shutil.copyfile(checkpoint, args.out_dir / release_head_name)

    standardizer = json.loads((run_dir / "standardizer.json").read_text(encoding="utf-8"))

    spec = {
        "schema": "voxsentinel.inference_release.v1",
        "release_id": args.out_dir.name,
        "validation_status": "pilot development only; not production validated",
        "head": {
            "file": release_head_name,
            "sha256": sha256_file(checkpoint),
            "model_name": "aasist",
            "epoch": payload["epoch"],
            "architecture": (
                "embedding-adapted AASIST research head (ported graph-attention backend over frozen "
                "IndicWav2Vec embeddings); NOT the published waveform AASIST"
            ),
            "attribution": "docs/third_party/AASIST_NOTICE.md (NAVER MIT, upstream commit a04c9863)",
            "configuration": payload["model_config"],
        },
        "encoder": dict(encoder),
        "preprocessing_version": PINNED_PREPROCESSING,
        "window": window,
        "threshold": threshold,
        "training_languages": languages,
        "training_run": str(run_dir),
        "training_identity": identity,
        "development_selection": {
            "split": "dev",
            "windows": metrics["windows"],
            "selected_epoch": selected_epoch,
            "selection_metric": {"name": "eer", "value": metrics["eer"], "mode": "min", "tie_break": "earliest epoch"},
            "selection_recomputed_from_history": {
                "epoch": recomputed_epoch,
                "eer": recomputed_eer,
                "matches_reported": True,
            },
            "threshold": threshold,
            "metrics": metrics,
            "epoch_row": selected_history,
            "note": (
                "Development split used for checkpoint selection only; it selected the checkpoint and never "
                "supplied gradients. These are development numbers, not an independent benchmark, and they "
                "must not be conflated with the GRU development result."
            ),
        },
        "feature_standardization": {
            **standardizer,
            "note": "fixed train-frame statistics serialized inside the checkpoint; never refitted at inference",
        },
        "cache_generation": {
            "mode": "fresh extraction with the pinned frozen encoder (no historical v1 reuse was available)",
            "frame_totals": frame_totals,
            "gru_reference_frame_totals": gru_totals,
            "frame_totals_match_gru_release_identity": (
                (frame_totals == gru_totals) if gru_totals is not None else None
            ),
            "parity_report": "artifacts/datasets-v3-coverage/features/reports/parity.json",
        },
        "limits": [
            "No Indian-English training yet",
            "Single selected window, not whole-file localization",
            "Uncalibrated score",
            "Gender and source-condition coverage gaps",
            (
                f"Embedding-adapted research head; worse than the GRU baseline on the same development "
                f"split (EER {metrics['eer']:.4f} vs 0.0230, lower is better)"
            ),
        ],
        "usage": (
            "Research pilot; source datasets retain their terms, including IndicSynth CC BY-NC 4.0; "
            "graph blocks adapted from AASIST (MIT, NAVER); no commercial clearance asserted"
        ),
    }
    (args.out_dir / "release.json").write_text(json.dumps(spec, indent=2) + "\n", encoding="utf-8")

    notice = (
        "VoxSentinel Indic AASIST v3 research baseline. Noncommercial research use only. No production or "
        "commercial clearance asserted.\n"
        "Embedding-adapted AASIST graph-attention head over frozen IndicWav2Vec embeddings; it is NOT the "
        "published waveform AASIST. Graph attention blocks are ported from https://github.com/clovaai/aasist "
        "(commit a04c9863f63d44471dde8a6abcb3b082b07cd1d1), MIT licence, Copyright (c) 2021-present NAVER "
        "Corp. See docs/third_party/AASIST_NOTICE.md for the full notice and the list of modifications.\n"
        "Training sources: Kathbath https://github.com/AI4Bharat/IndicSUPERB ; IndicSynth "
        "https://huggingface.co/datasets/vdivyasharma/IndicSynth (dataset card: CC BY-NC 4.0). Encoder: "
        "https://github.com/AI4Bharat/IndicWav2Vec . Preserve attribution and review upstream terms. No "
        "dataset audio or encoder weights included.\n"
        f"Epoch {selected_epoch} (selected by minimum development EER {metrics['eer']:.4f}); development "
        f"accuracy {metrics['accuracy']:.4f}, EER {metrics['eer']:.4f}; benchmark not rescored. The GRU "
        "baseline's development numbers (97.04% accuracy, 2.30% EER) describe the GRU head only. See "
        "docs/AASIST_TRAINING_RESULTS.md in the matching feature branch.\n"
    )
    (args.out_dir / "RESEARCH_USE.txt").write_text(notice, encoding="utf-8")

    lines = []
    for name in ("RESEARCH_USE.txt", release_head_name, "release.json"):
        lines.append(f"{sha256_file(args.out_dir / name)}  {name}\n")
    (args.out_dir / "SHA256SUMS").write_text("".join(lines), encoding="utf-8")

    print(json.dumps({
        "release_dir": str(args.out_dir),
        "release_json": str(args.out_dir / "release.json"),
        "head_sha256": spec["head"]["sha256"],
        "selected_epoch": selected_epoch,
        "dev_eer": metrics["eer"],
        "dev_accuracy": metrics["accuracy"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
