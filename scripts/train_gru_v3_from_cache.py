"""Dataset-aware GRU training entry point for the expanded v3 coverage caches.

This is a clearly scoped v3 runner that shares the existing training
components (``src.detectors.training.fit``, optimizer/loss builders, checkpoint
helpers, the collator) but never touches the historical ``artifacts/datasets``
recovery machinery: dataset-dependent helpers here load the v3 version record,
the v3 caches and the v3 development split explicitly.

Usage (encoder environment):

    ./.venv/bin/python -m scripts.train_gru_v3_from_cache --check-only
    ./.venv/bin/python -m scripts.train_gru_v3_from_cache --run-dir artifacts/runs/<name>

``--check-only`` validates the version contract, both caches (deep audit incl.
prepared-audio hashes), the model configuration and the required identities,
then exits without fitting, optimizer steps, forward passes or predictions.
A real training run refuses to start unless that audit passes; it fits the
standardizer on v3 TRAINING valid frames only, selects the checkpoint by
minimum development EER (earliest epoch wins ties) at a fixed 0.5 threshold
and reports accuracy alongside balanced accuracy, genuine false-alarm rate,
spoof miss rate and counts derived from the inputs.

No model has been trained on v3 yet, and the evaluated benchmark has not been
rescored; this module has been exercised through ``--check-only`` only.
"""
from __future__ import annotations

import argparse
import hashlib
import time
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.data.batch import batch_iterator
from src.dataset_prep.dataset_version import load_dataset_version
from src.dataset_prep.features_v3 import audit_v3_cache, load_v3_examples
from scripts.train_gru_from_cache import (
    _environment_block,
    _code_state,
    _history_rows,
    _now,
    _predict,
    _sha256_file,
    _subgroup_breakdown,
    _unique_run_dir,
    _write_history,
)

DEFAULT_CONFIGS = ["configs/base.yaml", "configs/gru.yaml", "configs/gru_standardized.yaml"]
INITIAL_COMPARISON = {
    "epochs": 10,
    "batch_size": 8,
    "seed": 0,
    "optimizer": {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0},
    "loss": "cross_entropy",
    "threshold": 0.5,
    "best_metric": "eer",
    "best_mode": "min",
    "feature_standardization": True,
    "model": {"name": "gru", "parameters": {"input_dim": 1024, "hidden_size": 256, "num_layers": 1, "dropout": 0.0, "num_classes": 2, "feature_standardization": True}},
    "shuffle": True, "use_amp": False, "device": "cpu",
}


def _resolved_settings(config, config_paths: Sequence[str]) -> dict[str, Any]:
    import yaml

    raw: dict[str, Any] = {}
    for path in config_paths:
        from src.config import _merge
        raw = _merge(raw, yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    return {
        "config_paths": list(config_paths),
        "raw_merged_config": raw,
        "resolved": {
            "model": {"name": config.model.name, "parameters": dict(config.model.parameters)},
            "optimizer": {
                "name": config.optimizer.name,
                "learning_rate": config.optimizer.learning_rate,
                "weight_decay": config.optimizer.weight_decay,
            },
            "training": {
                "batch_size": config.training.batch_size,
                "epochs": config.training.epochs,
                "device": config.training.device,
                "use_amp": config.training.use_amp,
                "shuffle": config.training.shuffle,
                "seed": config.training.seed,
                "loss": config.training.loss,
            },
            "evaluation": {"threshold": config.evaluation.threshold},
            "checkpoint": {
                "format_version": config.checkpoint.format_version,
                "strict_load": config.checkpoint.strict_load,
                "directory": config.checkpoint.directory,
                "best_metric": config.checkpoint.best_metric,
                "best_mode": config.checkpoint.best_mode,
            },
        },
        "notes": [
            "no learning-rate scheduler is configured",
            "no gradient clipping is configured",
            "unweighted cross-entropy: NO class weighting, resampling or oversampling is applied",
            "the genuine/synthetic class ratio is a property of the expanded v3 dataset and is reported from the inputs",
            "checkpoint selection: minimum development EER; earliest epoch wins ties; fixed threshold 0.5",
            "label mapping: 0 = genuine, 1 = synthetic; two-logit cross-entropy",
            "frozen encoder, head-only training on detached cached embeddings; CPU float32 execution",
            "v3 development results are not directly comparable to the historical 96-window development results",
        ],
    }


def _initial_comparison_check(settings: Mapping[str, Any]) -> dict[str, Any]:
    resolved = settings["resolved"]
    actual = {
        "model": resolved["model"],
        "shuffle": resolved["training"]["shuffle"],
        "use_amp": resolved["training"]["use_amp"],
        "device": resolved["training"]["device"],
        "epochs": resolved["training"]["epochs"],
        "batch_size": resolved["training"]["batch_size"],
        "seed": resolved["training"]["seed"],
        "optimizer": resolved["optimizer"],
        "loss": resolved["training"]["loss"],
        "threshold": resolved["evaluation"]["threshold"],
        "best_metric": resolved["checkpoint"]["best_metric"],
        "best_mode": resolved["checkpoint"]["best_mode"],
        "feature_standardization": bool(resolved["model"]["parameters"].get("feature_standardization", False)),
    }
    mismatches = {key: {"expected": value, "actual": actual.get(key)} for key, value in INITIAL_COMPARISON.items()
                  if actual.get(key) != value}
    return {"matches_initial_comparison": not mismatches, "mismatches": mismatches, "actual": actual}


def checkpoint_encoder_identity(cache_encoder):
    """The inference/checkpoint identity is narrower than cache provenance."""
    return {"checkpoint_sha256": cache_encoder["checkpoint_sha256"],
            "selected_layer": cache_encoder["output_layer"],
            "embedding_dim": cache_encoder["embedding_dim"],
            "preprocessing_version": cache_encoder["preprocessing_version"],
            "dtype": cache_encoder["dtype"]}


def _dataset_identity(version_path: Path, features_dir: Path, repo_root: Path) -> dict[str, Any]:
    version = load_dataset_version(version_path, repo_root=repo_root)
    identity: dict[str, Any] = {
        "dataset_version": {
            "path": str(version.path),
            "sha256": version.sha256,
            "protocol": version.protocol,
            "strict_conversion_family_compliant": version.strict_conversion_family_compliant,
        },
        "flag_note": (
            "strict_conversion_family_compliant is reported, not resolved: conversion-family "
            "co-participation chains are documented non-edges and cross-language person "
            "independence is unproven"
        ),
        "splits": {},
    }
    for split in ("train", "dev"):
        spec = version.split(split)
        sidecar_path = features_dir / f"{split}.meta.json"
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.exists() else {}
        identity["splits"][split] = {
            "manifest": str(spec.manifest_path),
            "manifest_declared_sha256": spec.declared_sha256,
            "manifest_file_sha256": _sha256_file(spec.manifest_path),
            "windows": spec.windows,
            "retained": spec.retained,
            "additions": spec.additions,
            "bundle": str(features_dir / f"{split}.pt"),
            "bundle_sha256": _sha256_file(features_dir / f"{split}.pt") if (features_dir / f"{split}.pt").exists() else None,
            "cache_identity": sidecar.get("identity"),
            "counts": sidecar.get("counts"),
        }
    identity["label_mapping"] = {"0": "genuine", "1": "synthetic/spoof"}
    ledger_path = repo_root / "artifacts" / "datasets" / "manifests" / "downloads.json"
    identity["ledger_observation"] = (
        json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
    )
    identity["ledger_note"] = (
        "mutable acquisition accounting recorded for provenance only; it is not part of the "
        "cache identity and never invalidates identical audio"
    )
    return identity


def _class_ratios(counts: Mapping[str, Any]) -> dict[str, Any]:
    genuine, synthetic = int(counts.get("genuine", 0)), int(counts.get("synthetic", 0))
    total = genuine + synthetic
    return {
        "windows": total,
        "genuine": genuine,
        "synthetic": synthetic,
        "genuine_fraction": round(genuine / total, 4) if total else None,
        "synthetic_fraction": round(synthetic / total, 4) if total else None,
        "balanced": genuine == synthetic,
        "note": "unweighted cross-entropy on the natural class ratio; no balancing applied",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", default="artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json")
    parser.add_argument("--features-dir", default="artifacts/datasets-v3-coverage/features")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--detector-config", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--json-out", default=None, help="write the check-only evidence JSON to this path")
    parser.add_argument("--device", default=None)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    repo_root = Path(args.repo_root)
    version_path = Path(args.dataset_version)
    features_dir = Path(args.features_dir)

    audit = audit_v3_cache(version_path, features_dir, repo_root=repo_root, check_audio_hashes=True)
    summary: dict[str, Any] = {"cache_audit": {"ready": audit["ready"], "n_failures": len(audit["failures"]),
                                              "failures": audit["failures"][:8]}}
    if not audit["ready"]:
        print(json.dumps(summary, indent=2))
        print("refusing to proceed: the v3 cache failed its readiness audit (see failures above).", file=sys.stderr)
        return 2

    from src.config import load_config

    config = load_config(args.detector_config)
    version_record = load_dataset_version(version_path, repo_root=repo_root)
    settings = _resolved_settings(config, args.detector_config)
    comparison = _initial_comparison_check(settings)
    if not comparison["matches_initial_comparison"] or args.threads != 4 or args.device not in (None, "cpu"):
        raise ValueError("Controlled v3 run requires the complete initial configuration, CPU and four threads")
    identity = _dataset_identity(version_path, features_dir, repo_root)
    ratios = {
        split: _class_ratios((identity["splits"][split].get("counts") or {}))
        for split in ("train", "dev")
    }
    expected_counts = {
        split: {"windows": version_record.split(split).windows,
                "retained": version_record.split(split).retained,
                "additions": version_record.split(split).additions}
        for split in ("train", "dev")
    }
    counts_match = all(
        int((identity["splits"][split].get("counts") or {}).get("windows", -1)) == expected_counts[split]["windows"]
        and int(identity["splits"][split]["additions"]) == expected_counts[split]["additions"]
        for split in ("train", "dev")
    )

    summary.update({
        "settings": settings["resolved"],
        "initial_comparison": comparison,
        "dataset_identity": identity["dataset_version"],
        "counts": {split: {"windows": identity["splits"][split]["windows"],
                           "retained": identity["splits"][split]["retained"],
                           "additions": identity["splits"][split]["additions"],
                           "bundle_sha256": identity["splits"][split]["bundle_sha256"]}
                   for split in ("train", "dev")},
        "class_ratios": ratios,
        "counts_match_version_record": counts_match,
        "encoder_identity": (identity["splits"]["train"].get("cache_identity") or {}).get("encoder"),
    })

    if args.check_only:
        summary["status"] = "check-only"
        summary["note"] = (
            "manifests, caches, model configuration and identities validated; "
            "NO fitting, optimizer steps, forward passes or predictions performed."
        )
        if args.json_out:
            out = Path(args.json_out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            summary["report"] = str(out)
        print(json.dumps(summary, indent=2))
        ok = audit["ready"] and counts_match
        if not comparison["matches_initial_comparison"]:
            print(f"warning: settings differ from the documented initial comparison: {comparison['mismatches']}",
                  file=sys.stderr)
        return 0 if ok else 1

    # Real training path (NOT executed in this task).  Fail-closed audit above,
    # then the shared training components drive the head.
    from src.detectors.checkpoints import restore_checkpoint, save_checkpoint
    from src.detectors.registry import create_detector
    from src.detectors.standardization import fit_detector_standardizer
    from src.detectors.training import _metrics_dict, build_loss, build_optimizer, fit

    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    device = torch.device(args.device or config.training.device)

    train_examples, train_records = load_v3_examples(version_path, features_dir, "train", repo_root=repo_root)
    dev_examples, dev_records = load_v3_examples(version_path, features_dir, "dev", repo_root=repo_root)
    if len(train_examples) != identity["splits"]["train"]["windows"]:
        raise AssertionError("train example count does not match the version record")
    if len(dev_examples) != identity["splits"]["dev"]["windows"]:
        raise AssertionError("dev example count does not match the version record")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(Path(args.run_dir) if args.run_dir else Path("artifacts/runs") / f"gru-v3-{stamp}")
    print(f"run directory: {run_dir}")
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (run_dir / "dataset_identity.json").write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    (run_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    (run_dir / "class_ratios.json").write_text(json.dumps(ratios, indent=2) + "\n", encoding="utf-8")

    started, start_clock = _now(), time.perf_counter()
    (run_dir / "run_status.json").write_text(json.dumps({"status":"running", "started":started})+"\n")
    try:
        environment = _environment_block(config)
        environment["device"] = str(device)
        (run_dir / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
        (run_dir / "code_state.json").write_text(json.dumps(_code_state(run_dir), indent=2)+"\n")
        seed = int(config.training.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        detector = create_detector(config.model)
        detector.to(device)
        rng_before = torch.random.get_rng_state().clone()
        transform_metadata = fit_detector_standardizer(
            detector, train_examples, cache_signature=identity["splits"]["train"]["bundle_sha256"])
        if not torch.equal(rng_before, torch.random.get_rng_state()):
            raise AssertionError("Fitting the standardizer changed the RNG state")
        if transform_metadata is not None:
            torch.save(detector.model.standardizer.state_dict(), run_dir / "standardizer.pt")
            (run_dir / "standardizer.json").write_text(json.dumps(transform_metadata, indent=2) + "\n", encoding="utf-8")

        optimizer = build_optimizer(detector, config.optimizer)
        loss_fn = build_loss(config.training.loss)
        batch_size = int(config.training.batch_size)
        threshold = float(config.evaluation.threshold)
        metadata = {
            "run_directory": str(run_dir),
            "dataset_version": identity["dataset_version"],
            "feature_cache": {split: identity["splits"][split] for split in ("train", "dev")},
            "class_ratios": ratios,
            "encoder": checkpoint_encoder_identity(identity["splits"]["train"]["cache_identity"]["encoder"]),
            "label_mapping": identity["label_mapping"],
            "seed": seed,
            "device": str(device),
            "checkpoint_selection": {"metric": config.checkpoint.best_metric, "mode": config.checkpoint.best_mode,
                                     "tie_break": "earliest epoch"},
            "threshold": threshold,
        }
        if transform_metadata is not None:
            metadata["feature_standardization"] = transform_metadata

        best_path = run_dir / "gru_best.pt"
        epoch_logits = {}
        epoch_evidence = []

        def record_epoch(model, summary):
            state = torch.random.get_rng_state().clone()
            live_logits, _ = _predict(model, dev_examples, device=device, batch_size=batch_size)
            if not torch.equal(state, torch.random.get_rng_state()):
                raise AssertionError("Read-only epoch evaluation consumed RNG")
            epoch_logits[summary.epoch] = live_logits.copy()
            epoch_evidence.append({"epoch": summary.epoch,
                                   "dev_logits_sha256": hashlib.sha256(live_logits.tobytes()).hexdigest()})
            (run_dir / "live_epoch_evidence.json").write_text(json.dumps(epoch_evidence, indent=2)+"\n")
            print(f"epoch {summary.epoch}: train CE {summary.train_loss:.5f}; dev CE {summary.val_loss:.5f}; dev EER {summary.metrics.eer:.5f}", flush=True)

        result = fit(
            detector,
            lambda epoch: batch_iterator(train_examples, batch_size, shuffle=config.training.shuffle, seed=seed + epoch),
            lambda epoch: batch_iterator(dev_examples, batch_size, shuffle=False),
            optimizer,
            model_name=config.model.name,
            model_config=dict(config.model.parameters),
            epochs=int(config.training.epochs),
            device=device,
            loss_fn=loss_fn,
            checkpoint_path=best_path,
            monitor=config.checkpoint.best_metric,
            mode=config.checkpoint.best_mode,
            threshold=threshold,
            metadata=metadata,
            epoch_callback=record_epoch,
        )
        history = _history_rows(result)
        _write_history(run_dir, history)
        last_epoch = result.epochs[-1]
        save_checkpoint(
            run_dir / "gru_final.pt",
            detector,
            config.model.name,
            dict(config.model.parameters),
            optimizer=optimizer,
            epoch=last_epoch.epoch,
            global_step=sum(epoch.train_steps for epoch in result.epochs),
            metrics=_metrics_dict(last_epoch.metrics),
            metadata=metadata,
        )

        best_detector = create_detector(config.model)
        restore_checkpoint(best_path, best_detector, model_name=config.model.name)
        best_detector.to(device)
        logits, scores = _predict(best_detector, dev_examples, device=device, batch_size=batch_size)
        reload_error = float(np.abs(logits - epoch_logits[result.best_epoch]).max())
        if reload_error != 0:
            raise AssertionError(f"Selected epoch live/reload mismatch: {reload_error}")
        final_detector = create_detector(config.model)
        restore_checkpoint(run_dir / "gru_final.pt", final_detector, model_name=config.model.name)
        final_detector.to(device)
        final_logits, _ = _predict(final_detector, dev_examples, device=device, batch_size=batch_size)
        final_error = float(np.abs(final_logits - epoch_logits[last_epoch.epoch]).max())
        if final_error != 0:
            raise AssertionError(f"Final live/reload mismatch: {final_error}")
        from src.dataset_prep.dataset_version import load_split_rows
        from scripts.gru_identity_audit import reference
        manifest_rows = {r['window_id']: r for r in load_split_rows(version_record, 'dev', repo_root=repo_root)}
        predictions = []
        for record, values, score in zip(dev_records, logits, scores):
            row = manifest_rows[record['window_id']]
            parent = row.get('parent_refs') or {}
            ref = reference(row['source_file'] if row['label'] == 0 else parent.get('target_reference'))
            predictions.append({**record, 'speaker_ids': row.get('speaker_ids'), 'parent_refs': parent,
                                'gender_role': 'genuine' if row['label'] == 0 else 'synthetic_target',
                                'gender': ref['gender'] if ref else 'unknown',
                                'cohort': 'addition' if row.get('path_base') else 'retained',
                                'logits': values.tolist(), 'synthetic_score': float(score)})
        (run_dir / 'dev_predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in predictions))
        from scripts.evaluate_frozen_v2_test import summarize
        from collections import defaultdict
        groups = defaultdict(list)
        for row in predictions:
            groups['cohort/'+row['cohort']].append(row)
            groups[row['gender_role']+'/'+row['gender']].append(row)
        (run_dir / 'coverage_breakdown.json').write_text(json.dumps({k:summarize(v) for k,v in groups.items()}, indent=2)+'\n')
        labels = np.array([example.label for example in dev_examples])
        from src.scoring.metrics import binary_metrics

        metrics = binary_metrics(torch.from_numpy(scores), torch.from_numpy(labels), threshold=threshold)
        tn, fp, fn, tp = metrics.true_negative, metrics.false_positive, metrics.false_negative, metrics.true_positive
        specificity = tn / (tn + fp) if tn + fp else None
        balanced = (metrics.recall + specificity) / 2 if specificity is not None else None
        overall = {
            "windows": len(dev_examples),
            "threshold": threshold,
            "score_direction": "higher score = more synthetic",
            "accuracy": metrics.accuracy,
            "balanced_accuracy": balanced,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "eer": metrics.eer,
            "true_negative": tn,
            "false_positive": fp,
            "false_negative": fn,
            "true_positive": tp,
            "genuine_false_alarm_rate": metrics.genuine_false_alarm_rate,
            "spoof_miss_rate": metrics.spoof_miss_rate,
        }
        breakdown = _subgroup_breakdown(
            [{"spoken_language": record["spoken_language"], "generator": record["generator"],
              "label": record["label"], "window_id": record["window_id"]} for record in dev_records],
            scores, threshold)
        (run_dir / "metrics.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
        (run_dir / "breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n", encoding="utf-8")

        status = {
            "status": "completed",
            "started": started,
            "finished": _now(),
            "elapsed_seconds": time.perf_counter()-start_clock,
            "final_live_reload_max_logit_error": final_error,
            "test_evaluated": False,
            "selected_epoch": result.best_epoch,
            "selected_metric": {"name": result.monitor, "value": result.best_metric, "mode": result.mode,
                                "tie_break": "earliest epoch"},
            "device": str(device),
            "controlled_frame_standardization": transform_metadata is not None,
            "best_checkpoint_reload_max_logit_error": reload_error,
            "note": "features reused from the v3 cache; no encoder forward pass during training",
        }
        (run_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"run_dir": str(run_dir), "selected_epoch": result.best_epoch, "metrics": overall}, indent=2))
        return 0
    except BaseException as error:
        (run_dir / "run_status.json").write_text(json.dumps({"status":"failed", "started":started,
            "finished":_now(), "error":repr(error)}, indent=2)+"\n")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
