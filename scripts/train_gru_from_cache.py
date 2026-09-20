"""Audited, reproducible GRU training entry point over the prepared feature cache.

This wires the frozen-cache bundles produced by
``scripts.prepare_datasets features`` into the existing B2 training components
(``src.detectors.training.fit`` with the registry detector, optimizer, loss and
checkpoint helpers — the same building blocks ``runner.run_training`` uses).

Usage (encoder environment, after the preparation pipeline has run):

    .venv/bin/python -m scripts.train_gru_from_cache --check-only
    .venv/bin/python -m scripts.train_gru_from_cache [--run-dir artifacts/runs/<name>]

``--check-only`` runs the full cache audit (completeness markers, manifest
freshness, item counts, both labels present, every core language contributing
both classes, bundle identity) and exits without touching the trainer.  A real
training run refuses to start unless that audit passes for both splits, so a
single-class or stale cache can never silently train a detector.

A training run writes one unique run directory containing the resolved
settings, dataset identity (manifest/cache/bundle hashes and lineage status
counts), environment, code snapshot (diff + file hashes), epoch history,
best/final checkpoints, best-checkpoint validation predictions and a concise
training report.  The best checkpoint is selected by the configured validation
metric (default: lowest validation EER, earliest epoch wins ties).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from src.data.batch import EmbeddingExample, batch_iterator
from src.dataset_prep.features import load_feature_bundle, validate_core_training_cache

CORE_LANGUAGES = [
    "Bengali",
    "Gujarati",
    "Hindi",
    "Kannada",
    "Malayalam",
    "Marathi",
    "Odia",
    "Punjabi",
    "Sanskrit",
    "Tamil",
    "Telugu",
    "Urdu",
]

SNAPSHOT_DIRS = ("src", "scripts", "configs", "docs", "README.md")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip()


def _unique_run_dir(path: Path) -> Path:
    candidate = path
    suffix = 2
    while candidate.exists() and any(candidate.iterdir()):
        candidate = path.with_name(f"{path.name}-{suffix}")
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def _resolved_settings(config, config_paths: Sequence[str]) -> dict[str, Any]:
    import yaml

    raw = {}
    for path in config_paths:
        raw.update(yaml.safe_load(Path(path).read_text()) or {})
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
        "scheduler": None,
        "gradient_clipping": None,
        "notes": [
            "no learning-rate scheduler is configured",
            "no gradient clipping is configured",
            "balanced genuine/synthetic sampling is a property of the dataset (equal counts)",
            "no class weighting or oversampling is applied",
            "checkpoint selection: lowest validation ER (configured metric), earliest epoch wins ties",
            "CPU float32 execution (config training.device)",
            "label mapping: 0 = genuine, 1 = synthetic; the head uses two-logit cross-entropy",
        ],
    }


def _environment_block(config) -> dict[str, Any]:
    def version(name: str) -> str:
        try:
            from importlib.metadata import version as _v

            return _v(name)
        except Exception:
            return "unknown"

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "torchaudio": version("torchaudio"),
        "numpy": np.__version__,
        "fairseq": version("fairseq"),
        "device": str(config.training.device),
        "dtype": "float32",
    }


def _code_state(run_dir: Path) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    branch = _git(repo, "branch", "--show-current") or "unknown"
    commit = _git(repo, "rev-parse", "HEAD") or "unknown"
    status = _git(repo, "status", "--porcelain")
    diff = _git(repo, "diff", "HEAD")
    (run_dir / "code_diff.patch").write_text(diff + "\n", encoding="utf-8")
    (run_dir / "git_status.txt").write_text(status + "\n", encoding="utf-8")

    changed = {
        line.split()[-1]
        for line in _git(repo, "diff", "--name-only", "HEAD").splitlines()
        if line.strip()
    }
    untracked = {
        line.strip()
        for line in _git(repo, "ls-files", "--others", "--exclude-standard", *SNAPSHOT_DIRS).splitlines()
        if line.strip()
    }
    hashes: dict[str, str] = {}
    for name in sorted(changed | untracked):
        path = repo / name
        if path.is_file() and not name.startswith("artifacts/"):
            hashes[name] = _sha256_file(path)
    import tarfile
    snapshot_hashes = {}
    with tarfile.open(run_dir / "source_snapshot.tar.gz", "w:gz") as archive:
        for pattern in ("src/**/*.py", "scripts/**/*.py", "scripts/*.sh", "configs/*", "docs/**/*.md", "requirements*.txt", "README.md", ".python-version"):
            for path in sorted(repo.glob(pattern)):
                if path.is_file() and "__pycache__" not in path.parts:
                    name = str(path.relative_to(repo))
                    archive.add(path, arcname=name)
                    snapshot_hashes[name] = _sha256_file(path)
    return {
        "source_snapshot": "source_snapshot.tar.gz",
        "source_snapshot_sha256": _sha256_file(run_dir / "source_snapshot.tar.gz"),
        "snapshot_file_sha256": snapshot_hashes,
        "branch": branch,
        "commit": commit,
        "dirty": bool(status),
        "diff_patch": "code_diff.patch",
        "git_status": "git_status.txt",
        "changed_or_untracked_source_hashes": hashes,
        "note": "Full source contents archived, including untracked files; commit/diff/hashes alone are insufficient.",
    }


def _dataset_identity(root: Path) -> dict[str, Any]:
    manifests = root / "manifests"
    identity: dict[str, Any] = {"dataset_root": str(root)}
    for split in ("train", "val"):
        manifest = manifests / f"windows.core_{split}.jsonl"
        bundle = root / "features" / f"{split}.pt"
        sidecar = root / "features" / f"{split}.meta.json"
        entry: dict[str, Any] = {
            "manifest": str(manifest),
            "manifest_sha256": _sha256_file(manifest) if manifest.exists() else None,
            "bundle": str(bundle),
            "bundle_sha256": _sha256_file(bundle) if bundle.exists() else None,
        }
        if sidecar.exists():
            meta = json.loads(sidecar.read_text())
            entry["cache_identity"] = meta.get("identity")
            entry["label_counts"] = meta.get("label_counts")
            entry["by_language"] = meta.get("by_language")
        identity[split] = entry
    identity["label_mapping"] = {"0": "genuine", "1": "synthetic/spoof"}
    identity["ledger"] = json.loads((manifests / "downloads.json").read_text()) if (
        manifests / "downloads.json"
    ).exists() else None
    return identity


def _verification_summary(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    path = root / "manifests" / "source_windows.indicsynth.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    status_counts: dict[str, int] = {}
    evidence_methods: dict[str, int] = {}
    for row in rows:
        parent = row.get("parent_refs") or {}
        for role in ("source", "target"):
            status = str(parent.get(f"{role}_parent_verification"))
            status_counts[status] = status_counts.get(status, 0) + 1
        evidence = parent.get("reference_evidence") or {}
        for role in ("source", "target"):
            method = str((evidence.get(role) or {}).get("method") or "none")
            evidence_methods[method] = evidence_methods.get(method, 0) + 1
    replacements: dict[str, Any] = {}
    for path in sorted(root.glob("staging/joint/lineage_excluded.*.json")):
        items = json.loads(path.read_text())
        if items:
            replacements[path.stem.split(".", 1)[1]] = items
    exclusion_history: dict[str, int] = {}
    excluded_path = root / "manifests" / "windows.excluded.jsonl"
    if excluded_path.exists():
        for line in excluded_path.read_text().splitlines():
            if not line.strip():
                continue
            reason = str(json.loads(line).get("reason") or "").split(":", 1)[0].strip()
            if reason in (
                "lineage_replacement_required",
                "reference_unresolvable_precheck",
                "upstream_held_out_parent",
            ):
                exclusion_history[reason] = exclusion_history.get(reason, 0) + 1
    return {
        "synthetic_windows": len(rows),
        "verification_status_counts": dict(sorted(status_counts.items())),
        "verification_evidence_methods": dict(sorted(evidence_methods.items())),
        "rows_replaced_by_language": replacements,
        "reference_exclusions_by_reason": dict(sorted(exclusion_history.items())),
    }


def _history_rows(result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for epoch in result.epochs:
        metrics = epoch.metrics
        tn, fp = metrics.true_negative, metrics.false_positive
        fn, tp = metrics.false_negative, metrics.true_positive
        specificity = tn / (tn + fp) if tn is not None and tn + fp else None
        balanced = (
            (metrics.recall + specificity) / 2 if metrics.recall is not None and specificity is not None else None
        )
        rows.append(
            {
                "epoch": epoch.epoch,
                "train_loss": epoch.train_loss,
                "val_loss": epoch.val_loss,
                "val_accuracy": metrics.accuracy,
                "val_balanced_accuracy": balanced,
                "val_precision": metrics.precision,
                "val_recall": metrics.recall,
                "val_f1": metrics.f1,
                "val_eer": metrics.eer,
                "val_threshold": metrics.threshold,
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp,
                "genuine_false_alarm_rate": metrics.genuine_false_alarm_rate,
                "spoof_miss_rate": metrics.spoof_miss_rate,
                "train_steps": epoch.train_steps,
                "train_examples": epoch.train_examples,
            }
        )
    return rows


def _write_history(run_dir: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    (run_dir / "history.json").write_text(json.dumps(list(rows), indent=2) + "\n", encoding="utf-8")
    if rows:
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def _predict(detector, examples: Sequence[EmbeddingExample], *, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    detector.eval()
    logits_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in batch_iterator(examples, batch_size, shuffle=False):
            moved = {key: value.to(device) for key, value in batch.items()}
            output = detector(moved)
            logits_chunks.append(output.logits.detach().cpu().numpy())
    logits = np.concatenate(logits_chunks, axis=0)
    probs = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    return logits, probs


def _subgroup_breakdown(
    records: Sequence[Mapping[str, Any]], scores: np.ndarray, threshold: float
) -> dict[str, Any]:
    from collections import defaultdict

    groups: dict[str, dict[str, Any]] = defaultdict(lambda: {"windows": 0, "genuine": 0, "spoof": 0, "TP": 0, "FP": 0, "FN": 0, "TN": 0})
    for record, score in zip(records, scores):
        for key in (f"language:{record['spoken_language']}", f"generator:{record.get('generator') or 'genuine'}"):
            entry = groups[key]
            entry["windows"] += 1
            label = int(record["label"])
            predicted_spoof = bool(score >= threshold)
            if label == 0:
                entry["genuine"] += 1
                entry["FP"] += int(predicted_spoof)
                entry["TN"] += int(not predicted_spoof)
            else:
                entry["spoof"] += 1
                entry["TP"] += int(predicted_spoof)
                entry["FN"] += int(not predicted_spoof)
    result: dict[str, Any] = {}
    for key, entry in sorted(groups.items()):
        genuine, spoof = entry["genuine"], entry["spoof"]
        accuracy = (entry["TP"] + entry["TN"]) / entry["windows"] if entry["windows"] else None
        fpr = entry["FP"] / genuine if genuine else None
        miss = entry["FN"] / spoof if spoof else None
        balanced = (
            0.5 * ((1 - miss) + (1 - fpr)) if genuine and spoof else None
        )
        result[key] = {
            **entry,
            "accuracy": accuracy,
            "genuine_false_alarm_rate": fpr,
            "spoof_miss_rate": miss,
            "balanced_accuracy": balanced,
            "note": (
                "single-class subgroup: error rates use the applicable side only; no binary EER"
                if not (genuine and spoof)
                else None
            ),
        }
    return result


def _training_report(
    run_dir: Path,
    *,
    settings: Mapping[str, Any],
    dataset: Mapping[str, Any],
    verification: Mapping[str, Any],
    audit: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    result,
    overall: Mapping[str, Any],
    breakdown: Mapping[str, Any],
    elapsed_seconds: float,
) -> None:
    lines: list[str] = []
    lines.append("# Frozen-IndicWav2Vec → GRU training run")
    lines.append("")
    lines.append(f"Run directory: `{run_dir}`")
    lines.append(f"Generated: {_now()}")
    lines.append("")
    lines.append("Dataset preparation and lineage repairs predate this run; see the original manager report.")
    lines.append("This run reuses the frozen caches. Optional frame standardization is recorded in settings and checkpoint state.")
    lines.append("")
    lines.append("## Readiness")
    lines.append("")
    lines.append(f"- Cache audit ready: {audit['ready']} (failures: {audit['failures'] or 'none'})")
    lines.append("- Training data = `windows.core_train.jsonl` only; validation = `windows.core_val.jsonl` only.")
    lines.append("- Label mapping: 0 = genuine, 1 = synthetic (two-logit cross-entropy).")
    lines.append("")
    lines.append("## Resolved settings")
    lines.append("")
    resolved = settings["resolved"]
    lines.append(f"- model: `{resolved['model']['name']}` {json.dumps(resolved['model']['parameters'])}")
    lines.append(f"- optimizer: {json.dumps(resolved['optimizer'])}")
    lines.append(f"- training: {json.dumps(resolved['training'])}")
    lines.append("- scheduler: none; gradient clipping: none; class weights/oversampling: none")
    lines.append(f"- checkpoint selection: lowest `{resolved['checkpoint']['best_metric']}` (mode {resolved['checkpoint']['best_mode']}), earliest epoch wins ties")
    lines.append("")
    lines.append("## Dataset composition")
    lines.append("")
    for split in ("train", "val"):
        entry = dataset[split]
        lines.append(
            f"- {split}: {entry.get('label_counts')} | languages={len(entry.get('by_language') or {})}"
        )
    lines.append("")
    lines.append("## Epoch history")
    lines.append("")
    lines.append("| epoch | train loss | val loss | val acc | balanced acc | val EER |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in history:
        def fmt(value):
            return "n/a" if value is None else (f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append(
            f"| {row['epoch']} | {fmt(row['train_loss'])} | {fmt(row['val_loss'])} | "
            f"{fmt(row['val_accuracy'])} | {fmt(row['val_balanced_accuracy'])} | {fmt(row['val_eer'])} |"
        )
    lines.append("")
    lines.append(f"Selected epoch (best `{result.monitor}` = {result.best_metric}): **{result.best_epoch}**")
    lines.append(f"Head-training time: {elapsed_seconds:.1f} s ({_environment_block_runtime()})")
    lines.append("")
    lines.append("## Best-checkpoint validation metrics (threshold 0.5, synthetic = positive)")
    lines.append("")
    m = overall
    def _fmt_float(value):
        return "n/a" if value is None else f"{float(value):.4f}"

    lines.append(
        f"- accuracy {_fmt_float(m['accuracy'])} | balanced accuracy {_fmt_float(m['balanced_accuracy'])} | "
        f"EER {_fmt_float(m['eer'])}"
    )
    lines.append(
        f"- confusion (rows = true, cols = predicted; order genuine, synthetic): "
        f"TN {m['true_negative']}, FP {m['false_positive']}, FN {m['false_negative']}, TP {m['true_positive']}"
    )
    lines.append(
        f"- genuine false-alarm rate {_fmt_float(m['genuine_false_alarm_rate'])} | "
        f"synthetic miss rate {_fmt_float(m['spoof_miss_rate'])}"
    )
    lines.append("- EER is a ranking summary at its own operating point; it is not a deployed threshold.")
    lines.append("")
    lines.append("## Per-language / per-generator breakdown")
    lines.append("")
    lines.append("| group | windows | genuine | spoof | FP | FN | FPR | miss | balanced acc |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for key, entry in breakdown.items():
        def fmt(value):
            return "–" if value is None else f"{value:.3f}"
        lines.append(
            f"| {key} | {entry['windows']} | {entry['genuine']} | {entry['spoof']} | "
            f"{entry['FP']} | {entry['FN']} | {fmt(entry['genuine_false_alarm_rate'])} | "
            f"{fmt(entry['spoof_miss_rate'])} | {fmt(entry['balanced_accuracy'])} |"
        )
    lines.append("")
    lines.append("## Limitations")
    lines.append("")
    lines.append("- 384 training / 96 validation windows; each language contributes 8 validation examples.")
    lines.append("- Validation selected the checkpoint; these are development results, not independent test performance.")
    lines.append("- No unseen-generator, Indian-English, calibration or latency claim is made.")
    lines.append("- Exact hashing cannot exclude near-duplicates or unidentified cross-corpus speakers.")
    lines.append("- Head-only training on detached cached embeddings; the encoder is frozen and not updated.")
    lines.append("")
    lines.append("Artifacts: `settings.json`, `dataset_identity.json`, `verification_summary.json`,")
    lines.append("`environment.json`, `code_state.json`, `code_diff.patch`, `history.{csv,json}`,")
    lines.append("`gru_best.pt`, `gru_final.pt`, `val_predictions.{csv,jsonl}`, `metrics.json`,")
    lines.append("`breakdown.json`, `run_status.json`. Learning-curve plot deferred: matplotlib is not")
    lines.append("installed in the pinned encoder environment (CSV/JSON history is complete).")
    (run_dir / "training_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _environment_block_runtime() -> str:
    return f"{platform.platform()}, torch {torch.__version__}, torch threads {torch.get_num_threads()}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="artifacts/datasets")
    parser.add_argument("--detector-config", nargs="+", default=["configs/base.yaml", "configs/gru.yaml"])
    parser.add_argument("--run-dir", default=None, help="unique output directory (refuses to reuse a non-empty one)")
    parser.add_argument("--check-only", action="store_true", help="audit the cache without training")
    parser.add_argument("--device", default=None, help="override the configured device (default: config value)")
    parser.add_argument("--threads", type=int, default=None, help="torch CPU thread override")
    parser.add_argument("--reference-run", type=Path, default=None, help="required original comparator for the controlled standardized experiment")
    parser.add_argument("--split-version", type=Path, help="verify v2 train/dev match existing audited cache manifests; never load test")
    args = parser.parse_args()

    root = Path(args.dataset_root)
    split_contract = None
    if args.split_version is not None:
        from scripts.training_split_contract import verify_split_contract
        split_contract = verify_split_contract(args.split_version, root)
    train_path = root / "features" / "train.pt"
    val_path = root / "features" / "val.pt"

    audit = validate_core_training_cache(
        root / "features",
        train_manifest=root / "manifests" / "windows.core_train.jsonl",
        val_manifest=root / "manifests" / "windows.core_val.jsonl",
        expected_languages=CORE_LANGUAGES,
    )
    print(json.dumps({"cache_audit": audit}, indent=2))
    if args.check_only:
        print(
            "check-only requested: cache audit report above; training NOT started."
            if audit["ready"]
            else "check-only requested: cache is NOT training-ready (see failures above); training NOT started."
        )
        return 0 if audit["ready"] else 1
    if not audit["ready"]:
        print(
            "refusing to train: the core feature cache failed its readiness audit: "
            + ", ".join(audit["failures"]),
            file=sys.stderr,
        )
        return 2

    from src.config import load_config
    from src.detectors.checkpoints import restore_checkpoint, save_checkpoint
    from src.detectors.registry import create_detector
    from src.detectors.training import build_loss, build_optimizer, fit

    config = load_config(args.detector_config)
    if args.threads is not None:
        torch.set_num_threads(int(args.threads))
    device = torch.device(args.device or config.training.device)

    recovery_readiness = None
    standardized = bool(config.model.parameters.get("feature_standardization", False))
    if standardized:
        if args.reference_run is None:
            raise ValueError("Standardized experiment requires --reference-run")
        if device.type != "cpu" or torch.get_num_threads() != 4:
            raise ValueError("Controlled experiment requires CPU and --threads 4")
        from scripts.gru_recovery import readiness
        recovery_readiness = readiness(root, args.reference_run)
        if not recovery_readiness["ready"]:
            raise ValueError(f"Recovery readiness failed: {recovery_readiness['failures']}")
        expected = json.loads((args.reference_run / "settings.json").read_text())["resolved"]
        actual = _resolved_settings(config, args.detector_config)["resolved"]
        import copy
        comparable = copy.deepcopy(actual)
        comparable["model"]["parameters"].pop("feature_standardization")
        if comparable != expected:
            raise ValueError("Controlled settings must match original except feature_standardization")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_dir = _unique_run_dir(Path(args.run_dir) if args.run_dir else Path("artifacts/runs") / f"gru-core-{stamp}")
    print(f"run directory: {run_dir}")
    if recovery_readiness is not None:
        (run_dir / "readiness.json").write_text(json.dumps(recovery_readiness, indent=2) + "\n")

    settings = _resolved_settings(config, args.detector_config)
    dataset = _dataset_identity(root)
    if split_contract is not None:
        dataset["split_contract"] = split_contract
        (run_dir / "split_contract.json").write_text(json.dumps(split_contract, indent=2) + "\n")
    verification = _verification_summary(root)
    environment = _environment_block(config)
    code_state = _code_state(run_dir)
    (run_dir / "settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (run_dir / "dataset_identity.json").write_text(json.dumps(dataset, indent=2) + "\n", encoding="utf-8")
    (run_dir / "verification_summary.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    (run_dir / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    (run_dir / "code_state.json").write_text(json.dumps(code_state, indent=2) + "\n", encoding="utf-8")
    (run_dir / "cache_audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")

    train_bundle = load_feature_bundle(train_path)
    val_bundle = load_feature_bundle(val_path)
    train_examples = [EmbeddingExample(features=item["features"], label=int(item["label"])) for item in train_bundle["items"]]
    val_examples = [EmbeddingExample(features=item["features"], label=int(item["label"])) for item in val_bundle["items"]]
    print(
        json.dumps(
            {
                "train": {"items": len(train_examples), "dim": train_examples[0].embedding_dim},
                "val": {"items": len(val_examples), "dim": val_examples[0].embedding_dim},
            }
        )
    )

    seed = int(config.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    detector = create_detector(config.model)
    detector.to(device)
    from src.detectors.standardization import fit_detector_standardizer
    rng_before = torch.random.get_rng_state().clone()
    transform_metadata = fit_detector_standardizer(detector, train_examples, cache_signature=dataset["train"]["bundle_sha256"])
    if not torch.equal(rng_before, torch.random.get_rng_state()):
        raise AssertionError("Fitting the standardizer changed the RNG state")
    if transform_metadata is not None:
        torch.save(detector.model.standardizer.state_dict(), run_dir / "standardizer.pt")
        (run_dir / "standardizer.json").write_text(json.dumps(transform_metadata, indent=2) + "\n")
    optimizer = build_optimizer(detector, config.optimizer)
    loss_fn = build_loss(config.training.loss)
    batch_size = int(config.training.batch_size)
    metadata = {
        "run_directory": str(run_dir),
        "feature_cache": {
            "train": dataset["train"],
            "val": dataset["val"],
        },
        "encoder": {
            "checkpoint_sha256": (dataset["train"].get("cache_identity") or {}).get(
                "encoder_checkpoint_sha256"
            ),
            "selected_layer": (dataset["train"].get("cache_identity") or {}).get("output_layer"),
            "preprocessing_version": (dataset["train"].get("cache_identity") or {}).get("preprocessing_version"),
            "embedding_dim": (dataset["train"].get("cache_identity") or {}).get("embedding_dim"),
            "dtype": (dataset["train"].get("cache_identity") or {}).get("dtype"),
        },
        "label_mapping": {"0": "genuine", "1": "synthetic/spoof"},
        "seed": seed,
        "device": str(device),
    }
    if split_contract is not None:
        metadata["split_contract"] = split_contract
    if transform_metadata is not None:
        metadata["feature_standardization"] = transform_metadata
    best_path = run_dir / "gru_best.pt"
    eval_history = []
    final_live_logits = None

    def record_epoch(model, summary):
        nonlocal final_live_logits
        from scripts.gru_recovery import evaluate_model
        # Read-only eval; save RNG and buffer identity to enforce no training effect.
        state = torch.random.get_rng_state().clone()
        train_eval, _, _ = evaluate_model(model, train_examples)
        val_eval, final_live_logits, _ = evaluate_model(model, val_examples)
        if not torch.equal(state, torch.random.get_rng_state()):
            raise AssertionError("Evaluation callback consumed RNG")
        eval_history.append({"epoch": summary.epoch, "online_train_loss": summary.train_loss,
                             "train_eval": train_eval, "val_eval": val_eval,
                             "val_logits_sha256": hashlib.sha256(final_live_logits.tobytes()).hexdigest()})
        (run_dir / "eval_history.json").write_text(json.dumps(eval_history, indent=2) + "\n")
        print(f"epoch {summary.epoch}: online CE {summary.train_loss:.5f}; train eval CE {train_eval['cross_entropy']:.5f}; dev CE {val_eval['cross_entropy']:.5f}; dev EER {summary.metrics.eer:.5f}", flush=True)


    start = _now()
    started = time.perf_counter()
    try:
        result = fit(
            detector,
            lambda epoch: batch_iterator(train_examples, batch_size, shuffle=config.training.shuffle, seed=seed + epoch),
            lambda epoch: batch_iterator(val_examples, batch_size, shuffle=False),
            optimizer,
            model_name=config.model.name,
            model_config=dict(config.model.parameters),
            epochs=int(config.training.epochs),
            device=device,
            loss_fn=loss_fn,
            checkpoint_path=best_path,
            monitor=config.checkpoint.best_metric,
            mode=config.checkpoint.best_mode,
            threshold=float(config.evaluation.threshold),
            metadata=metadata,
            epoch_callback=record_epoch if standardized else None,
        )
    except BaseException as error:  # mark the failed attempt before re-raising
        (run_dir / "run_status.json").write_text(
            json.dumps({"status": "failed", "started": start, "finished": _now(), "error": repr(error)}, indent=2) + "\n",
            encoding="utf-8",
        )
        raise
    elapsed = time.perf_counter() - started

    history = _history_rows(result)
    _write_history(run_dir, history)

    last_epoch = result.epochs[-1]
    from src.detectors.training import _metrics_dict  # existing helper: same metric extraction

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

    if standardized:
        from scripts.gru_recovery import evaluate_run, rebuild
        recovery_results = evaluate_run(run_dir, run_dir / "evaluation")
        restored_final, _ = rebuild(run_dir / "gru_final.pt")
        restored_logits, _ = _predict(restored_final, val_examples, device=device, batch_size=batch_size)
        parity = float(np.abs(restored_logits - final_live_logits).max())
        if parity != 0:
            raise AssertionError(f"Live final/reload parity failed: {parity}")
        recovery_results["checks"]["live_final_reload_max_logit_error"] = parity
        (run_dir / "evaluation" / "evaluation.json").write_text(json.dumps(recovery_results, indent=2) + "\n")

    # Predictions come from the *selected* best checkpoint, which also verifies
    # that the saved checkpoint reconstructs the model.
    best_detector = create_detector(config.model)
    restore_checkpoint(best_path, best_detector, model_name=config.model.name)
    best_detector.to(device)
    logits, scores = _predict(best_detector, val_examples, device=device, batch_size=batch_size)

    if standardized:
        expected_hash = eval_history[result.best_epoch - 1]["val_logits_sha256"]
        if hashlib.sha256(logits.tobytes()).hexdigest() != expected_hash:
            raise AssertionError("Best checkpoint does not match live selected-epoch predictions")
        recovery_results["checks"]["live_best_reload_exact"] = True
        (run_dir / "evaluation" / "evaluation.json").write_text(json.dumps(recovery_results, indent=2) + "\n")

    manifest_rows = {
        json.loads(line)["window_id"]: json.loads(line)
        for line in (root / "manifests" / "windows.core_val.jsonl").read_text().splitlines()
        if line.strip()
    }
    records: list[dict[str, Any]] = []
    for index, item in enumerate(val_bundle["items"]):
        manifest = manifest_rows.get(str(item["window_id"])) or {}
        parent = manifest.get("parent_refs") or {}
        records.append(
            {
                "window_id": item["window_id"],
                "recording_id": item["recording_id"],
                "dataset_id": item["dataset_id"],
                "generator": item.get("generator"),
                "spoken_language": item.get("spoken_language"),
                "split": manifest.get("split"),
                "label": int(item["label"]),
                "label_name": manifest.get("label_name"),
                "num_frames": int(item["num_frames"]),
                "sample_count": int(item["sample_count"]),
                "prepared_sha256": item.get("prepared_sha256"),
                "speaker_ids": manifest.get("speaker_ids"),
                "source_kind": parent.get("source_kind"),
                "source_reference": parent.get("source_reference"),
                "target_reference": parent.get("target_reference"),
                "source_parent_verification": parent.get("source_parent_verification"),
                "target_parent_verification": parent.get("target_parent_verification"),
                "logit_genuine": float(logits[index, 0]),
                "logit_synthetic": float(logits[index, 1]),
                "synthetic_score": float(scores[index]),
            }
        )
    with (run_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        for record in records:
            row = dict(record)
            row["speaker_ids"] = json.dumps(row["speaker_ids"], sort_keys=True)
            writer.writerow(row)
    with (run_dir / "val_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    from src.scoring.metrics import binary_metrics

    labels = np.array([int(item["label"]) for item in val_bundle["items"]])
    metrics = binary_metrics(torch.from_numpy(scores), torch.from_numpy(labels), threshold=float(config.evaluation.threshold))
    tn, fp, fn, tp = metrics.true_negative, metrics.false_positive, metrics.false_negative, metrics.true_positive
    specificity = tn / (tn + fp) if tn + fp else None
    balanced = (metrics.recall + specificity) / 2 if specificity is not None else None
    overall = {
        "windows": len(records),
        "threshold": float(config.evaluation.threshold),
        "score_direction": "higher score = more synthetic",
        "accuracy": metrics.accuracy,
        "balanced_accuracy": balanced,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "eer": metrics.eer,
        "eer_note": "ranking metric at its own operating point, not a deployed threshold",
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
        "confusion_order": "rows = true label, columns = predicted; order [genuine, synthetic]",
        "genuine_false_alarm_rate": metrics.genuine_false_alarm_rate,
        "spoof_miss_rate": metrics.spoof_miss_rate,
    }
    breakdown = _subgroup_breakdown(records, scores, float(config.evaluation.threshold))
    (run_dir / "metrics.json").write_text(json.dumps(overall, indent=2) + "\n", encoding="utf-8")
    (run_dir / "breakdown.json").write_text(json.dumps(breakdown, indent=2) + "\n", encoding="utf-8")

    _training_report(
        run_dir,
        settings=settings,
        dataset=dataset,
        verification=verification,
        audit=audit,
        history=history,
        result=result,
        overall=overall,
        breakdown=breakdown,
        elapsed_seconds=elapsed,
    )

    status = {
        "status": "completed",
        "started": start,
        "finished": _now(),
        "head_training_seconds": elapsed,
        "timing_note": "fit wall time includes checkpoint writes and, for standardized runs, full train/dev epoch evaluation",
        "feature_regeneration_seconds": 0,
        "device": str(device),
        "selected_epoch": result.best_epoch,
        "selected_metric": {"name": result.monitor, "value": result.best_metric, "mode": result.mode},
        "checkpoint_reload_verified": True,
        "controlled_frame_standardization": standardized,
        "resumed": False,
        "note": "features were reused from the frozen cache; no encoder forward pass during training",
    }
    (run_dir / "run_status.json").write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"run_dir": str(run_dir), "selected_epoch": result.best_epoch, "metrics": overall}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
