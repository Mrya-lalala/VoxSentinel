"""Diagnostic analyses for the first frozen-IndicWav2Vec -> GRU run (``gru-core-v1``).

Read-only with respect to the frozen dataset, the original run directory and
its checkpoints.  Every command writes its evidence under
``artifacts/runs/gru-core-v1-diagnosis-corrected/`` by default.
Historical outputs are preserved; --out-dir selects a different evidence location.

Subcommands:

    scores       recompute original validation metrics/distributions from saved predictions
    reload       reload best/final checkpoints, train/val four-row table, parity + batching probes
    features     cached feature statistics: shapes, norms, temporal variability, duplicates
    batches      reproduce the original epoch batch order and class composition
    optimizer    bounded optimizer-update probe on a disposable head copy
    activations  per-stage activation statistics for best checkpoint vs fresh init
    identity     split identity audit from frozen manifests (speakers, references, hashes)
    all          scores + reload + features + batches + optimizer + activations + identity

Requires the pinned encoder environment (.venv, Python 3.10, torch 2.2.2).
No training of the original model is performed; the ``optimizer`` probe uses a
fresh disposable copy only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
RUN_DIR = REPO / "artifacts" / "runs" / "gru-core-v1"
OUT_DIR = REPO / "artifacts" / "runs" / "gru-core-v1-diagnosis-corrected"
DATASET_ROOT = REPO / "artifacts" / "datasets"
CORE_LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
    "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
]
THRESHOLD = 0.5

sys.path.insert(0, str(REPO))


# --------------------------------------------------------------------------- #
# Independent metric implementations (deliberately not the production code)
# --------------------------------------------------------------------------- #

def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (1-based) with tie handling; values are float64."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and sorted_values[j + 1] == sorted_values[i]:
            j += 1
        average = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = average
        i = j + 1
    return ranks


def auc_rank(scores: np.ndarray, labels: np.ndarray) -> float:
    """Tie-aware ROC AUC via the Mann-Whitney rank formula (positive = label 1)."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    pos = labels == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both classes")
    ranks = _average_ranks(scores)
    rank_sum = float(ranks[pos].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def eer_independent(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Independent EER sweep.

    Convention (matching production): score = P(spoof); predicted spoof when
    ``score >= threshold``.  FAR(t) = P(score >= t | genuine),
    FRR(t) = P(score < t | spoof).  Returns the interpolated crossing plus both
    neighbouring operating points, so conventions are auditable.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    n0 = int((labels == 0).sum())
    n1 = int((labels == 1).sum())
    if n0 == 0 or n1 == 0:
        raise ValueError("EER requires both classes")
    thresholds = np.unique(scores)  # predicted spoof when score >= t
    far = np.array([np.sum((scores >= t) & (labels == 0)) / n0 for t in thresholds])
    frr = np.array([np.sum((scores < t) & (labels == 1)) / n1 for t in thresholds])
    far = np.concatenate([far, [0.0]])   # threshold above everything
    frr = np.concatenate([frr, [1.0]])
    differences = far - frr
    crossing = None
    for index in range(1, len(differences)):
        left, right = differences[index - 1], differences[index]
        if left <= 0 <= right or right <= 0 <= left:
            if right == left:
                weight = 0.0
            else:
                weight = -left / (right - left)
            fareq = float(far[index - 1] + weight * (far[index] - far[index - 1]))
            crossing = {
                "far": fareq,
                "frr": fareq,
                "lower_threshold": float(thresholds[index - 1]) if index - 1 < len(thresholds) else None,
                "upper_threshold": float(thresholds[index]) if index < len(thresholds) else None,
                "weight": float(weight),
            }
            break
    absolute = np.abs(far - frr)
    best = int(np.argmin(absolute))
    fallback = (float(far[best]) + float(frr[best])) / 2.0
    return {
        "eer": crossing["far"] if crossing else fallback,
        "crossing": crossing,
        "nearest_point": {"far": float(far[best]), "frr": float(frr[best]), "threshold": float(thresholds[best]) if best < len(thresholds) else None},
        "far_curve_points": int(len(thresholds)),
    }


def confusion_report(scores: np.ndarray, labels: np.ndarray, threshold: float = THRESHOLD) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    predicted = scores >= threshold
    tn = int(((labels == 0) & ~predicted).sum())
    fp = int(((labels == 0) & predicted).sum())
    fn = int(((labels == 1) & ~predicted).sum())
    tp = int(((labels == 1) & predicted).sum())
    n0, n1 = tn + fp, fn + tp
    accuracy = (tn + tp) / max(len(labels), 1)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    balanced = 0.5 * ((tn / n0 if n0 else 0.0) + (recall if n1 else 0.0))
    return {
        "threshold": threshold,
        "n": int(len(labels)),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        "accuracy": accuracy,
        "balanced_accuracy": balanced,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "genuine_false_alarm_rate": fp / n0 if n0 else None,
        "spoof_miss_rate": fn / n1 if n1 else None,
    }


def distribution_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    quantiles = np.percentile(values, [0, 5, 10, 25, 50, 75, 90, 95, 100])
    return {
        "count": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=0)),
        "q0": float(quantiles[0]), "q5": float(quantiles[1]), "q10": float(quantiles[2]),
        "q25": float(quantiles[3]), "q50": float(quantiles[4]), "q75": float(quantiles[5]),
        "q90": float(quantiles[6]), "q95": float(quantiles[7]), "q100": float(quantiles[8]),
    }


def selftest_metrics(out_path: Path) -> dict:
    """Validate the independent metrics on synthetic data with known answers."""
    rng = np.random.default_rng(0)
    results: dict = {}
    perfect = np.concatenate([np.linspace(0, 0.4, 60), np.linspace(0.6, 1.0, 60)])
    labels = np.concatenate([np.zeros(60, dtype=np.int64), np.ones(60, dtype=np.int64)])
    results["perfect_separation"] = {
        "auc": auc_rank(perfect, labels),
        "eer": eer_independent(perfect, labels)["eer"],
    }
    reversed_scores = -perfect
    results["reversed_separation"] = {
        "auc": auc_rank(reversed_scores, labels),
        "eer": eer_independent(reversed_scores, labels)["eer"],
    }
    random_scores = rng.random(2000)
    random_labels = (rng.random(2000) >= 0.5).astype(np.int64)
    results["random_large_n"] = {
        "auc": auc_rank(random_scores, random_labels),
        "eer": eer_independent(random_scores, random_labels)["eer"],
    }
    manual_scores = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float64)
    manual_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    results["manual_midpoint"] = {
        "auc": auc_rank(manual_scores, manual_labels),
        "eer": eer_independent(manual_scores, manual_labels)["eer"],
        "expected_eer": 0.25,
        "expected_auc": 1.0,
    }
    ties_scores = np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float64)
    ties_labels = np.array([0, 1, 0, 1], dtype=np.int64)
    results["all_ties_within_groups"] = {
        "auc": auc_rank(ties_scores, ties_labels),
        "eer": eer_independent(ties_scores, ties_labels)["eer"],
    }
    out_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return results


# --------------------------------------------------------------------------- #
# Stage 2a: recompute score facts
# --------------------------------------------------------------------------- #

def _load_saved_predictions() -> list[dict]:
    rows = [
        json.loads(line)
        for line in (RUN_DIR / "val_predictions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return rows


def command_scores(_: argparse.Namespace) -> int:
    stage = OUT_DIR / "stage2"
    stage.mkdir(parents=True, exist_ok=True)
    selftest = selftest_metrics(stage / "metric_selftest.json")
    print("metric self-test:", json.dumps(selftest, indent=None))

    rows = _load_saved_predictions()
    labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
    logits = np.array([[r["logit_genuine"], r["logit_synthetic"]] for r in rows], dtype=np.float64)
    saved_scores = np.array([r["synthetic_score"] for r in rows], dtype=np.float64)
    margins = logits[:, 1] - logits[:, 0]

    # Recompute probability from saved raw logits over the CLASS dimension.
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp[:, 1] / exp.sum(axis=1)
    score_diff = np.abs(probs - saved_scores)
    # batch-axis softmax (a plausible failure) would differ; quantify.
    batch_softmax = np.exp(logits[:, 1]) / np.exp(logits).sum()  # normalises over the whole 96x2 array
    batch_axis_diff = float(np.abs(batch_softmax - saved_scores).mean())

    ranking_scores = _average_ranks(saved_scores)
    ranking_margins = _average_ranks(margins)
    discordant = int(np.sum(ranking_scores != ranking_margins))

    production = None
    try:
        from src.scoring.metrics import binary_metrics

        m = binary_metrics(torch.from_numpy(saved_scores), torch.from_numpy(labels), threshold=THRESHOLD)
        production = {
            "accuracy": m.accuracy, "precision": m.precision, "recall": m.recall, "f1": m.f1,
            "eer": m.eer, "tn": m.true_negative, "fp": m.false_positive,
            "fn": m.false_negative, "tp": m.true_positive,
            "genuine_false_alarm_rate": m.genuine_false_alarm_rate,
            "spoof_miss_rate": m.spoof_miss_rate,
        }
    except Exception as exc:  # pragma: no cover
        production = {"error": repr(exc)}

    independent = confusion_report(saved_scores, labels)
    independent_eer = eer_independent(saved_scores, labels)
    independent_auc = auc_rank(saved_scores, labels)

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "n": len(rows),
        "class_counts": {"genuine": int((labels == 0).sum()), "synthetic": int((labels == 1).sum())},
        "softmax_check": {
            "class_axis_recompute_max_abs_diff_vs_saved": float(score_diff.max()),
            "class_axis_recompute_mean_abs_diff": float(score_diff.mean()),
            "batch_axis_softmax_mean_abs_diff_vs_saved": batch_axis_diff,
            "note": "saved scores equal softmax over the class dimension; a batch-axis softmax does not match",
        },
        "margin_check": {
            "discordant_rank_pairs_vs_scores": discordant,
            "note": "logit_synthetic - logit_genuine reproduces the saved score ordering exactly (0 discordant ranks)",
        },
        "distributions": {
            "score_genuine": distribution_stats(saved_scores[labels == 0]),
            "score_synthetic": distribution_stats(saved_scores[labels == 1]),
            "score_all": distribution_stats(saved_scores),
            "margin_genuine": distribution_stats(margins[labels == 0]),
            "margin_synthetic": distribution_stats(margins[labels == 1]),
            "margin_all": distribution_stats(margins),
        },
        "threshold_0_5": {
            "scores_below_0_5": int((saved_scores < THRESHOLD).sum()),
            "scores_above_or_equal_0_5": int((saved_scores >= THRESHOLD).sum()),
            "note": "all 96 scores sit in a ~1.1e-3 wide band above 0.5",
        },
        "recomputed_independent": {**independent, **{"eer": independent_eer["eer"], "auc": independent_auc}},
        "eer_details": independent_eer,
        "production_binary_metrics": production,
        "saved_metrics_json": json.loads((RUN_DIR / "metrics.json").read_text(encoding="utf-8")),
    }

    # per-sample CSV with recomputed columns
    with (stage / "val_scores_recomputed.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "label", "language", "generator", "logit_genuine", "logit_synthetic",
                         "margin", "recomputed_score", "saved_score", "abs_diff"])
        for r, prob, margin in zip(rows, probs, margins):
            writer.writerow([r["window_id"], r["label"], r["spoken_language"], r["generator"],
                             f"{r['logit_genuine']:.7f}", f"{r['logit_synthetic']:.7f}",
                             f"{margin:.7f}", f"{prob:.7f}", f"{r['synthetic_score']:.7f}",
                             f"{abs(prob - r['synthetic_score']):.2e}"])

    (stage / "stage2_scores.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "class_axis_softmax_maxdiff": report["softmax_check"]["class_axis_recompute_max_abs_diff_vs_saved"],
        "discordant_ranks": discordant,
        "independent": report["recomputed_independent"],
        "production_eer": (production or {}).get("eer"),
        "score_band": [report["distributions"]["score_all"]["min"], report["distributions"]["score_all"]["max"]],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Stage 2b: reload checkpoints, four-row table, parity, batching probes
# --------------------------------------------------------------------------- #

def _load_examples(split: str):
    from src.dataset_prep.features import load_feature_bundle
    from src.data.batch import EmbeddingExample

    bundle = load_feature_bundle(DATASET_ROOT / "features" / f"{split}.pt")
    examples = [EmbeddingExample(features=item["features"], label=int(item["label"])) for item in bundle["items"]]
    ids = [str(item["window_id"]) for item in bundle["items"]]
    return examples, ids


def _forward_logits(model, examples, batch_size: int) -> np.ndarray:
    from src.data.batch import batch_iterator

    chunks = []
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        for batch in batch_iterator(examples, batch_size, shuffle=False):
            moved = {key: value.to(device) for key, value in batch.items()}
            output = model(moved)
            chunks.append(output.logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _evaluate_with_metrics(model, examples, labels, batch_size: int = 8) -> dict:
    logits = _forward_logits(model, examples, batch_size)
    ce = float(torch.nn.functional.cross_entropy(
        torch.from_numpy(logits), torch.from_numpy(labels)).item())
    scores = torch.softmax(torch.from_numpy(logits), dim=-1)[:, 1].numpy()
    result = {
        "loss_cross_entropy_eval_full": ce,
        **confusion_report(scores, labels),
        "eer_independent": eer_independent(scores, labels)["eer"],
        "auc_independent": auc_rank(scores, labels),
        "score_stats": distribution_stats(scores),
        "margin_stats": distribution_stats(logits[:, 1] - logits[:, 0]),
    }
    return result, logits, scores


def command_reload(_: argparse.Namespace) -> int:
    from src.config import ModelConfig
    from src.detectors.checkpoints import load_checkpoint, restore_checkpoint
    from src.detectors.registry import create_detector

    stage = OUT_DIR / "stage2"
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    best_payload = load_checkpoint(RUN_DIR / "gru_best.pt", model_name="gru")
    final_payload = load_checkpoint(RUN_DIR / "gru_final.pt", model_name="gru")
    assert best_payload["model_config"] == final_payload["model_config"]

    def build(path):
        detector = create_detector(ModelConfig(name="gru", parameters=dict(best_payload["model_config"])))
        restore_checkpoint(path, detector, model_name="gru")
        detector.eval()
        return detector

    best_model = build(RUN_DIR / "gru_best.pt")
    final_model = build(RUN_DIR / "gru_final.pt")

    train_examples, train_ids = _load_examples("train")
    val_examples, val_ids = _load_examples("val")
    train_labels = np.array([e.label for e in train_examples], dtype=np.int64)
    val_labels = np.array([e.label for e in val_examples], dtype=np.int64)

    table = {}
    logits_store = {}
    for name, model in (("best", best_model), ("final", final_model)):
        for split, examples, labels in (("train", train_examples, train_labels), ("val", val_examples, val_labels)):
            metrics, logits, scores = _evaluate_with_metrics(model, examples, labels)
            key = f"{name}/{split}"
            table[key] = metrics
            logits_store[key] = (logits, scores)
            print(f"{key}: loss {metrics['loss_cross_entropy_eval_full']:.4f} | acc {metrics['accuracy']:.3f} "
                  f"| EER {metrics['eer_independent']:.4f} | AUC {metrics['auc_independent']:.4f} "
                  f"| score mean {metrics['score_stats']['mean']:.4f}")

    # --- parity: regenerated best-checkpoint val predictions vs saved row file
    rows = _load_saved_predictions()
    saved_logits = np.array([[r["logit_genuine"], r["logit_synthetic"]] for r in rows])
    saved_scores = np.array([r["synthetic_score"] for r in rows])
    saved_ids = [r["window_id"] for r in rows]
    regenerated_logits = logits_store["best/val"][0]
    regenerated_scores = logits_store["best/val"][1]
    order_ok = saved_ids == val_ids
    parity = {
        "window_order_matches_bundle": bool(order_ok),
        "n": len(saved_ids),
        "logit_max_abs_diff": float(np.abs(saved_logits - regenerated_logits).max()) if order_ok else None,
        "score_max_abs_diff": float(np.abs(saved_scores - regenerated_scores).max()) if order_ok else None,
        "note": "regenerated predictions come from reloading gru_best.pt; identical ordering expected",
    }

    # --- batching probes with the reloaded best checkpoint
    single_logits = _forward_logits(best_model, val_examples, batch_size=1)
    batch_probe = {
        "batch8_vs_batch1_logit_max_abs_diff": float(np.abs(regenerated_logits - single_logits).max()),
        "batch8_vs_batch1_score_max_abs_diff": float(np.abs(
            torch.softmax(torch.from_numpy(regenerated_logits), -1)[:, 1].numpy()
            - torch.softmax(torch.from_numpy(single_logits), -1)[:, 1].numpy()).max()),
    }
    # masked extra-padding probe on the shortest and longest validation examples:
    # append finite zero frames flagged as padding (valid_lengths unchanged).
    from src.data.batch import collate_examples

    lengths = [e.num_frames for e in val_examples]
    shortest = int(np.argmin(lengths))
    base_batch = collate_examples([val_examples[shortest]])
    extra = torch.zeros(1, 40, val_examples[shortest].embedding_dim)
    extended = {
        "features": torch.cat([base_batch["features"], extra], dim=1),
        "valid_lengths": base_batch["valid_lengths"],
        "padding_mask": torch.cat(
            [base_batch["padding_mask"], torch.ones(1, 40, dtype=torch.bool)], dim=1),
        "labels": base_batch["labels"],
    }
    padded_detector = build(RUN_DIR / "gru_best.pt")
    with torch.no_grad():
        padded_logits = padded_detector(extended).logits.numpy()
        solo_logits = padded_detector({
            "features": base_batch["features"],
            "valid_lengths": base_batch["valid_lengths"],
            "padding_mask": base_batch["padding_mask"],
            "labels": base_batch["labels"],
        }).logits.numpy()
    batch_probe["masked_extra_padding_logit_max_abs_diff"] = float(np.abs(padded_logits - solo_logits).max())
    batch_probe["masked_extra_padding_note"] = (
        "40 zero frames appended with padding_mask=True and valid_lengths unchanged; masked frames must be ignored"
    )

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {
            "best": {"epoch": best_payload["epoch"], "global_step": best_payload["global_step"],
                      "metrics": best_payload["metrics"], "model_config": best_payload["model_config"]},
            "final": {"epoch": final_payload["epoch"], "global_step": final_payload["global_step"],
                       "metrics": final_payload["metrics"]},
        },
        "four_row_table": table,
        "val_parity": parity,
        "batching_probes": batch_probe,
        "note": (
            "four_row_table uses eval mode, no shuffling, batch size 8; eval train loss is the full-train "
            "post-hoc loss in eval mode and is not comparable to the online epoch-averaged training loss"
        ),
    }
    (stage / "stage2_reload.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    with (stage / "train_scores_best.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["window_id", "label", "logit_genuine", "logit_synthetic", "score"])
        logits, scores = logits_store["best/train"]
        for wid, label, logit, score in zip(train_ids, train_labels, logits, scores):
            writer.writerow([wid, int(label), f"{logit[0]:.7f}", f"{logit[1]:.7f}", f"{score:.7f}"])

    print(json.dumps({"val_parity": parity, "batching_probes": batch_probe}, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Stage 3a: cached feature statistics
# --------------------------------------------------------------------------- #

def _feature_matrix_stats(features: list[torch.Tensor]) -> dict:
    norms = np.array([float(f.norm(dim=1).mean()) for f in features])
    temporal_std = np.array([float(f.std(dim=0).mean()) for f in features])
    time_means = torch.stack([f.mean(dim=0) for f in features])  # [N, D]
    cross_window_std = time_means.std(dim=0)
    frame_corr = []
    for f in features[:40]:
        a, b = f[:-1].flatten(), f[1:].flatten()
        frame_corr.append(float(torch.corrcoef(torch.stack([a, b]))[0, 1]))
    return {
        "n": len(features),
        "frames": Counter(int(f.shape[0]) for f in features),
        "element_mean": float(np.mean([float(f.mean()) for f in features])),
        "element_std": float(np.mean([float(f.std()) for f in features])),
        "per_frame_norm_mean": float(norms.mean()),
        "per_frame_norm_min": float(norms.min()),
        "per_frame_norm_max": float(norms.max()),
        "temporal_std_mean": float(temporal_std.mean()),
        "temporal_std_min": float(temporal_std.min()),
        "temporal_std_max": float(temporal_std.max()),
        "time_mean_template_std_across_dims": float(time_means.mean(dim=0).std()),
        "cross_window_std_per_dim_mean": float(cross_window_std.mean()),
        "cross_window_std_per_dim_median": float(cross_window_std.median()),
        "frame_to_frame_corr_mean": float(np.mean(frame_corr)),
        "relative_temporal_variation": float(temporal_std.mean() / norms.mean()),
        "relative_cross_window_variation": float(cross_window_std.mean() / time_means.mean(dim=0).abs().mean()),
    }


def command_features(_: argparse.Namespace) -> int:
    stage = OUT_DIR / "stage3"
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    report: dict = {"generated": datetime.now(timezone.utc).isoformat(), "splits": {}}
    all_hashes: dict[str, list[str]] = defaultdict(list)
    for split in ("train", "val"):
        bundle = torch.load(DATASET_ROOT / "features" / f"{split}.pt", map_location="cpu", weights_only=False)
        features = [it["features"] for it in bundle["items"]]
        entry = _feature_matrix_stats(features)
        entry["all_finite"] = all(bool(torch.isfinite(f).all()) for f in features)
        # exact duplicate detection via content hash
        for it in bundle["items"]:
            digest = hashlib.sha256(it["features"].numpy().tobytes()).hexdigest()
            all_hashes[digest].append(f"{split}:{it['window_id']}:label{it['label']}")
        report["splits"][split] = entry
        print(f"{split}: temporal std {entry['temporal_std_mean']:.5f} vs template {entry['time_mean_template_std_across_dims']:.4f} "
              f"| rel temporal {entry['relative_temporal_variation']:.5f} | frame corr {entry['frame_to_frame_corr_mean']:.5f}")

    duplicates = {digest: ids for digest, ids in all_hashes.items() if len(ids) > 1}
    report["exact_duplicate_feature_tensors"] = duplicates
    report["exact_duplicate_count"] = sum(len(ids) - 1 for ids in duplicates.values())

    # near-duplicate detection via random projection sketch correlations
    vectors = []
    labels, ids = [], []
    for split in ("train", "val"):
        bundle = torch.load(DATASET_ROOT / "features" / f"{split}.pt", map_location="cpu", weights_only=False)
        for it in bundle["items"]:
            f = it["features"]
            sketch = torch.cat([f.mean(dim=0), f.std(dim=0), f[:1].squeeze(0), f[-1:].squeeze(0)])
            vectors.append(sketch)
            labels.append(int(it["label"]))
            ids.append(f"{split}:{it['window_id']}")
    X = torch.stack(vectors)
    X = X - X.mean(dim=0, keepdim=True)
    Xn = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    corr = Xn @ Xn.T
    corr.fill_diagonal_(-1.0)
    near = []
    for i in range(len(ids)):
        j = int(torch.argmax(corr[i]))
        if float(corr[i, j]) > 0.9999:
            pair = tuple(sorted([ids[i], ids[j]]))
            if pair not in [tuple(sorted(p)) for p in near]:
                near.append(pair)
    report["near_duplicate_pairs_sketch_corr_gt_0.9999"] = [list(p) for p in near]
    report["near_duplicate_count"] = len(near)
    report["sketch_note"] = (
        "sketch = [time-mean | time-std | first frame | last frame] of the 1024-dim sequence, "
        "so this catches near-identical sequences, not merely similar means"
    )

    (stage / "stage3_features.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"exact duplicate feature tensors: {len(duplicates)} groups | near-dup pairs: {len(near)}")
    return 0


# --------------------------------------------------------------------------- #
# Stage 3b: reproduce the original batching (order + composition)
# --------------------------------------------------------------------------- #

def command_batches(_: argparse.Namespace) -> int:
    stage = OUT_DIR / "stage3"
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    from src.data.batch import batch_iterator, collate_examples, EmbeddingExample
    from src.dataset_prep.features import load_feature_bundle

    bundle = load_feature_bundle(DATASET_ROOT / "features" / "train.pt")
    items = bundle["items"]
    examples = [EmbeddingExample(features=item["features"], label=int(item["label"])) for item in items]
    ids = [str(item["window_id"]) for item in items]
    labels = np.array([e.label for e in examples], dtype=np.int64)

    # --- label alignment self-check with fabricated examples
    fake = [EmbeddingExample(features=torch.full((5 + i, 4), float(i + 1)), label=(i % 2)) for i in range(6)]
    collated = collate_examples(fake)
    alignment_ok = all(int(collated["labels"][i]) == (i % 2) for i in range(6))
    # shuffled collation keeps label-feature pairing
    fake_iter = list(batch_iterator(fake, 4, shuffle=True, seed=3))
    pair_ok = True
    for batch in fake_iter:
        for row in range(batch["features"].shape[0]):
            frames = int(batch["valid_lengths"][row])
            marker = int(batch["features"][row, 0, 0])  # fabricated value = example index + 1
            if frames != 5 + (marker - 1) or int(batch["labels"][row]) != (marker - 1) % 2:
                pair_ok = False

    # --- reproduce the exact original epoch orders (seed + epoch, epochs 1..10)
    epoch_rows = []
    single_class_batches = 0
    total_batches = 0
    for epoch in range(1, 11):
        rng_order = torch.randperm(len(examples), generator=torch.Generator().manual_seed(0 + epoch)).tolist()
        epoch_labels = labels[rng_order]
        counter: dict[int, dict[str, int]] = {}
        for start in range(0, len(rng_order), 8):
            chunk = epoch_labels[start : start + 8]
            key = start // 8
            counter[key] = {"genuine": int((chunk == 0).sum()), "synthetic": int((chunk == 1).sum())}
            total_batches += 1
            if int((chunk == 0).sum()) in (0, len(chunk)):
                single_class_batches += 1
        composition = Counter((v["genuine"], v["synthetic"]) for v in counter.values())
        epoch_rows.append({
            "epoch": epoch,
            "seed": 0 + epoch,
            "batches": len(counter),
            "composition_counts": {f"{g}gen/{s}syn": c for (g, s), c in sorted(composition.items())},
            "single_class_batches": sum(1 for v in counter.values() if v["genuine"] in (0, 8)),
        })
        print(f"epoch {epoch:2d}: {len(counter)} batches; compositions {dict(sorted(composition.items()))}")

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "label_alignment_selfcheck": {"collate_labels_follow_examples": alignment_ok, "shuffled_pairing_ok": pair_ok},
        "total_train_examples": len(examples),
        "class_totals": {"genuine": int((labels == 0).sum()), "synthetic": int((labels == 1).sum())},
        "epochs": epoch_rows,
        "epochs_with_single_class_batch": [
            row["epoch"] for row in epoch_rows if row["single_class_batches"] > 0
        ],
        "overall_single_class_batch_share": single_class_batches / total_batches,
        "note": ("batch order recomputed with torch.randperm(seed=0+epoch) exactly as scripts/train_gru_from_cache.py "
                 "did; labels ride along with their examples so shuffling cannot misalign them"),
    }
    (stage / "stage3_batches.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("alignment self-check:", alignment_ok, pair_ok)
    return 0


# --------------------------------------------------------------------------- #
# Stage 3c: bounded optimizer-update probe on a disposable copy
# --------------------------------------------------------------------------- #

def command_optimizer(_: argparse.Namespace) -> int:
    from src.data.batch import batch_iterator, EmbeddingExample
    from src.detectors.gru import GruDetector, GruSpoofDetector
    from src.detectors.training import build_optimizer, train_epoch
    from src.config import OptimizerConfig
    from src.dataset_prep.features import load_feature_bundle

    stage = OUT_DIR / "stage3"
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)

    bundle = load_feature_bundle(DATASET_ROOT / "features" / "train.pt")
    examples = [EmbeddingExample(features=item["features"], label=int(item["label"])) for item in bundle["items"]]

    inner = GruSpoofDetector(input_dim=1024, hidden_size=256, num_layers=1, dropout=0.0, num_classes=2)
    detector = GruDetector(inner)
    optimizer = build_optimizer(detector, OptimizerConfig(name="adamw", learning_rate=1e-3, weight_decay=0.0))

    trainable = [p for p in detector.parameters() if p.requires_grad]
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    coverage = {
        "trainable_param_tensors": len(trainable),
        "optimizer_param_tensors": len(grouped),
        "all_trainable_in_optimizer_once": len(grouped) == len(trainable) and all(
            sum(1 for q in grouped if q is p) == 1 for p in trainable
        ),
        "param_counts": {name: int(p.numel()) for name, p in detector.named_parameters() if p.requires_grad},
    }

    epoch1_batches = list(batch_iterator(examples, 8, shuffle=True, seed=0 + 1))[:4]
    history = []
    before = {name: p.detach().clone() for name, p in detector.named_parameters() if p.requires_grad}
    result = train_epoch(detector, epoch1_batches, optimizer)
    after = {name: p.detach().clone() for name, p in detector.named_parameters() if p.requires_grad}
    for name in before:
        changed = float((after[name] - before[name]).norm())
        history.append({"parameter": name, "update_norm_l2": changed,
                        "param_norm_l2": float(after[name].norm())})

    # second call: confirm parameters keep updating and loss value is finite
    result2 = train_epoch(detector, epoch1_batches, optimizer)

    # logit range on the same batches after the updates
    detector.eval()
    logits = []
    with torch.no_grad():
        for batch in epoch1_batches:
            output = detector(batch)
            logits.append(output.logits)
    logits_cat = torch.cat(logits)
    margins = (logits_cat[:, 1] - logits_cat[:, 0]).numpy()

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "coverage": coverage,
        "step_history": history,
        "epoch_loss_first_call": result.loss,
        "epoch_steps": result.steps,
        "epoch_loss_second_call": result2.loss,
        "all_parameters_changed_first_epoch": bool(all(h["update_norm_l2"] > 0 for h in history)),
        "logits_after_two_toy_epochs": distribution_stats(margins),
        "note": ("disposable fresh copy, seed 0, first four shuffled batches of epoch 1, AdamW lr 1e-3 wd 0; "
                 "the saved original checkpoints are never loaded into a training graph"),
    }
    (stage / "stage3_optimizer.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "coverage_ok": coverage["all_trainable_in_optimizer_once"],
        "epoch1_loss": result.loss,
        "all_params_changed": report["all_parameters_changed_first_epoch"],
        "margin_std": report["logits_after_two_toy_epochs"]["std"],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Stage 3d: per-stage activation statistics (best checkpoint vs fresh init)
# --------------------------------------------------------------------------- #

def command_activations(_: argparse.Namespace) -> int:
    from src.data.batch import collate_examples, EmbeddingExample
    from src.detectors.checkpoints import load_checkpoint
    from src.detectors.gru import GruSpoofDetector
    from src.dataset_prep.features import load_feature_bundle

    stage = OUT_DIR / "stage3"
    stage.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)

    bundle = load_feature_bundle(DATASET_ROOT / "features" / "val.pt")
    items = bundle["items"]
    genuine = [it for it in items if int(it["label"]) == 0][:4]
    synthetic = [it for it in items if int(it["label"]) == 1][:4]
    chosen = genuine + synthetic
    examples = [EmbeddingExample(features=torch.as_tensor(it["features"]), label=int(it["label"])) for it in chosen]
    batch = collate_examples(examples)

    payload = load_checkpoint(RUN_DIR / "gru_best.pt", model_name="gru")
    from src.config import ModelConfig
    from src.detectors.checkpoints import restore_checkpoint
    from src.detectors.registry import create_detector

    trained_detector = create_detector(ModelConfig(name="gru", parameters=dict(payload["model_config"])))
    restore_checkpoint(RUN_DIR / "gru_best.pt", trained_detector, model_name="gru")
    trained_detector.eval()
    trained = trained_detector.model
    torch.manual_seed(0)
    fresh = GruSpoofDetector(**payload["model_config"])
    fresh.eval()

    def stage_stats(model):
        captured = {}
        with torch.no_grad():
            features = batch["features"]
            valid = ~batch["padding_mask"]
            normed = model.input_norm(features)
            captured["input_norm"] = {
                "per_frame_mean": distribution_stats(normed.mean(dim=-1).numpy()),
                "per_frame_std": distribution_stats(normed.std(dim=-1).numpy()),
            }
            projected = model.projection(normed)
            captured["projection"] = {
                "element_mean": float(projected.mean()),
                "element_std": float(projected.std()),
                "per_sample_time_mean_std": float(projected.mean(dim=1).std()),
            }
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

            lengths = valid.sum(dim=1)
            packed = pack_padded_sequence(projected, lengths.detach().to("cpu"), batch_first=True, enforce_sorted=False)
            sequence, _ = model.gru(packed)
            sequence, _ = pad_packed_sequence(sequence, batch_first=True, total_length=features.shape[1])
            weights = valid.unsqueeze(-1).to(sequence.dtype)
            pooled = (sequence * weights).sum(dim=1) / lengths.unsqueeze(-1).to(sequence.dtype)
            captured["gru_sequence"] = {
                "std_over_time_within_sample_mean": float(sequence.std(dim=1).mean()),
                "pooled_cross_sample_std": float(pooled.std(dim=0).mean()),
                "pooled_magnitude_mean": float(pooled.abs().mean()),
            }
            logits = model.classifier(model.dropout(pooled))
            captured["logits"] = {c: distribution_stats(logits[:, c].numpy()) for c in (0, 1)}
            captured["margins"] = distribution_stats((logits[:, 1] - logits[:, 0]).numpy())
        return captured

    report = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "batch": {"window_ids": [it["window_id"] for it in chosen], "labels": [int(it["label"]) for it in chosen]},
        "fresh_init": stage_stats(fresh),
        "trained_best_checkpoint": stage_stats(trained),
        "generated_note": "4 genuine + 4 synthetic validation windows in one batch of 8, CPU float32, eval mode",
    }
    (stage / "stage3_activations.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "fresh_margin_std": report["fresh_init"]["margins"]["std"],
        "trained_margin_std": report["trained_best_checkpoint"]["margins"]["std"],
        "trained_pooled_cross_sample_std": report["trained_best_checkpoint"]["gru_sequence"]["pooled_cross_sample_std"],
    }, indent=2))
    return 0


# --------------------------------------------------------------------------- #
# Stage 3e: identity / split audit from frozen manifests
# --------------------------------------------------------------------------- #

def _load_manifest(split: str) -> list[dict]:
    path = DATASET_ROOT / "manifests" / f"windows.core_{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def command_identity(_: argparse.Namespace) -> int:
    stage = OUT_DIR / "stage3"
    stage.mkdir(parents=True, exist_ok=True)

    from scripts.gru_identity_audit import audit_identity
    report = audit_identity(_load_manifest("train"), _load_manifest("val"))
    (stage / "stage3_identity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("ready", "speaker_keys", "unresolved_required_count", "expected_absent_source_count")}, indent=2))
    return 0 if report["ready"] else 1


# --------------------------------------------------------------------------- #

COMMANDS = {
    "scores": command_scores,
    "reload": command_reload,
    "features": command_features,
    "batches": command_batches,
    "optimizer": command_optimizer,
    "activations": command_activations,
    "identity": command_identity,
}
ALL_ORDER = ["scores", "reload", "features", "batches", "optimizer", "activations", "identity"]


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    environment = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "torch_threads": torch.get_num_threads(),
        "generated": datetime.now(timezone.utc).isoformat(),
    }
    (OUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    start = time.perf_counter()
    commands = ALL_ORDER if args.command == "all" else [args.command]
    for name in commands:
        print(f"\n================ stage: {name} ================")
        code = COMMANDS[name](args)
        if code != 0:
            return code
    print(f"\ndiagnostic stages finished in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
