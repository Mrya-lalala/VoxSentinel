"""Stage-5 pooled linear baseline on the frozen gru-core-v1 split.

Diagnostic reference only: mean-pools valid frames (1024 features per window),
optionally applies per-dimension standardization fitted on TRAIN ONLY, and fits
a two-class logistic regression with deterministic PyTorch LBFGS (max 200
iterations) on cross-entropy plus a fixed 1e-4 sum-of-squared-weights penalty
(excluding bias).  The final model is the endpoint of that objective; the
validation split is never used for fitting or selection.

Metrics use the same positive-class convention as the GRU (label 1 =
synthetic) and are reported both via the production metric module and the
independent implementations in ``scripts/diagnose_gru.py``.

Writes into ``artifacts/runs/gru-core-v1-diagnosis/stage5/``:
``summary.json``, ``model_scaled.pt``, ``model_unscaled.pt`` (weights + scaler),
``predictions_{split}.csv`` for the scaled variant.

Run:  .venv/bin/python -m scripts.linear_baseline_gru_cache
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "artifacts" / "runs" / "gru-core-v1-diagnosis" / "stage5"
DATASET_ROOT = REPO / "artifacts" / "datasets"

EPSILON = 1e-6
L2_PENALTY = 1e-4
MAX_ITER = 200


def load_pooled(split: str):
    from src.dataset_prep.features import load_feature_bundle

    bundle = load_feature_bundle(DATASET_ROOT / "features" / f"{split}.pt")
    X = torch.stack([torch.as_tensor(item["features"]).mean(dim=0) for item in bundle["items"]])
    y = torch.tensor([int(item["label"]) for item in bundle["items"]], dtype=torch.long)
    ids = [str(item["window_id"]) for item in bundle["items"]]
    return X, y, ids


def independent_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from scripts.diagnose_gru import auc_rank, confusion_report, eer_independent

    return {
        "auc": auc_rank(scores, labels),
        "eer": eer_independent(scores, labels)["eer"],
        **confusion_report(scores, labels),
    }


def production_metrics(scores: np.ndarray, labels: np.ndarray) -> dict:
    from src.scoring.metrics import binary_metrics

    m = binary_metrics(torch.from_numpy(scores), torch.from_numpy(labels), threshold=0.5)
    return {"eer": m.eer, "accuracy": m.accuracy, "f1": m.f1,
            "tn": m.true_negative, "fp": m.false_positive, "fn": m.false_negative, "tp": m.true_positive}


def fit_logreg(X_train: torch.Tensor, y_train: torch.Tensor) -> dict:
    weight = torch.zeros(X_train.shape[1], requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=MAX_ITER, history_size=10, line_search_fn=None)
    counters = {"closure_calls": 0}

    def closure():
        counters["closure_calls"] += 1
        optimizer.zero_grad()
        raw_response = X_train @ weight + bias
        logits = torch.stack([-raw_response, raw_response], dim=1)
        objective = torch.nn.functional.cross_entropy(logits, y_train) + L2_PENALTY * (weight * weight).sum()
        objective.backward()
        return objective

    optimizer.step(closure)
    state = optimizer.state[weight]
    final_objective = float(closure().detach())
    n_iter = int(state.get("n_iter")) if state.get("n_iter") is not None else None
    with torch.no_grad():
        raw_responses_train = (X_train @ weight + bias).numpy()
    return {
        "weight": weight.detach(), "bias": bias.detach().numpy(),
        "final_objective": final_objective,
        "iterations_reported": n_iter,
        "closure_calls": counters["closure_calls"],
        "objective_definition": f"cross_entropy(logits, y) + {L2_PENALTY:g} * sum(w^2), bias unpenalized",
        "raw_responses_train": raw_responses_train,
    }


def evaluate(model: dict, X: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        raw_response = X @ model["weight"] + float(model["bias"][0])
        logits = torch.stack([-raw_response, raw_response], dim=1)
        margins = (logits[:, 1] - logits[:, 0]).numpy()
        scores = torch.softmax(logits, dim=1)[:, 1].numpy()
        ce = float(torch.nn.functional.cross_entropy(logits, y))
    from scripts.diagnose_gru import auc_rank, eer_independent
    # Ranking on the actual logit difference avoids probability-saturation ties.
    result = independent_metrics(scores, y.numpy())
    result["probability_auc"] = result["auc"]
    result["probability_eer"] = result["eer"]
    result["auc"] = auc_rank(margins, y.numpy())
    result["eer"] = eer_independent(margins, y.numpy())["eer"]
    result["ranking_basis"] = "logit_synthetic - logit_genuine"
    result["unique_probabilities"] = int(len(np.unique(scores)))
    result["unique_margins"] = int(len(np.unique(margins)))
    return {"cross_entropy": ce, "scores": scores, "margins": margins,
            **result, "production": production_metrics(scores, y.numpy())}


def reevaluate_saved(model_path: Path, out: Path) -> int:
    """Correct exports using the saved fitted weights AND scaler; never optimize."""
    import csv
    import hashlib
    from scripts.diagnose_gru import distribution_stats
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    before = hashlib.sha256(model_path.read_bytes()).hexdigest()
    torch.set_num_threads(4)
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    report = {"model_path": str(model_path), "model_sha256": before, "refitted": False,
              "score_definition": "softmax([-raw_response, raw_response])[1] = sigmoid(2*raw_response)",
              "scaler_definition": "saved training-window-mean scaler, sample std (ddof=1); NOT frame scaler",
              "splits": {}}
    for split in ("train", "val"):
        x, y, ids = load_pooled(split)
        result = evaluate(model, (x-model["scaler_mean"])/model["scaler_std"], y)
        scores, margins = result.pop("scores"), result.pop("margins")
        result["score_stats"] = distribution_stats(scores)
        result["margin_stats"] = distribution_stats(margins)
        result["by_class"] = {str(c): {"scores": distribution_stats(scores[y.numpy()==c]),
                                             "margins": distribution_stats(margins[y.numpy()==c])} for c in (0,1)}
        old_scores = torch.sigmoid(torch.from_numpy(margins)/2).numpy()
        result["old_vs_corrected_decisions_changed"] = int(((old_scores>=.5)!=(scores>=.5)).sum())
        result["example"] = {"window_id":ids[0], "old_score":float(old_scores[0]),
                             "corrected_score":float(scores[0]), "logit_margin":float(margins[0])}
        report["splits"][split] = result
        with (out/f"predictions_{split}.csv").open("w",newline="") as f:
            writer=csv.writer(f);writer.writerow(["window_id","label","raw_response","logit_margin","score"])
            writer.writerows((wid,int(label),float(m)/2,float(m),float(s)) for wid,label,m,s in zip(ids,y,margins,scores))
    report["model_unchanged"] = before == hashlib.sha256(model_path.read_bytes()).hexdigest()
    (out/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reevaluate-saved", type=Path)
    parser.add_argument("--out-dir", type=Path, default=REPO / "artifacts/runs/gru-core-v1-diagnosis-corrected/linear")
    args = parser.parse_args()
    if args.reevaluate_saved:
        return reevaluate_saved(args.reevaluate_saved, args.out_dir)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError("Historical baseline directory exists; use --reevaluate-saved and a new --out-dir")
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(0)

    X_train, y_train, ids_train = load_pooled("train")
    X_val, y_val, ids_val = load_pooled("val")
    print(f"pooled: train {tuple(X_train.shape)} val {tuple(X_val.shape)}")

    scaler_mu = X_train.mean(dim=0)
    scaler_sd = X_train.std(dim=0).clamp_min(EPSILON)
    X_train_scaled = (X_train - scaler_mu) / scaler_sd
    X_val_scaled = (X_val - scaler_mu) / scaler_sd
    clamped = int((X_train.std(dim=0) < EPSILON).sum())

    report: dict = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "scaler": {"fitted_on": "train only", "epsilon": EPSILON, "dims_clamped": clamped,
                    "mean_l2": float(scaler_mu.norm()), "sd_min": float(scaler_sd.min()),
                    "sd_max": float(scaler_sd.max())},
        "solver": {"optimizer": "torch.optim.LBFGS", "max_iter": MAX_ITER, "history_size": 10,
                    "l2_penalty": L2_PENALTY, "bias_penalized": False,
                    "seed": 0, "init": "zeros"},
        "variants": {},
    }

    for name, (Xtr, Xva) in (("scaled", (X_train_scaled, X_val_scaled)), ("unscaled", (X_train, X_val))):
        model = fit_logreg(Xtr, y_train)
        train_eval = evaluate(model, Xtr, y_train)
        val_eval = evaluate(model, Xva, y_val)
        report["variants"][name] = {
            "model": {
                "final_objective": model["final_objective"],
                "iterations_reported": model["iterations_reported"],
                "closure_calls": model["closure_calls"],
                "objective_definition": model["objective_definition"],
                "weight_l2": float(model["weight"].norm()), "bias": float(model["bias"][0]),
            },
            "train": {k: v for k, v in train_eval.items() if k not in ("scores", "margins")},
            "val": {k: v for k, v in val_eval.items() if k not in ("scores", "margins")},
            "val_score_stats": {
                "genuine_mean": float(val_eval["scores"][y_val.numpy() == 0].mean()),
                "synthetic_mean": float(val_eval["scores"][y_val.numpy() == 1].mean()),
                "min": float(val_eval["scores"].min()), "max": float(val_eval["scores"].max()),
            },
        }
        print(f"[{name}] obj {model['final_objective']:.4f} | iters {model['iterations_reported']} | "
              f"train CE {train_eval['cross_entropy']:.4f} acc {train_eval['accuracy']:.3f} | "
              f"val CE {val_eval['cross_entropy']:.4f} acc {val_eval['accuracy']:.3f} "
              f"EER {val_eval['eer']:.4f} AUC {val_eval['auc']:.4f}")

        if name == "scaled":
            torch.save({
                "weight": model["weight"], "bias": model["bias"],
                "scaler_mean": scaler_mu, "scaler_std": scaler_sd,
                "epsilon": EPSILON, "l2_penalty": L2_PENALTY, "max_iter": MAX_ITER,
                "fitted_on": "windows.core_train.jsonl pooled time-mean features",
            }, OUT / "model_scaled.pt")
            import csv

            for split, ids, scores, margins, labels in (
                ("train", ids_train, train_eval["scores"], train_eval["margins"], y_train.numpy()),
                ("val", ids_val, val_eval["scores"], val_eval["margins"], y_val.numpy()),
            ):
                with (OUT / f"predictions_{split}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["window_id", "label", "logit_margin", "score"])
                    for wid, label, margin, score in zip(ids, labels, margins, scores):
                        writer.writerow([wid, int(label), f"{margin:.7f}", f"{score:.7f}"])
        if name == "unscaled":
            torch.save({"weight": model["weight"], "bias": model["bias"],
                        "note": "unscaled variant is a diagnostic comparison row"}, OUT / "model_unscaled.pt")

    (OUT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
