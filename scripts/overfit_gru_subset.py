"""Stage-4 tiny balanced-subset overfit diagnostic for the GRU head.

Determines whether the SAME head architecture, optimizer and learning rate can
fit a small fixed set of 24 training windows (one genuine + one synthetic per
core language, deterministic seed 0).  Success demonstrates fit capacity on
these samples only; it is not spoof-generalization evidence.

Target (diagnostic convenience): 100 % subset accuracy with cross-entropy
below 0.05 on five consecutive evaluations; runs cap at 500 optimizer steps.
If the lr 1e-3 attempt fails, ONE fallback attempt at lr 1e-4 with the same
subset/seed is allowed (clearly labelled).  No wider search.

Writes into ``artifacts/runs/gru-core-v1-diagnosis/stage4/``:
``subset_manifest.json``, ``attempt1_lr1e-3/{history.json,history.csv,summary.json,model.pt}``,
``attempt2_lr1e-4/...`` if needed.

Run:  .venv/bin/python -m scripts.overfit_gru_subset
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
OUT = REPO / "artifacts" / "runs" / "gru-core-v1-diagnosis" / "stage4"
DATASET_ROOT = REPO / "artifacts" / "datasets"
CORE_LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", "Marathi",
    "Odia", "Punjabi", "Sanskrit", "Tamil", "Telugu", "Urdu",
]
SEED = 0


def select_subset() -> list[dict]:
    from src.dataset_prep.features import load_feature_bundle

    bundle = load_feature_bundle(DATASET_ROOT / "features" / "train.pt")
    items = [dict(item) for item in bundle["items"]]
    rng = random.Random(SEED)
    picks: list[dict] = []
    for language in CORE_LANGUAGES:
        for label in (0, 1):
            candidates = sorted(
                (item for item in items if item["spoken_language"] == language and int(item["label"]) == label),
                key=lambda item: str(item["window_id"]),
            )
            if not candidates:
                raise SystemExit(f"no {label} training window for {language}")
            picks.append(rng.choice(candidates))
    return picks


def train_attempt(picks: list[dict], learning_rate: float, max_steps: int = 500) -> dict:
    from src.config import OptimizerConfig
    from src.data.batch import collate_examples, EmbeddingExample
    from src.detectors.gru import GruDetector, GruSpoofDetector
    from src.detectors.training import build_optimizer

    label = f"lr{learning_rate:g}"
    attempt_dir = OUT / f"attempt-{label}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(4)
    torch.manual_seed(SEED)
    examples = [EmbeddingExample(features=torch.as_tensor(p["features"]), label=int(p["label"])) for p in picks]
    batch = collate_examples(examples)
    labels = batch["labels"]

    inner = GruSpoofDetector(input_dim=1024, hidden_size=256, num_layers=1, dropout=0.0, num_classes=2)
    detector = GruDetector(inner)
    assert inner.dropout.p == 0.0
    optimizer = build_optimizer(detector, OptimizerConfig(name="adamw", learning_rate=learning_rate, weight_decay=0.0))

    def evaluate():
        detector.eval()
        with torch.no_grad():
            output = detector(batch)
            loss = torch.nn.functional.cross_entropy(output.logits, labels)
            predictions = output.logits.argmax(dim=1)
            accuracy = float((predictions == labels).float().mean())
        return float(loss), accuracy

    history = []
    streak = 0
    reached_at = None
    start = time.perf_counter()
    detector.train()
    for step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output = detector(batch)
        loss = torch.nn.functional.cross_entropy(output.logits, labels)
        loss.backward()
        if step % 50 == 1 or step == max_steps:
            grad_norm = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in detector.parameters() if p.grad is not None)))
        else:
            grad_norm = None
        optimizer.step()
        if step % 5 == 0 or step == 1:
            eval_loss, eval_accuracy = evaluate()
            detector.train()
            history.append({
                "step": step,
                "train_loss": float(loss.detach()),
                "eval_loss": eval_loss,
                "eval_accuracy": eval_accuracy,
                "grad_norm": grad_norm,
            })
            if eval_accuracy == 1.0 and eval_loss < 0.05:
                streak += 1
                if streak >= 5 and reached_at is None:
                    reached_at = step
                    break
            else:
                streak = 0
    elapsed = time.perf_counter() - start

    final_loss, final_accuracy = evaluate()
    summary = {
        "learning_rate": learning_rate,
        "steps_run": history[-1]["step"],
        "reached_target": reached_at is not None,
        "reached_target_at_step": reached_at,
        "final_eval_loss": final_loss,
        "final_eval_accuracy": final_accuracy,
        "min_eval_loss": min(row["eval_loss"] for row in history),
        "best_eval_accuracy": max(row["eval_accuracy"] for row in history),
        "elapsed_seconds": elapsed,
        "selection_seed": SEED,
        "note": "diagnostic fit-capacity test on 24 fixed training windows; not generalization evidence",
    }
    (attempt_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    with (attempt_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    (attempt_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    torch.save({"model_state_dict": detector.state_dict(), "history": history,
                "subset_window_ids": [p["window_id"] for p in picks]}, attempt_dir / "model.pt")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    picks = select_subset()
    manifest = {
        "seed": SEED,
        "selection": "one genuine + one synthetic per core language from windows.core_train.jsonl, rng.choice over window_id-sorted candidates",
        "count": len(picks),
        "windows": [
            {"window_id": p["window_id"], "label": int(p["label"]), "spoken_language": p["spoken_language"],
             "dataset_id": p["dataset_id"], "generator": p.get("generator")}
            for p in picks
        ],
    }
    (OUT / "subset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"subset: {len(picks)} windows ({sum(p['label']==0 for p in picks)} genuine / "
          f"{sum(p['label']==1 for p in picks)} synthetic)")

    summary1 = train_attempt(picks, learning_rate=1e-3)
    print("attempt lr1e-3:", json.dumps({k: summary1[k] for k in
          ("reached_target", "steps_run", "final_eval_loss", "final_eval_accuracy", "elapsed_seconds")}, indent=2))

    summaries = {"attempt_lr1e-3": summary1}
    if not summary1["reached_target"]:
        summary2 = train_attempt(picks, learning_rate=1e-4)
        print("attempt lr1e-4:", json.dumps({k: summary2[k] for k in
              ("reached_target", "steps_run", "final_eval_loss", "final_eval_accuracy", "elapsed_seconds")}, indent=2))
        summaries["attempt_lr1e-4"] = summary2
    (OUT / "summary.json").write_text(json.dumps(
        {"generated": datetime.now(timezone.utc).isoformat(), **summaries}, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
