"""Stage-3 supplementary probe: separation of classes at each head stage.

Measures a scale-free separation scalar (d' style: between-class mean distance /
within-class isotropic scatter) for:

    1. raw cached time-mean features (1024-d)
    2. projection outputs (256-d time-mean), fresh init
    3. pooled GRU outputs (256-d), fresh init
    4. pooled GRU outputs (256-d), trained best checkpoint

computed on the frozen train (384) and val (96) splits.  Writes
``stage3_fisher.json`` next to the other stage-3 evidence.

Run:  .venv/bin/python artifacts/runs/gru-core-v1-diagnosis/stage3_fisher_probe.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
OUT = Path(__file__).resolve().parent / "stage3" / "stage3_fisher.json"
DATASET_ROOT = REPO / "artifacts" / "datasets"


def separation(matrix: np.ndarray, labels: np.ndarray) -> dict:
    mu0, mu1 = matrix[labels == 0].mean(axis=0), matrix[labels == 1].mean(axis=0)
    c0, c1 = matrix[labels == 0], matrix[labels == 1]
    s0 = float(np.mean(np.var(c0, axis=0)))
    s1 = float(np.mean(np.var(c1, axis=0)))
    d_prime = float(np.linalg.norm(mu1 - mu0) / np.sqrt(s0 + s1 + 1e-30))
    return {
        "d_prime_isotropic": d_prime,
        "between_mean_l2": float(np.linalg.norm(mu1 - mu0)),
        "within_std_avg": float(np.sqrt((s0 + s1) / 2.0)),
        "mean_scalar_std_across_dims": float(matrix.mean(axis=0).std()),
    }


def pooled_time_mean(features: torch.Tensor) -> np.ndarray:
    return features.mean(dim=0).numpy()


def main() -> int:
    from src.config import ModelConfig
    from src.data.batch import batch_iterator, EmbeddingExample
    from src.detectors.checkpoints import restore_checkpoint
    from src.detectors.registry import create_detector
    from src.dataset_prep.features import load_feature_bundle

    torch.set_num_threads(4)
    result: dict = {"generated": datetime.now(timezone.utc).isoformat(), "stages": {}}
    for split in ("train", "val"):
        bundle = load_feature_bundle(DATASET_ROOT / "features" / f"{split}.pt")
        features = [torch.as_tensor(item["features"]) for item in bundle["items"]]
        labels = np.array([int(item["label"]) for item in bundle["items"]], dtype=np.int64)

        raw = np.stack([pooled_time_mean(f) for f in features])
        entry = {"raw_feature_time_mean": separation(raw, labels)}

        torch.manual_seed(0)
        fresh_detector = create_detector(ModelConfig(name="gru", parameters={
            "input_dim": 1024, "hidden_size": 256, "num_layers": 1, "dropout": 0.0, "num_classes": 2,
        }))

        def collect(detector) -> tuple[np.ndarray, np.ndarray]:
            examples = [EmbeddingExample(features=f, label=int(l)) for f, l in zip(features, labels)]
            projected_means, pooled = [], []
            detector.eval()
            with torch.no_grad():
                for batch in batch_iterator(examples, 8, shuffle=False):
                    valid = ~batch["padding_mask"]
                    lengths = valid.sum(dim=1)
                    normed = detector.model.input_norm(batch["features"])
                    proj = detector.model.projection(normed)
                    projected_means.append(proj.mean(dim=1).numpy())
                    from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

                    packed = pack_padded_sequence(proj, lengths, batch_first=True, enforce_sorted=False)
                    seq, _ = detector.model.gru(packed)
                    seq, _ = pad_packed_sequence(seq, batch_first=True, total_length=proj.shape[1])
                    weights = valid.unsqueeze(-1).to(seq.dtype)
                    pooled.append(((seq * weights).sum(dim=1) / lengths.unsqueeze(-1).to(seq.dtype)).numpy())
            return np.concatenate(projected_means), np.concatenate(pooled)

        proj_fresh, pooled_fresh = collect(fresh_detector)
        entry["projection_time_mean_fresh_init"] = separation(proj_fresh, labels)
        entry["pooled_gru_fresh_init"] = separation(pooled_fresh, labels)

        trained = create_detector(ModelConfig(name="gru", parameters={
            "input_dim": 1024, "hidden_size": 256, "num_layers": 1, "dropout": 0.0, "num_classes": 2,
        }))
        restore_checkpoint(REPO / "artifacts/runs/gru-core-v1/gru_best.pt", trained, model_name="gru")
        _, pooled_trained = collect(trained)
        entry["pooled_gru_trained_best"] = separation(pooled_trained, labels)

        result["stages"][split] = entry
        print(f"[{split}]")
        for name, stats in entry.items():
            print(f"  {name:32s} d' {stats['d_prime_isotropic']:.4f} | "
                  f"between {stats['between_mean_l2']:.5f} | within-std {stats['within_std_avg']:.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
