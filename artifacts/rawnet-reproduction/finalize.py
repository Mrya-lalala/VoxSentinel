"""Audit calibration and save the accepted model and threshold together."""
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

from model import create_model
from prepare import CACHE, ROOT, digest, write_json

sys.path.insert(0, str(ROOT.parents[1]))
from src.scoring.metrics import binary_metrics


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    result_path = ROOT / "calibration/result.json"
    calibration = json.loads(result_path.read_text())
    checkpoint = Path(calibration["checkpoint"])
    assert digest(checkpoint) == calibration["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["data_sha256"] == digest(CACHE / "ready.json")
    ready = json.loads((CACHE / "ready.json").read_text())
    records = calibration["calibration_records"]
    assert len(records) == 773
    original = {row["original_sha256"] for rows in ready["splits"].values() for row in rows}
    pcm = {row["pcm_sha256"] for rows in ready["splits"].values() for row in rows}
    assert not original.intersection(row["original_sha256"] for row in records)
    assert not pcm.intersection(row["pcm_sha256"] for row in records)
    assert len({row["original_sha256"] for row in records}) == len(records)
    assert len({row["pcm_sha256"] for row in records}) == len(records)

    scores = torch.tensor(calibration["calibration_scores"], dtype=torch.float32)
    labels = torch.tensor([row["label"] for row in records])
    threshold = float(np.float32(calibration["threshold"]))
    assert np.isfinite(threshold) and 0 <= threshold <= 1
    targets = labels.numpy()
    values = scores.numpy()

    def balanced_error(cutoff):
        predictions = values >= cutoff
        return float((predictions[targets == 0].mean() + (~predictions[targets == 1]).mean()) / 2)

    # Every distinct float32 decision boundary is checked independently.
    boundaries = np.unique(np.r_[values, np.float32(0), np.float32(1)])
    minimum_error = min(balanced_error(value) for value in boundaries)
    assert abs(balanced_error(threshold) - minimum_error) < 1e-12
    calibrated_metrics = asdict(binary_metrics(scores, labels, threshold))
    for key in ("true_positive", "true_negative", "false_positive", "false_negative"):
        assert calibrated_metrics[key] == calibration["calibration"][key]

    output = ROOT / "final"
    output.mkdir(exist_ok=True)
    payload = {
        "schema": "voxsentinel.rawnet3.calibrated.v1",
        "model": state["model"],
        "model_config": {
            "architecture": "RawNet3", "C": 1024, "model_scale": 8,
            "context": True, "summed": True, "encoder_type": "ECA",
            "nOut": 1, "out_bn": False, "sinc_stride": 10,
            "log_sinc": True, "norm_sinc": "mean",
        },
        "threshold": threshold,
        "threshold_selection": "minimum calibration balanced error; tie closest to .5",
        "labels": {"0": "genuine", "1": "spoof"},
        "decision_rule": "sigmoid(logit) >= threshold means spoof; compare in float32",
        "preprocessing": {
            "sample_rate": 16000, "mono": "arithmetic channel mean",
            "resample": "soxr HQ", "samples": 80000,
            "long_audio": "first five seconds", "short_audio": "repeat to five seconds",
            "dtype": "float32", "gain": "none",
        },
        "metadata": {
            "source_epoch": 37, "epochs_completed": 50,
            "batchnorm": "training-only cumulative statistics, batch 64",
            "source_checkpoint_sha256": calibration["checkpoint_sha256"],
            "data_sha256": state["data_sha256"],
            "calibration_result_sha256": digest(result_path),
            "upstream": json.loads((ROOT / "upstream-provenance.json").read_text()),
            "test_audio_used": False,
        },
    }
    destination = output / "rawnet3_calibrated.pt"
    temporary = destination.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(destination)
    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    assert loaded["threshold"] == threshold
    for name, value in state["model"].items():
        assert torch.equal(value, loaded["model"][name]), name

    model = create_model().cuda().eval().requires_grad_(False)
    model.load_state_dict(loaded["model"], strict=True)
    parts = [torch.load(CACHE / name, map_location="cpu", weights_only=True)
             for name in sorted(ready["parts"]) if name.startswith("dev-")]
    waves = torch.cat([part["waveforms"] for part in parts])
    dev_labels = torch.tensor([row["label"] for part in parts for row in part["records"]])
    replay = []
    with torch.no_grad():
        for start in range(0, len(waves), 16):
            replay.append(model(waves[start:start + 16].cuda()).squeeze(1).sigmoid().cpu())
    replay = torch.cat(replay)
    torch.testing.assert_close(replay, state["scores"], atol=1e-6, rtol=0)
    development = asdict(binary_metrics(replay, dev_labels, threshold))
    summary = {
        "status": "complete_user_accepted",
        "checkpoint": str(destination), "checkpoint_sha256": digest(destination),
        "threshold": threshold, "calibration_examples": len(records),
        "calibration_metrics": calibrated_metrics, "development_examples": len(waves),
        "development_metrics": development,
        "replay_max_absolute_error": float((replay - state["scores"]).abs().max()),
        "state_round_trip_verified": True,
        "threshold_minimum_balanced_error_verified": True,
        "calibration_train_dev_overlap": False,
        "final_test_audio_used": False,
        "decision": "User accepted current performance and requested calibration completion, then stop",
    }
    write_json(output / "verification.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
