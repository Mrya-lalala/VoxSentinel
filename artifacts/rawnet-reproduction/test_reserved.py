"""One frozen-model evaluation of the reserved IFD test partition."""
from collections import Counter
from contextlib import ExitStack
from dataclasses import asdict
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from time import perf_counter
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import soxr
import torch

from model import create_model
from prepare import CACHE, REPO, ROOT, digest, write_json

sys.path.insert(0, str(ROOT.parents[1]))
from src.scoring.metrics import binary_metrics


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = ROOT / "final/rawnet3_calibrated.pt"
    verification = json.loads((ROOT / "final/verification.json").read_text())
    checkpoint_hash = digest(checkpoint)
    assert checkpoint_hash == verification["checkpoint_sha256"]
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    threshold = state["threshold"]
    assert threshold == verification["threshold"]
    ready = json.loads((CACHE / "ready.json").read_text())
    assert digest(CACHE / "ready.json") == state["metadata"]["data_sha256"]
    index_path = REPO / "artifacts/datasets-ifd-first-run/archive_index.json"
    assert digest(index_path) == ready["provenance"]["index_sha256"]
    index = json.loads(index_path.read_text())
    pilot_path = index_path.with_name("inventory.json")
    assert digest(pilot_path) == ready["provenance"]["pilot_sha256"]
    pilot = json.loads(pilot_path.read_text())
    calibration = json.loads((ROOT / "calibration/result.json").read_text())
    assert digest(ROOT / "calibration/result.json") == state["metadata"]["calibration_result_sha256"]
    folders = set(index["speaker_groups"]["test"])
    assert not folders.intersection(
        index["speaker_groups"]["train"] + index["speaker_groups"]["dev"]
        + index["speaker_groups"]["calibration"]
    )
    rows = [row for row in index["rows"] if row["split"] == "test"]
    assert len(rows) == 1903
    assert Counter(row["label"] for row in rows) == {0: 593, 1: 1310}
    known = [row for group in ready["splits"].values() for row in group]
    known += calibration["calibration_records"]
    original_hashes = {row["original_sha256"] for row in known}
    pcm_hashes = {row["pcm_sha256"] for row in known}
    output = ROOT / "final-test"
    output.mkdir(exist_ok=True)
    result_path = output / "result.json"
    if result_path.exists():
        raise RuntimeError("Test evaluation already completed; inspect the recorded result")
    exposure = {
        "status": "started", "checkpoint_sha256": checkpoint_hash,
        "threshold": threshold, "index_sha256": digest(index_path),
        "implementation_sha256": digest(Path(__file__)),
        "test_folders": sorted(folders),
        "purpose": "Frozen checkpoint evaluation requested by user; no model or threshold selection",
    }
    write_json(output / "exposure.json", exposure)
    records, waves, skips = [], [], []
    started = perf_counter()
    with ExitStack() as stack:
        archives = {}
        for entry in pilot["archives"]:
            path = Path.home() / "Downloads" / entry["name"]
            assert digest(path) == entry["sha256"]
            archives[entry["name"]] = stack.enter_context(ZipFile(path))
        for row in rows:
            assert row["speaker_folder"] in folders
            reason = None
            if row["original_bytes"] > 32 * 1024**2:
                skips.append({**row, "reason": "above_size_limit"})
                continue
            info = archives[row["archive"]].getinfo(row["member"])
            assert info.file_size == row["original_bytes"]
            assert f"{info.CRC:08x}" == row["crc32"]
            blob = archives[row["archive"]].read(info)
            original = sha256(blob).hexdigest()
            if original in original_hashes:
                skips.append({**row, "reason": "duplicate_original"})
                continue
            try:
                x, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                x = x.mean(1)
                if rate != 16000:
                    x = soxr.resample(x, rate, 16000, quality="HQ")
                if not len(x) or not np.isfinite(x).all():
                    raise ValueError("empty or nonfinite")
            except (RuntimeError, ValueError) as error:
                skips.append({**row, "reason": "decode_failure", "error": str(error)})
                continue
            x = np.tile(x, (80000 + len(x) - 1) // len(x))[:80000].copy()
            pcm = sha256(x.tobytes()).hexdigest()
            if pcm in pcm_hashes:
                skips.append({**row, "reason": "duplicate_pcm"})
                continue
            original_hashes.add(original)
            pcm_hashes.add(pcm)
            waves.append(torch.from_numpy(x))
            records.append({
                "archive": row["archive"], "member": row["member"],
                "speaker_folder": row["speaker_folder"], "label": row["label"],
                "original_sha256": original, "pcm_sha256": pcm,
            })
            if len(records) % 512 == 0:
                print("Prepared test clips", len(records), flush=True)
    model = create_model().cuda().eval().requires_grad_(False)
    model.load_state_dict(state["model"], strict=True)
    versions = {name: value._version for name, value in model.state_dict().items()}
    scores = []
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        for start in range(0, len(waves), 16):
            batch = torch.stack(waves[start:start + 16]).cuda()
            values = model(batch).squeeze(1).sigmoid().cpu()
            assert torch.isfinite(values).all()
            scores.append(values)
    scores = torch.cat(scores)
    labels = torch.tensor([row["label"] for row in records])
    metrics = asdict(binary_metrics(scores, labels, threshold))
    balanced_accuracy = (
        metrics["true_negative"] / int((labels == 0).sum())
        + metrics["true_positive"] / int((labels == 1).sum())
    ) / 2
    by_folder = {}
    for folder in sorted(folders):
        positions = [i for i, row in enumerate(records) if row["speaker_folder"] == folder]
        by_folder[folder] = {
            "count": len(positions),
            "metrics": asdict(binary_metrics(scores[positions], labels[positions], threshold)),
        }
    assert all(not module.training for module in model.modules())
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(value._version == versions[name] for name, value in model.state_dict().items())
    assert digest(checkpoint) == checkpoint_hash
    predictions = [
        {**row, "synthetic_score": float(score), "predicted_label": int(score >= threshold)}
        for row, score in zip(records, scores)
    ]
    result = {
        **exposure, "status": "completed", "indexed_count": len(rows),
        "evaluated_count": len(records), "class_counts": dict(Counter(labels.tolist())),
        "metrics": metrics, "balanced_accuracy": balanced_accuracy, "by_folder": by_folder,
        "skips": skips, "skip_counts": dict(Counter(row["reason"] for row in skips)),
        "elapsed_seconds": perf_counter() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "model_parameters_and_buffers_unchanged": True,
        "checkpoint_file_unchanged": True,
        "predictions": predictions,
    }
    write_json(result_path, result)
    write_json(output / "exposure.json", {
        **exposure, "status": "evaluated_benchmark", "result_sha256": digest(result_path),
        "note": "Results exposed; preserve this frozen benchmark record",
    })
    print(json.dumps({key: result[key] for key in (
        "status", "indexed_count", "evaluated_count", "class_counts",
        "metrics", "balanced_accuracy", "skip_counts", "by_folder", "elapsed_seconds"
    )}), flush=True)


if __name__ == "__main__":
    main()
