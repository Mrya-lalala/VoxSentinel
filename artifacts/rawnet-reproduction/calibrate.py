"""Select one frozen RawNet3 threshold using the reserved calibration folders."""
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
    ready = json.loads((CACHE / "ready.json").read_text())
    index_path = REPO / "artifacts/datasets-ifd-first-run/archive_index.json"
    assert digest(index_path) == ready["provenance"]["index_sha256"]
    index = json.loads(index_path.read_text())
    pilot = json.loads(index_path.with_name("inventory.json").read_text())
    assert not set(index["speaker_groups"]["calibration"]) & set(
        index["speaker_groups"]["train"] + index["speaker_groups"]["dev"]
    )
    checkpoint = ROOT / "batchnorm-diagnostic/improved.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert state["data_sha256"] == digest(CACHE / "ready.json")
    model = create_model().cuda().eval().requires_grad_(False)
    model.load_state_dict(state["model"])
    seen_original = {row["original_sha256"] for rows in ready["splits"].values() for row in rows}
    seen_pcm = {row["pcm_sha256"] for rows in ready["splits"].values() for row in rows}
    records, waves, skips = [], [], Counter()
    started = perf_counter()
    with ExitStack() as stack:
        archives = {}
        for entry in pilot["archives"]:
            path = Path.home() / "Downloads" / entry["name"]
            assert digest(path) == entry["sha256"]
            archives[entry["name"]] = stack.enter_context(ZipFile(path))
        for row in index["rows"]:
            if row["split"] != "calibration":
                continue
            assert row["speaker_folder"] in index["speaker_groups"]["calibration"]
            if row["original_bytes"] > 32 * 1024**2:
                skips["above_size_limit"] += 1
                continue
            blob = archives[row["archive"]].read(row["member"])
            original = sha256(blob).hexdigest()
            if original in seen_original:
                skips["duplicate_original"] += 1
                continue
            try:
                x, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                x = x.mean(1)
                if rate != 16000:
                    x = soxr.resample(x, rate, 16000, quality="HQ")
                if not len(x) or not np.isfinite(x).all():
                    raise ValueError("empty or nonfinite")
            except (RuntimeError, ValueError):
                skips["decode_failure"] += 1
                continue
            x = np.tile(x, (80000 + len(x) - 1) // len(x))[:80000].copy()
            pcm = sha256(x.tobytes()).hexdigest()
            if pcm in seen_pcm:
                skips["duplicate_pcm"] += 1
                continue
            seen_original.add(original)
            seen_pcm.add(pcm)
            waves.append(torch.from_numpy(x))
            records.append({
                "member": row["member"], "speaker_folder": row["speaker_folder"],
                "label": row["label"], "original_sha256": original, "pcm_sha256": pcm,
            })
    scores = []
    with torch.no_grad():
        for start in range(0, len(waves), 16):
            batch = torch.stack(waves[start:start + 16]).cuda()
            scores.append(model(batch).squeeze(1).sigmoid().cpu())
    scores = torch.cat(scores).numpy()
    labels = np.asarray([row["label"] for row in records])
    assert set(labels.tolist()) == {0, 1}
    values = np.unique(scores.astype(np.float64))
    thresholds = np.r_[0., (values[:-1] + values[1:]) / 2, 1.]
    ranking = []
    for threshold in thresholds:
        predictions = scores >= threshold
        far = np.mean(predictions[labels == 0])
        frr = np.mean(~predictions[labels == 1])
        ranking.append((round(float((far + frr) / 2), 12), abs(threshold - .5), threshold))
    threshold = float(min(ranking)[2])
    calibration_metrics = asdict(binary_metrics(torch.from_numpy(scores), torch.from_numpy(labels), threshold))
    dev_labels = torch.tensor([row["label"] for row in ready["splits"]["dev"]])
    development_metrics = asdict(binary_metrics(state["scores"], dev_labels, threshold))
    output = ROOT / "calibration"
    output.mkdir(exist_ok=True)
    result = {
        "checkpoint": str(checkpoint), "checkpoint_sha256": digest(checkpoint),
        "data_sha256": state["data_sha256"], "index_sha256": digest(index_path),
        "implementation_sha256": digest(Path(__file__)),
        "selection": "minimum calibration balanced error; tie closest to .5",
        "threshold": threshold, "counts": dict(Counter(labels.tolist())),
        "calibration": calibration_metrics, "development": development_metrics,
        "skips": dict(skips), "test_audio_loaded": False,
        "elapsed_seconds": perf_counter() - started,
        "calibration_records": records,
        "calibration_scores": scores.tolist(),
    }
    write_json(output / "result.json", result)
    print(json.dumps({key: result[key] for key in (
        "threshold", "counts", "calibration", "development", "skips", "elapsed_seconds"
    )}), flush=True)


if __name__ == "__main__":
    main()
