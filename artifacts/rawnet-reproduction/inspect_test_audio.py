"""Describe test audio properties without modifying the frozen detector."""
from collections import Counter, defaultdict
from contextlib import ExitStack
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import soxr

from prepare import REPO, ROOT, digest, write_json

result = json.loads((ROOT / "final-test/result.json").read_text())
pilot = json.loads((REPO / "artifacts/datasets-ifd-first-run/inventory.json").read_text())
properties = []
with ExitStack() as stack:
    archives = {}
    for entry in pilot["archives"]:
        path = Path.home() / "Downloads" / entry["name"]
        assert digest(path) == entry["sha256"]
        archives[entry["name"]] = stack.enter_context(ZipFile(path))
    for row in result["predictions"]:
        blob = archives[row["archive"]].read(row["member"])
        assert sha256(blob).hexdigest() == row["original_sha256"]
        source = io.BytesIO(blob)
        info = sf.info(source)
        source.seek(0)
        x, rate = sf.read(source, dtype="float32", always_2d=True)
        x = x.mean(1)
        duration = len(x) / rate
        if rate != 16000:
            x = soxr.resample(x, rate, 16000, quality="HQ")
        x = np.tile(x, (80000 + len(x) - 1) // len(x))[:80000].copy()
        assert sha256(x.tobytes()).hexdigest() == row["pcm_sha256"]
        power = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        frequencies = np.fft.rfftfreq(len(x), 1 / 16000)
        total = max(power.sum(), 1e-20)
        properties.append({
            "speaker": row["speaker_folder"], "label": row["label"],
            "correct": row["label"] == row["predicted_label"],
            "sample_rate": int(rate), "channels": info.channels, "subtype": info.subtype,
            "duration": duration, "rms": float(np.sqrt(np.mean(x * x))),
            "near_silence_fraction": float(np.mean(np.abs(x) < .001)),
            "clipped_fraction": float(np.mean(np.abs(x) >= .999)),
            "high_frequency_fraction": float(power[frequencies > 4000].sum() / total),
        })
groups = defaultdict(list)
for row in properties:
    groups[row["speaker"], row["label"]].append(row)
summary = {}
for (speaker, label), rows in sorted(groups.items()):
    item = {
        "count": len(rows),
        "sample_rates": dict(Counter(row["sample_rate"] for row in rows)),
        "channels": dict(Counter(row["channels"] for row in rows)),
        "subtypes": dict(Counter(row["subtype"] for row in rows)),
    }
    for key in ("duration", "rms", "near_silence_fraction", "clipped_fraction", "high_frequency_fraction"):
        item["median_" + key] = float(np.median([row[key] for row in rows]))
    summary[f"{speaker}/{'genuine' if label == 0 else 'spoof'}"] = item
write_json(ROOT / "final-test/audio-properties.json", {"summary": summary, "records": properties})
print(json.dumps(summary, indent=2), flush=True)
