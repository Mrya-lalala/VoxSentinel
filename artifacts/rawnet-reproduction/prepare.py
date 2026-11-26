"""Hash-bound five-second IFD waveform cache for the RawNet3 experiment."""
from contextlib import ExitStack
from hashlib import sha256
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import soundfile as sf
import soxr
import torch

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
CACHE = ROOT / "waveforms"


def digest(path):
    result = sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temp.replace(path)


def prepare():
    CACHE.mkdir(exist_ok=True)
    index_path = REPO / "artifacts/datasets-ifd-first-run/archive_index.json"
    pilot_path = index_path.with_name("inventory.json")
    dev_path = REPO / "artifacts/mixed-gru-first-run/manifest.json"
    index = json.loads(index_path.read_text())
    pilot = json.loads(pilot_path.read_text())
    mixed = json.loads(dev_path.read_text())
    provenance = {
        "index_sha256": digest(index_path), "pilot_sha256": digest(pilot_path),
        "development_manifest_sha256": digest(dev_path),
        "prepare_sha256": digest(Path(__file__)), "samples": 80000,
        "policy": "mono mean; soxr HQ to 16k; first five seconds; short clips repeat",
        "calibration_and_test_audio_loaded": False,
    }
    ready_path = CACHE / "ready.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        assert ready["provenance"] == provenance
        assert all(digest(CACHE / name) == value for name, value in ready["parts"].items())
        return ready
    groups = index["speaker_groups"]
    assert not set(groups["train"]) & set(groups["dev"])
    splits = {
        "dev": mixed["splits"]["dev"],
        "train": [row for row in index["rows"] if row["split"] == "train"],
    }
    seen_original, seen_pcm = set(), set()
    parts, retained, skips = {}, {}, {}
    with ExitStack() as stack:
        archives = {}
        for entry in pilot["archives"]:
            path = Path.home() / "Downloads" / entry["name"]
            assert digest(path) == entry["sha256"]
            archives[entry["name"]] = stack.enter_context(ZipFile(path))
        for split, rows in splits.items():
            values, records, all_records = [], [], []
            for row in rows:
                if row["original_bytes"] > 32 * 1024**2:
                    skips["above_size_limit"] = skips.get("above_size_limit", 0) + 1
                    continue
                blob = archives[row["archive"]].read(row["member"])
                original = sha256(blob).hexdigest()
                if row.get("original_sha256") is not None:
                    assert row["original_sha256"] == original
                if original in seen_original:
                    skips["duplicate_original"] = skips.get("duplicate_original", 0) + 1
                    continue
                try:
                    x, rate = sf.read(io.BytesIO(blob), dtype="float32", always_2d=True)
                    x = x.mean(1)
                    if rate != 16000:
                        x = soxr.resample(x, rate, 16000, quality="HQ")
                    if not len(x) or not np.isfinite(x).all():
                        raise ValueError("empty or nonfinite")
                except (RuntimeError, ValueError) as error:
                    skips["decode_failure"] = skips.get("decode_failure", 0) + 1
                    continue
                original_samples = len(x)
                x = np.tile(x, (80000 + len(x) - 1) // len(x))[:80000].copy()
                pcm = sha256(x.tobytes()).hexdigest()
                if pcm in seen_pcm:
                    skips["duplicate_pcm"] = skips.get("duplicate_pcm", 0) + 1
                    continue
                seen_original.add(original)
                seen_pcm.add(pcm)
                record = {
                    "member": row["member"], "archive": row["archive"],
                    "label": row["label"], "speaker_folder": row["speaker_folder"],
                    "original_sha256": original, "pcm_sha256": pcm,
                    "original_samples_16k": original_samples,
                }
                values.append(torch.from_numpy(x))
                records.append(record)
                all_records.append(record)
                if len(values) == 128:
                    name = f"{split}-{len(all_records) // 128 - 1:03d}.pt"
                    torch.save({"waveforms": torch.stack(values), "records": records}, CACHE / name)
                    parts[name] = digest(CACHE / name)
                    values, records = [], []
                    print(split, len(all_records), "/", len(rows), flush=True)
            if values:
                name = f"{split}-{len(all_records) // 128:03d}.pt"
                torch.save({"waveforms": torch.stack(values), "records": records}, CACHE / name)
                parts[name] = digest(CACHE / name)
            retained[split] = all_records
    assert len(retained["dev"]) == 200, "development inputs must remain complete"
    ready = {"provenance": provenance, "parts": parts, "splits": retained, "skips": skips}
    write_json(ready_path, ready)
    print("RawNet data ready", {key: len(rows) for key, rows in retained.items()}, skips, flush=True)
    return ready


def load_data():
    ready = prepare()
    data = {}
    for split in ("train", "dev"):
        waves, labels = [], []
        for name in sorted(ready["parts"]):
            if name.startswith(split + "-"):
                part = torch.load(CACHE / name, map_location="cpu", weights_only=True)
                waves.append(part["waveforms"])
                labels.extend(row["label"] for row in part["records"])
        data[split] = torch.cat(waves), torch.tensor(labels, dtype=torch.float32)
    return data, ready


if __name__ == "__main__":
    prepare()
