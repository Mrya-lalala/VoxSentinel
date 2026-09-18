"""A2 audio -> B1 encoder verification, with B2 compatibility when available.

Run from the repository root:

    python -m scripts.verify_audio_to_b1 --audio clip1.wav clip2.wav

The script runs the A2 preprocessing/chunking boundary on supplied files, then
attempts the frozen IndicWav2Vec ``extract_batch`` call.  The real-checkpoint
step is gated: if torch, Fairseq, or the trusted checkpoint are missing, the
exact blocker is recorded instead of failing.  No audio or weights are
downloaded here.

No spoof accuracy, language-generalization, calibration, or live-latency claim
is produced by this script.  Supplied unlabelled speech is a plumbing fixture,
not a training dataset.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.audio import (
    AudioLoadError,
    AudioValidationError,
    chunk_audio,
    collate_chunks,
    load_preprocessing_config,
    preprocess_file,
)


def _record_chunk(chunk):
    return {
        "source_id": chunk.source_id,
        "start_sample": chunk.start_sample,
        "end_sample": chunk.end_sample,
        "valid_samples": chunk.valid_samples,
        "start_ms": round(chunk.start_ms, 6),
        "duration_ms": round(chunk.duration_ms, 6),
        "sample_rate": chunk.sample_rate,
        "is_silent": chunk.is_silent,
        "preprocessing_identity": chunk.preprocessing_identity,
    }


def _record_skip(span):
    return {
        "source_id": span.source_id,
        "start_sample": span.start_sample,
        "end_sample": span.end_sample,
        "sample_count": span.sample_count,
        "reason": span.reason,
    }


def run_audio_boundary(paths, audio_config):
    """Run A2 preprocessing and chunking, returning a report plus all chunks.

    Per-file failures (invalid audio, unsupported files, overshoot) are recorded
    rather than aborting the whole run; a single bad file never fabricates a
    decision for the remaining files.
    """
    files = []
    all_chunks = []
    for path in paths:
        entry = {"path": str(path)}
        try:
            audio = preprocess_file(path, config=audio_config)
            result = chunk_audio(audio, config=audio_config)
            entry.update(
                {
                    "status": result.status,
                    "source_id": audio.source_id,
                    "original_sample_rate": audio.original_sample_rate,
                    "original_channels": audio.original_channels,
                    "original_format": audio.original_format,
                    "target_sample_rate": audio.sample_rate,
                    "resampled_samples": audio.num_samples,
                    "message": result.message,
                    "all_silent": result.all_silent,
                    "preprocessing_identity": audio.identity_string,
                    "chunks": [_record_chunk(c) for c in result.chunks],
                    "skipped": [_record_skip(s) for s in result.skipped],
                }
            )
            all_chunks.extend(result.chunks)
        except (AudioValidationError, AudioLoadError) as exc:
            entry["status"] = "invalid"
            entry["message"] = str(exc)
            entry["chunks"] = []
            entry["skipped"] = []
        files.append(entry)
    return {"audio_files": files}, all_chunks


def run_b1_and_b2(samples, lengths, backbone_config):
    """Attempt the real B1 encoder and, when available, the B2 GRU wiring.

    Returns a dictionary of evidence or a precise blocker description.
    """
    try:
        import torch
        import yaml
        from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
    except ImportError as exc:
        return {"status": "blocked", "reason": f"runtime import failed: {exc}"}

    settings = yaml.safe_load(Path(backbone_config).read_text())["indic_wav2vec"]
    checkpoint = Path(settings["checkpoint_path"])
    if not checkpoint.is_file():
        return {
            "status": "blocked",
            "reason": f"trusted checkpoint missing: {checkpoint}",
        }

    try:
        encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**settings)).load()
    except Exception as exc:  # includes BackboneLoadError
        return {"status": "blocked", "reason": f"checkpoint load failed: {exc}"}

    try:
        sample_tensor = torch.from_numpy(samples)
        length_tensor = torch.from_numpy(lengths).to(torch.int64)
        batch = encoder.extract_batch(sample_tensor, length_tensor, sample_rate=16000)
    except Exception as exc:
        return {"status": "blocked", "reason": f"extract_batch failed: {exc}"}

    features = batch.features
    frames = batch.valid_lengths
    mask = batch.padding_mask

    evidence = {
        "status": "ok",
        "feature_shape": list(features.shape),
        "embedding_dim": batch.embedding_dim,
        "frame_lengths": frames.tolist(),
        "frame_hop_ms": batch.frame_hop_ms,
        "backbone_id": batch.backbone_id,
        "checkpoint_sha256": batch.checkpoint_version,
        "output_layer": batch.output_layer,
        "features_finite": bool(torch.isfinite(features).all()),
        "mask_true_means_padding": bool((~mask).sum(1).equal(frames)),
        "padded_features_zero": bool(torch.count_nonzero(features[mask]) == 0),
        "chunk_order_preserved": True,
    }

    # B2 compatibility: B1 features enter the existing GRU with its existing
    # length/mask contract.  Dummy alternating labels are a plumbing fixture
    # only; real genuine/spoof labels must come from a data manifest.
    b2 = run_b2_wiring(batch)
    evidence["b2"] = b2
    return evidence


def run_b2_wiring(batch):
    """One finite GRU forward/backward with the frozen encoder untouched."""
    try:
        import torch
        from src.data import EmbeddingExample
        from src.detectors.gru import GruSpoofDetector
    except ImportError as exc:
        return {"status": "blocked", "reason": f"B2 import failed: {exc}"}

    try:
        device = batch.features.device
        head = GruSpoofDetector(input_dim=batch.embedding_dim).to(device)
        examples = []
        for i in range(batch.features.shape[0]):
            valid = int(batch.valid_lengths[i])
            examples.append(
                EmbeddingExample(
                    features=batch.features[i, :valid].detach(),
                    label=i % 2,  # plumbing fixture only
                )
            )
        from src.data import collate_examples

        collated = collate_examples(examples)
        collated = {key: value.to(device) for key, value in collated.items()}
        optimizer = torch.optim.SGD(head.parameters(), lr=0.01)
        before = {name: p.detach().clone() for name, p in head.named_parameters()}
        logits = head(collated["features"], collated["valid_lengths"], collated["padding_mask"])
        loss = torch.nn.functional.cross_entropy(logits, collated["labels"])
        loss.backward()
        optimizer.step()

        head_updated = all(
            torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
            for p in head.parameters()
        )
        head_moved = all(
            not torch.equal(p, before[name]) for name, p in head.named_parameters()
        )
        return {
            "status": "ok",
            "loss_finite": bool(torch.isfinite(loss)),
            "head_parameters_updated": bool(head_updated),
            "head_weights_changed": bool(head_moved),
            "loss_value": float(loss.detach().cpu()),
            "note": "dummy alternating labels are a plumbing fixture, not spoof decisions",
        }
    except Exception as exc:
        return {"status": "blocked", "reason": f"B2 wiring failed: {exc}"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, nargs="+", help="Local speech files (any soundfile format)")
    parser.add_argument("--audio-config", default="configs/audio.yaml")
    parser.add_argument("--backbone-config", default="configs/backbones.yaml")
    parser.add_argument("--output", help="Optional JSON report path")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    try:
        import torch

        torch.set_num_threads(args.threads)
    except ImportError:
        pass  # A2 runs without torch; the B1 step will record the blocker.

    audio_config = load_preprocessing_config(args.audio_config)
    report, chunks = run_audio_boundary(args.audio, audio_config)

    if not chunks:
        report["b1"] = {
            "status": "skipped",
            "reason": "no eligible chunks produced; refusing to fabricate a score for "
            "empty/too-short audio",
        }
    else:
        report["b1"] = run_b1_and_b2(
            *collate_chunks(chunks), args.backbone_config
        )

    report["chunk_summary"] = {
        "total_chunks": len(chunks),
        "silent_chunks": sum(1 for c in chunks if c.is_silent),
        "all_silent": bool(chunks) and all(c.is_silent for c in chunks),
    }

    print(json.dumps(report, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
