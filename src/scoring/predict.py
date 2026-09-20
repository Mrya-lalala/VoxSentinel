"""Versioned local audio-file prediction using a pinned encoder/head pair.

This implements the pilot's one highest-energy window contract. It is not a
whole-recording spoof localization system or a language router.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch

from ..audio.prepare import decode_mono_16k
from ..audio.identity import assert_trained_preprocessing
from ..backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from ..config import ModelConfig
from ..dataset_prep.config import WindowPolicy
from ..dataset_prep.materialize import sha256_file
from ..dataset_prep.windows import select_window
from ..detectors.checkpoints import load_checkpoint, restore_checkpoint
from ..detectors.registry import create_detector


class FilePredictor:
    """Load once; predict local files with an immutable preprocessing contract."""

    def __init__(self, release: str | Path, *, encoder_path: str | Path | None = None):
        self.release_path = Path(release).resolve()
        self.spec = json.loads(self.release_path.read_text())
        if self.spec.get("schema") != "voxsentinel.inference_release.v1":
            raise ValueError("Unsupported prediction release schema")
        assert_trained_preprocessing(self.spec.get("preprocessing_version"))
        self.head_path = self.release_path.parent / self.spec["head"]["file"]
        if sha256_file(self.head_path) != self.spec["head"]["sha256"]:
            raise ValueError("Detector checkpoint hash mismatch")
        payload = load_checkpoint(self.head_path, model_name="gru")
        settings = dict(self.spec["encoder"])
        if encoder_path is not None:
            settings["checkpoint_path"] = str(encoder_path)
        training_identity = payload.get("metadata", {}).get("encoder", {})
        expected = {
            "checkpoint_sha256": settings["checkpoint_sha256"],
            "selected_layer": settings["output_layer"],
            "embedding_dim": settings["expected_embedding_dim"],
            "preprocessing_version": self.spec["preprocessing_version"],
            "dtype": "float32",
        }
        if training_identity != expected:
            raise ValueError("Release encoder/preprocessing differs from detector training identity")
        if settings["device"] != "cpu":
            raise ValueError("This release has only been validated on CPU")
        threshold = float(self.spec["threshold"])
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be within [0,1]")
        self.policy = WindowPolicy(**self.spec["window"])
        self.detector = create_detector(ModelConfig("gru", payload["model_config"])).eval()
        restore_checkpoint(self.head_path, self.detector, model_name="gru")
        self.detector.requires_grad_(False)
        self.encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**settings))
        self.threshold = threshold

    def predict(self, path: str | Path, *, language: str | None = None) -> dict:
        started = time.perf_counter()
        audio = decode_mono_16k(path)
        window = select_window(audio.samples, audio.sample_rate, self.policy)
        base = {
            "schema": "voxsentinel.file_prediction.v1", "release_id": self.spec["release_id"],
            "input_file": str(path), "input_sha256": sha256_file(path),
            "original_audio": audio.original_facts(), "processing": audio.processing_facts(),
            "window": asdict(window), "sample_rate": audio.sample_rate,
            "coverage": "one selected window; not a decision about every region of a long file",
            "language": language,
            "language_in_training_scope": (language or "").strip().lower() in self.spec["training_languages"],
            "validation_status": self.spec["validation_status"],
            "threshold": self.threshold, "threshold_kind": "fixed, uncalibrated",
            "encoder_sha256": self.spec["encoder"]["checkpoint_sha256"],
            "head_sha256": self.spec["head"]["sha256"],
        }
        if not window.usable:
            return {**base, "status": "insufficient_audio", "prediction": None,
                    "synthetic_score": None, "elapsed_seconds": time.perf_counter() - started}
        samples = torch.from_numpy(audio.samples[window.start_sample:window.end_sample].copy())
        encoded = self.encoder.extract_batch(samples[None], torch.tensor([samples.numel()]), sample_rate=16000)
        with torch.no_grad():
            output = self.detector({"features": encoded.features, "valid_lengths": encoded.valid_lengths,
                                    "padding_mask": encoded.padding_mask})
            logits = output.logits[0]
            score = float(torch.softmax(logits, dim=0)[1])
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite classifier output")
        return {**base, "status": "ok", "prediction": "spoof" if score >= self.threshold else "genuine",
                "synthetic_score": score, "score_is_calibrated": False,
                "logits": logits.tolist(), "valid_frames": int(encoded.valid_lengths[0]),
                "window_start_seconds": window.start_sample/16000,
                "window_end_seconds": window.end_sample/16000,
                "elapsed_seconds": time.perf_counter() - started}
