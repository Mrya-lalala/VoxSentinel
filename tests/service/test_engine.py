"""C1 engine tests: threshold resolution, checkpoint metadata, payload errors."""

import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.audio.errors import AudioLoadError
from src.config import ModelConfig
from src.detectors.checkpoints import save_checkpoint
from src.detectors.registry import create_detector
from src.service.engine import (
    DEFAULT_THRESHOLD,
    LoadedDetector,
    ServiceEngine,
    load_detector,
    resolve_threshold,
)

MODEL_PARAMETERS = {"input_dim": 8}


def build_detector():
    torch.manual_seed(0)
    return create_detector(ModelConfig(name="gru", parameters=MODEL_PARAMETERS))


class ResolveThresholdTests(unittest.TestCase):
    def test_configured_threshold_wins(self) -> None:
        self.assertEqual(
            resolve_threshold(configured=0.7, checkpoint=0.2),
            (0.7, "configured"),
        )

    def test_checkpoint_threshold_used_when_unconfigured(self) -> None:
        self.assertEqual(
            resolve_threshold(checkpoint=0.2),
            (0.2, "checkpoint"),
        )

    def test_default_used_without_any_threshold(self) -> None:
        self.assertEqual(
            resolve_threshold(),
            (DEFAULT_THRESHOLD, "default"),
        )


class LoadDetectorThresholdTests(unittest.TestCase):
    def save(self, path: Path, *, metadata=None) -> None:
        save_checkpoint(
            path,
            build_detector(),
            "gru",
            dict(MODEL_PARAMETERS),
            metadata=metadata,
        )

    def test_reads_threshold_from_checkpoint_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru_best.pt"
            self.save(path, metadata={"threshold": 0.25, "commit": "abc"})

            loaded = load_detector(path)

        self.assertIsInstance(loaded, LoadedDetector)
        self.assertEqual(loaded.threshold, 0.25)
        self.assertFalse(loaded.detector.training)

    def test_reports_no_threshold_without_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "gru_best.pt"
            self.save(path)

            self.assertIsNone(load_detector(path).threshold)

    def test_ignores_unusable_threshold_values(self) -> None:
        cases = ("0.25", 5.0, -0.1, True)

        for value in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "gru_best.pt"
                self.save(path, metadata={"threshold": value})

                self.assertIsNone(load_detector(path).threshold)


class ServiceEngineTests(unittest.TestCase):
    class StubEncoder:
        def extract_batch(self, waveforms, valid_lengths, *, sample_rate):
            raise AssertionError("encoding should not be reached")

    def build(self, **kwargs) -> ServiceEngine:
        return ServiceEngine(
            encoder=self.StubEncoder(),
            detector=build_detector(),
            model_id="test-model",
            encoder_id="test-encoder",
            **kwargs,
        )

    def test_exposes_resolved_threshold(self) -> None:
        engine = self.build(threshold=0.3, threshold_source="checkpoint")

        self.assertEqual(engine.threshold, 0.3)
        self.assertEqual(engine.threshold_source, "checkpoint")

    def test_defaults_to_service_default_threshold(self) -> None:
        engine = self.build()

        self.assertEqual(engine.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(engine.threshold_source, "default")

    def test_undecodable_payload_raises_audio_load_error(self) -> None:
        engine = self.build()

        with self.assertRaises(AudioLoadError):
            engine.predict(b"this is not a wav file", source_id="corrupt")

    def test_rejects_empty_audio_before_inference(self) -> None:
        engine = self.build()
        buffer = io.BytesIO()
        sf.write(buffer, np.zeros(0, dtype=np.float32), 16000, format="WAV")

        with self.assertRaises(ValueError):
            engine.predict(buffer.getvalue(), source_id="empty")


if __name__ == "__main__":
    unittest.main()
