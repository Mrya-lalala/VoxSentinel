"""C1 API contract tests: client-error mapping, upload cap, threshold use."""

import io
import os
import unittest
from unittest import mock

import numpy as np
import soundfile as sf
import torch
from fastapi.testclient import TestClient
from torch import nn

from src.backbones.tensor_interface import EncoderBatch
from src.config import ModelConfig
from src.detectors.registry import create_detector
from src.service import api
from src.service.engine import DEFAULT_THRESHOLD, LoadedDetector, ServiceEngine

TARGET_RATE = 16000
EMBEDDING_DIM = 8
FRAMES = 4


class StubEncoder:
    """Fixed-shape B1 stand-in: no checkpoint decoding, deterministic batch."""

    def __init__(self) -> None:
        self.calls = 0

    def extract_batch(self, waveforms, valid_lengths, *, sample_rate):
        self.calls += 1
        batch = int(waveforms.shape[0])
        return EncoderBatch(
            features=torch.zeros(batch, FRAMES, EMBEDDING_DIM),
            valid_lengths=torch.full((batch,), FRAMES, dtype=torch.int64),
            padding_mask=torch.zeros(batch, FRAMES, dtype=torch.bool),
            frame_hop_ms=20.0,
            backbone_id="stub",
            checkpoint_version="stub",
            output_layer=None,
        )


def wav_bytes(samples: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, samples, TARGET_RATE, format="WAV")
    return buffer.getvalue()


def noise_wav(seconds: float = 8.0) -> bytes:
    generator = np.random.default_rng(0)
    samples = generator.standard_normal(int(TARGET_RATE * seconds)) * 0.1
    return wav_bytes(samples.astype(np.float32))


def build_engine(
    *,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_source: str = "default",
) -> ServiceEngine:
    torch.manual_seed(0)
    detector = create_detector(
        ModelConfig(name="gru", parameters={"input_dim": EMBEDDING_DIM})
    )
    return ServiceEngine(
        encoder=StubEncoder(),
        detector=detector,
        model_id="test-model",
        encoder_id="test-encoder",
        threshold=threshold,
        threshold_source=threshold_source,
    )


class HealthEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def test_reports_degraded_without_engine(self) -> None:
        with mock.patch.object(api, "engine", None):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["detector_ready"])

    def test_reports_ok_with_engine(self) -> None:
        with mock.patch.object(api, "engine", build_engine()):
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertTrue(response.json()["detector_ready"])


class DetectEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(api.app)

    def post(
        self,
        content: bytes,
        filename: str = "sample.wav",
        content_type: str = "audio/wav",
    ):
        return self.client.post(
            "/detect",
            files={"file": (filename, content, content_type)},
        )

    def test_returns_503_without_engine(self) -> None:
        with mock.patch.object(api, "engine", None):
            response = self.post(noise_wav())

        self.assertEqual(response.status_code, 503)

    def test_returns_503_without_detector(self) -> None:
        engine = build_engine()
        engine.detector = None

        with mock.patch.object(api, "engine", engine):
            response = self.post(noise_wav())

        self.assertEqual(response.status_code, 503)

    def test_rejects_non_wav_filename(self) -> None:
        with mock.patch.object(api, "engine", build_engine()):
            response = self.post(noise_wav(), filename="sample.mp3")

        self.assertEqual(response.status_code, 400)

    def test_rejects_unsupported_content_type(self) -> None:
        with mock.patch.object(api, "engine", build_engine()):
            response = self.post(noise_wav(), content_type="audio/mpeg")

        self.assertEqual(response.status_code, 400)

    def test_rejects_empty_upload(self) -> None:
        with mock.patch.object(api, "engine", build_engine()):
            response = self.post(b"")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], "Uploaded file is empty.")

    def test_maps_undecodable_upload_to_client_error(self) -> None:
        with mock.patch.object(api, "engine", build_engine()):
            response = self.post(b"this is not a wav file")

        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertIn("could not be decoded", detail)
        self.assertNotIn("/", detail)

    def test_rejects_upload_over_the_size_cap(self) -> None:
        engine = build_engine()
        oversize = b"\x00" * (2 * 1024 * 1024)

        with mock.patch.object(api, "engine", engine), mock.patch.object(
            api, "MAX_UPLOAD_BYTES", 1024 * 1024
        ):
            response = self.post(oversize)

        self.assertEqual(response.status_code, 413)
        self.assertEqual(engine.encoder.calls, 0)

    def test_size_cap_is_checked_before_decoding(self) -> None:
        engine = build_engine()
        within_cap = b"\x00" * (512 * 1024)

        with mock.patch.object(api, "engine", engine), mock.patch.object(
            api, "MAX_UPLOAD_BYTES", 1024 * 1024
        ):
            response = self.post(within_cap)

        # Rejected for its content, not for its size.
        self.assertEqual(response.status_code, 400)

    def test_reports_insufficient_speech_for_short_recording(self) -> None:
        engine = build_engine(threshold=0.4, threshold_source="checkpoint")

        with mock.patch.object(api, "engine", engine):
            response = self.post(wav_bytes(np.zeros(200, dtype=np.float32)))

        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "insufficient_speech")
        self.assertIsNone(body["score"])
        self.assertIsNone(body["decision"])
        self.assertEqual(body["chunks_total"], 0)
        self.assertEqual(body["threshold"], 0.4)
        self.assertEqual(body["threshold_source"], "checkpoint")

    def test_decision_uses_the_effective_threshold(self) -> None:
        cases = ((0.0, "spoof"), (1.0, "genuine"))

        for threshold, expected in cases:
            with self.subTest(threshold=threshold):
                engine = build_engine(
                    threshold=threshold,
                    threshold_source="configured",
                )

                with mock.patch.object(api, "engine", engine):
                    response = self.post(noise_wav())

                body = response.json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["status"], "success")
                self.assertEqual(body["decision"], expected)
                self.assertEqual(body["threshold"], threshold)
                self.assertEqual(body["threshold_source"], "configured")


class CreateEngineThresholdTests(unittest.TestCase):
    def create_engine(
        self,
        *,
        checkpoint_threshold,
        env_threshold,
        detector_checkpoint="detector.pt",
    ) -> ServiceEngine:
        env = {
            key: value
            for key, value in os.environ.items()
            if key != api.THRESHOLD_ENV
        }
        if env_threshold is not None:
            env[api.THRESHOLD_ENV] = str(env_threshold)

        loaded = LoadedDetector(
            detector=nn.Module(),
            threshold=checkpoint_threshold,
        )

        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(
            api, "ENCODER_CHECKPOINT", "encoder.pt"
        ), mock.patch.object(
            api, "DETECTOR_CHECKPOINT", detector_checkpoint
        ), mock.patch.object(
            api, "load_encoder", return_value=StubEncoder()
        ), mock.patch.object(
            api, "load_detector", return_value=loaded
        ):
            return api.create_engine()

    def test_uses_checkpoint_threshold_when_not_configured(self) -> None:
        engine = self.create_engine(checkpoint_threshold=0.25, env_threshold=None)

        self.assertEqual(engine.threshold, 0.25)
        self.assertEqual(engine.threshold_source, "checkpoint")

    def test_configured_threshold_overrides_checkpoint(self) -> None:
        engine = self.create_engine(checkpoint_threshold=0.25, env_threshold=0.7)

        self.assertEqual(engine.threshold, 0.7)
        self.assertEqual(engine.threshold_source, "configured")

    def test_falls_back_to_default_without_checkpoint_metadata(self) -> None:
        engine = self.create_engine(checkpoint_threshold=None, env_threshold=None)

        self.assertEqual(engine.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(engine.threshold_source, "default")

    def test_falls_back_to_default_without_detector_checkpoint(self) -> None:
        engine = self.create_engine(
            checkpoint_threshold=None,
            env_threshold=None,
            detector_checkpoint=None,
        )

        self.assertEqual(engine.threshold, DEFAULT_THRESHOLD)
        self.assertEqual(engine.threshold_source, "default")
        self.assertFalse(engine.detector_ready)

    def test_blank_threshold_override_is_ignored(self) -> None:
        engine = self.create_engine(checkpoint_threshold=0.25, env_threshold=" ")

        self.assertEqual(engine.threshold, 0.25)
        self.assertEqual(engine.threshold_source, "checkpoint")


if __name__ == "__main__":
    unittest.main()
