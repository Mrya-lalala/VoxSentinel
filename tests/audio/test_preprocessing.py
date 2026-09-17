"""Entry-point parity tests: file decoding vs already-decoded arrays."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from src.audio.config import PreprocessingConfig
from src.audio.errors import AudioLoadError, AudioValidationError
from src.audio.preprocessing import preprocess_array, preprocess_file


class FileDecodingParityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dir = Path(self.tmp.name)

    def _write_pcm(self, pcm: np.ndarray, rate: int, subtype: str) -> Path:
        path = self.dir / f"clip_{subtype}.wav"
        sf.write(str(path), pcm, rate, subtype=subtype)
        return path

    def test_int16_stereo_file_matches_decoded_array(self):
        rng = np.random.default_rng(1)
        pcm = rng.integers(-32768, 32768, size=(8000, 2), dtype=np.int16)
        path = self._write_pcm(pcm, 16000, "PCM_16")
        from_file = preprocess_file(path, source_id="f")
        from_array = preprocess_array(pcm, 16000, source_id="a")
        np.testing.assert_array_equal(from_file.samples, from_array.samples)
        self.assertEqual(from_file.original_channels, 2)
        self.assertEqual(from_file.original_sample_rate, 16000)

    def test_mono_file_matches_decoded_array(self):
        rng = np.random.default_rng(2)
        pcm = rng.integers(-32768, 32768, size=4000, dtype=np.int16)
        path = self._write_pcm(pcm, 16000, "PCM_16")
        from_file = preprocess_file(path)
        from_array = preprocess_array(pcm, 16000)
        np.testing.assert_array_equal(from_file.samples, from_array.samples)

    def test_file_load_reports_source_identity(self):
        pcm = np.zeros(400, dtype=np.int16)
        path = self._write_pcm(pcm, 16000, "PCM_16")
        audio = preprocess_file(path)
        self.assertEqual(audio.source_id, str(path))
        self.assertEqual(audio.original_sample_rate, 16000)
        self.assertEqual(audio.original_channels, 1)
        self.assertEqual(audio.sample_rate, 16000)

    def test_float_file_not_integer_rescaled(self):
        # A float WAV at amplitude 0.5 must not be treated as raw int16 0.5.
        samples = np.full(4000, 0.5, dtype=np.float32)
        path = self.dir / "float.wav"
        sf.write(str(path), samples, 16000, subtype="FLOAT")
        audio = preprocess_file(path)
        self.assertLessEqual(float(audio.samples.max()), 0.5 + 1e-6)

    def test_file_8k_stereo_downmix_and_resample(self):
        # Realistic correlated audio (not full-scale random noise) so resampler
        # ringing stays within the documented overshoot tolerance.
        t = np.arange(8000) / 8000
        sine = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)
        pcm = np.clip(np.rint(sine * 32768), -32768, 32767).astype(np.int16)
        path = self._write_pcm(np.stack([pcm, pcm], axis=-1), 8000, "PCM_16")
        audio = preprocess_file(path)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.num_samples, 16000)
        self.assertEqual(audio.original_sample_rate, 8000)
        self.assertEqual(audio.original_channels, 2)


class FileErrorTests(unittest.TestCase):
    def test_missing_file(self):
        with self.assertRaises(AudioLoadError):
            preprocess_file("definitely/not/here.wav")

    def test_invalid_container(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.wav"
            path.write_bytes(b"not a wav file at all")
            with self.assertRaises(AudioLoadError):
                preprocess_file(path)


class InvalidInputTests(unittest.TestCase):
    def test_float_out_of_range_rejected(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.array([1.1, 0.0], dtype=np.float32), 16000)

    def test_zero_channels_rejected(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.empty((4, 0), dtype=np.float32), 16000)

    def test_unsupported_rank_rejected(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.zeros((1, 1, 1), dtype=np.float32), 16000)

    def test_invalid_rate_rejected(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.zeros(400, dtype=np.float32), 0)

    def test_nan_rejected(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.array([np.nan], dtype=np.float32), 16000)


if __name__ == "__main__":
    unittest.main()
