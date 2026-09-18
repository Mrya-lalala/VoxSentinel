"""Tests for the real polyphase resampler and the outgoing PCM contract."""

import unittest

import numpy as np

from src.audio.config import PreprocessingConfig
from src.audio.errors import AudioValidationError
from src.audio.preprocessing import preprocess_array
from src.audio.resampling import expected_output_samples, resample_polyphase


class ResamplerLengthTests(unittest.TestCase):
    def test_8k_to_16k_length(self):
        mono = np.zeros(8000, dtype=np.float64)
        self.assertEqual(resample_polyphase(mono, 8000, 16000).size, 16000)
        self.assertEqual(expected_output_samples(8000, 8000, 16000), 16000)

    def test_48k_to_16k_length(self):
        mono = np.zeros(48000, dtype=np.float64)
        self.assertEqual(resample_polyphase(mono, 48000, 16000).size, 16000)
        self.assertEqual(expected_output_samples(48000, 48000, 16000), 16000)

    def test_441k_to_16k_length(self):
        mono = np.zeros(44100, dtype=np.float64)
        self.assertEqual(resample_polyphase(mono, 44100, 16000).size, 16000)
        self.assertEqual(expected_output_samples(44100, 44100, 16000), 16000)

    def test_same_rate_is_copy(self):
        mono = np.arange(10, dtype=np.float64)
        out = resample_polyphase(mono, 16000, 16000)
        np.testing.assert_array_equal(out, mono.astype(np.float32))
        self.assertIsNot(out, mono)

    def test_output_is_finite(self):
        rng = np.random.default_rng(0)
        mono = (rng.uniform(-0.5, 0.5, size=8000)).astype(np.float64)
        self.assertTrue(np.isfinite(resample_polyphase(mono, 8000, 16000)).all())

    def test_rejects_unsupported_method(self):
        with self.assertRaises(AudioValidationError):
            resample_polyphase(np.zeros(8), 8000, 16000, method="linear_interp_mock")


class ResamplerRangePolicyTests(unittest.TestCase):
    def _sine(self, rate, amplitude=1.0, seconds=1.0):
        t = np.arange(int(rate * seconds)) / rate
        return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)

    def test_normal_level_passes(self):
        audio = preprocess_array(self._sine(8000, amplitude=0.5), 8000)
        self.assertEqual(audio.num_samples, 16000)
        self.assertTrue(np.isfinite(audio.samples).all())
        self.assertLessEqual(float(audio.samples.max()), 1.0 + 1e-6)
        self.assertGreaterEqual(float(audio.samples.min()), -1.0 - 1e-6)

    def test_near_full_scale_sine_clamped_within_tolerance(self):
        # Smooth near-full-scale signal overshoots by ~0.03%; bounded correction.
        audio = preprocess_array(self._sine(8000, amplitude=0.999), 8000)
        self.assertLessEqual(float(audio.samples.max()), 1.0 + 1e-6)

    def test_material_overshoot_rejected(self):
        # A full-scale DC edge rings ~13% above full scale: rejected, not clamped.
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.full(8000, 0.999, dtype=np.float64), 8000)

    def test_malformed_input_rejected_before_resampling(self):
        with self.assertRaises(AudioValidationError):
            preprocess_array(np.full(8000, 1.1, dtype=np.float32), 8000)

    def test_nonfinite_rejected(self):
        for bad in (np.nan, np.inf, -np.inf):
            with self.subTest(bad=bad):
                signal = np.full(8000, bad, dtype=np.float32)
                with self.assertRaises(AudioValidationError):
                    preprocess_array(signal, 8000)

    def test_no_normalization_applied(self):
        # A quiet recording must stay quiet: no AGC/peak normalization may
        # rescale it toward full scale.  Resampler ringing is allowed, so the
        # bound is a "still quiet" check, not an exact amplitude check.
        quiet = preprocess_array(np.full(8000, 0.05, dtype=np.float64), 8000)
        self.assertLess(float(quiet.samples.max()), 0.1)
        self.assertGreater(float(quiet.samples.min()), -0.1)


class OutgoingContractTests(unittest.TestCase):
    def test_preprocess_produces_mono_float32_16k(self):
        stereo = np.stack([self._sine(8000, 0.5), self._sine(8000, 0.5)], axis=-1)
        audio = preprocess_array(stereo, 8000, source_id="unit")
        self.assertEqual(audio.samples.dtype, np.float32)
        self.assertEqual(audio.samples.ndim, 1)
        self.assertEqual(audio.sample_rate, 16000)
        self.assertEqual(audio.original_channels, 2)
        self.assertEqual(audio.original_sample_rate, 8000)

    def _sine(self, rate, amplitude=1.0):
        t = np.arange(rate) / rate
        return (amplitude * np.sin(2 * np.pi * 440.0 * t)).astype(np.float64)


if __name__ == "__main__":
    unittest.main()
