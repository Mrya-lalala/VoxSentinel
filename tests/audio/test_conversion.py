"""Unit tests for A2 PCM validation, scaling, and channel downmix."""

import unittest

import numpy as np

from src.audio.conversion import (
    downmix_to_mono,
    pcm_to_float,
    validate_float_amplitudes,
    validate_sample_rate,
    validate_waveform_array,
)
from src.audio.errors import AudioValidationError


class PcmScalingTests(unittest.TestCase):
    def test_int16_mono_endpoints(self):
        pcm = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        result = pcm_to_float(pcm)
        np.testing.assert_allclose(
            result,
            [-1.0, -0.5, 0.0, 0.5, 32767 / 32768],
            rtol=1e-7,
            atol=1e-7,
        )

    def test_int16_equivalent_stereo_channels_equal_mono(self):
        # The reviewed bug: identical int16 channels produced 16384.0 while
        # equivalent mono produced 0.5.  Scaling must happen before downmix.
        mono = np.array([16384, -32768], dtype=np.int16)
        stereo = np.stack([mono, mono], axis=-1)
        np.testing.assert_array_equal(downmix_to_mono(pcm_to_float(stereo)),
                                      downmix_to_mono(pcm_to_float(mono)))

    def test_uint8_midpoint_mapping(self):
        pcm = np.array([0, 128, 255], dtype=np.uint8)
        np.testing.assert_allclose(pcm_to_float(pcm), [-1.0, 0.0, 127 / 128], rtol=1e-7, atol=1e-7)

    def test_uint8_silence_128_is_zero_not_positive(self):
        # 128 must never be interpreted as +0.502.
        self.assertEqual(float(pcm_to_float(np.array([128], dtype=np.uint8))[0]), 0.0)

    def test_int32_endpoints(self):
        pcm = np.array([-(2 ** 31), 0, 2 ** 31 - 1], dtype=np.int32)
        np.testing.assert_allclose(
            pcm_to_float(pcm),
            [-1.0, 0.0, (2 ** 31 - 1) / (2 ** 31)],
            rtol=1e-7,
            atol=1e-7,
        )

    def test_float_passthrough_not_rescaled(self):
        samples = np.array([0.25, -0.75], dtype=np.float32)
        result = pcm_to_float(samples)
        self.assertEqual(result.dtype, np.float64)
        np.testing.assert_array_equal(result, samples.astype(np.float64))

    def test_stereo_downmix_averages_channels(self):
        stereo = np.array([[0.1, 0.3], [0.5, 0.5], [-0.2, 0.2]], dtype=np.float64)
        np.testing.assert_allclose(downmix_to_mono(stereo), [0.2, 0.5, 0.0])


class UnsupportedInputTests(unittest.TestCase):
    def test_rejects_complex(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.array([1 + 0j], dtype=np.complex64))

    def test_rejects_boolean(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.array([True, False]))

    def test_rejects_object_dtype(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.array([1, "two"], dtype=object))

    def test_rejects_string_dtype(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.array(["a", "b"]))

    def test_rejects_unsupported_int_dtypes(self):
        for dtype in (np.int8, np.int64, np.uint16, np.uint32, np.uint64):
            with self.subTest(dtype=dtype):
                with self.assertRaises(AudioValidationError):
                    pcm_to_float(np.array([1, 2], dtype=dtype))

    def test_rejects_zero_channels(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.empty((4, 0), dtype=np.float32))

    def test_rejects_rank_three(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.zeros((1, 1, 1), dtype=np.float32))

    def test_rejects_empty(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array(np.empty(0, dtype=np.float32))

    def test_rejects_non_numpy_input(self):
        with self.assertRaises(AudioValidationError):
            validate_waveform_array([0.0, 0.1])


class SampleRateTests(unittest.TestCase):
    def test_accepts_positive_int(self):
        self.assertEqual(validate_sample_rate(16000), 16000)

    def test_accepts_integral_float(self):
        self.assertEqual(validate_sample_rate(16000.0), 16000)

    def test_rejects_zero(self):
        with self.assertRaises(AudioValidationError):
            validate_sample_rate(0)

    def test_rejects_negative(self):
        with self.assertRaises(AudioValidationError):
            validate_sample_rate(-16000)

    def test_rejects_fractional(self):
        with self.assertRaises(AudioValidationError):
            validate_sample_rate(16000.5)

    def test_rejects_nonfinite(self):
        with self.assertRaises(AudioValidationError):
            validate_sample_rate(float("nan"))
        with self.assertRaises(AudioValidationError):
            validate_sample_rate(float("inf"))


class AmplitudeValidationTests(unittest.TestCase):
    def test_accepts_in_range(self):
        validate_float_amplitudes(np.array([-1.0, 0.0, 1.0]), tolerance=1e-6, context="test")

    def test_rejects_out_of_range(self):
        with self.assertRaises(AudioValidationError):
            validate_float_amplitudes(np.array([1.1]), tolerance=1e-6, context="test")

    def test_rejects_nan(self):
        with self.assertRaises(AudioValidationError):
            validate_float_amplitudes(np.array([np.nan]), tolerance=1e-6, context="test")

    def test_rejects_inf(self):
        with self.assertRaises(AudioValidationError):
            validate_float_amplitudes(np.array([np.inf]), tolerance=1e-6, context="test")


if __name__ == "__main__":
    unittest.main()
