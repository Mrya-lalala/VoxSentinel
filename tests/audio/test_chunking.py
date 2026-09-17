"""Tests for B1-safe chunking: tail policy, accounting, and per-chunk status."""

import unittest

import numpy as np

from src.audio.chunking import (
    STATUS_EMPTY,
    STATUS_OK,
    STATUS_TOO_SHORT,
    chunk_audio,
    collate_chunks,
)
from src.audio.config import PreprocessingConfig
from src.audio.errors import AudioValidationError
from src.audio.preprocessing import PreprocessedAudio
from src.audio.identity import PreprocessingIdentity


def _audio(num_samples: int, source_id: str = "unit") -> PreprocessedAudio:
    identity = PreprocessingIdentity()
    return PreprocessedAudio(
        samples=np.zeros(num_samples, dtype=np.float32),
        sample_rate=16000,
        source_id=source_id,
        original_sample_rate=16000,
        original_channels=1,
        original_format=None,
        identity=identity,
    )


class TailPolicyTests(unittest.TestCase):
    def test_exact_multiple_single_chunk(self):
        result = chunk_audio(_audio(64000))
        self.assertEqual(result.status, STATUS_OK)
        self.assertEqual([c.valid_samples for c in result.chunks], [64000])
        self.assertEqual(result.skipped, [])

    def test_final_retained_tail_400(self):
        result = chunk_audio(_audio(64400))
        self.assertEqual([c.valid_samples for c in result.chunks], [64000, 400])
        self.assertEqual(result.skipped, [])
        self.assertEqual(result.chunks[1].start_sample, 64000)
        self.assertEqual(result.chunks[1].end_sample, 64400)

    def test_tail_of_1_dropped(self):
        result = chunk_audio(_audio(64001))
        self.assertEqual([c.valid_samples for c in result.chunks], [64000])
        self.assertEqual([s.sample_count for s in result.skipped], [1])
        self.assertEqual(result.skipped[0].start_sample, 64000)
        self.assertEqual(result.skipped[0].reason, "tail_below_min_valid_samples")

    def test_tail_of_399_dropped(self):
        result = chunk_audio(_audio(64399))
        self.assertEqual([c.valid_samples for c in result.chunks], [64000])
        self.assertEqual([s.sample_count for s in result.skipped], [399])

    def test_recording_400_emits_one_short_chunk(self):
        result = chunk_audio(_audio(400))
        self.assertEqual([c.valid_samples for c in result.chunks], [400])
        self.assertEqual(result.skipped, [])

    def test_recordings_below_minimum_are_too_short(self):
        for size in (0, 1, 399):
            with self.subTest(size=size):
                result = chunk_audio(_audio(size))
                self.assertEqual(result.chunks, [])
                if size == 0:
                    self.assertEqual(result.status, STATUS_EMPTY)
                else:
                    self.assertEqual(result.status, STATUS_TOO_SHORT)
                    self.assertEqual([s.sample_count for s in result.skipped], [size])

    def test_many_windows_preserve_lengths(self):
        result = chunk_audio(_audio(3 * 64000 + 8000))
        self.assertEqual([c.valid_samples for c in result.chunks], [64000, 64000, 64000, 8000])
        self.assertEqual(result.skipped, [])


class AccountingTests(unittest.TestCase):
    def test_emitted_and_skipped_cover_all_samples(self):
        result = chunk_audio(_audio(64000 * 2 + 100))
        covered = []
        for chunk in result.chunks:
            covered.append((chunk.start_sample, chunk.end_sample))
        for span in result.skipped:
            covered.append((span.start_sample, span.end_sample))
        covered.sort()
        self.assertEqual(covered[0], (0, 64000))
        self.assertEqual(covered[-1], (128000, 128100))
        total = sum(end - start for start, end in covered)
        self.assertEqual(total, 64000 * 2 + 100)

    def test_spans_are_contiguous_and_ordered(self):
        result = chunk_audio(_audio(64000 * 2 + 100))
        spans = [(c.start_sample, c.end_sample) for c in result.chunks]
        spans += [(s.start_sample, s.end_sample) for s in result.skipped]
        spans.sort()
        for previous, current in zip(spans, spans[1:]):
            self.assertEqual(previous[1], current[0])


class CollateTests(unittest.TestCase):
    def test_right_padding_and_sample_lengths(self):
        result = chunk_audio(_audio(64400))
        samples, lengths = collate_chunks(result.chunks)
        self.assertEqual(samples.shape, (2, 64000))
        self.assertEqual(samples.dtype, np.float32)
        np.testing.assert_array_equal(lengths, np.array([64000, 400], dtype=np.int64))
        self.assertTrue(np.isfinite(samples).all())

    def test_empty_collate_rejected(self):
        with self.assertRaises(AudioValidationError):
            collate_chunks([])


class ChunkStatusTests(unittest.TestCase):
    def test_silent_window_inside_nonzero_recording_is_silent(self):
        # First window nonzero, second window all zero: the second must be marked
        # silent independently of the recording-wide status.
        identity = PreprocessingIdentity()
        samples = np.zeros(64000 * 2, dtype=np.float32)
        samples[:64000] = 0.1
        audio = PreprocessedAudio(samples, 16000, "unit", 16000, 1, None, identity)
        result = chunk_audio(audio)
        self.assertEqual([c.is_silent for c in result.chunks], [False, True])
        self.assertFalse(result.all_silent)

    def test_all_silent_flag(self):
        result = chunk_audio(_audio(64000))
        self.assertTrue(result.chunks[0].is_silent)
        self.assertTrue(result.all_silent)

    def test_nonzero_chunk_not_silent(self):
        identity = PreprocessingIdentity()
        samples = np.full(64000, 0.01, dtype=np.float32)
        audio = PreprocessedAudio(samples, 16000, "unit", 16000, 1, None, identity)
        result = chunk_audio(audio)
        self.assertFalse(result.chunks[0].is_silent)
        self.assertFalse(result.all_silent)

    def test_status_is_ok_for_silent_audio(self):
        # Silence is an acoustic input, not an invalid recording.
        self.assertEqual(chunk_audio(_audio(64000)).status, STATUS_OK)


class ConfigTests(unittest.TestCase):
    def test_two_second_windows_differ(self):
        config = PreprocessingConfig(window_seconds=2.0, hop_seconds=2.0)
        result = chunk_audio(_audio(64000), config=config)
        self.assertEqual([c.valid_samples for c in result.chunks], [32000, 32000])

    def test_overlap_rejected_for_this_milestone(self):
        config = PreprocessingConfig(window_seconds=4.0, hop_seconds=2.0)
        with self.assertRaises(AudioValidationError):
            chunk_audio(_audio(64000), config=config)


if __name__ == "__main__":
    unittest.main()
