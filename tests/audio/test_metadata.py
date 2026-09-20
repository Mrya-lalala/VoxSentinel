"""Tests for truthful, reproducible preprocessing identity and metadata."""

import unittest

import numpy as np

from src.audio.chunking import chunk_audio
from src.audio.config import PreprocessingConfig
from src.audio.identity import PreprocessingIdentity
from src.audio.preprocessing import PreprocessedAudio


def _audio(num_samples: int, source_id: str = "src-1") -> PreprocessedAudio:
    return PreprocessedAudio(
        samples=np.zeros(num_samples, dtype=np.float32),
        sample_rate=16000,
        source_id=source_id,
        original_sample_rate=16000,
        original_channels=1,
        original_format=None,
        identity=PreprocessingIdentity(),
    )


class IdentityTests(unittest.TestCase):
    def test_fingerprint_is_repeatable(self):
        identity = PreprocessingIdentity()
        self.assertEqual(identity.fingerprint, PreprocessingIdentity().fingerprint)
        self.assertEqual(identity.identity_string, PreprocessingIdentity().identity_string)

    def test_two_second_vs_four_second_differ(self):
        four = PreprocessingIdentity.from_config(PreprocessingConfig())
        two = PreprocessingIdentity.from_config(
            PreprocessingConfig(window_seconds=2.0, hop_seconds=2.0)
        )
        self.assertNotEqual(four.identity_string, two.identity_string)

    def test_schema_version_separate_from_config(self):
        identity = PreprocessingIdentity()
        self.assertEqual(identity.schema_version, "audio-v1")
        self.assertNotIn(identity.schema_version, identity.canonical_key())
        self.assertTrue(identity.identity_string.startswith("audio-v1:sha256:"))

    def test_different_resampler_methods_differ(self):
        a = PreprocessingIdentity.from_config(PreprocessingConfig(resampling_method="scipy_polyphase"))
        # A hypothetical different backend must change the identity.
        other = PreprocessingIdentity(
            resampling_method="torchaudio_soxr",
            window_samples=a.window_samples,
            hop_samples=a.hop_samples,
        )
        self.assertNotEqual(a.identity_string, other.identity_string)


class ChunkMetadataTests(unittest.TestCase):
    def test_offsets_index_resampled_waveform(self):
        audio = _audio(64400, source_id="src-1")
        result = chunk_audio(audio)
        self.assertEqual([c.start_sample for c in result.chunks], [0, 64000])
        self.assertEqual([c.end_sample for c in result.chunks], [64000, 64400])
        self.assertEqual(result.chunks[0].start_ms, 0.0)
        self.assertEqual(result.chunks[0].duration_ms, 4000.0)
        self.assertEqual(result.chunks[1].start_ms, 4000.0)
        self.assertEqual(result.chunks[1].duration_ms, 25.0)

    def test_source_id_preserved_per_chunk(self):
        result = chunk_audio(_audio(128000, source_id="caller/rec.wav"))
        self.assertTrue(all(c.source_id == "caller/rec.wav" for c in result.chunks))

    def test_order_preserved(self):
        result = chunk_audio(_audio(3 * 64000 + 400, source_id="s"))
        starts = [c.start_sample for c in result.chunks]
        self.assertEqual(starts, sorted(starts))

    def test_identity_attached_to_chunks_and_preprocessed(self):
        audio = _audio(64000)
        result = chunk_audio(audio)
        self.assertEqual(result.chunks[0].preprocessing_identity, audio.identity_string)
        self.assertTrue(result.chunks[0].preprocessing_identity.startswith("audio-v1:sha256:"))


if __name__ == "__main__":
    unittest.main()
