import unittest

import numpy as np

from src.backbones.base import BackboneExtractor
from src.backbones.router import BackboneRouter
from src.backbones.schemas import ChunkMetadata, EmbeddingSequence


class FakeExtractor(BackboneExtractor):
    def __init__(self, backbone_id: str, route: str) -> None:
        self.backbone_id = backbone_id
        self.route = route

    def extract(self, samples, metadata):
        waveform = self.validate_samples(samples, metadata)
        return EmbeddingSequence(np.ones((2, 3), dtype=np.float32), 20.0, self.backbone_id, self.route, metadata)


class BackboneRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = BackboneRouter(FakeExtractor("indic-test", "wrong"), FakeExtractor("global-test", "wrong"))
        self.samples = np.zeros(16_000, dtype=np.float32)

    def metadata(self, language):
        return ChunkMetadata("fixture", 0, 1_000, language=language)

    def test_routes_known_indic_language(self):
        result = self.router.extract(self.samples, self.metadata("hi-IN"))
        self.assertEqual(result.route, "indic")
        self.assertEqual(result.backbone_id, "indic-test")

    def test_routes_missing_or_unknown_language_to_global_fallback(self):
        for language in (None, "en", "fr-CA"):
            with self.subTest(language=language):
                result = self.router.extract(self.samples, self.metadata(language))
                self.assertEqual(result.route, "global_fallback")
                self.assertEqual(result.backbone_id, "global-test")

    def test_rejects_non_normalized_audio(self):
        metadata = ChunkMetadata("fixture", 0, 1_000, sample_rate=8_000, language="hi")
        with self.assertRaisesRegex(ValueError, "16 kHz"):
            self.router.extract(self.samples, metadata)
