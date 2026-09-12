import unittest

import numpy as np

from src.backbones.schemas import ChunkMetadata, EmbeddingSequence


class EmbeddingContractTests(unittest.TestCase):
    def test_normalizes_features_to_float32(self):
        result = EmbeddingSequence(np.ones((3, 4), dtype=np.float64), 20.0, "test", "indic", ChunkMetadata("x", 0, 60))
        self.assertEqual(result.features.dtype, np.float32)
        self.assertEqual(result.features.shape, (3, 4))

    def test_rejects_non_time_major_features(self):
        with self.assertRaisesRegex(ValueError, "frames"):
            EmbeddingSequence(np.ones(4), 20.0, "test", "indic", ChunkMetadata("x", 0, 60))
