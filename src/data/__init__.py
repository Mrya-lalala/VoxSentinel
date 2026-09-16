"""Dataset contracts for VoxSentinel: manifest-ready examples and batching."""

from .batch import EmbeddingExample, batch_iterator, collate_examples
from .labels import GENUINE, SPOOF, validate_label, validate_label_tensor
from .synthetic import synthetic_examples

__all__ = [
    "EmbeddingExample",
    "GENUINE",
    "SPOOF",
    "batch_iterator",
    "collate_examples",
    "synthetic_examples",
    "validate_label",
    "validate_label_tensor",
]
