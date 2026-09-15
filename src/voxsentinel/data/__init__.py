"""In-memory B2 training data: embedding sequences, labels, and batching."""

from .batch import EmbeddingExample, GENUINE, SPOOF, batch_iterator, collate_examples
from .synthetic import synthetic_examples

__all__ = [
    "EmbeddingExample",
    "GENUINE",
    "SPOOF",
    "batch_iterator",
    "collate_examples",
    "synthetic_examples",
]
