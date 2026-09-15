"""Tiny synthetic embedding fixture used only to verify the B2 training loop.

Genuine examples are i.i.d. noise across the embedding dim.  Spoof examples
add a fixed sinusoidal pattern across that dim, which survives the detector's
per-frame LayerNorm and is easy for the GRU to separate.  This is a smoke-test
fixture, not a benchmark dataset.
"""

from __future__ import annotations

import torch

from .batch import GENUINE, SPOOF, EmbeddingExample


def synthetic_examples(
    num_examples: int = 32,
    *,
    input_dim: int = 1024,
    min_frames: int = 8,
    max_frames: int = 24,
    seed: int = 0,
) -> list[EmbeddingExample]:
    """Return an even split of genuine (0) and spoof (1) embedding sequences."""
    if num_examples <= 0:
        raise ValueError("num_examples must be positive.")
    if input_dim <= 0:
        raise ValueError("input_dim must be positive.")
    if min_frames < 1 or max_frames < min_frames:
        raise ValueError("Require 1 <= min_frames <= max_frames.")

    generator = torch.Generator().manual_seed(int(seed))
    positions = torch.arange(input_dim, dtype=torch.float32)
    spoof_pattern = torch.sin(positions * torch.pi / 8)

    examples: list[EmbeddingExample] = []
    for index in range(num_examples):
        label = index % 2
        frames = int(torch.randint(min_frames, max_frames + 1, (1,), generator=generator).item())
        features = torch.randn(frames, input_dim, generator=generator)
        if label == SPOOF:
            features = 0.5 * features + 3.0 * spoof_pattern.unsqueeze(0)
        examples.append(EmbeddingExample(features=features, label=label))
    assert {example.label for example in examples} <= {GENUINE, SPOOF}
    return examples
