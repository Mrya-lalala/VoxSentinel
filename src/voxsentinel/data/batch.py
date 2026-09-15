"""B2 training data contract: one labelled embedding sequence per example.

B2 never loads or preprocesses audio.  A dataset adapter added later is
responsible for running the frozen B1 encoder and handing B2 a list of
``EmbeddingExample`` objects.  ``collate_examples`` turns such a list into the
exact tensor batch the GRU detector consumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Mapping, Sequence

import torch
from torch import Tensor

GENUINE = 0
SPOOF = 1


@dataclass(frozen=True)
class EmbeddingExample:
    """One labelled speech excerpt: features ``[T, D]`` and label 0/1."""

    features: Tensor
    label: int

    def __post_init__(self) -> None:
        features = torch.as_tensor(self.features, dtype=torch.float32)
        if features.ndim != 2:
            raise ValueError(f"features must have shape [frames, dim]; got {tuple(features.shape)}.")
        if features.shape[0] < 1 or features.shape[1] < 1:
            raise ValueError("features must have at least one frame and one dimension.")
        if int(self.label) not in (GENUINE, SPOOF):
            raise ValueError("label must be 0 (genuine) or 1 (spoof).")
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "label", int(self.label))

    @property
    def num_frames(self) -> int:
        return int(self.features.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.features.shape[1])

    @classmethod
    def from_array(cls, features: Tensor, label: int) -> "EmbeddingExample":
        """Build an example from anything ``torch.as_tensor`` accepts (e.g. a B1 ndarray)."""
        return cls(features=torch.as_tensor(features, dtype=torch.float32), label=label)


def collate_examples(examples: Sequence[EmbeddingExample], *, pad_value: float = 0.0) -> dict[str, Tensor]:
    """Right-pad a list of examples into a single training batch.

    Returns ``features`` ``[B, T, D]``, ``valid_lengths`` ``[B]``,
    ``padding_mask`` ``[B, T]`` (True = padding), and ``labels`` ``[B]``.
    Padding is zero by default, matching the B1 encoder batch contract.
    """
    if len(examples) == 0:
        raise ValueError("collate_examples requires at least one example.")
    dims = {example.embedding_dim for example in examples}
    if len(dims) != 1:
        raise ValueError(f"All examples must share one embedding dim; got {sorted(dims)}.")

    dim = dims.pop()
    max_frames = max(example.num_frames for example in examples)
    features = torch.full((len(examples), max_frames, dim), float(pad_value), dtype=torch.float32)
    valid_lengths = torch.empty(len(examples), dtype=torch.long)
    padding_mask = torch.ones((len(examples), max_frames), dtype=torch.bool)
    labels = torch.empty(len(examples), dtype=torch.long)

    for index, example in enumerate(examples):
        frames = example.num_frames
        features[index, :frames] = example.features
        valid_lengths[index] = frames
        padding_mask[index, :frames] = False
        labels[index] = example.label

    return {
        "features": features,
        "valid_lengths": valid_lengths,
        "padding_mask": padding_mask,
        "labels": labels,
    }


def batch_iterator(
    examples: Sequence[EmbeddingExample],
    batch_size: int,
    *,
    shuffle: bool = False,
    seed: int = 0,
) -> Iterator[dict[str, Tensor]]:
    """Yield collated batches; ``seed`` makes shuffled order reproducible."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if len(examples) == 0:
        raise ValueError("Cannot batch an empty dataset.")

    order = list(range(len(examples)))
    if shuffle:
        generator = torch.Generator().manual_seed(int(seed))
        order = torch.randperm(len(examples), generator=generator).tolist()

    for start in range(0, len(order), batch_size):
        selected = [examples[index] for index in order[start : start + batch_size]]
        yield collate_examples(selected)
