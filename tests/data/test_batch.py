import numpy as np
import pytest
import torch

from src.data import EmbeddingExample, batch_iterator, collate_examples, synthetic_examples


def test_embedding_example_normalizes_dtype_and_validates_shape():
    example = EmbeddingExample(torch.ones(4, 3, dtype=torch.float64), 1)
    assert example.features.dtype == torch.float32
    assert (example.num_frames, example.embedding_dim, example.label) == (4, 3, 1)
    with pytest.raises(ValueError, match="frames, dim"):
        EmbeddingExample(torch.ones(4), 0)
    with pytest.raises(ValueError, match="non-finite"):
        EmbeddingExample(torch.tensor([[float("nan"), 1.0]]), 0)


def test_labels_are_validated_before_integer_conversion():
    with pytest.raises(ValueError, match="binary"):
        EmbeddingExample(torch.ones(2, 2), 0.5)
    with pytest.raises(ValueError, match="binary"):
        EmbeddingExample(torch.ones(2, 2), 2)
    with pytest.raises(ValueError, match="boolean"):
        EmbeddingExample(torch.ones(2, 2), True)
    assert EmbeddingExample(torch.ones(2, 2), np.int64(1)).label == 1


def test_from_array_accepts_b1_style_ndarray():
    example = EmbeddingExample.from_array(np.ones((5, 8), dtype=np.float64), 0)
    assert example.features.dtype == torch.float32
    assert (example.num_frames, example.embedding_dim) == (5, 8)


def test_collate_pads_right_marks_padding_and_preserves_order():
    examples = [
        EmbeddingExample(torch.full((3, 4), 1.0), 0),
        EmbeddingExample(torch.full((5, 4), 2.0), 1),
    ]
    batch = collate_examples(examples)
    assert batch["features"].shape == (2, 5, 4)
    assert batch["valid_lengths"].tolist() == [3, 5]
    assert batch["labels"].tolist() == [0, 1]
    assert batch["padding_mask"].dtype == torch.bool
    assert batch["padding_mask"][0].tolist() == [False, False, False, True, True]
    assert batch["padding_mask"][1].tolist() == [False] * 5
    assert torch.equal(batch["features"][0, 3:], torch.zeros(2, 4))
    assert torch.allclose(batch["features"][0, :3], torch.full((3, 4), 1.0))


def test_collate_rejects_empty_input_and_mixed_dims():
    with pytest.raises(ValueError, match="at least one"):
        collate_examples([])
    with pytest.raises(ValueError, match="embedding dim"):
        collate_examples([EmbeddingExample(torch.ones(2, 4), 0), EmbeddingExample(torch.ones(2, 5), 1)])


def test_batch_iterator_sizes_and_seeded_shuffle():
    examples = synthetic_examples(5, input_dim=4, min_frames=3, max_frames=5, seed=0)
    ordered = list(batch_iterator(examples, batch_size=2))
    assert [batch["features"].shape[0] for batch in ordered] == [2, 2, 1]
    assert all(batch["features"].shape[-1] == 4 for batch in ordered)
    first = [batch["labels"].tolist() for batch in batch_iterator(examples, 2, shuffle=True, seed=3)]
    second = [batch["labels"].tolist() for batch in batch_iterator(examples, 2, shuffle=True, seed=3)]
    assert first == second
    with pytest.raises(ValueError, match="batch_size"):
        list(batch_iterator(examples, 0))


def test_synthetic_fixture_matches_the_head_contract():
    examples = synthetic_examples(8, input_dim=1024, min_frames=10, max_frames=20, seed=0)
    assert len(examples) == 8
    assert {example.label for example in examples} == {0, 1}
    batch = collate_examples(examples)
    assert batch["features"].shape == (8, 20, 1024)
    assert batch["valid_lengths"].shape == (8,)
    assert batch["padding_mask"].shape == (8, 20)
    assert batch["labels"].shape == (8,)
