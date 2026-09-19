from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Sequence


AggregationMethod = Literal["mean", "max", "topk_mean"]


@dataclass(frozen=True)
class AggregationResult:
    score: float
    method: AggregationMethod
    chunks_used: int
    selected_indices: tuple[int, ...]


def _validate_scores(scores: Sequence[float]) -> None:
    if not scores:
        raise ValueError("Cannot aggregate an empty score sequence.")

    for score in scores:
        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"Chunk score must be in [0, 1], got {score}."
            )


def aggregate_scores(
    scores: Iterable[float],
    method: AggregationMethod = "mean",
    top_k: int | None = None,
) -> AggregationResult:
    """
    Convert chunk-level spoof scores into one file-level score.

    Parameters
    ----------
    scores:
        Chunk-level scores. Higher means more spoof-like.

    method:
        - mean: arithmetic mean of all chunks
        - max: maximum chunk score
        - topk_mean: mean of the highest-scoring k chunks

    top_k:
        Required for topk_mean.

    Returns
    -------
    AggregationResult
    """

    values = tuple(float(score) for score in scores)

    _validate_scores(values)

    if method == "mean":
        selected_indices = tuple(range(len(values)))
        score = sum(values) / len(values)

    elif method == "max":
        max_index = max(
            range(len(values)),
            key=lambda index: values[index],
        )

        selected_indices = (max_index,)
        score = values[max_index]

    elif method == "topk_mean":
        if top_k is None:
            raise ValueError(
                "top_k must be provided when method='topk_mean'."
            )

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        k = min(top_k, len(values))

        ranked_indices = sorted(
            range(len(values)),
            key=lambda index: values[index],
            reverse=True,
        )

        selected_indices = tuple(ranked_indices[:k])

        score = sum(
            values[index] for index in selected_indices
        ) / k

    else:
        raise ValueError(
            f"Unsupported aggregation method: {method}"
        )

    return AggregationResult(
        score=float(score),
        method=method,
        chunks_used=len(selected_indices),
        selected_indices=selected_indices,
    )