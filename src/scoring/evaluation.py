"""Aggregate detector scores over batches and compute B2 metrics."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor

from ..data.labels import validate_label_tensor
from .metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: BinaryMetrics
    scores: Tensor
    labels: Tensor
    loss: float | None = None
    batches: int = field(default=0)


def _detector_device(detector: torch.nn.Module) -> torch.device:
    try:
        return next(detector.parameters()).device
    except StopIteration:  # pragma: no cover - detectors always have parameters
        return torch.device("cpu")


def evaluate(
    detector,
    batches: Iterable[Mapping[str, Tensor]],
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
    loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> EvaluationResult:
    """Run ``detector`` over ``batches`` without gradients.

    The detector must already live on ``device``; mismatches raise instead of
    silently relocating a model (which would invalidate optimizer state).  Loss
    is aggregated as an example-weighted mean so a short final batch is not
    overweighted.
    """
    target_device = torch.device(device)
    detector_device = _detector_device(detector)
    if detector_device != target_device:
        raise ValueError(
            f"Detector is on {detector_device} but evaluation requested {target_device}; "
            "move the detector before evaluating."
        )

    detector.eval()
    scores: list[Tensor] = []
    labels: list[Tensor] = []
    total_loss, total_examples, batch_count = 0.0, 0, 0
    with torch.no_grad():
        for batch in batches:
            moved = {key: value.to(target_device) if isinstance(value, Tensor) else value for key, value in batch.items()}
            if "labels" not in moved:
                raise KeyError("Evaluation batches must contain a 'labels' tensor.")
            moved = {**moved, "labels": validate_label_tensor(moved["labels"]).flatten()}
            output = detector(moved)
            loss = loss_fn(output.logits, moved["labels"]) if loss_fn is not None else detector.loss(output, moved)
            examples = int(moved["labels"].shape[0])
            total_loss += float(loss.detach()) * examples
            total_examples += examples
            batch_count += 1
            scores.append(output.scores.detach().cpu())
            labels.append(moved["labels"].detach().cpu())

    if batch_count == 0:
        raise ValueError("Evaluation requires at least one batch.")
    all_scores, all_labels = torch.cat(scores), torch.cat(labels)
    return EvaluationResult(
        metrics=binary_metrics(all_scores, all_labels, threshold),
        scores=all_scores,
        labels=all_labels,
        loss=total_loss / total_examples,
        batches=batch_count,
    )
