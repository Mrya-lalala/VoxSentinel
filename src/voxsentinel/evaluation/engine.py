"""Evaluation loop that aggregates detector scores before computing metrics."""

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor

from ..detectors.base import Detector
from .metrics import BinaryMetrics, binary_metrics


@dataclass(frozen=True)
class EvaluationResult:
    metrics: BinaryMetrics
    scores: Tensor
    labels: Tensor
    loss: float | None = None


@torch.no_grad()
def evaluate(
    detector: Detector,
    batches: Iterable[Mapping[str, Tensor]],
    device: torch.device | str = "cpu",
    threshold: float = 0.5,
) -> EvaluationResult:
    detector.eval()
    scores, labels, losses = [], [], []
    for batch in batches:
        moved = {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}
        if "labels" not in moved:
            raise KeyError("Evaluation batches must contain a 'labels' tensor.")
        output = detector(moved)
        scores.append(output.scores.detach().cpu())
        labels.append(moved["labels"].detach().cpu())
        losses.append(detector.loss(output, moved).detach().item())
    if not scores:
        raise ValueError("Evaluation requires at least one batch.")
    all_scores, all_labels = torch.cat(scores), torch.cat(labels)
    return EvaluationResult(
        binary_metrics(all_scores, all_labels, threshold),
        all_scores,
        all_labels,
        loss=sum(losses) / len(losses),
    )
