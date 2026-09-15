"""Dependency-light metrics for binary bona-fide/spoof classification."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float | None


def _eer(scores: Tensor, labels: Tensor) -> float | None:
    positives, negatives = scores[labels == 1], scores[labels == 0]
    if not len(positives) or not len(negatives):
        return None
    thresholds = torch.unique(scores).sort().values
    candidates: list[tuple[float, float]] = []
    for threshold in thresholds:
        far = (negatives >= threshold).float().mean().item()
        frr = (positives < threshold).float().mean().item()
        candidates.append((abs(far - frr), (far + frr) / 2))
    return min(candidates, key=lambda item: item[0])[1]


def binary_metrics(scores: Tensor, labels: Tensor, threshold: float = 0.5) -> BinaryMetrics:
    scores, labels = scores.detach().flatten().cpu(), labels.detach().flatten().cpu().long()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    if scores.numel() == 0:
        raise ValueError("metrics require at least one example.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary with spoof encoded as 1.")
    predictions = (scores >= threshold).long()
    true_positive = ((predictions == 1) & (labels == 1)).sum().item()
    false_positive = ((predictions == 1) & (labels == 0)).sum().item()
    false_negative = ((predictions == 0) & (labels == 1)).sum().item()
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    return BinaryMetrics(
        accuracy=(predictions == labels).float().mean().item(),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        eer=_eer(scores, labels),
    )
