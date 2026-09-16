"""Binary genuine/spoof metrics with an O(N log N) EER.

Conventions: class 0 = genuine, class 1 = spoof, score = P(spoof), and a sample
is predicted spoof when ``score >= threshold``.  Error rates are named from the
screening perspective:

- ``genuine_false_alarm_rate``: genuine audio flagged as spoof (FP / genuine).
- ``spoof_miss_rate``: spoof audio accepted as genuine (FN / spoof).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.labels import GENUINE, SPOOF, validate_label_tensor


@dataclass(frozen=True)
class BinaryMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    eer: float | None
    threshold: float | None = None
    true_negative: int | None = None
    false_positive: int | None = None
    false_negative: int | None = None
    true_positive: int | None = None
    genuine_false_alarm_rate: float | None = None
    spoof_miss_rate: float | None = None


def _validate_scores(scores: Tensor) -> Tensor:
    values = torch.as_tensor(scores, dtype=torch.float32).flatten().cpu()
    if values.numel() == 0:
        raise ValueError("metrics require at least one example.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores contain non-finite values.")
    return values


def error_rates(scores: Tensor, labels: Tensor, threshold: float) -> tuple[float, float]:
    """Return ``(genuine_false_alarm_rate, spoof_miss_rate)`` at ``threshold``."""
    values = _validate_scores(scores)
    targets = validate_label_tensor(labels).flatten().cpu()
    if values.numel() != targets.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be a finite number.")

    predicted_spoof = values >= float(threshold)
    genuine, spoof = targets == GENUINE, targets == SPOOF
    genuine_count = int(genuine.sum())
    spoof_count = int(spoof.sum())
    false_alarms = int((predicted_spoof & genuine).sum())
    misses = int((~predicted_spoof & spoof).sum())
    false_alarm_rate = false_alarms / genuine_count if genuine_count else 0.0
    miss_rate = misses / spoof_count if spoof_count else 0.0
    return false_alarm_rate, miss_rate


def _curve(values: Tensor, targets: Tensor) -> tuple[Tensor, Tensor]:
    """Return FAR/FRR in descending-threshold order, including both extremes.

    Index 0 is "threshold above the highest score" (predict nothing as spoof)
    and the last index is "threshold below the lowest score" (predict
    everything as spoof), so FAR is non-decreasing along the curve.
    """
    genuine_count = int((targets == GENUINE).sum())
    spoof_count = int((targets == SPOOF).sum())
    _, inverse = torch.unique(values, sorted=True, return_inverse=True)
    per_bucket_genuine = torch.zeros(int(inverse.max()) + 1, dtype=torch.float64)
    per_bucket_spoof = torch.zeros(int(inverse.max()) + 1, dtype=torch.float64)
    per_bucket_genuine.index_add_(0, inverse, (targets == GENUINE).to(torch.float64))
    per_bucket_spoof.index_add_(0, inverse, (targets == SPOOF).to(torch.float64))

    # Buckets are ascending by score; flip to descending-threshold order first.
    false_positives = torch.cumsum(torch.flip(per_bucket_genuine, [0]), 0)
    true_positives = torch.cumsum(torch.flip(per_bucket_spoof, [0]), 0)
    far = torch.cat(
        [torch.zeros(1, dtype=torch.float64), false_positives / genuine_count, torch.ones(1, dtype=torch.float64)]
    )
    frr = torch.cat(
        [torch.ones(1, dtype=torch.float64), 1.0 - true_positives / spoof_count, torch.zeros(1, dtype=torch.float64)]
    )
    return far, frr


def _eer(values: Tensor, targets: Tensor) -> float:
    """Equal-error-rate point from the piecewise-linear ROC curve.

    The crossing is found on the ROC segment where ``FAR - FRR`` changes sign;
    the returned value is the interpolated FAR (equal to FRR) at that point.
    """
    far, frr = _curve(values, targets)
    difference = far - frr
    for index in range(1, difference.numel()):
        left, right = float(difference[index - 1]), float(difference[index])
        if right < 0:
            continue
        if right - left <= 1e-12:
            weight = 0.0
        else:
            weight = -left / (right - left)
        return float(far[index - 1]) + weight * (float(far[index]) - float(far[index - 1]))
    return 0.0


def binary_metrics(scores: Tensor, labels: Tensor, threshold: float = 0.5) -> BinaryMetrics:
    """Classification metrics at ``threshold`` plus the ranking EER.

    ``eer`` is ``None`` when only one class is present.  The threshold is a
    classification parameter selected elsewhere; it is never tuned on held-out
    scores.
    """
    values = _validate_scores(scores)
    targets = validate_label_tensor(labels).flatten().cpu()
    if values.numel() != targets.numel():
        raise ValueError("scores and labels must have the same number of elements.")
    if not isinstance(threshold, (int, float)) or not math.isfinite(float(threshold)):
        raise ValueError("threshold must be a finite number.")
    if values.numel() == 0:
        raise ValueError("metrics require at least one example.")

    predicted_spoof = values >= float(threshold)
    genuine, spoof = targets == GENUINE, targets == SPOOF
    true_positive = int((predicted_spoof & spoof).sum())
    false_positive = int((predicted_spoof & genuine).sum())
    false_negative = int((~predicted_spoof & spoof).sum())
    true_negative = int((~predicted_spoof & genuine).sum())

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    genuine_count, spoof_count = int(genuine.sum()), int(spoof.sum())
    eer = _eer(values, targets) if genuine_count and spoof_count else None

    return BinaryMetrics(
        accuracy=(predicted_spoof == spoof).float().mean().item(),
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        eer=eer,
        threshold=float(threshold),
        true_negative=true_negative,
        false_positive=false_positive,
        false_negative=false_negative,
        true_positive=true_positive,
        genuine_false_alarm_rate=false_positive / genuine_count if genuine_count else 0.0,
        spoof_miss_rate=false_negative / spoof_count if spoof_count else 0.0,
    )
