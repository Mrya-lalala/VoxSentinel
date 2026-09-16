"""Realizable operating-threshold selection on development scores.

The EER is a ranking metric and its interpolated value need not be attainable
by any single deterministic threshold.  This module therefore selects the
observed score threshold whose error rates are closest to the crossing, and
records the error rates at that threshold.  Select on development data only and
freeze the result before touching held-out metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..data.labels import GENUINE, SPOOF, validate_label_tensor


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    genuine_false_alarm_rate: float
    spoof_miss_rate: float
    rule: str = "closest_empirical_crossing"


def select_threshold(scores: Tensor, labels: Tensor) -> ThresholdSelection:
    """Pick the observed threshold minimizing |FAR - FRR| on a dev split.

    Tie-breaking: smaller |FAR - FRR|, then smaller FAR + FRR, then the higher
    threshold.  Requires both classes; raises otherwise.
    """
    values = torch.as_tensor(scores, dtype=torch.float32).flatten().cpu()
    targets = validate_label_tensor(labels).flatten().cpu()
    if values.numel() != targets.numel() or values.numel() == 0:
        raise ValueError("scores and labels must be non-empty and equally sized.")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("scores contain non-finite values.")
    genuine_count = int((targets == GENUINE).sum())
    spoof_count = int((targets == SPOOF).sum())
    if not genuine_count or not spoof_count:
        raise ValueError("threshold selection requires both genuine and spoof examples.")

    unique_scores, inverse = torch.unique(values, sorted=True, return_inverse=True)
    per_bucket_genuine = torch.zeros(unique_scores.numel(), dtype=torch.float64)
    per_bucket_spoof = torch.zeros(unique_scores.numel(), dtype=torch.float64)
    per_bucket_genuine.index_add_(0, inverse, (targets == GENUINE).to(torch.float64))
    per_bucket_spoof.index_add_(0, inverse, (targets == SPOOF).to(torch.float64))

    thresholds = torch.flip(unique_scores, [0])  # descending, matching the curve
    false_alarms = torch.cumsum(torch.flip(per_bucket_genuine, [0]), 0) / genuine_count
    misses = 1.0 - torch.cumsum(torch.flip(per_bucket_spoof, [0]), 0) / spoof_count

    # Tie-break: closest crossing, then lowest total error, then highest threshold.
    index = min(
        range(thresholds.numel()),
        key=lambda i: (abs(float(false_alarms[i]) - float(misses[i])), float(false_alarms[i]) + float(misses[i]), i),
    )
    return ThresholdSelection(
        threshold=float(thresholds[index]),
        genuine_false_alarm_rate=float(false_alarms[index]),
        spoof_miss_rate=float(misses[index]),
    )
