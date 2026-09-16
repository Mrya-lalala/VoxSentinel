"""Shared evaluation primitives: metrics, thresholds and score aggregation.

Ownership note: C1 owns the shared metric layer.  B2 provides this initial
implementation (it is needed for head training/validation); C1 should extend it
in place rather than creating a parallel copy.
"""

from .evaluation import EvaluationResult, evaluate
from .metrics import BinaryMetrics, binary_metrics, error_rates
from .thresholds import ThresholdSelection, select_threshold

__all__ = [
    "BinaryMetrics",
    "EvaluationResult",
    "ThresholdSelection",
    "binary_metrics",
    "error_rates",
    "evaluate",
    "select_threshold",
]
