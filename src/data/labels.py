"""Label convention shared by data, detector and metric boundaries.

Class 0 = genuine (bona fide), class 1 = spoof.  Validation happens *before*
any integer conversion so fractional values such as 0.5 cannot silently become
a valid-looking class index.
"""

from __future__ import annotations

import math
import numbers

import torch
from torch import Tensor

GENUINE = 0
SPOOF = 1

_INT_DTYPES = (
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
)


def validate_label(value: object) -> int:
    """Return the label as ``int`` or raise for anything that is not 0 or 1."""
    if isinstance(value, bool):
        raise ValueError("labels must be the integers 0 (genuine) or 1 (spoof), not booleans.")
    if isinstance(value, numbers.Integral):
        label = int(value)
    elif isinstance(value, numbers.Real):
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            raise ValueError(f"labels must be binary (0=genuine, 1=spoof) and integral; got {value!r}.")
        label = int(number)
    else:
        raise ValueError(f"labels must be binary (0=genuine, 1=spoof); got {value!r}.")
    if label not in (GENUINE, SPOOF):
        raise ValueError(f"labels must be binary (0=genuine, 1=spoof); got {label}.")
    return label


def validate_label_tensor(labels: Tensor) -> Tensor:
    """Return a validated ``int64`` label tensor (any shape) with values in {0,1}."""
    tensor = torch.as_tensor(labels)
    if tensor.dtype == torch.bool:
        raise ValueError("labels must be binary (0=genuine, 1=spoof), not boolean.")
    if tensor.is_floating_point():
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError("labels contain non-finite values.")
        if not torch.equal(tensor, tensor.round()):
            raise ValueError("labels must be binary (0=genuine, 1=spoof) and integral.")
    elif tensor.dtype not in _INT_DTYPES:
        raise ValueError(f"labels must be integer-valued; got dtype {tensor.dtype}.")
    tensor = tensor.long()
    if tensor.numel() and not bool(((tensor == GENUINE) | (tensor == SPOOF)).all()):
        raise ValueError("labels must be binary (0=genuine, 1=spoof).")
    return tensor
