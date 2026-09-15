"""Minimal reusable loops for B2 detector adapters."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
from torch import Tensor

from ..detectors.base import Detector

LossFn = Callable[[Tensor, Tensor], Tensor]


@dataclass(frozen=True)
class EpochResult:
    loss: float
    steps: int
    examples: int


def _move(batch: Mapping[str, Tensor], device: torch.device | str) -> dict[str, Tensor]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def train_epoch(
    detector: Detector,
    batches: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    device: torch.device | str = "cpu",
    loss_fn: LossFn | None = None,
) -> EpochResult:
    detector.train()
    total_loss, steps, examples = 0.0, 0, 0
    for batch in batches:
        moved = _move(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = detector(moved)
        loss = loss_fn(output.logits, moved["labels"].long()) if loss_fn is not None else detector.loss(output, moved)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach().item()
        steps += 1
        examples += int(moved["labels"].shape[0])
    return EpochResult(loss=total_loss / steps if steps else 0.0, steps=steps, examples=examples)


@torch.no_grad()
def validate_epoch(
    detector: Detector,
    batches: Iterable[Mapping[str, Tensor]],
    device: torch.device | str = "cpu",
    loss_fn: LossFn | None = None,
) -> EpochResult:
    detector.eval()
    total_loss, steps, examples = 0.0, 0, 0
    for batch in batches:
        moved = _move(batch, device)
        output = detector(moved)
        loss = loss_fn(output.logits, moved["labels"].long()) if loss_fn is not None else detector.loss(output, moved)
        total_loss += loss.item()
        steps += 1
        examples += int(moved["labels"].shape[0])
    return EpochResult(loss=total_loss / steps if steps else 0.0, steps=steps, examples=examples)
