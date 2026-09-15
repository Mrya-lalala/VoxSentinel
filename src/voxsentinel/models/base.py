"""Contract B2 detector backbones must satisfy."""

from abc import ABC, abstractmethod

from torch import Tensor, nn


class Backbone(nn.Module, ABC):
    """Maps already-prepared embedding sequences to two-class logits."""

    @abstractmethod
    def forward(
        self,
        features: Tensor,
        valid_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return logits with shape ``[batch, 2]`` for features ``[B, T, D]``."""
        raise NotImplementedError
