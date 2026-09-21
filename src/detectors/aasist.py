"""Embedding-adapted AASIST research head for frozen IndicWav2Vec features.

This is not the published waveform AASIST: the sinc/convolutional front-end is
replaced by a validated embedding adapter, and the graph backend is the
MIT-licensed AASIST graph code (see ``aasist_blocks.py`` and
``docs/third_party/AASIST_NOTICE.md``). The F axis below is a learned latent
feature axis, not a physical frequency axis.

Pipeline (Gate 1 plan in ``docs/AASIST_IMPLEMENTATION_PLAN.md``):

    features [B, T, 1024] + valid_lengths + padding_mask (True = padding)
      -> fixed FrameStandardizer (train-fit, fail-closed) on valid frames
      -> per-frame LayerNorm(1024)
      -> valid-slice adaptive average pooling to K = 32 temporal bins
      -> Linear(1024 -> C * F) with C = 32, F = 16 -> [B, C, F, K]
      -> latent-feature nodes [B, F, C] (max|.| over K) + learned pos_F
         temporal nodes       [B, K, C] (max|.| over F)
      -> GAT_S / GAT_T (dense temperature attention) + learned GraphPool
      -> two heterogeneous branches (type-masked attention + master nodes)
         with residual adds, fused by elementwise max
      -> readout concat[T_max, T_avg, S_max, S_avg, master]
      -> Linear(5 * gat_dims[1], 2) genuine/spoof logits
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import ModelConfig
from .aasist_blocks import GraphAttentionLayer, GraphPool, HtrgGraphAttentionLayer
from .base import Detector, DetectorOutput
from .registry import register_detector
from .standardization import FrameStandardizer


class AasistSpoofDetector(nn.Module):
    """Trainable genuine/spoof head adapted from AASIST for embedding input."""

    def __init__(
        self,
        *,
        input_dim: int = 1024,
        pooled_bins: int = 32,
        channels: int = 32,
        latent_nodes: int = 16,
        gat_dims: Sequence[int] = (64, 32),
        pool_ratios: Sequence[float] = (0.5, 0.7, 0.5),
        temperatures: Sequence[float] = (2.0, 2.0, 100.0),
        dropout_attention: float = 0.2,
        dropout_pool: float = 0.3,
        dropout_fusion: float = 0.2,
        dropout_readout: float = 0.5,
        feature_standardization: bool = True,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive.")
        if pooled_bins < 2:
            raise ValueError("pooled_bins must be at least 2.")
        if channels <= 0 or latent_nodes <= 0:
            raise ValueError("channels and latent_nodes must be positive.")
        if len(gat_dims) != 2 or any(int(dim) <= 0 for dim in gat_dims):
            raise ValueError("gat_dims must contain two positive dimensions.")
        if len(pool_ratios) < 3 or any(not 0.0 < float(ratio) <= 1.0 for ratio in pool_ratios[:3]):
            raise ValueError("pool_ratios must provide three ratios in (0, 1].")
        if len(temperatures) < 3 or any(float(temp) <= 0 for temp in temperatures[:3]):
            raise ValueError("temperatures must provide three positive values.")
        if num_classes != 2:
            raise ValueError("The AASIST head uses exactly two classes (0=genuine, 1=spoof).")

        self.input_dim = int(input_dim)
        self.pooled_bins = int(pooled_bins)
        self.channels = int(channels)
        self.latent_nodes = int(latent_nodes)
        self.gat_dims = (int(gat_dims[0]), int(gat_dims[1]))
        self.pool_ratios = tuple(float(ratio) for ratio in pool_ratios[:3])
        self.temperatures = tuple(float(temp) for temp in temperatures[:3])
        self.num_classes = int(num_classes)

        self.standardizer = FrameStandardizer(self.input_dim) if feature_standardization else None
        self.input_norm = nn.LayerNorm(self.input_dim)
        self.projection = nn.Linear(self.input_dim, self.channels * self.latent_nodes)
        self.pos_feature = nn.Parameter(torch.randn(1, self.latent_nodes, self.channels))

        self.gat_s = GraphAttentionLayer(
            self.channels, self.gat_dims[0], temperature=self.temperatures[0], dropout=dropout_attention
        )
        self.gat_t = GraphAttentionLayer(
            self.channels, self.gat_dims[0], temperature=self.temperatures[1], dropout=dropout_attention
        )
        self.pool_s = GraphPool(self.pool_ratios[0], self.gat_dims[0], dropout_pool)
        self.pool_t = GraphPool(self.pool_ratios[1], self.gat_dims[0], dropout_pool)

        self.master1 = nn.Parameter(torch.randn(1, 1, self.gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, self.gat_dims[0]))

        self.hetero_st11 = HtrgGraphAttentionLayer(
            self.gat_dims[0], self.gat_dims[1], temperature=self.temperatures[2], dropout=dropout_attention
        )
        self.hetero_st12 = HtrgGraphAttentionLayer(
            self.gat_dims[1], self.gat_dims[1], temperature=self.temperatures[2], dropout=dropout_attention
        )
        self.hetero_st21 = HtrgGraphAttentionLayer(
            self.gat_dims[0], self.gat_dims[1], temperature=self.temperatures[2], dropout=dropout_attention
        )
        self.hetero_st22 = HtrgGraphAttentionLayer(
            self.gat_dims[1], self.gat_dims[1], temperature=self.temperatures[2], dropout=dropout_attention
        )
        self.pool_hs1 = GraphPool(self.pool_ratios[2], self.gat_dims[1], dropout_pool)
        self.pool_ht1 = GraphPool(self.pool_ratios[2], self.gat_dims[1], dropout_pool)
        self.pool_hs2 = GraphPool(self.pool_ratios[2], self.gat_dims[1], dropout_pool)
        self.pool_ht2 = GraphPool(self.pool_ratios[2], self.gat_dims[1], dropout_pool)

        self.dropout_fusion = nn.Dropout(dropout_fusion)
        self.dropout_readout = nn.Dropout(dropout_readout)
        self.out_layer = nn.Linear(5 * self.gat_dims[1], self.num_classes)

    # ------------------------------------------------------------------ input
    def _valid_mask(
        self,
        features: Tensor,
        valid_lengths: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        """Resolve a contiguous right-prefix valid mask or raise.

        Same semantics as the GRU head: ``padding_mask`` is boolean with
        True = padding, ``valid_lengths`` counts output frames, and the two are
        intersected when both are supplied.
        """
        batch, frames, _ = features.shape
        positions = torch.arange(frames, device=features.device).unsqueeze(0)
        valid = torch.ones((batch, frames), dtype=torch.bool, device=features.device)

        if valid_lengths is not None:
            lengths = valid_lengths
            if lengths.dtype not in (torch.int32, torch.int64):
                raise ValueError(f"valid_lengths must be an integer tensor; got dtype {lengths.dtype}.")
            lengths = lengths.to(device=features.device, dtype=torch.long)
            if lengths.ndim != 1 or lengths.shape[0] != batch:
                raise ValueError(f"valid_lengths must have shape [batch]; got {tuple(valid_lengths.shape)}.")
            if bool(((lengths < 0) | (lengths > frames)).any()):
                raise ValueError(f"valid_lengths must be within [0, {frames}] (output frames, not samples).")
            valid = valid & (positions < lengths.unsqueeze(1))

        if padding_mask is not None:
            if padding_mask.dtype != torch.bool:
                raise ValueError(f"padding_mask must be boolean with True = padding; got dtype {padding_mask.dtype}.")
            if padding_mask.ndim != 2 or tuple(padding_mask.shape) != (batch, frames):
                raise ValueError(f"padding_mask must have shape [batch, time] = [{batch}, {frames}].")
            valid = valid & ~padding_mask.to(device=features.device)

        expected = positions < valid.sum(dim=1, keepdim=True)
        if not torch.equal(valid, expected):
            raise ValueError(
                "valid_lengths/padding_mask must describe right padding only; "
                "left padding or masks with holes are not supported."
            )
        if bool((valid.sum(dim=1) == 0).any()):
            raise ValueError("Every sequence must contain at least one valid frame.")
        return valid

    def _validate_inputs(
        self,
        features: Tensor,
        valid_lengths: Tensor | None,
        padding_mask: Tensor | None,
    ) -> Tensor:
        if features.ndim != 3:
            raise ValueError(f"AASIST head expects features [batch, time, dim]; got {tuple(features.shape)}.")
        if features.shape[-1] != self.input_dim:
            raise ValueError(f"Expected embedding dim {self.input_dim}; got {features.shape[-1]}.")
        if features.dtype != torch.float32:
            raise ValueError(f"AASIST head expects float32 features; got {features.dtype}.")
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features contain non-finite values.")
        if features.shape[1] < self.pooled_bins:
            raise ValueError(
                f"Unsupported sequence length {features.shape[1]}: at least {self.pooled_bins} frames "
                "(the pooled-bin count) are required; shorter windows are rejected rather than repeated or padded."
            )

        valid = self._valid_mask(features, valid_lengths, padding_mask)
        lengths = valid.sum(dim=1)
        if int(lengths.min()) < self.pooled_bins:
            raise ValueError(
                f"Unsupported valid length {int(lengths.min())}: at least {self.pooled_bins} valid frames "
                "are required for the fixed-size pooling contract."
            )
        return valid

    # ---------------------------------------------------------------- adapter
    def _pool_valid_frames(self, features: Tensor, valid: Tensor) -> Tensor:
        """Adaptive-average-pool each valid slice into ``pooled_bins`` bins."""
        batch, _, dim = features.shape
        lengths = valid.sum(dim=1).tolist()
        pooled = features.new_zeros((batch, self.pooled_bins, dim))
        for index, length in enumerate(lengths):
            item = features[index, : int(length)].transpose(0, 1).unsqueeze(0)  # [1, dim, length]
            item = F.adaptive_avg_pool1d(item, self.pooled_bins).squeeze(0)  # [dim, bins]
            pooled[index] = item.transpose(0, 1)
        return pooled

    def _project_nodes(self, features: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        pooled = self._pool_valid_frames(features, valid)
        projected = self.projection(pooled)
        batch = features.shape[0]
        projected = projected.reshape(batch, self.pooled_bins, self.channels, self.latent_nodes)
        projected = projected.permute(0, 2, 3, 1).contiguous()  # [B, C, F, K]
        if tuple(projected.shape) != (batch, self.channels, self.latent_nodes, self.pooled_bins):
            raise RuntimeError(f"Unexpected adapter tensor shape {tuple(projected.shape)}.")

        # Upstream reduction convention: max-absolute over the opposite axis.
        feature_nodes = torch.max(torch.abs(projected), dim=3).values  # [B, C, F]
        feature_nodes = feature_nodes.transpose(1, 2) + self.pos_feature  # [B, F, C]
        temporal_nodes = torch.max(torch.abs(projected), dim=2).values  # [B, C, K]
        temporal_nodes = temporal_nodes.transpose(1, 2)  # [B, K, C]
        return temporal_nodes, feature_nodes

    def _run_branch(
        self,
        first: HtrgGraphAttentionLayer,
        pool_hs: GraphPool,
        pool_ht: GraphPool,
        second: HtrgGraphAttentionLayer,
        temporal_nodes: Tensor,
        feature_nodes: Tensor,
        master: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        out_t, out_s, master = first(temporal_nodes, feature_nodes, master=master)
        out_s = pool_hs(out_s)
        out_t = pool_ht(out_t)
        aug_t, aug_s, aug_master = second(out_t, out_s, master=master)
        return out_t + aug_t, out_s + aug_s, master + aug_master

    # ---------------------------------------------------------------- forward
    def forward(
        self,
        features: Tensor,
        valid_lengths: Tensor | None = None,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Return genuine/spoof logits ``[batch, 2]``."""
        valid = self._validate_inputs(features, valid_lengths, padding_mask)

        if self.standardizer is not None:
            features = self.standardizer(features, valid)
        features = self.input_norm(features)
        features = features.masked_fill(~valid.unsqueeze(-1), 0.0)

        temporal_nodes, feature_nodes = self._project_nodes(features, valid)

        gat_s = self.gat_s(feature_nodes)
        out_s = self.pool_s(gat_s)
        gat_t = self.gat_t(temporal_nodes)
        out_t = self.pool_t(gat_t)

        batch = features.shape[0]
        master1 = self.master1.expand(batch, -1, -1)
        master2 = self.master2.expand(batch, -1, -1)

        out_t1, out_s1, master1 = self._run_branch(
            self.hetero_st11, self.pool_hs1, self.pool_ht1, self.hetero_st12, out_t, out_s, master1
        )
        out_t2, out_s2, master2 = self._run_branch(
            self.hetero_st21, self.pool_hs2, self.pool_ht2, self.hetero_st22, out_t, out_s, master2
        )

        out_t1, out_t2 = self.dropout_fusion(out_t1), self.dropout_fusion(out_t2)
        out_s1, out_s2 = self.dropout_fusion(out_s1), self.dropout_fusion(out_s2)
        master1, master2 = self.dropout_fusion(master1), self.dropout_fusion(master2)

        out_t = torch.maximum(out_t1, out_t2)
        out_s = torch.maximum(out_s1, out_s2)
        master = torch.maximum(master1, master2)

        t_max = torch.max(torch.abs(out_t), dim=1).values
        t_avg = torch.mean(out_t, dim=1)
        s_max = torch.max(torch.abs(out_s), dim=1).values
        s_avg = torch.mean(out_s, dim=1)

        last_hidden = torch.cat([t_max, t_avg, s_max, s_avg, master.squeeze(1)], dim=1)
        return self.out_layer(self.dropout_readout(last_hidden))


class AasistDetector(Detector):
    """Batch adapter: ``features`` plus optional ``valid_lengths``/``padding_mask``."""

    def __init__(self, model: AasistSpoofDetector) -> None:
        super().__init__()
        self.model = model

    @classmethod
    def from_config(cls, config: ModelConfig) -> "AasistDetector":
        return cls(AasistSpoofDetector(**config.parameters))

    def forward(self, batch: Mapping[str, Tensor]) -> DetectorOutput:
        if "features" not in batch:
            raise KeyError("AASIST detector batches must contain a 'features' tensor [batch, time, dim].")
        return DetectorOutput(
            logits=self.model(
                batch["features"],
                batch.get("valid_lengths"),
                batch.get("padding_mask"),
            )
        )


register_detector("aasist", AasistDetector.from_config)

__all__ = ["AasistDetector", "AasistSpoofDetector"]
