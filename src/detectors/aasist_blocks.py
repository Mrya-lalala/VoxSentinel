"""AASIST graph blocks ported for embedding input.

Ported from https://github.com/clovaai/aasist at commit
``a04c9863f63d44471dde8a6abcb3b082b07cd1d1`` (``models/AASIST.py``):
``GraphAttentionLayer``, ``HtrgGraphAttentionLayer`` and ``GraphPool``.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license - see ``docs/third_party/AASIST_NOTICE.md``.

Local modifications (the attention math, temperature handling, batch
normalization, SELU activations, type-masked heterogeneous attention and
top-k graph pooling are unchanged):

* dropout probabilities are constructor arguments instead of hardcoded values,
  and dropout is non-inplace;
* ``_apply_BN`` uses ``reshape`` so non-contiguous inputs are accepted;
* master nodes are passed explicitly with shape ``[B, 1, dim]``.

These blocks operate on fixed-size node sets; they know nothing about the
encoder, the adapter or the padding mask.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class GraphAttentionLayer(nn.Module):
    """Homogeneous dense graph attention over one node set ``[B, N, dim]``."""

    def __init__(self, in_dim: int, out_dim: int, *, temperature: float = 1.0, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.SELU(inplace=True)
        self.temp = float(temperature)

    def forward(self, x: Tensor) -> Tensor:
        """x: ``[B, N, dim]`` -> ``[B, N, out_dim]``."""
        x = self.input_drop(x)
        att_map = self._derive_att_map(x)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        return self.act(x)

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x: Tensor) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.reshape(-1, org_size[-1])
        x = self.bn(x)
        return x.reshape(org_size)

    def _init_new_params(self, *size: int) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    """Heterogeneous attention over two node types plus one master node."""

    def __init__(self, in_dim: int, out_dim: int, *, temperature: float = 1.0, dropout: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=dropout)
        self.act = nn.SELU(inplace=True)
        self.temp = float(temperature)

    def forward(self, x1: Tensor, x2: Tensor, master: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """``x1``/``x2``: ``[B, N1|N2, dim]``; ``master``: ``[B, 1, dim]``."""
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)
        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        x = self.input_drop(x)
        att_map = self._derive_att_map(x, num_type1, num_type2)
        master = self._update_master(x, master)
        x = self._project(x, att_map)
        x = self._apply_BN(x)
        x = self.act(x)

        return x.narrow(1, 0, num_type1), x.narrow(1, num_type1, num_type2), master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = self._derive_att_map_master(x, master)
        return self._project_master(x, master, att_map)

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        return F.softmax(att_map, dim=-2)

    def _derive_att_map(self, x: Tensor, num_type1: int, num_type2: int) -> Tensor:
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)
        att_board[:, :num_type1, :num_type1, :] = torch.matmul(att_map[:, :num_type1, :num_type1, :], self.att_weight11)
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(att_map[:, num_type1:, num_type1:, :], self.att_weight22)
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(att_map[:, :num_type1, num_type1:, :], self.att_weight12)
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(att_map[:, num_type1:, :num_type1, :], self.att_weight12)

        att_map = att_board / self.temp
        return F.softmax(att_map, dim=-2)

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(self, x: Tensor, master: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_attM(torch.matmul(att_map.squeeze(-1).unsqueeze(1), x))
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.reshape(-1, org_size[-1])
        x = self.bn(x)
        return x.reshape(org_size)

    def _init_new_params(self, *size: int) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    """Learned top-k node selection with sigmoid score weighting."""

    def __init__(self, k: float, in_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        if not 0.0 < float(k) <= 1.0:
            raise ValueError("GraphPool ratio must be in (0, 1].")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        self.k = float(k)
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        z = self.drop(h)
        scores = self.sigmoid(self.proj(z))
        return self.top_k_graph(scores, h, self.k)

    def top_k_graph(self, scores: Tensor, h: Tensor, k: float) -> Tensor:
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)
        h = h * scores
        return torch.gather(h, 1, idx)


__all__ = ["GraphAttentionLayer", "GraphPool", "HtrgGraphAttentionLayer"]
