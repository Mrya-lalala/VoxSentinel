"""Fixed training-frame statistics, serialized with the detector, never optimized."""
from __future__ import annotations

import hashlib
from typing import Any, Sequence

import torch
from torch import nn

SCHEMA = "voxsentinel.frame_standardizer.v1"


class FrameStandardizer(nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dimension))
        self.register_buffer("scale", torch.ones(dimension))
        self.register_buffer("population_variance", torch.zeros(dimension, dtype=torch.float64))
        self.register_buffer("fitted", torch.tensor(False))
        self.metadata: dict[str, Any] = {}

    def get_extra_state(self):
        return dict(self.metadata)

    def set_extra_state(self, state):
        if not isinstance(state, dict) or state.get("schema") != SCHEMA:
            raise ValueError("Missing or incompatible frame-standardizer metadata")
        self.metadata = dict(state)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Mandatory even for strict=False: inference must never silently omit scaling.
        required = ("mean", "scale", "population_variance", "fitted", "_extra_state")
        if any(prefix + key not in state_dict for key in required):
            error_msgs.append("Required frame-standardizer state is missing")
        else:
            mean, scale = state_dict[prefix + "mean"], state_dict[prefix + "scale"]
            if not bool(state_dict[prefix + "fitted"]) or not torch.isfinite(mean).all() or not torch.isfinite(scale).all() or not (scale > 0).all():
                error_msgs.append("Invalid or unfitted frame-standardizer state")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                     missing_keys, unexpected_keys, error_msgs)

    @torch.no_grad()
    def fit(self, examples: Sequence, *, cache_signature: str | None = None) -> dict:
        if bool(self.fitted):
            raise ValueError("Refusing to refit a frozen standardizer")
        if not examples:
            raise ValueError("Training examples are required")
        n = 0
        mean = torch.zeros(self.mean.numel(), dtype=torch.float64)
        m2 = torch.zeros_like(mean)
        digest = hashlib.sha256()
        for example in examples:
            x = example.features.detach().cpu()
            if x.ndim != 2 or x.shape[1] != mean.numel() or not x.shape[0] or not torch.isfinite(x).all():
                raise ValueError("Expected finite unpadded training frames [T,D]")
            digest.update(str((tuple(x.shape), str(x.dtype), int(example.label))).encode())
            digest.update(x.contiguous().numpy().tobytes())
            x = x.double()
            count = x.shape[0]
            batch_mean = x.mean(0)
            batch_m2 = ((x - batch_mean) ** 2).sum(0)
            delta = batch_mean - mean
            total = n + count
            m2 += batch_m2 + delta.square() * (n * count / total)
            mean += delta * (count / total)
            n = total
        variance = m2 / n
        sd = variance.sqrt()
        self.mean.copy_(mean)
        self.scale.copy_(sd.clamp_min(1e-6))
        self.population_variance.copy_(variance)
        self.metadata = {
            "schema": SCHEMA, "placement": "before_existing_input_LayerNorm",
            "fitted_on": "training only", "weighting": "valid-frame weighted",
            "variance": "population (ddof=0), float64 parallel Welford accumulation",
            "epsilon": 1e-6, "valid_frames": n, "windows": len(examples),
            "dimension": mean.numel(), "clamped_dimensions": int((sd < 1e-6).sum()),
            "applied_dtype": str(self.mean.dtype), "training_cache_sha256": cache_signature,
            "training_examples_sha256": digest.hexdigest(),
        }
        self.fitted.fill_(True)
        return dict(self.metadata)

    def forward(self, features, valid):
        if not bool(self.fitted):
            raise ValueError("Frame standardization enabled but training statistics are absent")
        return ((features - self.mean) / self.scale).masked_fill(~valid.unsqueeze(-1), 0.0)


def fit_detector_standardizer(detector, examples, *, cache_signature=None):
    transform = getattr(getattr(detector, "model", None), "standardizer", None)
    return transform.fit(examples, cache_signature=cache_signature) if transform is not None else None
