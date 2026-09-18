"""Configuration for the bounded dataset-preparation pilot.

Loads ``configs/datasets.yaml`` into small dataclasses.  Paths in the manifest
are always stored relative to :attr:`PrepConfig.dataset_root`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/datasets.yaml"


@dataclass(frozen=True)
class WindowPolicy:
    max_seconds: float = 4.0
    min_seconds: float = 1.0
    scan_hop_seconds: float = 0.25
    silence_rms_threshold: float = 0.00316
    activity_frame_ms: float = 30.0
    activity_rms_threshold: float = 0.0025
    min_active_fraction: float = 0.2


@dataclass(frozen=True)
class BudgetLimits:
    max_download_bytes: int = 5 * 1024**3
    max_metadata_requests_per_source: int = 400
    request_timeout_seconds: float = 60.0


@dataclass(frozen=True)
class PrepConfig:
    dataset_root: Path
    seed: int
    window: WindowPolicy
    budget: BudgetLimits
    quotas: dict[str, Any]
    releases: dict[str, Any]
    preprocessing: dict[str, Any]
    features: dict[str, Any]
    config_path: Path

    # ---- path helpers -------------------------------------------------

    @property
    def manifests_dir(self) -> Path:
        return self.dataset_root / "manifests"

    @property
    def raw_dir(self) -> Path:
        return self.dataset_root / "raw"

    @property
    def prepared_dir(self) -> Path:
        return self.dataset_root / "prepared"

    @property
    def features_dir(self) -> Path:
        return self.dataset_root / "features"

    @property
    def reports_dir(self) -> Path:
        return self.dataset_root / "reports"

    @property
    def staging_dir(self) -> Path:
        return self.dataset_root / "staging"

    def ensure_dirs(self) -> None:
        for path in (
            self.manifests_dir,
            self.raw_dir,
            self.prepared_dir,
            self.features_dir,
            self.reports_dir,
            self.staging_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def release(self, source_id: str) -> dict[str, Any]:
        return dict(self.releases.get(source_id, {}))

    def quota(self, source_id: str) -> dict[str, Any]:
        return dict(self.quotas.get(source_id, {}))

    def source_enabled(self, source_id: str) -> bool:
        return bool(self.quota(source_id).get("enabled", False))


def load_prep_config(
    path: str | Path = DEFAULT_CONFIG_PATH,
    *,
    dataset_root: str | Path | None = None,
) -> PrepConfig:
    """Load and validate the pilot configuration."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping.")

    root = Path(dataset_root) if dataset_root is not None else Path(data.get("dataset_root", "artifacts/datasets"))
    window_raw = data.get("window", {}) or {}
    budget_raw = data.get("budget", {}) or {}

    window = WindowPolicy(
        max_seconds=float(window_raw.get("max_seconds", 4.0)),
        min_seconds=float(window_raw.get("min_seconds", 1.0)),
        scan_hop_seconds=float(window_raw.get("scan_hop_seconds", 0.25)),
        silence_rms_threshold=float(window_raw.get("silence_rms_threshold", 0.00316)),
        activity_frame_ms=float(window_raw.get("activity_frame_ms", 30.0)),
        activity_rms_threshold=float(window_raw.get("activity_rms_threshold", 0.0025)),
        min_active_fraction=float(window_raw.get("min_active_fraction", 0.2)),
    )
    if not 0 < window.min_seconds <= window.max_seconds:
        raise ValueError("window.min_seconds must be positive and not exceed window.max_seconds.")
    if window.scan_hop_seconds <= 0:
        raise ValueError("window.scan_hop_seconds must be positive.")
    if not 0.0 <= window.min_active_fraction <= 1.0:
        raise ValueError("window.min_active_fraction must be in [0, 1].")

    budget = BudgetLimits(
        max_download_bytes=int(budget_raw.get("max_download_bytes", 5 * 1024**3)),
        max_metadata_requests_per_source=int(budget_raw.get("max_metadata_requests_per_source", 400)),
        request_timeout_seconds=float(budget_raw.get("request_timeout_seconds", 60.0)),
    )
    if budget.max_download_bytes <= 0:
        raise ValueError("budget.max_download_bytes must be positive.")

    return PrepConfig(
        dataset_root=root,
        seed=int(data.get("seed", 42)),
        window=window,
        budget=budget,
        quotas=dict(data.get("quotas", {}) or {}),
        releases=dict(data.get("releases", {}) or {}),
        preprocessing=dict(data.get("preprocessing", {}) or {}),
        features=dict(data.get("features", {}) or {}),
        config_path=config_path,
    )


__all__ = ["BudgetLimits", "DEFAULT_CONFIG_PATH", "PrepConfig", "WindowPolicy", "load_prep_config"]
