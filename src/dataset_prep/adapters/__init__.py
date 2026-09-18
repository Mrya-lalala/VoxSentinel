"""Source adapters for the bounded preparation pilot.

Accessible sources have working ``fetch_*`` functions.  Gated or bounded
sources are documented in :mod:`.catalog`; no adapter ever accepts terms,
handles tokens or downloads archives beyond the configured budget.
"""

from __future__ import annotations

from typing import Callable

from ..budget import DownloadLedger
from ..config import PrepConfig
from .asvspoof2019 import fetch_asvspoof2019
from .catalog import SourceEntry, source_catalog
from .common import SourceRun
from .indicsynth import fetch_indicsynth
from .nisp import fetch_nisp
from .nptel import fetch_nptel
from .svarah import fetch_svarah
from .synthetic_english import import_synthetic_english

FETCHERS: dict[str, Callable[..., SourceRun]] = {
    "indicsynth": fetch_indicsynth,
    "nisp": fetch_nisp,
    "asvspoof2019": fetch_asvspoof2019,
    "nptel": fetch_nptel,
    "svarah": fetch_svarah,
    "synthetic_english": import_synthetic_english,
}

__all__ = ["FETCHERS", "SourceEntry", "SourceRun", "source_catalog"]
