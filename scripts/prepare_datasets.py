"""Entry point for the bounded dataset-preparation pilot.

Run from the repository root, e.g.:

    python -m scripts.prepare_datasets catalog
    python -m scripts.prepare_datasets fetch --source all
    python -m scripts.prepare_datasets split
    .venv/bin/python -m scripts.prepare_datasets features
    python -m scripts.prepare_datasets report

Download/inventory/materialization stages need the data environment
(``.venv-data``); the ``features`` stage needs the pinned encoder environment
(``.venv``).  See ``docs/archive/history/dataset-preparation-pilot.md``.
"""

from __future__ import annotations

from src.dataset_prep.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
