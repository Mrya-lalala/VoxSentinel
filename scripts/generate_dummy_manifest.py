"""Generate a small dummy ``manifest.csv`` for A1 deliverable checks.

This is a *fixture* generator only.  It writes a hand-written set of rows that
exercise the exact ``manifest.csv`` contract defined in
``src/data/dataset_setup.py``:

    sample_id, path, label, language, dataset_version

Pilot languages (hindi, bengali, marathi) are covered, with both label
conventions (``0`` = genuine via Kathbath, ``1`` = spoof via IndicSynth).

The rows are built with :class:`~src.data.dataset_setup.ManifestRow` and written
with :func:`~src.data.dataset_setup.write_manifest_csv`, so the fixture is
label-validated and column-ordered by the very same code that produces the real
manifest.  Run from the repository root:

    python scripts/generate_dummy_manifest.py
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_SETUP_PATH = REPO_ROOT / "src" / "data" / "dataset_setup.py"


def _load_dataset_setup() -> ModuleType:
    """Load ``dataset_setup.py`` directly, bypassing the torch-heavy package __init__."""
    spec = importlib.util.spec_from_file_location("dataset_setup", DATASET_SETUP_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"Cannot load dataset_setup from {DATASET_SETUP_PATH}.")
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves type annotations via sys.modules[cls.__module__].
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_dataset_setup = _load_dataset_setup()

INDICSYNTH = _dataset_setup.INDICSYNTH
KATHBATH = _dataset_setup.KATHBATH
LANGUAGE_CODES = _dataset_setup.LANGUAGE_CODES
MANIFEST_COLUMNS = _dataset_setup.MANIFEST_COLUMNS
PILOT_LANGUAGES = _dataset_setup.PILOT_LANGUAGES
ManifestRow = _dataset_setup.ManifestRow
write_manifest_csv = _dataset_setup.write_manifest_csv

#: Where the dummy manifest is written (per the A1 request).
DEFAULT_OUTPUT = REPO_ROOT / "src" / "data" / "manifest.csv"


def build_dummy_rows() -> list[ManifestRow]:
    """Return one genuine (label 0) and one spoof (label 1) row per pilot language."""
    rows: list[ManifestRow] = []
    for index, language in enumerate(PILOT_LANGUAGES, start=1):
        code = LANGUAGE_CODES[language]

        # Genuine sample from Kathbath (label 0).
        rows.append(
            ManifestRow(
                sample_id=f"{KATHBATH}_{code}_{index:06d}",
                path=f"data/{KATHBATH}/processed/{code}/{code}_{index:04d}.wav",
                label=0,
                language=language,
                dataset_version="kathbath-v1.0-pilot",
            )
        )

        # Spoof sample from IndicSynth (label 1).
        rows.append(
            ManifestRow(
                sample_id=f"{INDICSYNTH}_{code}_{index:06d}",
                path=f"data/{INDICSYNTH}/processed/{code}/{code}_{index:04d}_synth.wav",
                label=1,
                language=language,
                dataset_version="indicsynth-v1.0-pilot",
            )
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Destination manifest (default: {DEFAULT_OUTPUT}).",
    )
    args = parser.parse_args(argv)

    rows = build_dummy_rows()
    destination = write_manifest_csv(rows, args.output)

    print(f"Wrote {len(rows)} rows to {destination}")
    print("columns: " + ", ".join(MANIFEST_COLUMNS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())