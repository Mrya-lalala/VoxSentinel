"""Dataset setup for VoxSentinel's three core corpora.

This module owns the on-disk *filesystem layout* and *metadata conventions* for
the corpora named in the 02-DATASET-TRAINING plan:

    * IndicSynth - synthetic (TTS/cloned) speech, i.e. spoof material.
    * Kathbath   - genuine (bona fide) Indic speech.
    * Svarah     - held-out evaluation data; never used for training.

Every adapter produced here must follow the project-wide label convention:

    class 0 = genuine (bona fide)
    class 1 = spoof

IndicSynth and Kathbath are the starting 2-3 pilot languages that seed the
training/dev splits.  Svarah is strictly held-out evaluation data and is
guarded by :func:`assert_training_allowed`.

The ``download_*`` and ``parse_*`` functions are deliberately inert starter
stubs: they validate their arguments and prepare directories, then raise
``NotImplementedError``.  No network access, licence acceptance or corpus
ingestion happens until a human explicitly authorises it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from .labels import GENUINE, SPOOF, validate_label

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_DATA_ROOT = Path("data")

#: The 2-3 pilot languages seed the initial train/dev pool.  This is a config
#: point: widen it once the pilot track is validated, but Svarah stays held out.
PILOT_LANGUAGES: tuple[str, ...] = ("hindi", "tamil", "bengali")

#: Standard sub-directories created for every corpus.
SUBDIRECTORIES: tuple[str, ...] = ("raw", "processed", "manifests")

INDICSYNTH = "indicsynth"
KATHBATH = "kathbath"
SVARAH = "svarah"

#: Corpora permitted to contribute to training/dev splits.
TRAINABLE_KEYS: tuple[str, ...] = (INDICSYNTH, KATHBATH)

#: Corpora that must never be trained on or tuned against.
EVALUATION_ONLY_KEYS: tuple[str, ...] = (SVARAH,)


@dataclass(frozen=True)
class DatasetSpec:
    """Static description of one corpus: role, label policy and layout."""

    key: str
    name: str
    role: str  # "synthetic" | "genuine" | "evaluation"
    default_label: int | None
    languages: tuple[str, ...]
    notes: str

    @property
    def is_evaluation_only(self) -> bool:
        return self.role == "evaluation"


DATASETS: dict[str, DatasetSpec] = {
    INDICSYNTH: DatasetSpec(
        key=INDICSYNTH,
        name="IndicSynth",
        role="synthetic",
        default_label=SPOOF,
        languages=PILOT_LANGUAGES,
        notes=(
            "Synthetic Indic speech. Primary label is 1 (spoof); any genuine "
            "reference audio must come from Kathbath, not from IndicSynth."
        ),
    ),
    KATHBATH: DatasetSpec(
        key=KATHBATH,
        name="Kathbath",
        role="genuine",
        default_label=GENUINE,
        languages=PILOT_LANGUAGES,
        notes=(
            "Genuine Indic speech. Primary label is 0 (genuine); provide the "
            "source audio that IndicSynth spoofs are conditioned against."
        ),
    ),
    SVARAH: DatasetSpec(
        key=SVARAH,
        name="Svarah",
        role="evaluation",
        default_label=None,
        languages=PILOT_LANGUAGES,
        notes=(
            "Held-out evaluation only. Contains mixed genuine/spoof labels; "
            "never trained on, never used for model selection or thresholds."
        ),
    ),
}


# --------------------------------------------------------------------------- #
# Manifest records
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ManifestRecord:
    """One row of a dataset manifest, ready to be serialised to JSONL."""

    path: str
    label: int
    dataset: str
    language: str
    split: str
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", validate_label(self.label))
        if not self.path:
            raise ValueError("ManifestRecord.path must be a non-empty string.")
        if not self.dataset:
            raise ValueError("ManifestRecord.dataset must be a non-empty string.")
        if not self.language:
            raise ValueError("ManifestRecord.language must be a non-empty string.")
        if not self.split:
            raise ValueError("ManifestRecord.split must be a non-empty string.")


def write_manifest(records: Iterable[ManifestRecord], destination: str | Path) -> Path:
    """Write ``records`` as JSONL to ``destination`` and return the path."""
    rows = list(records)
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in rows:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return output


def iter_manifest(path: str | Path) -> Iterator[ManifestRecord]:
    """Yield validated :class:`ManifestRecord` objects from a JSONL manifest."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as error:  # pragma: no cover - defensive
                raise ValueError(f"{path}:{line_number}: invalid JSON: {error}") from error
            yield ManifestRecord(**payload)


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #


def dataset_root(root: str | Path, key: str) -> Path:
    """Return the directory that holds one corpus under ``root``."""
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key {key!r}; expected one of {sorted(DATASETS)}.")
    return Path(root) / key


def setup_dataset_directories(root: str | Path = DEFAULT_DATA_ROOT) -> dict[str, Path]:
    """Create the folder structure for all three corpora and return the paths.

    Each corpus gets ``raw/``, ``processed/`` and ``manifests/`` sub-directories.
    The function is idempotent: existing directories are left untouched.
    """
    base = Path(root)
    created: dict[str, Path] = {}
    for key in DATASETS:
        corpus = dataset_root(base, key)
        for name in SUBDIRECTORIES:
            (corpus / name).mkdir(parents=True, exist_ok=True)
        created[key] = corpus
    return created


def assert_training_allowed(key: str) -> None:
    """Guard against accidental use of held-out evaluation corpora."""
    if key in EVALUATION_ONLY_KEYS:
        raise ValueError(
            f"{DATASETS[key].name} is strictly held-out evaluation data and must not be "
            "used for training, tuning or threshold selection."
        )


def training_specs() -> list[DatasetSpec]:
    """Return the corpora allowed to contribute to training/dev splits."""
    return [DATASETS[key] for key in TRAINABLE_KEYS]


def evaluation_specs() -> list[DatasetSpec]:
    """Return the held-out evaluation corpora."""
    return [DATASETS[key] for key in EVALUATION_ONLY_KEYS]


def _require_empty_target(destination: Path, *, force: bool) -> None:
    """Refuse to overwrite a populated destination unless ``force`` is set."""
    if destination.exists() and any(destination.iterdir()) and not force:
        raise FileExistsError(
            f"{destination} is not empty; pass force=True to overwrite an existing download."
        )


def _resolve_languages(languages: Sequence[str] | None) -> tuple[str, ...]:
    """Validate/normalise a requested language selection."""
    if languages is None:
        return PILOT_LANGUAGES
    requested = tuple(languages)
    if not requested:
        raise ValueError("languages must be a non-empty selection when provided.")
    unknown = sorted(set(requested) - set(PILOT_LANGUAGES))
    if unknown:
        raise ValueError(
            f"Unsupported pilot language(s) {unknown}; allowed: {sorted(PILOT_LANGUAGES)}."
        )
    return requested


# --------------------------------------------------------------------------- #
# Starter adapters (intentionally inert; authorise before implementing)
# --------------------------------------------------------------------------- #


def download_indicsynth(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    force: bool = False,
) -> Path:
    """Download IndicSynth synthetic speech into ``root/indicsynth/raw``.

    Every produced item carries label 1 (spoof).  Not yet implemented: requires
    a licence decision and network access before any bytes are fetched.
    """
    selected = _resolve_languages(languages)
    destination = dataset_root(root, INDICSYNTH) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    _require_empty_target(destination, force=force)
    raise NotImplementedError(
        f"IndicSynth download is not implemented yet (languages={selected}). "
        "Implement the corpus-specific fetcher once access is authorised."
    )


def download_kathbath(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    force: bool = False,
) -> Path:
    """Download Kathbath genuine speech into ``root/kathbath/raw``.

    Every produced item carries label 0 (genuine).  Not yet implemented: the
    real fetcher must be added after its licence/access terms are confirmed.
    """
    selected = _resolve_languages(languages)
    destination = dataset_root(root, KATHBATH) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    _require_empty_target(destination, force=force)
    raise NotImplementedError(
        f"Kathbath download is not implemented yet (languages={selected}). "
        "Implement the corpus-specific fetcher once access is authorised."
    )


def download_svarah(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    force: bool = False,
) -> Path:
    """Download Svarah (held-out evaluation) into ``root/svarah/raw``.

    Not yet implemented.  This corpus is evaluation-only; see
    :func:`assert_training_allowed`.
    """
    destination = dataset_root(root, SVARAH) / "raw"
    destination.mkdir(parents=True, exist_ok=True)
    _require_empty_target(destination, force=force)
    raise NotImplementedError(
        "Svarah download is not implemented yet. Implement the corpus-specific "
        "fetcher once access is authorised."
    )


def _parse_corpus(
    key: str,
    *,
    root: str | Path,
    manifest_path: str | Path | None,
    default_manifest_name: str,
    languages: Sequence[str] | None,
) -> Path:
    """Shared scaffolding for the per-corpus ``parse_*`` starter functions."""
    spec = DATASETS[key]
    raw_dir = dataset_root(root, key) / "raw"
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist; run download_{key}() or place the corpus there first."
        )
    output = (
        Path(manifest_path)
        if manifest_path is not None
        else dataset_root(root, key) / "manifests" / default_manifest_name
    )
    selected = _resolve_languages(languages)
    raise NotImplementedError(
        f"{spec.name} parsing is not implemented yet (languages={selected}). "
        f"Emit ManifestRecord rows to {output} using label {spec.default_label} "
        "and the 0=genuine / 1=spoof convention."
    )


def parse_indicsynth(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
) -> Path:
    """Parse raw IndicSynth audio into a JSONL manifest (label 1, spoof)."""
    return _parse_corpus(
        INDICSYNTH,
        root=root,
        manifest_path=manifest_path,
        default_manifest_name="indicsynth.jsonl",
        languages=languages,
    )


def parse_kathbath(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
) -> Path:
    """Parse raw Kathbath audio into a JSONL manifest (label 0, genuine)."""
    return _parse_corpus(
        KATHBATH,
        root=root,
        manifest_path=manifest_path,
        default_manifest_name="kathbath.jsonl",
        languages=languages,
    )


def parse_svarah(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Parse raw Svarah audio into an evaluation-only JSONL manifest.

    Labels are read from the corpus (mixed genuine/spoof); no default label is
    applied.  This corpus must never feed a training split.
    """
    return _parse_corpus(
        SVARAH,
        root=root,
        manifest_path=manifest_path,
        default_manifest_name="svarah.jsonl",
        languages=None,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main(argv: Sequence[str] | None = None) -> int:
    """Create the dataset folder layout, then print a short summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Dataset root directory.")
    args = parser.parse_args(argv)

    created = setup_dataset_directories(args.root)
    trainable = ", ".join(spec.name for spec in training_specs())
    evaluation = ", ".join(spec.name for spec in evaluation_specs())
    print(f"Dataset root: {Path(args.root)}")
    for key, path in created.items():
        print(f"  {DATASETS[key].name:<11} ({DATASETS[key].role:<10}) -> {path}")
    print(f"Training/dev corpora: {trainable}")
    print(f"Held-out evaluation:  {evaluation}")
    return 0


__all__ = [
    "DATASETS",
    "DEFAULT_DATA_ROOT",
    "EVALUATION_ONLY_KEYS",
    "INDICSYNTH",
    "KATHBATH",
    "ManifestRecord",
    "PILOT_LANGUAGES",
    "SUBDIRECTORIES",
    "SVARAH",
    "TRAINABLE_KEYS",
    "DatasetSpec",
    "assert_training_allowed",
    "dataset_root",
    "download_indicsynth",
    "download_kathbath",
    "download_svarah",
    "evaluation_specs",
    "iter_manifest",
    "main",
    "parse_indicsynth",
    "parse_kathbath",
    "parse_svarah",
    "setup_dataset_directories",
    "training_specs",
    "write_manifest",
]


if __name__ == "__main__":
    raise SystemExit(main())