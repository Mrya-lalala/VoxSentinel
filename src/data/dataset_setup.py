"""Dataset setup for VoxSentinel's three core corpora.

This module owns the on-disk *filesystem layout*, *acquisition* and *metadata
conventions* for the corpora named in the 02-DATASET-TRAINING plan:

    * IndicSynth - synthetic (TTS/cloned) speech, i.e. spoof material.
    * Kathbath   - genuine (bona fide) Indic speech.
    * Svarah     - held-out evaluation data; never used for training.

Every adapter produced here must follow the project-wide label convention:

    class 0 = genuine (bona fide)
    class 1 = spoof

IndicSynth and Kathbath are the starting 2-3 pilot languages that seed the
training/dev splits.  Svarah is strictly held-out evaluation data and is
guarded by :func:`assert_training_allowed`.

The acquisition engine (URL streaming, SHA-256 verification and safe archive
extraction) uses the Python standard library only, so dataset preparation can
run without the heavy model dependencies.  The *exact* upstream locations live
in the verified-accessible list from 02-DATASET-TRAINING: supply them via a
source file (``--source-file``, default ``configs/dataset_sources.json``) or the
``VOXSENTINEL_DATASET_SOURCES`` environment variable.  No URL is invented here;
:func:`select_sources` fails loudly if a pilot language has no configured URL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import numbers
import os
import sys
import tarfile
import urllib.error
import urllib.request
import zipfile
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Callable, Iterable, Iterator, Mapping, Sequence

try:  # Preferred: the shared source of truth when the full package is importable.
    from .labels import GENUINE, SPOOF, validate_label
except ImportError:  # stdlib-only run, e.g. `python src/data/dataset_setup.py`.
    GENUINE = 0
    SPOOF = 1

    def validate_label(value: object) -> int:
        """Return the label as ``int`` or raise for anything that is not 0 or 1."""
        if isinstance(value, bool):
            raise ValueError("labels must be the integers 0 (genuine) or 1 (spoof), not booleans.")
        if isinstance(value, numbers.Integral):
            label = int(value)
        elif isinstance(value, numbers.Real):
            number = float(value)
            if not math.isfinite(number) or not number.is_integer():
                raise ValueError(f"labels must be binary (0=genuine, 1=spoof) and integral; got {value!r}.")
            label = int(number)
        else:
            raise ValueError(f"labels must be binary (0=genuine, 1=spoof); got {value!r}.")
        if label not in (GENUINE, SPOOF):
            raise ValueError(f"labels must be binary (0=genuine, 1=spoof); got {label}.")
        return label


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

DEFAULT_DATA_ROOT = Path("data")

#: The 2-3 pilot languages seed the initial train/dev pool.  Selected from the
#: verified-accessible list in 02-DATASET-TRAINING.  Widen once pilots validate.
PILOT_LANGUAGES: tuple[str, ...] = ("hindi", "bengali", "marathi")

#: Short language codes used in paths, ids and future manifests.
LANGUAGE_CODES: dict[str, str] = {"hindi": "hi", "bengali": "bn", "marathi": "mr"}

#: Standard sub-directories created for every corpus.
SUBDIRECTORIES: tuple[str, ...] = ("raw", "processed", "manifests")

#: Audio file extensions recognised during discovery.
AUDIO_EXTENSIONS: tuple[str, ...] = (".wav", ".flac", ".mp3", ".ogg", ".opus", ".m4a", ".aac")

INDICSYNTH = "indicsynth"
KATHBATH = "kathbath"
SVARAH = "svarah"

#: Corpora permitted to contribute to training/dev splits.
TRAINABLE_KEYS: tuple[str, ...] = (INDICSYNTH, KATHBATH)

#: Corpora that must never be trained on or tuned against.
EVALUATION_ONLY_KEYS: tuple[str, ...] = (SVARAH,)

#: Dataset version stamped into every manifest row (override per run).
DEFAULT_VERSIONS: dict[str, str] = {
    INDICSYNTH: "indicsynth-v1.0-pilot",
    KATHBATH: "kathbath-v1.0-pilot",
    SVARAH: "svarah-v1.0-eval",
}

#: A single combined manifest; column order is fixed by the A1 interface.
MANIFEST_FILENAME = "manifest.csv"
MANIFEST_COLUMNS: tuple[str, ...] = ("sample_id", "path", "label", "language", "dataset_version")

DEFAULT_SOURCE_FILE = Path("configs/dataset_sources.json")
SOURCE_FILE_ENV = "VOXSENTINEL_DATASET_SOURCES"
DEFAULT_USER_AGENT = "VoxSentinel-A1-dataset-setup/1.0"
DEFAULT_CHUNK_SIZE = 1 << 20  # 1 MiB


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
# Source registry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class DatasetSource:
    """One downloadable subset (per corpus + pilot language) to acquire.

    ``url`` may be an ``http(s)://`` location, a ``file://`` URI or a local
    filesystem path (handy for offline runs and tests).  ``sha256`` is optional
    but strongly recommended.  ``archive`` selects the extraction strategy and is
    inferred from the filename suffix when omitted.
    """

    dataset: str
    language: str
    url: str | None
    version: str
    sha256: str | None = None
    archive: str | None = None  # "tar" | "zip" | "none" | None (infer)
    notes: str = ""

    def __post_init__(self) -> None:
        if self.dataset not in DATASETS:
            raise ValueError(f"source.dataset must be one of {sorted(DATASETS)}; got {self.dataset!r}.")
        if self.language not in LANGUAGE_CODES:
            raise ValueError(f"source.language must be a pilot language {sorted(LANGUAGE_CODES)}; got {self.language!r}.")
        if not self.version:
            raise ValueError("source.version must be a non-empty string.")

    @property
    def filename(self) -> str:
        """A safe local filename for this source's payload."""
        if self.url:
            tail = self.url.split("?", 1)[0].rstrip("/").split("/")[-1]
            if tail:
                return tail
        return f"{self.dataset}-{LANGUAGE_CODES[self.language]}.tar.gz"


def _source_from_mapping(entry: Mapping[str, object]) -> DatasetSource:
    """Build a :class:`DatasetSource` from a JSON-style mapping, ignoring extras."""
    known = {field.name for field in fields(DatasetSource)}
    payload = {key: value for key, value in entry.items() if key in known}
    return DatasetSource(**payload)  # type: ignore[arg-type]


def load_sources(path: str | Path) -> list[DatasetSource]:
    """Load sources from a JSON file (a list, or an object with ``"sources"``)."""
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    entries = data.get("sources", []) if isinstance(data, dict) else data
    if not isinstance(entries, list):
        raise ValueError(f"{path}: expected a list of sources or an object with a 'sources' list.")
    return [_source_from_mapping(entry) for entry in entries]


def select_sources(
    dataset_key: str,
    languages: Sequence[str],
    *,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
) -> list[DatasetSource]:
    """Return configured sources for ``dataset_key`` and the chosen languages.

    Raises a clear, actionable error when a requested language has no configured
    URL, because the verified-accessible locations are owned by the master doc.
    """
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}; expected one of {sorted(DATASETS)}.")

    env_path = os.environ.get(SOURCE_FILE_ENV)
    candidate = Path(env_path) if env_path else Path(source_file)
    available: list[DatasetSource] = []
    if candidate.exists():
        available = load_sources(candidate)
    else:
        raise FileNotFoundError(
            f"No dataset source file found at {candidate}. Create it (or set {SOURCE_FILE_ENV}) "
            "with the verified IndicSynth/Kathbath subset URLs from 02-DATASET-TRAINING."
        )

    selected: list[DatasetSource] = []
    for language in languages:
        matches = [s for s in available if s.dataset == dataset_key and s.language == language]
        if not matches:
            raise ValueError(
                f"No source configured for dataset={dataset_key!r} language={language!r} in {candidate}. "
                "Add an entry with the verified accessible URL before downloading."
            )
        for source in matches:
            if not source.url:
                raise ValueError(
                    f"Source for {dataset_key}/{language} in {candidate} has no URL. "
                    "Populate the verified accessible URL from 02-DATASET-TRAINING."
                )
        selected.extend(matches)
    return selected


def sources_template() -> dict[str, object]:
    """Return a JSON-serialisable template pre-filled for the pilot languages."""
    return {
        "_comment": (
            "Fill each 'url' with the verified accessible subset URL from "
            "02-DATASET-TRAINING. 'sha256' is optional but recommended. 'archive' "
            "is inferred from the URL suffix (tar/zip) when omitted."
        ),
        "sources": [
            {
                "dataset": dataset,
                "language": language,
                "url": "",
                "version": DEFAULT_VERSIONS[dataset],
                "sha256": None,
                "notes": f"{DATASETS[dataset].name} {language} pilot subset",
            }
            for dataset in TRAINABLE_KEYS
            for language in PILOT_LANGUAGES
        ],
    }


# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #


def dataset_root(root: str | Path, key: str) -> Path:
    """Return the directory that holds one corpus under ``root``."""
    if key not in DATASETS:
        raise ValueError(f"Unknown dataset key {key!r}; expected one of {sorted(DATASETS)}.")
    return Path(root) / key


def processed_dir(root: str | Path, key: str, language: str) -> Path:
    """Return the extracted-audio directory for one corpus + language."""
    if language not in LANGUAGE_CODES:
        raise ValueError(f"language must be a pilot language {sorted(LANGUAGE_CODES)}; got {language!r}.")
    return dataset_root(root, key) / "processed" / LANGUAGE_CODES[language]


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


def resolve_version(dataset_key: str, versions: Mapping[str, str] | None = None) -> str:
    """Return the dataset version stamp for ``dataset_key``."""
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown dataset key {dataset_key!r}.")
    if versions and dataset_key in versions:
        return str(versions[dataset_key])
    return DEFAULT_VERSIONS[dataset_key]


def _resolve_languages(languages: Sequence[str] | None) -> tuple[str, ...]:
    """Validate/normalise a requested language selection."""
    if languages is None:
        return PILOT_LANGUAGES
    requested = tuple(languages)
    if not requested:
        raise ValueError("languages must be a non-empty selection when provided.")
    unknown = sorted(set(requested) - set(LANGUAGE_CODES))
    if unknown:
        raise ValueError(f"Unsupported pilot language(s) {unknown}; allowed: {sorted(LANGUAGE_CODES)}.")
    return requested


# --------------------------------------------------------------------------- #
# Acquisition engine (standard library only)
# --------------------------------------------------------------------------- #


def sha256_file(path: str | Path, *, chunk_size: int = DEFAULT_CHUNK_SIZE) -> str:
    """Return the hex SHA-256 digest of ``path``."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _to_url(value: str) -> str:
    """Return an openable URL; local paths become ``file://`` URIs."""
    if "://" in value:
        return value
    return Path(value).expanduser().resolve().as_uri()


def download_file(
    url: str,
    destination: str | Path,
    *,
    expected_sha256: str | None = None,
    force: bool = False,
    timeout: float = 60.0,
    user_agent: str = DEFAULT_USER_AGENT,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Stream ``url`` to ``destination``, verifying an optional SHA-256 digest.

    A pre-existing destination is reused when it already matches the expected
    digest (or when no digest is given and ``force`` is False).  Downloads land
    in a ``.part`` file first so an interrupted run never leaves a partial file
    looking complete.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _message: None)

    if destination.exists() and not force:
        if expected_sha256 is None or sha256_file(destination) == expected_sha256.lower():
            log(f"cached   {destination}")
            return destination

    if destination.exists() and force:
        destination.unlink()

    request = urllib.request.Request(_to_url(url), headers={"User-Agent": user_agent})
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    log(f"download {url} -> {destination}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as handle:
            total_header = response.headers.get("Content-Length") if response.headers else None
            total = int(total_header) if total_header else None
            written = 0
            last_report = 0
            while chunk := response.read(chunk_size):
                handle.write(chunk)
                digest.update(chunk)
                written += len(chunk)
                if total and written - last_report >= 50 * chunk_size:
                    last_report = written
                    log(f"         {written / total:.0%} of {total} bytes")
    except urllib.error.URLError as error:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"failed to download {url}: {error}") from error

    actual = digest.hexdigest()
    if expected_sha256 and actual != expected_sha256.lower():
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"SHA-256 mismatch for {url}: expected {expected_sha256.lower()}, got {actual}."
        )
    partial.replace(destination)
    log(f"verified {destination} (sha256={actual[:12]}...)")
    return destination


def _is_within(base: Path, target: Path) -> bool:
    """True when ``target`` is ``base`` or lives underneath it (no traversal)."""
    base = base.resolve()
    target = target.resolve()
    return base == target or base in target.parents


def _archive_kind(path: Path) -> str:
    """Infer the archive strategy from a filename suffix."""
    name = path.name.lower()
    if name.endswith((".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tbz2", ".tar.xz", ".txz")):
        return "tar"
    if name.endswith(".zip"):
        return "zip"
    return "none"


def extract_archive(
    archive: str | Path,
    destination: str | Path,
    *,
    kind: str | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Safely extract ``archive`` (tar/zip) or place a bare audio file.

    Member paths are resolved and rejected if they escape ``destination`` so a
    malicious archive cannot write outside the target directory.
    """
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    log = progress or (lambda _message: None)
    strategy = kind or _archive_kind(archive)

    if strategy == "none":
        target = destination / archive.name
        if target.exists() and not force:
            log(f"present  {target}")
            return target
        data = archive.read_bytes()
        target.write_bytes(data)
        log(f"placed   {target}")
        return target

    if strategy == "zip":
        with zipfile.ZipFile(archive) as bundle:
            for name in bundle.namelist():
                if not _is_within(destination, destination / name):
                    raise ValueError(f"Refusing unsafe archive member {name!r} in {archive}.")
            bundle.extractall(destination)
        log(f"extracted {archive} -> {destination}")
        return destination

    if strategy == "tar":
        with tarfile.open(archive) as bundle:
            for member in bundle.getmembers():
                if not _is_within(destination, destination / member.name):
                    raise ValueError(f"Refusing unsafe archive member {member.name!r} in {archive}.")
            bundle.extractall(destination)
        log(f"extracted {archive} -> {destination}")
        return destination

    raise ValueError(f"Unsupported archive strategy {strategy!r} for {archive}.")


def discover_audio(
    directory: str | Path,
    *,
    extensions: Sequence[str] = AUDIO_EXTENSIONS,
) -> list[Path]:
    """Return sorted audio files under ``directory`` (recursive)."""
    base = Path(directory)
    if not base.exists():
        return []
    suffixes = tuple(ext.lower() for ext in extensions)
    return sorted(
        path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in suffixes
    )


# --------------------------------------------------------------------------- #
# Manifest (CSV) interface
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ManifestRow:
    """One ``manifest.csv`` row: the minimum information downstream needs."""

    sample_id: str
    path: str
    label: int
    language: str
    dataset_version: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", validate_label(self.label))
        for name in ("sample_id", "path", "language", "dataset_version"):
            if not getattr(self, name):
                raise ValueError(f"ManifestRow.{name} must be a non-empty string.")


def write_manifest_csv(rows: Iterable[ManifestRow], destination: str | Path) -> Path:
    """Write rows to ``destination`` as ``manifest.csv`` and return the path."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(MANIFEST_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "sample_id": row.sample_id,
                    "path": row.path,
                    "label": row.label,
                    "language": row.language,
                    "dataset_version": row.dataset_version,
                }
            )
    return output


def read_manifest_csv(path: str | Path) -> Iterator[ManifestRow]:
    """Yield validated :class:`ManifestRow` objects from a ``manifest.csv``."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: missing required columns {sorted(missing)}.")
        for row in reader:
            yield ManifestRow(
                sample_id=row["sample_id"],
                path=row["path"],
                label=int(row["label"]),
                language=row["language"],
                dataset_version=row["dataset_version"],
            )


def _relative_posix(path: Path, root: Path) -> str:
    """Return ``path`` relative to ``root`` as a POSIX string when possible."""
    try:
        return path.resolve().relative_to(Path(root).resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def build_manifest_rows(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    datasets: Sequence[str] = TRAINABLE_KEYS,
    languages: Sequence[str] | None = None,
    versions: Mapping[str, str] | None = None,
    limit: int | None = None,
) -> list[ManifestRow]:
    """Discover processed audio and build manifest rows (label per corpus).

    ``limit`` caps the number of audio files *per corpus per language*, keeping
    the pilot subset tractable.
    """
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive when provided.")
    base = Path(root)
    selected = _resolve_languages(languages)
    rows: list[ManifestRow] = []

    for key in datasets:
        if key not in DATASETS:
            raise ValueError(f"Unknown dataset key {key!r}; expected one of {sorted(DATASETS)}.")
        spec = DATASETS[key]
        if spec.default_label is None:
            raise ValueError(f"{spec.name} does not have a default label and cannot seed a manifest.")
        version = resolve_version(key, versions)
        for language in selected:
            files = discover_audio(processed_dir(base, key, language))
            if limit is not None:
                files = files[:limit]
            code = LANGUAGE_CODES[language]
            for index, path in enumerate(files, start=1):
                rows.append(
                    ManifestRow(
                        sample_id=f"{key}_{code}_{index:06d}",
                        path=_relative_posix(path, base),
                        label=spec.default_label,
                        language=language,
                        dataset_version=version,
                    )
                )
    return rows


def build_manifest(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    datasets: Sequence[str] = TRAINABLE_KEYS,
    languages: Sequence[str] | None = None,
    versions: Mapping[str, str] | None = None,
    limit: int | None = None,
    manifest_path: str | Path | None = None,
) -> Path:
    """Build the combined ``manifest.csv`` and return its path."""
    rows = build_manifest_rows(
        root, datasets=datasets, languages=languages, versions=versions, limit=limit
    )
    destination = Path(manifest_path) if manifest_path is not None else Path(root) / MANIFEST_FILENAME
    return write_manifest_csv(rows, destination)


# --------------------------------------------------------------------------- #
# Corpus acquisition
# --------------------------------------------------------------------------- #


def preview_corpus(
    dataset_key: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Log the planned download/extract work without touching network or disk.

    Tolerates an unconfigured source file so the plan can be previewed before the
    verified URLs are filled in.
    """
    assert_training_allowed(dataset_key)
    selected = _resolve_languages(languages)
    log = progress or (lambda message: print(message, file=sys.stderr))

    env_path = os.environ.get(SOURCE_FILE_ENV)
    candidate = Path(env_path) if env_path else Path(source_file)
    available: list[DatasetSource] = []
    if candidate.exists():
        available = load_sources(candidate)
    else:
        log(f"[dry-run] no source file at {candidate}; nothing configured yet")

    result: dict[str, Path] = {}
    for language in selected:
        extract_to = processed_dir(root, dataset_key, language)
        matches = [s for s in available if s.dataset == dataset_key and s.language == language]
        if not matches:
            log(f"[dry-run] {dataset_key}/{language}: (no source configured) -> {extract_to}")
        for source in matches:
            where = source.url or "(no URL configured)"
            log(f"[dry-run] {dataset_key}/{language}: {where} -> {extract_to}")
        result[language] = extract_to
    return result


def download_corpus(
    dataset_key: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
    user_agent: str = DEFAULT_USER_AGENT,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Download and extract one corpus for the pilot languages.

    Returns a mapping of language -> extracted audio directory.  ``dry_run``
    prints the planned work without touching the network or the filesystem.
    """
    selected = _resolve_languages(languages)
    if dry_run:
        return preview_corpus(
            dataset_key, root, languages=selected, source_file=source_file, progress=progress
        )

    assert_training_allowed(dataset_key)
    sources = select_sources(dataset_key, selected, source_file=source_file)
    log = progress or (lambda message: print(message, file=sys.stderr))
    raw_root = dataset_root(root, dataset_key) / "raw"

    result: dict[str, Path] = {}
    for language in selected:
        code = LANGUAGE_CODES[language]
        language_sources = [s for s in sources if s.language == language]
        extract_to = processed_dir(root, dataset_key, language)
        raw_root.mkdir(parents=True, exist_ok=True)
        extract_to.mkdir(parents=True, exist_ok=True)
        for source in language_sources:
            payload = download_file(
                source.url or "",
                raw_root / code / source.filename,
                expected_sha256=source.sha256,
                force=force,
                timeout=timeout,
                user_agent=user_agent,
                progress=log,
            )
            extract_archive(payload, extract_to, kind=source.archive, force=force, progress=log)
        result[language] = extract_to
    return result


def download_indicsynth(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Fetch and extract the IndicSynth pilot subset (label 1, spoof)."""
    return download_corpus(
        INDICSYNTH,
        root,
        languages=languages,
        source_file=source_file,
        force=force,
        dry_run=dry_run,
        timeout=timeout,
        progress=progress,
    )


def download_kathbath(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Path]:
    """Fetch and extract the Kathbath pilot subset (label 0, genuine)."""
    return download_corpus(
        KATHBATH,
        root,
        languages=languages,
        source_file=source_file,
        force=force,
        dry_run=dry_run,
        timeout=timeout,
        progress=progress,
    )


def download_svarah(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    force: bool = False,
) -> Path:
    """Refuse to acquire Svarah here; it is strictly held-out evaluation data."""
    assert_training_allowed(SVARAH)
    raise NotImplementedError(
        "Svarah is strictly held-out evaluation data and is intentionally not "
        "downloaded by the training-data setup script."
    )


def prepare_pilot_datasets(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    source_file: str | Path = DEFAULT_SOURCE_FILE,
    versions: Mapping[str, str] | None = None,
    limit: int | None = None,
    force: bool = False,
    dry_run: bool = False,
    timeout: float = 60.0,
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Create the layout, fetch IndicSynth + Kathbath pilots, and write manifest.csv."""
    setup_dataset_directories(root)
    download_indicsynth(
        root, languages=languages, source_file=source_file, force=force, dry_run=dry_run,
        timeout=timeout, progress=progress,
    )
    download_kathbath(
        root, languages=languages, source_file=source_file, force=force, dry_run=dry_run,
        timeout=timeout, progress=progress,
    )
    if dry_run:
        return Path(root) / MANIFEST_FILENAME
    return build_manifest(root, languages=languages, versions=versions, limit=limit)


# --------------------------------------------------------------------------- #
# JSONL manifest (legacy convenience)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ManifestRecord:
    """One JSONL manifest row (kept for downstream tooling compatibility)."""

    path: str
    label: int
    dataset: str
    language: str
    split: str
    speaker_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", validate_label(self.label))
        for name in ("path", "dataset", "language", "split"):
            if not getattr(self, name):
                raise ValueError(f"ManifestRecord.{name} must be a non-empty string.")


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


def parse_corpus(
    dataset_key: str,
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    languages: Sequence[str] | None = None,
    manifest_path: str | Path | None = None,
    split: str = "train",
) -> Path:
    """Discover extracted audio and emit a JSONL manifest for one corpus."""
    spec = DATASETS[dataset_key]
    if spec.default_label is None:
        raise ValueError(f"{spec.name} has no default label; its labels must come from the corpus.")
    selected = _resolve_languages(languages)
    base = Path(root)
    records: list[ManifestRecord] = []
    for language in selected:
        for path in discover_audio(processed_dir(base, dataset_key, language)):
            records.append(
                ManifestRecord(
                    path=_relative_posix(path, base),
                    label=spec.default_label,
                    dataset=dataset_key,
                    language=language,
                    split=split,
                )
            )
    destination = (
        Path(manifest_path)
        if manifest_path is not None
        else dataset_root(base, dataset_key) / "manifests" / f"{dataset_key}.jsonl"
    )
    return write_manifest(records, destination)


def parse_indicsynth(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
) -> Path:
    """Parse extracted IndicSynth audio into a JSONL manifest (label 1, spoof)."""
    return parse_corpus(INDICSYNTH, root, languages=languages, manifest_path=manifest_path)


def parse_kathbath(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
    languages: Sequence[str] | None = None,
) -> Path:
    """Parse extracted Kathbath audio into a JSONL manifest (label 0, genuine)."""
    return parse_corpus(KATHBATH, root, languages=languages, manifest_path=manifest_path)


def parse_svarah(
    root: str | Path = DEFAULT_DATA_ROOT,
    *,
    manifest_path: str | Path | None = None,
) -> Path:
    """Refuse to parse Svarah here; it is held out from training tooling."""
    raise NotImplementedError(
        "Svarah is strictly held-out evaluation data; its manifest is produced "
        "separately and must never feed a training split."
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", default=str(DEFAULT_DATA_ROOT), help="Dataset root directory.")
    parser.add_argument(
        "--languages", nargs="+", choices=sorted(LANGUAGE_CODES), default=None,
        help="Pilot languages to operate on (default: all).",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dataset_setup", description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    layout = subparsers.add_parser("layout", help="Create the dataset folder structure.")
    _add_common_arguments(layout)

    download = subparsers.add_parser("download", help="Fetch and extract pilot subsets.")
    _add_common_arguments(download)
    download.add_argument("--source-file", default=str(DEFAULT_SOURCE_FILE))
    download.add_argument("--force", action="store_true", help="Re-download/re-extract.")
    download.add_argument("--dry-run", action="store_true", help="Print the plan only.")

    manifest = subparsers.add_parser("manifest", help="Write the combined manifest.csv.")
    _add_common_arguments(manifest)
    manifest.add_argument("--limit", type=int, default=None, help="Cap files per corpus/language.")
    manifest.add_argument("--output", default=None, help="Manifest path (default: <root>/manifest.csv).")

    everything = subparsers.add_parser("all", help="layout + download + manifest.")
    _add_common_arguments(everything)
    everything.add_argument("--source-file", default=str(DEFAULT_SOURCE_FILE))
    everything.add_argument("--force", action="store_true")
    everything.add_argument("--dry-run", action="store_true")
    everything.add_argument("--limit", type=int, default=None)

    sources = subparsers.add_parser("sources-template", help="Print a source-file template.")
    sources.add_argument("--output", default=None, help="Write template here instead of stdout.")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for ``python src/data/dataset_setup.py <command>``."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    command = args.command or "layout"

    if command == "layout":
        created = setup_dataset_directories(args.root)
        print(f"Dataset root: {Path(args.root)}")
        for key, path in created.items():
            print(f"  {DATASETS[key].name:<11} ({DATASETS[key].role:<10}) -> {path}")
        print(f"Training/dev corpora: {', '.join(s.name for s in training_specs())}")
        print(f"Held-out evaluation:  {', '.join(s.name for s in evaluation_specs())}")
        return 0

    if command == "sources-template":
        payload = json.dumps(sources_template(), indent=2) + "\n"
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(payload, encoding="utf-8")
            print(f"Wrote source template to {out}")
        else:
            print(payload, end="")
        return 0

    if command == "download":
        setup_dataset_directories(args.root)
        for key in TRAINABLE_KEYS:
            download_corpus(
                key, args.root, languages=args.languages, source_file=args.source_file,
                force=args.force, dry_run=args.dry_run,
            )
        print("Download complete." if not args.dry_run else "Dry run complete.")
        return 0

    if command == "manifest":
        output = build_manifest(
            args.root, languages=args.languages, limit=args.limit, manifest_path=args.output
        )
        print(f"Wrote {output}")
        return 0

    if command == "all":
        output = prepare_pilot_datasets(
            args.root, languages=args.languages, source_file=args.source_file,
            limit=args.limit, force=args.force, dry_run=args.dry_run,
        )
        print(f"Wrote {output}" if not args.dry_run else "Dry run complete.")
        return 0

    parser.error(f"unknown command {command!r}")  # pragma: no cover - argparse guards this
    return 2


__all__ = [
    "AUDIO_EXTENSIONS",
    "DATASETS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_SOURCE_FILE",
    "DEFAULT_VERSIONS",
    "EVALUATION_ONLY_KEYS",
    "INDICSYNTH",
    "KATHBATH",
    "LANGUAGE_CODES",
    "MANIFEST_COLUMNS",
    "MANIFEST_FILENAME",
    "PILOT_LANGUAGES",
    "SOURCE_FILE_ENV",
    "SUBDIRECTORIES",
    "SVARAH",
    "TRAINABLE_KEYS",
    "DatasetSource",
    "DatasetSpec",
    "ManifestRecord",
    "ManifestRow",
    "assert_training_allowed",
    "build_manifest",
    "build_manifest_rows",
    "dataset_root",
    "discover_audio",
    "download_corpus",
    "download_file",
    "download_indicsynth",
    "download_kathbath",
    "download_svarah",
    "evaluation_specs",
    "extract_archive",
    "iter_manifest",
    "load_sources",
    "main",
    "parse_corpus",
    "parse_indicsynth",
    "parse_kathbath",
    "parse_svarah",
    "prepare_pilot_datasets",
    "preview_corpus",
    "processed_dir",
    "read_manifest_csv",
    "resolve_version",
    "select_sources",
    "setup_dataset_directories",
    "sha256_file",
    "sources_template",
    "training_specs",
    "write_manifest",
    "write_manifest_csv",
]


if __name__ == "__main__":
    raise SystemExit(main())