"""Dataset-version contract for the expanded v3 coverage dataset.

This module owns the *pure* (no-torch) side of dataset-aware path handling:

* loading and validating ``dataset-version.v3.json`` (schema, declared
  manifest/retained/benchmark hashes, counts, declared split assignments);
* loading split rows in deterministic manifest order with per-row validation;
* resolving each row's ``prepared_audio.path`` through the declared
  ``path_base`` rule - one explicit resolver, no silent fallback to other
  roots.

Split-name normalization is intentional: the v3 **dev** manifest's rows carry
the historical raw value ``"val"`` (retained v2 development rows and the v3
additions alike), while the cache/training split name is ``"dev"``.  Every
row is validated against the allowed raw values of its manifest, and the
loader exposes the normalized split name for cache/training consumers.

Nothing here reads audio, features or the benchmark rows' audio - the
benchmark manifest and exposure hashes are verified as declared provenance
only.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

SCHEMA_V3 = "voxsentinel.dataset_version.v3-coverage.v1"
SPLIT_NAMES = ("train", "dev")
PROTOCOL = "speaker_recording_disjoint_v2"
PREPROCESSING_VERSION = "voxsentinel-prep-2"

# Raw row ``split`` values accepted per manifest, and the normalized name the
# cache/training layer uses.  ``val`` is the historical development value.
_ROW_SPLITS = {"train": frozenset({"train"}), "dev": frozenset({"dev", "val"})}


class DatasetVersionError(ValueError):
    """The version record, a manifest, or a manifest row violates the contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetVersionError(f"{path}:{line_number}: invalid JSON: {error}") from error
        if not isinstance(row, dict):
            raise DatasetVersionError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


@dataclass(frozen=True)
class SplitSpec:
    """One split of the version record, already checked against its manifest."""

    name: str
    manifest_path: Path
    declared_sha256: str
    windows: int
    retained: int
    additions: int

    @property
    def row_splits(self) -> frozenset[str]:
        return _ROW_SPLITS[self.name]


@dataclass(frozen=True)
class DatasetVersion:
    """Validated ``dataset-version.v3.json`` plus the paths it declares."""

    path: Path
    sha256: str
    spec: Mapping[str, Any]
    protocol: str
    strict_conversion_family_compliant: bool
    default_base: str
    extra_bases: frozenset[str]
    splits: Mapping[str, SplitSpec]
    benchmark: Mapping[str, Any]
    benchmark_manifest_path: Path
    benchmark_exposure_path: Path

    @property
    def bases(self) -> frozenset[str]:
        return frozenset({self.default_base}) | self.extra_bases

    def split(self, name: str) -> SplitSpec:
        if name not in self.splits:
            raise DatasetVersionError(f"unknown split {name!r}; expected one of {SPLIT_NAMES}")
        return self.splits[name]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DatasetVersionError(message)


def _check_declared_file(entry: Mapping[str, Any] | None, *, label: str, root: Path) -> tuple[Path, str]:
    _require(isinstance(entry, dict), f"{label}: missing declared file entry")
    relative = Path(str(entry.get("path", "")))
    declared = str(entry.get("sha256", ""))
    _require(bool(relative.name), f"{label}: empty path")
    _require(not relative.is_absolute() and ".." not in relative.parts,
             f"{label}: declared path must be repo-relative and safe: {relative}")
    _require(bool(declared) and len(declared) == 64, f"{label}: invalid declared sha256")
    path = root / relative
    _require(path.is_file(), f"{label}: declared file does not exist: {path}")
    actual = sha256_file(path)
    _require(actual == declared, f"{label}: file sha256 {actual} != declared {declared}")
    return path, actual


def load_dataset_version(path: str | Path, *, repo_root: str | Path | None = None) -> DatasetVersion:
    """Load and fully validate a v3 dataset-version record.

    All declared files (manifests, retained sources, benchmark manifest and
    exposure listing) are checked against their declared SHA-256 values, and
    every manifest is parsed and counted.  A record that does not verify is
    rejected - never partially trusted.
    """
    version_path = Path(path)
    _require(version_path.is_file(), f"dataset version file not found: {version_path}")
    root = Path(repo_root if repo_root is not None else ".").resolve()
    try:
        spec = json.loads(version_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise DatasetVersionError(f"{version_path}: invalid JSON: {error}") from error
    _require(isinstance(spec, dict), f"{version_path}: version record must be an object")
    _require(spec.get("schema") == SCHEMA_V3, f"unsupported dataset version schema: {spec.get('schema')!r}")
    _require(spec.get("protocol") == PROTOCOL, f"unexpected protocol: {spec.get('protocol')!r}")

    resolution = spec.get("path_resolution") or {}
    default_base = str(resolution.get("default_base", ""))
    _require(bool(default_base), "path_resolution.default_base is required")
    bases_block = resolution.get("bases") or {}
    _require(isinstance(bases_block, dict), "path_resolution.bases must be an object")
    extra_bases = frozenset(str(base) for base in bases_block)
    for base in {default_base} | extra_bases:
        _require(not Path(base).is_absolute(), f"path base must be repo-relative: {base}")
        _require(".." not in Path(base).parts, f"path base must not contain '..': {base}")

    manifests = spec.get("manifests") or {}
    splits: dict[str, SplitSpec] = {}
    for name in SPLIT_NAMES:
        entry = manifests.get(name)
        _require(isinstance(entry, dict), f"manifests.{name} is missing")
        manifest_path, declared = _check_declared_file(entry, label=f"manifests.{name}", root=root)
        rows = _read_jsonl(manifest_path)
        windows = int(entry.get("windows", -1))
        retained = int(entry.get("retained", -1))
        additions = int(entry.get("additions", -1))
        _require(len(rows) == windows, f"manifests.{name}: {len(rows)} rows != declared {windows}")
        _require(retained >= 0 and additions >= 0 and retained + additions == windows,
                 f"manifests.{name}: retained+additions != windows")
        counts = {"retained": 0, "additions": 0}
        for row in rows:
            if row.get("coverage_addition"):
                counts["additions"] += 1
            else:
                counts["retained"] += 1
            raw_split = row.get("split")
            _require(raw_split in _ROW_SPLITS[name],
                     f"{name}: row {row.get('window_id')!r} has split {raw_split!r}; "
                     f"allowed {sorted(_ROW_SPLITS[name])}")
        _require(counts["retained"] == retained and counts["additions"] == additions,
                 f"manifests.{name}: retained/additions {counts} != declared {retained}/{additions}")
        splits[name] = SplitSpec(
            name=name,
            manifest_path=manifest_path,
            declared_sha256=declared,
            windows=windows,
            retained=retained,
            additions=additions,
        )

    retained_sources = spec.get("retained_sources") or {}
    for name in SPLIT_NAMES:
        _check_declared_file(retained_sources.get(name), label=f"retained_sources.{name}", root=root)

    benchmark = spec.get("benchmark") or {}
    _require(isinstance(benchmark, dict) and benchmark, "benchmark declaration is required")
    benchmark_manifest_path, _ = _check_declared_file(
        {"path": benchmark.get("manifest"), "sha256": benchmark.get("manifest_sha256")},
        label="benchmark.manifest", root=root,
    )
    benchmark_exposure_path, _ = _check_declared_file(
        {"path": benchmark.get("exposure"), "sha256": benchmark.get("exposure_sha256")},
        label="benchmark.exposure", root=root,
    )

    version_sha = sha256_file(version_path)
    return DatasetVersion(
        path=version_path,
        sha256=version_sha,
        spec=spec,
        protocol=str(spec["protocol"]),
        strict_conversion_family_compliant=bool(spec.get("strict_conversion_family_compliant", False)),
        default_base=default_base,
        extra_bases=extra_bases,
        splits=splits,
        benchmark=benchmark,
        benchmark_manifest_path=benchmark_manifest_path,
        benchmark_exposure_path=benchmark_exposure_path,
    )


def _validate_row(version: DatasetVersion, name: str, row: Mapping[str, Any]) -> None:
    window_id = row.get("window_id")
    _require(isinstance(window_id, str) and window_id, f"{name}: row without window_id")
    label = row.get("label")
    _require(label in (0, 1), f"{name}: row {window_id}: label must be 0/1")
    _require(isinstance(row.get("spoken_language"), str) and row["spoken_language"],
             f"{name}: row {window_id}: spoken_language is required")
    audio = row.get("prepared_audio")
    _require(isinstance(audio, dict), f"{name}: row {window_id}: prepared_audio is required")
    rel = audio.get("path")
    _require(isinstance(rel, str) and rel, f"{name}: row {window_id}: prepared_audio.path is required")
    rel_path = Path(rel)
    _require(not rel_path.is_absolute(), f"{name}: row {window_id}: prepared path must be repo-relative")
    _require(".." not in rel_path.parts, f"{name}: row {window_id}: prepared path must not contain '..'")
    digest = audio.get("sha256")
    _require(isinstance(digest, str) and len(digest) == 64, f"{name}: row {window_id}: invalid prepared sha256")
    base = row.get("path_base")
    if base is not None:
        _require(base in version.bases, f"{name}: row {window_id}: undeclared path_base {base!r}")
    _require(row.get("preprocessing_version") == PREPROCESSING_VERSION,
             f"{name}: row {window_id}: preprocessing_version {row.get('preprocessing_version')!r} "
             f"!= {PREPROCESSING_VERSION!r}")
    revisions = version.spec.get("revisions") or {}
    declared_revision = revisions.get(str(row.get("dataset_id") or ""))
    if declared_revision:
        actual_revision = (row.get("parent_refs") or {}).get("dataset_revision")
        _require(actual_revision == declared_revision,
                 f"{name}: row {window_id}: {row.get('dataset_id')} source revision {actual_revision!r} "
                 f"!= declared {declared_revision!r}")


def load_split_rows(
    version: DatasetVersion,
    split: str,
    *,
    repo_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Load one split's rows in deterministic manifest order, validated.

    The returned rows are shallow copies carrying the normalized
    ``resolved_split`` (``train``/``dev``) and ``resolved_path`` (absolute
    prepared-audio path) fields; the on-disk manifest itself is never
    modified.
    """
    _require(split in SPLIT_NAMES, f"unknown split {split!r}; expected one of {SPLIT_NAMES}")
    spec = version.split(split)
    rows = _read_jsonl(spec.manifest_path)
    _require(len(rows) == spec.windows, f"{split}: manifest changed since version load")
    seen: set[str] = set()
    resolved: list[dict[str, Any]] = []
    for row in rows:
        _validate_row(version, split, row)
        window_id = str(row["window_id"])
        _require(window_id not in seen, f"{split}: duplicate window_id {window_id}")
        seen.add(window_id)
        prepared = dict(row)
        prepared["resolved_split"] = split
        prepared["resolved_path"] = str(resolve_prepared_path(row, version, repo_root=repo_root))
        resolved.append(prepared)
    return resolved


def resolve_prepared_path(
    row: Mapping[str, Any],
    version: DatasetVersion,
    *,
    repo_root: str | Path | None = None,
) -> Path:
    """Resolve one row's prepared audio path through its declared base.

    Rows without ``path_base`` resolve under ``path_resolution.default_base``;
    rows carrying ``path_base`` must name a base declared by the version
    record.  The resolver never searches other roots and rejects any result
    that escapes its base directory.
    """
    root = Path(repo_root or ".").resolve()
    audio = row.get("prepared_audio") or {}
    rel = Path(str(audio.get("path", "")))
    _require(bool(rel.name), f"row {row.get('window_id')!r}: missing prepared path")
    _require(not rel.is_absolute() and ".." not in rel.parts,
             f"row {row.get('window_id')!r}: prepared path must be safe and relative")
    base = row.get("path_base")
    if base is None:
        base = version.default_base
    _require(base in version.bases, f"row {row.get('window_id')!r}: undeclared path_base {base!r}")
    base_dir = (root / str(base)).resolve()
    target = (base_dir / rel).resolve()
    try:
        inside = os.path.commonpath([str(base_dir), str(target)]) == str(base_dir)
    except ValueError:
        inside = False
    _require(inside, f"row {row.get('window_id')!r}: prepared path escapes base {base!r}")
    return target


def load_split_pair(
    version: DatasetVersion,
    *,
    repo_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load train and dev together and enforce cross-split invariants.

    * window ids are unique across both splits;
    * no train/dev window id appears in the evaluated benchmark manifest
      (ids/hashes are read from that file as provenance only - its audio is
      never touched);
    * the declared split membership counts hold.
    """
    train = load_split_rows(version, "train", repo_root=repo_root)
    dev = load_split_rows(version, "dev", repo_root=repo_root)
    train_ids = {str(row["window_id"]) for row in train}
    dev_ids = {str(row["window_id"]) for row in dev}
    overlap = train_ids & dev_ids
    _require(not overlap, f"{len(overlap)} window ids shared between train and dev: {sorted(overlap)[:5]}")

    benchmark_path = version.benchmark_manifest_path
    benchmark_ids: set[str] = set()
    if benchmark_path.is_file():
        for row in _read_jsonl(benchmark_path):
            window_id = row.get("window_id")
            if isinstance(window_id, str):
                benchmark_ids.add(window_id)
    leaked = (train_ids | dev_ids) & benchmark_ids
    _require(not leaked,
             f"{len(leaked)} train/dev window ids appear in the evaluated benchmark: {sorted(leaked)[:5]}")
    return train, dev


def iter_rows(rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
    """Deterministic iteration helper (manifest order is the canonical order)."""
    return rows


__all__ = [
    "DatasetVersion",
    "DatasetVersionError",
    "PREPROCESSING_VERSION",
    "PROTOCOL",
    "SCHEMA_V3",
    "SPLIT_NAMES",
    "SplitSpec",
    "iter_rows",
    "load_dataset_version",
    "load_split_pair",
    "load_split_rows",
    "resolve_prepared_path",
    "sha256_file",
]
