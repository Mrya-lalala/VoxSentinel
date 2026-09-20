# Datasets (`src/data/`)

This directory owns the dataset layout, acquisition, manifests and label
conventions for VoxSentinel's three core corpora, as defined by the
`02-DATASET-TRAINING` plan.

## The three corpora

| Corpus | Role | Default label | Used for |
| --- | --- | --- | --- |
| **IndicSynth** | Synthetic (spoof) | 1 — spoof | Training / dev |
| **Kathbath** | Genuine (bona fide) | 0 — genuine | Training / dev |
| **Svarah** | Evaluation only | mixed | **Held-out evaluation only** |

### Label convention

All adapters and manifests use the project-wide binary convention:

- `0` = genuine (bona fide)
- `1` = spoof

This matches `src/data/labels.py` (`GENUINE = 0`, `SPOOF = 1`) and is validated
before any label is written to a manifest.

### Pilot languages

**IndicSynth** and **Kathbath** are our starting **2–3 pilot languages** taken
from the verified-accessible list. The set is declared once in
`dataset_setup.PILOT_LANGUAGES` (`hindi`, `bengali`, `marathi`) with short codes
in `LANGUAGE_CODES`, so it can be widened after the pilot track is validated —
without changing adapter code.

### Svarah is strictly held out

**Svarah** is **strictly held-out evaluation data**. It must never be used for
training, hyper-parameter tuning, model selection or threshold selection.
`assert_training_allowed()` and `download_svarah()` make accidental use fail
loudly.

## Folder structure

```
data/
├── indicsynth/
│   ├── raw/<code>/        # downloaded archives (not committed)
│   ├── processed/<code>/  # extracted source audio
│   └── manifests/
├── kathbath/
│   ├── raw/<code>/
│   ├── processed/<code>/
│   └── manifests/
├── svarah/                 # held-out evaluation only
│   └── ...
└── manifest.csv            # combined training manifest
```

(`<code>` is `hi` / `bn` / `mr` for the pilot languages.)

## Quick start

```bash
# 1. Create the folder structure.
python src/data/dataset_setup.py layout --root data

# 2. Fill in the verified accessible subset URLs and optional SHA-256 digests.
python src/data/dataset_setup.py sources-template --output configs/dataset_sources.json

# 3. (Optional) Preview the plan without downloading anything.
python src/data/dataset_setup.py download --dry-run

# 4. Download + extract the IndicSynth and Kathbath pilots, then write manifest.csv.
python src/data/dataset_setup.py all --root data

# Or run the steps individually:
python src/data/dataset_setup.py download --root data
python src/data/dataset_setup.py manifest --root data --limit 200
```

## Source configuration

The exact upstream locations live in the verified-accessible list from the master
document. Supply them in `configs/dataset_sources.json` (or point
`VOXSENTINEL_DATASET_SOURCES` at another file). Each entry is:

```json
{"dataset": "kathbath", "language": "hindi", "url": "https://.../hi.tar.gz",
 "version": "kathbath-v1.0-pilot", "sha256": null, "archive": null}
```

- `url` may be `http(s)://`, a `file://` URI or a local path (useful offline).
- `sha256` is optional but recommended; downloads are digest-verified.
- `archive` is inferred from the URL suffix (`tar`/`zip`); a bare audio file is
  copied as-is (`archive: "none"`).

No URL is invented in code — `select_sources()` fails loudly if a pilot language
has no configured URL.

## Manifest (`manifest.csv`)

`manifest.csv` lives at the data root (default `data/manifest.csv`) and contains
the minimum information downstream needs — exactly these five columns, in order:

| column | meaning |
| --- | --- |
| `sample_id` | stable id, e.g. `kathbath_hi_000001` |
| `path` | audio path relative to the data root |
| `label` | `0` = genuine, `1` = spoof |
| `language` | spoken language (`hindi` / `bengali` / `marathi`) |
| `dataset_version` | corpus version stamp, e.g. `kathbath-v1.0-pilot` |

An optional JSONL manifest (`ManifestRecord`, `write_manifest` / `iter_manifest`)
is retained for `split`/`speaker_id` tooling that needs more than the CSV.

## API overview

| Function | Purpose |
| --- | --- |
| `setup_dataset_directories(root)` | Create the layout for all three corpora. |
| `load_sources` / `select_sources` | Read and validate the source registry. |
| `download_file(url, dest, expected_sha256=...)` | Stream + digest-verify a file. |
| `extract_archive(archive, dest)` | Safe tar/zip extraction (path-traversal guarded). |
| `download_indicsynth(root, languages=...)` | Fetch synthetic audio (label 1). |
| `download_kathbath(root, languages=...)` | Fetch genuine audio (label 0). |
| `download_svarah(root)` | Refuses: Svarah is held out. |
| `build_manifest(root, limit=...)` | Discover audio and write `manifest.csv`. |
| `prepare_pilot_datasets(root, ...)` | layout + download + manifest in one call. |
| `assert_training_allowed(key)` | Raise if `key` is a held-out corpus. |
| `training_specs()` / `evaluation_specs()` | Train/dev vs held-out corpora. |

## Related modules

- `labels.py` — the `0`/`1` convention and validators (source of truth).
- `batch.py` — in-memory embedding batches consumed by the detector head.
- `synthetic.py` — a plumbing-only synthetic fixture for loop tests.