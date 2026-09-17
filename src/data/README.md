# Datasets (`src/data/`)

This directory owns the dataset layout, manifests and label conventions for
VoxSentinel's three core corpora, as defined by the `02-DATASET-TRAINING` plan.

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
by `validate_label` before any label is written to a manifest.

### Pilot languages

**IndicSynth** and **Kathbath** are our starting **2–3 pilot languages** and
seed the initial train/dev pool. The pilot set is declared once in
`dataset_setup.PILOT_LANGUAGES` so it can be widened after the pilot track is
validated — without changing any adapter code.

### Svarah is strictly held out

**Svarah** is **strictly held-out evaluation data**. It must never be used for
training, hyper-parameter tuning, model selection or threshold selection. The
`assert_training_allowed()` guard exists to make accidental use fail loudly.

## Folder structure

Running the setup script creates this layout (relative to the data root,
default `data/`, which is git-ignored):

```
data/
├── indicsynth/
│   ├── raw/          # downloaded source audio (not committed)
│   ├── processed/    # resampled / chunked audio
│   └── manifests/    # JSONL manifests
├── kathbath/
│   ├── raw/
│   ├── processed/
│   └── manifests/
└── svarah/           # held-out evaluation only
    ├── raw/
    ├── processed/
    └── manifests/
```

Create the structure with:

```bash
python -m src.data.dataset_setup --root data
```

## Manifests

Each corpus produces a JSONL manifest with one `ManifestRecord` per audio item:

```json
{"path": "data/kathbath/raw/hi_0001.wav", "label": 0, "dataset": "kathbath", "language": "hindi", "split": "train", "speaker_id": "spk_07"}
```

Records are validated on construction and on read:

- `write_manifest(records, destination)` — serialise records to JSONL.
- `iter_manifest(path)` — stream validated records back.

## Starter functions

The download and parse adapters in `dataset_setup.py` are **intentionally
inert stubs**. They validate their arguments and prepare directories, then raise
`NotImplementedError` until corpus access is authorised. No bytes are fetched
and no licences are accepted automatically.

| Function | Purpose |
| --- | --- |
| `setup_dataset_directories(root)` | Create the layout for all three corpora. |
| `download_indicsynth(root, languages=...)` | Fetch synthetic audio (label 1). |
| `download_kathbath(root, languages=...)` | Fetch genuine audio (label 0). |
| `download_svarah(root)` | Fetch held-out evaluation audio. |
| `parse_indicsynth(root, manifest_path=...)` | Emit the IndicSynth JSONL manifest. |
| `parse_kathbath(root, manifest_path=...)` | Emit the Kathbath JSONL manifest. |
| `parse_svarah(root, manifest_path=...)` | Emit the Svarah evaluation JSONL manifest. |
| `assert_training_allowed(key)` | Raise if `key` is a held-out corpus. |
| `training_specs()` / `evaluation_specs()` | Split corpora into train/dev vs held-out. |

## Related modules

- `labels.py` — the `0`/`1` convention and validators (source of truth).
- `batch.py` — in-memory embedding batches consumed by the detector head.
- `synthetic.py` — a plumbing-only synthetic fixture for loop tests.