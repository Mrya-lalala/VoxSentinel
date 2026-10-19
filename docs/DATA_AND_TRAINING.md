# Current data and training

V3 contains **458 training windows** (266 genuine / 192 synthetic) and **135 development windows** (87 genuine / 48 synthetic), across twelve Indic languages. All original examples were retained; additions contributed 74 training and 39 development male genuine recordings. Malayalam has no male additions, Odia has one development addition only, and Bengali has one new speaker per split.

Data and caches live under `artifacts/datasets-v3-coverage/` and are not supplied by a normal source checkout. For inference, use the [quickstart](TEAMMATE_RESEARCH_BASELINE.md); training assets are unnecessary.

## Contracts and commands

`manifests/dataset-version.v3.json` binds train/dev manifests, revisions and path resolution. Retained audio resolves under `artifacts/datasets`; additions resolve under their explicit v3 `path_base`. Never move rows between splits or reuse the old complete-cache bundles as though they contained the additions.

The validated v3 feature bundles contain 458 train / 135 dev items. They reuse 480 verified historical embeddings and add 113 fresh embeddings from the frozen IndicWav2Vec final layer. Standardization must be fitted on training frames only.

The local caches in this checkout were rebuilt by fresh extraction with the pinned frozen encoder because the canonical features bundle was not supplied; counts, labels, order and per-split frame totals are identical, but no historical embedding was reused. Verify with `scripts.prepare_dataset_v3_features check` / `parity --threads 12` and see [AASIST training results](AASIST_TRAINING_RESULTS.md) §1–2.

From the repository root, with the required local artifacts present:

```sh
# Readiness only: no fitting or model inference
.venv/bin/python -m scripts.train_gru_v3_from_cache --check-only

# Explicitly starts a NEW training run; preserves prior run directories
.venv/bin/python -m scripts.train_gru_v3_from_cache \
  --run-dir artifacts/runs/gru-v3-expanded-standardized --threads 4

# Embedding-adapted AASIST over the same caches, splits and protocol
.venv/bin/python -m scripts.train_aasist_v3_from_cache --check-only
.venv/bin/python -m scripts.train_aasist_v3_from_cache \
  --run-dir artifacts/runs/aasist-v3-expanded-standardized --threads 4
```

The fixed comparison uses ten epochs, seed zero, batch eight, unweighted cross-entropy, AdamW learning rate 0.001, and threshold 0.5. Epoch selection uses minimum development EER with the earliest tie. The completed GRU run selected epoch three; see [results](V3_TRAINING_RESULTS.md). The AASIST sibling runner enforces the identical protocol for the embedding-adapted AASIST head; its completed run selected epoch ten and scored a development EER of 8.05% — higher (worse) than the GRU's 2.30% on the same split, with 91.85% vs 97.04% accuracy — see [AASIST training results](AASIST_TRAINING_RESULTS.md). Do not treat development performance as independent-test evidence.

The previously scored 96-example benchmark is excluded from additions. Its results informed coverage acquisition, so any future repeat evaluation is on an already-seen benchmark. The adopted speaker/recording-disjoint protocol does not establish strict conversion-family isolation or cross-language person independence. Indian-English remains pending IndieFake access.

Acquisition commands are not part of routine setup. They consume a shared capped ledger; the recorded remaining allowance after v3 acquisition is 342,905,206 bytes. Re-read the actual ledger before any new acquisition.

## Detailed historical evidence

- [Acquisition handoff](archive/history/GENUINE_COVERAGE_ACQUISITION_HANDOFF.md) and [independent review](archive/history/GENUINE_COVERAGE_ACQUISITION_REVIEW.md).
- [Cache construction handoff](archive/history/V3_CACHE_AND_TRAINING_SUPPORT_HANDOFF.md) and [pre-training review](archive/history/V3_CACHE_AND_TRAINING_SUPPORT_REVIEW.md). Their “not trained yet” status is historical; the fixes and completed run are documented in the current training results.
- [Earlier v2 benchmark results](archive/history/V2_FROZEN_TEST_RESULTS.md). These are not v3 scores.
