# Training reproduction with the v2 split contract

Completed 19 September 2026. This is a fresh ten-epoch training run from the original seed, not a resumed run or a new-data improvement experiment.

## Changes and data used

Added `--split-version` to `scripts/train_gru_from_cache.py`, backed by `scripts/training_split_contract.py`. Before training it verifies the v2 train/development file hashes against both the version metadata and the original cache manifests, and rejects empty or incorrect split assignments. The verified contract is stored in the run and checkpoint metadata. Test identity is recorded as metadata only; test rows, features and audio are not loaded by this guard or training.

This adapter deliberately supports the present byte-identical v2 train/development copies only. If future training data changes, it fails instead of silently reusing stale features. The existing cache/readiness audits still verify labels, language coverage, audio hashes, encoder identity, features and canonical split identities.

The run used 384 training and 96 development windows. All genuine training examples remain female: the newly acquired male genuine examples are in the reserved test set. Moving those examples into training would destroy this test's intended role. Future coverage expansion should acquire/select separate unexposed training components and create new audited caches.

## Result

| Item | Result |
|---|---:|
| Epochs | 10 |
| Selected epoch (development EER; earliest tie) | 5 |
| Development accuracy | 94.7917% |
| Development EER | 4.1667% |
| Development genuine false-alarm rate at 0.5 | 4.1667% |
| Development spoof miss rate at 0.5 | 6.25% |
| Confusion: TN / FP / FN / TP | 46 / 2 / 3 / 45 |
| Training loop wall time, including checkpointing/evaluation | 31.50 seconds |

The best and final checkpoint model states (including the fitted standardizer) and optimizer states exactly match the earlier standardized run. All epoch history values match, and exported development predictions are byte-identical. Checkpoint file hashes differ because run metadata now includes the new run path and v2 contract; this does not indicate changed learned parameters.

The encoder was frozen and its audited cached features were reused. The standardizer was freshly fitted on valid training frames only. The selected checkpoint was verified against live selected-epoch predictions; restore, batch-size and masked-padding checks passed. All 11 data-preparation/contract tests and 8 training/checkpoint tests passed. The reserved test manifest hash is unchanged.

## Scope

No independent test predictions were generated, no threshold was tuned, and no data was downloaded. This demonstrates reproducibility, not improved accuracy or production readiness. The v2 test protocol remains `speaker_recording_disjoint_v2`; strict conversion-family isolation is not met. Indian-English coverage is pending IndieFake access. No model package was replaced, and nothing was committed, pushed or merged.

The next evaluation can use the fixed epoch-five checkpoint and threshold 0.5 on the reserved 96-window test, reporting gender/language/generator subsets with small-sample limitations. Freeze that evaluation plan first; do not use test results to select among checkpoints or retune the threshold.

## Artifacts and reproduction

Run: `artifacts/runs/gru-v2-split-reproduction/`.

Key files: `gru_best.pt`, `gru_final.pt`, `split_contract.json`, `reproduction_checks.json`, `run_status.json`, `metrics.json`, `history.json`, `evaluation/evaluation.json`, `val_predictions.jsonl`, and the full `source_snapshot.tar.gz` with file hashes.

```sh
.venv/bin/python -m scripts.train_gru_from_cache \
  --split-version artifacts/datasets-v2/manifests/dataset-version.v2.json \
  --detector-config configs/base.yaml configs/gru.yaml configs/gru_standardized.yaml \
  --reference-run artifacts/runs/gru-core-v1 \
  --threads 4 \
  --run-dir artifacts/runs/gru-v2-split-reproduction
```

An existing nonempty run directory is preserved and a numbered sibling is created. This remains the controlled CPU, frozen-cache training path; it is not a generic loader for arbitrary future datasets.
