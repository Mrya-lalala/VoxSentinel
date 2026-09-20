# VoxSentinel
SIH26104 — voice-cloning and impersonation detection research.

**Current deliverable: Indic research baseline, not production validated.**
Start with the [teammate quickstart](docs/TEAMMATE_RESEARCH_BASELINE.md).
V3 development: 97.04% accuracy / 2.30% EER on 135 examples. The v3 model
has not been evaluated on the existing benchmark; English and streaming
performance remain unvalidated.

- [Frozen encoder setup and interface](docs/b1-setup.md)
- [Real speech encoder → GRU smoke test and measured results](docs/encoder-gru-smoke.md)
- [Dataset preparation: paired core, audits and handoff](docs/dataset-preparation-pilot.md)
  — complete 12-language paired core (384 train / 96 val windows), Svarah external
  evaluation pool, feature caches with a training-readiness gate, full reference-lineage
  evidence, and the first controlled GRU run (`artifacts/runs/gru-core-v1/`, dev-val EER
  0.229 at epoch 2; scores collapse to one side at threshold 0.5 — no threshold tuning
  was performed). Reports: `artifacts/datasets/reports/preparation-report.{json,md}`
  (`python -m scripts.prepare_datasets report`).
- [Manager report: lineage resolution, dataset freeze and first GRU run](docs/DATASET_LINEAGE_AND_FIRST_GRU_RUN_MANAGER_REPORT.md)
  — compliance matrix, exact metrics, risk register, reproduce runbook and next steps.
- [GRU diagnostic report: why the first run scored everything synthetic](docs/GRU_DIAGNOSTIC_MANAGER_REPORT.md)
  — independent metric re-derivation, checkpoint-reload parity, feature/batching/optimizer
  audits, the 24-example overfit diagnostic, and a pooled linear baseline on the same frozen
  features (val EER 4.2 % / AUC 0.988) — evidence `artifacts/runs/gru-core-v1-diagnosis/`.
  Historical conclusions are superseded by the recovery report below.
- [Controlled GRU recovery report](docs/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md)
  — corrected linear score export and cross-class identity audit; one training-only
  frame-standardized GRU run selected epoch 5 (development accuracy 94.79%, EER 4.17%).
  These are pilot development results, not independent test or deployment evidence.
- [Dataset decisions, prediction tests and baseline release readiness](docs/BASELINE_DATASET_AND_PREDICTION_REPORT.md)
  — Kathbath + IndicSynth for Indic speech; IndieFake selected pending access for
  Indian English; balanced coverage and an untouched test remain required.
- [Run a local audio-file prediction](docs/prediction-path.md)
  — hash-pinned pilot package, real encoder/GRU inference and raw-audio parity checks.
- [Training reproduction with the v2 split contract](docs/V2_SPLIT_TRAINING_REPRODUCTION.md)
  — ten-epoch run exactly reproduced the standardized baseline; test data remains unused.
- [First fixed-checkpoint v2 test results](docs/V2_FROZEN_TEST_RESULTS.md)
  — 79.17% test accuracy; substantial male genuine false alarms; research baseline only.
- [V3 expanded-coverage training results](docs/V3_TRAINING_RESULTS.md)
  — selected epoch 3, development accuracy 97.04%, EER 2.30%; benchmark not rescored.
