# VoxSentinel
SIH26104 — AI-powered real-time, language-agnostic voice-cloning and impersonation detection.

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
  Verdict: training-dynamics/convergence failure on weak, ill-conditioned inputs; one
  controlled whitened-input rerun recommended.
