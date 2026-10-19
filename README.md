# VoxSentinel

SIH26104 — voice-cloning and impersonation detection research.

**Current deliverable: Indic research baseline, not production validated.**
The released GRU v3 model achieved 97.04% development accuracy and 2.30% EER on
135 examples. The embedding-adapted AASIST research head, trained under the same
protocol, reached 91.85% accuracy and 8.05% EER on the same split — worse than
the GRU, and not a claim of improvement.
Development was used for checkpoint selection; v3 benchmark performance,
Indian-English support and streaming performance remain unvalidated.

- [Teammate quickstart: setup, model bundle and predictions](docs/TEAMMATE_RESEARCH_BASELINE.md)
- [Current data and training workflow](docs/DATA_AND_TRAINING.md)
- [V3 training results and limitations](docs/V3_TRAINING_RESULTS.md)
- [Documentation index](docs/README.md)

Earlier reports and completed agent prompts are in the [historical archive](docs/archive/README.md).
