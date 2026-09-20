> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Frozen baseline: first v2 test evaluation

Completed 19 September 2026. **Research baseline only; these results do not support production promotion.**

The epoch-five checkpoint and threshold 0.5 were fixed before inference in `artifacts/evaluations/gru-v2-test-epoch5/plan.json`. All 96 prepared windows were scored through the real frozen encoder and restored standardizer/GRU, with no skipped examples, parameter updates or threshold tuning. The checkpoint and manifest hashes remained unchanged.

## Overall results

| Metric | Development (selection set) | New test |
|---|---:|---:|
| Accuracy | 94.79% | 79.17% |
| EER | 4.17% | 16.67% |
| Genuine false-alarm rate at 0.5 | 4.17% | 35.42% |
| Spoof miss rate at 0.5 | 6.25% | 6.25% |

Test confusion: **31 genuine accepted, 17 genuine falsely flagged, 45 spoofs detected, 3 spoofs missed**. Margin AUC is 0.91276; probability and logit-margin EER agree. EER is a ranking summary at its own operating point, not a recommended deployment threshold. Counts were independently recomputed from the saved predictions.

## Gender and generator coverage

| Genuine group | Windows | False alarms | False-alarm rate |
|---|---:|---:|---:|
| Documented female | 24 | 1 | 4.17% |
| Documented male | 24 | 16 | 66.67% |

The training genuine examples were all female. This test shows a substantial disparity consistent with inadequate coverage. It does not establish that gender alone caused the errors: speaker, recording channel, content and dataset-selection effects may also differ. Adding representative male genuine TRAIN/DEV examples is a justified next experiment, using identities separate from this evaluated benchmark. Do not repurpose its 24 male genuine recordings as training data while retaining a held-out claim.

Synthetic misses: FreeVC24 0/25, XTTSv2 3/21, VITS 0/2. Two VITS examples cannot establish reliable generator performance. Synthetic target-gender misses: female 2/21, male 1/27.

## Language results

Each language has four genuine and four synthetic windows. Counts are more informative than fine-grained percentages at this sample size.

| Language | Genuine false alarms / 4 | Spoof misses / 4 |
|---|---:|---:|
| Bengali | 1 | 0 |
| Gujarati | 1 | 0 |
| Hindi | 1 | 1 |
| Kannada | 2 | 0 |
| Malayalam | 1 | 0 |
| Marathi | 2 | 0 |
| Odia | 2 | 1 |
| Punjabi | 1 | 0 |
| Sanskrit | 3 | 0 |
| Tamil | 2 | 1 |
| Telugu | 1 | 0 |
| Urdu | 0 | 0 |

## Uncertainty and scope

Exploratory 95% percentile bootstrap intervals, resampling 84 admitted relationship components with 2,000 replicates: accuracy **71.11–86.46%**, genuine false-alarm rate **21.95–48.89%**, spoof miss rate **0–14.29%**. These intervals are conditional on this small, selected sample and the admitted grouping assumptions; they do not cover unidentified cross-language people or the stricter conversion-chain relationships.

Protocol: `speaker_recording_disjoint_v2`. Strict conversion-family isolation is not met. English remains unvalidated. Inference covers a preselected 1–4 second window, not every region of a long recording. No real-time or service-level latency claim is made.

## Decision and next experiment

Retain the model as a reproducible research baseline; do not promote it as a production detector. Improve genuine training/development coverage through separate male and female speakers, audit recording-condition differences, and retrain using development-only selection. Do not calibrate a threshold to these test errors. Future adaptive work should obtain a new protected evaluation set; this 96-example set is now an evaluated benchmark.

The evaluated identities are recorded in `artifacts/evaluations/gru-v2-test-epoch5/exposure.json` for future split exclusion. Future acquisition must explicitly consume this registry; the old v2 registry is historical and has not been silently changed.

## Evidence

`artifacts/evaluations/gru-v2-test-epoch5/` contains the pre-inference plan, pinned release, source snapshot, 96 per-example predictions, full metrics/subgroups/intervals, and exposure record. No failed/abstained inputs occurred. Seventeen scoring/reporting tests passed before evaluation. No downloads, training, package replacement, commit, push or merge occurred.
