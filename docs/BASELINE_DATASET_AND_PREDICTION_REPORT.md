# Dataset decision and prediction-path report

Status: 19 September 2026. Local research pilot, not a final production release.

**Subsequent user decision:** Preserve acquired data; do not downsample for exact gender balance. Check upstream metadata before concluding male speech is unavailable. The equal-gender quotas below are superseded by `DATA_SPLIT_AGENT_MASTER_PROMPT.md` and the updated planning config. IndieFake access blocks Indian-English preparation, not an honestly scoped Indic-only split. Earlier statements that male acquisition is mandatory describe the previous plan; documented limitations may remain in a provisional Indic split.

The standardized GRU is now packaged locally and can score real audio files through the same preprocessing and frozen encoder used for training. Integration checks passed. The next combined Indic / Indian-English training run is **not ready**: IndieFake access is pending, genuine male coverage must be acquired, and a new untouched test must be selected and audited. No new training, push or merge was performed in this phase.

## Dataset decision

| Dataset | Selected role | Conditions |
|---|---|---|
| Kathbath | Genuine Indic training, development and independent test | Acquire documented male as well as female candidates; pin existing revision; audit related identities before splitting. |
| IndicSynth | Synthetic Indic training, development and independent test | Retain verified source/target lineage and report each generator; preserve CC BY-NC 4.0 restrictions. |
| IndieFake | Genuine and synthetic Indian-English training, development and test | User has requested access. No local archive or download link yet. Audit actual release, terms, speakers, originals, augmentations and generator labels first. |
| Svarah | External Indian-English genuine false-alarm evaluation | Genuine-only evaluation cannot estimate spoof recall or EER. Keep separate from fitting and threshold selection. |
| ASVspoof 2019 LA | Separate general-English reference experiment | Not a substitute for Indian-English evidence; do not silently pool with the main training set. |
| NISP | Deferred supplementary genuine-speech coverage | Add only with a matched synthetic/condition strategy and identity audit. |
| NPTEL | Deferred | Lecturer identity unresolved; exclude from the primary benchmark. |

IndieFake is an appropriate candidate because its paper describes both genuine and generated Indian-accent English. Its reported speaker and augmentation structure means we must inspect lineage rather than assume every file has a unique genuine/fake counterpart. Dataset access is via the [official project](https://indie-fake-dataset.netlify.app/); methodology is described in the [paper](https://arxiv.org/html/2506.19014v1). No second access request was submitted. Svarah's role follows its [official repository](https://github.com/AI4Bharat/Svarah).

## What the present baseline actually learned

The frozen core contains 384 training windows and 96 development windows, evenly divided between genuine and synthetic speech across 12 Indic languages. Each language contributes 16 examples per class to training and four per class to development. Genuine recordings are all female. Synthetic target gender is 97 female / 95 male in training and 48 female in development. The cached Kathbath candidate inventories also contain only female entries: re-running selection on those inventories cannot fix the gap.

Preprocessing decodes audio, averages channels, resamples to 16 kHz using soxr HQ, attenuates overflow only, and selects a contiguous 1–4 second window. Longer recordings use the highest-energy four-second window, scanned every 0.25 seconds plus the tail. A short-time activity check rejects insufficient speech. There is no denoising, loudness normalization or augmentation in this baseline.

The pinned IndicWav2Vec encoder stays frozen. Its final-layer 1,024-dimensional frame features feed a training-only frame standardizer, LayerNorm, a projection to 256 dimensions, a one-layer 256-unit GRU, valid-frame mean pooling and a two-logit classifier. The standardizer excludes padding and is restored with the head. The head has 659,714 parameters. Epoch five of the controlled ten-epoch run was selected on development performance: 94.79% accuracy, 4.17% EER, AUC 0.986979. These are repeatedly inspected development results, not independent generalization estimates. Full evidence and training details are in [the recovery report](ASTRA_GRU_RECOVERY_MANAGER_REPORT.md).

## Next dataset contract

`configs/datasets_v2_plan.yaml` is a planning contract, not an active acquisition configuration. A controlled Indic comparison targets the same 384 training / 96 development windows plus 96 new test windows, with equal documented male/female coverage within each language/class/split. These small quotas support diagnosis, not production claims. English quotas remain unset until IndieFake's actual independent speaker groups are known. Do not duplicate examples or relax identity rules to fill quotas.

Split connected components of genuine speakers, conversion source speakers, target speakers, parent recordings and augmentation originals **before** extracting windows. Keep upstream test material reserved. Report source and target gender separately; use the documented output/target identity for synthetic gender balancing. Quarantine unresolved required lineage. Audit near-duplicates and cross-language identities; language-scoped numeric IDs alone do not prove different people.

The new audit exports 154 previously exposed speaker keys and 521 parent-recording keys from all v1 core train/development examples to `artifacts/datasets-v2/planning/prior_exposure_exclusions.json`, with input hashes. This is an initial exclusion registry, not a completed test split: add any other previously fitted or inspected examples, and resolve identity aliases before assigning the test set. The tool does not yet enforce exclusions in an acquisition pipeline.

Keep the old manifests, caches and runs frozen. New material belongs under `artifacts/datasets-v2`. The existing shared download ledger has 1,748,211,957 bytes remaining under its 5 GiB cap. Do not reset that ledger for v2. Metadata and acquisition planning must account for the remaining capacity; full IndieFake acquisition is not assumed to fit.

After selection and audit, cache features using the same pinned encoder. Refit the standardizer on new training frames only, then train the unchanged head. Select epoch and any threshold using development data only. Freeze both before a one-time test evaluation. Report per-language, gender, generator and recording-condition performance, plus false alarms and missed spoofs at the chosen threshold. Review confidence intervals grouped by independent speakers/components. The present training runner's strict reference-run checks need a deliberate v2 adaptation before using different manifests; do not pass new caches as though they were the old frozen core.

## Complete implemented file-prediction path

New files: `src/scoring/predict.py`, `scripts/predict_audio.py`, `scripts/package_pilot_baseline.py`, `scripts/verify_prediction_path.py`, and `requirements-inference.txt`.

The local release is `artifacts/releases/indic-gru-pilot-v0.1/release.json`. It pins the head hash, encoder identity, window policy and threshold. Encoder weights remain an external local asset. The predictor verifies the head hash and encoder/training identity, restores the saved standardizer and GRU, decodes the input file, runs the real frozen encoder and returns an uncalibrated synthetic score and genuine/spoof label at threshold 0.5. Insufficient audio returns no classification. Explicit language metadata flags whether the language appeared in training; it does not detect language or establish performance on it.

Validation in `artifacts/prediction-path/report.json` passed:

- Four raw recordings: Hindi and Tamil, both genuine and synthetic. Included 16 kHz genuine audio and 24 kHz synthetic audio requiring resampling.
- Raw-file logits matched cached-feature logits exactly in these four cases; selected offsets and valid-frame counts also matched.
- Equal-channel stereo and mono produced identical logits.
- Silence and half-second audio abstained; corrupt audio and a wrong head hash were rejected.
- Encoder and head remained frozen with no gradients; checkpoint bytes were unchanged.

The four development clips verify integration, not independent accuracy. The first prediction included lazy encoder startup and took about nine seconds; later examples took about 0.23–0.59 seconds on this machine. These are observations from one run, not a latency benchmark. One selected window does not establish that an entire long recording is genuine. Microphone capture, streaming, a service endpoint and a UI are outside this implemented path. The path uses the trained Indic encoder directly; routing into WavLM would require a compatible trained head and separate validation.

## Release and production decision

The current model is a reproducible **Indic research pilot**. The local package is useful for integration work but does not meet the requirements for the final shared Indic / Indian-English baseline. Before promotion: acquire IndieFake; complete balanced selection and untouched-test audits; run the controlled training comparison; pass agreed false-alarm/miss requirements; verify license-compatible weight distribution and a clean teammate setup. Do not merge merely because development accuracy is high. Artifacts are gitignored, so source-code publication alone would not distribute a usable checkpoint.

Larger, more representative training is needed before production, but it should happen iteratively alongside development, not only after the application is finished. Start with a reproducible small baseline; expand speakers, generators, languages and real recording channels; evaluate every frozen candidate on protected tests. More data alone cannot fix leakage, source/label confounding or missing deployment conditions. Production size should be determined by learning curves and operational error targets, not a single arbitrary clip count.

## Reproduce

See [prediction setup and usage](prediction-path.md). Run `.venv/bin/python -m scripts.audit_dataset_v2_readiness` to regenerate planning evidence without downloading or changing frozen data. A `not_ready_to_train_or_release` report is the expected current result; a successful audit command is not authorization to train or release.
