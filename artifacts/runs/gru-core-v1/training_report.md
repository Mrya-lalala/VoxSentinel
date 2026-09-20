# First frozen-IndicWav2Vec → GRU training run

Run directory: `artifacts/runs/gru-core-v1`
Generated: 2026-09-18T07:44:35.013568+00:00

## What was fixed before this run

- Removed the stale duplicate preprocessing section (4.25 s allowance) from the docs.
- Audited all 480 core prepared files against `voxsentinel-prep-2` (mono/16 kHz/finite/≤1.0 peak/
  1–4 s contiguous spans/no padding; every prepared SHA-256 matched its manifest).
- Resolved `parsed_only` reference lineage with an evidence-based full train-shard scan
  (footer `fname` statistics + targeted column reads).
  Status counts: {"not_applicable_tts": 114, "verified_train": 366}
  Evidence methods: {"full_train_shard_scan": 236, "none": 114, "scanned_inventory": 130}
- Raw upstream row metadata is preserved per record (`parent_refs.upstream`) and TTS
  conditioning/reference recordings remain in the relationship graph and split checks.

## Readiness

- Cache audit ready: True (failures: none)
- Training data = `windows.core_train.jsonl` only; validation = `windows.core_val.jsonl` only.
- Label mapping: 0 = genuine, 1 = synthetic (two-logit cross-entropy).

## Resolved settings

- model: `gru` {"input_dim": 1024, "hidden_size": 256, "num_layers": 1, "dropout": 0.0, "num_classes": 2}
- optimizer: {"name": "adamw", "learning_rate": 0.001, "weight_decay": 0.0}
- training: {"batch_size": 8, "epochs": 10, "device": "cpu", "use_amp": false, "shuffle": true, "seed": 0, "loss": "cross_entropy"}
- scheduler: none; gradient clipping: none; class weights/oversampling: none
- checkpoint selection: lowest `eer` (mode min), earliest epoch wins ties

## Dataset composition

- train: {'0': 192, '1': 192} | languages=12
- val: {'0': 48, '1': 48} | languages=12

## Epoch history

| epoch | train loss | val loss | val acc | balanced acc | val EER |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7346 | 0.7215 | 0.5000 | 0.5000 | 0.3958 |
| 2 | 0.7113 | 0.7031 | 0.5000 | 0.5000 | 0.2292 |
| 3 | 0.7131 | 0.6940 | 0.5000 | 0.5000 | 0.4167 |
| 4 | 0.7074 | 0.7271 | 0.5000 | 0.5000 | 0.3125 |
| 5 | 0.6998 | 0.7047 | 0.5000 | 0.5000 | 0.3125 |
| 6 | 0.6953 | 0.7181 | 0.5000 | 0.5000 | 0.2708 |
| 7 | 0.7000 | 0.7090 | 0.5000 | 0.5000 | 0.2500 |
| 8 | 0.7226 | 0.7179 | 0.5000 | 0.5000 | 0.4167 |
| 9 | 0.7052 | 0.6963 | 0.5000 | 0.5000 | 0.3750 |
| 10 | 0.7006 | 0.6939 | 0.5000 | 0.5000 | 0.3333 |

Selected epoch (best `eer` = 0.22916666666666666): **2**
Head-training time: 43.9 s (macOS-26.5.2-arm64-arm-64bit, torch 2.2.2, torch threads 8)

## Best-checkpoint validation metrics (threshold 0.5, synthetic = positive)

- accuracy 0.5000 | balanced accuracy 0.5000 | EER 0.2292
- confusion (rows = true, cols = predicted; order genuine, synthetic): TN 0, FP 48, FN 0, TP 48
- genuine false-alarm rate 1.0000 | synthetic miss rate 0.0000
- EER is a ranking summary at its own operating point; it is not a deployed threshold.

## Per-language / per-generator breakdown

| group | windows | genuine | spoof | FP | FN | FPR | miss | balanced acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| generator:freevc24 | 25 | 0 | 25 | 0 | 0 | – | 0.000 | – |
| generator:genuine | 48 | 48 | 0 | 48 | 0 | 1.000 | – | – |
| generator:vits | 2 | 0 | 2 | 0 | 0 | – | 0.000 | – |
| generator:xtts_v2 | 21 | 0 | 21 | 0 | 0 | – | 0.000 | – |
| language:Bengali | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Gujarati | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Hindi | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Kannada | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Malayalam | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Marathi | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Odia | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Punjabi | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Sanskrit | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Tamil | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Telugu | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |
| language:Urdu | 8 | 4 | 4 | 4 | 0 | 1.000 | 0.000 | 0.500 |

## Limitations

- 384 training / 96 validation windows; each language contributes 8 validation examples.
- Validation selected the checkpoint; these are development results, not independent test performance.
- No unseen-generator, Indian-English, calibration or latency claim is made.
- Exact hashing cannot exclude near-duplicates or unidentified cross-corpus speakers.
- Head-only training on detached cached embeddings; the encoder is frozen and not updated.

Artifacts: `settings.json`, `dataset_identity.json`, `verification_summary.json`,
`environment.json`, `code_state.json`, `code_diff.patch`, `history.{csv,json}`,
`gru_best.pt`, `gru_final.pt`, `val_predictions.{csv,jsonl}`, `metrics.json`,
`breakdown.json`, `run_status.json`. Learning-curve plot deferred: matplotlib is not
installed in the pinned encoder environment (CSV/JSON history is complete).
