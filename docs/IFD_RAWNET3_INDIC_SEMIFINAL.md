# IFD RawNet3 and Indic GRU semifinal

Two independent spoof detectors were evaluated. These are experiment candidates;
the existing service has not been switched to these checkpoints.

| Candidate | Selected epoch | Calibration threshold | Final test clips | Accuracy | EER |
|---|---:|---:|---:|---:|---:|
| IFD RawNet3 | 37 of 50 | 0.6729692220687866 | 1,903 | 87.49% | 14.17% |
| Frozen IndicWav2Vec + GRU | 3 of 10 | 0.4190840721130371 | 5,670 | 98.69% | 0.99% |

Thresholds were selected on calibration data before final testing. Both test
sets are now exposed benchmarks and must not be called untouched holdouts.
The IFD experiment did not reach the earlier 6.1% EER target on its final test.
Indic results are aggregate over nine evaluated languages. Bengali, Malayalam
and Odia have no held-out evaluation. Some cohorts have few voices and synthetic
generator coverage is narrow; corpus and encoding shortcuts remain possible.

## Tracked experiment files

- `artifacts/rawnet-reproduction`: preparation, 50-epoch training, training-only
  BatchNorm recalibration, threshold calibration and reserved-test scripts.
  `results-summary.json` retains compact final metrics and checkpoint identity.
- `artifacts/regional-vnext-run`: disk-backed frozen feature extraction, GRU
  training, memory-safe resumption and final calibration/test scripts. The
  training settings, epoch history and result JSON files bind local inputs.
- `src/backbones/trainable_indic_wav2vec.py`: optional encoder adaptation interface;
  the selected Indic run used the frozen encoder. See `trainable-indicwav2vec.md`.

The experiment scripts use the original workspace artifact layout. Dataset
manifests, prepared audio, feature caches, environments and model weights are
local assets and are not included in Git. They must be supplied separately to
reproduce the runs. Original absolute paths in result provenance describe the
machine on which the experiments ran.

RawNet3 model source is from Jungjee/RawNet commit
`fd6121db44d994dce476bf7ad4a8ef067f841958`, with its MIT license retained.
The local RawNet dependency is `asteroid-filterbanks==0.4.0`; dependencies are
not vendored. Existing project requirements supply the encoder environment.

## Local calibrated checkpoints

- IFD: `artifacts/rawnet-reproduction/final/rawnet3_calibrated.pt`
  SHA-256 `3214b36343dbc20b165df8578c274d86a0083bd27e2976d1ac84ac3fe75270be`.
- Indic: `artifacts/regional-vnext-run/final-evaluation/gru_calibrated.pt`.
  Its SHA-256 and complete calibration/test metrics are in the tracked
  `artifacts/regional-vnext-run/final-evaluation/result.json`.

Both use class 0 genuine, class 1 synthetic. The Indic checkpoint contains the
training-only feature standardizer and requires the pinned final-layer
IndicWav2Vec encoder. It does not contain that pretrained encoder's weights.

## Validation

Before branch publication: repository suite 302 passed, 8 skipped. The training
smoke verified padding invariance, finite gradients and checkpoint roundtrip.
The final Indic calibrated checkpoint preserved all parameter tensors and
reproduced the final replay batch exactly. Large-asset experiment reproduction
is separate from the unit suite.
