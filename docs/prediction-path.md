# Local audio prediction (research pilot)

Run from the repository root with the Python 3.10 encoder environment described in [encoder setup](b1-setup.md). Install the extra pinned resampler:

```sh
.venv/bin/python -m pip install -r requirements-inference.txt
```

The current local package is `artifacts/releases/indic-gru-pilot-v0.1/`. It contains a release manifest and the standardized GRU head. Its head SHA-256 is `44ce5ffebd1b94a5ff34de3dcc08fa6b776005c36064c178aa93c5f22d96a2b5`. The pinned encoder is supplied separately at `models/indicwav2vec_large.pt`; override its location with `--encoder-path` without changing its identity.

```sh
.venv/bin/python -m scripts.predict_audio \
  --release artifacts/releases/indic-gru-pilot-v0.1/release.json \
  --audio path/to/recording.wav \
  --language Hindi \
  --output artifacts/prediction.json
```

`--audio` accepts multiple files; the encoder loads once per process. `--language` is optional metadata, using the full language name. JSON contains the synthetic score, prediction, selected window, input and checkpoint hashes, processing facts, and scope flags. The score is not a calibrated probability of fraud. Silent or too-short inputs return `insufficient_audio` and a null prediction. Invalid inputs return a nonzero exit status with an error on stderr. Batch failure currently aborts the batch.

This model was trained only on the twelve Indic languages listed in its manifest; Indian English remains unvalidated. A language outside that list is flagged but still scored. Long files contribute one high-energy window, so this is not a whole-recording guarantee. CPU execution is the validated path.

To regenerate a package from the existing local training run (choose an empty destination):

```sh
.venv/bin/python -m scripts.package_pilot_baseline \
  --out-dir artifacts/releases/indic-gru-pilot-local
```

To check raw-audio/cached-feature parity and invalid-input handling with the local pilot data:

```sh
.venv/bin/python -m scripts.verify_prediction_path
```

A fresh clone does not contain the gitignored encoder, run, release or validation audio. Supply authorized artifacts before running these commands. Packaging does not download them. This pilot includes training data with noncommercial terms; no commercial clearance is asserted. See [dataset and release readiness](BASELINE_DATASET_AND_PREDICTION_REPORT.md).
