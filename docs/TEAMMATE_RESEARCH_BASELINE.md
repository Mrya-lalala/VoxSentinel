# Teammate quickstart: Indic GRU research baseline

Use branch `feature/a1-datasets-manifests`. This is a local CPU, single-window research detector. It is not a production fraud service, a streaming detector or a validated Indian-English model.

## Get the code and runtime

```sh
git clone --branch feature/a1-datasets-manifests https://github.com/Mrya-lalala/VoxSentinel.git
cd VoxSentinel
bash scripts/setup_encoder_env.sh
```

Install Python 3.10 and platform build tools first. The script installs the pinned encoder runtime, inference resampler and test dependencies, then checks dependencies and runs tests. Development validation used macOS arm64; clean dependency installation on other platforms has not been verified. Do not upgrade the legacy Fairseq/Hydra/OmegaConf stack independently. Full setup and the official encoder download are in [encoder setup](b1-setup.md).

## Obtain model artifacts

Weights are deliberately outside Git. The repository owner has the handoff file **`voxsentinel-indic-gru-v3-research.tar.gz`**, produced at `artifacts/handoff/` in the training workspace. Request that exact archive from the owner through the team's approved file-sharing channel. No public weights URL or GitHub release is claimed. The owner should distribute this file alongside the branch for immediate inference use.

The archive includes the detector head, `release.json`, checksums and research-use notes. It contains no encoder or training audio. Verify the archive against the checksum below before extracting it in the repository root:

```sh
shasum -a 256 voxsentinel-indic-gru-v3-research.tar.gz
tar -xzf voxsentinel-indic-gru-v3-research.tar.gz
```

Archive SHA-256: `b7c77b0a1e71484a5a698c991afda72628ca7b23fc7c95e932da2166c491f512`.

Separately obtain `models/indicwav2vec_large.pt` from the official source documented in [encoder setup](b1-setup.md). Expected SHA-256:
`26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`.
The CLI verifies the head and encoder identities and does not download weights automatically.

## Predict

```sh
.venv/bin/python -m scripts.predict_audio \
  --release artifacts/releases/indic-gru-v3-research/release.json \
  --audio /path/to/your/recording.wav \
  --language Hindi \
  --output artifacts/my-prediction.json
```

Use a trusted local recording. `--language` is metadata, not automatic language detection. Multiple paths can follow `--audio`; the encoder is loaded once. Silence/too-short clips return `insufficient_audio`, not a genuine label. A long file contributes only one selected 1–4 second window. The synthetic score is uncalibrated and the fixed threshold is 0.5. Do not equate a genuine result with a guarantee about the complete recording or its speaker's identity.

## Evidence and limits

V3 training: 458 examples, development: 135. Epoch three was selected by development EER. Development accuracy 97.04%, EER 2.30%; false alarms 0/87, missed spoofs 4/48. Development was used for selection and is not independent-test evidence. The earlier v2 model scored 79.17% on the evaluated benchmark; that result is NOT the v3 model's score. V3 has not rescored that benchmark.

Read [training results](V3_TRAINING_RESULTS.md) and [prior benchmark results](archive/history/V2_FROZEN_TEST_RESULTS.md). Malayalam/Odia/Bengali male coverage remains limited. Cross-language identities, strict conversion-family isolation, Indian English, production channels and streaming are unvalidated.

This handoff is supplied for **noncommercial research use only**. Retain attribution to the source datasets and encoder and review their applicable terms before any redistribution or broader use. Recorded training provenance includes Kathbath and IndicSynth; IndicSynth's dataset card declares CC BY-NC 4.0. This handoff does not grant commercial rights or override third-party terms. Source references: [IndicSynth](https://huggingface.co/datasets/vdivyasharma/IndicSynth), [Kathbath](https://github.com/AI4Bharat/IndicSUPERB), [IndicWav2Vec](https://github.com/AI4Bharat/IndicWav2Vec). No dataset audio is redistributed in the model bundle.

## Tests and training

```sh
.venv/bin/python -m pytest tests -q
```

Tests requiring gitignored local datasets skip in a fresh source checkout. To reproduce training, obtain the separately documented datasets/caches and follow [v3 cache handoff](archive/history/V3_CACHE_AND_TRAINING_SUPPORT_HANDOFF.md). They are not needed for inference. Do not run acquisition casually: it consumes the shared capped ledger. Historical reports and agent prompts describe previous phases; this page and the final result reports describe the current teammate deliverable.
