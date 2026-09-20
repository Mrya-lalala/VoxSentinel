# C1 inference service

C1 exposes the v3 research detector over HTTP. Use the release manifest to keep
its encoder, detector, preprocessing and window settings together. This remains
an uncalibrated research baseline; deployment does not expand its validation scope.

From the repository root, install the runtime with
`bash scripts/setup_encoder_env.sh`. Obtain the weights as described in the
[teammate quickstart](TEAMMATE_RESEARCH_BASELINE.md), then run:

```bash
INFERENCE_RELEASE=artifacts/releases/indic-gru-v3-research/release.json \
ENCODER_CHECKPOINT=models/indicwav2vec_large.pt \
.venv/bin/python -m uvicorn src.service.api:app --host 127.0.0.1 --port 8000
```

```bash
curl http://127.0.0.1:8000/health
curl -F 'file=@sample.wav;type=audio/wav' http://127.0.0.1:8000/detect
```

## Contract and preprocessing

`GET /health` returns status, model/encoder identifiers and detector readiness.
Check `detector_ready`: HTTP 200 alone does not indicate inference readiness.
`POST /detect` accepts multipart field `file`, a `.wav` filename and WAV MIME type
(`audio/wav`, `audio/x-wav`, `audio/wave`, or absent). It returns status, spoof
score, decision, threshold and its source, chunk counts and coverage notes.
Invalid uploads return 400, oversized uploads 413, unavailable models 503 and
inference/configuration failures 500. Short/silent audio returns 200 with
`insufficient_speech`, null score and null decision. `/docs` exposes OpenAPI.

The trained path is `voxsentinel-prep-2`: `decode_mono_16k` decodes float32,
mean-downmixes channels, resamples with soxr HQ and attenuates the entire clip
by `1/peak` only if its peak exceeds 1. No clipping, loudness boost or denoising.
The encoder alone performs checkpoint-required waveform normalization.

C1's `one-selected-window-v1` contract calls the same `select_window` as file
prediction. With the v3 release it selects one highest-energy window of at most
4 seconds (0.25-second scan, including the flush tail), rejects clips shorter
than 1 second and applies the release's activity check. The complete `WindowPolicy`
is read from `release.json`; legacy checkpoint-only mode uses code defaults.
The independent legacy A2 array/chunk API still has its truthful `audio-v1`
identity and scipy/range policy. It is **not** the trained C1 path and has not
been relabeled as `voxsentinel-prep-2`.

Successful requests produce one score. `coverage_ratio=1` means the selected
window was processed, not that every part of the recording was assessed.
Coverage notes identify the contract and selected sample range. Mean/max/top-k
aggregation remains available for response compatibility, but with one selected
window all yield the same score. This changes the earlier C1 whole-file chunking
behavior; clients must not interpret the response as whole-file localization.

## Environment and threshold provenance

| Variable | Meaning/default |
|---|---|
| `INFERENCE_RELEASE` | Recommended: release JSON; verifies head hash, encoder training identity, supported preprocessing and CPU policy. Takes precedence over `DETECTOR_CHECKPOINT`. |
| `ENCODER_CHECKPOINT` | Required local trusted encoder path; release mode verifies its SHA-256 on load. |
| `DETECTOR_CHECKPOINT` | Legacy trusted head path when no release is set; without it detection is unavailable. |
| `DEVICE` | `cpu`; research release mode rejects other devices. |
| `MODEL_ID` | Legacy display label, default `voxsentinel-gru`; release mode uses release ID. |
| `ENCODER_ID` | Display label, default `indicwav2vec`. |
| `DETECTION_THRESHOLD` | Optional finite number in [0,1]; blank means no override. |
| `MAX_UPLOAD_BYTES` | Default 26214400 (25 MiB); upload file cap. |
| `AGGREGATION_METHOD` | Default `mean`; also `max` or `topk_mean`. |
| `AGGREGATION_TOP_K` | Default 3. |

Threshold precedence is operator override → release threshold (release mode),
or operator override → valid checkpoint metadata threshold → default 0.5
(legacy mode). Responses report `configured`, `release`, `checkpoint` or
`default` respectively. The released 0.5 threshold is **fixed, uncalibrated**,
not a threshold optimized on validation data. Score is softmax class 1;
`score >= threshold` means spoof.

An explicitly incompatible checkpoint preprocessing identity fails loading.
Legacy heads with no identity emit a warning: compatibility is unverified.
Use release mode to verify the full encoder identity and window configuration.
Startup failure leaves the API degraded and `/detect` unavailable.

## Dependencies and checks

`requirements.txt` declares dataset acquisition's `requests` and
`huggingface_hub`. Optional WavLM support uses `requirements-backbones.txt`
(`transformers`); it is not needed by the frozen IndicWav2Vec release.
The setup script installs audio, inference, service and pytest requirements and
runs the full suite. Install optional WavLM with
`python -m pip install -r requirements-backbones.txt` after the base runtime.

The `Preprocessing parity / preprocessing-parity` CI job checks legacy A2 and C1
contracts on every PR and main push. Its offline scan covers six sample rates,
stereo, overflow, short/silent and long recordings, and compares exact encoder
inputs, window boundaries and scores against file prediction. It also checks
release window overrides and preprocessing identity rejection. No model assets
are required. Repository maintainers can mark this job required in branch
protection; adding a workflow alone does not change GitHub protection settings.

Local verification on 2026-09-20: 245 tests passed, 5 asset-dependent tests
skipped in the isolated checkout. Dataset acquisition imports, optional WavLM
construction and `pip check` passed in a temporary environment extending the
pinned runtime. This was not a fresh bootstrap installation. With the actual
v3 encoder/head, HTTP `/detect` matched `FilePredictor` exactly (score error
0.0) on one known development clip and a derived long 44.1 kHz stereo clip
with overflow. These are integration checks, not additional accuracy results.
GitHub-hosted execution and branch-protection enforcement remain pending.
