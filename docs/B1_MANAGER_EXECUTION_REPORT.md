# VoxSentinel B1 — frozen IndicWav2Vec execution report

Prepared for the manager agent from the local execution record dated 2026-09-12.
This report describes implemented behavior, measured verification, and residual
requirements. It does not assert completion of the full detection pipeline.

## 1. Outcome and acceptance status

The official multilingual pretrained IndicWav2Vec Large checkpoint was downloaded,
loaded with strict weight compatibility, and exercised on real speech on the
user's Mac. The frozen wrapper returns contextual tensor features, valid output
lengths, and an explicit downstream padding mask. A clean environment reproduced
the installation and passed the real-checkpoint verification.

The encoder is ready for independent head-interface integration on the verified
CPU configuration. **Do not mark every original requirement unconditionally
complete:** upstream intermediate-state retention has not been removed, and an
actual parent detector's training lifecycle has not been integrated/tested.

| Requirement | Evidence / status |
| --- | --- |
| Real multilingual pretrained checkpoint | Passed: official Large checkpoint; strict state loading; configured SHA-256 enforcement |
| Mono, resampling, normalization ownership | Explicit: caller owns decoding/mono/resampling; wrapper alone owns checkpoint waveform normalization. Upstream audio implementation was not audited |
| 16 kHz boundary | Enforced; wrong declared sample rate rejected |
| No default denoising | No denoising added |
| Frozen encoder | Passed: all parameters frozen; full state hash unchanged after downstream update |
| Evaluation during extraction | Passed: extraction restores eval after deliberate `model.train()` |
| Parent-detector training integration | Not built; wrapper is not an `nn.Module`; see section 5 |
| Downstream gradients | Passed: disposable linear probe received finite nonzero gradients and updated; encoder/input gradients remained None |
| Contextual `[B,T,D]` and `[B]` lengths | Passed; actual D=1024, configurable expected dimension |
| Actual downsampling lengths | Derived from loaded Conv1d geometry; verified against real outputs |
| Encoder masks and downstream lengths | Exact-length groups avoid waveform padding; upstream mask=None; downstream lengths and mask always returned |
| Final/intermediate layer selection | Passed: None and indices 0–23 match direct upstream calls |
| Avoid unnecessary all-layer retention | Partially met: no extra forward passes or all-layer consumer output, but Fairseq internally retains `layer_results` |
| Invalid inputs | Passed focused checks; see section 6 |
| Real-speech batching and repeatability | Passed with quantified zero differences for tested exact-length-group policy |
| Measured runtime | Recorded CPU-only measurements; no GPU or end-to-end measurements |

## 2. Repository and transfer state

- Project: VoxSentinel, SIH26104, anti-spoofing/impersonated-speech risk assessment.
- Repository: https://github.com/Mrya-lalala/VoxSentinel.git
- Local checkout: `/Users/mouryabs/VoxSentinel`.
- Branch: **`feature/indicVac2Wav`**, preserving spelling and case.
- Baseline before this update: `8328fa290e9b9d0fc7104b2ce975a114add4a4b0`.
- This report accompanies the encoder implementation in the feature-branch
  update authorized for commit and push by the user. The baseline above is not
  the implementation commit; use `git log -1 -- docs/B1_MANAGER_EXECUTION_REPORT.md`
  to identify the commit containing this report in your checkout.
- No PR, merge to main, or message to teammates is part of this update.
- Another machine/agent must fetch and check out `feature/indicVac2Wav` and verify
  that the files listed below are present. Main must not be assumed to contain them.
- Model weights, environments, fixtures, feature exports and raw generated reports
  are Git-ignored. The documents preserve the essential measured results.

### Scope held

Work was limited to the Indic encoder, its batch tensor contract, relevant config,
setup, tests, examples and documentation. Routing code was not changed. Existing
WavLM code/config was preserved, but its runtime was not installed or exercised.
XLS-R was neither implemented nor loaded. No GRU, AASIST, RawNet2 adapter, detector
training, scoring, calibration, aggregation, API, streaming or mobile component
was implemented. The only trainable layer was a disposable verification probe.

## 3. Artifact inventory

Paths are relative to the repository root.

| File | Purpose |
| --- | --- |
| `src/backbones/indic_wav2vec.py` | Real checkpoint loader, normalization, freezing, grouped batch extraction, legacy single-clip bridge |
| `src/backbones/tensor_interface.py` | Structural `TensorEncoder` protocol and `EncoderBatch` dataclass |
| `configs/backbones.yaml` | Indic checkpoint SHA-256, expected dimension, CPU device and layer setting; routing/fallback sections preserved |
| `requirements.txt` | Exact resolved encoder runtime dependencies and pinned upstream Git commit |
| `requirements-bootstrap.txt` | Legacy native-extension build prerequisites and pip version |
| `requirements-fixtures.txt` | Optional PyArrow for unpacking verification audio only |
| `scripts/setup_encoder_env.sh` | Reproducible Python 3.10 setup, dependency check and unit tests |
| `scripts/extract_indic.py` | Runnable local-file extraction example with optional tensor export |
| `scripts/prepare_encoder_fixtures.py` | Hash-check and unpack two public speech recordings |
| `scripts/verify_indic.py` | Real-checkpoint parity, gradient, validation and CPU timing checks |
| `tests/backbones/test_indic_encoder.py` | Nine tests with a tiny random upstream model; complements five existing tests |
| `.python-version` | Records Python 3.10.18 |
| `.gitignore` | Excludes local environments, models, artifacts, Python caches and secrets |
| `docs/b1-setup.md` | Setup, API semantics, reproduction commands and measured results |
| `docs/B1_TEAM_AGENT_HANDOUT.md` | Self-contained consumer instructions for teammate agents |

Local evidence files:

- `artifacts/encoder-verification-clean.json`: authoritative final clean-env run.
- `artifacts/encoder-verification-clean.log`: associated execution log.
- `artifacts/encoder-verification.json`: earlier project-env real verification.
- `artifacts/example-extraction.json`: saved single-clip example metadata.
- `artifacts/example-features.pt`: checked `[1,292,1024]` tensor export.

These local artifacts are excluded from Git. Do not assume they accompany a code
checkout; regenerate with the provided commands or transfer them separately.

## 4. Exact model and runtime identity

Model: official **IndicWav2Vec Large multilingual pretrained acoustic checkpoint**,
not a language-specific CTC/ASR fine-tune.

- Official listing: https://github.com/AI4Bharat/IndicWav2Vec#download-models
- Download URL: https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt
- Local path: `models/indicwav2vec_large.pt`.
- File size: **3,808,863,698 bytes**.
- SHA-256: **`26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`**.
- Digest was computed from the downloaded official file, not obtained as an
  upstream-signed verification statement.
- Fairseq fork: https://github.com/AI4Bharat/fairseq-indicwav2vec-v1
- Fairseq commit: **`d11b8fb30a6f0156c2c1152644221ac4031cadcc`**.
- Installed Fairseq package version: **0.12.1**.

The loader hashes the local file before deserialization, compares the configured
hash, uses `torch.load(..., mmap=True, weights_only=False)`, merges checkpoint
model settings with upstream `Wav2Vec2Config` defaults, constructs upstream
`Wav2Vec2Model`, and strictly loads the state dictionary. It checks the model
identity, normalization setting, selected layer, optional expected D, and
supported convolution geometry. Memory mapping avoids materializing optimizer
state. Training-task setup and original dataset paths are not needed.

Checkpoints contain Python configuration objects; this is a trusted-checkpoint
path, not a general untrusted-model ingestion service. The wrapper never fetches
weights automatically. Upstream licenses remain linked in the source repositories;
no weights or upstream source tree are committed here.

### Verified hardware and software

| Item | Recorded value |
| --- | --- |
| CPU | Apple M3, 8 physical cores |
| RAM | 17,179,869,184 bytes / 16 GiB |
| OS | macOS 26.5.2, build 25F84, arm64 |
| Python | 3.10.18 |
| PyTorch / torchaudio | 2.2.2 / 2.2.2 |
| NumPy | 1.23.5 |
| Hydra / OmegaConf | 1.0.7 / 2.0.6 |
| SoundFile | 0.12.1 |
| pip / setuptools / wheel / Cython | 24.0 / 69.5.1 / 0.43.0 / 0.29.37 |
| Encoder device and computation | CPU, FP32, 4 PyTorch threads for real verification |

The complete transitive dependency list is in `requirements.txt`. Build uses
bootstrap installation followed by `--no-build-isolation`. pip 24.0 accommodates
the legacy OmegaConf metadata rejected by pip >=24.1. Fairseq's tensorboardX
suggestion and weight_norm deprecation warning were observed; neither prevented
the verified extraction path. No tensorboardX, ASR decoder, KenLM, or Transformers
installation was needed for this Indic-only execution.

## 5. Implemented encoder behavior

### Audio ownership

Caller supplies floating mono PCM `[B,S]` at a declared 16 kHz and integer valid
sample counts `[B]`. It owns decoding, scaling integer PCM, mono conversion,
resampling, and chunking. The wrapper cannot infer whether an upstream caller
already applied normalization; callers must follow the documented ownership
contract to avoid duplicate processing. No upstream audio pipeline was audited.

The wrapper rejects incompatible declarations/shapes and performs exactly one
checkpoint-specific normalization operation per encoded group: `F.layer_norm`
over each clip's valid sample axis, default epsilon `1e-5`. Normalization is
independent per clip and excludes padding. The checkpoint's `task.normalize`
is True; feature extractor mode is `layer_norm`. No denoising was introduced.

### Tensor output and length geometry

Output `EncoderBatch` contains float32 contextual `features [B,T,D]`, int64
`valid_lengths [B]` in **frames**, bool `padding_mask [B,T]`, frame hop, backbone
ID, actual checkpoint digest, and selected layer. Mask True means ignore/padding.
Padded output features are exactly zero. Batch order is preserved.

D is discovered from the checkpoint and is **1024** here. The configured expected
D is an assertion, not a projection. Model parameter count as loaded, including
retained pretraining components, was **317,390,592**. No text head is invoked.

Actual Conv1d kernel/stride geometry:
`(10,5), (3,2), (3,2), (3,2), (3,2), (2,2), (2,2)`.
Length is computed successively as `(length-kernel)//stride + 1`. Total stride is
320 samples = **20 ms**. Minimum convolution receptive input is 400 samples =
25 ms. For this checkpoint: `floor((N-400)/320)+1` valid frames, for N>=400.
Technical minimum length does not establish speech quality or a detection window.

### Batching and encoder mask

The wrapper groups input clips by exact valid sample count, strips their right
padding, encodes each group, then pads contextual features across groups. Upstream
receives `padding_mask=None` because every clip in an encoder call is unpadded.
Fairseq may manage its own internal alignment padding; the wrapper does not
replace that implementation. A separate downstream feature mask is always returned.

Tested standalone-versus-grouped-batch differences were exactly zero. This result
supports the implemented policy; it does **not** establish that one naive,
unequal-length, waveform-padded upstream call would be equivalent. That alternative
was not benchmarked or compared. The implementation does not overwrite or round
features to manufacture parity. Distinct lengths require separate calls, so
batch speedup is not promised.

### Freezing and downstream training

Loading calls `eval()` and `requires_grad_(False)`. Every extraction reasserts
both. Compute is under `torch.inference_mode(False)`, `torch.no_grad()`, and
autocast disabled. Input is detached; output is an ordinary detached tensor,
not an inference-only tensor that prevents a head from saving inputs for backward.
The outer inference/autocast-context case was explicitly verified.

The wrapper itself is a plain `BackboneExtractor`, **not `torch.nn.Module`**.
A parent detector has not been built. Verification deliberately called the
private underlying model's `train()`, then confirmed extraction restored eval,
repeatability, and frozen gradients. There is no override preventing that private
model from temporarily entering train mode between extraction calls. Consumers
must call the public extraction API rather than the private model directly.
Future real parent-detector integration needs its own lifecycle check.

### Layer selection and the retained-state qualification

`output_layer=None` returns the final encoder output. Integer indices 0–23 select
zero-based transformer blocks using upstream `layer=index`. This checkpoint has
`layer_norm_first=True`: None applies final encoder layer normalization, whereas
23 selects the last block before that normalization. These are distinct contracts.
All 25 selections matched direct upstream calls exactly on the test input.

Normal extraction makes one upstream call per length group and returns only
`result['x']`. However, the pinned upstream transformer loop appends per-layer
values to `layer_results` internally and returns them in the upstream result.
The wrapper does not suppress that retention. Selecting an earlier block stops
later blocks, but avoiding all unnecessary intermediate-state retention remains
an optimization requirement, not a completed claim. This report corrects any
broader impression from earlier completion summaries. No code was changed to
address that point during report preparation.

### Existing architecture compatibility

The old `extract(numpy_waveform, ChunkMetadata)` entry point still returns
`EmbeddingSequence [T,D]` and route `indic`. It validates duration against sample
count to within one sample, returns the actual frame hop, and records the real
checkpoint hash as version. Batch metadata remains caller-owned. Router and
fallback selection logic were not modified.

## 6. Verification evidence

### Fast tests

**14 unit tests passed** in both the project environment and a freshly installed
clean environment. Five existing tests cover routing/contracts; nine new tests
use a tiny random upstream wav2vec2 model. They validate dynamic dimension,
geometry, loading failures, padding, layer selection, frozen gradients, outer
inference/autocast, invalid inputs, minimum length, and the NumPy bridge. They
are separate from pretrained-model evidence.

### Real speech provenance

Source: https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy
(clean validation fixture, English LibriSpeech speech). These are two recordings
from the same speaker/chapter, not representative language or speaker coverage.

Archive SHA-256:
`4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603`.

| Clip ID | Samples | Duration | Audio SHA-256 |
| --- | ---: | ---: | --- |
| 1272-128104-0000 | 93,680 | 5.855 s | `4e25e22555cd16e90edb0a3b49fdcf1fe652b2a1250ab643634db33895c75b41` |
| 1272-128104-0001 | 77,040 | 4.815 s | `46a9d58622b4675b29564da2d9ba73e702241c5fa969f12c387cad4aa984276a` |

The PyTorch tutorial audio URL initially returned HTTP 403; the successful
verification used these hash-checked fixtures instead. No audio was committed.

### Focused real-checkpoint outcomes

| Check | Recorded outcome |
| --- | --- |
| Single full clip | `[1,292,1024]`, length `[292]`; saved/reloaded tensor export verified |
| Two full unequal clips | `[2,292,1024]`, lengths `[292,240]` |
| 4s/3s/4s batch | `[3,199,1024]`, lengths `[199,149,199]` |
| Mask/feature padding | bool mask matches lengths; padded features exactly zero |
| Single versus grouped batch | Maximum absolute errors `[0.0,0.0,0.0]` |
| Changed waveform padding | Replacing finite padding +77 with -77 left output exactly unchanged |
| All layer selections | Maximum absolute error 0.0 for None and each block 0–23 against direct upstream |
| Repeatability | Exact repeated features after deliberate training-mode change |
| Downstream probe | Finite nonzero gradients; own weights updated |
| Encoder freezing | All encoder gradients None; all parameters frozen; full state hash unchanged |
| Input differentiation | Input gradient None |
| Outer inference + autocast | Output ordinary FP32 tensor; downstream backward worked |
| Invalid input cases | 12 cases rejected in real-checkpoint verification |
| Minimum and silence | 400 samples yielded one finite frame; silence yielded finite embeddings |
| Legacy bridge | NumPy output exactly matched tensor extraction |

Invalid real cases covered wrong rank, integer PCM, zero/oversized/too-short
lengths, noninteger lengths, incorrect length shape, empty batch, NaN, infinity,
amplitude outside valid PCM range, and wrong sample rate. Tiny-model tests also
covered missing/corrupt files, wrong hash, wrong expected D and out-of-range layer.
Not every adversarial tensor/device/config combination has been exhaustively tested.

## 7. Measured runtime

Authoritative final run: `artifacts/encoder-verification-clean.json`.
CPU FP32, four PyTorch threads, one untimed warm-up per case, five repetitions.
Timing covers in-memory boundary validation, normalization and encoder extraction.

| Samples per call | Audio | Median | Min–max | Median real-time factor (total audio) |
| --- | --- | ---: | ---: | ---: |
| [48000] | 3s | 187.259 ms | 185.064–189.934 ms | 0.062420 |
| [64000] | 4s | 246.116 ms | 242.171–248.521 ms | 0.061529 |
| [64000,48000] | 4s + 3s | 439.115 ms | 433.215–445.116 ms | 0.062731 |
| [64000,64000] | 4s + 4s | 474.822 ms | 468.514–479.577 ms | 0.059353 |

Individual run durations in milliseconds:

- 3s: 187.435, 186.781, 187.259, 185.064, 189.934.
- 4s: 242.171, 246.116, 246.673, 248.521, 243.083.
- 4s+3s: 433.215, 438.091, 439.115, 445.116, 441.303.
- 4s+4s: 468.514, 472.613, 474.822, 478.659, 479.577.

Load time including checkpoint hashing: **7.784 s**; OS file cache may have been
warm. Peak process RSS over the entire verification: **2,702,393,344 bytes**
(~2.52 GiB). This includes model-state hashing and test work; it is not an isolated
steady-state model-memory measurement. The separate single full-clip example
recorded 390.756 ms extraction, but that is one run, not a benchmark median.

Excluded: file decoding, live audio collection, network transfer, detector work,
aggregation and end-to-end orchestration. A 3-second live window still needs to
be collected; these numbers do not establish live alert latency.

The user's Ryzen 9900X / DDR5-6000 CL30 / RTX 4080 Super PC was discussed only.
No code ran there and no speedup was measured. CUDA acceleration is an expectation
requiring validation, not an execution result. The current verification script
explicitly supports CPU timing only; GPU measurements need synchronization and
correctness checks before reporting numbers.

## 8. Reproduction and transfer

From the repository root on a compatible Mac:

```sh
bash scripts/setup_encoder_env.sh
source .venv/bin/activate
mkdir -p models
curl -fL --retry 3 \
  https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt \
  -o models/indicwav2vec_large.pt
python -m scripts.extract_indic --audio /path/to/mono-16k-speech.wav
```

The configured loader checks the checkpoint digest. See `docs/b1-setup.md` for
hash-checked public fixture download/preparation commands, then run:

```sh
python -m scripts.verify_indic \
  --audio artifacts/speech/1272-128104-0000.flac artifacts/speech/1272-128104-0001.flac \
  --output artifacts/encoder-verification.json
```

Raw reports are generated locally, not committed. Archive essential results with
code revision, config, dependency identity and hardware when repeating elsewhere.

## 9. Remaining work and manager guidance

1. Treat this as a verified CPU encoder subsystem, not completed detector integration.
2. If strict avoidance of all-layer retention is an acceptance gate, address the
   upstream `layer_results` behavior and rerun parity/memory measurements.
3. Verify the actual parent training lifecycle once a detector team integrates
   through the public API. Do not bypass the wrapper's freezing behavior.
4. Audit caller-side mono conversion/resampling/normalization ownership when the
   audio subsystem is connected; current evidence covers the encoder boundary.
5. Test additional speech, especially Indian languages, if making robustness
   claims. No spoof accuracy, language coverage or calibration was measured.
6. Validate CUDA/MPS/Windows/Linux separately. The legacy dependency pins and shell
   installer were verified on macOS arm64 only. Long audio, concurrency, ONNX,
   mixed-precision encoder execution and mobile remain unverified.
7. Shared `[B,T,D]` interface does not make IndicWav2Vec-trained head weights
   compatible with XLS-R, even if dimensions happen to match. Preserve checkpoint,
   selected layer, normalization and mask/length semantics in downstream configs.
8. Keep deferred fallback work and detector work with their owners. Preserve the
   feature-branch workflow and teammates' changes; a branch update is not a merge
   into main or a deployment.
