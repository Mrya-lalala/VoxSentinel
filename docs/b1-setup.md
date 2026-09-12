# Frozen IndicWav2Vec encoder: setup and verification

Implementation and verification completed on 2026-09-12 for the official
multilingual **pretrained Large acoustic checkpoint**. No ASR decoder is invoked.
Routing and the existing single-chunk interface remain in place. No fallback
encoder is installed by this setup. No detector is implemented.

## Reproduce the environment

Verified on Python 3.10.18, macOS 26.5.2 arm64, Apple M3 (8 CPU cores), 16 GiB RAM.
Install Python 3.10 and Apple's command-line build tools before running:

```sh
bash scripts/setup_encoder_env.sh
source .venv/bin/activate
```

The script installs `requirements-bootstrap.txt`, then `requirements.txt` with
`--no-build-isolation`, runs `pip check`, and runs the 14 unit tests. It was also
run successfully into a fresh `/private/tmp/voxsentinel-clean-env` environment;
that environment passed the full real-checkpoint verification below.

Core pins: PyTorch/torchaudio 2.2.2, NumPy 1.23.5, Hydra 1.0.7, OmegaConf 2.0.6,
SoundFile 0.12.1, pip 24.0, setuptools 69.5.1, wheel 0.43.0, Cython 0.29.37.
All runtime transitive versions are recorded in `requirements.txt`.
The legacy OmegaConf metadata requires pip <24.1; do not blindly upgrade pip in
this environment. Fairseq's tensorboardX suggestion and weight_norm deprecation
warning do not prevent extraction; tensorboardX is not required for this path.

Fairseq is built from [AI4Bharat's fork](https://github.com/AI4Bharat/fairseq-indicwav2vec-v1/tree/d11b8fb30a6f0156c2c1152644221ac4031cadcc),
commit `d11b8fb30a6f0156c2c1152644221ac4031cadcc` (package version 0.12.1).
The [official IndicWav2Vec setup](https://github.com/AI4Bharat/IndicWav2Vec#setting-up-your-environment)
links to this fork through its former `AI4Bharat/fairseq` name. Source licenses
are available in the upstream repositories. No third-party source or weights
are vendored in this repository.

`.python-version` records the interpreter; it does not install or activate it.
Point your editor at `.venv/bin/python`. This dependency file covers the B1
Indic encoder; it is not a complete application or fallback runtime.

## Checkpoint identity

[Official model listing](https://github.com/AI4Bharat/IndicWav2Vec#download-models)
provides the multilingual pretrained Large model:

- URL: https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt
- Local path: `models/indicwav2vec_large.pt`
- Size: 3,808,863,698 bytes
- Recorded SHA-256: `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`

This is the digest of the official URL downloaded during verification, not an
upstream-published signature. The configured loader checks it **before** loading
Python checkpoint objects. Only use trusted checkpoints. Weights are Git-ignored
and are never downloaded implicitly by the wrapper.

For a new checkout, download explicitly:

```sh
mkdir -p models
curl -fL --retry 3 \
  https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt \
  -o models/indicwav2vec_large.pt
```

The extraction command below enforces the configured digest. The loader builds
upstream `Wav2Vec2Model` from checkpoint model configuration, strictly loads its
weights, and reads `task.normalize`. Memory-mapped loading avoids materializing
the checkpoint's optimizer state. Training task/data paths are not needed.
A CTC fine-tune, missing normalization declaration, corrupt file, wrong hash,
or unexpected embedding dimension fails explicitly.

## Tensor interface

`src.backbones.tensor_interface.TensorEncoder` describes the common tensor API.
`IndicWav2VecExtractor.extract_batch(samples, valid_lengths, sample_rate=16000)`
returns `EncoderBatch`:

| Field | Shape / meaning |
| --- | --- |
| Input `samples` | Floating tensor `[B,S]`, mono PCM, valid amplitudes in `[-1,1]`, all values finite |
| Input `valid_lengths` | int32/int64 `[B]`, number of valid waveform samples; right-padding only |
| Output `features` | Detached, ordinary float32 tensor `[B,T,D]` on encoder device; padded frames exactly zero |
| Output `valid_lengths` | int64 `[B]`, number of valid **output frames** |
| Output `padding_mask` | bool `[B,T]`: **True = padding/ignore; False = valid** |
| `embedding_dim` | Actual D inferred from loaded encoder; **1024** for this checkpoint |
| `frame_hop_ms` | **20 ms**, from convolution strides |
| `checkpoint_version` | Actual SHA-256 digest; config's optional version label is descriptive only |
| `output_layer` | Selected zero-based transformer block, or None for final output |

The original `extract(numpy_waveform, ChunkMetadata)` remains available for the
router and returns NumPy `[T,D]` in `EmbeddingSequence`. Its duration metadata
must match the sample count (within one sample). Batch callers retain their own
source/chunk metadata alongside the unchanged input/output batch order.

`expected_embedding_dim` is an optional configuration assertion, not a projection.
Do not hard-code 1024 into the shared interface. A future encoder can expose a
different D. Even equal D does **not** imply that detector weights trained on
IndicWav2Vec transfer unchanged to XLS-R or another encoder.

### Preprocessing ownership and lengths

The audio/calling subsystem owns file decoding, integer PCM scaling, mono
conversion, resampling to 16 kHz, optional audio cleanup, and chunk boundaries.
This wrapper does none of those operations. It owns checkpoint-specific waveform
normalization: this model has `normalize=True`; `F.layer_norm` with its default
`eps=1e-5` is applied independently to each valid utterance. Do not normalize
across clips or include padding in waveform statistics.

Unequal lengths are grouped by exact valid sample count, encoded without input
padding, then padded in feature space. Finite input padding is ignored. This
conservative implementation preserves standalone behavior at the cost of
separate encoder calls for different lengths. It does not promise batch-speedup.

Convolution `(kernel,stride)` sequence is `(10,5), (3,2) x4, (2,2) x2`.
The minimum is **400 samples (25 ms)** for one frame; the total stride is 320
samples. For this checkpoint, valid frame count is
`floor((N - 400) / 320) + 1`. Thus 48,000 samples produce 149 frames, and 64,000
produce 199. The technical minimum is not a recommendation for usable speech
quality. Silence gives finite embeddings; this subsystem makes no risk decision.
Contextual embeddings use attention over the full supplied chunk.

### Layer selection and freezing

Configure `output_layer: null` for the final encoder representation (recommended
starting contract). Integers **0 through 23** select the output of that zero-based
transformer block using upstream `extract_features(..., layer=index)`.
All selections return D=1024. Because `layer_norm_first=True`, None applies the
final encoder layer norm; **23 is not equivalent to None**. All 25 selections
(None and 24 block indices) matched direct upstream calls exactly in verification.

Every parameter has `requires_grad=False`; extraction reasserts eval mode and
uses `mask=False` (no random pretraining masks). Encoder computation uses FP32
with autocast disabled and no gradients. Returned tensors can still be inputs
to a trainable head's backward pass, including when extraction was called inside
an outer inference-mode/autocast context. No encoder or input gradients are kept.
The caller owns the downstream optimizer and must honor the output padding mask.

## Runnable extraction

Use local mono 16 kHz speech files:

```sh
python -m scripts.extract_indic --audio clip1.wav clip2.wav
python -m scripts.extract_indic --audio clip1.wav --output-layer 11
python -m scripts.extract_indic --audio clip1.wav --output artifacts/features.pt
```

The example reads only the Indic configuration, loads no fallback, and prints
shape, lengths, checkpoint digest and timing. To call from Python:

```python
from pathlib import Path
import torch
import yaml
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor

config = yaml.safe_load(Path('configs/backbones.yaml').read_text())['indic_wav2vec']
encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**config)).load()
# waveform_a and waveform_b: caller-provided 1-D floating PCM tensors at 16 kHz.
lengths = torch.tensor([waveform_a.numel(), waveform_b.numel()])
samples = torch.nn.utils.rnn.pad_sequence([waveform_a, waveform_b], batch_first=True)
batch = encoder.extract_batch(samples, lengths)
# batch.features [B,T,D]; batch.valid_lengths [B]; batch.padding_mask [B,T]
```

## Verification fixtures and commands

Real-speech checks used two English recordings, `1272-128104-0000` and
`1272-128104-0001`, from the [Hugging Face LibriSpeech dummy validation fixture](https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy).
They are speech fixtures, not a training/evaluation dataset for this project.
No audio is committed. The optional preparation script checks archive and audio
hashes, extracts only these two recordings, and performs no resampling.

```sh
mkdir -p artifacts
curl -fL \
  https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/resolve/main/clean/validation-00000-of-00001.parquet \
  -o artifacts/speech.parquet
python -m pip install -r requirements-fixtures.txt
python -m scripts.prepare_encoder_fixtures --parquet artifacts/speech.parquet
python -m scripts.verify_indic \
  --audio artifacts/speech/1272-128104-0000.flac artifacts/speech/1272-128104-0001.flac \
  --output artifacts/encoder-verification.json
```

Archive SHA-256: `4e69a06fa5edc90921e5e7e39a7084881f8b3ed9c805c574f4f39c6fde27c603`.
Audio SHA-256 values and IDs are pinned in `scripts/prepare_encoder_fixtures.py`.

## Recorded results and limits

14 unit tests passed in both the project and clean environment. They exercise a
tiny random upstream model, including dynamic D, checksum/corruption failures,
invalid layers, padding, frozen parameters, and the existing router contract.
Separately, the real Large checkpoint passed:

- Two full speech clips: `[2,292,1024]`, valid lengths `[292,240]`.
- 4s/3s/4s batch: `[3,199,1024]`, lengths `[199,149,199]`; zero output padding,
  correct mask, exact standalone parity and invariance to changed input padding.
- All layers matched direct upstream output; repeated extraction was exact.
- A disposable linear probe got nonzero finite gradients and updated its own
  weights. The complete encoder state hash stayed unchanged, all encoder
  gradients stayed None, and input gradients stayed None.
- Wrong rank/rate, integer PCM, nonfinite/out-of-range samples, empty batches,
  invalid length shape/type/bounds and too-short clips were rejected. Minimum
  length, silence, and the existing NumPy bridge produced valid outputs.

Final clean-environment CPU measurements, four PyTorch threads, FP32, one warm-up
per case followed by five timed repetitions:

| Audio per call | Median | Observed min–max | Median time / total audio duration |
| --- | ---: | ---: | ---: |
| One 3s clip | 187.3 ms | 185.1–189.9 ms | 0.0624 |
| One 4s clip | 246.1 ms | 242.2–248.5 ms | 0.0615 |
| Unequal batch: 4s + 3s | 439.1 ms | 433.2–445.1 ms | 0.0627 |
| Equal batch: 4s + 4s | 474.8 ms | 468.5–479.6 ms | 0.0594 |

Load including SHA-256 verification took 7.78s in that process (OS file cache may
be warm). Peak process RSS across the full verification was 2,702,393,344 bytes
(~2.52 GiB), including model-state hashing and test work, not an isolated
steady-state inference measurement. Full raw results remain locally in
`artifacts/encoder-verification-clean.json` (Git-ignored).

These measurements cover in-memory waveform validation, normalization and
encoder extraction, excluding file decoding, audio collection, network transfer,
detector work and end-to-end orchestration. Only this CPU platform and short
fixtures were verified. MPS/CUDA, concurrent serving, long recordings, mobile,
ONNX, language robustness and spoof-detection quality remain unverified. The
English fixtures establish encoder mechanics, not Indian-language performance.
