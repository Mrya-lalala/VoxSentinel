# VoxSentinel — IndicWav2Vec encoder handout for teammate agents

Use this document to integrate your team's components with the verified B1
encoder. The execution was verified on 2026-09-12. The detailed evidence and
remaining acceptance qualifications are in `B1_MANAGER_EXECUTION_REPORT.md`.

## What you are receiving

A frozen, real **IndicWav2Vec Large multilingual pretrained acoustic encoder**.
It consumes mono waveform samples and emits contextual acoustic features. It does
not transcribe speech, identify a speaker, assign spoof probabilities, or decide
whether audio is genuine. No detector or fallback is loaded by the example below.

The wrapper, setup and tests accompany this handout in the user-authorized update
to `feature/indicVac2Wav` in https://github.com/Mrya-lalala/VoxSentinel.git.
Baseline HEAD was `8328fa290e9b9d0fc7104b2ce975a114add4a4b0`; it does not identify
the implementation commit. Fetch and check out the feature branch, then confirm
that your checkout contains the implementation. Use
`git log -1 -- docs/B1_TEAM_AGENT_HANDOUT.md` to identify the handout's commit.
Main is not updated by this branch handoff. Model weights and generated artifacts
are excluded from Git.

## Ownership boundary

| Responsibility | Owner / expected action |
| --- | --- |
| Decode file/container and scale integer PCM | Audio caller; supply valid audio amplitudes in `[-1,1]` (finite right-padding is excluded from this amplitude bound) |
| Convert channels to mono | Audio caller, before the encoder boundary |
| Resample to 16,000 Hz | Audio caller, before the encoder boundary |
| Choose chunk boundaries and maintain provenance | Audio/calling subsystem |
| Checkpoint-specific waveform normalization | **B1 wrapper only**; do not pre-apply it upstream |
| Optional denoising policy | Not introduced here; do not assume denoising occurs |
| Frozen acoustic representation | B1 wrapper |
| Head/adapter architecture, optimizer and output interpretation | Your owning team; outside this B1 implementation |
| Length/mask use in a head | Head owner; never count padded frames as valid audio |
| Risk aggregation, thresholds and calibration | Their owning teams; no outputs supplied by this encoder |

The wrapper cannot detect a falsely declared sample rate or a clip already
normalized by an upstream pipeline. Honor this boundary explicitly. It validates
the declared rate and tensor values/shapes; it does not audit how audio was made.

## Setup and exact identity

Verified configuration: Python **3.10.18**, macOS **26.5.2 arm64**, Apple M3,
16 GiB RAM, CPU FP32. CUDA/MPS and native Windows/Linux installation are not
verified by this handout.

From the repository root:

```sh
bash scripts/setup_encoder_env.sh
source .venv/bin/activate
```

Prerequisites are Python 3.10 and native build tools (Apple command-line tools on
the tested Mac). This script bootstraps build dependencies, installs the pinned
runtime with `--no-build-isolation`, checks dependencies and runs 14 unit tests.
Use `.venv/bin/python` in your editor. Keep pip at 24.0 for the pinned legacy
OmegaConf metadata. Do not substitute a generic latest Fairseq package.

Runtime identity:

- Fairseq: AI4Bharat `fairseq-indicwav2vec-v1`, commit
  `d11b8fb30a6f0156c2c1152644221ac4031cadcc` (package 0.12.1).
- PyTorch and torchaudio: 2.2.2; NumPy: 1.23.5.
- Hydra: 1.0.7; OmegaConf: 2.0.6; SoundFile: 0.12.1.
- Exact remaining pins: `requirements.txt` and `requirements-bootstrap.txt`.
- Optional fixture preparation only: `requirements-fixtures.txt`.

Download the checkpoint explicitly if it is absent:

```sh
mkdir -p models
curl -fL --retry 3 \
  https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt \
  -o models/indicwav2vec_large.pt
```

Identity: **3,808,863,698 bytes**, SHA-256
**`26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`**.
This hash is enforced by `configs/backbones.yaml` before loading. Only load the
trusted official checkpoint: deserialization includes Python configuration objects.
The wrapper never downloads models implicitly. Official model references are at
https://github.com/AI4Bharat/IndicWav2Vec#download-models.

## Consumer contract

Import:

```python
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor
from src.backbones.tensor_interface import EncoderBatch, TensorEncoder
```

`TensorEncoder` is a structural protocol. The wrapper is a plain Python object,
not a `torch.nn.Module`. Use its public API:

```python
encoder.load()  # Optional explicit eager load; otherwise the first call loads.
batch = encoder.extract_batch(samples, valid_lengths, sample_rate=16000)
```

### Device placement and reconstruction

When this plain wrapper is stored as an ordinary attribute of a parent
`torch.nn.Module`, its private encoder is not a registered child module.
Consequently, parent `.to(device)` does **not** move this encoder, and parent
`.state_dict()` does **not** include its weights or `IndicWav2VecConfig`. Loading
the parent's state dictionary alone does not reconstruct the encoder either.
The wrapper exposes no public `.to()`, `.state_dict()`, or `.load_state_dict()`
method.

Set `IndicWav2VecConfig.device` (the YAML `indic_wav2vec.device` setting) before
constructing/loading the wrapper. The loader uses `torch.device(config.device)`
and moves the model there in FP32. **`cpu` is the default and only verified device
configuration.** Other PyTorch device strings are delegated to the backend; CUDA
and MPS execution are unverified and must not be treated as supported by these
CPU checks. There is no automatic device selection or CPU fallback.

Loading is cached: changing/replacing the configuration after a model is loaded
does not move or reload that model. To change device through the public interface,
construct a new wrapper with the intended device and call `.load()`. Extraction
moves each valid waveform group to the actual model-parameter device. Returned
features, frame lengths and mask are on that device; moving a parent head does
not automatically move these outputs. The integrating team must align its head
and encoder/output devices explicitly.

Save an encoder manifest alongside your parent/head state dictionary containing
the checkpoint reference and SHA-256, selected layer, expected embedding
dimension, backbone/version labels and the config needed to reconstruct the
wrapper. Preserve the pinned runtime identity as well. On restore, make the
trusted checkpoint available locally, resolve its path, choose the deployment
device, then reconstruct with
`IndicWav2VecExtractor(IndicWav2VecConfig(**saved_config)).load()`.
The configured SHA-256 is checked again. This reconstruction path restores the
frozen encoder from its external checkpoint; it is independent of loading the
parent/head state dictionary.

### Inputs

| Input | Contract |
| --- | --- |
| `samples` | Floating, finite, dense/strided tensor `[B,S]`; each row represents mono audio |
| `valid_lengths` | int32 or int64 tensor `[B]`; sample counts in each row, not milliseconds or frame counts |
| `sample_rate` | Must be 16000 |
| Padding | Right-padding only; every padding value must be finite, but its magnitude is unrestricted and it is excluded from normalization/encoding |
| Valid amplitude | `[-1,1]` applies only to `samples[i, :valid_lengths[i]]`; integer PCM must be decoded/scaled upstream |
| Minimum length | 400 valid samples for this checkpoint; this is only a convolution limit |

A stereo `[channels,samples]` array is not a batch of mono clips. Convert channels
upstream and add the batch dimension deliberately. Samples are cast to FP32 by
the wrapper. Input lengths need not be on the model device; returned frame lengths,
mask and features are on the actual encoder device. The default and verified
device is CPU.

Validation first checks finiteness across the **entire** input tensor, including
padding, then checks `[-1,1]` only on each valid audio slice. Thus the +77/-77 test
is intentional: those finite values occur exclusively after a row's valid length
and are stripped before normalization and encoding. A value of +77 or -77 inside
valid audio is rejected; NaN or infinity is rejected even in padding. The wrapper's
class-level docstring broadly describes PCM in `[-1,1]`; the precise behavior is
the valid-slice rule above, not an amplitude restriction on finite padding.

### Outputs

| `EncoderBatch` field | Shape / semantics |
| --- | --- |
| `features` | float32 `[B,T,D]`; ordinary detached tensor; zero in padded frames |
| `valid_lengths` | int64 `[B]`, **output frame counts** |
| `padding_mask` | bool `[B,T]`; **True = padding/ignore, False = valid** |
| `frame_hop_ms` | 20.0 for this checkpoint |
| `backbone_id` | `indicwav2vec-large` with the supplied config |
| `checkpoint_version` | Actual checkpoint SHA-256, not the optional descriptive config label |
| `output_layer` | None or selected zero-based block index |
| `embedding_dim` | Property returning actual feature dimension D |

D is **1024** for the verified checkpoint. Discover it from `encoder.embedding_dim`
or `batch.embedding_dim`. `expected_embedding_dim` checks compatibility when
loading; it does not resize or project features. Build your head against its
recorded training encoder and layer. A future encoder can have another D, and
matching D alone never makes head weights interchangeable with XLS-R.

Batch row order is unchanged. Keep source IDs, offsets and other provenance in
parallel caller-owned records; the tensor batch object does not carry chunk
metadata. Existing router users can continue to call
`extract(numpy_waveform, ChunkMetadata)` and receive NumPy `[T,D]`.

## Exact lengths, normalization and batching

For this checkpoint the waveform is normalized with per-clip `F.layer_norm`
over valid samples, epsilon `1e-5`. No normalization across the batch and no
normalization over padding. Do not add another copy of this operation in your head.

Convolution kernels/strides are `(10,5), (3,2) x4, (2,2) x2`. Lengths are computed
from the actual loaded convolution modules. With N valid samples:

```text
valid output frames = floor((N - 400) / 320) + 1, for N >= 400
48,000 samples (3s) -> 149 frames
64,000 samples (4s) -> 199 frames
```

Do not compute frame hop as `duration / frames`. The real hop is 320 samples,
20 ms. Features are contextual over the supplied clip, not independent local
25 ms measurements.

The wrapper groups clips by exact valid sample length, runs each group without
waveform padding (`padding_mask=None` upstream), and then pads the feature
sequences. This is the selected conservative batching policy. There may be
multiple upstream calls for one consumer batch. Do not replace it with one
waveform-padded call without rechecking parity and normalization behavior.

At your head boundary, use valid lengths or the output mask as required by your
implementation. Check that head-library mask polarity matches **True=ignore**.
When a library requires lengths on CPU, transfer the lengths explicitly. Pooling,
loss computation and evidence generation must exclude padded frames. Do not infer
validity from whether a feature vector happens to contain zeros.

## Layers and frozen behavior

- `output_layer: null` is the default final encoder representation.
- `output_layer: 0` through `23` selects that zero-based transformer block.
- None and 23 differ: the checkpoint uses `layer_norm_first=True`, and None
  includes final encoder layer normalization after the last block.
- Record the selected layer with your head's configuration and weights.

All encoder parameters are frozen. Extraction restores eval mode, disables random
pretraining masking, detaches input and performs FP32 computation without encoder
gradients. Ordinary detached output tensors allow your trainable components to
save their inputs for backward. Put **your head's forward and loss/backward outside
any surrounding inference/no-grad scope**; the wrapper only manages its own scope.

Call `extract_batch`, not `encoder._model` directly. The underlying model's mode
can temporarily be changed to train outside extraction, but extraction reasserts
eval. There is no parent-detector module wrapper or custom `train()` override here.
When integrating a real parent model, verify that its training lifecycle still
routes encoder work through this API. A deliberate mode-change simulation passed;
a full parent detector has not been integrated.

The wrapper returns only the chosen contextual tensor. The pinned Fairseq code
still retains intermediate `layer_results` internally during a forward pass;
removing that unnecessary retention remains an optimization item. No extra
per-layer forward passes occur during a normal extraction.

## Runnable first extraction

For your own local mono 16 kHz speech:

```sh
python -m scripts.extract_indic --audio clip1.wav clip2.wav
python -m scripts.extract_indic --audio clip1.wav --output-layer 11
python -m scripts.extract_indic --audio clip1.wav --output artifacts/features.pt
```

This CLI reads only the Indic section of `configs/backbones.yaml`. It rejects
files requiring channel conversion or resampling instead of silently doing it.

The following consumer code uses already-decoded mono tensors from your audio
subsystem. It constructs no detector:

```python
from pathlib import Path
import torch
import yaml
from src.backbones.indic_wav2vec import IndicWav2VecConfig, IndicWav2VecExtractor

settings = yaml.safe_load(Path('configs/backbones.yaml').read_text())['indic_wav2vec']
encoder = IndicWav2VecExtractor(IndicWav2VecConfig(**settings)).load()

# clips: caller-provided list of 1-D floating PCM tensors at 16 kHz.
def encode_clips(clips):
    lengths = torch.tensor([clip.numel() for clip in clips], dtype=torch.int64)
    samples = torch.nn.utils.rnn.pad_sequence(clips, batch_first=True)
    return encoder.extract_batch(samples, lengths, sample_rate=16000)

# output = encode_clips(clips)
# Your existing head consumes output.features plus lengths/mask as appropriate.
```

Handle `BackboneInputError` as an invalid-input boundary failure and
`BackboneLoadError` as a model/configuration/loading failure. Unexpected backend
runtime failures can also propagate. None of these outcomes is a spoof score.
Do not fabricate a risk decision for infrastructure failures. Silence is accepted
as an acoustic input and returns embeddings; this does not establish usable speech.

## Evidence you can rely on

14 unit tests and the real-checkpoint suite passed in both the project setup and
a clean installation. Real speech used two English LibriSpeech recordings from
the same speaker/chapter. Recorded outcomes:

- Single full clip: `[1,292,1024]`, length `[292]`.
- Two full unequal clips: `[2,292,1024]`, lengths `[292,240]`.
- 4s/3s/4s batch: `[3,199,1024]`, lengths `[199,149,199]`.
- Standalone-versus-grouped-batch maximum absolute errors: `[0.0,0.0,0.0]`.
- Finite input padding changed from +77 to -77 without changing features.
- All block selections and final output matched direct upstream calls exactly.
- A disposable linear probe updated with finite nonzero gradients; the full
  encoder state hash remained unchanged and encoder/input gradients stayed None.
- Repeatability, wrong sample rate, malformed shape/lengths, empty/too-short,
  nonfinite and out-of-range audio checks passed.

The zero parity result applies to exact-length grouping, not a hypothetical
single waveform-padded upstream batch. No comparison of that alternative policy
was completed. No artificial rounding or forced feature replacement was used.

For reproduction, follow the fixture preparation section in `docs/b1-setup.md`, then:

```sh
python -m scripts.verify_indic \
  --audio artifacts/speech/1272-128104-0000.flac artifacts/speech/1272-128104-0001.flac \
  --output artifacts/encoder-verification.json
```

## Runtime expectations and limits

Apple M3 CPU, four PyTorch threads, FP32; one warm-up and five timed repeats:

| Audio per call | Median encoder extraction |
| --- | ---: |
| One 3s clip | 187.3 ms |
| One 4s clip | 246.1 ms |
| 4s + 3s | 439.1 ms |
| 4s + 4s | 474.8 ms |

Load including checksum verification was 7.78s, with possibly warm OS cache.
Peak process RSS across the full verification was approximately 2.52 GiB.
These are CPU encoder measurements, excluding audio collection/decoding and
head/aggregation work. They are not end-to-end latency or throughput guarantees.

No measurements exist for the user's 9900X / RTX 4080 Super PC. GPU acceleration
requires a validated CUDA setup and synchronized timing. The current verification
script supports CPU timing only. Indian-language robustness, spoof accuracy,
CUDA/MPS, long audio, concurrency, ONNX and mobile readiness are unverified.

## Instructions for receiving agents

1. Inspect the feature-branch checkout and configuration; do not assume main
   contains the update. Download the external checkpoint separately.
2. Preserve B1 preprocessing ownership, axes, length units, mask polarity,
   checkpoint identity and layer semantics.
3. Integrate through the public API and keep your component's changes within
   your assigned ownership. Do not silently change routing or add a fallback.
4. Validate your own head's mask handling, dimension, device and training lifecycle.
   The existing probe is evidence for tensor/backward compatibility, not for your
   head's architecture or pretrained weights.
5. Use `feature/indicVac2Wav` or a coordinated feature branch. Do not overwrite
   teammates' work, force-push shared history, or commit weights/generated artifacts.
6. Report your own measured results separately from the B1 CPU evidence. Shared
   interfaces do not establish model interchangeability or detection quality.
