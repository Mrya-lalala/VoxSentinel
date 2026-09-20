# VoxSentinel A2 — audio preprocessing repair and integration handoff

Prepared for the manager agent from the working tree dated 2026-09-17.  This
report describes the implemented A2 audio boundary, the executed verification,
and the exact remaining B1/B2 integration boundary.

## 1. What changed and why

The reviewed A2 implementation had five defects, all addressed in this update:

| Review finding | Fix |
| --- | --- |
| Stereo integer PCM was downmixed to float *before* integer scaling, so identical int16 channels containing 16384 produced `16384.0` while equivalent mono produced `0.5`. | One shared conversion path scales integer PCM to float **before** channel averaging (`src/audio/conversion.py` + `src/audio/preprocessing.py`). |
| SoundFile loading kept a second drifting copy of mono/resampling/status logic and could integer-scale already-float samples. | `preprocess_file` decodes via SoundFile into float amplitudes and delegates to the same `preprocess_array` used for arrays. SoundFile-decoded floats are never integer-scaled again. |
| A 64,001-sample recording emitted `[64000,1]`, violating B1's 400-sample minimum. | `src/audio/chunking.py` retains final windows with >= 400 samples at actual length, drops tails of 1–399 samples, records each omitted span, and marks recordings shorter than 400 samples as `too_short`. |
| Per-chunk status inherited the whole recording's status, so an all-silent window inside a nonzero recording was marked `USABLE`. | Each chunk's silence is computed from its own samples via an exact-zero check (`is_silent`); `ChunkResult.all_silent` aggregates. No near-silence threshold or speech classifier was added. |
| The constant `audio-v1-16khz-mono-chunk4s` was attached to 2-second chunks. | `PreprocessingIdentity` separates a schema version from effective configuration and derives a SHA-256 fingerprint from canonical serialization of the effective settings. 2-second and 4-second windows produce different identities. |

No existing file was modified: all changes are additive under `src/audio/`,
`tests/audio/`, `configs/`, `scripts/`, and `docs/`.  B1 normalization,
routing, detector, data, and scoring code are untouched.

## 2. API, shapes, and conventions

Import:

```python
from src.audio import (
    preprocess_array, preprocess_file, chunk_audio, collate_chunks,
    load_preprocessing_config, PreprocessingConfig,
)
```

### Channel axis

The channel axis is always the **last** axis:

- Mono: `[S]`
- Samples-first multichannel: `[S, C]`

Channels-first `[C, S]` is rejected explicitly; A2 never guesses the layout.
Downmix is a mean over the last axis (after integer scaling).

### Sample length vs frame length

These are different quantities and must not be conflated:

| Quantity | Units | Where |
| --- | --- | --- |
| Sample length | audio samples (400 min for one encoder frame) | `AudioChunk.valid_samples`, B1 `valid_lengths` **input** |
| Frame length | encoder output frames (199 for 64,000 samples, 149 for 48,000) | B1 `EncoderBatch.valid_lengths` **output** |

### `preprocess_array(samples, sample_rate, *, source_id, original_format, config)`

- Accepts mono `[S]` or `[S, C]`; integer (`int16`, `int32`, `uint8`) or float
  (`float32`/`float64`) dtypes.
- Integer PCM is scaled before downmix: int16 `-32768 -> -1.0`,
  `32767 -> 32767/32768`; int32 by `2**31`; uint8 by `(x - 128) / 128` so
  midpoint 128 is 0.0 (never `+0.502`).
- Complex, boolean, object/string, packed 24-bit, and other unsigned dtypes are
  rejected, not coerced.
- Returns `PreprocessedAudio` with mono float32 `samples` `[S]` at 16 kHz,
  `sample_rate`, `source_id`, `original_sample_rate`, `original_channels`,
  `original_format`, and `identity`.

### `preprocess_file(path, *, source_id, config)`

Decodes with SoundFile (`dtype='float64'`) and delegates to `preprocess_array`.
Supported containers are whatever libsndfile/SoundFile supports; no arbitrary
container is promised.  Source identity defaults to the file path.

### `chunk_audio(audio, *, config) -> ChunkResult`

- `chunks`: list of `AudioChunk(source_id, samples [S] float32, sample_rate,
  start_sample, end_sample, is_silent, preprocessing_identity)`.
- `skipped`: list of `SkippedSpan(source_id, start_sample, end_sample,
  sample_count, reason)`.
- `status` in `{"ok", "empty", "too_short"}`; `all_silent` bool.
- `collate_chunks(chunks) -> (samples [B, S] float32, valid_lengths [B] int64)`
  right-pads with finite zeros; `valid_lengths` are **sample** counts.

Offsets (`start_sample`/`end_sample`) index the **resampled 16 kHz waveform**,
not the original-rate array; timestamps (`start_ms`, `duration_ms`) are derived
from those offsets at 16 kHz.

## 3. Effective config, identity, and policies

`configs/audio.yaml` (loaded by `load_preprocessing_config`):

```yaml
audio:
  target_sample_rate: 16000
  window_seconds: 4.0
  hop_seconds: 4.0            # contiguous non-overlapping for this milestone
  min_valid_samples: 400
  resampling_method: scipy_polyphase
  overshoot_tolerance: 0.001
  input_range_tolerance: 0.000001
```

- **Resampling**: `scipy.signal.resample_poly` — a real Kaiser-windowed
  polyphase FIR resampler, not an interpolation mock.  The backend name is
  recorded in the identity.  Output length is
  `ceil(N * target_rate / source_rate)`.
- **Range policy**: malformed float input outside `[-1, 1]` (tolerance
  `1e-6` for float representation only) is rejected **before** resampling, with
  no correction.  After resampling, ringing within `overshoot_tolerance` is
  clamped to `[-1, 1]`; anything larger is rejected as material overshoot.
  No peak/RMS normalization, rescaling, denoising, silence trimming, VAD, or
  AGC exists in the default path.
- **Tail policy**: contiguous non-overlapping windows; final windows with
  >= 400 samples are retained at actual length; tails of 1–399 samples are
  dropped and recorded; whole recordings < 400 samples are `too_short`.
  Padding never turns synthetic zeros into valid audio.
- **Silence policy**: per-chunk exact-zero check only.  `is_silent`/`all_silent`
  describe sample values, not speech presence or spoof confidence.
- **Unsupported files/dtypes**: rejected with `AudioLoadError` (decode failure)
  or `AudioValidationError` (rank/dtype/rate/range/channel violations).

The complete identity is `audio-v1:sha256:<16 hex>` derived from the canonical
serialization of the *effective settings only* (schema version excluded).
Different window sizes, resamplers, tolerances, or tail policies produce
different identities.  The identity is attached to every `PreprocessedAudio`
and every `AudioChunk`.

## 4. Handling empty / invalid / silent / too-short audio

A caller must not fabricate a score from these outcomes:

- **Invalid** (NaN/Inf, out-of-range float, wrong rank/dtype, zero channels,
  material overshoot): an `AudioValidationError`/`AudioLoadError` is raised —
  an infrastructure/input failure, never a spoof decision.
- **Empty** (`status == "empty"`): zero decoded samples; no chunks.
- **Too short** (`status == "too_short"`): nonempty but < 400 samples; no
  chunks and a skipped span with reason `recording_too_short`.
- **Silent** (`all_silent`): chunks exist but every sample is exactly zero.
  Silence is a valid acoustic input, **not** evidence of genuine speech.  The
  integration example (`scripts/verify_audio_to_b1.py`) reports `all_silent`
  and must not treat it as a meaningful genuine/spoof decision.
- **Skipped tails** (1–399 samples): recorded in `ChunkResult.skipped` with
  source ID, offsets, count, and reason.  They are omitted from encoding, never
  padded into validity.

`collate_chunks` raises on an empty list; callers must branch on `status`/no
chunks before collating.

## 5. Reproduction, versions, and evidence

### Executed (this machine)

Environment: Windows, Python 3.13.7, numpy 2.3.2, scipy 1.18.1,
soundfile 0.12.1, PyYAML 6.0.3.

```powershell
python -m unittest discover -s tests/audio -v
```

Result: **76 tests collected and passed** (audio units only).

Audio verification without the real checkpoint (A2 boundary fully runs; the
B1 step records its blocker):

```powershell
python -m scripts.verify_audio_to_b1 --audio clip1.wav clip2.wav
```

### Blocked (not executed here)

The real B1 extraction requires the pinned legacy Fairseq runtime, torch, and
the trusted 3.8 GB IndicWav2Vec checkpoint.  On this Windows Python 3.13 box,
`torch`/`fairseq` are not installed and `models/indicwav2vec_large.pt` is
absent.  The script therefore reports
`runtime import failed: No module named 'torch'` (or the missing-checkpoint /
load-failure reason) instead of failing.  Reproduce the full real-checkpoint
check on the pinned macOS Python 3.10.18 environment:

```sh
bash scripts/setup_encoder_env.sh
source .venv/bin/activate
python -m pip install -r requirements-audio.txt
mkdir -p models
curl -fL --retry 3 \
  https://objectstore.e2enetworks.net/indic-superb/aaai_ckpts/pretrained_models/indicw2v_large_pretrained.pt \
  -o models/indicwav2vec_large.pt
python -m scripts.verify_audio_to_b1 --audio clip1.flac clip2.flac --output artifacts/a2-verification.json
```

The pinned production dependency file is `requirements-audio.txt`
(`scipy==1.11.4`; numpy/soundfile/PyYAML already pinned in `requirements.txt`).
scipy 1.18.1 was used for the executed local runs because 1.11.4 has no wheels
for Python 3.13; `resample_poly` behavior is stable across these versions and
the tests use range bounds with margin.

### Repository state

- Branch: `main`, HEAD `c250bbd0ac3b2af1ae345ce48960fcc63d200557`.
- The prompt named `feature/a2-audio-preprocessing` as the working branch; the
  current checkout is on `main`.  No commit/push authorization was present in
  this session, so no commit, branch switch, or push was performed.
- All changes are **new, untracked files**; no existing file was modified:

```
configs/audio.yaml
requirements-audio.txt
scripts/verify_audio_to_b1.py
src/audio/{__init__,chunking,config,conversion,errors,identity,preprocessing,resampling}.py
tests/audio/{__init__,test_chunking,test_conversion,test_metadata,test_preprocessing,test_resampling}.py
docs/A2_AUDIO_HANDOFF.md
```

### Test coverage vs required table

| Area | Coverage |
| --- | --- |
| PCM | signed mono/stereo scaling; equivalent channels equal mono; uint8 midpoint/endpoints; 128 -> 0.0; int32; unsupported dtypes rejected |
| Entry-point parity | int16 mono/stereo file vs decoded array (exact); float file not rescaled |
| Resampling | real 8k/44.1k/48k -> 16k lengths; finite outputs; near-full-scale sine clamped within tolerance; full-scale DC edge rejected; same-rate copy |
| Chunking | exact multiples; retained tails; sizes 0/1/399/400/64000/64001/64399/64400; full emitted+skipped accounting |
| Status | silent window inside nonzero recording; all-silent flag; nonzero chunk; silent audio still `ok` |
| Metadata | exact offsets/timestamps; source/order preservation; 2s vs 4s identities differ; fingerprint repeatability |
| Invalid input | NaN/Inf, bad rate/duration, rank/dtype, zero channels, out-of-range float; post-resampling overshoot policy |
| B1 boundary | `collate_chunks` sample lengths + right padding; mixed lengths |

## 6. B2 / data handoff

- **Source-manifest linkage**: `AudioChunk.source_id` and offsets are the keys
  for joining to the data manifest.  A2 never invents labels, speakers,
  languages, generators, or dataset splits.  Data owners split related
  speakers/source recordings/generated variants **before** chunking; chunks
  inherit their source's split.  A2 does not randomly split chunks.
- **Preprocessing identity**: every chunk carries
  `audio-v1:sha256:<fingerprint>`.  A training manifest must store this identity
  (and the pinned runtime identity) so features can be reproduced.
- **Feature slicing/masks**: after
  `encoder.extract_batch(samples, sample_lengths, sample_rate=16000)`, construct
  `EmbeddingExample(features=features[i, :frame_lengths[i]], label=...)` so
  feature padding is never relabelled valid.  The B2 GRU already consumes
  `[B, T, D]` with `valid_lengths` (frames) and `padding_mask` (True = padding).
- **Device alignment**: B1's wrapper is a plain object, not an `nn.Module`;
  moving a parent module does not move B1.  Align the head and encoder/output
  devices explicitly, and keep the encoder frozen with no gradients while the
  head updates.
- **Remaining ownership**: real B1 extraction and real B2 backward verification
  were not executed here (see blockers in section 5).  The A2 code provides the
  `collate_chunks` -> `extract_batch` -> `EmbeddingExample` path; the owning B2
  team must run it with the trusted checkpoint and real labels, and must not
  treat the plumbing-only dummy labels or any synthetic fixture as spoof
  accuracy, language-generalization, calibration, or live-latency evidence.

The 400-sample minimum is a technical convolution minimum, not evidence that a
25 ms clip supports reliable spoof detection.
