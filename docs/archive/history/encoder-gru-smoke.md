> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Real speech encoder → GRU integration check

Verified locally on 2026-09-17, following the GRU merge at `c250bbd`.
The existing frozen IndicWav2Vec wrapper works with the existing GRU head,
collator, optimizer factory, and training loop. No changes to their implementation
or routing were necessary.

## Repeat the check

From the repository root, with the Python 3.10 encoder environment installed:

```sh
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest tests -q
.venv/bin/python -m scripts.verify_encoder_gru
```

New checkouts should first follow [encoder setup](../../b1-setup.md), including the
checkpoint download and **Verification fixtures and commands** section. The
smoke script requires these local, Git-ignored files and downloads nothing:

- `models/indicwav2vec_large.pt`
- `artifacts/speech/1272-128104-0000.flac`
- `artifacts/speech/1272-128104-0001.flac`

Both audio files and the checkpoint are SHA-256 checked. Checkpoint identity:
`26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`.
The audio fixtures are genuine English LibriSpeech recordings from one speaker.
They retain label **0 = genuine**; neither is relabelled as spoof for this test.

The script reads `configs/backbones.yaml` and layers `configs/gru.yaml` over
`configs/base.yaml`. It uses their encoder, GRU architecture and optimizer
settings, while fixing the smoke run to two examples, one cross-entropy training
step, CPU float32, seed 42 and four threads. It does not run the configured
10-epoch training schedule or EER-based checkpoint selection. Optional arguments
include `--speech-dir`, `--output`, `--threads`, `--backbone-config` and
`--detector-config`. Non-CPU configurations are rejected.

## What is exercised

1. Decode the fixtures and take their first 3 and 4 seconds. Require mono 16 kHz;
   no resampling, mono conversion or denoising is needed. The encoder alone
   applies the checkpoint's waveform normalization to valid samples.
2. Extract contextual features using the real final encoder layer. Derive frame
   counts from the loaded convolution geometry and check the B1/B2 collation
   boundary. `valid_lengths` passed to the GRU count **frames**, not waveform
   samples; `padding_mask=True` means ignored right-padding.
3. Compare standalone and unequal-batch features and GRU logits. Change ignored
   waveform padding from +77 to −77 and feature padding from zero to +77.
   Valid waveform amplitudes still must be in `[-1,1]`; all padding must be finite.
4. Run the existing `train_epoch` for one AdamW update on detached encoder features.
   Check finite gradients and updates in the input normalization, projection,
   GRU and classifier; verify absent waveform/encoder gradients, encoder eval
   mode, and unchanged SHA-256 over every encoder parameter and persistent buffer.

The encoder remains a separate non-`nn.Module` object. Configure its device
explicitly; the detector's `.to()` and `.state_dict()` do not manage it. This
script loads the encoder from checkpoint/config and explicitly moves the head
before constructing its optimizer. It writes no detector checkpoint.

## Measured results

Environment: macOS 26.5.2 arm64, Python 3.10.18, PyTorch 2.2.2, Fairseq 0.12.1,
NumPy 1.23.5, SoundFile 0.12.1, pytest 8.3.5; CPU, four PyTorch threads.

| Check | Result |
| --- | --- |
| Existing encoder and B2 suite | 58 passed; three upstream weight-norm deprecation warnings |
| Dependency consistency | `pip check`: no broken requirements |
| Encoder output | `[2,199,1024]`; valid frame counts `[149,199]` |
| B1 output versus B2 collator | Features, lengths and mask match exactly |
| GRU logits | Finite `[2,2]` |
| Standalone versus batch features | Maximum absolute error 0 for both clips |
| Standalone versus batch logits | Maximum absolute error `8.94e-8` |
| Changed waveform/feature padding | Outputs unchanged exactly |
| Cross-entropy before update | `0.6779804229736328` |
| Head gradients and update | All 10 parameter tensors received finite, nonzero gradients and changed |
| Encoder state and gradients | Full state hash unchanged; all gradients absent; eval mode retained |

Final smoke-run timing observations:

| Operation | Elapsed |
| --- | --- |
| Load encoder, including checkpoint hash | 6.525 s |
| First extraction of the 3s + 4s batch | 427.7 ms |
| First GRU evaluation forward | 6.51 ms |
| One training step on cached features | 33.65 ms |

These are individual observations, not a warmed latency benchmark. Training-step
time excludes encoder extraction. File decoding and complete application latency
are not represented. Full configurations, fixture hashes, gradient norms, state
hashes and timings are saved locally in `artifacts/encoder-gru-smoke.json`.

## Remaining work

This verifies integration mechanics. A randomly initialized head updated once
on two genuine clips provides no evidence of spoof-detection quality. Accuracy,
EER, calibration, Indian-language robustness and CUDA/MPS performance remain
unmeasured. No production dataset/audio pipeline is tested here.

The next detector-team task is a small labelled genuine/spoof dataset with
independent training and validation splits, followed by a controlled training
and evaluation run. Keep encoder checkpoint, layer selection and preprocessing
fixed and record them with detector checkpoints.
