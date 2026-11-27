# Trainable IndicWav2Vec Large

## Implementation plan

1. Add `TrainableIndicWav2Vec(config, *, mode="frozen")` with a
   `forward(samples, valid_lengths, *, sample_rate=16000)` that returns
   `EncoderBatch`. Use the existing `indic_wav2vec` YAML section with a
   caller-selected device and require a SHA-256 pin, final-layer output,
   and 1024 features.
2. Share input checks and the exact-length encoding, waveform normalization,
   frame counting, and padding path. Keep the extractor's detached,
   inference-safe outputs and its existing callers.
3. Keep the Fairseq model and encoder container in evaluation mode. In
   `last_two` mode enable gradients for the final two blocks, and let their
   internal dropout follow the wrapper's train/eval state. Block-local
   normalization affine parameters follow the blocks. The encoder-wide
   final norm stays frozen, and the encoder container's eval mode disables
   layer dropping.
4. Use `named_parameters()` and `parameters()` for parameter selection.
   Attach a version, verified base hash, mode, and resolved block paths
   to the normal PyTorch state dict and validate them on load.
5. Run same-device FP32 parity on synthetic PCM WAVs, including unequal
   lengths, altered right padding, and the minimum window. Check input
   errors, a CUDA backward pass, serialized state reload, and the existing
   frozen encoder regressions.

## Trainer handoff

The later optimization experiments also add `mode="all_blocks"`. This
enables all 24 `model.encoder.layers.0` through `.23` transformer blocks,
including their block-local normalization parameters. The convolutional
front end and encoder-wide final norm stay frozen. Block dropout follows
train/eval, and encoder-level layer dropping stays disabled. A disposable
CUDA backward check on two four-second windows verified finite gradients
in every block and no gradients in frozen parameters, with 2,447.9 MiB peak
allocation before optimizer state. This mode is intended for experiments.

Use the repository's `indic_wav2vec` settings. They pin the base checkpoint
to SHA-256
`26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`.
Supply an absolute checkpoint path when running from a worktree.

```python
from pathlib import Path
import yaml
from src.backbones.indic_wav2vec import IndicWav2VecConfig
from src.backbones.trainable_indic_wav2vec import TrainableIndicWav2Vec

settings = yaml.safe_load(Path("configs/backbones.yaml").read_text(encoding="utf-8"))
config = IndicWav2VecConfig(**{
    **settings["indic_wav2vec"],
    "checkpoint_path": str(Path("../../../models/indicwav2vec_large.pt").resolve()),
    "device": "cuda",
})
encoder = TrainableIndicWav2Vec(config, mode="last_two").train()
batch = encoder(waveforms, sample_lengths)
trainable_parameters = [p for p in encoder.parameters() if p.requires_grad]
```

`waveforms` is dense float32 `[B, S]` prepared mono 16 kHz PCM and
`sample_lengths` is dense int32 or int64 `[B]`. Valid lengths are 400 to
`S` samples, inclusive. Every value must be finite. Valid samples must be in
`[-1, 1]`. Finite right-padding values are ignored. Invalid inputs raise
`BackboneInputError`. Waveforms are detached, with gradients tracked for
the selected model parameters. Exact-length groups are normalized separately
using Fairseq's checkpoint setting `task.normalize=True`.

Outputs are on the model device. `batch.features` is float32 `[B, T, 1024]`
with zero-filled padding. `batch.valid_lengths` is int64 `[B]` in frames
and `batch.padding_mask` is bool `[B, T]` with `True` marking ignored
frames. The minimum window (400 samples) yields one frame. The frame hop is
320 samples (20 ms). The final-layer output includes the encoder-wide final
LayerNorm, with `output_layer=None`.

In `last_two` mode the exact prefixes from `encoder.named_parameters()`
are `model.encoder.layers.22.` and `model.encoder.layers.23.`. Each has
these exact suffixes, with both `.weight` and `.bias` for every entry:

- `self_attn.k_proj`, `self_attn.v_proj`, `self_attn.q_proj`, `self_attn.out_proj`
- `self_attn_layer_norm`, `fc1`, `fc2`, `final_layer_norm`

Only these blocks and their children enter training mode with `train()`.
Their internal dropout follows that mode. The feature extractor, encoder
container, positional convolution, earlier blocks, and encoder-wide
LayerNorm stay in eval mode and frozen. Layer dropping stays disabled.
`eval()` disables all dropout and preserves the selected parameters'
`requires_grad` flags. Use `torch.no_grad()` when evaluating `last_two`
without a graph. In `frozen` mode every forward yields detached features
usable by a downstream trainable head, and all Fairseq modules stay in eval
mode.

Save and restore the encoder with the standard PyTorch APIs:

```python
import torch

torch.save(encoder.state_dict(), "encoder-state.pt")
restored = TrainableIndicWav2Vec(config, mode="last_two")
state = torch.load("encoder-state.pt", map_location="cpu", weights_only=True, mmap=True)
restored.load_state_dict(state)
```

The state includes all model parameters and buffers, about 1.2 GiB in FP32,
and metadata with the schema version, verified base SHA-256, mode, and block
paths. Loading checks that metadata against the newly constructed encoder.
Set the desired train/eval mode after reload. Keep the encoder in FP32.
Forward disables autocast for the encoder. CUDA memory use grows with batch
size and duration, and the smoke check excludes optimizer and head memory.

## Engineering verification

Run from the `artifacts/worktrees/ifd-dataset` worktree:

```powershell
& ../../environments/mixed-gru-cuda/Scripts/python.exe -m scripts.verify_trainable_encoder --checkpoint ../../../models/indicwav2vec_large.pt
& ../../../.venv/Scripts/python.exe -m pytest -q tests/backbones tests/service
```

- The CUDA smoke passed on an RTX 4080 SUPER with PyTorch 2.2.2+cu121.
  Four synthetic 16 kHz WAVs produced shape `[4, 12, 1024]` and frame
  lengths `[12, 5, 1, 12]`. Frozen encoder parity and altered-padding
  invariance both had maximum absolute error `0` with matching masks.
  The checks require exact equality for the same groups and precision.
  Singleton error was `9.54e-7`, within the `1e-4` absolute limit for the
  different CUDA kernel batch shapes. Autocast and TF32 were disabled.
- All 32 selected parameter tensors received finite gradients, with
  nonzero signal in both blocks. Every frozen parameter had no gradient.
  Peak allocated CUDA memory was 1,390.9 MiB for this batch and backward
  pass. Five invalid configurations and eight invalid inputs were rejected.
- All saved tensors matched a clean load of the pretrained checkpoint.
  Restored tensors and evaluation features matched exactly. The loader
  rejected a wrong hash, wrong mode, and missing provenance metadata.
- The CPU regression command passed 95 tests with zero failures,
  including the frozen encoder, FilePredictor, and service tests.
