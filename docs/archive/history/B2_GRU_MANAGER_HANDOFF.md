> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# VoxSentinel B2 - GRU detector and training pipeline handoff

Prepared for the manager agent from the current repository state (2026-09-15).
This report describes implemented behavior, the evidence gathered from focused
local test runs, and what remains before real spoof/genuine training.

**Headline status:** the custom GRU detector, its embedding-sequence data
contract, the training/validation loop, best-checkpoint selection, evaluation
metrics, configuration, and tests are complete in the working tree. Training has
only been exercised on a synthetic fixture and a synthetic real-dimension
(D=1024) sanity run. No real spoof/genuine dataset has been downloaded or used,
and no real-data spoof-detection performance is claimed anywhere in this report.

---

## 0. Scope and repository state

- Scope owned by B2: detector models, the B2 data contract, training loop,
  evaluation aggregation/metrics, checkpoint format, B2 configs, B2 tests.
- Out of scope and untouched: B1 IndicWav2Vec implementation, backbone routing,
  audio loading/preprocessing, API, UI, deployment, AASIST implementation,
  real dataset ingestion, `main`.
- Current branch: `feature/gru`.
- `HEAD` = `827d526` ("version 0 frozen IndicWav2Vec encoder"), which is also
  `origin/main`. Local `main` is at `f899df8` (4 commits behind `HEAD`).
- `feature/gru` currently tracks `origin/rawnet2-aasist` and is "ahead 4" of
  that remote ref. `origin/rawnet2-aasist` (`f899df8`) contains only the initial
  scaffold; `origin/main` contains the B1 encoder work and **no B2 code**
  (`src/voxsentinel/`, GRU, or RawNet2 paths are absent from `origin/main`).
- Nothing has been committed or pushed by B2. All B2 work exists as uncommitted
  working-tree/index changes in this checkout.
- The git index still contains the earlier RawNet2-era scaffold as *staged new
  files* (`.gitignore`, `src/voxsentinel/**`, `configs/aasist.yaml`,
  `configs/rawnet2.yaml`, `tests/**`), while the worktree has moved on to the GRU
  implementation. `git status` therefore shows `AD` entries for
  `configs/rawnet2.yaml`, `src/voxsentinel/models/rawnet2.py`,
  `src/voxsentinel/detectors/rawnet2.py`, and `tests/test_rawnet2.py`.
  **Before any commit, run `git add -A` (or `git rm -r --cached` on the deleted
  paths) so the commit does not resurrect RawNet2.** New GRU files are currently
  untracked (`??`).

### Read this first: what is and is not true

| Claim | Reality |
| --- | --- |
| GRU architecture implemented | True, in `src/voxsentinel/models/gru.py` |
| Training/validation/checkpoint plumbing implemented | True, in `src/voxsentinel/training/` |
| Pipeline verified end to end | True, but only on synthetic embedding fixtures |
| Real spoof/genuine training completed | **False - not started** |
| Real-data accuracy/EER numbers | **None exist; do not cite any** |
| AASIST implemented | **False - reserved name and empty config only** |
| RawNet2 remains the final detector | **False - files removed from the worktree** |
| B2 merged into `main` / pushed | **False - uncommitted in this checkout** |

---

## 1. Completed work

### 1.1 Custom GRU detector (replaces the original RawNet2 model)

File: `src/voxsentinel/models/gru.py`. Class: `GruSpoofDetector`, registered as
`"gru"` in the model registry.

Architecture (exactly as implemented):

```
features [B, T, D]  (D defaults to 1024)
  -> LayerNorm(D)
  -> Linear(D, hidden_size)          # hidden_size default 256
  -> unidirectional GRU(hidden_size, hidden_size, num_layers=1, batch_first=True)
  -> valid-frame masked mean pooling over the GRU output sequence
  -> Dropout(dropout)                # dropout default 0.0
  -> Linear(hidden_size, 2)          # genuine/spoof logits
```

- Defaults: `input_dim=1024`, `hidden_size=256`, `num_layers=1`, `dropout=0.0`,
  `num_classes=2` (only 2 is accepted).
- Parameter counts for the defaults (measured on this checkout):
  **659,714 total, 659,714 trainable** (all parameters require grad).
- The model is a `torch.nn.Module` (`Backbone`) and is fully trainable on CPU.
- Logit convention is fixed and documented: index 0 = genuine, index 1 = spoof.
  `DetectorOutput.scores` returns `softmax(logits)[..., 1]`, i.e. P(spoof).
- Rationale recorded in the module docstring: the RawNet2 *idea* of recurrent
  modelling over frame-level features is reused, but the RawNet2 front-end
  (fixed sinc filterbank, residual/FMS blocks, raw waveform input) was removed
  and is not part of the final detector.

### 1.2 Variable-length and padding handling

The GRU consumes the B1 batch contract and ignores padded frames in two places:

1. Recurrence: `valid_lengths` is converted to a `[B, T]` boolean mask, reduced
   to per-sequence lengths, and passed to
   `pack_padded_sequence(..., batch_first=True, enforce_sorted=False)`. Padded
   frames therefore never enter the GRU computation; `pad_packed_sequence`
   restores the time axis afterwards.
2. Pooling: the masked mean divides by the number of valid frames per sequence,
   so padded positions contribute nothing to the pooled vector.

Additional behavior:

- `padding_mask` is `[B, T]` bool with **True = padding**, matching the B1
  contract. If only `padding_mask` is supplied, lengths are derived from it.
- If both `valid_lengths` and `padding_mask` are supplied, the two constraints
  are intersected (the stricter of the two wins).
- If neither is supplied, every frame is treated as valid.
- Right/trailing padding is assumed (the B1 encoder contract); this assumption is
  stated in the class docstring.
- A sequence with zero valid frames raises `ValueError`, as does an embedding
  whose last dimension does not match the configured `input_dim`.
- Shape validation rejects anything that is not `[B, T, D]`.

### 1.3 Detector adapter and registry

- `src/voxsentinel/detectors/base.py`: `Detector` (abstract `forward(batch)`) and
  `DetectorOutput(logits)` with the spoof-probability `scores` property.
  `Detector.loss(output, batch)` is cross-entropy over `[B, 2]` logits and
  `labels.long()`, so label 1 = spoof.
- `src/voxsentinel/detectors/gru.py`: `GruDetector` reads
  `batch["features"]` and optional `batch["valid_lengths"]` /
  `batch["padding_mask"]`, and returns `DetectorOutput`. It raises a clear
  `KeyError` when `features` is missing.
- `src/voxsentinel/detectors/registry.py` and
  `src/voxsentinel/models/registry.py`: `_RESERVED = ("gru", "aasist")`;
  `available_models()` / `available_detectors()` both return
  `('aasist', 'gru')`. Creating `"gru"` works; creating `"aasist"` raises
  `NotImplementedError("The aasist detector adapter has not been implemented yet.")`
  (verified by execution).

### 1.4 B2 training data interface

New package `src/voxsentinel/data/`:

- `batch.py`
  - `EmbeddingExample(features, label)`: one labelled example, `features`
    `[T, D]` coerced to `float32`, label in `{0 (genuine), 1 (spoof)}`; validates
    rank, non-empty frames/dims, and the label convention.
  - `EmbeddingExample.from_array(features, label)`: accepts anything
    `torch.as_tensor` handles, including a B1 `EmbeddingSequence.features`
    `float32` ndarray of shape `[frames, hidden_dim]`.
  - `collate_examples(examples, pad_value=0.0)`: right-pads a list into the exact
    detector batch - `features [B, T, D]` float32, `valid_lengths [B]` int64,
    `padding_mask [B, T]` bool (True = padding), `labels [B]` int64. Padding is
    zero by default, matching B1's "padded frames are zero" contract. Rejects
    empty lists and mixed embedding dims.
  - `batch_iterator(examples, batch_size, shuffle=False, seed=0)`: yields
    collated batches; deterministic shuffle via `torch.Generator` for a given
    seed.
- `synthetic.py`
  - `synthetic_examples(num_examples=32, input_dim=1024, min_frames=8,
    max_frames=24, seed=0)`: alternating genuine/spoof examples. Genuine features
    are i.i.d. noise; spoof features add a fixed sinusoidal pattern across the
    embedding dim (a signal that survives the detector's per-frame LayerNorm and
    is trivially separable). This is explicitly a smoke-test fixture, not a
    benchmark and not a model of real spoofing artifacts.

### 1.5 Training loop, optimizer, loss

- `src/voxsentinel/training/engine.py` (pre-existing, extended):
  - `train_epoch(detector, batches, optimizer, device="cpu", loss_fn=None)`:
    `zero_grad -> forward -> loss -> backward -> optimizer.step`, returns
    `EpochResult(loss, steps, examples)`.
  - `validate_epoch(...)`: forward-only loss aggregation (kept for backward
    compatibility with existing tests).
  - `loss_fn` is optional; when omitted the detector's built-in cross-entropy is
    used.
- `src/voxsentinel/training/builders.py`:
  - `build_optimizer(model, OptimizerConfig)`: supports `adamw`, `adam`, `sgd`;
    rejects unknown names and non-positive learning rates; only optimizes
    parameters with `requires_grad=True`.
  - `build_loss(name)`: supports `cross_entropy` (aliases `ce`); rejects unknown
    names.
- `src/voxsentinel/training/trainer.py`:
  - `fit(detector, train_batches, val_batches, optimizer, *, model_name,
    model_config, epochs, device="cpu", loss_fn=None, checkpoint_path=None,
    monitor="eer", mode="min", threshold=0.5)`.
  - Per epoch: train, then `evaluate` on the validation split, then compare the
    monitored metric and save the best checkpoint on strict improvement.
  - Returns `TrainingResult(epochs=tuple[EpochSummary], best_epoch,
    best_metric, best_checkpoint, monitor, mode)`, where each `EpochSummary`
    holds `train_loss`, `val_loss`, `metrics` (`BinaryMetrics`), `train_steps`,
    `train_examples`.
  - Batch sources may be plain iterables or callables taking the epoch number
    (the latter enables a fresh shuffle each epoch). A plain one-shot generator
    is snapshotted once so multi-epoch runs do not silently exhaust it.
  - Ties do not count as improvement, so with equal monitored values the
    earliest best epoch is kept.
- `src/voxsentinel/training/runner.py`:
  - `run_training(config, train_examples, val_examples, *, checkpoint_path=None,
    device=None)`: seeds torch, builds the detector via `create_detector`,
    builds the optimizer and loss from config, and runs `fit` with per-epoch
    shuffling (`seed + epoch`) and unshuffled validation.
  - Raises `NotImplementedError` when `training.use_amp` is true (mixed
    precision is intentionally not implemented yet).
  - Default checkpoint path:
    `<checkpoint.directory>/<model.name>_best.pt`, e.g.
    `artifacts/gru/gru_best.pt` (the `artifacts/` directory is git-ignored).

### 1.6 Evaluation

- `src/voxsentinel/evaluation/metrics.py` implements, without sklearn:
  accuracy, precision, recall, F1, and EER (EER from the threshold sweep that
  minimizes |FAR - FRR|; `None` when only one class is present). Labels must be
  binary with spoof = 1.
- `src/voxsentinel/evaluation/engine.py`: `evaluate(detector, batches, device,
  threshold=0.5)` aggregates spoof probabilities across batches and returns
  `EvaluationResult(metrics, scores, labels, loss)`. The mean validation loss
  field was added in this work so the trainer gets loss and metrics from one
  validation pass.

### 1.7 Checkpoints

Existing versioned format reused unchanged (`src/voxsentinel/checkpoints/`):

- Payload: `format_version` (currently 1), `model_name`, `model_config`,
  `model_state_dict`, optional `optimizer_state_dict`, `epoch`, `global_step`,
  `metrics`.
- `save_checkpoint(path, model, model_name, model_config, *, optimizer, epoch,
  global_step, metrics)`; `load_checkpoint(path, map_location="cpu",
  model_name=None)` validates required keys, format version, and model-name
  match; `restore_checkpoint(path, model, optimizer=None, strict=True, ...)`
  restores weights (and optimizer state when present).
- The trainer writes only the best checkpoint per run, including the optimizer
  state and the metric dict (`accuracy`, `precision`, `recall`, `f1`, `eer`) of
  the selected epoch.

### 1.8 Configuration

- `configs/base.yaml` (shared defaults):
  `optimizer.name=adamw`, `optimizer.learning_rate=0.001`,
  `optimizer.weight_decay=0.0`; `training.batch_size=8`, `training.epochs=1`,
  `training.device=cpu`, `training.use_amp=false`, `training.shuffle=true`,
  `training.seed=0`, `training.loss=cross_entropy`; `evaluation.threshold=0.5`;
  `checkpoint.format_version=1`, `checkpoint.strict_load=true`,
  `checkpoint.directory=artifacts/gru`, `checkpoint.best_metric=eer`,
  `checkpoint.best_mode=min`.
- `configs/gru.yaml` (GRU overlay):
  `model.name=gru`; parameters `input_dim=1024`, `hidden_size=256`,
  `num_layers=1`, `dropout=0.0`, `num_classes=2`; `optimizer.name=adamw`,
  `learning_rate=0.001`, `weight_decay=0.0`; `training.batch_size=8`,
  `training.epochs=10`, `training.loss=cross_entropy`;
  `evaluation.threshold=0.5`; `checkpoint.directory=artifacts/gru`,
  `best_metric=eer`, `best_mode=min`.
  (`hidden_size` is the existing GRU width parameter; there is no `hidden_dim`
  key in code.)
- `configs/aasist.yaml` exists but contains only
  `model: {name: aasist, parameters: {}}` - it is a placeholder for the
  reserved, unimplemented AASIST detector.
- The loader (`src/voxsentinel/config/loader.py`) performs the documented
  "later file wins" overlay, so `base.yaml` + `gru.yaml` yields the GRU setup.

### 1.9 Removal of the old RawNet2 implementation

Removed from the worktree as the final detector:

- `src/voxsentinel/models/rawnet2.py` (fixed sinc filterbank, residual/FMS
  blocks, waveform input)
- `src/voxsentinel/detectors/rawnet2.py` (waveform batch adapter)
- `configs/rawnet2.yaml`
- `tests/test_rawnet2.py`

Caveats for the manager:

- These four paths are still present in the **git index** (staged as added,
  deleted in the worktree). They must be unstaged/removed before committing.
- RawNet2 is not restored anywhere in the worktree; `rg --files` finds no
  `rawnet` path, and the model/detector registries no longer mention it.
- The label/class-name convention (0 genuine, 1 spoof) and the generic B2
  infrastructure (checkpoints, metrics, config, training engine) came from the
  earlier scaffold and were reused; only the model-specific parts were replaced.

### 1.10 AASIST status

- Not implemented, by design for this stage.
- `"aasist"` is reserved in both registries and `configs/aasist.yaml` exists as
  a placeholder, so a future implementation can slot into the same
  `Detector`/`create_detector` and config-overlay machinery.
- Any attempt to build it today raises `NotImplementedError`.

---

## 2. Verified behavior and evidence

Environment used for all verification below: Windows, Python 3.14.3,
`torch 2.14.0+cpu`, CPU device.

### 2.1 Focused B2 test run (current state)

```
python -m pytest tests/test_data_batching.py tests/test_training_pipeline.py \
  tests/test_config.py tests/test_gru_detector.py tests/test_training_engine.py \
  tests/test_evaluation_engine.py tests/test_checkpoints.py tests/test_metrics.py \
  tests/test_model_registry.py tests/test_detector_interface.py \
  -p no:cacheprovider
```

Result: **30 passed in 3.92s**. `tests/backbones/` was intentionally excluded
because it requires Fairseq (installed separately for B1). `-p no:cacheprovider`
is used because this Windows environment has a known pytest temp/cache
permission issue; this is an environment workaround, not a code failure.

| Test file | Tests | What it verifies |
| --- | --- | --- |
| `tests/test_gru_detector.py` | 9 | GRU defaults/architecture, `[B,T,D]` forward for variable lengths, padded frames ignored, `valid_lengths`-only vs `padding_mask`-only equivalence, stricter mask honored, zero-valid-frame rejection, adapter + registry, gradients reach projection/GRU/classifier, checkpoint round trip |
| `tests/test_data_batching.py` | 6 | example validation and dtype coercion, B1-style ndarray input, right-padding and mask correctness, rejection of empty/mixed-dim input, batching sizes and deterministic seeded shuffle, synthetic fixture shape `[B,T,1024]` |
| `tests/test_training_pipeline.py` | 5 | weight updates during `fit`, validation each epoch, best-checkpoint learning + payload contents, restore reproduces validation metrics, padded frames do not change collated-batch logits, builder/AMP error paths |
| `tests/test_config.py` | 2 | `base.yaml` + `gru.yaml` overlay values (input_dim 1024, hidden 256, adamw, lr 1e-3, batch 8, epochs 10, cross-entropy, EER/min) and missing-`model.name` rejection |
| `tests/test_model_registry.py` | 2 | `gru`/`aasist` visible in the registry; GRU default construction |
| `tests/test_training_engine.py` | 1 | legacy `train_epoch`/`validate_epoch` counts and non-negative losses |
| `tests/test_evaluation_engine.py` | 1 | evaluation aggregates scores/labels and produces bounded accuracy |
| `tests/test_metrics.py` | 2 | perfect-classification metrics and binary-label rejection |
| `tests/test_checkpoints.py` | 1 | checkpoint save/load/restore round trip |
| `tests/test_detector_interface.py` | 1 | detector scores/loss contract |

### 2.2 Specific evidence points

- **Padded frames do not affect the GRU.** Model-level test perturbs all padded
  frames to `1000.0` and requires identical logits; pipeline-level test does the
  same on a batch produced by `collate_examples`.
- **GRU parameters actually update.** `fit` test compares `state_dict` before and
  after training and requires changed tensors for `model.input_norm.weight`,
  `model.projection.weight`, `model.gru.weight_ih_l0`, and
  `model.classifier.weight`.
- **Validation runs.** Every epoch records `train_steps > 0`,
  `train_examples > 0`, non-negative train/val loss, and bounded accuracy; EER is
  present because the synthetic split contains both classes.
- **Best checkpoint.** After `run_training`, the checkpoint exists, its
  `epoch` equals `result.best_epoch`, its `model_config` equals the configured
  parameters, it contains `optimizer_state_dict`, and its `metrics` include
  `accuracy`, `precision`, `recall`, `f1`, `eer`.
- **Checkpoint reload.** Restoring the best checkpoint into a fresh detector and
  re-running `evaluate` reproduces the saved `accuracy` and `eer` (validation is
  deterministic in `eval()` mode with dropout 0.0).
- **Registry behavior.** Executed: `available_detectors()` ->
  `('aasist', 'gru')`, `available_models()` -> `('aasist', 'gru')`, and
  `create_detector(ModelConfig(name="aasist"))` raises the
  not-implemented error.
- **Parameter count.** Executed: defaults yield 659,714 trainable parameters.

### 2.3 Synthetic sanity training at the real embedding dimension

Run with the actual configs (`configs/base.yaml` + `configs/gru.yaml`,
`input_dim=1024`, `hidden_size=256`), 64 synthetic training examples and 32
synthetic validation examples (20-40 frames each), 5 epochs, batch size 8,
AdamW lr 1e-3, checkpoint written to a temporary directory (no repo artifacts
left behind). Wall time for the whole script was about 9 seconds on CPU.

| Epoch | Train loss | Val loss | Accuracy | F1 | EER |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.3576 | 0.1947 | 1.000 | 1.000 | 0.000 |
| 2 | 0.0644 | 0.0071 | 1.000 | 1.000 | 0.000 |
| 3 | 0.0008 | 0.0003 | 1.000 | 1.000 | 0.000 |
| 4 | 0.0001 | 0.0001 | 1.000 | 1.000 | 0.000 |
| 5 | 0.0000 | 0.0001 | 1.000 | 1.000 | 0.000 |

Best epoch reported as 1 because EER ties at 0.000 for all epochs and the
trainer only replaces the best on strict improvement. These numbers describe a
separable synthetic fixture only; they say nothing about real spoof detection.

---

## 3. Synthetic-only validation vs real-data validation

**Synthetic-only (all validation done so far):**

- Purpose: prove the plumbing trains, validates, selects a best checkpoint,
  saves/loads it, and evaluates - including padding invariance and gradient flow.
- Both a small-width fixture (`input_dim=32`) in tests and a real-dimension
  `input_dim=1024` sanity run were used.
- The fixture is separable by construction (i.i.d. noise vs a fixed sinusoidal
  pattern), so high accuracy/EER 0.000 is expected and is not evidence of
  spoof-detection capability.

**Real-data validation (none yet):**

- No ASVspoof, ASVspoof 5, IndicSynth, Svarah, or any other spoof/genuine corpus
  has been downloaded, prepared, or traversed.
- No real B1 embeddings have been produced in this checkout during B2 work, and
  no real-data accuracy, precision, recall, F1, or EER has been measured.
- `tests/backbones/` (B1 encoder tests) was not run in this work because the
  Fairseq environment is not installed here.
- Therefore: the GRU architecture and training plumbing are complete, but real
  multi-dataset training has **not** been completed, and no claim of real spoof
  detection performance should be made.

---

## 4. What is still missing

1. **Real dataset ingestion.** Manifest/dataset adapters for the chosen corpora
   (e.g. ASVspoof 5, IndicSynth, Svarah), including label mapping to 0/1 and
   official train/dev/eval splits. Nothing here has been started.
2. **B1 to B2 feature bridge.** A component that takes decoded audio, calls the
   frozen B1 `extract_batch`/`extract` path, and produces `EmbeddingExample`
   objects (`[T, 1024]`) plus `valid_lengths`/`padding_mask` at the batch level.
   B2 currently defines the contract and can consume arrays via
   `EmbeddingExample.from_array`, but does not call B1 itself.
3. **Embedding caching.** Real training needs persisted embeddings (and their
   metadata) so the frozen encoder is not re-run every epoch.
4. **Real-data training run.** No end-to-end run against genuine/spoof speech.
5. **Threshold selection and calibration.** The evaluation threshold is fixed at
   0.5; a dev-set threshold sweep (or EER-matched threshold) is still needed.
6. **Scale/performance work.** AMP is rejected with `NotImplementedError`;
   per-epoch shuffling exists but there is no multi-worker DataLoader, no
   gradient accumulation, and no LR schedule/early stopping.
7. **Class imbalance handling.** No class weighting or sampling strategy yet;
   the loss is plain cross-entropy.
8. **AASIST.** Not implemented; only the reserved name and empty config exist.
9. **Repository hygiene.** Stage the GRU tree (`git add -A`) so the index no
   longer contains RawNet2, decide the real branch/PR path, and later remove the
   stale `origin/rawnet2-aasist` tracking.
10. **Real reporting.** A real-data evaluation report (per-corpus and
    cross-corpus) does not exist yet.

---

## 5. Immediate next steps according to the B2 plan

1. **Wire real embeddings in.** Add a dataset adapter that reads a manifest,
   groups equal-length audio to avoid padding where practical, calls the frozen
   B1 encoder, and emits `EmbeddingExample` objects plus lengths/masks. Cache the
   embeddings to disk.
2. **Run the first real training.** Use `run_training` with `configs/base.yaml` +
   `configs/gru.yaml` on the real train/dev split; tune batch size, learning
   rate, and epochs against dev EER.
3. **Add dev-set threshold selection** and report accuracy/precision/recall/F1/EER
   on the official eval split using the existing `evaluate`.
4. **Then scale to multiple datasets** (and cross-corpus evaluation) before
   treating the GRU as the B2 baseline.
5. **Then implement AASIST** behind the same `Detector`/config/checkpoint
   contracts and compare it against the GRU with identical metrics.
6. **Housekeeping:** resolve the index/RawNet2 staging issue, get a review commit
   on a proper B2 branch, and keep `main` untouched until review.

---

## 6. Dependencies and blockers

| Dependency | State / impact |
| --- | --- |
| B1 IndicWav2Vec frozen features | Contract exists (`src/backbones/tensor_interface.py`): `features [B,T,D]` detached float32, `valid_lengths` int64 `[B]` in output frames, `padding_mask` bool `[B,T]` True=padding, plus `frame_hop_ms`, `backbone_id`, `checkpoint_version`, `output_layer`. B2 uses the same key names, so integration is a data-plumbing task, not an interface redesign. D=1024 for the current encoder. |
| Encoder checkpoint | `configs/backbones.yaml` points to `models/indicwav2vec_large.pt` with SHA-256 enforcement; weights are git-ignored and must be provisioned per `docs/b1-setup.md`. Without them, real embeddings cannot be produced. |
| Fairseq environment | Required to run the encoder and `tests/backbones/`; not installed here, which is why B2 verification stopped at synthetic embeddings. |
| Real datasets | Not downloaded (no authorization given), no license/split decisions made. This is the main blocker for real training. |
| Compute | Verification was CPU-only; real training scale (epochs, batch size, runtime) is unmeasured. |
| Windows pytest temp/cache permissions | Known environment issue; B2 tests run with `-p no:cacheprovider`. It is not a code defect. |
| Frozen-encoder assumption | B2 treats embeddings as fixed inputs; there is no gradient path into the encoder by design. If end-to-end fine-tuning is ever wanted, it is a separate decision and out of B2 scope. |

---

## 7. Architectural decisions and ownership boundaries

**Ownership**

- B1 owns audio decode, mono conversion, resampling, normalization, the frozen
  IndicWav2Vec encoder, output-length derivation, and mask construction. B1 was
  not modified.
- B2 owns detector models, the embedding-sequence data contract and collation,
  training/validation loops, metrics, checkpoint format, B2 configs, and B2
  tests.
- Integration (audio -> B1 -> B2) is a new component that neither B1 nor B2 owns
  yet; it needs an explicit owner before real training starts.

**Decisions recorded**

- The final B2 direction is: custom GRU detector + AASIST. The original RawNet2
  implementation is explicitly **not** the final detector and was removed from
  the worktree.
- The detector consumes embeddings, never raw waveforms. The waveform/sinc
  front-end is not revived; if a RawNet2-style model is ever revisited it would
  need an embedding-compatible redesign and would not be the final model.
- Label convention: 0 = genuine, 1 = spoof; logits `[B, 2]`; score = P(spoof);
  threshold is configurable (`evaluation.threshold`, default 0.5).
- Padding convention: right-padded, `padding_mask` True = padding, padded frames
  zero-filled at collation; the GRU additionally excludes them from recurrence
  via packing and from pooling via the mask.
- The encoder is frozen: features are detached float32 tensors; B2 never
  backpropagates into B1.
- Model selection default: monitor EER, mode min, strict improvement (ties keep
  the earlier epoch). This is configurable via `checkpoint.best_metric` /
  `checkpoint.best_mode`.
- Config style: `configs/base.yaml` holds shared defaults; `configs/gru.yaml`
  overlays GRU-specific model/training values; later files win.
- Dependency discipline: B2 uses only `torch` plus `PyYAML`; metrics are
  implemented in-repo (no sklearn).
- Minimalism: the pipeline is intentionally small (no LR schedules, no AMP, no
  DataLoader workers, no class weighting) so it stays understandable until real
  data proves what is needed.

**Boundaries still to be respected by the next agent**

- Do not restore RawNet2; do not implement AASIST until the real-data baseline
  path is working.
- Do not modify B1, IndicWav2Vec, routing, or audio preprocessing.
- Do not download datasets or weights without explicit authorization.
- Do not commit or push `main`.

---

## 8. Reproduction quick reference

Focused B2 tests:

```
python -m pytest tests/test_data_batching.py tests/test_training_pipeline.py \
  tests/test_config.py tests/test_gru_detector.py tests/test_training_engine.py \
  tests/test_evaluation_engine.py tests/test_checkpoints.py tests/test_metrics.py \
  tests/test_model_registry.py tests/test_detector_interface.py \
  -p no:cacheprovider
```

Synthetic end-to-end run with the real 1024-dim GRU config (writes the best
checkpoint to `artifacts/gru/gru_best.pt`, which is git-ignored):

```python
from voxsentinel.config import load_config
from voxsentinel.data import synthetic_examples
from voxsentinel.training import run_training

config = load_config(["configs/base.yaml", "configs/gru.yaml"])
result = run_training(
    config,
    synthetic_examples(64, input_dim=1024, min_frames=20, max_frames=40, seed=0),
    synthetic_examples(32, input_dim=1024, min_frames=20, max_frames=40, seed=1),
)
print(result.best_epoch, result.best_metric, result.best_checkpoint)
```

Restoring and evaluating the best checkpoint:

```python
from voxsentinel.checkpoints import restore_checkpoint
from voxsentinel.data import collate_examples, synthetic_examples
from voxsentinel.detectors import create_detector
from voxsentinel.evaluation import evaluate

detector = create_detector(config.model)
restore_checkpoint(result.best_checkpoint, detector, model_name="gru")
print(evaluate(detector, [collate_examples(synthetic_examples(32, seed=1))]).metrics)
```
