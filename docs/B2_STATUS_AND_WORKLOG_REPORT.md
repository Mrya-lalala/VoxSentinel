# VoxSentinel B2 - status and worklog report

Generated 2026-09-15 from the live repository state. Audience: manager agent
and teammates who need to know exactly what exists and what was done without
reading chat history.

How to read this report:

- "Committed" means it is in git history and pushed to `origin/feature/gru`.
- "Working tree" means it exists on disk but is **not committed**.
- Nothing in this report is real spoof-detection performance. All measured
  classification numbers come from a synthetic fixture.

---

## 1. One-paragraph summary

B2 has a trainable custom GRU spoof head for frozen IndicWav2Vec embedding
sequences, plus a complete training/validation/best-checkpoint/evaluation,
metrics-and-threshold, config and batching stack with focused tests. That work
is committed as `baa9a60` on `feature/gru` (pushed, 1 commit ahead of
`origin/main`) **in the earlier `src/voxsentinel/` layout that the current B2
master prompt rejects**. A completed migration of the same code into the
project's original `src/<area>/` structure, with the reliability fixes the
master prompt requires, currently exists as **uncommitted working-tree changes**
and passes 46 focused tests. Real-data work has not started: no dataset, no
manifest, no audio->B1 bridge, no feature cache, no real B1 extraction (Fairseq
is not installed and `models/indicwav2vec_large.pt` is absent), and therefore no
real accuracy or EER exists. AASIST is not implemented (reserved name only) and
RawNet2 is not restored.

---

## 2. Repository state right now

| Item | Current value |
| --- | --- |
| Branch | `feature/gru` |
| HEAD | `baa9a60` "feat: add GRU detector and B2 training pipeline" |
| Upstream | `origin/feature/gru`, branch is up to date (in sync) |
| Relation to main | 1 commit ahead of `origin/main` (`827d526` "version 0 frozen IndicWav2Vec encoder") |
| Staged changes | none |
| Uncommitted changes | 39 tracked files modified/deleted (diff stat: +313 / -2061) plus 16 new source modules and 8 new test files untracked |
| This agent's commits/pushes | none in this step; `baa9a60` was not created by this agent |

### 2.1 What the committed commit contains

`baa9a60` adds 43 files / 2190 insertions and contains the **pre-migration**
state:

- `src/voxsentinel/**` - the rejected parallel package: GRU head, detector
  adapter, data batch + synthetic fixture, training engine/builders/trainer/
  runner, evaluation engine + metrics, checkpoints, config loader/schema
  (26 files in the committed tree).
- `configs/base.yaml`, `configs/gru.yaml`, `configs/aasist.yaml`.
- Root-level `tests/test_*.py` for the B2 pipeline.
- `pyproject.toml` (a B2-added packaging/pytest file).
- Fixed `.gitignore` (merge markers removed, caches/venvs/env ignored).
- `docs/B2_GRU_MANAGER_HANDOFF.md` (first version).

### 2.2 What the uncommitted working tree contains

The master-prompt migration plus hardening, not yet committed:

- Deleted: `pyproject.toml`, all of `src/voxsentinel/**`, all B2 root
  `tests/test_*.py`.
- Modified: `tests/conftest.py`, `tests/test_config.py`,
  `docs/B2_GRU_MANAGER_HANDOFF.md`.
- Untracked new source: `src/config.py`, `src/data/{__init__,batch,labels,
  synthetic}.py`, `src/detectors/{__init__,base,checkpoints,gru,registry,
  runner,training}.py`, `src/scoring/{__init__,evaluation,metrics,
  thresholds}.py`.
- Untracked new tests: `tests/data/test_batch.py`,
  `tests/detectors/{test_gru,test_training,test_checkpoints,test_registry}.py`,
  `tests/scoring/{test_metrics,test_thresholds,test_evaluation}.py`.

**Consequence:** the pushed branch still shows the rejected layout. The
corrected structure and the reliability fixes exist only locally until the
migration is committed and pushed.

---

## 3. What we have now

### 3.1 B1 assets (pre-existing; not modified)

- `src/backbones/`: `indic_wav2vec.py`, `tensor_interface.py`, `base.py`,
  `registry.py`, `router.py`, `schemas.py`, `wavlm.py`, `errors.py`.
- `configs/backbones.yaml`: IndicWav2Vec checkpoint path/SHA-256, expected
  embedding dim 1024, device `cpu`, output layer, routing policy.
- `scripts/`: encoder setup, fixture preparation, extraction and verification
  commands.
- `docs/`: B1 execution report, B1 team handout, b1-setup guide.
- Pinned runtime: `.python-version` = 3.10.18, `requirements.txt`,
  `requirements-bootstrap.txt`, `requirements-fixtures.txt`.
- Contract B2 consumes: `features [B,T,D]` detached FP32 (zero in padding),
  `valid_lengths [B]` int64 **output frames**, `padding_mask [B,T]` bool
  **True = padding**, `frame_hop_ms` = 20.0, `backbone_id`,
  `checkpoint_version`, `output_layer`; D = 1024; encoder frozen and in eval
  mode during extraction.

### 3.2 B2 code (current, working tree)

| Path | What it provides |
| --- | --- |
| `src/detectors/gru.py` | `GruSpoofDetector`: LayerNorm(1024) -> Linear(1024,256) -> 1-layer unidirectional GRU -> valid-frame mean pooling -> Dropout -> Linear(256,2); `GruDetector` batch adapter; registration as `gru` |
| `src/detectors/base.py` | `Detector` contract, `DetectorOutput` (`scores` = softmax[...,1] = uncalibrated P(spoof)), cross-entropy `loss` |
| `src/detectors/registry.py` | name -> detector factory; `gru` implemented, `aasist` reserved (raises `NotImplementedError`) |
| `src/detectors/checkpoints.py` | `build/save/load/restore_checkpoint`; format v1 payload (model state, model name/config, optional optimizer state, epoch, global step, metrics, optional metadata for encoder identity/threshold/manifest hashes) |
| `src/detectors/training.py` | `train_epoch` (example-weighted loss, finite-loss check), `build_optimizer` (adamw/adam/sgd), `build_loss` (cross-entropy), `fit` (train -> validate -> best checkpoint), `EpochResult`/`EpochSummary`/`TrainingResult`, device assertions |
| `src/detectors/runner.py` | `run_training(config, train_examples, val_examples, checkpoint_path=None, device=None, metadata=None)` |
| `src/data/labels.py` | label convention 0=genuine, 1=spoof; validation before integer conversion (fractional/boolean rejected) |
| `src/data/batch.py` | `EmbeddingExample` `[T,D]` + label; `collate_examples` -> `features [B,T,D]`, `valid_lengths [B]`, `padding_mask [B,T]`, `labels [B]`; `batch_iterator` with seeded shuffling |
| `src/data/synthetic.py` | synthetic genuine/spoof fixture for plumbing checks only |
| `src/scoring/metrics.py` | accuracy, precision, recall, F1, example counts, genuine-false-alarm rate, spoof-miss rate, O(N log N) EER with grouped ties and both curve endpoints |
| `src/scoring/thresholds.py` | `select_threshold`: realizable dev threshold closest to the empirical crossing, documented tie-breaks, requires both classes |
| `src/scoring/evaluation.py` | `evaluate`: aggregates scores/labels, example-weighted loss (uses the same configured loss as training), returns `EvaluationResult` |
| `src/config.py` | `AppConfig` dataclasses and YAML overlay `load_config` |

### 3.3 Configs

- `configs/base.yaml`: AdamW lr 1e-3 wd 0.0; batch 8, epochs 1, device cpu,
  `use_amp false`, shuffle true, seed 0, loss `cross_entropy`; threshold 0.5;
  checkpoint format 1, strict load, directory `artifacts/gru`, best metric
  `eer`, mode `min`.
- `configs/gru.yaml`: `model.name gru` with `input_dim 1024`, `hidden_size
  256`, `num_layers 1`, `dropout 0.0`, `num_classes 2`; AdamW lr 1e-3;
  batch 8, epochs 10, cross-entropy; EER/min selection; `artifacts/gru`.
- `configs/aasist.yaml`: placeholder `model.name aasist`, empty parameters.
- `configs/backbones.yaml`: B1 encoder configuration (unchanged).

### 3.4 Tests present on disk (current layout)

| Test file | Tests |
| --- | --- |
| `tests/data/test_batch.py` | 7 |
| `tests/detectors/test_gru.py` | 9 |
| `tests/detectors/test_training.py` | 6 |
| `tests/detectors/test_checkpoints.py` | 2 |
| `tests/detectors/test_registry.py` | 3 |
| `tests/scoring/test_metrics.py` | 7 |
| `tests/scoring/test_thresholds.py` | 3 |
| `tests/scoring/test_evaluation.py` | 4 |
| `tests/test_config.py` | 3 |
| `tests/backbones/test_embedding_contract.py` | 2 (B1, untouched) |

`tests/backbones/test_indic_encoder.py` and `tests/backbones/test_router.py`
exist but need the Fairseq runtime and were not run here.

### 3.5 Docs

- `docs/B2_GRU_MANAGER_HANDOFF.md` - corrected B2 handoff (current layout,
  contracts, evidence, blockers, next steps).
- `docs/B2_STATUS_AND_WORKLOG_REPORT.md` - this report.
- `docs/B1_*` and `docs/b1-setup.md` - B1 documents (pre-existing).

---

## 4. What we did (worklog, in order)

1. **Surveyed the existing B2 code.** Found a waveform-based RawNet2 model and
  adapter plus generic checkpoint/metrics/config/training scaffolding, and no
  dataset/manifest layer.
2. **Replaced RawNet2 with the custom GRU head over embeddings.** Implemented
  LayerNorm -> Linear -> unidirectional GRU -> valid-frame masked mean pooling ->
  Dropout -> 2 logits; configurable `input_dim` (default 1024); packed sequences
  so padded frames never enter the recurrence; masked pooling so they never
  contribute to the pooled vector; genuine/spoof logit convention documented;
  config (`configs/gru.yaml`), registry entries (`gru` implemented, `aasist`
  reserved), and tests including padding invariance, variable lengths,
  trainability and checkpoint round trip.
3. **Removed the RawNet2 implementation** (`models/rawnet2.py`,
  `detectors/rawnet2.py`, `configs/rawnet2.yaml`, `tests/test_rawnet2.py`) from
  the working tree so RawNet2 is not the final detector.
4. **Repaired `.gitignore`.** Removed merge-conflict markers, kept Python
  caches/`.pytest_cache`/`*.egg-info`/venvs/`.env` ignored, and fixed the
  over-broad `models/` rule that was hiding a source module.
5. **Wrote the first B2 manager handoff report** describing the committed
  layout at that time.
6. **Built the training pipeline** around the head: labelled embedding-example
  contract and collation, synthetic fixture, training config (batch size,
  optimizer, loss, epochs, seed/shuffle), optimizer/loss factories, epoch
  driver with validation and best-checkpoint selection, config-driven
  `run_training`, metrics reuse, checkpoint metadata, and end-to-end tests
  including a real-dimension (1024) synthetic sanity run.
7. **Read the four manager documents in the required order** (B2 master prompt
  first, then 01-AGENT-MASTER, the project AI-agent handoff, then the B1
  handout) and reconciled them with the checkout: the original structure is
  flat `src/<area>/`, B2's `src/voxsentinel/` package was accidental, and the
  master prompt lists concrete defects to fix.
8. **Priority 0 - restored the original structure.** Migrated GRU, adapters,
  training, batching, metrics, checkpoints and config into `src/detectors/`,
  `src/data/`, `src/scoring/` and `src/config.py`; collapsed the duplicate
  model/detector registries into one; moved B2 tests to `tests/<area>/`;
  deleted `src/voxsentinel/`, `pyproject.toml` and the `tests/conftest.py`
  sys.path shim; removed the leftover package directory; verified no imports or
  configs still reference `voxsentinel`.
9. **Priority 1 - hardened the pipeline** (details in section 5): device
  handling, example-weighted losses, one objective for training and
  validation, O(N log N) EER with a documented crossing policy, realizable
  threshold selection, strict label/length/mask/feature validation, empty and
  non-finite failure paths, one-shot iterator policy, and optional checkpoint
  metadata.
10. **Re-verified everything and rewrote the handoff** to match the corrected
  implementation, with explicit "synthetic only / no real-data numbers"
  statements and the remaining blockers.

---

## 5. Verified evidence (exact, from this checkout)

Environment: Windows, Python 3.14.3, `torch 2.14.0+cpu`, pytest 9.1.1, CPU.

### 5.1 Test suite

```
python -m pytest tests/data tests/detectors tests/scoring tests/test_config.py \
  tests/backbones/test_embedding_contract.py -p no:cacheprovider
```

Result: **46 passed in 2.19s** (44 B2 + 2 B1 embedding-contract tests).

Covered behaviour: padded frames ignored through collated batches; variable
lengths and lengths/mask equivalence; stricter mask accepted; left padding and
mask holes rejected; integer length/boolean mask/non-finite feature validation;
GRU gradients reach projection/GRU/classifier; example-weighted short-batch
loss (0.75 for 3x1.0 + 1x0.0); empty training epoch rejected; device-mismatch
rejection; one-shot generator rejection; best-checkpoint contents and restore;
registry behaviour; EER on perfect/reversed/all-tied/partial-tie/one-class
inputs; threshold selection; evaluation aggregation and configured-loss use;
config overlay validation.

### 5.2 Synthetic end-to-end run at the real embedding dimension

`run_training` with `configs/base.yaml` + `configs/gru.yaml`, 64 train / 32
validation synthetic examples (20-40 frames), batch 8, 3 epochs, temporary
checkpoint, metadata `{"encoder": "indicwav2vec-large", "output_layer": null}`:

| Epoch | Train loss | Val loss | Accuracy | F1 | EER | Genuine false alarms | Spoof misses |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.3576 | 0.1947 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 2 | 0.0644 | 0.0071 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |
| 3 | 0.0008 | 0.0003 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

Best epoch 1 (EER ties at 0; only strict improvement replaces the best);
checkpoint and metadata written. Restoring that checkpoint reproduces the saved
accuracy and EER. This is a separable synthetic fixture, not a benchmark.

### 5.3 Other verified facts

- GRU parameter count at the default configuration: **659,714 trainable
  parameters**.
- `available_detectors()` returns `('aasist', 'gru')`; creating the `aasist`
  detector raises `The aasist detector has not been implemented yet.`
- `src/backbones/` and `tests/backbones/` show no modifications.
- No `voxsentinel` references remain in `src/`, `tests/`, `configs/` or
  `scripts/` (only an unrelated temp-path string in B1's `docs/b1-setup.md`).

---

## 6. What we do NOT have, and what is blocked

| Missing item | Why it matters / blocker |
| --- | --- |
| Real dataset (ASVspoof / IndicSynth / Svarah / other) | No genuine/spoof audio, manifests, splits or licenses in the checkout; nothing downloaded, no authorization given |
| Manifest + split/leakage audit | A1/A2-owned interface not yet agreed; needed to avoid speaker/source/generator leakage |
| Audio -> frozen B1 bridge | Needs the B1 runtime; not implemented |
| Feature cache (FP32 CPU, per-row valid frames, identity fields) | Not implemented |
| Real audio -> head -> loss -> backward smoke test | Needs audio + B1 + cache; not performed |
| Real train/dev/holdout baseline and held-out metrics | Not performed; no real accuracy/EER exists |
| Threshold frozen on dev and reported error budget | `select_threshold` exists and is tested, but never applied to real dev scores |
| Fairseq runtime | Not installed here (`import fairseq` fails); required for real extraction and two B1 tests |
| B1 checkpoint | `models/indicwav2vec_large.pt` is absent (git-ignored, SHA-256 enforced by config) |
| AASIST | Not implemented; only the reserved name and empty config placeholder |
| AMP / ONNX / distributed / schedules / early stopping / class weighting | Deliberately not implemented; no demonstrated need yet |
| CLI commands for manifest validation, cache build, threshold selection, evaluation, inference | Not yet implemented; today the entry point is `src.detectors.runner.run_training` in Python |
| Committed migration | The corrected structure and hardening are uncommitted; the pushed branch still has the rejected layout |

---

## 7. Immediate next steps (in order)

1. **Commit and push the migration** (per current authorization) so
   `origin/feature/gru` contains the corrected `src/<area>` structure, the
   hardening fixes, and the updated handoff - and so the rejected
   `src/voxsentinel/` tree and `pyproject.toml` are gone from history going
   forward. Verify `tests/` and imports in a fresh checkout afterwards.
2. **Agree the manifest contract with A1/A2** (stable id, audio reference,
   label, language/accent, dataset/version, split, speaker id, source
   recording, segment offsets, leakage-group id, generator family, provenance,
   chunk-policy version) and freeze splits before chunking.
3. **Implement the audio -> frozen B1 bridge** with row-order/provenance
   preservation and per-row caching of `features[i, :valid_lengths[i]]` only.
4. **Implement the bounded cache identity + lazy dataset** (encoder id,
   checkpoint SHA-256, output layer including `None`, embedding dim, frame hop,
   chunk policy, schema version; atomic writes; resumable).
5. **Real-audio integration smoke test**: decode -> B1 -> collate -> head loss
   -> backward; verify head gradients/updates, detached features, and no
   encoder gradients or weight changes.
6. **Tiny real overfit diagnostic**, then the real run: train, select the best
   checkpoint on dev EER, load it into a fresh detector, select and freeze the
   dev threshold, then evaluate the reserved holdout once and report EER plus
   confusion counts, false-alarm rate, miss rate, precision, recall, F1 and
   accuracy.
7. **Add the documented CLI commands** for the steps above.
8. **Only then implement AASIST** behind the same `Detector`/config/checkpoint
   contracts and compare on identical splits, encoder layer and reporting
   protocol.

---

## 8. Operating rules and ownership boundaries in force

- B1 owns decode/resample/mono/normalization, frozen extraction, output lengths
  and masks; B1 is not modified and routing is untouched.
- A1 owns manifests/splits/labels; A2 owns audio chunking; the bridge contract
  must be agreed with them.
- B2 owns detector heads/adapters, optimizer/loss, training loop, checkpoint
  identity and checkpoint selection.
- `src/scoring/` is C1's designated home; B2's metrics/threshold/evaluation
  code there is an initial implementation to extend in place, not a parallel
  stack.
- Label/score convention: 0 = genuine, 1 = spoof, logits `[B,2]`,
  score = softmax[...,1] = P(spoof), `score >= threshold` means spoof; the
  score is uncalibrated and must not be presented as a probability of fraud.
- Padding convention: right padding only, True = padding, padded frames are
  never valid audio.
- Frozen-encoder assumption: the head consumes detached FP32 features and
  never trains B1.
- Not to be done in this milestone: restore RawNet2, implement AASIST before a
  real GRU baseline, add AMP/distributed/ONNX complexity, download datasets or
  weights, touch B1/audio/API/UI/deployment, or merge to `main`.
- Never present fixture numbers as spoof-detection performance.

---

## 9. Appendix

### 9.1 Current working-tree inventory (verified)

- Source: `src/__init__.py`, `src/config.py`,
  `src/data/{__init__,batch,labels,synthetic}.py`,
  `src/detectors/{__init__,base,checkpoints,gru,registry,runner,training}.py`,
  `src/scoring/{__init__,evaluation,metrics,thresholds}.py`,
  `src/backbones/*` (B1).
- Tests: `tests/conftest.py`, `tests/test_config.py`,
  `tests/data/test_batch.py`,
  `tests/detectors/{test_gru,test_training,test_checkpoints,test_registry}.py`,
  `tests/scoring/{test_metrics,test_thresholds,test_evaluation}.py`,
  `tests/backbones/*` (B1).
- Configs: `configs/{base,gru,aasist,backbones}.yaml`.
- Scripts: `scripts/{extract_indic.py,prepare_encoder_fixtures.py,
  setup_encoder_env.sh,verify_indic.py}`.
- Docs: `docs/{B1_MANAGER_EXECUTION_REPORT,B1_TEAM_AGENT_HANDOUT,b1-setup,
  B2_GRU_MANAGER_HANDOFF,B2_STATUS_AND_WORKLOG_REPORT}.md`.
- Project files: `README.md`, `.gitignore`, `.python-version` (3.10.18),
  `requirements.txt`, `requirements-bootstrap.txt`, `requirements-fixtures.txt`.

### 9.2 Commands used for this report

```
git branch --show-current; git log --oneline -6; git status
git show --stat --oneline HEAD
git diff --stat
python -m pytest tests/data tests/detectors tests/scoring tests/test_config.py \
  tests/backbones/test_embedding_contract.py -p no:cacheprovider
```

### 9.3 Synthetic-only reproduction (writes a temp checkpoint)

```python
import tempfile

from src.config import AppConfig, TrainingConfig, load_config
from src.data import collate_examples, synthetic_examples
from src.detectors import create_detector, restore_checkpoint, run_training
from src.scoring import evaluate, select_threshold

base = load_config(["configs/base.yaml", "configs/gru.yaml"])
config = AppConfig(
    model=base.model,
    optimizer=base.optimizer,
    training=TrainingConfig(batch_size=8, epochs=3, device="cpu", loss="cross_entropy"),
    evaluation=base.evaluation,
    checkpoint=base.checkpoint,
)
train = synthetic_examples(64, input_dim=1024, min_frames=20, max_frames=40, seed=0)
val = synthetic_examples(32, input_dim=1024, min_frames=20, max_frames=40, seed=1)

with tempfile.TemporaryDirectory() as tmp:
    result = run_training(config, train, val, checkpoint_path=f"{tmp}/gru_best.pt")
    detector = create_detector(config.model)
    restore_checkpoint(result.best_checkpoint, detector, model_name="gru")
    evaluation = evaluate(detector, [collate_examples(val)])
    selection = select_threshold(evaluation.scores, evaluation.labels)
    print(result.best_epoch, result.best_metric, evaluation.metrics, selection)
```
