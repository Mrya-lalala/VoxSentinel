# AASIST implementation plan (embedding-adapted, Gate 1)

Prepared 2026-09-20 on branch `feature/aasist` at `a00e290`. This document is
the Gate 1 deliverable required before any implementation. It records the
upstream mapping, the verified tensor/graph behaviour, the exact adaptation
design and its intentional deviations, the integration plan, and the local
asset status. No implementation file exists yet.

Scope reminder: this is an **embedding-adapted AASIST research head** evaluated
on frozen IndicWav2Vec embeddings. It is not a reproduction of the waveform
AASIST and is not promised to beat the GRU. No training was run for this plan.

---

## 1. Provenance and licence

Reference inspected (not added to this repository; fetched read-only into a
temp directory outside the checkout):

- Repository: https://github.com/clovaai/aasist
- Commit: `a04c9863f63d44471dde8a6abcb3b082b07cd1d1` (inspected 2026-09-20)
- Files and SHA-256 of the fetched copies:

| Upstream path | SHA-256 |
| --- | --- |
| `models/AASIST.py` | `9E0D3E80937DD0577BEEA7883098465A479DA23A198EBC0D712ABCC59B0BEC50` |
| `config/AASIST.conf` | `C25023331685027CCE90E1B9A0D2DF10AA04B2A27D9B27D5AFA36E6815B0FE76` |
| `LICENSE` | `DA2E79B8592D166EF505224300968B80EBE1E4C217C43B94A5EC627D81CD4142` |
| `NOTICE` | `70EDB07F04DDC88E7155D6E826495E5EF99C3B1F0BD4E1B7FA35897B3854DBD0` |

- Paper: https://arxiv.org/abs/2110.01200

Licence: AASIST is MIT (`Copyright (c) 2021-present NAVER Corp.`). The upstream
`NOTICE` additionally lists subcomponents: ASVspoof 2021 `Baseline-RawNet2`
(MIT, eurecom-asp), `Jungjee/RawNet` (MIT, Jee-weon Jung), and the ASVspoof2019
t-DCF Python package (**CC BY-NC-SA 4.0**). Only the MIT-licensed graph
components of `models/AASIST.py` are reused here
(`GraphAttentionLayer`, `HtrgGraphAttentionLayer`, `GraphPool`). The waveform
front-end (`CONV`, `Residual_block`) is not reused, and the t-DCF package is
not imported or copied.

Required attribution action for Gate 2: carry the NAVER/MIT notice in the
adapted module header (and, if a separate graph-block module is created, in
that module and in a small adjacent provenance/notice file), stating that the
code is adapted and listing the modifications.

---

## 2. Repository state and local asset availability (inspected)

- Git root `C:/Users/anany/VoxSentinel`; branch `feature/aasist`; HEAD `a00e290`
  (= `main` = `origin/main`); working tree clean; branch is local only.
- AASIST is absent: `configs/aasist.yaml` holds only `model.name: aasist` with
  empty parameters, `"aasist"` is listed in `src/detectors/registry.py`
  `_RESERVED`, and `tests/detectors/test_registry.py` asserts it raises
  `NotImplementedError`. No AASIST code, config values, tests or docs exist.
- Reusable B2 contracts confirmed present: `Detector`/`DetectorOutput` in
  `src/detectors/base.py`; `FrameStandardizer` and `fit_detector_standardizer`
  in `src/detectors/standardization.py`; checkpoint build/save/load/restore in
  `src/detectors/checkpoints.py`; `train_epoch`/`fit`/`build_optimizer`/
  `build_loss` in `src/detectors/training.py`; `EmbeddingExample`,
  `collate_examples`, `batch_iterator` in `src/data/batch.py`; labels in
  `src/data/labels.py`; metrics in `src/scoring/metrics.py`; release/predict
  conventions in `src/scoring/predict.py` (GRU-specific, to be left unchanged).
- Encoder contract (`src/backbones/tensor_interface.py`): detached float32
  `features [B,T,D]`, `valid_lengths [B]` in output frames, `padding_mask
  [B,T]` bool with True = padding, `True`-ignore polarity, right padding only,
  D = 1024, frame hop 20 ms (1 s ~ 49 frames, 4 s ~ 199 frames).
- Local assets: **v3 caches absent** (`artifacts/datasets-v3-coverage/` does
  not exist). `artifacts/datasets/manifests/` holds acquisition manifests only
  (600 recording rows, no audio or features). Only v1-era GRU checkpoints exist
  (`artifacts/runs/gru-core-v1/`). `fairseq` is not installed and there is no
  `.venv`. Therefore Gates 3-5 cannot run in this checkout; Gates 1-2 can.

---

## 3. Verified upstream architecture (`models/AASIST.py`, 607 lines)

### 3.1 Classes and roles

| Class | Role |
| --- | --- |
| `GraphAttentionLayer` | Homogeneous dense graph attention over one node set |
| `HtrgGraphAttentionLayer` | Heterogeneous attention over two node types plus master node |
| `GraphPool` | Learned top-k node selection with score weighting |
| `CONV` | Fixed sinc/mel filterbank front-end (waveform only) |
| `Residual_block` | 2-D residual conv block of the waveform encoder |
| `Model` | Front-end + encoder + two-branch graph backend + readout |

### 3.2 Upstream shape flow (config: `nb_samp 64600`, `first_conv 128`,
`filts [70,[1,32],[32,32],[32,64],[64,64]]`, `gat_dims [64,32]`,
`pool_ratios [0.5,0.7,0.5,0.5]`, `temperatures [2.0,2.0,100.0,100.0]`)

```
waveform [B, 64600]
  -> CONV(70 filters, kernel 128, stride 1)          [B, 70, 64473]
  -> unsqueeze(1) + abs + MaxPool2d((3,3))           [B, 1, 23, 21491]
  -> BatchNorm2d(1) + SELU
  -> 6 x Residual_block (conv2d H preserved, MaxPool2d((1,3)))
                                                     e: [B, 64, 23, 29]
  -> S branch: max|.| over time -> transpose + pos_S [B, 23, 64]
       -> GAT_layer_S (64->64, temp 2.0)             [B, 23, 64]
       -> pool_S (0.5)                               [B, 11, 64]
  -> T branch: max|.| over freq -> transpose         [B, 29, 64]
       -> GAT_layer_T (64->64, temp 2.0)             [B, 29, 64]
       -> pool_T (0.7)                               [B, 20, 64]
  -> masters master1/master2: [1,1,64]
  -> branch 1: ST11 (64->32, temp 100) -> pools 0.5 [B,5,32]/[B,10,32]
               ST12 (32->32, temp 100) + residual
  -> branch 2: ST21 (64->32, temp 100) -> pools 0.5 [B,5,32]/[B,10,32]
               ST22 (32->32, temp 100) + residual
  -> elementwise max across branches; Dropout(0.2)
  -> readout: concat[T_max, T_avg, S_max, S_avg, master] = 5*32 = 160
  -> Dropout(0.5) -> Linear(160, 2)
```

Notes verified in code: `23` is the encoder's frequency-node count
(`pos_S = Parameter(1, 23, 64)`), `29` is the residual block's time output for
64600 samples, and `pool_ratios[3]` / `temperatures[3]` are never consumed.

### 3.3 Graph construction and attention semantics

- **No adjacency matrix.** Both attention layers build a dense all-pairs map
  by pairwise node multiplication (`x[:, :, None, :] * x[:, None, :, :]`),
  project with `tanh`, score with a learned weight vector, divide by
  `temperature`, and `softmax` over `dim=-2` (the source-node axis of the
  `[B, N, N, 1]` map). The "graph" is complete within each node set.
- `GraphAttentionLayer`: `out = proj_with_att(att^T x) + proj_without_att(x)`,
  then `BatchNorm1d(out_dim)` applied to the flattened `(B*N, out_dim)`, then
  `SELU`. Input dropout `p=0.2`.
- `HtrgGraphAttentionLayer`: projects each type with its own `proj_type1/2`,
  concatenates them, and builds a **type-masked attention board** from three
  learned weight vectors: `att_weight11` (type1-type1), `att_weight22`
  (type2-type2) and `att_weight12` shared by both cross-type blocks. Master
  attention is separate: `att_map = softmax(tanh(att_projM(x * master)) *
  att_weightM / temp, dim=-2)`; the master update is
  `proj_with_attM(att^T x) + proj_without_attM(master)`. Nodes then receive
  `proj_with_att(att^T x) + proj_without_att(x)`, `BatchNorm1d`, `SELU`.
  The layer returns `(x_type1, x_type2, master)` and does not feed the master
  back into node attention.
- **Master nodes are learnable parameters** `[1, 1, gat_dims[0]]`. The two
  `self.masterN.expand(x.size(0), ...)` statements are dead code: the first
  heterogeneous call passes `self.masterN` unexpanded and relies on implicit
  broadcasting (`[B,N,dim] * [1,1,dim]`). After the first call the master is
  materialized as `[B, 1, dim]` and is used in the branch residual
  (`master = master + master_aug`).
- `GraphPool`: `scores = sigmoid(Linear(dropout(h)))`, `k = max(int(N*k), 1)`,
  `topk` over nodes, then `h = gather(h * scores, idx)`. It is node selection
  with score weighting, not aggregation; dropout `p=0.3`.
- **Branch/readout**: two independent branches (`ST11+ST12` with `master1`,
  `ST21+ST22` with `master2`), pooled identically (0.5) so shapes match, then
  fused by elementwise `max` on T nodes, S nodes and master. Readout
  concatenates `max|.|` and `mean` over nodes for both node sets plus the
  master, applies `Dropout(0.5)` and `Linear(5*gat_dims[1], 2)`.
- `Model.forward` returns `(last_hidden, output)`; training uses `output`.
  `Freq_aug` masks raw sinc filters and is waveform-specific.
- Normalization inventory: `BatchNorm2d(1)` after the front-end pool;
  `BatchNorm1d(out_dim)` inside every attention layer (over `B*N` rows);
  `GraphPool` has no norm; there is a `Dropout(0.5, inplace=True)` before the
  classifier and `Dropout(0.2, inplace=True)` for the branch fusion.
- No external graph libraries are used: only `torch`, `torch.nn`,
  `torch.nn.functional`, `numpy`, `random`.

### 3.4 Upstream training recipe (`config/AASIST.conf`)

Adam (`betas 0.9/0.999`, `lr 1e-4`, `weight_decay 1e-4`), cosine schedule to
`5e-6`, `batch_size 24`, `num_epochs 100`, loss `CCE` (categorical
cross-entropy), 64600-sample waveforms, checkpoint `models/weights/AASIST.pth`.
Recorded for provenance only; this task uses the GRU-matched pilot budget
(section 6.4).

---

## 4. Comparison with the proposed `[B,T,1024]` adapter

Our pipeline has no waveform and no time-frequency map. Every upstream
assumption that depends on waveform geometry must be replaced or re-parameterized:

| Upstream element | Depends on | Embedding-adapted replacement |
| --- | --- | --- |
| `CONV` sinc filterbank, `first_conv`, `nb_samp`, `filts` | raw waveform, fixed 64600 samples | dropped; `1024 -> C*F` learned linear projection |
| `Residual_block` encoder producing `[B,64,23,29]` | 2-D time-frequency grid | dropped; valid-slice temporal pooling to `K=32` bins |
| `pos_S` shape `[1,23,64]` | 23 frequency nodes | learned type parameter `[1,F,C]` with `F=16` |
| node feature dim 64 | 64 encoder filters | `C=32` (adapter channels) |
| fixed-length waveform batches | 64600 samples | `[B,T,1024]` + `valid_lengths` + `padding_mask`; right padding excluded before any pooling |
| `BatchNorm2d(1)` after front-end pool | waveform conv output | dropped (front-end removed; `LayerNorm` retained per frame) |
| `Freq_aug` filter masking | sinc filters | dropped (not applicable) |
| `max|.|` reductions over time/frequency | 2-D grid | kept as `max|.|` over K (temporal nodes) and over F (latent-feature nodes) |
| GAT/HS-GAL/GraphPool/two-branch readout | dimension-parameterized | kept, with `in_dim = C` |
| master broadcast | implicit | explicit expansion to `[B,1,64]` |

The F axis is a **learned latent-feature axis**, not a physical frequency axis;
no claim is made that embedding coordinates are spectral. Uniform temporal
pooling to 32 bins compresses local detail and may discard spoof cues; this is
a stated limitation of the first comparison.

---

## 5. Final Gate 1 architecture (exact)

### 5.1 Input validation (before any tensor math)

1. `features` must be `[B,T,1024]`, float, finite; `T >= 32` (reject shorter
   with a clear error; no repetition, no silent padding of speech).
2. `valid_lengths` integer `[B]` within `[32, T]`; `padding_mask` boolean
   `[B,T]`; when both are given they must agree; right-prefix only (holes and
   left padding rejected); no empty sequence after masking.
3. Reuse the same validation conventions as `GruSpoofDetector._valid_mask`
   (True = padding, contiguous prefix).

### 5.2 Adapter

1. `FrameStandardizer(1024)` applied on valid frames only (fixed buffers,
   serialized in the checkpoint, mandatory state; fitted on TRAIN frames only
   via the existing `fit_detector_standardizer`).
2. `nn.LayerNorm(1024)` (learnable, per frame), kept after the fixed
   standardizer, mirroring the GRU head and separating fixed scaling from
   learnable normalization.
3. Valid-slice temporal pooling: per item take `x[i, :len_i]`, transpose to
   `[1024, len_i]`, `adaptive_avg_pool1d` to `K = 32` bins, transpose back to
   `[B, 32, 1024]`. Padded positions never enter the pool.
4. Projection: `nn.Linear(1024, C*F)` with `C = 32`, `F = 16` (512 outputs),
   reshaped with explicit assertions to `[B, C, F, K] = [B, 32, 16, 32]`.

### 5.3 Node construction (upstream max-absolute convention)

- Temporal nodes `T`: `max|.|` over the F axis -> `[B, C, K]` -> transpose ->
  `[B, K, C] = [B, 32, 32]`.
- Latent-feature nodes `S`: `max|.|` over the K axis -> `[B, C, F]` ->
  transpose -> `[B, F, C] = [B, 16, 32]`; add learned `pos_F` `[1, 16, 32]`.
- Both node sets are fixed-size, so the upstream backend runs with no padded
  graph nodes and no per-item graph bookkeeping.

### 5.4 Graph backend (upstream blocks, parameterized)

| Stage | Configuration | Input -> output nodes |
| --- | --- | --- |
| `GAT_layer_S` | `GraphAttentionLayer(C=32 -> 64, temperature 2.0)` | `[B,16,32] -> [B,16,64]` |
| `pool_S` | `GraphPool(0.5, 64, dropout 0.3)` | `16 -> 8` |
| `GAT_layer_T` | `GraphAttentionLayer(32 -> 64, temperature 2.0)` | `[B,32,32] -> [B,32,64]` |
| `pool_T` | `GraphPool(0.7, 64, dropout 0.3)` | `32 -> 22` |
| `ST11` / `ST21` | `HtrgGraphAttentionLayer(64 -> 32, temperature 100.0)` | T 22 + S 8 = 30 nodes |
| `pool_hS1`, `pool_hS2` | `GraphPool(0.5, 32, 0.3)` | `8 -> 4` |
| `pool_hT1`, `pool_hT2` | `GraphPool(0.5, 32, 0.3)` | `22 -> 11` |
| `ST12` / `ST22` | `HtrgGraphAttentionLayer(32 -> 32, temperature 100.0)` | T 11 + S 4 = 15 nodes, residual add |

Masters: `master1`, `master2` `[1,1,64]`, expanded to `[B,1,64]` before the
first heterogeneous call. Branch fusion: elementwise `max` on T nodes, S nodes
and master. Readout: `concat[T_max, T_avg, S_max, S_avg, master]` =
`5 * 32 = 160` -> `Dropout(0.5)` -> `Linear(160, 2)`.

Output: `DetectorOutput(logits [B,2])`; class 0 = genuine, class 1 = spoof;
score = `softmax(logits, dim=1)[:, 1]`; margin = `logits[:,1] - logits[:,0]`.

### 5.5 Dropout (configurable; upstream values as defaults)

attention input `0.2`, graph-pool `0.3`, branch fusion `0.2`, readout `0.5`.
Exposed as config parameters so the Gate 3 diagnostic can zero them via an
explicit override without code edits; production-pilot values stay separate.
Implementations use non-inplace dropout (behaviour-preserving change).

### 5.6 Proposed `configs/aasist.yaml` (applied only in Gate 2)

```yaml
model:
  name: aasist
  parameters:
    input_dim: 1024
    pooled_bins: 32          # K, minimum input frames
    channels: 32             # C
    latent_nodes: 16         # F
    gat_dims: [64, 32]
    pool_ratios: [0.5, 0.7, 0.5]
    temperatures: [2.0, 2.0, 100.0]
    dropout_attention: 0.2
    dropout_pool: 0.3
    dropout_fusion: 0.2
    dropout_readout: 0.5
    feature_standardization: true
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0
training:
  batch_size: 8
  epochs: 10
  device: cpu
  use_amp: false
  shuffle: true
  seed: 0
  loss: cross_entropy
evaluation:
  threshold: 0.5
checkpoint:
  format_version: 1
  strict_load: true
  directory: artifacts/aasist
  best_metric: eer
  best_mode: min
```

---

## 6. Intentional deviations from upstream (with reasons)

1. **Waveform front-end removed** (sinc/conv + 6 residual blocks +
   `BatchNorm2d(1)` + absolute max-pool). Reason: only frozen embeddings are
   available; no waveform path, no faked time-frequency reshaping.
2. **Adapter added** (fixed frame standardizer + learnable per-frame LayerNorm +
   valid-slice 32-bin pooling + linear projection). Reason: maps `[B,T,1024]`
   onto the graph backends; the standardizer matches the current GRU protocol.
3. **`pos_S` hardcoded 23 -> `pos_F` sized `F=16`.** Reason: our latent-feature
   node count is `F`, not the waveform encoder's frequency-bin count.
4. **Node feature dim 64 -> `C=32`.** Reason: bounded adapter
   (~596k head parameters). The graph blocks are dimension-parameterized, so
   this is a configuration choice; `C=64` remains available and would mirror
   the upstream `in_dim` exactly (documented alternative, not the default).
5. **Fixed node counts `F=16`, `K=32`.** Reason: bounded all-pairs attention
   and no padded graph nodes; `T < 32` is rejected instead of repeating or
   padding speech.
6. **Masters expanded explicitly** to `[B,1,64]` (upstream relies on implicit
   broadcasting for the first call and its `expand` lines are unused). Reason:
   removes a latent batch-axis hazard; numerically equivalent.
7. **Dropout configurable, non-inplace**, upstream defaults retained. Reason:
   enables the Gate 3 zero-dropout diagnostic override and avoids inplace
   autograd/logging surprises.
8. **`Freq_aug` / CONV masking removed.** Reason: it masks sinc filters; there
   is no waveform filterbank in the adapter.
9. **Detector wrapper returns only `DetectorOutput(logits)`.** Reason: the
   repository contract; upstream returns `(last_hidden, output)`.
10. **Training budget is GRU-matched**, not the upstream recipe: AdamW
    `lr 1e-3`, `weight_decay 0`, seed 0, batch 8, 10 epochs, CPU FP32 -
    per the pilot protocol, so the comparison is on identical inputs and
    budget. The upstream Adam/cosine/100-epoch recipe stays recorded for any
    future authorized tuning but is not used here.
11. **Standardizer is new** relative to upstream (fixed training-frame scaling,
    never optimized, serialized with the head).
12. **BatchNorm kept, not substituted.** The upstream `BatchNorm1d` over
    `(B*N, out_dim)` is preserved; train/eval behaviour and a batch-size-one
    with multiple nodes case must be verified in Gate 2.

Preserved from upstream (with attribution): dense pairwise attention with
temperature softmax; attention/without-attention projection sum; `SELU`;
input dropout; type-masked heterogeneous attention with shared cross-type
weights; master-node attention and projection; learned top-k `GraphPool` with
`h * scores`; two-branch structure with elementwise-max fusion; `max|.|` /
`mean` graph readout with the 5-way concatenation and `Linear(->2)` head.

---

## 7. Parameter and attention-cost estimates (to be confirmed in Gate 2)

| Component | Parameters |
| --- | --- |
| LayerNorm(1024) | 2,048 |
| Linear(1024 -> 512) | 524,800 |
| `pos_F` (16 x 32) | 512 |
| `GAT_layer_S` + `GAT_layer_T` (32 -> 64 each) | 13,056 |
| `ST11` + `ST21` (64 -> 32 each) | 37,824 |
| `ST12` + `ST22` (32 -> 32 each) | 17,280 |
| 6 `GraphPool` layers | 262 |
| 2 masters | 128 |
| `Linear(160 -> 2)` | 322 |
| **Total (measured in Gate 2)** | **600,392** |

For reference, the GRU head has 659,714 trainable parameters; the standardizer
adds non-trainable buffers to both.

All-pairs tensors (per item, before batch): `GAT_S` `16x16x32`, `GAT_T`
`32x32x32`, `ST11/ST21` `30x30x64`, `ST12/ST22` `15x15x32`. At batch 8 the
largest intermediate is `[8,30,30,64]` FP32 (~1.8 MB). No all-pairs attention
across 1,024 embedding dimensions or full recordings.

---

## 8. Integration plan and exact files (Gate 2 onward)

New (Gate 2):

| Path | Purpose |
| --- | --- |
| `src/detectors/aasist.py` | Adapted head (`AasistSpoofDetector`) + `AasistDetector(Detector)` wrapper + `register_detector("aasist", ...)`; upstream attribution header; exposes `.model.standardizer` |
| `src/detectors/aasist_blocks.py` (conditional) | `GraphAttentionLayer`, `HtrgGraphAttentionLayer`, `GraphPool` ports with attribution; created only if keeping them separate improves reviewability |
| `tests/detectors/test_aasist.py` | Focused correctness checks (section 9) |
| `docs/third_party/AASIST_NOTICE.md` (name may be adjusted) | Upstream commit, MIT notice, list of modifications |

Modified (Gate 2, minimal):

| Path | Change |
| --- | --- |
| `configs/aasist.yaml` | Fill the placeholder with the parameters in section 5.6 |
| `src/detectors/__init__.py` | Import/export `AasistDetector` so registration runs |
| `tests/detectors/test_registry.py` | Assert `aasist` constructs instead of raising `NotImplementedError` |

Gate 4 (only with local v3 assets): new `scripts/train_aasist_v3_from_cache.py`
with `--check-only` and its own configuration guard, reusing
`load_dataset_version`, `audit_v3_cache`, `load_v3_examples`,
`fit_detector_standardizer`, `build_optimizer`, `build_loss`, `fit` and the
existing evidence-capture helpers. The GRU runner's guard must not be
weakened; helper extraction from it is acceptable only if behaviour and tests
are preserved.

Gate 5 documents: `docs/AASIST_IMPLEMENTATION_REPORT.md`, plus optional
`scripts/verify_aasist_audio.py` for a narrow audio -> preprocessing ->
encoder -> head parity check when local assets exist.

Explicitly unchanged: `src/detectors/{gru,base,registry,training,runner,
checkpoints,standardization}.py`, `src/data/**`, `src/backbones/**`,
`src/audio/**`, `src/scoring/predict.py`, `src/service/**`,
`configs/gru*.yaml`, `tests/detectors/test_gru.py`,
`scripts/train_gru_v3_from_cache.py`. Service dispatch and general release
packaging stay GRU-only for this task.

---

## 9. What can be tested without real data, and what cannot

Testable now with synthetic tensors (never as accuracy evidence):

1. Shapes for `B=1` and `B>1` with `T=49` and `T=199`; `[B,2]` logits; finite
   outputs; label/dtype validation.
2. Eval-mode equivalence: single item versus mixed-length batch within a
   stated FP32 tolerance; appending finite padding (`+77/-77`) without
   changing lengths/masks leaves logits unchanged; mask holes, contradictory
   lengths/masks, `T<32` and empty sequences fail clearly.
3. Padding excluded before adaptive pooling, attention, BatchNorm and
   `GraphPool` (assert on a fixture where padded values would otherwise change
   the pooled bins).
4. One CE backward step updates adapter, attention, pooling, master and
   classifier parameters with finite gradients; standardizer buffers have no
   gradients; inputs remain detached.
5. Train/eval dropout and BatchNorm behaviour, including batch size 1 with
   more than one node per node set.
6. Standardizer: fitted on training fixtures only, persisted in the
   checkpoint, missing/incompatible state rejected, no fitting during forward.
7. Checkpoint parity: live logits equal restored logits; wrong `model_name`
   fails; existing GRU checkpoint/restore tests still pass.
8. Upstream-parity checks on random tensors for the ported graph blocks
   (compare against the fetched reference copies where licensing allows).
9. Parameter counts and a tiny random-batch training smoke test (plumbing
   only, explicitly not performance).

Must wait for real v3 assets: Gate 3 memorization diagnostic, Gate 4 pilot
training/selection/threshold, benchmark or held-out scoring, GRU comparison,
latency timing, and audio-to-head parity.

---

## 10. Gate status and blockers

| Gate | Status |
| --- | --- |
| Gate 1 (inspection + architecture plan) | **Complete (this document)**, pending review |
| Gate 2 (head + focused checks) | Not started; no code written |
| Gate 3 (24-example diagnostic) | Blocked: v3 caches absent locally |
| Gate 4 (v3 pilot) | Blocked: v3 caches + pinned encoder environment absent |
| Gate 5 (comparison/handoff) | Blocked: requires Gate 4 assets; GRU v3 predictions/caches absent locally |

Additional blockers for later gates: no `fairseq`, no `.venv`, no
`models/indicwav2vec_large.pt`, and `artifacts/datasets-v3-coverage/` missing.
None of these affect Gates 1-2.

## 11. Decisions requested before Gate 2

1. Confirm `C=32, F=16, K=32` (the bounded hypothesis) rather than the
   closer-to-upstream `C=64` variant.
2. Confirm keeping upstream `BatchNorm1d` in attention layers (with the
   batch-size-one check) instead of substituting another norm.
3. Confirm that the head reuses `FrameStandardizer` with the same
   fail-closed checkpoint guard as the GRU.
4. Confirm the module split (`aasist.py` plus optional `aasist_blocks.py`) and
   the provenance file name.
