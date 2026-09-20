> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# V3 Cache and Training Support Handoff

Status: **feature caching and verified training support complete — STOPPED
before training, as instructed.** No standardizer fitting, GRU training,
classifier predictions, threshold tuning, benchmark scoring, model release,
commit, push or merge was performed.

**No model has yet been trained on v3, and the evaluated benchmark has not
been rescored.** All evidence below was produced by cache construction and
validation commands; the training command exists and passed its check-only
gate but has never been executed.

## 1. Exact inputs

- Version contract: `artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json`
  (sha256 `768787e20a18afffabeb6a25ab63e520ee6665a7aea435ff24a8ecf0af574be0`),
  schema `voxsentinel.dataset_version.v3-coverage.v1`, protocol
  `speaker_recording_disjoint_v2`, `strict_conversion_family_compliant: false`.
- Train: `artifacts/datasets-v3-coverage/manifests/windows.train.jsonl`
  (458 windows = 384 retained + 74 additions; sha256 `cfa61534…4445ece`).
- Dev: `artifacts/datasets-v3-coverage/manifests/windows.dev.jsonl`
  (135 windows = 96 retained + 39 additions; sha256 `87a2a851…b00e134`).
- Path resolution (enforced by one resolver, no fallback roots): rows without
  `path_base` resolve `prepared_audio.path` under `artifacts/datasets`; the
  113 additions carry `path_base: artifacts/datasets-v3-coverage`.
- Split normalization: dev-manifest rows carry the historical raw value `val`
  (retained and additions alike) and are normalized to `dev`; every row is
  validated against its manifest's allowed raw values.
- Encoder: `models/indicwav2vec_large.pt`, sha256
  `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`,
  `output_layer: null` (final output, not layer 23), 1024-dim float32,
  ~20 ms hop; preprocessing `voxsentinel-prep-2` (mono 16 kHz prepared
  windows, no re-cropping/denoising/normalization).
- Runtime: `.venv` Python 3.10.18, Torch 2.2.2, torchaudio, numpy, fairseq
  0.12.1 (`.venv-data` was used only for JSON-side inspection).

## 2. Emitted caches (complete train + dev bundles)

Under `artifacts/datasets-v3-coverage/features/`:

| Split | Items | Genuine | Synthetic | Reused (`reused_v1`) | Fresh (`extracted_v3`) | Valid frames | Bundle sha256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | 458 | 266 | 192 | 384 | 74 | 91,117 | `25a5c30d9f3143ffb1528583a5888afa81b97589e4733ffd313c56f3d7e58d0c` |
| dev | 135 | 87 | 48 | 96 | 39 | 26,853 | `03be8bb06017097b30ebc5f162a3465a07769f69e7f1f607cc73dd2ede51c8ee` |

- Item contract: unpadded float32 `features [frames, 1024]`, `num_frames`
  equal to the encoder-reported count, plus ids, language, generator,
  prepared sha256, sample count, normalized split and
  `feature_source`/`reused_from` provenance per item.
- All 480 retained windows were reused per example from the historical v1
  caches (`artifacts/datasets/features/train.pt` / `val.pt`) only after the
  full identity match: window id, label, language, prepared waveform hash,
  preprocessing version, encoder checkpoint hash, selected layer, dim/dtype,
  implementation and torch runtime version. Zero reuse mismatches; the 113
  additions were freshly extracted with the pinned encoder using the same
  exact-length batching the v1 builder used.
- Sidecars (`train.meta.json` / `dev.meta.json`) bind each bundle to the
  dataset-version hash, exact manifest file hash, window membership/signature,
  label/language distributions, prepared-audio identities, encoder identity +
  runtime, bundle sha256, per-item order (manifest order) and an explicit
  `complete: true` marker. Construction is single-pass and atomic (bundle
  temp+rename first, marker last); an interrupted build leaves no valid
  marker, and a stale bundle with a valid sidecar fails hash/unreadability
  checks — it is never silently accepted. Re-running `build` against current
  caches is a verified no-op (`already_current`, 0.6 s).
- No benchmark/test features were created. The benchmark files are verified
  as declared provenance only; no benchmark audio or features are ever read
  by the cache or training-input path.

## 3. Verification evidence (`features/reports/`)

| File | Content |
| --- | --- |
| `cache_build_report.json` | Build provenance: status, timings, reuse audit, per-split counts, identity, bundle hashes |
| `cache_check.json` | Deep audit result: `ready: true`, 0 failures (membership + order, labels, languages, prepared hashes re-computed for all 593 files, feature contract per item, encoder/version identities, benchmark isolation) |
| `parity.json` | Encoder parity on a predeclared 10-sample set: 6 languages, retained/addition x train/dev, shortest/longest; `passed: true`, max abs error 5.245e-06 (reused items), **0.0 bit-exact for fresh extractions**, tolerance 1e-5, autograd-compatibility true for every sample |
| `preservation.json` | All 27 pre-task frozen artifacts re-hash to the planning snapshot (`all_match: true`); encoder checkpoint hash unchanged |
| `check_only_training.json` | Training-support gate: cache ready, counts match the version record, resolved settings match the documented initial comparison |
| `pytest_dataset_prep.log`, `pytest_full.log`, `build.log` | Focused (34 passed) and full (95 passed) test runs; build log |

Parity tolerance policy: items extracted in the current session must
reproduce **bit-exactly** (asserted `== 0`); items reused from the historical
v1 cache may drift at unit-in-the-last-place level (observed ≤ 5.25e-06)
because CPU reduction order depends on the thread count of the original
extraction session. This is encoder feature parity, not classifier
evaluation; the paired `torch.save`/`load` round-trip is bit-preserving.

Preservation: `scripts.prepare_dataset_v3_coverage verify` re-run after the
cache build → 27/27 frozen artifacts unchanged (includes v1 manifests,
features, v2 manifests incl. the frozen test manifest, runs, configs,
evaluated-benchmark files). Historical caches were read-only throughout.

## 4. Code changes (historical commands and evidence untouched)

New files (historical runners, `gru_recovery.py`, `training_split_contract.py`
and `src/dataset_prep/features.py` were not modified):

- `src/dataset_prep/dataset_version.py` — version-contract loader (schema,
  declared manifest/retained/benchmark hashes, revisions, counts), row loader
  with per-row validation (ids, labels, languages, split values, path safety,
  preprocessing identity, source revisions), the single deterministic path
  resolver, and cross-split invariants (train/dev disjoint, no benchmark id
  appears in either split).
- `src/dataset_prep/features_v3.py` — v3 cache builder (per-example reuse +
  pinned-encoder extraction), deep auditor (fail-closed, collects all
  failures), the training loader (train/dev roles only; manifest order;
  refuses stale caches) and the parity checker.
- `scripts/prepare_dataset_v3_features.py` — CLI: `build`, `check`, `parity`.
- `scripts/train_gru_v3_from_cache.py` — dataset-aware v3 training runner
  sharing the existing components (`fit`, optimizer/loss builders, checkpoint
  helpers, collator, `fit_detector_standardizer`) and deliberately NOT
  touching the old-root recovery helpers (`scripts/gru_recovery.py`), so a v3
  run can never select checkpoints or report metrics against the old
  96-window development set.
- `tests/dataset_prep/test_v3_dataset_version.py` (7 tests),
  `test_v3_feature_cache.py` (11 tests), `test_v3_training_support.py`
  (4 tests, skipped when the untracked v3 artifacts are absent).

Covered behaviors: mixed retained/addition resolution; undeclared base,
absolute path and `..` escape rejection; wrong raw split value; declared-hash
tamper; wrong preprocessing version / source revision; train/dev overlap and
benchmark-id leak; cache membership/order/duplicate/label/prepared-hash
mutations; v1-sidecar-as-v3 and incomplete-marker rejection; encoder
checkpoint/layer and dataset-version identity mismatches; truncated-bundle
interruption; non-finite features; loader role enforcement (`test`/`val`
refused) and manifest-order loading; dev plumbing uses all 135 examples, not
the old fixed 96.

## 5. Commands

Verification (offline, zero-charge): `check` and `--check-only` do not run forward passes; `parity` runs the frozen encoder, not the classifier:

```bash
./.venv/bin/python -m scripts.prepare_dataset_v3_features check
./.venv/bin/python -m scripts.prepare_dataset_v3_features parity
./.venv/bin/python -m scripts.train_gru_v3_from_cache --check-only \
  --json-out artifacts/datasets-v3-coverage/features/reports/check_only_training.json
```

The **exact future training command** (initial controlled comparison;
standardized GRU, seed 0, batch 8, AdamW lr 1e-3, 10 epochs, unweighted
cross-entropy, fixed 0.5 threshold, development-EER checkpoint selection with
earliest-epoch tie-break, CPU 4 threads as in the v2 standardized run):

```bash
./.venv/bin/python -m scripts.train_gru_v3_from_cache \
  --run-dir artifacts/runs/gru-v3-expanded-standardized --threads 4
```

It was **not executed**. Its real-training path fails closed unless the full
cache audit passes, fits the standardizer on v3 training valid frames only,
writes dataset/cache identities, protocol and class ratios into the run and
checkpoint metadata, verifies best-checkpoint reload parity, and derives all
counts and reports from the inputs.

## 6. Limitations carried into provenance

- `strict_conversion_family_compliant: false` is **reported, not resolved**:
  conversion-family co-participation chains remain documented non-edges and
  cross-language person independence is unproven. These flags are embedded in
  the cache identity and future run metadata.
- Class ratios changed by the acquisition: train 266/192 (58.1% genuine),
  dev 87/48 (64.4% genuine) — not balanced. The first v3 comparison keeps
  unweighted cross-entropy and applies **no** class weighting, resampling or
  oversampling; ratios are reported explicitly. A weighting experiment is a
  separate modeling decision for the primary agent.
- v3 development results are not directly comparable to the historical
  96-window development results (different composition).
- Coverage shortfalls remain from acquisition: Malayalam no additions, Odia
  one development addition, Bengali one new speaker per split.
- Reused v1 features differ from a fresh extraction only at ulp level
  (≤ 5.25e-06, documented above); fresh features are bit-exact.
- The evaluated benchmark (`artifacts/evaluations/gru-v2-test-epoch5/`) is
  unchanged and was not rescored; it influenced the acquisition objective and
  is not an untouched future test for adaptive improvements.
- The unchanged `--split-version` v2 guard still accepts only byte-identical
  v2 splits; the v3 runner deliberately does not use it and instead validates
  the v3 version contract. Do not point old commands at the v3 caches.

## 7. Remaining work for the primary agent

1. Review this implementation and evidence.
2. Fit/standardize + train with the exact command in §5 (or an audited
   variant), then analyze development metrics with the reported ratios.
3. Any comparison against the evaluated benchmark must be labeled as a repeat
   evaluation on an already-seen benchmark, not untouched-test performance.
