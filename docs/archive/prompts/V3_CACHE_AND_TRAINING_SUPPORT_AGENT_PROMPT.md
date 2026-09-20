> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# VoxSentinel: implement v3 feature caching and training support

Work in `/Users/mouryabs/VoxSentinel`. Implement and verify feature caching and dataset-aware training support for the newly expanded v3 train/development manifests. Generate the required caches. **Stop before actual model training**; the primary agent will review the implementation, train, analyze and handle branch publication.

## Read first and preserve existing work

Read applicable AGENTS.md and:

- `docs/archive/history/GENUINE_COVERAGE_ACQUISITION_HANDOFF.md`
- `docs/archive/history/GENUINE_COVERAGE_ACQUISITION_REVIEW.md`
- `docs/archive/history/V2_SPLIT_TRAINING_REPRODUCTION.md`
- `docs/archive/history/V2_FROZEN_TEST_RESULTS.md`
- `artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json`
- `src/dataset_prep/features.py`, `src/data/batch.py`
- `scripts/train_gru_from_cache.py`, `scripts/training_split_contract.py`, `scripts/gru_recovery.py`
- `src/detectors/{standardization,training,checkpoints,gru}.py`
- `src/backbones/indic_wav2vec.py`, `configs/backbones.yaml`
- `scripts/validate_dataset_v3_coverage.py` and related tests.

Inspect Git status. This checkout has intentional uncommitted changes: preserve them. Do not reset, clean, stash, commit, push or merge. Do not rerun acquisition commands or overwrite historical manifests, runs, caches, models or evaluation evidence.

## Verified context

The independently reviewed v3 dataset contains:

| Split | Windows | Genuine | Synthetic | New male genuine |
|---|---:|---:|---:|---:|
| Train | 458 | 266 | 192 | 74 from 37 language-scoped speakers |
| Development | 135 | 87 | 48 | 39 from 20 language-scoped speakers |

All 384 old training and 96 old development examples remain unchanged. The 113 additions passed independent raw-to-prepared sample equality checks; identity overlaps across train/dev and the evaluated benchmark were zero under the adopted protocol. Malayalam has no additions, Odia has one new development recording and no training additions, and Bengali has one new speaker per split. Nine of twelve languages reached acquisition targets.

Old caches cover only 384 train / 96 development examples. They are valid historical caches but incomplete for v3. The existing `--split-version` guard intentionally accepts only byte-identical v2 copies. Do not disable that check and feed expanded manifests to old bundles.

The existing standardized GRU uses a frozen IndicWav2Vec encoder, training-only frame standardization, LayerNorm, projection to 256, a one-layer 256-unit GRU, valid-frame mean pooling and two output logits. Labels: genuine=0, synthetic=1. Standardization excludes padding and is restored with checkpoint state.

The earlier frozen model achieved 94.79% development accuracy but 79.17% on the 96-window v2 benchmark, with male genuine false alarms 16/24. These results motivated coverage acquisition; they do not prove a causal gender-only failure. The benchmark is already evaluated and must not guide checkpoint selection, preprocessing, loss weighting or thresholds. It remains excluded from train/dev. No fresh untouched test is being created here.

Protocol: `speaker_recording_disjoint_v2`; strict conversion-family isolation is not met and cross-language person independence is unproven. Carry these limits into provenance rather than marking them resolved.

## 1. Dataset-aware manifest and path handling

Use `dataset-version.v3.json` as the explicit contract. Verify its schema, declared manifest hashes, source revisions and preprocessing identity before building caches or declaring training readiness.

- Train: `artifacts/datasets-v3-coverage/manifests/windows.train.jsonl`.
- Dev: `artifacts/datasets-v3-coverage/manifests/windows.dev.jsonl`.
- Retained rows lack `path_base`; resolve their `prepared_audio.path` under `artifacts/datasets`.
- Additions have `path_base: artifacts/datasets-v3-coverage`; resolve under that base.

Implement one tested resolver consistent with the version record. Validate allowed bases, existing paths, hashes, duplicate IDs and split assignments. Do not search other roots silently when a path fails. Make repo-relative resolution explicit and deterministic. Do not conflate the `dev` manifest name with historical rows whose split field is `val`; normalize intentionally and validate every row.

Re-run the existing offline v3 integrity validator or independently bind a successful audit to the exact current manifest/audio inputs. Do not accept a stale `ready: true` report detached from its inputs. Check retained/addition accounting, labels, language coverage and disjoint train/dev identities. Test benchmark paths may be recorded as provenance but never loaded as training or development examples.

## 2. Frozen encoder and preprocessing contract

Use the existing local encoder and pinned environment; no downloads or dependency upgrades:

- Encoder checkpoint: `models/indicwav2vec_large.pt`.
- SHA-256: `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`.
- Final encoder output: `output_layer: null` (do not substitute layer 23).
- Output dimension: 1024, float32, approximately 20 ms frame hop.
- Existing preprocessing: `voxsentinel-prep-2`, mono 16 kHz prepared windows.
- Runtime: existing `.venv` Python 3.10 / Torch 2.2.2 pinned encoder environment. `.venv-data` is the acquisition environment, not the default model runtime.

Load prepared windows directly through the established encoder contract; do not recrop, augment, denoise or normalize them again. Preserve the encoder's own valid-sample waveform normalization and existing exact-length batching/padding behavior. Encoder stays in eval mode with gradients disabled. Cache ordinary detached tensors compatible with subsequent autograd through the head; avoid inference-mode tensors that cannot be saved for backward.

## 3. Build complete v3 caches safely

Write under `artifacts/datasets-v3-coverage/features/`, preserving all historical caches.

Prefer validated per-example reuse for the unchanged 480 original windows, then extract features for the 113 additions. Reuse only after matching window ID, label, language, prepared waveform hash, preprocessing version, encoder checkpoint hash/layer, feature dimension/dtype and relevant extraction contract. If compatibility cannot be established, regenerate the affected features with the same pinned encoder and document why. Never rely on row position alone.

Emit complete train and dev bundles in deterministic manifest order. Do not create a bundle containing only additions and call it the full split. Preserve item IDs, labels, language, generator, prepared hash, sample count and valid frame count. Reject missing, duplicate, extra, nonfinite or shape-inconsistent features.

Cache sidecars/completion markers must bind each bundle to:

- dataset-version identity and exact split manifest hash;
- window IDs, counts and label/language distributions;
- prepared-audio identities;
- pinned encoder/preprocessing/extractor identity and runtime;
- bundle hash and explicit completeness status;
- per-item reuse versus fresh extraction provenance.

Write atomically; failed/interrupted construction must not leave a success marker. Resume only when partial entries still match the current identities. Distinguish cached provenance from mutable acquisition ledger totals: do not invalidate identical audio because the ledger grew, or accept changed audio because its path stayed the same.

Do not create benchmark/test features in this task. No model scoring or predictions.

## 4. Extend training support without weakening historical safeguards

Add explicit support for the v3 version schema and its caches while preserving historical v1/v2 commands and checks. Keep the model and optimizer behavior unchanged.

The existing standardized runner calls recovery helpers tied to the old dataset root. Inspect all of these dependencies. A v3 run must not accidentally train on v3 examples but select a checkpoint, generate predictions, perform reload checks or report metrics using the old 96 development examples. Parameterize dataset-dependent helpers or introduce a clearly scoped v3 runner using shared training components. Preserve historical recovery reproducibility rather than rewriting old evidence.

Before any future training run:

- Validate the complete v3 cache contract and exact train/dev sample membership.
- Fit the standardizer on v3 TRAINING valid frames only; do not fit now.
- Use v3 development for epoch selection and development predictions.
- Store the new dataset/cache identities and protocol in checkpoints and reports.
- Fail closed on a v2 cache presented as v3, mismatched manifests, missing samples, altered labels, mixed encoder identities, or test rows presented as train/dev.

Configure the initial future comparison using the existing standardized GRU, seed, batch size, optimizer, ten epochs and fixed 0.5 threshold. Keep unweighted cross-entropy for this first controlled comparison. Report the changed class ratios explicitly; do not silently add class weights, resampling, oversampling or balancing. Preserve all acquired data. A weighting experiment may be considered separately by the primary agent.

Make selection metric/direction/tie-breaking explicit as in the existing runner: development EER, minimum, earliest epoch on ties. Report accuracy alongside balanced accuracy, genuine false-alarm rate, spoof miss rate and counts. Update hard-coded 384/96 or “balanced classes” narrative so future v3 reports derive their counts from the actual inputs. Do not claim the old and expanded development accuracies are directly comparable on identical populations.

Provide a `--check-only`/dry-run path that checks manifests, caches, model configuration and required identities without fitting, optimizer steps or predictions. The training command should be ready to run later, but DO NOT execute it now.

## 5. Focused verification

Add meaningful tests, not just fixture-count assertions:

- Mixed retained/addition path resolution; missing/wrong base rejection.
- Deterministic manifest ordering independent of cache order.
- Missing/extra/duplicate IDs; changed audio hash or label; old-cache/v3 mismatch rejection.
- Encoder hash/layer/preprocessing mismatch rejection.
- Interrupted/incomplete cache rejection and safe resume identity checks.
- Train/dev/test role enforcement and absent benchmark access in the cache/training-input path.
- V3 development plumbing uses all 135 examples, not the old fixed 96. Use lightweight fixtures or mocks; do not run the real detector on them.

Verify all real emitted feature tensors and sidecars. On a small predeclared sample spanning retained/addition, train/dev, languages and available lengths, compare cache entries against direct frozen encoder extraction at an explicit tight tolerance; record maximum errors. This is encoder feature parity, not classifier evaluation. Where feasible verify batch/padding parity through existing encoder tests. Confirm cached tensors are detached ordinary tensors and compatible with a trivial synthetic autograd operation; do not fit or run the actual classifier as a substitute for this check.

Run relevant cache, batching, configuration and checkpoint tests. Verify frozen original artifacts remain unchanged. Any validation that requires network access must be replaced with a local fixture or marked untested; there should be no new acquisition in this task.

## 6. Deliverables and stopping point

Produce:

- Dataset-aware cache builder/resolver and training-support changes.
- Complete validated v3 train/dev feature bundles and sidecars.
- Check-only validation evidence, focused test results, encoder parity evidence and preservation hashes under a new v3 cache-report directory.
- `docs/archive/history/V3_CACHE_AND_TRAINING_SUPPORT_HANDOFF.md` describing exact inputs, feature counts/shapes/frame counts, reused versus extracted examples, hashes/runtime, code changes, limitations and commands.
- One exact command for future training and one safe check-only command. Explain explicitly that no model has yet been trained on v3 and the evaluated benchmark has not been rescored.

If a reliability check fails, resolve it within this scope or provide the precise blocker. Never bypass a readiness guard to produce a green report. Preserve failed-attempt evidence and avoid claiming checks were run when they were only inspected.

**STOP after feature caching and verified training support. No standardizer fitting, GRU training, classifier predictions, threshold tuning, benchmark evaluation, model release, commit, push or merge.** The primary agent will review, then conduct training and analysis before preparing the teammate branch handoff.
