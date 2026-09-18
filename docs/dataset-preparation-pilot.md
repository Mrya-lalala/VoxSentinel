# VoxSentinel dataset preparation — complete paired core, audits and handoff

Bounded preparation of real data for the frozen **IndicWav2Vec → GRU** pipeline,
executed 2026-09-18 (continuing the 2026-09-17 pilot). This run completes and
repairs the training/validation datasets: the paired core is finished at target
quota for all 12 Indic languages, newly accessible sources are processed, the
preprocessing contract is tightened and the handoff artifacts are rebuilt with
accurate counts.

**No tests, detector training or evaluation were run.**

- Checkout: branch `feature/indicVac2Wav`, commit `c250bbd`; nothing was
  committed, pushed or force-pushed. All pre-existing and new working-tree
  changes are preserved (see the JSON report's `repo` block).
- Dataset root: `artifacts/datasets/` (Git-ignored; config
  `configs/datasets.yaml`).
- Reports: `artifacts/datasets/reports/preparation-report.{json,md}`
  (`python -m scripts.prepare_datasets report`).

## 1. What was actually prepared

| Pool | Windows | Content | Manifest |
| --- | ---: | --- | --- |
| core train | 384 | 12 languages × (16 genuine Kathbath + 16 synthetic IndicSynth) | `manifests/windows.core_train.jsonl` |
| core validation | 96 | 12 languages × (4 genuine Kathbath + 4 synthetic IndicSynth) | `manifests/windows.core_val.jsonl` |
| supplementary | 10 | NPTEL2020 pure-set genuine English (identity unresolved) | `manifests/windows.supplementary.jsonl` |
| external evaluation | 20 | Svarah genuine Indian-English read speech (20 speakers / 10 states) | `manifests/windows.external_eval.jsonl` |
| fallback/baseline | 50 | ASVspoof 2019 LA: train 20 genuine + 20 spoof, dev 5 + 5 | `manifests/windows.fallback_baseline.jsonl` |
| unpaired candidate | 40 | NISP genuine Indian-English, 5 native-language groups | `manifests/windows.unpaired_candidate.jsonl` |
| excluded | 8,799 | pair-candidate trims (`candidate_pool_not_selected`) | `manifests/windows.excluded.jsonl` |
| missing coverage | 4 | SPIRE-SIES / Indic TIMIT / IndicVoices / synthetic English | `manifests/windows.missing_coverage.jsonl` |

Every selected window has a recording row in `manifests/recordings.jsonl`; each
source keeps its own `source_windows.*`, `source_recordings.*` and
`source_exclusions.*` files. Source-access catalog:
`manifests/source-catalog.json`.

### Core windows by language

Each language contributes 16 + 16 training windows (genuine + synthetic) and
4 + 4 validation windows, built from **strictly speaker-disjoint relationship
pairs** (§5). Generators across the core (train + val, 480 windows): `freevc24`
128, `xtts_v2` 107, `vits` 5, plus 240 genuine Kathbath windows; per-language
counts, unique originals and generator mixes are in the JSON report
(`core_identity.languages`, `core_build`, `core_by_generator`).

### Cross-split integrity

The report records unique speakers/originals per split and the cross-split
overlap for the core pools (`core_identity`): train and validation share **no
speaker, no recording and no upstream reference** by construction (disjoint
pairs; genuine material attached to the same components or assigned as
exclusive free agents), and the audit re-checks this from the manifests:
speaker overlap 0, reference overlap 0, prepared-waveform overlap 0 and
original-audio overlap 0.

### Lineage resolution (full evidence)

Every selected synthetic window's references were resolved against the
**complete** Kathbath train shards with stored evidence:
**366/366 applicable reference roles `verified_train`** (130 via the scanned
inventory, 236 via the full-shard footer-statistics scan) and 114 roles
`not_applicable_tts` (TTS rows have no voice-conversion source step). No
`parsed_only` status remains in the frozen files. Across the repair rounds,
**12 selected rows** were replaced with verified alternatives (quotas
preserved in every language/split/class) and 72 further candidates were
pre-excluded by reference knowledge before selection. Replaced/or excluded
rows are itemised in `windows.excluded.jsonl` and summarised in the report's
`lineage_resolution` block.

### Fallback pool (ASVspoof 2019 LA, official membership preserved)

| Official split | genuine | spoof | attack systems |
| --- | ---: | ---: | --- |
| train | 20 | 20 | A01–A06 (labels from the official protocol key, verified) |
| dev | 5 | 5 | A01–A05 (labels from the official protocol key, verified) |

## 2. Feature cache (ready for the existing B2 contract)

`artifacts/datasets/features/train.pt` (384 items) and `features/val.pt`
(96 items), plus `*.meta.json` sidecars, extracted with the pinned encoder
environment (`.venv`).

- Format `voxsentinel.embedding_cache.v1`; each item stores **unpadded** float32
  `features [frames, 1024]`, `label` (0 = genuine, 1 = synthetic), `num_frames`,
  window identity and the prepared-waveform SHA-256.
- Cache identity includes the **manifest content signature** — a hash over window
  membership, labels, spoken languages and prepared-waveform hashes — so
  metadata-only edits (provenance/evidence blocks) reuse cached features, while
  any audio, label or membership change invalidates the cache. The raw manifest
  file hash is recorded separately for audit. The encoder checkpoint SHA-256
  `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`, the final
  encoder layer output, dim 1024, frame hop 20 ms, float32, implementation
  `src.backbones.indic_wav2vec:extract_batch`, preprocessing version
  `voxsentinel-prep-2` and the pinned runtime identity complete the key. A cache
  whose identity, sidecar or per-item waveform hashes do not match is rebuilt,
  never silently reused.
- Writes are atomic (temp file + `os.replace`), and a `complete: true` sidecar
  marker with per-language label counts is written after the bundle. A bundle
  without a complete sidecar is treated as stale.
- **Training guard:** `validate_core_training_cache()` audits completeness,
  manifest freshness, item counts, both-label presence and per-language
  dual-class coverage; `scripts/train_gru_from_cache.py` runs the audit in
  `--check-only` mode and **refuses to start training** when it fails. The same
  audit is embedded in the report (`features_validation`).
- Frame counts come from the encoder itself (never estimated); windows at the
  strict 4 s cap produce 199 frames.

## 3. Sources, revisions and terms (as of this run)

| Source | Status | Route | License recorded |
| --- | --- | --- | --- |
| IndicSynth (`vdivyasharma/IndicSynth`) | ✅ processed | HF; metadata-first row selection; only selected assets downloaded; revision SHA per row | CC BY-NC 4.0 |
| Kathbath (`ai4bharat/Kathbath`) | ✅ processed | HF gate accepted by the account owner; parquet shards read selectively by row group; per-language revision SHA recorded | CC BY 4.0 |
| Svarah (`ai4bharat/Svarah`) | ✅ processed | HF gate accepted; fixed row-group plan over 3 parquet files; revision SHA recorded | CC BY 4.0 |
| NISP (`iiscleap/NISP-Dataset`) | ✅ refreshed | official `train_spkrID` speakers only | CC BY 4.0 |
| NPTEL2020 (`AI4Bharat/…-Speech-Dataset`) | ✅ refreshed | published pure-set release asset only | CC BY |
| ASVspoof 2019 LA (DataShare 10283/3336) | ✅ refreshed + licence text | `LA.zip` (7.64 GB) never downloaded whole; selected FLACs via HTTP range reads | licence text extracted to `raw/asvspoof2019/license/` (LICENSE.txt sha256 `99a4d3e0…c1555d`, README.LA.txt sha256 `c99a875d…748fe`) |
| IndicVoices (`ai4bharat/IndicVoices`) | 🟡 accessible, not processed | gate accepted mid-run; file reads verified (e.g. `hindi/train-00000-of-00082.parquet`, 5,429 rows) | CC BY 4.0 |
| SPIRE-SIES / Indic TIMIT | ⛔ awaiting user links | portal is request-based; user reported archives unavailable this session | portal T&C |
| Synthetic Indian-English | ⛔ missing source | no traceable documented source; undocumented TTS dumps rejected | – |

ASVspoof 2021 LA/DF is explicitly **out of scope** for this stage and has been
removed from the catalog and the gap inventory.

**User actions still required** (the tooling never accepts terms or handles
tokens):

- IndicVoices (optional): extend the external-evaluation pool with IndicVoices
  shards in a follow-up run; the parquet/row-group access pattern is proven.
- SPIRE-SIES / Indic TIMIT: provide local archive paths.
- Synthetic Indian-English: supply authorized assets following
  `configs/synthetic_english_template.json`.

## 4. Audio processing policy (`voxsentinel-prep-2`)

Identical for genuine and synthetic audio:

- decode → mean-of-channels mono → one resample to 16 kHz with `soxr` HQ; the
  decoder identity is recorded per window (`prepared_audio.processing.decoder`)
  — in this run all 600 selected assets decoded via `libsndfile`, including the
  Kathbath `.m4a`-named blobs (their payload is actually FLAC, 16 kHz PCM); the
  documented ffmpeg fallback remains available for formats `libsndfile` cannot
  read and would be recorded the same way;
- **no** denoising, loudness normalization, speed/pitch changes or codec
  augmentation;
- explicit overflow policy: if resampling leaves amplitudes > 1.0, attenuate by
  exactly `1/peak` — a **bounded attenuation ≤ 0 dB**, recorded as
  `overflow_gain`/`overflow_gain_db` with `peak_before_overflow_gain`; never
  clip, never upscale;
- prepared files: mono 16 kHz WAV/FLOAT, byte-deterministic;
- **strict 4-second cap (no grace):** recordings with ≤ 64,000 prepared samples
  are kept whole; longer recordings contribute the highest-energy contiguous
  4 s window on a 0.25 s grid (ties → earliest). Minimum usable length 1 s.
  Speech-activity facts are recorded per window (`active_fraction` under the
  30 ms / RMS policy), and windows with too little activity are excluded with
  an explicit reason instead of silently accepted;
- window offsets are recorded in seconds **and** sample coordinates at both the
  prepared and original rates.

The refresh pass (`scripts/refresh_prepared_windows.py`) re-materialized the
existing non-core pools from local raw audio under this policy without any
re-download: NISP 40/40, NPTEL 10/10 and ASVspoof 50/50 windows refreshed; the
three previously over-length ASVspoof windows are now at the cap
(over-length 3 → 0). The frozen dataset state is stamped in
`manifests/dataset-version.json` (per-manifest hashes, cache identities,
preprocessing version, seed, readiness flags).

## 5. Selection and split rules

- **Paired core (Kathbath ↔ IndicSynth):** per language, up to 8 candidate
  pairs are selected **strictly speaker-disjoint** — no two pairs share a
  source speaker, target speaker or reference recording, so relationship
  components cannot merge and no cross-component bridge exists. Participants
  are excluded from every other pair (both sides for voice-conversion rows;
  target only for TTS rows, whose source fields are nulled with
  `source_kind: tts_target_only`).
- Validation takes **two relationship components with distinct generators**
  where available (4 synthetic + 4 genuine windows) and genuine material from
  ≥ 2 speakers; training takes the remaining components (16 + 16, ≥ 3 speakers).
  Genuine material attaches to its pair's component where possible; otherwise
  speakers with real recordings but no selected synthetic row ("free agents")
  are assigned exclusively to one split to fill quotas.
- Every synthetic row passes lineage checks before selection: the parsed
  reference must match the declared speaker, target lineage must resolve, and
  rows whose upstream reference points into the upstream `valid`/`test`
  partitions are excluded (`upstream_held_out_parent`). After selection, each
  remaining reference is resolved against the **complete** Kathbath train
  shards by `scripts.resolve_lineage.py` using parquet footer `fname`
  statistics as an exact index plus targeted column reads; evidence (shard,
  row group, row index, exact upstream fname) is stored per role in
  `parent_refs.reference_evidence`, with raw upstream row metadata preserved in
  `parent_refs.upstream`. Rows whose references live in the upstream
  evaluation partitions or exist nowhere are replaced by verified alternatives
  from the same language/class/split pool.
- **NISP:** only official `train_spkrID` speakers; official test speakers are
  never selected. Folder native language is the accent background; spoken
  language is `en`.
- **ASVspoof 2019 LA:** labels come from the official protocol keys, never from
  filenames; official train/dev membership preserved; selection spreads
  speakers (genuine) and attack systems (spoof).
- **NPTEL:** only the published pure set; lecturer identity is unresolved, so
  the clips are supplementary/inventory-only and **not** claimed to be
  speaker-disjoint.
- **Svarah:** fixed, documented row-group plan across the three parquet files;
  one window per recording; speakers used once while fresh speakers remain,
  states preferred when not yet covered; ≤ 2 windows per speaker. All windows
  are genuine (`label: 0`), live in `external_eval`, and never enter training,
  validation or threshold tuning.
- No core test split exists; the external and fallback pools protect future
  evaluation from this pilot's data.

## 6. Downloads (bounded, resumable, one ledger)

`manifests/downloads.json` holds the final per-source totals against the
5 GiB budget: **3.18 GiB used, ~1.82 GiB remaining**.

| Source | Bytes | Note |
| --- | ---: | --- |
| Kathbath | 1,542,601,246 | 12 languages: shard metadata scans + row-group audio reads for selected recordings |
| NISP | 1,294,991,360 | archive parts (earlier pilot; reused, not re-charged) |
| IndicSynth | 228,226,054 | selected row assets only (never whole parquet shards) |
| NPTEL | 188,815,224 | pure-set release asset |
| Svarah | 142,472,502 | ten row groups → the 20 selected windows |
| ASVspoof 2019 | 16,272,703 | remote-ZIP range reads + 50 selected FLACs (archive is 7.64 GB) |

All downloads are resumable, charged before use, and rerun-safe: credentials
that already exist on disk are skipped without a second charge, and the ledger
re-syncs with the on-disk file before every charge so concurrent stages cannot
lose accounting.

## 7. Commands actually executed (data env `.venv-data`, Python 3.12)

```text
python -m scripts.prepare_datasets catalog
python -m scripts.prepare_datasets core --plan-only --languages Malayalam
python -m scripts.prepare_datasets core --languages Malayalam
python -m scripts.prepare_datasets core --languages Bengali Gujarati Hindi Kannada Marathi Odia Punjabi Sanskrit Tamil Telugu Urdu
python -m scripts.prepare_datasets core --languages Malayalam    # merge pass: keeps all 12 languages staged
python -m scripts.prepare_datasets fetch --source svarah
python -m scripts.refresh_prepared_windows --sources asvspoof2019 nisp nptel
python -m scripts.resolve_lineage scan                          # full train-shard reference verification
python -m scripts.resolve_lineage apply                         # evidence + raw upstream metadata into manifests
python -m scripts.prepare_datasets core --languages Bengali Gujarati Kannada Marathi Odia Sanskrit   # verified replacements (rounds 1-3)
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets split
.venv/bin/python -m scripts.prepare_datasets features --force --threads 4
python -m scripts.prepare_datasets report
```

(The full timestamped log is `manifests/run-log.jsonl`.)

## 8. NOT RUN — exact next commands for your own checks and training

The commands below were **not executed** during preparation:

1. Inspect the preparation state (any Python env):

   ```sh
   python -m scripts.prepare_datasets status
   cat artifacts/datasets/reports/preparation-report.md
   ```

2. Audit the feature cache without training (encoder env, `.venv`):

   ```sh
   .venv/bin/python -m scripts.train_gru_from_cache --check-only
   ```

3. Train the GRU detector once the audit passes (encoder env):

   ```sh
   .venv/bin/python -m scripts.train_gru_from_cache        # configs/base.yaml + configs/gru.yaml
   ```

   (`scripts/train_gru_from_cache.py` wires the cache into the existing
   `src.detectors.runner.run_training`; it refuses to start if the readiness
   audit fails.)

4. Optional follow-up: extend `external_eval` with IndicVoices shards (access
   is now available; same parquet/row-group pattern as Svarah).

## 9. Honest qualification

- No unit tests, smoke tests, detector forward passes, training, evaluation,
  benchmarks or optimizer steps were run for this task; no accuracy, EER,
  calibration or robustness number is claimed anywhere.
- `indian_english_training_ready` remains **false**: no traceable synthetic
  Indian-English source is paired with the genuine English pool; NISP windows
  stay `unpaired_candidate` and are excluded from the default core manifest.
- IndicVoices access became available mid-run but is intentionally not
  processed in this bounded run (`accessible_not_processed`).
- SPIRE-SIES and Indic TIMIT remain blocked on user-supplied archives
  (`awaiting_user_links`).
- Exact hashing cannot prove the absence of near-duplicates or unidentified
  shared speakers; final assembly also drops exact-duplicate prepared windows.
- Accent verification of any future synthetic Indian-English import is
  unreviewed until actually reviewed; an `en` TTS setting alone is not
  evidence.

## 10. First controlled GRU run (completed 2026-09-18)

After the frozen dataset version passed the readiness audit, the first
frozen-embedding GRU experiment was executed:

```sh
.venv/bin/python -m scripts.train_gru_from_cache --run-dir artifacts/runs/gru-core-v1
```

- Settings (resolved from `configs/base.yaml` + `configs/gru.yaml`): GRU head
  (1024→256, 1 layer, dropout 0), AdamW lr 1e-3 / wd 0, batch 8, **10 epochs**,
  seed 0, CPU float32, no scheduler, no gradient clipping, no class weights;
  checkpoint selection = lowest validation EER (earliest epoch wins ties).
  Head training took **43.9 s**; cached features were reused (no encoder pass).
- **Selected epoch 2** with validation EER **0.229** (chance = 0.5). At the
  fixed 0.5 threshold every validation window scored ≥ 0.5, so all 48 genuine
  windows were flagged (FPR 1.00) and all 48 synthetic windows caught
  (miss 0.00): accuracy 0.50, balanced accuracy 0.50
  (confusion TN 0, FP 48, FN 0, TP 48).
- The pattern is a score-bias collapse (constant-side predictions at 0.5), not
  a per-language or per-generator failure; the ranking is weakly informative
  (EER 0.229 well below chance) but no deployed threshold or calibration is
  claimed — none was tuned in this task.
- Validation selected the checkpoint and each language contributes only 8
  development examples: these are development results, not test performance.
  Next evidence needed: decision-threshold/calibration analysis on an
  independent set, more data and speakers per language, alternative heads or
  encoder layers, and unseen-generator evaluation.
- Full run artifacts (settings, dataset/version hashes, code diff + hashes,
  environment, epoch history, best/final checkpoints, best-checkpoint
  predictions with IDs/languages/generators, metrics, per-language/generator
  breakdown, training report): `artifacts/runs/gru-core-v1/`.

**Follow-up (2026-09-18): the failure mode above was diagnosed.** See
`docs/GRU_DIAGNOSTIC_MANAGER_REPORT.md` and
`artifacts/runs/gru-core-v1-diagnosis/`. In brief: the near-constant scores are
a training-dynamics/convergence failure on weak, ill-conditioned inputs — not
a code, metric, checkpoint or data defect (all reproduced/refuted with
bit-exact evidence). On the same frozen features a pooled linear probe reaches
validation EER 4.2 % / AUC 0.988, and the same GRU head memorizes a 24-window
subset to zero loss, so the evidenced next step is one controlled
whitened-input rerun (per-dim standardization fitted on train only) rather
than data expansion.
