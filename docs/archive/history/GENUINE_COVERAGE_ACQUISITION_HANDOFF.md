> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Genuine Coverage Acquisition Handoff (v3 male-genuine additions)

Status: **acquisition, manifest addition, audio preparation and data-integrity
verification complete — STOPPED before features/training as instructed.**
No encoder features, standardizer fitting, training, prediction, test scoring,
threshold tuning, model packaging, commit, push or merge was performed.

- Task scope: acquire documented **male genuine** Kathbath speech (pinned
  revision `5b9e92849222026d9141acba4e8434fe816396bf`) and build combined v3
  train/dev manifests under `artifacts/datasets-v3-coverage/`.
- Soft targets: +8 train / +4 dev windows per language, ≥4 train speakers /
  ≥2 dev speakers per language where available, ≤2 windows per speaker per
  split. Missing targets are reported as shortfalls for languages whose
  admissible pool is exhausted (see § 7).
- Protocol: `speaker_recording_disjoint_v2`.

## 1. Deliverables

| Artifact | Path |
| --- | --- |
| Combined train manifest (384 retained + 74 additions = 458 rows) | `artifacts/datasets-v3-coverage/manifests/windows.train.jsonl` (sha256 `cfa61534a2d7206745897e06d0d0ddb74e8291f0d2e9643214aa8caa44445ece`) |
| Combined dev manifest (96 retained + 39 additions = 135 rows) | `artifacts/datasets-v3-coverage/manifests/windows.dev.jsonl` (sha256 `87a2a851b6ab160a792edd699a3d5abceba6af93b1ee469e341220cc9b00e134`) |
| Version/pointers record | `artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json` |
| Frozen plan + feasibility | `artifacts/datasets-v3-coverage/planning/{coverage_plan.json, feasibility.json, feasibility.md, candidate_snapshot.json, group_sizes.*.json, footer_charges.json, snapshot_before.json, snapshot_after_check.json}` |
| Materialization evidence | `artifacts/datasets-v3-coverage/staging/{additions.jsonl, dispositions.jsonl, materialization_summary.json}` |
| Failed-attempt evidence (preserved) | `artifacts/datasets-v3-coverage/staging_failed_attempt1/` |
| Raw + prepared audio (113 recordings) | `artifacts/datasets-v3-coverage/raw/kathbath/…` and `…/prepared/kathbath/…` (prep-2: mono mean, soxr HQ 16 kHz, 1–4 s high-energy windows) |
| Independent audit | `artifacts/datasets-v3-coverage/reports/audit.json` (`ready: true`, 0 issues) and `reports/replay_check.json` |
| Pipeline + CLI | `src/dataset_prep/coverage_v3.py`, `scripts/prepare_dataset_v3_coverage.py` |
| Validator + tests | `scripts/validate_dataset_v3_coverage.py`, `tests/dataset_prep/test_v3_coverage_validation.py` |

Digests: full frozen planning record `ab577f8baca46b90596b22fa62b05dec750bc1cbd047b03e40b1f7ef462d128a`
(includes point-in-time accounting); deterministic **selection digest**
`f0992c4ec1a3445c28b3704d3ff20434f5da9b4c5afe460ba4be29caf42ded20`.

## 2. Achieved coverage (actual, from the emitted manifests)

| Language | +train | train spk | +dev | dev spk | add. duration (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bengali | 2 | 1 | 2 | 1 | 8.0 / 8.0 |
| Gujarati | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Hindi | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Kannada | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Marathi | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Malayalam | 0 | 0 | 0 | 0 | 0.0 / 0.0 |
| Odia | 0 | 0 | 1 | 1 | 0.0 / 4.0 |
| Punjabi | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Sanskrit | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Tamil | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Telugu | 8 | 4 | 4 | 2 | 32.0 / 16.0 |
| Urdu | 8 | 4 | 4 | 2 | 32.0 / 16.0 |

- Totals: **74 train + 39 dev additions**, **57 unique speakers**
  (language-scoped), 296.0 s train / 156.0 s dev of added audio.
- All additions are documented male (`-m` Kathbath filename convention),
  read/scripted Kathbath speech, `raw_source: fetched_v3`, 0 replacement
  rows used, max 2 windows per speaker per split (target respected in all
  languages).
- 9 of 12 languages hit both soft targets; Bengali/Odia/Malayalam are
  pool-limited (see § 7).
- Retained rows: **384 train + 96 dev v2 rows carried over byte-identically**
  (row-level equality checked; see § 4).

## 3. Exclusion and protocol compliance

- Exactly one admissible-edge definition was used (matches the v2 review
  outcome): per-row-side person↔parent-recording links plus Kathbath
  inventory record↔speaker links. A candidate (speaker / reference / record
  node) is rejected iff its connected component intersects an anchor set:
  v1 prior-exposure speakers/refs/records, upstream held-out records, and the
  **evaluated benchmark** = `artifacts/evaluations/gru-v2-test-epoch5/`
  `exposure.json` plus **all roles of the full frozen v2 test manifest**
  (sha256 `1d7042f9a1c564b46b4df5d163733b7b46843f14aa44a47558d9e7a859a7115e`).
- Outcome: **0 of 113 additions hit any exclusion anchor** (closure audit:
  113 checked, 0 hits in `reports/audit.json`). Per-language exclusion
  tallies (e.g. `excluded_benchmark_evaluated_test`, `excluded_v1_prior_exposure`,
  `not_male`, `recording_too_short`) are in `planning/feasibility.json`.
- Static checks enforced in the validator: benchmark/v1 identity exclusions,
  no speaker shared across added splits, no duplicate ids/hashes, path-base
  enforcement (`path_base` required for every addition), near-duplicate
  fingerprints.
- `strict_conversion_family_compliant: false` is **reported, not passed** —
  conversion co-participation chains remain documented non-edges per the v2
  review decision; this is an explicit limitation, not a compliance claim.
- The benchmark was frozen: planning snapshot (27 artifacts) verified
  unchanged before and after (`snapshot_after_check.json`, 27/27 unchanged).

## 4. Data-integrity verification (independent)

All run on the frozen artifacts:

1. `scripts/prepare_dataset_v3_coverage.py replay` — zero-charge replan
   reproduces the frozen selection exactly (`selection_match: true`,
   digests equal, `replay_charges_bytes: 0`, manifest hashes match,
   additions match). Point-in-time fields (ledger snapshots, projected
   footer charges, planning estimate) are excluded from the selection
   digest and instead checked by an exact identity:
   `stored_est − footer = recomputed_est` (319,978,966 − 5,898,240 =
   314,080,726) — verified true.
2. `scripts/prepare_dataset_v3_coverage verify` — 27/27 frozen artifacts
   unchanged/not missing.
3. `scripts/validate_dataset_v3_coverage --fixtures` — 11/11 behavioral
   cases pass (benchmark male genuine; benchmark synthetic participant; old
   exposure identity; shared speaker across splits; clean acceptance; wrong
   path base; transitive bridge to benchmark; renamed duplicate hash;
   single-speaker dev-only acceptance; failed-read charge retention;
   allowance exhaustion blocks before IO).
4. `scripts/validate_dataset_v3_coverage` — **`ready: true`, 0 issues**:
   retention 384/96 exact, assignment + manifest hashes match, audio
   problems 0 (hash/decode/16 kHz mono/finite/1–4 s all pass), near-duplicate
   pairs 0, closure 113 checked / 0 hits, determinism all true, snapshot ok.
5. `pytest tests/dataset_prep/test_v3_coverage_validation.py` — 1 passed
   (wraps the fixture suite so regressions fail the test suite).

## 5. Ledger reconciliation (single shared ledger, cap 5,368,709,120)

| Checkpoint | Used bytes | Remaining | Events | Source |
| --- | ---: | ---: | ---: | --- |
| Task start (planning snapshot) | 4,377,793,162 | 990,915,958 | 2,144 | `planning/snapshot_before.json` |
| Frozen plan recorded (`ledger_before` = `ledger_after_planning`; planning charged **0**) | 4,377,858,778 | 990,850,342 | 2,147 | `planning/coverage_plan.json` |
| After **failed** materialize attempt 1 (all charges retained) | 4,699,365,861 | 669,343,259 | 2,283 | `staging_failed_attempt1/materialization_summary.json` |
| After successful materialize attempt 2 (final) | 5,025,803,914 | 342,905,206 | 2,332 | `staging/materialization_summary.json` |

Deltas and identity:

- Snapshot → plan: +65,616 bytes / +3 events of pre-plan probes; the plan
  projection was 314,080,726 (23 row-group reads) + 5,898,240 projected
  footer probes (90 probes × 64 KiB, `footer_charges.json`) = 319,978,966.
- Failed attempt 1: +321,507,083 = 23 row-group shard reads (315,608,843,
  `charged_skips`) + 90 footer probes (5,898,240). Cause: the new
  `SeekableRangeReader` initially lacked the `.bytes` attribute read by the
  fetch path, so all payloads were discarded **after** fully charged reads.
  Charges are retained on failure by design (pre-charge before read).
  Evidence is preserved under `staging_failed_attempt1/`; the failure and
  fix are valid audit history.
- Successful attempt 2: +326,438,053 (same read structure + minor retries);
  113/113 payloads written, all `fetched_v3`.
- Plan → completion: **+647,945,136 bytes** (`kathbath` 2,451,904,609 →
  3,099,849,745). Including the 65,616 pre-plan probe bytes, snapshot → completion
  consumed **648,010,752 bytes**, within the 750,000,000 soft cap for this task.
- Final headroom: **342,905,206 bytes** against the 5,368,709,120 cap.

## 6. Reproduce / re-verify

> The pipeline commands are archival: `snapshot`, `plan` and `materialize`
> write the frozen evidence files listed in § 1. Re-running them would
> overwrite that evidence (and `materialize` would charge the ledger again),
> so only the offline verification block should be re-run against this
> delivery; a fresh acquisition must use a new output root.

```bash
# pipeline (network / ledger-touching; do not re-run casually)
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage snapshot
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage plan
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage materialize
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage finalize
# verification (offline, zero-charge)
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage replay
./.venv-data/bin/python -m scripts.prepare_dataset_v3_coverage verify
./.venv-data/bin/python -m scripts.validate_dataset_v3_coverage --fixtures
./.venv-data/bin/python -m scripts.validate_dataset_v3_coverage
./.venv/bin/python -m pytest tests/dataset_prep/test_v3_coverage_validation.py -q
```

## 7. Shortfalls and limitations

- **Malayalam 0/0**: no eligible male speaker remains — every male pool
  candidate is under v1 prior exposure (15,445 recordings) or the benchmark
  (792 recordings); 12,468 non-male.
- **Odia 0 train / 1 dev**: the entire eligible male pool is
  benchmark-linked (23,863 recordings excluded); exactly 1 eligible
  recording survives (dev-only).
- **Bengali 2/2 (1 speaker per split)**: only 2 eligible speakers survive
  (4,435 v1-exposed, 4,860 benchmark); per-speaker cap allows 2 windows per
  split, so the soft targets cannot be met without new admissible speakers.
- These are **pool exhaustion** outcomes under the enforced exclusions, not
  planning or fetch failures; `trim_log` is empty (no global-budget trim).
- Near-duplicate detection is audio-only (RMS+ZCR 64+64 fixed bins, cosine
  ≥ 0.995, |Δduration| ≤ 0.05 s); metadata/provenance is not part of the
  fingerprint — documented limitation, 0 pairs flagged among additions.
- Gender is taken from the documented Kathbath filename convention covered
  per row; no independent acoustic gender verification was performed.

## 8. Handoff note for the next agent (training blocker)

- **Existing feature caches are stale** for the expanded v3 manifests
  (`artifacts/datasets/features/*.pt` and any detector-side window caches
  were built from the v2 train/dev split).
- `scripts/training_split_contract.py --split-version` accepts **only
  byte-identical v2 copies**; the v3 manifests are a new (combined) dataset,
  so the guard will correctly refuse reuse. The next agent must implement
  dataset-aware feature-cache generation and training that reads the v3
  manifests with `path_base` resolution (`dataset-version.v3.json` documents
  the resolution rule) **before any training**. Do not bypass the guard or
  reuse old caches with new labels/IDs.
- `artifacts/datasets-v3-coverage/manifests/dataset-version.v3.json`
  carries an explicit `cache_staleness` warning for the same reason.
- The evaluated benchmark (`artifacts/evaluations/gru-v2-test-epoch5/`) is
  unchanged and remains the already-evaluated benchmark; no evaluation of v3 is
  claimed here.
