> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Manager report — reference-lineage resolution, dataset freeze and first GRU run

Prepared for: managing agent / reviewer
Prepared: 2026-09-18 (Asia/Kolkata)
Branch: `feature/indicVac2Wav` @ `c250bbd0ac3b2af1ae345ce48960fcc63d200557` (17 dirty items preserved; **nothing committed**)
Companion documents: `docs/archive/history/dataset-preparation-pilot.md` (§10 = run record), `artifacts/runs/gru-core-v1/training_report.md`, `artifacts/datasets/reports/preparation-report.{json,md}`

---

## 1. Executive summary

| Goal (from the authorizing prompt) | Status | Evidence |
| --- | --- | --- |
| Resolve `parsed_only` / weaker reference lineage (§2A) | **Done** — 366/366 applicable roles `verified_train`, 0 unresolved; 12 rows replaced + 72 candidates pre-excluded; all quotas preserved | `manifests/source_windows.indicsynth.jsonl` (`reference_evidence`, `upstream`); report `lineage_resolution` block |
| TTS vs VC field semantics; preserve raw metadata; keep conditioning material in graph/checks (§2B) | **Done** — documented; `parent_refs.upstream` per record; TTS conditioning recordings retained in relationship graph and split checks | same manifest fields; split audit below |
| Remove stale 4.25 s section; verify actual audio vs `voxsentinel-prep-2` (§3) | **Done** — stale section removed; 480/480 files audited clean (mono/16 kHz/finite/≤1.0/1–4 s contiguous/no padding; SHAs match) | audit output captured in session; `windows.core_*.jsonl` |
| Freeze a complete consistent dataset version (§3) | **Done** — `manifests/dataset-version.json` (14 manifest hashes, cache identities, seed 42, prep v2) | file |
| Readiness checks via existing entry point (§4) | **Done** — `--check-only` audit passes with **zero failures** (counts, labels, per-language dual-class, manifest content signature, deep bundle identity) | `artifacts/runs/gru-core-v1/cache_audit.json` |
| ONE controlled training run (§5–§6) | **Completed** — 10 epochs, 43.9 s head-training on frozen features; selected epoch 2; full artifacts saved | `artifacts/runs/gru-core-v1/` |
| Reproducible artifacts + final report (§7) | **Done** — settings/dataset identity/code diff+hashes/env/history/checkpoints/predictions/metrics/breakdown/report | run dir listing below |

**Headline result:** validation EER **0.229** at the selected epoch, but the model's scores collapse to one side at the fixed 0.5 threshold (all 48 genuine flagged, all 48 synthetic caught). This is reported as an honest negative/partial result — **no success claim**.

---

## 2. What was done, in order

### 2.1 Reference-lineage resolution (prompt §2A)

- **Scope found by audit:** across 240 synthetic core windows, 194 distinct references (41 VC-source, 153 target/conditioning) were weaker than confirmed membership (`parsed_only`).
- **Method:** new tool `scripts/resolve_lineage.py` (`scan | apply | crosscheck`):
  - reads every train shard's parquet footer (~64 KiB) and uses the `fname` column's min/max statistics as an exact index (record id = prefix of `fname`);
  - reads only the bracketing row groups' `fname` column chunks to *confirm* membership, producing evidence `{method, shard, row_group, row_index, fname}`;
  - searches are parallel (6–8 workers) and charged to the normal download ledger.
- **Outcome:** all 12 languages processed; **366/366 applicable roles verified** (130 via scanned inventory, 236 via full-shard scan); 114 TTS-source roles `not_applicable_tts` (upstream fields are null — generation semantics, not missing data). No `parsed_only` status remains.
- **Replacements (evidence-driven):** references that exist nowhere in the release were replaced, never concealed:
  - round 1: 27 rows (Bengali 3, Gujarati 2, Kannada 4, Marathi 2, Odia 3, Sanskrit 13);
  - rounds 2–3 re-verified the new selections with a **reference-knowledge pre-exclusion** (candidates whose refs are cached `not_found`/`upstream_eval_parent` can no longer be selected) — final residual: 0.
  - totals: **12 selected rows replaced** (`lineage_replacement_required`), **72 candidates pre-excluded** (`reference_unresolvable_precheck`), itemised in `manifests/windows.excluded.jsonl`.
- **Deviations recorded:** a cross-folder confirmation pass (references may in principle live in another language's Kathbath folder) was attempted with a hard 120 MB cap, but the CDN's sustained SSL/read instability made it non-convergent; per policy the remaining refs were treated as unresolved and **replaced** instead of half-verified. All replacement rows are verified in their own language folder or the scan recorded their absence explicitly.

### 2.2 TTS vs VC fields (prompt §2B)

- Raw schema in `staging/joint/indicsynth_candidates.*.jsonl` shows: `freevc24` rows carry `source_record/source_reference/source_recording_speaker` + target side; `xtts_v2`/`vits` rows have **null** source fields upstream — a VC-specific source step does not exist for TTS (recorded `source_kind: tts_target_only`, `not_applicable_tts`).
- The real TTS conditioning material (`target_reference` + target speaker) is **kept**, unioned into the relationship graph, and counted in all split checks.
- **Raw metadata is preserved per record** in `parent_refs.upstream` (dataset/language/split/row_index/generator/transcript + per-side record/reference/declared speaker/kind).

### 2.3 Cleanup, freeze, and integrity (prompt §3)

- Stale duplicate "4.25 s allowance" preprocessing section deleted (docs now describe the strict 4 s `voxsentinel-prep-2` contract only).
- Actual-file audit of all 480 core windows: mono, 16 kHz, float, finite, max peak ≤ 1.0, duration 1–4 s, contiguous spans inside originals, prepared SHA-256 matches manifests — **0 problems**.
- Cross-split audit (scoped by dataset+language): **speaker overlap 0, reference overlap 0, prepared-SHA overlap 0, original-audio-SHA overlap 0**.
- Pool isolation confirmed: training reads only `windows.core_train.jsonl`, validation only `windows.core_val.jsonl`; `window_id` sets disjoint.
- Label mapping confirmed against the two-logit cross-entropy head: **0 = genuine, 1 = synthetic/spoof**.
- Cache identity upgraded to a **content signature** (window membership + labels + languages + prepared hashes) with the raw manifest file hash kept informationally — metadata-only edits no longer invalidate features; a one-time rebuild produced **384/96** items with `complete: true` sidecars.

### 2.4 Repairs to shared infrastructure (found during the work; all verified)

| Issue found | Fix | Where |
| --- | --- | --- |
| Incremental `core` runs unioned rows by id — a re-planned language kept stale superseded selections (observed: 507 windows) | A re-planned language now **replaces** its previous rows; staged 480 promoted exactly; 27/12 orphaned prepared files pruned | `src/dataset_prep/joint_core.py` staging merge |
| Ledger accounting lost Svarah charges (concurrent stage writes) | `charge()` re-syncs with the on-disk ledger before every charge; Svarah's 142,472,502 bytes reconciled explicitly | `src/dataset_prep/budget.py` |
| Resolver wrote caches under `staging/staging/joint` (path constant double-prefix) | Path fixed; stray caches relocated; `not_found` re-classified in `apply` | `scripts/resolve_lineage.py` |
| Replacement plans could be 1 window short (Sanskrit val 3/4) | Soft quota top-up pass with global per-speaker caps retained | `_select_synthetic` top-up in `joint_core.py` |
| Stale exclusions contradicting manifests | `assemble_pools` prunes exclusions for materialized recordings | `src/dataset_prep/assemble.py` |
| Kathbath `.m4a` blobs | Payloads are FLAC 16 kHz PCM (libsndfile decode, recorded per window); ffmpeg fallback remains for unsupported formats | materialization facts |

---

## 3. Final dataset composition (frozen)

**Core (this experiment):** 480 windows = 12 languages × (16 genuine + 16 synthetic train, 4 + 4 val); labels balanced per split (192/192, 48/48).

| Dataset | Windows | Revision (recorded) | Notes |
| --- | ---: | --- | --- |
| Kathbath (genuine) | 240 | `5b9e92849222d9141acba4e8434fe816396bf` | row-group reads; per-language shards |
| IndicSynth (synthetic) | 240 | `c0a10386b723717aff682f757bd67f72983f269f` | generators: freevc24 127, xtts_v2 108, vits 5 |

**Other pools (out of scope for training, intact):** external_eval 20 (Svarah `ebbf7777fe77…`), fallback_baseline 50 (ASVspoof 2019 LA; license text extracted, sha `99a4d3e0…`), supplementary 10 (NPTEL), unpaired_candidate 40 (NISP), missing_coverage 4 (SPIRE-SIES, Indic TIMIT, IndicVoices, synthetic English).

**Ledger:** 3,620,497,163 of 5,368,709,120 bytes used (~3.37 GiB; headroom ~1.63 GiB). Per source: kathbath 1.73 GB, nisp 1.29 GB, indicsynth 248 MB, nptel 189 MB, svarah 142 MB, asvspoof 16 MB.

**Exclusions manifest:** 8,853 rows — 8,768 `candidate_pool_not_selected`, 12 `lineage_replacement_required`, 72 `reference_unresolvable_precheck`, 1 `audio_fetch_missing` (historical, verified superseded).

---

## 4. Readiness checks actually executed (prompt §4)

| Check | Result |
| --- | --- |
| `validate_core_training_cache` (via `.venv/bin/python -m scripts.train_gru_from_cache --check-only`) | **ready: true**, failures: none, skipped: none — completeness markers, manifest content signatures, item counts 384/96, dual-class per split, per-language dual-class (12 languages), deep bundle identity + item counts |
| Label mapping vs head contract | 0 = genuine, 1 = synthetic; head uses two-logit cross-entropy (documented in `gru.py`, `training.py`, `metrics.py`) |
| Pool isolation | train/val bundles built from separate manifests; no code path combines pools |
| Lineage (separate from the cache audit) | `resolve_lineage` evidence complete (this report §2.1) |
| Audio contract | 480-file audit clean (§2.3) |

No broad test-suite expansion was run (not requested); checks were limited to the gates above.

---

## 5. The training run (prompt §5–§6)

**Command:** `.venv/bin/python -m scripts.train_gru_from_cache --run-dir artifacts/runs/gru-core-v1`
**Environment:** `.venv` (Python 3.10.18, torch/torchaudio 2.2.2, fairseq 0.12.1, numpy 1.23.5), CPU float32, 4 torch threads.

**Resolved settings:** GRU head (1024→256, 1 layer, dropout 0); AdamW lr 1e-3, wd 0; batch 8; **10 epochs**; seed 0; no scheduler; no gradient clipping; no class weights/oversampling; checkpoint selection = lowest validation EER, earliest epoch wins ties. Features reused from the frozen cache (feature-regeneration time = 0; head training **43.9 s**).

**Epoch history (extract):**

| epoch | train loss | val loss | val acc | val EER |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.7346 | 0.7215 | 0.500 | 0.396 |
| 2 | 0.7113 | 0.7031 | 0.500 | **0.229** |
| 10 | 0.7006 | 0.6939 | 0.500 | 0.333 |

**Selected epoch 2** (best val EER 0.2292, mode min). Best-checkpoint reload was **performed** (`restore_checkpoint`) and predictions come from the reloaded checkpoint.

**Best-checkpoint validation metrics at threshold 0.5** (synthetic = positive; score = softmax P(synthetic)):

- accuracy 0.50; balanced accuracy 0.50; precision 0.50; recall 1.00; F1 0.667; **EER 0.229**
- confusion (rows = true, cols = predicted; order [genuine, synthetic]): **TN 0, FP 48, FN 0, TP 48**
- **genuine false-alarm rate 1.00; synthetic miss rate 0.00**

**Failure pattern:** at the fixed 0.5 threshold every validation window scores ≥ 0.5 — a score-bias collapse, not a per-language or per-generator failure (breakdown shows miss 0 / FPR 1.0 uniformly; e.g. `generator:freevc24` 25/25 detected, `generator:xtts_v2` 21/21, `generator:vits` 2/2, `generator:genuine` 48/48 false alarms). The ranking is weakly informative (EER 0.229 vs chance 0.5). No threshold or calibration was tuned (per instructions); EER is a ranking summary, not a deployed threshold.

**Interpretation (modest):** 384 train / 96 dev windows (≤ 25.6 / 6.4 minutes of audio); 8 dev windows per language; checkpoint chosen on this validation. These are development results; no production accuracy, unseen-generator robustness, Indian-English coverage, calibration, or latency claim exists.

**Run artifacts (`artifacts/runs/gru-core-v1/`):** `settings.json`, `dataset_identity.json` (manifest/bundle SHA-256s, cache identities, label mapping, ledger snapshot), `verification_summary.json`, `cache_audit.json`, `environment.json`, `code_state.json` + `code_diff.patch` + `git_status.txt` (39 changed/untracked source files hashed; commit `c250bbd` alone is insufficient for the dirty checkout), `history.{csv,json}`, `gru_best.pt`, `gru_final.pt`, `val_predictions.{csv,jsonl}` (window/recording/dataset ids, language, generator, split, label, speaker ids, source/target references + verification statuses, logits both classes, synthetic score), `metrics.json`, `breakdown.json` (per language and per generator with denominators), `training_report.md`, `run_status.json` (completed; started/finished timestamps; `checkpoint_reload_verified: true`; `resumed: false`).

**Encoder manifest alongside the detector:** the encoder reference (checkpoint SHA-256 `26bb5ada…1559`, layer = final output, dim 1024, prep `voxsentinel-prep-2`) travels in `metadata` inside both checkpoints and in `dataset_identity.json`/`settings.json`; the detector checkpoint does not embed B1's weights.

---

## 6. Repository & environment state

- Branch `feature/indicVac2Wav` @ `c250bbd`; 17 dirty items preserved; no commit/push/reset performed (not authorized).
- Data environments: `.venv-data` (Python 3.12) for acquisition/preparation; `.venv` (Python 3.10.18) for the frozen encoder/training. No dependency upgrades were made to satisfy acquisition tooling.
- Artifacts are Git-ignored under `artifacts/` (raw, prepared, manifests, features, runs, reports).

---

## 7. Known issues / risk register

| # | Item | Severity | Status / mitigation |
| --- | --- | --- | --- |
| 1 | Score-bias collapse at threshold 0.5 (FPR 1.0) | High (for deployment claims) | Reported; threshold/calibration study deferred by instruction; no claim made |
| 2 | Weak ranking signal (EER 0.229) on dev | Medium | More data/speakers, head/layer variants, per-language diagnostics queued as next evidence |
| 3 | Cross-folder lineage confirmation incomplete (CDN instability; 120 MB cap) | Low | Affected refs were replaced with locally verified ones; absence recorded in exclusion manifest |
| 4 | Indian-accented English training blocked | Medium | `synthetic_english` still `missing_source`; NISP stays `unpaired_candidate` |
| 5 | IndicVoices accessible but unprocessed | Low | `accessible_not_processed`; same parquet/row-group pattern as Svarah |
| 6 | SPIRE-SIES / Indic TIMIT | Low | `awaiting_user_links` |
| 7 | No ROC-AUC/plot tooling in the pinned env | Low | EER provided by existing metrics; CSV/JSON history complete (matplotlib absent) |
| 8 | Dirty working tree not committed | Info | Per authorization; `code_diff.patch` + hashes reconstruct the run |

---

## 8. Reproduce / verify runbook

```sh
# data environment: verify frozen state
python -m scripts.prepare_datasets status
python -m scripts.prepare_datasets report            # regenerates reports + dataset-version.json
python -m scripts.resolve_lineage apply              # re-apply evidence idempotently

# encoder environment: readiness + training
.venv/bin/python -m scripts.train_gru_from_cache --check-only
.venv/bin/python -m scripts.train_gru_from_cache --run-dir artifacts/runs/<unique-name>
```

Expected: audit `ready: true`; a new unique run directory; `run_status.json.status == "completed"`; best checkpoint selected by lowest validation EER.

---

## 9. Recommended next steps (for the managing agent)

1. **Decision-threshold / calibration analysis** on an independent set (e.g. Svarah/IndicVoices external pools for English; a held-out slice for Indic) — the EER operating point is not a deployed threshold.
2. **Data growth**: more speakers/windows per language (Kathbath row-group reads make this cheap), keeping disjoint-pair construction and the verification loop (`scan → apply → replace`).
3. **Model/feature ablations (bounded)**: alternative encoder layers, sequence pooling variants, dropout/regularization — pre-register the selection rule.
4. **Indian-English track**: only after a traceable synthetic-English source is supplied (`configs/synthetic_english_template.json`).
5. **External-eval extension**: add IndicVoices windows during the next acquisition window (pattern proven).
6. **Packaging**: if checkpoints must ship, keep encoder reference + dataset-version hashes alongside; never commit weights to the repo (existing policy).

---

## 10. Evidence index (paths)

- Lineage: `manifests/source_windows.indicsynth.jsonl` (+`source_recordings.*`), `staging/joint/lineage_resolution.*.json`, `staging/joint/lineage_excluded.*.json`
- Pools: `manifests/windows.core_{train,val}.jsonl`, `windows.excluded.jsonl`, `windows.missing_coverage.jsonl`
- Freeze: `manifests/dataset-version.json`, `reports/preparation-report.{json,md}`
- Features: `features/train.pt` (384) + `train.meta.json`, `features/val.pt` (96) + `val.meta.json`
- Run: `artifacts/runs/gru-core-v1/` (see §5 listing)
- Code changes: `code_diff.patch`, `code_state.json` (39 hashes) in the run dir; sources under `src/dataset_prep/`, `scripts/`

No tests, detector training beyond the single configured run, benchmarks or evaluation claims are made beyond the numbers above.
