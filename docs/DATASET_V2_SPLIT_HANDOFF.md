# VoxSentinel v2 Indic split — handoff document

Status date: 2026-09-19. Branch context: `feature/a1-datasets-manifests` (nothing about
this work was committed, pushed or merged; files live in the working tree).

This document is for teammates who need to reproduce, audit, or consume the v2 Indic
train/development/test split. It lists required gitignored assets, the exact build
commands, verified results, and known limitations.

## Final scope and budget correction

The retained split uses **`speaker_recording_disjoint_v2`**: speaker and parent-recording identities are disjoint across all roles and admitted recordings stay within their split. Conversion pair chains through unused candidates are not exclusion edges. **The original strict conversion-family contract is not met (67/96 diagnostic hits).** This is an explicit revised research protocol, not a claim that strict isolation passed or that indirect relationships are proven harmless. No model evaluation has occurred.

The audio-shard fetcher now reads serially, disables read-ahead and charges each bounded read **before** I/O against the shared global ledger and a shared per-language allowance. Failed and short reads retain their reservation, so new accounting is conservative requested payload bytes, not exact wire traffic. Run only one acquisition job at a time; metadata discovery and other download adapters have not been hardened by this targeted fix. Historical reconciliation remains an estimate. No downloads occurred during this correction.

Verification: 8 focused tests passed, including cumulative global-cap rejection before I/O, language-cap rejection, unbounded-read rejection, failure accounting and seek/readinto behavior. A local PyArrow read through the new wrapper also passed. Live network compatibility was not exercised.

## Status fields

| Field | Value |
|---|---|
| `indic_split_integrity` | **pass under speaker_recording_disjoint_v2**; original strict family rule not met |
| `coverage_limitations` | thin male pools for Odia (4 upstream speakers) and Malayalam (7); synthetic transcript unknowns (25/48); near-dup method scope (audio-only fingerprints); no cross-language identity linkage; strict co-participation adjacency reported as diagnostic (see closure section) |
| `untouched_test_status` | **audited**: 96 windows outside prior-exposure closure — direct membership, transitive admissible-relationship closure, cross-split components and upstream held-out all zero hits |
| `indian_english_status` | **pending IndieFake access** (requested; no archive/link provided yet). This is not a failure of Indic manifest integrity. |
| `model_evaluation_performed` | **false** |

## Review follow-up (2026-09-19, after `DATASET_V2_REVIEW.md`)

The agent’s earlier follow-up is recorded below. The final scope/budget correction above supersedes its claims of complete compliance:

1. **Exposure closure** — the reviewer's 67-window co-participation result was reproduced exactly
   (`reports/closure_audit.json`, variants matrix + per-window shortest paths). Edge provenance was then
   audited: no shared person, recording or hash connects any of those windows to exposure — every path is a
   conversion co-participation chain (source↔target voice pairing in scanned rows; e.g. Bengali 278→960,
   1174→960, 99→1174, 99→1038). The strict cross-union reading is unsatisfiable on real data (100% of
   Malayalam/Marathi synthetic candidates poisoned → no compliant test possible), so it is reported as a
   **diagnostic**, not an exclusion rule. The enforced closure is the *admissible* relationship graph
   (person↔parent-recording links per row side, transitive through shared exact keys, kathbath inventory
   label-conflict merges, held-out anchors), computed both in selection (`build_closure_state`) and
   independently in the validator (`load_closure_data`/`audit_closure`): **0 hits over 96 windows**,
   now guarded by fixtures (multi-hop bridge rejected; disjoint case passed).
2. **Parent verification** — verified target record **and** verified source record (or `not_applicable_tts`)
   are now hard admission rules; `parsed_only` parents are quarantine-by-admission with counted reasons
   (1,371 target + 631 source candidates excluded; **29 selected synthetic windows were replaced**).
   Both previously flagged windows (`indicsynth-bengali-012765`, `indicsynth-sanskrit-102314`) are out.
   Manifests now carry **actual lookup evidence** (`reference_evidence.*.found_in` = v1_scan_inventory /
   v2_shard_scan, matching record id) instead of restating the reference; the validator independently
   re-checks both roles, evidence completeness and record membership. Unknown-metadata reporting now
   includes the source role (0 parsed-only in the final test for both roles).
3. **Ledger** — `fetch_kathbath_audio_pinned` now charges **every shard read** whether or not the payload is
   retained, and skips over-cap shards **before reading** when cached footer estimates exist; skip events
   are recorded with charged/estimated bytes. The historical uncharged discard (the pre-fix cap-skip of
   `malayalam/train-00027-of-00033.parquet` row group 12) was computed from pinned shard metadata
   (33,915,343 compressed audio bytes + footer) and **charged as an explicit reconciliation entry**
   (`reports/ledger_reconciliation.json`; estimate method and ±delta uncertainty documented).
4. **Tests** — fixture matrix expanded 11 → **19 cases + 2 audio cases**, all passing, including: the
   transitive A→B→C bridge (reject) with a disjoint control (accept), quota-overflow component split
   across splits (reject) vs contained within test (accept), unverified source/target parents, missing
   evidence, ghost record membership, and a re-encoded (non-byte-identical) audio pair that the
   fingerprint detector must catch while hash equality alone would miss it. The pytest suite now asserts
   behaviors (closure multi-hop unit test, required-case effectiveness, fingerprint vs hash), not counts.

## What the split is

- Source corpora (revision-pinned): Kathbath (genuine) @ `5b9e92849222026d9141acba4e8434fe816396bf`;
  IndicSynth (synthetic) @ `c0a10386b723717aff682f757bd67f72983f269f`.
- 12 languages x (4 genuine + 4 synthetic) = **96 test windows / 96 recordings**.
- Test content: 48 genuine (24 m / 24 f), 48 synthetic (freevc24 25, xtts_v2 21, vits 2).
- Unique test identities: 48 genuine speakers, 24 synthetic source speakers, 47 synthetic
  target speakers; relationship graph resolves to **84 components** (per language 4–8).
- Duration: test 382.9 s (0.106 h); dev 383.8 s (0.107 h); train 1535.5 s (0.426 h).
- Original rates: Kathbath 16 kHz (48); IndicSynth 24 kHz (46), 22.05 kHz (2).
- Metadata unknowns in test manifests: transcript not exposed upstream (25 synthetic);
  source/target parent parsed-only: **0 / 0** (verified-parent admission).

Splits and counts by language are in `artifacts/datasets-v2/reports/report.md`
(+ `report.json`). All 12 languages: 4 genuine + 4 synthetic.

## Manifest outputs (schema `voxsentinel.dataset_version.v2` seed 42)

| File | sha256 |
|---|---|
| `artifacts/datasets-v2/manifests/windows.train.jsonl` | `1b361e14…53ad4` (byte-identical copy of v1 core train) |
| `artifacts/datasets-v2/manifests/windows.dev.jsonl` | `a23acb1c…e3342` (byte-identical copy of v1 core val) |
| `artifacts/datasets-v2/manifests/windows.test.jsonl` | `1d7042f9…a7115e` |
| `artifacts/datasets-v2/manifests/windows.test.excluded.jsonl` | `e3b0c442…b855` (empty) |
| `artifacts/datasets-v2/manifests/dataset-version.v2.json` | full manifest/config hashes + revisions |

Prepared window files use preprocessing `voxsentinel-prep-2` (mono mean, soxr HQ to
16 kHz, overflow attenuation only, 1–4 s high-energy windowing) and resolve under
`artifacts/datasets-v2/prepared/...`.

## What was reused vs. added

- **Reused unchanged (frozen v1)**: all 384 train + 96 dev windows and their prepared
  audio; feature caches; GRU run artifacts; configs. 17/17 snapshot hashes verified
  unchanged (`planning/snapshot_before.json` vs `planning/snapshot_after_check.json`);
  only the shared ledger changed (documented exception).
- **Added**: 96 test windows materialized from previously-unscanned Kathbath shards
  (genuine) and fresh IndicSynth rows (synthetic), all outside the v1 exposure closure.
  After the review follow-up, **29 of the 48 synthetic windows were re-selected** to satisfy
  the verified-parent admission rule (the deterministic seed and plan changed accordingly;
  no genuine windows were affected).
- **Reserved / excluded (accounted, never silently dropped)**:
  - 387,827 shortlisted-but-unused candidates in `planning/accounting.<Language>.jsonl`;
  - 2 quarantined Malayalam recordings with missing upstream audio on first fetch
    (`planning/quarantine.json`; replacements selected and materialized);
  - a superset of fetched raw audio retained read-only in `raw/` (some belongs to
    earlier selection passes); nothing was deleted anywhere.
- **Upstream held-out protected**: Kathbath valid/test record ids verified for all 12
  languages (`staging/kathbath_heldout.json`, status `verified` each); 0 overlap in test.

## Verification evidence (all on the final state)

- `reports/audit.json` — independent validator over the emitted manifests:
  `ready: true`, 0 issues / 0 problems; 96/96 accounted; held-out overlap 0;
  direct closure 0; **transitive admissible closure 0/96** (strict co-participation
  diagnostic: 67, adjudicated non-leak — see closure section); cross-split components 0;
  parent verification/evidence checks 0 problems (source and target).
- `reports/replay_check.json` — deterministic replay from frozen inputs:
  `selection_match: true`, `input_hash_match: true`, 96 selected.
- Adversarial fixtures: `validate_dataset_v2 --fixtures` → passed — **19 cases + 2 audio cases**
  (cross-split identity leaks incl. multi-hop A→B→C via injected closure edges, quota-overflow
  components split vs contained, renamed duplicates, unverified/ghost parents, int/str id
  collisions, and a re-encoded near-duplicate the fingerprint detector catches while hash
  equality would miss it).
- `tests/dataset_prep/test_v2_validation.py` → **4 passed** (behavioral assertions, incl. a direct
  multi-hop closure unit test and fingerprint-vs-hash test; no count-only assertions).
- Determinism: two consecutive full `plan` runs produce identical selections
  (`diff` clean), including from a warm byte-size cache.
- Defects fixed during this work (all re-verified from scratch afterwards):
  1. row-group size cache keys were int in memory but strings after JSON round-trip, silently
     degrading cost-aware selection on the cached path (fixed by int-key normalization in
     `estimate_group_bytes`);
  2. the cap-skip path in `fetch_kathbath_audio_pinned` discarded already-read shards without
     charging (now: every read charged, pre-read gating via cached estimates, skip events
     recorded; historical discarded read reconciled into the ledger).

## Ledger reconciliation

`artifacts/datasets/manifests/downloads.json` (shared 5,368,709,120-byte cap):

| | total used | remaining | events |
|---|---|---|---|
| before (session start) | 3,620,497,163 | 1,748,211,957 | 1,379 |
| after | **4,377,793,162** | **990,915,958** | 2,144 |

Per source (after): kathbath 2,451,838,993 · nisp 1,294,991,360 · indicsynth 283,402,380 ·
nptel 188,815,224 · svarah 142,472,502 · asvspoof2019 16,272,703.
V2 work consumed ~758 MB (shard scans, held-out scans, footer counts, materialization,
experiments, reconciliation). The historical uncharged discard (pre-fix cap skip of one
Malayalam row group, 33,915,343 compressed bytes + footer) was computed from pinned shard
metadata and charged as an explicit reconciliation entry — method and ±delta uncertainty in
`reports/ledger_reconciliation.json`.

## Required gitignored assets (must exist locally)

- `models/indicwav2vec_large.pt` (large checkout asset; not used by this task).
- `artifacts/datasets-v2/raw/**` (fetched genuine/synthetic source audio cache),
  `artifacts/datasets-v2/prepared/**` (prep-2 windows), `staging/**` (scan inventories),
  `planning/**` (exposure registry, byte-size caches, plan/selection/accounting),
  `manifests/**`, `reports/**`.
- Shared ledger `artifacts/datasets/manifests/downloads.json` and frozen v1 tree
  `artifacts/datasets/**` (read-only).
- Python envs: `.venv-data` (acquisition/prep, py3.12) and `.venv` (tests, py3.10).

## Rebuild commands (idempotent, revision-pinned, ledger-charged)

```bash
V=./.venv-data/bin/python
$V -m scripts.prepare_dataset_v2 discover-kathbath      # all 12 languages, full shard scan
$V -m scripts.prepare_dataset_v2 discover-heldout       # upstream valid/test record ids
$V -m scripts.prepare_dataset_v2 discover-indicsynth    # fresh synthetic candidates
$V -m scripts.prepare_dataset_v2 scan-counts            # footer-only unparsed accounting
$V -m scripts.prepare_dataset_v2 plan                   # seed 42, 4/class/language
$V -m scripts.prepare_dataset_v2 materialize --byte-cap-per-language 128000000
$V -m scripts.prepare_dataset_v2 finalize
$V -m scripts.validate_dataset_v2                       # independent audit -> reports/audit.json
$V -m scripts.prepare_dataset_v2 replay                 # determinism check
$V -m scripts.report_dataset_v2                         # reports/report.{md,json}
$V -m scripts.verify_v2_snapshot                        # v1 freeze verification
```

## Limitations (read before using test)

1. Near-duplicate protection is audio-only (fixed-bin RMS+ZCR fingerprints, 0.995
   threshold, 0.05 s duration tolerance). It catches direct re-encodes/duplicates; it is
   not a guarantee against all transcode families. 0 flagged pairs in the final test.
2. Cross-language speaker independence is NOT claimed; upstream identity documentation
   cannot support it. Identity checks are language-scoped.
3. Synthetic transcripts are upstream-partial (25/48 test synthetic rows lack exposed
   transcripts); flags are carried in provenance, not imputed.
4. Odia used 2 of 4 available upstream male speakers; Malayalam 2 of 7. Male pools are
   genuinely thin for these two languages upstream. Under the verified-parent rule,
   Malayalam has only 3 distinct eligible synthetic target speakers (one target repeats
   across two test windows — permitted within a split, integrity over quotas).
5. Generator representation is uneven (freevc24 25 / xtts_v2 21 / vits 2). This is not
   an unseen-generator evaluation.
6. Byte-size estimates for Kathbath row groups are compressed-column footers, not exact
   transfer costs; selection is cost-aware, not cost-exact. The reconciliation charge for
   the historical discarded read is an estimate with documented ±delta uncertainty.
7. **Closure adjudication**: conversion co-participation adjacency (source↔target voice
   pairing anywhere in the scanned candidate graph) is outside the revised exclusion rule after edge
   provenance audit — no shared person/recording/hash connects the flagged windows to
   exposure, and the strict reading cannot be satisfied by any materialized pool for
   Malayalam/Marathi. If the project later decides co-participation implies a family
   relationship that must be split-exclusive, the affected-window list is ready in
   `reports/closure_audit.json` (strict diagnostic: 67/96 windows) and the small
   admissible-component case is already enforced.
8. IndieFake / Indian-English remains pending (status above).

No training, feature extraction, detector evaluation, threshold calibration, commit, or
push was performed. No files were deleted.
