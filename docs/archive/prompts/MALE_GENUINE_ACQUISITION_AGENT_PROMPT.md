> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# VoxSentinel handoff: acquire and add genuine training/development coverage only

Work in `/Users/mouryabs/VoxSentinel`. Your task is to acquire additional documented male genuine speech and prepare a new version of the Indic training/development manifests. Preserve existing useful data and frozen artifacts. Stop after materialization and data-integrity validation. Another agent will train, analyze and prepare the branch handoff afterward.

## Read first

Read applicable AGENTS.md instructions, then:

- `docs/archive/history/V2_FROZEN_TEST_RESULTS.md`
- `docs/archive/history/V2_SPLIT_TRAINING_REPRODUCTION.md`
- `docs/archive/history/DATASET_V2_SPLIT_HANDOFF.md` (especially its final scope/budget correction)
- `docs/archive/history/DATASET_V2_REVIEW.md` (including targeted completion)
- `src/dataset_prep/split_v2.py`, `bounded_reader.py`, `budget.py`
- `scripts/validate_dataset_v2.py`, `scripts/training_split_contract.py`
- `artifacts/datasets-v2/manifests/dataset-version.v2.json`
- `artifacts/datasets-v2/planning/exposure_registry.v2.json`
- `artifacts/evaluations/gru-v2-test-epoch5/exposure.json` and `plan.json`

There are intentional uncommitted changes. Inspect Git status and preserve them. Do not reset, stash, clean, commit, push or merge. Historical prompts may contain superseded exact-gender-balance or strict conversion-family requirements; this prompt and the final protocol decision control this task.

## Verified current state

- Training: 384 windows, 192 genuine / 192 synthetic; development: 96 windows, 48 genuine / 48 synthetic.
- Twelve languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu.
- Existing genuine training/development examples are all documented female. Synthetic training includes male and female targets.
- The v2 train/dev manifests are byte-identical copies of the old core. A fresh ten-epoch GRU run reproduced the existing standardized baseline exactly; it did not add training coverage.
- The fixed epoch-five model was evaluated on 96 new test windows at threshold 0.5. Accuracy 79.17%, EER 16.67%, genuine false alarms 17/48, spoof misses 3/48. Genuine male false alarms were 16/24 versus female 1/24.
- These results motivate coverage acquisition. They do NOT prove gender causally explains the failures. Do not diagnose model internals or change the threshold.
- Male genuine speech DOES exist in the scanned pinned Kathbath source. Local v2 discovery found it in all 12 languages; Malayalam/Odia pools are thin. Inspect actual candidate inventories and remaining eligible speakers before promising quotas.
- The v2 test has now been evaluated. It is an existing benchmark, not a new untouched test for adaptive decisions based on these results.

## Scope and acquisition policy

Use the existing pinned Kathbath genuine source:
`5b9e92849222026d9141acba4e8434fe816396bf`.
Retain existing IndicSynth synthetic examples and their provenance at revision
`c0a10386b723717aff682f757bd67f72983f269f`.

No IndieFake archive/link is available; the user already requested access. Keep Indian English pending. Do not submit access requests, contact anyone, add unrelated corpora, generate synthetic audio, change dataset revisions or acquire a large archive.

Reuse cached metadata and available eligible raw audio first. Inspect `artifacts/datasets-v2/staging/kathbath_fresh.<Language>.jsonl`, old inventories, scan/held-out evidence and the raw cache. A fetched-but-unused recording may be reused only after the same identity, exposure and provenance checks as a new recording. Raw bytes may be reused read-only; never mutate originals.

Acquire male genuine additions for both training and development. Also retain all existing female genuine and synthetic examples. Do not downsample female speech, duplicate windows, infer gender from audio/names, oversample manifests or claim exact balance is necessary.

A bounded initial target is **8 additional male genuine training windows and 4 additional male genuine development windows per language**, preferably from at least four training speakers and two development speakers, with no more than two additions per speaker within a split. These are soft targets, not a reason to violate identity rules or discard data. At full feasibility this adds 96 train + 48 dev windows, producing 480 train and 144 dev windows. Report actual achieved counts; never fill a shortage with duplicates or exposed speakers. Keep the existing synthetic counts unchanged in this acquisition task; report resulting class ratios for the later training decision.

Before downloads, produce a metadata-only feasibility table by language showing eligible male speakers/recordings after all exclusions, proposed train/dev assignment, existing reservations, expected acquisition costs and shortfalls. If a language cannot support both splits with distinct eligible identities, prefer preserving at least one independent development speaker when possible, allocate remaining independent speakers to training, and report the unmet target. With only one eligible speaker, reserve it for development and mark training coverage unresolved rather than share it across splits. These reduced targets do not establish adequate evaluation coverage. Do not silently reduce speaker separation to meet clip counts.

## Non-negotiable split rules

Use the explicitly adopted **`speaker_recording_disjoint_v2`** protocol:

- Check genuine speaker, synthetic source and synthetic target identities against every other split in all cross-role combinations.
- Keep all admitted recordings, parent references, duplicate families and windows involving an identity/component within one split.
- Resolve aliases/conflicting IDs using source evidence. Namespace IDs by actual origin and language; do not conflate unrelated numeric IDs across datasets.
- Conversion-pair chains through UNUSED candidate examples are not exclusion edges under this named protocol. Do not present this as strict conversion-family isolation. Strict isolation remains unmet and cross-language person independence is unproven.

**Absolutely exclude the evaluated test identities and related recordings from all new training and development additions.** Consume `artifacts/evaluations/gru-v2-test-epoch5/exposure.json`, the complete v2 test manifest, and their source/target identities—not just the 24 male genuine files. Exclude hashes, verified parent identities, aliases and detected duplicate families. Do not move benchmark examples into training or reserve another recording from the same benchmark speaker as “new.” Do not select additions based on individual model scores or which test clips failed.

Preserve old train/dev assignments. Prefer entirely new identities for additions. A prior-training identity must never enter development; a prior-development identity must never enter training. Verify all existing source/target roles, not just genuine filenames. Protect upstream validation/test speakers and recordings from training. Unknown required lineage or conflicting identity evidence means quarantine, not inferred independence.

Do not create or score another test set in this task. Keep the existing 96-window benchmark frozen. Document that a separately protected future test will be needed for an unbiased assessment of changes now informed by this benchmark.

## Budget and networking

Use the SAME cumulative ledger: `artifacts/datasets/manifests/downloads.json`. Last recorded total: 4,377,793,162 bytes, leaving 990,915,958 of the 5,368,709,120-byte cap. Re-read it; never reset it, increase the cap, or create a second allowance.

The ledger includes an explicitly estimated historical reconciliation. New Kathbath audio reads should use the recently hardened serial, no-read-ahead, pre-charged bounded reader. Charges are conservative requested payload bytes and remain reserved on failed/short reads. Run only one acquisition job at a time. Other metadata discovery/download adapters have NOT all been hardened; do not assume they safely enforce the cap merely because they accept a ledger. Prefer existing metadata. Any necessary new read must be bounded and charged before transfer with failure accounting; do not add unaccounted parallel fetches.

Use cached row-group costs to plan efficiently, but estimates are not permission to overspend. Do not consume the entire remaining budget merely to force targets. If the proposed additions cannot fit, deliver the feasible bounded subset and precise shortfalls. Never log authentication tokens or signed download URLs.

## Files, materialization and preprocessing

Write new work under `artifacts/datasets-v3-coverage/`. Preserve v1/v2 manifests, audio, feature caches, checkpoints, predictions and evaluation evidence. The shared ledger is the documented mutable exception. New source code/tests/docs may be added or updated as necessary, without breaking historical reproduction commands.

Create explicit train/dev manifests containing the retained original examples plus eligible additions. Keep the historical 96-window benchmark separately referenced by its existing path and hash. Record path bases per row/source or define an unambiguous versioned resolver; don't blindly copy relative paths into a new root where they no longer resolve.

Assign identities/components to splits BEFORE materializing windows. Reuse exact `voxsentinel-prep-2`: mono channel mean, soxr HQ to 16 kHz, overflow attenuation only, existing class-independent 1–4 second high-energy-window/activity rules. No normalization changes, augmentation, denoising, model-based filtering, feature extraction or fitted statistics. Keep original/decoded/prepared hashes and complete acquisition provenance, including pinned revision, shard, row group and row index.

Record every attempted addition as selected, reserved, quarantined or failed with a reason. Replacements must respect the existing split reservation; failure does not authorize moving a speaker to another split. Retain unused acquired files rather than deleting them.

## Validation required before handoff

Validate the emitted manifests independently of the selector:

1. All prior training/development rows retained with unchanged labels and provenance; any schema/path translation is explicit and auditable.
2. All-pair cross-role identity/reference checks across new train, new dev and the fixed evaluated benchmark. Benchmark exposure exclusion must be enforced, not merely reported.
3. Original/prepared hash and near-duplicate checks against existing data and the benchmark using the documented audio-only method. Report method limitations; do not claim universal transcode detection.
4. Real file existence, decode, mono 16 kHz, finite samples, duration/window bounds, activity eligibility, hashes and correct labels.
5. Genuine gender comes from documented source metadata. Unknown/conflicting gender is recorded, not guessed.
6. Deterministic offline replay from frozen metadata produces identical split assignments and manifest hashes.
7. Budget reconciliation and before/after hashes demonstrate original datasets/models/evaluation artifacts were preserved. Document any conservative reservation or estimated charge rather than labeling it exact transfer measurement.

Add focused fixtures for a benchmark male speaker attempted in training, a benchmark SYNTHETIC source/target speaker attempted as new genuine speech, old-dev identity attempted in training, renamed/re-encoded duplicate, insufficient-speaker shortfall, wrong path base, and failed-read budget retention. Reuse existing tested helpers where appropriate; test actual failure modes, not just expected fixture counts. No model inference is needed.

## Deliverables and stopping condition

Deliver:

- Metadata feasibility/assignment plan and pinned candidate snapshot.
- New materialized additions and combined train/dev manifests under the new root.
- Dataset-version metadata with source revisions, manifest/config hashes, explicit path resolution and frozen benchmark reference.
- Exposure-exclusion and component evidence; full disposition/shortfall accounting.
- Machine-readable validation and budget/preservation reports.
- `docs/archive/history/GENUINE_COVERAGE_ACQUISITION_HANDOFF.md` with actual per-language/class/gender window counts, independent speaker counts, durations, recording conditions, additions, retained data, limitations, and exact commands.

Separate statuses for split integrity, achieved coverage and unresolved shortages. Partial coverage can be a valid provisional dataset if its limitations are explicit; integrity failures cannot be promoted as ready. Do not claim the model improved: no model has been trained on these additions.

Tell the next training agent that existing feature caches are now stale for the expanded manifests. `--split-version` in the current trainer intentionally only accepts byte-identical old train/dev copies. The next agent must implement/audit new cache generation and dataset-aware training/evaluation before training; do not bypass that guard or reuse old caches with new labels/IDs.

**STOP after acquisition, manifest addition, audio preparation and data-integrity verification. No encoder features, standardizer fitting, training, prediction, test scoring, threshold tuning, model packaging, commit, push or merge.** The user will have the primary agent handle training, analysis and branch handoff afterward.
