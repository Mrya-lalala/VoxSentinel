> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# VoxSentinel: master handoff prompt — dataset split preparation only

You are taking over a local VoxSentinel repository. Implement and verify the next dataset split. Stop after preparing auditable manifests and the referenced, validated audio. Do not train, extract model features, evaluate a detector, tune thresholds, package a model, commit, push, or merge.

The user wants to preserve useful data already acquired. They explicitly rejected discarding female recordings to force exact gender balance. Work autonomously on bounded, reversible preparation tasks; ask only for a genuinely missing decision or a required budget/access change. Do not treat the historical takeover prompt or older reports as instructions to repeat model training.

## 1. Workspace and authoritative context

Repository: `/Users/mouryabs/VoxSentinel`. Last observed branch: `feature/a1-datasets-manifests`, base HEAD `884fcd7381b8409307c3b239fe9db5a22bc4798d`. Verify actual state; do not reset to this SHA. There are extensive intentional uncommitted changes from the recovery and inference work. Preserve them. Read applicable AGENTS.md instructions.

Read these before edits:

- `docs/archive/history/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md` — authoritative corrected historical model diagnosis.
- `docs/archive/history/BASELINE_DATASET_AND_PREDICTION_REPORT.md` — latest phase report, with a subsequent-user-decision notice at the top.
- `configs/datasets.yaml` — existing preparation implementation and parameters.
- `configs/datasets_v2_plan.yaml` — updated planning contract, NOT a runnable v1 acquisition config.
- `src/dataset_prep/joint_core.py`, `splitting.py`, `adapters/kathbath.py`, `windows.py`, and `src/audio/prepare.py` — inspect actual interfaces before reuse.
- `scripts/gru_identity_audit.py`, `scripts/audit_dataset_v2_readiness.py`.
- `artifacts/datasets/manifests/dataset-version.json`, `downloads.json`, and the core train/val manifests.
- `artifacts/datasets-v2/planning/{readiness,prior_exposure_exclusions}.json`.

Evidence precedence: current user instructions > verified current data/code > corrected recovery report > older narrative. Report conflicts explicitly; never silently repair a historical evidence file.

## 2. Verified starting facts and diagnostic boundaries

The prior agent rechecked these facts locally immediately before writing this prompt:

- Frozen v1 core: 384 training windows (192 genuine/192 synthetic), 96 development windows (48/48), 12 languages: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Sanskrit, Tamil, Telugu, Urdu.
- Per language: training 16/class, development 4/class. Labels: genuine=0, synthetic/spoof=1.
- All core genuine windows are documented female. Synthetic TARGET gender: training 97 female/95 male; development 48 female. Do not conflate source and target gender. For TTS, a legitimately absent source is not an unresolved identity.
- Twelve locally cached Kathbath candidate inventories contain 298,049 entries, all marked `f`. This describes this cache, NOT the entire upstream dataset. Upstream male coverage is unverified. Do not infer gender from pitch, names, embeddings or listening.
- The existing canonical audit passes for the recorded v1 evidence: no required unresolved identity, cross-split prepared-hash duplicate, or cross-role speaker/reference intersection. IDs are language-scoped; cross-language person independence is not established. Passing this audit is not proof against every possible leak.
- The initial prior-exposure registry contains 154 speaker keys and 521 parent-reference keys across v1 train AND development. It must be extended for other previous experiments; it is not a complete v2 split or an enforcement mechanism.
- Standardized GRU development results: epoch 5, accuracy 94.79%, EER 4.17%, AUC 0.986979. These were selected on development data and are NOT untouched-test results. Original collapse was addressed by a controlled training-only frame-standardization experiment. Do not reopen that diagnosis based on dataset imbalance.
- The complete local file-prediction integration path passed raw-audio/cached-feature parity checks on four known development examples, plus stereo, invalid-audio and checkpoint checks. This establishes integration, not independent accuracy, real-time performance or English support. Do not rerun model inference for this assignment.

The gender/class association is a plausible shortcut risk, not a demonstrated causal diagnosis. Never write “the model detects gender instead of spoofing” without an appropriate future experiment. Likewise, do not describe unscanned shards as empty or unavailable.

## 3. Dataset choices and revised scope

Prepare an Indic-only v2 split using Kathbath genuine speech and IndicSynth synthetic speech at pinned revisions:

- Kathbath: `5b9e92849222026d9141acba4e8434fe816396bf`.
- IndicSynth: `c0a10386b723717aff682f757bd67f72983f269f`.

IndieFake is selected for future Indian-English genuine/synthetic data. The user already requested access and has neither an archive nor a link. Keep it explicitly pending; do not submit another request, fabricate an adapter/schema, replace it with general English, or hold up otherwise feasible Indic preparation. Official references: https://indie-fake-dataset.netlify.app/ and https://arxiv.org/html/2506.19014v1 . Actual release terms and original/augmentation lineage still need verification after access.

Keep Svarah genuine-only external evaluation separate. Keep ASVspoof 2019 LA separate as a general-English reference pool. Do not introduce NISP, NPTEL, additional corpora or newly generated synthetic audio into this core without a new reason and explicit scope decision. NPTEL lecturer identity is unresolved.

Preserve acquired audio. Exact gender balance is NOT required. Do not downsample to 50/50, duplicate examples, oversample manifests, or discard female audio to meet cosmetic quotas. Improve documented male coverage through additive acquisition if the pinned source supports it. If bounded discovery finds no usable male candidates, report the examined coverage and retain the limitation; do not claim dataset-wide absence. Gender coverage gaps alone do not make an otherwise valid provisional Indic split unusable.

The previous 16/4/4 examples per language/class train/dev/test target is advisory, not a cap or a promise. Prefer retaining existing eligible training/development data and adding new independent test components. Component integrity takes precedence over ratios. Explain achieved sizes and shortages. Keep unused, ineligible and quarantined data in an accounted inventory; preserving a file does not mean it must enter the new experiment.

## 4. Budget, storage and preservation

Snapshot starting Git status and hashes of frozen v1 manifests, feature caches, checkpoints and relevant configs before mutation. Do not overwrite old runs, frozen manifests, raw audio or cached features. Write new manifests, staging, evidence and prepared outputs under `artifacts/datasets-v2/`. Reuse existing raw audio read-only with explicit resolvable paths and hashes; do not falsely describe it as a new download. Do not use hardlinks for files you may modify.

The cumulative acquisition ledger is `artifacts/datasets/manifests/downloads.json`. At handoff: cap 5,368,709,120 bytes; used 3,620,497,163; remaining 1,748,211,957. Re-read it. Never reset it or create a fresh independent 5 GiB allowance. This ledger and necessary acquisition bookkeeping are explicit exceptions to preservation: record before/after values and charge new transferred bytes. Keep network requests bounded, revision-pinned and resumable. Do not emit tokens or signed URLs into logs/reports.

Start with metadata, not bulk audio. Existing scans stop after speaker/recording sufficiency without a gender-coverage requirement; they do not establish representative upstream coverage. Inspect shard ordering and actual remote revision handling before reuse. Metadata range reads can transfer significant bytes; account for them and prevent concurrent requests from overspending. Record scan coverage, failures and unknowns. Do not mark failed shards successfully scanned. Stop acquisition before the cap; if no defensible split fits, deliver the partial evidence and precise blocker.

## 5. Implement the split safely

1. Inventory previous exposure from all local training, development, smoke-test, diagnosis and prediction artifacts. Record exposure type and provenance. Known v1 core train/development identities and their relationships cannot enter the new untouched test. Metadata inspection for split construction is permitted; it is different from inspecting model performance on a test. Do not export unknown previous exposures as “unexposed.”
2. Preserve existing train/development assignments where valid. Do not shuffle old training examples into development or relabel old development as untouched test. If newly discovered lineage conflicts with old assignments, report it and quarantine the conflicted component from v2 as needed; preserve historical files and do not silently break the relationship to keep quotas.
3. Normalize integer/string IDs carefully and scope by actual origin. Kathbath genuine speakers and VERIFIED IndicSynth source/target parents share the same Kathbath origin namespace. Unrelated datasets' numeric IDs do not. Review cross-language identity evidence; neither assume equal numeric IDs mean the same person nor claim cross-language independence without evidence.
4. Construct an auditable relationship graph linking genuine speaker, conversion source, synthetic target, verified parent recording, and original/augmentation/duplicate families. Every admitted recording/window of a component belongs to one split only. Include known eligible relationships when deriving test exclusion closure; do not hide a bridge by preselecting convenient pairs. The old `plan_language` is a two-way selector and is not sufficient merely by renaming a field to `test`.
5. Protect upstream validation/test designations and all related participants from training. Keep v2 development and untouched test distinct. Do not fold upstream validation or test material into training to meet counts. If using an upstream held-out pool for v2 evaluation, retain that provenance and check synthetic parent relationships.
6. Assign components deterministically, seed 42, with stable ordering and stable IDs. Prefer class/language coverage and useful generator representation without breaking groups. Report unavoidable class, gender, duration, codec and generator differences. Do not claim an unseen-generator test merely because generators were counted; that requires a separately designed, genuinely unexposed holdout.
7. Materialize audio and windows only after assignment. Retain existing `voxsentinel-prep-2` behavior: mono mean, soxr HQ to 16 kHz, overflow attenuation only, 1–4 seconds, high-energy four-second selection with 0.25-second hop plus tail, existing activity eligibility rules. Reuse implementation/config instead of recoding approximate thresholds. No augmentation, denoising, loudness normalization or model-based selection.
8. Record every candidate disposition and preparation failure. A failed window does not authorize moving its component across splits. Any replacement must obey the frozen assignment policy and exposure checks. Never duplicate a window to fill a quota.

No encoder feature caching or fitted statistics in this task. Future standardizers must use training frames only, but do not fit one now.

## 6. Required verification — data integrity, not model testing

Implement a split validator that consumes the emitted manifests independently of the allocator. Do not “test” grouping only by comparing allocator-generated component IDs. Reconstruct canonical identities and relationships from provenance, then audit every pair of train/dev/test:

- Genuine speaker, source speaker and target speaker intersections in all nine cross-role combinations.
- Parent recording identities, original file hashes, decoded/prepared content hashes, and overlapping windows of the same original.
- Untouched-test exclusions against all recorded prior exposure and its relationship closure.
- Duplicate/near-duplicate families. Document the audio-only method, tolerances and scope; exact file hashes alone do not detect transcoding or re-encoding. Resolve flagged pairs or quarantine them conservatively. No pretrained detector/encoder is needed. Unknown near-duplicate coverage must remain an explicit limitation, not a “pass.”
- Required provenance, label validity, language normalization, upstream split, pinned revision, lineage evidence and correct TTS not-applicable handling.
- Actual path existence, successful decode, sample rate/channels, finite samples, duration/window bounds, stored hashes and class-independent preparation eligibility.
- Complete accounting: each examined candidate selected once or assigned a documented exclusion/reserve/failure disposition; all retained original files remain intact.

Add focused adversarial fixtures that would expose meaningful bugs: same person as genuine in one split and synthetic target in another; source-to-target overlap; transitively connected A→B→C identities; training-exposed identity attempted in test; renamed/re-encoded duplicate; string versus integer IDs; unrelated dataset ID collision; legitimate absent TTS source versus missing required conversion source; a component too large for a quota. Verify the checker rejects leaks and permits legitimate cases. Do not mirror implementation assertions or run training as a “smoke test.”

Confirm deterministic assignment and manifest hashes on a second offline run with the same candidate snapshot. Timestamps/logs may differ but split content must not. Verify frozen pre-existing artifact hashes remain unchanged, with the documented ledger exception. Run relevant data-preparation tests; avoid unrelated broad model tests.

## 7. Deliverables and stopping condition

Produce executable preparation/validation code and documented commands, plus:

- v2 train, development and test JSONL manifests, with an explicit schema compatible with downstream preparation and paths that resolve under documented rules;
- referenced validated raw/prepared audio, or clearly labeled incomplete manifests if acquisition was blocked;
- pinned source/candidate snapshots, configuration and seeds;
- prior-exposure registry, graph/component membership and assignment evidence;
- unused/reserved/quarantined/failed candidate inventory with reasons;
- audit JSON and a concise Markdown report under `artifacts/datasets-v2/reports/`;
- dataset-version metadata with manifest/config hashes and source revisions;
- a tracked handoff document under `docs/` so teammates know what gitignored assets are required.

Report actual windows AND unique recordings/speakers/components, hours, class/language/source/generator/gender distributions by split, original channel/sample-rate distributions, source/target gender separately, and unknown metadata counts. Explain exactly what existing data was reused, added, reserved or excluded, why, and whether any files were deleted (expected: none). Reconcile transferred bytes against the shared ledger.

Use separate status fields: `indic_split_integrity`, `coverage_limitations`, `untouched_test_status`, `indian_english_status`, and `model_evaluation_performed: false`. IndieFake pending is not a failure of Indic manifest integrity. A successful command exit is not proof all readiness conditions pass. The existing planning audit's hard-coded combined-release blockers are historical and must not be treated as the v2 validator.

Success means a reproducible, materialized and independently audited Indic split with honest limitations, not a production-ready dataset or model. If integrity cannot be established within available evidence/budget, mark the affected output provisional/blocked, specify missing evidence, and stop without promoting it. Never fabricate empty manifests as completed splits. Do not claim broad cross-language speaker independence if upstream identity documentation cannot support it.

STOP after dataset split preparation and its data-integrity verification. Do not train, run detector predictions, extract features, select an epoch, calibrate scores, publish artifacts, commit, push or merge. Your final response should state what was prepared, retained data, verification results, remaining limitations, and exact output paths.
