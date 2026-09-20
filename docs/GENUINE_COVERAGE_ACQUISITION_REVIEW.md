# Independent review of genuine-coverage additions

Verdict: **Accept the materialized v3 dataset for feature-cache preparation**, under the documented `speaker_recording_disjoint_v2` protocol and coverage limitations. No blocking defect was found in the delivered rows in this review. This is not a model-performance or production approval.

Verified locally:

- 458 training windows: 266 genuine / 192 synthetic, including 74 new male genuine windows from 37 language-scoped speakers.
- 135 development windows: 87 genuine / 48 synthetic, including 39 new male genuine windows from 20 language-scoped speakers.
- Every original train/dev row retained exactly; unique emitted IDs.
- Independently reconstructed identity sets across genuine/source/target roles: zero train/dev overlap and zero overlap with the evaluated benchmark.
- Independently decoded all 113 raw additions, repeated the exact prep-2 window selection and compared waveform samples against the stored prepared files: all 113 matched exactly, with valid activity eligibility and matching offsets.
- Full delivered validator rerun: ready, zero issues, retention and replay match, no audio/near-duplicate problems. Separate evidence: `/private/tmp/vox-v3-independent-review.json`.
- All 12 dataset-preparation pytest tests passed. Validator also confirmed the frozen snapshot and no-charge deterministic replay.

Important limitations:

- Malayalam has no additions; Odia has one development example and no new training coverage; Bengali has only one new speaker per split. Nine of twelve languages reached both targets.
- Class balance changed. Training is about 58% genuine; development is about 64% genuine. The next experiment should preserve these examples, report balanced accuracy/FAR/miss and subgroup counts, and avoid treating raw accuracy as directly comparable across differently composed development sets. Class weighting is a separate modeling choice, not an acquisition failure or automatic requirement.
- The test benchmark was previously evaluated and influenced this acquisition objective. It remains unchanged and excluded from training, but is not an untouched future test for adaptive improvements. Any repeat benchmark result must be labeled accordingly.
- The failed first acquisition attempt consumed 321,507,083 accounted bytes; final remaining budget is 342,905,206 bytes. No data loss is apparent in the completed delivery. The final handoff discloses the failed attempt rather than hiding its cost. This review did not measure historical wire transfers independently.
- Near-duplicate and cross-language identity limits remain as documented. Strict conversion-family isolation is not claimed.

Corrected minor handoff wording: 9 of 12 languages (not 11), full snapshot-to-completion budget delta of 648,010,752 bytes including pre-plan probes, and “unchanged, already-evaluated benchmark” instead of ambiguous “untouched.”

Next: implement dataset-aware path resolution and cache identity for v3, generate/validate features for the expanded manifests, then fit the standardizer on training frames only and train a controlled comparison. Do not bypass the existing v2 cache guard or treat the old complete-cache bundles as covering new rows. Existing per-example features can only be reused with verified audio/encoder/preprocessing identity and correct new manifests.

This review performed no acquisition, feature extraction, model training or model inference. It did not change any split membership, audio, checkpoint or ledger.
