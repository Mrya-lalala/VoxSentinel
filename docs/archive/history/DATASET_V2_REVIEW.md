> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Independent review of the v2 split handoff

The path supplied for review, `BASELINE_DATASET_AND_PREDICTION_REPORT.md`, is the earlier baseline report. The new outputs are `DATASET_V2_SPLIT_HANDOFF.md` and `artifacts/datasets-v2/reports/report.md`.

## Verdict

Useful preparation work, but the unconditional integrity/untouched-test pass is not supported by the implemented checks. Treat the new test as provisional pending the issues below. No dataset, checkpoint or original audit evidence was modified during this review; no model was evaluated.

## Confirmed progress

- 384 training and 96 development windows retained; 96 new test windows materialized across 12 languages.
- New test genuine coverage: 24 male / 24 female. This resolves the question of whether male genuine recordings are available in the scanned pinned Kathbath source. Training itself remains unchanged and all genuine training examples are female.
- Reran the two existing fixture tests: both passed. Reran the supplied full validator, writing separate output to `/private/tmp/vox-v2-review-audit.json`: it reported no problems. Passing its current checks does not resolve gaps in what those checks enforce.

## Findings requiring correction

1. **Exposure closure is implemented as direct membership, not transitive closure over the known candidate graph.** `split_v2._plan` filters against registry speaker/reference sets before selecting rows, and `_component_summary` describes only selected rows. `validate_dataset_v2.audit_manifests` checks the same direct exposure membership. An independent union-find check joining parsed source/target speaker references in all 24 old and fresh per-language IndicSynth candidate files found 67 of the 96 test windows connected to a v1-exposed speaker component. Using only the fresh candidate files found zero; this illustrates why excluding the old candidate relationships matters. This is a candidate-metadata finding: verify admissible edge provenance before claiming confirmed leakage or excluding components. It does establish that the required broader closure check was not performed. It does not imply direct same-speaker overlap between emitted manifests or observed model contamination. Examples needing review include `kathbath-bengali-844424931060785-278-m`, `indicsynth-bengali-022696` and `indicsynth-gujarati-030701`.

2. **Required source-parent verification is allowed to remain unresolved.** `indicsynth-bengali-012765` and `indicsynth-sanskrit-102314` have `source_parent_verification=parsed_only`. The validator parses reference names and compares declared speaker IDs but does not require verified source/target parent evidence. Its unknown-metadata summary counts target parsed-only status but omits source parsed-only status. Verify the actual parent identities or quarantine those recordings; report both roles. For verified entries, preserve the actual lookup evidence rather than repeating the candidate reference as evidence of a successful scan.

3. **The download ledger is incomplete and its cap is enforced after transfer.** `fetch_kathbath_audio_pinned` fetches in parallel, then skips an already-read shard if it exceeds a language cap, without charging those bytes. The handoff acknowledges approximately 30 MB of such uncharged transfer. The displayed remaining allowance is therefore overstated. Account for skipped/failed transfers and enforce the cumulative cap before or during reads, not only when retaining payloads. This review did not determine the exact reconciliation amount or establish that the global cap was exceeded.

4. **The handoff overstates the adversarial test coverage.** It claims 11 cases including a transitive A→B→C bridge, actual re-encoding and quota-overflow components. The implemented 11 cases do not include the bridge or quota-overflow test. The duplicate fixture assigns identical prepared hashes; it does not exercise re-encoding and the audio fingerprint detector. The second pytest test checks the fixture count, not an independent behavior. Add the missing behavioral cases and update the report to describe only checks actually run.

## Next action

Repair closure construction/validation, resolve required parent provenance, reconcile transfers and strengthen the tests. Re-select only affected test material where necessary, preserving acquired audio and frozen train/development data. Re-run the data-only audit and determinism checks; update the readiness claim from that evidence. Keep all detector training/evaluation out of this repair task. Cross-language identity and limited near-duplicate-method coverage remain explicit limitations.

## Second review — updated handoff

Rechecked the updated implementation and artifacts. Four pytest tests pass; the full validator reports zero problems under its revised relationship policy (separate review output: `/private/tmp/vox-v2-updated-review-audit.json`). Independently matched all 73 applicable synthetic test parent references, including speaker/gender identity, against the cached Kathbath inventories: no mismatch. Frozen snapshot hashes remain unchanged except the permitted ledger; train/dev manifests are exact v1 copies. Both previously flagged parsed-only windows were replaced. These are real improvements.

Two findings remain open:

- **Original closure contract not met.** The implementation now treats source/target co-participation links in unused candidate conversions as diagnostic rather than excluding their connected components. The strict diagnostic still flags 67/96 windows. Such a chain does not by itself demonstrate actual shared-speaker or recording leakage, so it is reasonable to distinguish it from direct contamination; however, this is a changed experimental policy, not successful enforcement of the original prompt. Infeasibility of the original policy is a reason to report a blocked strict split or propose a revised protocol, not evidence that every excluded edge is scientifically irrelevant. The current split can be described as a candidate for a narrower identity/recording-disjoint protocol, pending explicit adoption of that protocol. The unconditional “all four findings fixed” wording is inaccurate.
- **Global budget protection remains incomplete.** The fetcher checks each estimated shard against the same unreserved ledger remainder and then submits all reads concurrently. Several individually affordable reads can jointly exceed the remainder; absent/underestimated costs also bypass protection. Charges still occur after reads. On `BudgetExceeded`, it records `charged_bytes: 0` despite completed transfer, and fetch exceptions can lose partial-transfer accounting. The historical reconciliation is explicitly estimated, not exact. Implement cumulative reservations or bounded sequential reads with pre/during-read enforcement, and account for failures. No new download was performed in this review and no actual cap breach is claimed.

Minor reporting correction: the updated validator reports 387,827 eligible-unused candidates; the handoff still says 389,829. Update from the final evidence.

No training, inference, feature extraction, acquisition, split mutation or publication was performed by this review.

## Targeted completion

Following the user's request to resolve the remaining issues, retained the data under the explicitly named `speaker_recording_disjoint_v2` protocol. Audit, generated report and dataset-version metadata now distinguish its passing integrity checks from `strict_conversion_family_compliant: false`; the original stricter rule remains unmet rather than being relabeled a pass. No split memberships changed.

Replaced parallel post-charged Kathbath audio reads with serial uncached reads, each conservatively charged before I/O. A shared per-language allowance bounds cumulative reads; failed/short reads retain charges. This targeted path requires a single acquisition job; other discovery/download adapters were not changed, and historical wire-byte accounting remains approximate. Eight focused tests and a local PyArrow integration check passed. Regenerated the data-only audit/report successfully; corrected the stale candidate count. No network acquisition, feature extraction, training or model inference was performed.
