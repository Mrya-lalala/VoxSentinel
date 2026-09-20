# Independent review: v3 cache and training support

Verdict: **accept the current feature caches; fix the training-runner integration/evidence gaps before starting the controlled run.** No model training or benchmark evaluation was performed in this review.

## Verified

- All 22 targeted v3 dataset-version, cache and training-support tests pass.
- Re-ran the actual training check-only gate: cache ready, zero failures; evidence saved separately to `/private/tmp/vox-v3-training-review.json`.
- Independently loaded complete bundles: train 458 items / 91,117 valid frames; dev 135 items / 26,853 valid frames.
- All 384 reused training and 96 reused development tensors are exactly equal to their historical cached tensors. All 593 feature tensors are ordinary tensors, not inference-mode tensors.
- The real loader and runner use v3 examples for training and development. No old-root recovery evaluator is called.
- Recorded encoder-parity evidence reports ten samples, tolerance 1e-5, maximum retained-item error about 5.25e-6 and fresh-item error zero. This review inspected that evidence rather than regenerating encoder outputs.

## Fix before training

1. **Checkpoint metadata is incompatible with the existing predictor.** The v3 runner copies the full cache `encoder` identity into checkpoint metadata. That structure contains `output_layer` plus cache-specific fields. `FilePredictor` requires an exact five-field identity with `selected_layer`, checkpoint hash, dimension, preprocessing version and dtype. The dicts do not match, so an otherwise correct v3 checkpoint would be rejected. Normalize runtime checkpoint identity to the established format, while retaining the full cache identity separately. Add a small compatibility test without scoring benchmark audio. The older packaging helper also assumes the historical dataset-identity layout; provide explicit v3 packaging support before distribution rather than silently reusing that helper.

2. **Training evidence is incomplete.** The runner imports but never writes runtime/environment evidence, does not archive source state, and does not export the computed per-example development logits/scores. It writes only aggregate and language/generator metrics. Save source/config/runtime provenance before fitting, an actual start time and completion/failure status, and per-example predictions with IDs and original manifest metadata. Include genuine/target gender and retained/addition subgroup counts for the intended coverage analysis. Current `started` is recorded after training finishes and is misleading. A failure currently leaves no explicit failed-run record.

3. **Reload verification checks two restores, not the trained model.** Loading the same checkpoint twice and comparing outputs can pass even if the wrong epoch/state was saved. Capture read-only v3 development logits per epoch (or at best-selection time), preserve RNG state around the callback, and compare the restored selected checkpoint with that epoch's live predictions. Verify final-live versus final-reloaded state as well. Existing shared components support this pattern. Keep checkpoint selection on v3 development EER, earliest tie, threshold 0.5.

## Minor reporting/guard issues

- The handoff's “no forward passes” verification block includes `parity`, which explicitly runs the encoder. Separate no-forward check-only commands from encoder-parity verification.
- `raw_merged_config` uses shallow dictionary updates, so it is not the actual nested merged configuration; use the same deep merge as the config loader or report resolved settings only.
- The initial-comparison check omits architecture, shuffle, AMP and execution constraints and only warns on mismatches. For the planned controlled run, validate the complete intended configuration and reject unintended changes. This is a guard improvement; the current default configuration itself matches the intended experiment.

No cache rebuild is indicated by these findings. Preserve the delivered bundles and historical data. Fix and test the runner first, then run one predeclared v3 training experiment. The already-evaluated benchmark remains outside selection and tuning.
