> Archived historical document. Instructions and status may be superseded. Start with the [current documentation](../../README.md).

# Fixed v2 test evaluation

The executable pre-inference plan is `artifacts/evaluations/gru-v2-test-epoch5/plan.json`. It was written before constructing the predictor or generating any test scores; it pins the checkpoint, manifest and data audit hashes. The source snapshot is stored with the evaluation.

Evaluate the epoch-five checkpoint from `gru-v2-split-reproduction` once at the existing 0.5 synthetic-score threshold. Use all 96 prepared test windows through the frozen encoder, restored training-only standardizer and GRU. Do not refit, choose among checkpoints, calibrate or change preprocessing using test outcomes. Any input failure must be reported rather than silently excluded.

Report confusion counts, accuracy, genuine false-alarm and spoof miss rates, probability/margin EER and margin AUC. Report language, generator, genuine gender and synthetic target-gender subsets; undefined single-class metrics remain null. Use 2,000 seeded component-bootstrap replicates for exploratory overall accuracy/FAR/miss intervals. Components follow the admitted speaker/recording protocol, not chains of unused candidate conversions.

The test is small (eight windows per language) and does not establish cross-language person independence or strict conversion-family separation. Its genuine male examples have no counterpart in the model's training coverage. Indian-English and whole-recording/streaming performance are outside scope.

Once scored, this set is an evaluated benchmark; it must not be described as untouched if its results subsequently influence training or model choices. Any adaptive improvement needs development data and a separate future protected test.
