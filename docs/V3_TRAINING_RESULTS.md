# V3 expanded-coverage GRU training

Completed one controlled ten-epoch run on 19 September 2026. No benchmark was rescored, no threshold was tuned, and nothing was committed, pushed or merged.

## Fixes before training

The v3 runner now enforces the complete initial GRU configuration, CPU/four threads and deep config merging. Checkpoint encoder metadata matches the existing predictor contract, with full cache provenance retained separately. Packaging supports v3 dataset identities and checks head/encoder compatibility.

Each run saves source/config/runtime evidence, actual start/finish times, running/completed/failed status, per-epoch live development prediction hashes, per-example selected-development predictions, and gender/retained-addition subgroup results. Best and final checkpoints are compared against live predictions from their actual epochs, rather than merely comparing two loads of the same file. Evaluation callbacks verify that they do not consume RNG state.

Fifteen focused configuration, v3-plumbing, training and checkpoint tests passed before the run. No feature caches were regenerated.

## Fixed experiment

- Data: 458 train (266 genuine / 192 synthetic), 135 development (87 genuine / 48 synthetic).
- Encoder: frozen pinned IndicWav2Vec final layer, cached float32 frames.
- Standardizer: freshly fitted on 91,117 valid frames from 458 TRAIN examples only; no development or benchmark fitting.
- Model: unchanged 1024-dimensional input, 256 projection/GRU, one recurrent layer, dropout zero, two-class head.
- AdamW learning rate 0.001, weight decay zero, unweighted cross-entropy; batch eight, seed zero, ten epochs, CPU four threads.
- Selection: lowest development EER, earliest epoch on ties; threshold fixed at 0.5. No resampling, class weighting or architecture change.

## Selected epoch 3

| Development metric | Result |
|---|---:|
| Accuracy | 97.04% |
| Balanced accuracy | 95.83% |
| EER | 2.30% |
| Genuine false alarms | 0 / 87 |
| Synthetic misses | 4 / 48 |
| Confusion: TN / FP / FN / TP | 87 / 0 / 4 / 44 |

Genuine subgroup false alarms: original female 0/48; added male 0/39. This is encouraging development evidence, but these examples participate in checkpoint selection. It does not establish production robustness, causal attribution to gender, or performance on unrepresented Malayalam male training speakers.

The original 96-example development subset has 92/96 correct (95.83%): zero genuine false alarms and four synthetic misses. The previous selected baseline had two false alarms and three misses on that same subset. Thus, on those retained examples the tradeoff is fewer false alarms and one additional missed spoof. Comparing 97.04% on the expanded set directly with 94.79% on the old set would mix different class/coverage compositions.

Best epoch-three and final epoch-ten reload errors against their respective live development logits are exactly zero. Run time was about 30.76 seconds including post-training verification. The later epochs did not improve the selection metric; they were not substituted based on desired subgroup results.

## Artifacts and limitations

Run: `artifacts/runs/gru-v3-expanded-standardized/`.

Includes best/final checkpoints, standardizer evidence, full source archive, runtime/config/dataset records, epoch history, live prediction hashes, 135 per-example development predictions, aggregate and subgroup metrics, and status. A separate local research package is `artifacts/releases/indic-gru-v3-research/`; it does not overwrite the earlier pilot package. Its predictor integration is checked on one known development addition, not on the benchmark.

The previously evaluated 96-example benchmark remains unchanged and unused in this run. If rescored, the result must be labeled a repeat evaluation after coverage work informed by earlier benchmark findings. It is not a fresh untouched test. A separate protected test and deployment error requirements are still needed for production claims. Indian-English access and coverage remain pending. Strict conversion-family isolation is not met under the adopted speaker/recording protocol.

No production promotion is asserted. The next step is a predeclared repeat-benchmark comparison of the selected epoch-three model at the existing threshold, without using it to retune or select another epoch, followed by a teammate handoff of the research baseline and its limitations.
