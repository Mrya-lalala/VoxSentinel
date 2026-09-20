# Astra GRU recovery manager report

Prepared 2026-09-18. Local repository `/Users/mouryabs/VoxSentinel`.

## Verdict

The single controlled intervention succeeded on this frozen pilot: training-fitted **frame standardization before the existing LayerNorm** enabled the GRU to fit training data and improved development performance. Selected epoch **5**: development accuracy **94.79%**, EER **4.17%**, AUC **0.986979**, CE **0.201876**. The original selected GRU had 50% accuracy, 22.92% EER, AUC 0.824653, and CE 0.703088. The selected standardized model makes two genuine false alarms and three synthetic misses at the unchanged threshold 0.5.

The linear score-export defect and the incomplete identity audit were repaired. The linear weights/scaler were reloaded, never refitted. The canonical cross-class audit passed. Original run artifacts, diagnostics, manifests, and raw caches remain unchanged (85 files verified by SHA-256).

This supports this intervention under seed 0 and the specified configuration. It does **not** establish input conditioning as the unique root cause, independent generalization, deployment readiness, calibrated confidence, Indian-English performance, or a reliable advantage over the linear reference. This 96-window set is development data and has informed checkpoint selection and subsequent decisions. No new dataset, encoder extraction/fine-tuning, sweep, threshold calibration, full test suite, commit, merge, or push was performed.

## 1. Scope and identities

Startup branch `feature/a1-datasets-manifests`, HEAD `884fcd7381b8409307c3b239fe9db5a22bc4798d`, clean working tree. It exactly matched the handoff's reviewed commit. The earlier report in the conversation inspected an older local state; its single-class-cache conclusion does not describe this frozen pilot.

The original run recorded `feature/indicVac2Wav` / `c250bbd0ac3b2af1ae345ce48960fcc63d200557` plus uncommitted code. Comparing all 39 recorded original-run file hashes to the reviewed commit found three differences: `.gitignore`, `README.md`, and `docs/dataset-preparation-pilot.md`. None changes model training. This corrects the prior diagnostic report's count of two differences. All actual repairs are local and uncommitted; unrelated A1 placeholders, audio, routing, streaming and services remain untouched.

Environment: Python 3.10.18, PyTorch 2.2.2, torchaudio 2.2.2, NumPy 1.23.5, Fairseq 0.12.1; macOS-26.5.2-arm64-arm-64bit; CPU FP32, four torch threads. Existing environment reused without installation. Encoder checkpoint identity: `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`; final `output_layer=None`, 1024 dimensions, 20 ms hop. `None` is not block 23. No encoder was loaded during recovery.

Frozen audio policy remains `voxsentinel-prep-2`: mono averaging, one SoXR HQ resampling pass when needed, 16 kHz FLOAT WAV, strict 1–4 second windows, recorded overflow attenuation, no denoising/loudness normalization/augmentation. The encoder's waveform normalization is unchanged and separate from downstream feature standardization.

| Split | Manifest SHA-256 | Cache bundle SHA-256 |
| --- | --- | --- |
| train | 1b361e14cb60cb739b6bbd55692529b94e8a7b37efc8530f4b03cb2ed2253ad4 | a893b5038cfc2375e027d7f2bc335aa48dd1f215af488c297eada13a4ea1b56b |
| val | a23acb1cdbe86a940c984bd9fb206abbd3dd6612733fa88b4de755b980e13342 | 069b3b78e2c4f767146966a1a4f2c1596b569eaa26663aecd6835df35ad03360 |

The readiness audit checked exact original manifest/bundle hashes; manifest/cache membership, order identity through unchanged bundle hashes, labels, language, waveform hashes, tensor dtype/dimension/length/finiteness/detachment; sidecar identities and both classes in every language; and all 480 actual prepared-file hashes. Encoder identity is compared between config, cache and original run metadata, not re-derived by fresh extraction. Evidence: `readiness.json`, `dataset_identity.json`, and the corrected canonical audit.

## 2. Findings ledger

| Finding | Status and exact location | Evidence and impact |
| --- | --- | --- |
| Wrong linear exported probability | Confirmed/fixed: `scripts/linear_baseline_gru_cache.py::evaluate`, `reevaluate_saved` | `sigmoid(m)` was incompatible with logits `[-m,m]`; now exports class-axis softmax and actual logit difference `2m`. CE/weights unchanged. |
| Kathbath speaker and cross-role lineage omitted | Confirmed/fixed: `scripts/diagnose_gru.py::command_identity`, new `scripts/gru_identity_audit.py::audit_identity` | Includes genuine `.speaker`, synthetic `.source/.target`, common verified Kathbath origin namespace, extension-normalized references, all nine role combinations, unresolved counts and scoped hash duplicates. |
| Gender coverage imbalance | Confirmed/documented; selection left frozen | All genuine and all development target/source identities with applicable metadata are female. See coverage and local inventory evidence below. This is a shortcut opportunity, not proof the model used it. |
| Report: zero linear scaler clamps | Confirmed report error | Original `stage5/summary.json` says eight. Correct value is eight for the pooled scaler, two for the new frame scaler. |
| Report: unscaled baseline predicts all genuine | Confirmed report error | Original JSON dev confusion is 0/48/0/48: all synthetic. |
| Nearest-centroid AUC 0.934 implies 7% nonseparable training pairs | Unsupported; corrected interpretation | One centroid score's ordering does not establish linear inseparability; fitted linear training AUC and accuracy are 1.0. |
| Standardization described as whitening | Incorrect terminology | Per-dimension scaling does not decorrelate features or whiten full covariance. |
| Unscaled LBFGS divergence proves recurrent AdamW root cause | Unsupported | LBFGS used `line_search_fn=None`; model, pooling, regularization and optimization differ. Scaling also changes the geometry of an L2 penalty. |
| Narrow score spread means no signal | Incorrect | Original best dev AUC 0.824653 reproduces; spread and ranking are distinct. |
| Root cause declared confirmed | Overstated | Original underfitting is observed; conditioning/optimization was a hypothesis. This matched intervention now supports a benefit from standardization without isolating a unique mechanism. |
| Shuffled labels prove absence of leakage | Incorrect | Controls do not replace speaker/reference provenance auditing. |
| Naive significance/SE at n=96 | Unsupported | Windows share speakers/lineage; no confidence interval or significance claim is made. |
| Supplemental Fisher padded projection mean | Confirmed in archived `stage3_fisher_probe.py:83` | `proj.mean(dim=1)` includes padding. Analysis not carried forward, no corrected rerun needed; archived script untouched. This is not a defect in head packing/masked pooling. |
| Fresh four-window extraction/depth-profile claims | Reported, not independently reproduced | `commands.txt` points to a transcript, but no standalone logs/script with those sample IDs were located. Earlier encoder-verification artifacts concern different checks. No expensive extraction was run to reconstruct missing evidence. |
| Cache readiness described as deep per-item checking | Existing gate narrower than its prose | Existing gate checks sidecar/bundle identities/counts; recovery adds explicit per-item and actual audio hash checks without changing raw data. |

The historical diagnostic report remains available with a superseding correction notice. Historical artifact copies were not rewritten to match the corrected conclusions.

## 3. Corrected linear baseline: no refit

The fitted model remains `artifacts/runs/gru-core-v1-diagnosis/stage5/model_scaled.pt`.

SHA-256: `32724f15239441fc631824da33968ee51b6aa79e6627e4bac07692e053b2a232`. Saved weights, bias, mean and scale were used unchanged; file hash before/after matches.

For raw response `m = X @ weight + bias`, logits are `[-m, m]`, logit margin is `2m`, and the correct score is `softmax(logits)[1] = sigmoid(2m)`. The old exporter used `sigmoid(m)`. The fitted objective already used the actual logits, so cross-entropy is unaffected.

Example `indicsynth-bengali-003638`: old score 0.997214139 → corrected 0.999992251, actual logit margin 11.760760307. Zero decisions changed at 0.5 in either split. Training CE 0.000072797, dev CE 0.224134922, dev confusion 47/1/6/42, accuracy 92.71%, EER 4.17%, and AUC 0.988281 remain unchanged.

Primary ranking metrics are tie-aware AUC and interpolated EER on actual logit margins. Probability-based metrics are saved separately. Corrected float32 softmax has 296 unique training probabilities versus 384 unique margins, and 94 versus 96 on development; saturation creates ties, but does not alter these aggregate AUC/EER values here. Neither scores nor margins are calibrated confidence. No operating threshold is selected.

The linear scaler used one time-mean vector per training window and sample standard deviation (`ddof=1`). Its eight clamped dimensions are not comparable to the two clamped dimensions of the new frame-weighted population scaler. Linear pooling, LBFGS and L2 regularization also differ from the GRU; this is a diagnostic reference, not a controlled architecture comparison.

## 4. Canonical split and coverage audit

The frozen pilot has 384 training windows (192 genuine, 192 synthetic) and 96 development windows (48/48). All upstream rows are recorded as training material. Detector partitions are audited separately from upstream split membership.

Canonical speaker key = `(Kathbath origin, normalized language, normalized numeric speaker ID)`. Genuine filenames and synthetic verified parent evidence put the two classes into that common namespace. Synthetic source and target references are compared to genuine `source_file` recording identities across all roles, ignoring container extensions. Unrelated datasets are not merged by coincident numeric IDs. Cross-language person identity remains unresolved.

Results: **117 train / 37 development unique speaker keys; zero speaker overlaps; zero reference-recording overlaps across every one of the nine role pairs; zero unresolved required identities; zero within-split or cross-split prepared-audio hash duplicates.** There are 114 expected absent TTS source roles (91 train, 23 development), preserved as null/not-applicable. Known within-split genuine/synthetic sharing is expected and recorded in JSON. The audit uses recorded evidence; it does not re-query upstream datasets or rule out unknown speakers/near-duplicates.

Generator counts below use the order **FreeVC24 / XTTS-v2 / VITS**:

| Language | Train genuine | Train synthetic F/X/V | Dev genuine | Dev synthetic F/X/V |
| --- | --- | --- | --- | --- |
| bengali | 16 | 6/7/3 | 4 | 2/0/2 |
| gujarati | 16 | 8/8/0 | 4 | 2/2/0 |
| hindi | 16 | 8/8/0 | 4 | 2/2/0 |
| kannada | 16 | 8/8/0 | 4 | 2/2/0 |
| malayalam | 16 | 8/8/0 | 4 | 2/2/0 |
| marathi | 16 | 16/0/0 | 4 | 4/0/0 |
| odia | 16 | 8/8/0 | 4 | 2/2/0 |
| punjabi | 16 | 8/8/0 | 4 | 2/2/0 |
| sanskrit | 16 | 7/9/0 | 4 | 1/3/0 |
| tamil | 16 | 8/8/0 | 4 | 2/2/0 |
| telugu | 16 | 8/8/0 | 4 | 2/2/0 |
| urdu | 16 | 8/8/0 | 4 | 2/2/0 |

Totals: synthetic 127 FreeVC24, 108 XTTS-v2, 5 VITS; development 25/21/2. Every language has 16 examples per class in training and four per class in development. Eight development windows per language and two development VITS examples do not support strong subgroup conclusions. Detailed language/class/generator and language/class/gender/role counts are saved in canonical audit JSON and coverage CSVs.

Gender values describe **recorded participant identities**, not inferred or perceived generated-voice gender:

| Split | Class | Role | Female | Male | Unknown/missing | Not applicable |
| --- | --- | --- | --- | --- | --- | --- |
| train | genuine | speaker | 192 | 0 | 0 | 0 |
| train | synthetic | target | 97 | 95 | 0 | 0 |
| train | synthetic | source | 60 | 41 | 0 | 91 |
| val | genuine | speaker | 48 | 0 | 0 | 0 |
| val | synthetic | target | 48 | 0 | 0 | 0 |
| val | synthetic | source | 25 | 0 | 0 | 23 |

A local scan of all 12 cached Kathbath candidate inventories found **298,049 female / 0 male** metadata rows. This directly explains why that cached candidate pool cannot supply male genuine examples. `joint_core.py::_ensure_inventory` reads lexicographically sorted training shards progressively and stops based on speaker/recording sufficiency, with no gender-coverage criterion. `select_final_pairs` prefers speakers with genuine inventory material; `plan_language` prioritizes attached genuine material for development. These mechanisms can propagate a biased inventory into genuine and development synthetic coverage. They are a plausible selection explanation, not an upstream corpus-wide gender claim. No sampling code/data was changed.

Original-rate coverage is also class-correlated: genuine train/dev 192/48 at 16 kHz; synthetic train 189 at 24 kHz and 3 at 22.05 kHz, dev 46 at 24 kHz and 2 at 22.05 kHz. Resampling everything to 16 kHz does not erase historical recording/codec/resampling cues.

For a later dataset version, inspect candidate metadata across more shard regions, require supported gender coverage in both classes and detector partitions, then select whole participant/recording components under feasible quotas. Report shortfalls and missing values instead of breaking grouping to fill quotas. This is future work: the current split stayed frozen. IFD remains only a candidate paired Indian-English source pending release/permissions/lineage audit; Svarah remains genuine-only Indian-English external false-alarm material; ASVspoof 2019 remains a separate general-English baseline. No ASVspoof 2021 acquisition is needed.

## 5. Controlled experiment specification

One fresh initialization, seed 0; exactly the original train/development IDs, labels and caches; LayerNorm(1024) → Linear(1024,256) → one unidirectional GRU(256) → valid-frame mean → Dropout(0) → Linear(256,2). AdamW lr 0.001, weight decay 0; batch 8; ten epochs / 480 optimizer steps; CPU FP32 / four threads. No scheduler, clipping, class weights, resampling, augmentation or encoder updates. Per-epoch shuffle uses a private generator seeded `0 + epoch`. Lowest development probability-EER selects the checkpoint; strict improvement preserves earliest ties; fixed reporting threshold 0.5.

The **sole modeling change** is a fixed affine transform before the existing input LayerNorm:

```text
N = sum of valid frame counts over training examples
mu[d] = sum(x[i,t,d]) / N
var[d] = sum((x[i,t,d] - mu[d])²) / N
scale[d] = max(sqrt(var[d]), 1e-6)
z[i,t,d] = (x[i,t,d] - mu[d]) / scale[d]
```

N=76,391, dimension=1024, training windows=384, clamped dimensions=2. Statistics use float64 parallel-Welford accumulation; population variance is retained in float64; applied mean/scale are serialized float32. This is frame weighting, not one-window-one-vote statistics. Development frames never fit/adjust the scaler. No per-clip or per-batch fitting occurs. Padded transformed positions are zero, and lengths/masks are unchanged.

`src/detectors/standardization.py::FrameStandardizer` stores mean, scale, variance, fitted flag and metadata with the head as nontrainable state. Config opt-in is `model.parameters.feature_standardization: true`. Historical defaults remain unchanged. Missing required transform state is rejected even with non-strict loading; mismatched enabled/disabled checkpoint construction is rejected. The generic `runner.run_training` and cache CLI both fit on training examples only. Raw caches stay immutable.

Checkpoint metadata records schema `voxsentinel.frame_standardizer.v1`, placement, epsilon, weighting, variance convention, frame count, clamp count, training-cache hash and training-feature-content signature. Applied transform is also exported in `standardizer.pt` and `standardizer.json`.

The constructor's parameter tensors and post-construction RNG state exactly matched the reviewed original class under seed 0. Fitting did not consume RNG. Actual batch order and feature/label pairing were checked against all ten original seed permutations; epoch order hashes are saved in `standardization_checks.json`. Additional full-training eval-mode CE is measured after each epoch, without RNG consumption or optimizer updates, and is clearly separated from online minibatch CE.

Run status **completed**; selected epoch **5**; fit wall time **29.623 s**, including checkpoint writes and additional full train/dev epoch evaluations. Start 2026-09-18T15:06:08.616549+00:00; completion 2026-09-18T15:06:41.086359+00:00. This timing is not a comparable speed benchmark against the original 43.9-second run. No technical retry or second scientific run occurred. A preliminary readiness gate flagged `.gitignore` hash divergence; inspection classified it as non-training metadata before the experiment started.

## 6. Results

CE is full-split evaluation-mode cross-entropy. Accuracy/FAR/miss/confusion use threshold 0.5; EER/AUC below use actual logit-margin ranking (probability versions also saved). Confusion order is TN/FP/FN/TP. “val” is development, not independent test data.

| Model / split | Eval CE | Accuracy | EER | AUC | TN/FP/FN/TP | FAR | Miss |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Original GRU best/train | 0.7029349 | 0.5000 | 0.1823 | 0.9019 | 0/192/0/192 | 1.0000 | 0.0000 |
| Original GRU best/val | 0.7030879 | 0.5000 | 0.2292 | 0.8247 | 0/48/0/48 | 1.0000 | 0.0000 |
| Original GRU final/train | 0.6938350 | 0.5000 | 0.2448 | 0.8262 | 192/0/192/0 | 0.0000 | 1.0000 |
| Original GRU final/val | 0.6938761 | 0.5000 | 0.3333 | 0.7348 | 48/0/48/0 | 0.0000 | 1.0000 |
| Standardized GRU best/train | 0.0034313 | 1.0000 | 0.0000 | 1.0000 | 192/0/0/192 | 0.0000 | 0.0000 |
| Standardized GRU best/val | 0.2018759 | 0.9479 | 0.0417 | 0.9870 | 46/2/3/45 | 0.0417 | 0.0625 |
| Standardized GRU final/train | 0.0001486 | 1.0000 | 0.0000 | 1.0000 | 192/0/0/192 | 0.0000 | 0.0000 |
| Standardized GRU final/val | 0.2858007 | 0.9375 | 0.0417 | 0.9865 | 47/1/5/43 | 0.0208 | 0.1042 |
| Corrected linear train | 0.0000728 | 1.0000 | 0.0000 | 1.0000 | 192/0/0/192 | 0.0000 | 0.0000 |
| Corrected linear val | 0.2241349 | 0.9271 | 0.0417 | 0.9883 | 47/1/6/42 | 0.0208 | 0.1250 |

The standardized head fits all training examples by the selected epoch. Development EER improves from 22.92% to 4.17%; accuracy improves by 44.79 percentage points at the fixed threshold. The final checkpoint further reduces training CE while increasing development CE and misses, so stronger training confidence does not imply better development behavior. Epoch 5 is selected because later epochs tie its EER; no threshold or epoch extension was tried. The linear reference has slightly higher AUC and fewer genuine false alarms, while selected GRU has fewer synthetic misses at 0.5; no significance or universal superiority claim is supported.

### Learning history

| Epoch | Online train CE | Full train eval CE | Dev eval CE | Dev accuracy | Dev EER |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.239345 | 0.075605 | 0.250361 | 0.8958 | 0.06250 |
| 2 | 0.052277 | 0.018962 | 0.179503 | 0.9479 | 0.06250 |
| 3 | 0.020135 | 0.010006 | 0.173150 | 0.9583 | 0.06250 |
| 4 | 0.072556 | 0.029108 | 0.187913 | 0.9375 | 0.06250 |
| 5 | 0.013498 | 0.003431 | 0.201876 | 0.9479 | 0.04167 |
| 6 | 0.002104 | 0.000785 | 0.267439 | 0.9271 | 0.04167 |
| 7 | 0.000488 | 0.000346 | 0.263263 | 0.9479 | 0.04167 |
| 8 | 0.000308 | 0.000244 | 0.273888 | 0.9271 | 0.04167 |
| 9 | 0.000223 | 0.000189 | 0.279435 | 0.9375 | 0.04167 |
| 10 | 0.000174 | 0.000149 | 0.285801 | 0.9375 | 0.04167 |

### Score and margin distributions

| Model / split | Score min–max | Score mean | Score std | Margin min–max | Margin mean |
| --- | --- | --- | --- | --- | --- |
| Original best/train | 0.570169–0.571692 | 0.571049 | 0.000297249 | 0.282539–0.288759 | 0.286131 |
| Original best/val | 0.570475–0.571575 | 0.571008 | 0.000220737 | 0.283788–0.288282 | 0.285965 |
| Original final/train | 0.479783–0.480362 | 0.480105 | 9.38183e-05 | -0.0809116–-0.0785911 | -0.0796202 |
| Original final/val | 0.479932–0.480276 | 0.480099 | 8.173e-05 | -0.0803165–-0.078939 | -0.0796444 |
| Standardized best/train | 7.00513e-07–0.999999 | 0.501613 | 0.497045 | -14.1715–14.079 | 0.265292 |
| Standardized best/val | 1.00254e-06–0.999995 | 0.47242 | 0.478806 | -13.813–12.1983 | -1.13488 |
| Standardized final/train | 3.72074e-08–1 | 0.499987 | 0.499852 | -17.1068–16.8977 | -0.161604 |
| Standardized final/val | 3.4765e-08–1 | 0.459449 | 0.484278 | -17.1747–15.7816 | -1.83017 |

Full quantiles and per-class distributions are in `evaluation/evaluation.json` and `eval_history.json`. Per-window CSVs retain window IDs, labels, languages, generators, both logits, true logit margin and corrected score. `breakdown.json` contains selected-checkpoint language/generator errors with denominators; subgroup numbers remain descriptive only.

## 7. Bounded validation actually performed

- Original 384/96 cache/manifest identity, explicit per-item checks, all 480 prepared-file SHA-256 checks, canonical speaker/reference audit, and exact original artifact preservation checks passed.
- Injected in-memory genuine→synthetic-target leakage is detected across class/role/extension/numeric formatting. Missing required identities are reported. Within-split and cross-split duplicate scopes are distinguished. No manifest was modified for these checks.
- Corrected linear evaluation loaded original weights/scaler with no optimizer invocation; original CE/confusion reproduced and zero threshold decisions changed. Probability saturation ties were counted; margin-ranking metrics retained.
- Original best/final GRU checkpoints were reloaded and all four train/dev metric rows independently recomputed. Historical default forward behavior remains bit-exact under the checked batch.
- Unequal-length toy arithmetic checks population frame weighting and epsilon clamp. Real statistics match an independent two-pass float64 oracle on six representative dimensions. There are no standardizer parameters or buffer gradients. Best/final transform state matches the pre-training exported state exactly.
- Initialization/RNG and ten epoch batch orders match historical code. Generic runner setup was exercised with `fit` intercepted; no extra training was performed.
- Standardized best/final live-to-restored predictions match exactly. Repeated reconstruction is exact. Masked +77 padding changes logits by zero; transformed padding remains zero.
- Batch-one versus batch-eight maximum logit differences: best 3.33786011e-06; final 5.7220459e-06 (bounded tolerance 1e-5).
- Required missing transform state fails even with `strict=False`. Restoring historical checkpoints still succeeds.

Skipped: full pytest suite, broad regression tests, new encoder forward passes/depth profiles, Fisher reanalysis, CUDA/MPS, streaming/inference-service integration, deployment calibration, independent held-out evaluation and new data acquisition. No new test-suite pass count is claimed. Fresh-cache matching in previous reports establishes fidelity only for its tested samples, not universal correctness of encoder choices.

## 8. Reproduction and artifacts

Use the existing `.venv` and pinned requirements; no added third-party dependencies. Run from the repository root. Original directories must stay intact. For corrected exports use new output directories (the linear correction refuses nonempty output); the training CLI allocates a unique suffix rather than overwriting an existing run.

```sh
# Read-only readiness / bounded diagnostics; no training:
.venv/bin/python -m scripts.gru_recovery readiness --out-dir artifacts/runs/recovery-check
.venv/bin/python -m scripts.diagnose_gru identity --out-dir artifacts/runs/recovery-check
.venv/bin/python -m scripts.linear_baseline_gru_cache --reevaluate-saved artifacts/runs/gru-core-v1-diagnosis/stage5/model_scaled.pt --out-dir artifacts/runs/recovery-check/linear
.venv/bin/python -m scripts.verify_gru_standardization

# Exact command executed ONCE for this report (do not rerun just to inspect results):
.venv/bin/python -m scripts.train_gru_from_cache --detector-config configs/base.yaml configs/gru.yaml configs/gru_standardized.yaml --reference-run artifacts/runs/gru-core-v1 --run-dir artifacts/runs/gru-core-v2-standardized --threads 4

# Evaluation-only reconstruction and report regeneration:
.venv/bin/python -m scripts.gru_recovery evaluate --run-dir artifacts/runs/gru-core-v2-standardized --out-dir artifacts/runs/recovery-check/standardized
.venv/bin/python -m scripts.report_gru_recovery
```

Checkpoint reconstruction (requires matching raw embedding identity, no external trainer-only scaler):

```python
from src.config import ModelConfig
from src.detectors.registry import create_detector
from src.detectors.checkpoints import load_checkpoint, restore_checkpoint
path = "artifacts/runs/gru-core-v2-standardized/gru_best.pt"
payload = load_checkpoint(path, model_name="gru")
model = create_detector(ModelConfig("gru", payload["model_config"]))
restore_checkpoint(path, model, model_name="gru")
model.eval()
# model(collated_raw_embedding_batch) applies restored standardization internally.
```

Main outputs:

- `docs/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md` — this report.
- `artifacts/runs/gru-core-v1-diagnosis-corrected/` — readiness, canonical audit, bounded check evidence, corrected linear predictions/metrics, re-evaluated original GRU, gender inventory evidence, preservation hashes and machine-readable recovery summary.
- `artifacts/runs/gru-core-v2-standardized/` — best/final checkpoints, transform, resolved settings, dataset/cache identities, environment, histories, selected-checkpoint predictions, complete best/final train/dev evaluation CSVs, status and log.
- New run `source_snapshot.tar.gz` — complete pre-run source/config/documentation and dependency-file contents, including untracked additions. `code_state.json` has per-file hashes and archive hash; `code_diff.patch`/`git_status.txt` preserve the dirty-tree context. `final-evidence/source_snapshot.tar.gz` additionally includes the report generator and final reporting changes; its own `code_state.json` records file/archive hashes. Hashes alone are not claimed to reconstruct missing uncommitted content.
- `artifact_hashes.json` — hashes of new supporting artifacts, excluding itself; original frozen evidence has a separate before/after hash inventory.

Checkpoint hashes:

| Artifact | SHA-256 |
| --- | --- |
| gru_best.pt | 44ce5ffebd1b94a5ff34de3dcc08fa6b776005c36064c178aa93c5f22d96a2b5 |
| gru_final.pt | 92c4bb0e84deed5ca38b1f584bf4cff8217cbfb0b38db4492a71aa37ee8d56d7 |
| standardizer.pt | 8c0d8ba241b1a9c421d7bdf7565b8e96fd48b63b9e20bb394f04b585a8572e9c |

Changed files (no unrelated code changes):

- `README.md`
- `configs/gru_standardized.yaml`
- `docs/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md`
- `docs/GRU_DIAGNOSTIC_MANAGER_REPORT.md`
- `docs/dataset-preparation-pilot.md`
- `scripts/diagnose_gru.py`
- `scripts/gru_identity_audit.py`
- `scripts/gru_recovery.py`
- `scripts/linear_baseline_gru_cache.py`
- `scripts/report_gru_recovery.py`
- `scripts/train_gru_from_cache.py`
- `scripts/verify_gru_standardization.py`
- `src/detectors/checkpoints.py`
- `src/detectors/gru.py`
- `src/detectors/runner.py`
- `src/detectors/standardization.py`
- `src/detectors/training.py`

## 9. One next recommendation

Freeze this standardized checkpoint as a **pilot reference**, then make the next phase a separately versioned coverage-and-independent-evaluation effort: build feasible gender coverage in both classes with the same participant/recording grouping rules, and reserve untouched speakers/lineage and synthesis systems before further tuning. Do not extend this experiment or sweep optimization settings: training fit has recovered, while independent generalization and source-condition shortcuts are now the larger evidence gaps. Any paired Indian-English expansion, including IFD, requires its own release/lineage audit and cannot be inferred from this Indic-language pilot.
