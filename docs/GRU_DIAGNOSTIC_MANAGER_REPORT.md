# GRU Diagnostic Manager Report — `gru-core-v1` failure analysis

> **Superseded findings — 2026-09-18:** See [the recovery manager report](ASTRA_GRU_RECOVERY_MANAGER_REPORT.md) for corrected diagnostics and the matched standardization experiment. This document is retained as a historical report. Its “confirmed root cause” claim, leakage-refuted conclusion, zero-clamp count, unscaled-predictor description, whitening terminology, linear-separability inference, and uncertainty claims are not supported as written. The original linear probability exports also used the wrong logit scaling; corrected outputs preserve the fitted model and its 0.5 decisions. Historical artifact files remain unchanged.

Date: 2026-09-18
Author: implementation agent (autonomous diagnostic run authorized by the diagnosis master prompt)
Scope: diagnose why the first frozen-IndicWav2Vec → GRU run (`artifacts/runs/gru-core-v1/`) barely learned and classified every validation window as synthetic at threshold 0.5; deliver evidence-backed findings, the bounded experiments (tiny-subset overfit, pooled linear baseline), and one recommended next step.
Constraints honored: no dataset expansion, no encoder fine-tuning, no architecture replacement, CPU float32 only, pinned environments, `gru-core-v1`, its checkpoints, frozen manifests and caches untouched. No commit, merge, reset or push. The original run directory was not modified; all new work lives under `artifacts/runs/gru-core-v1-diagnosis/`.

---

## 1. Executive verdict

**Confirmed root cause (high confidence).** The failing behaviour is a *training-dynamics / optimization-convergence failure on a weak, badly conditioned input representation* — not an implementation defect, not reversed scores and not a degenerate feature cache:

1. The original run's outputs consisted of a **near-constant score for every window** — validation scores spanned only `[0.57047, 0.57158]` (width 1.1e-3) with mean 0.5710 for both classes (class means 0.57089 vs 0.57113). Because all 96 scores sat *above* 0.5, every window was classified synthetic. The final checkpoint (epoch 10) behaved identically but landed *below* 0.5 (mean 0.4801 → every window classified genuine). Across epochs the decision side flips (synthetic for epochs 1, 2, 4, 9; genuine for 3, 5, 6, 7, 8, 10) — the signature of a bias-dominated constant output whose centre drifts around the 0.5 threshold, not of "everything is synthetic".
2. Training **never fit even the training set**: the example-weighted train loss moved only 0.7346 → 0.7006 over 10 epochs (480 optimizer steps), never reaching the balanced-constant level `ln 2 = 0.6931` meaningfully, while the same head architecture **memorizes a 24-window subset to 100 % accuracy / CE 0.011 at the original learning rate** (evidence §6).
3. The head inputs are extremely **ill-conditioned**: cached features are dominated by a near-static template (within-window temporal std 0.010 vs per-frame RMS 0.317 ≈ 3 %; cross-window time-mean std/dim ≈ 0.0022; consecutive-frame correlation 0.998). The class signal present in the features is tiny relative to scale and unevenly distributed across dimensions: the per-dim-standardized pooled linear probe reaches **val EER 4.2 % / AUC 0.988**, while the GRU (and the unscaled optimizer) could not practically reach it. Unscaled LBFGS on the same data *diverges* (objective 1.3e9), demonstrating the conditioning problem independently.
4. Optimization shows a **two-phase behaviour**: on a favourable 24-example subset the loss is flat for ~150 steps, then collapses to zero within ~100 more. The original 480-step minibatch schedule (with weaker full-set signal: nearest-centroid train AUC 0.934 vs 0.986 on the subset) plausibly ended inside its slow phase.

**Likely contributors (medium confidence).**
- Weak per-sample signal-to-noise in the full 384-window set (best linear readout training AUC 0.934, i.e. ~7 % of training pairs are not even linearly separable).
- Margin/score bias dynamics: the model stayed a near-constant predictor (best-checkpoint val margins 0.286 ± 0.0009; final-checkpoint val margins −0.080 ± 0.0003), so its deployed-threshold behaviour is controlled by the wandering of a single bias-like quantity.
- Checkpoint selection noise: min-val-EER selection over 10 flickering epochs (EER 0.23–0.42; standard error ≈ 0.06 at n=96) picked epoch 2; the selected checkpoint's ranking (val AUC 0.825) is not meaningfully better than neighbouring epochs. This is a selection-process observation, not a defect.

**Unresolved questions.**
- Whether a per-dim-standardized (whitened) retraining of the identical head reaches the linear probe's operating level — this is the recommended next controlled experiment (§8).
- Whether the mid/late transformer layer depth profile of this checkpoint (norm explosion to ~4.3e4 at layers 22–23 before the final LayerNorm; relative temporal variation richer at layers ~12–21) is intrinsic to the checkpoint or specific to this audio domain; no second checkpoint was available, and layer-choice experiments are out of scope for this task.
- Whether the residual val errors of any approach reflect spoof-cue difficulty or residual dataset artefacts (e.g. source-recording/systematic conversion differences correlated with the label by construction; the design deliberately pairs speaker/content, but recording-chain differences cannot be excluded with the current evidence).

**Nothing about the original metrics was "wrong".** They reproduced exactly (§4). EER 22.9 % is a valid ranking statistic of the selected checkpoint — it is *not* 77.1 % accuracy; at the deployed threshold 0.5 that checkpoint is a coin flip that says "synthetic" to everything (F1 0.667 is the trivial always-positive artefact).

---

## 2. What was inspected and executed

### 2.1 Code and configuration actually used by the run
- Original run harness: `scripts/train_gru_from_cache.py` (sha256 matches run snapshot `e24ccdc1…`), training loop `src/detectors/training.py`, head `src/detectors/gru.py` (LayerNorm → Linear 1024→256 → 1-layer GRU(256) → masked mean pooling → Dropout(0) → Linear(256→2)), collator `src/data/batch.py`, metrics `src/scoring/metrics.py`, evaluation `src/scoring/evaluation.py`, checkpoints `src/detectors/checkpoints.py`, label contract `src/data/labels.py`.
- Resolved settings (from `artifacts/runs/gru-core-v1/settings.json`): AdamW lr 1e-3, wd 0, batch 8, 10 epochs, seed 0, CPU float32, no scheduler, no clipping, no class weights; selection = lowest val EER (earliest tie); threshold 0.5; label 1 = synthetic positive class.
- **Original vs current source:** the run snapshot hashed 39 files. 37/39 still match byte-for-byte; the only differences are two post-run *documentation* edits (`README.md`, `docs/dataset-preparation-pilot.md`). All training-path code is identical, so all diagnoses below are about the same code that produced the run.

### 2.2 Environment and dataset identities
- Python 3.10.18; torch 2.2.2; torchaudio 2.2.2; fairseq 0.12.1; numpy 1.23.5; CPU (macOS arm64); no matplotlib/pandas/scipy/sklearn available (no analytics stack was installed; exact tables/CSV used instead of plots).
- Encoder: `models/indicwav2vec_large.pt`, sha256 `26bb5ada…ab59`, `output_layer: null` (final output incl. final LayerNorm; `normalize=true`, `layer_norm_first=true`), embedding dim 1024, frame hop 20 ms, 4 s → 199 frames.
- Cache/anchor identities: train manifest sha `1b361e14cb60cb73…`, train bundle sha `a893b5038cfc2375…`; val manifest sha `a23acb1cdbe86a94…`, val bundle sha `069b3b78e2c4f767…`; preprocessing `voxsentinel-prep-2`.
- Commit `c250bbd0ac3b2af1ae345ce48960fcc63d200557` (branch `feature/indicVac2Wav`), dirty checkout; run snapshot diff/hashes in `artifacts/runs/gru-core-v1/code_state.json`.

### 2.3 Executed (all artifacts under `artifacts/runs/gru-core-v1-diagnosis/`)
| Stage | Command | Outputs |
|---|---|---|
| Score facts | `scripts/diagnose_gru.py scores` | `stage2/stage2_scores.json`, `val_scores_recomputed.csv`, `metric_selftest.json`, bucket CSVs |
| Reload/parity | `scripts/diagnose_gru.py reload` | `stage2/stage2_reload.json`, `train_scores_best.csv` |
| Feature stats | `scripts/diagnose_gru.py features` | `stage3/stage3_features.json` |
| Batch audit | `scripts/diagnose_gru.py batches` | `stage3/stage3_batches.json` |
| Optimizer probe | `scripts/diagnose_gru.py optimizer` | `stage3/stage3_optimizer.json` |
| Activations | `scripts/diagnose_gru.py activations` | `stage3/stage3_activations.json` |
| Identity audit | `scripts/diagnose_gru.py identity` | `stage3/stage3_identity.json` |
| Stage separation probe | `stage3_fisher_probe.py` | `stage3/stage3_fisher.json` |
| Tiny-subset overfit | `scripts/overfit_gru_subset.py` | `stage4/` (manifest, history, model, summaries) |
| Linear baseline | `scripts/linear_baseline_gru_cache.py` | `stage5/` (summary, weights+scaler, predictions) |
| Bounded encoder check | inline (logged) | cache-vs-fresh comparison, per-layer profile |
| Decision record | — | `stage6_decision.json`, `commands.txt`, `artifact_hashes.txt` |

Preserved originals: `gru_best.pt` sha256 `013c7ada…`, `gru_final.pt` sha256 `866ab269…`, all run JSONs and manifests — unchanged (hashes in `artifact_hashes.txt`).

---

## 3. Independent recomputation of the original metrics (no retraining)

Saved predictions (`val_predictions.jsonl`) were re-scored with an independently written, self-tested metric implementation (tie-aware Mann-Whitney AUC; explicit threshold sweep for EER; validated on perfect / reversed / random / tie-only synthetic cases; `stage2/metric_selftest.json`).

- **Softmax check:** recomputing `softmax(logits[:, :2])[1]` reproduces the saved scores to max abs diff **5.6e-8** (float32 round-trip). A batch-axis softmax does **not** match (mean abs diff 0.007). No reversed direction, no class swap: the ranking of `logit_synthetic − logit_genuine` is identical to the saved-score ranking (**0/96 discordant ranks**).
- **Recomputed metrics, best checkpoint, val, threshold 0.5:** TN 0, FP 48, FN 0, TP 48; accuracy 0.500; balanced accuracy 0.500; precision 0.500; recall 1.0; F1 0.667; genuine false-alarm rate 1.0; synthetic miss rate 0.0. Exactly matches `metrics.json`.
- **AUC (tie-aware): 0.8247. EER: 0.22917 = 22/96**, matching the production metric to the last bit; the EER operating point is at score ≈ 0.57099 where FAR = FRR = 0.2292. (The production `_curve` and the independent sweep agree; tie/interpolation conventions are documented in `stage2_scores.json`.)
- **Interpretation:** the checkpoint does not "detect fake audio". It orders windows weakly (AUC 0.82) while its absolute scores say "synthetic" for everything. EER 22.9 % ≠ 77.1 % deployed accuracy (deployed accuracy at 0.5 is 50 %). An operating point at the EER crossing (score ≈ 0.57099) would give ≈ 77.1 % accuracy on this same 96-window set — but that point was fitted on the same data used for checkpoint selection, so it is not a validated threshold. The val set is development data: it already selected the checkpoint.

### 3.1 Score and margin distributions (best checkpoint, val, 96 windows)

| statistic | score genuine | score synthetic | margin genuine | margin synthetic |
|---|---:|---:|---:|---:|
| count | 48 | 48 | 48 | 48 |
| min | 0.570475 | 0.570580 | 0.283788 | 0.284217 |
| q25 | 0.570764 | 0.571012 | 0.284970 | 0.285982 |
| median | 0.570885 | 0.571172 | 0.285465 | 0.286634 |
| mean | 0.570887 | 0.571129 | 0.285472 | 0.286459 |
| q75 | 0.570984 | 0.571261 | 0.285869 | 0.286999 |
| max | 0.571322 | 0.571576 | 0.287247 | 0.288282 |
| std | 0.000165 | 0.000203 | 0.000672 | 0.000828 |

Both classes live inside a 1.1e-3-wide band; class means differ by ~1.3–1.5 within-class σ. Bucket tables (`stage2/score_buckets.csv`, `stage2/margin_buckets.csv`) show a monotone-ish trend — low buckets are genuine-heavy (12 vs 3), high buckets synthetic-heavy (12 vs 2 at the top end) — i.e. weak but real ordering, drowned in magnitude terms. There is no meaningful "threshold" anywhere near 0.5 that separates classes: everything is > 0.5. A distribution slightly above 0.5 with ordering information is exactly what is measured; nothing here proves "collapsed representations" by itself — the separation probes (§7) do the rest.

### 3.2 Best/final checkpoint table (reload, eval mode, no grad, no shuffling, batch 8)

| model / split | CE (eval) | accuracy @0.5 | EER | AUC | TN/FP/FN/TP | score mean ± std |
|---|---:|---:|---:|---:|---|---|
| best / train (384) | 0.7029 | 0.500 | 0.1823 | 0.9019 | 0/192/0/192 | 0.5710 ± 0.00030 |
| best / val (96) | 0.7031 | 0.500 | 0.2292 | 0.8247 | 0/48/0/48 | 0.5710 ± 0.00022 |
| final / train (384) | 0.6938 | 0.500 | 0.2448 | 0.8262 | 192/0/192/0 | 0.4801 ± 0.00009 |
| final / val (96) | 0.6939 | 0.500 | 0.3333 | 0.7348 | 48/0/48/0 | 0.4801 ± 0.00008 |

- **Reload parity:** regenerating validation predictions from reloaded `gru_best.pt` reproduces the saved logits and scores with **max abs diff 0.0** (bit-exact). The checkpoint stores and restores the model faithfully.
- Eval-mode full-train CE (0.7029) vs the online epoch-2 *training* loss (0.7113): different quantities (teacher-forced eval-mode vs online shuffled minibatch average); both confirm the model never fitted training data. At epoch 10: eval 0.6938 vs online 0.7006 — final model sits almost exactly on the balanced-constant level ln 2 = 0.6931 and predicts one class everywhere.
- The best checkpoint is *not* better in aggregate than the final one in any deployed sense: both are constant predictors on opposite sides of 0.5.

### 3.3 Batching, padding and masking probes (reloaded best checkpoint)
- Batch-8 vs batch-1 logits over all 96 val windows: max abs diff **3.6e-7** (float noise) — no cross-example contamination.
- Masked extra padding (40 zero frames appended with `padding_mask=True`, `valid_lengths` unchanged): logits diff **0.0** — masked frames are excluded exactly, as designed.
- (Transparency: a first version of this probe appended frames without masking and measured a 0.081 diff; that variant tested "feed junk as data", not padding handling. It was replaced by the correct masked probe above; both are recorded in the script history.)

---

## 4. Data-to-loss path findings (numeric evidence)

### 4.1 Feature tensors
- Shapes: `[T, 1024]`, T = 199 for 475/480 windows (train {199: 380, 193: 3, 192: 1}; val {199: 95, 187: 1}); all finite; per-frame L2 norm 10.07–10.21.
- **Near-static representation:** element std 0.317; within-window temporal std **0.0100** (~3 % of RMS); time-mean template std 0.316 vs per-dim cross-window std of window means **0.00216** (~0.7 % of RMS); consecutive-frame correlation **0.998**; 74.5 % of dims vary by < 1e-3 across time in a sampled window.
- **No feature corruption or reuse:** exact-content hashing of all 480 tensors → **0 duplicate groups**; sketch-based near-duplicate scan (time-mean + time-std + first/last frame, corr > 0.9999) → **0 pairs**.
- **Cache fidelity:** fresh bounded encoder re-extraction of 4 prepared windows (2 languages × both labels) matches the cached tensors with **max abs diff 0.0** each.
- Depth profile (single window, diagnostic): relative temporal variation is 0.7 % at layer 0, ~1.4–2.0 % at layers 8–20, then norms explode (~3.7e3 → ~4.3e4 at layers 22–23) and the final LayerNorm output drops to **0.1 %** relative variation. The final-layer output is what the deployed head consumed; it remains usable (see linear probe) but is dominated by a template direction.

### 4.2 Labels, lengths, masks, collation
- `collate_examples` verified by construction self-check and shuffled re-collation: labels cannot separate from their features; `valid_lengths` counts frames; `padding_mask` True = ignore; the GRU packs sequences (padding never enters the recurrence) and mean-pools with the mask (padding never enters pooling).
- Reproduced the exact original epoch orderings (`randperm(seed=0+epoch)`, epochs 1–10): 48 batches/epoch; batch class compositions binomial (most common 4/4); single-class batches occur at the expected random rate (1–3 per epoch; 0.6 % of all 480 batches; epochs 1, 2, 6). The original run's outcome is not explained by batch composition.

### 4.3 Optimizer and updates (disposable copy, seed 0, real batches)
- All 10 parameter tensors are trainable, present in AdamW **exactly once**, and all update within one epoch (update L2 norms 5.8e-4 … 0.94; e.g. GRU `weight_ih_l0` 0.92, projection 0.78).
- Loss on 4 fixed batches: 0.8234 → 0.6106 after the second pass — optimization can move this head on real data.
- Cosine/train/eval switching, `zero_grad(set_to_none=True)`, backward/step order, finite-loss guard: all as expected; head outputs are not detached; the frozen encoder is not in the graph.

### 4.4 Activation statistics (4 genuine + 4 synthetic val windows, one batch)
- Fresh init: projection input LN output behaves normally; GRU sequence varies over time by 0.025 (within window); **pooled vectors vary across samples by only 0.0043**; output margins std 2.7e-3.
- Trained best checkpoint: pooled vectors vary across samples by **0.0034**; margins 0.286 ± **0.00086**. Training barely increased cross-sample output variation; it mostly shifted a constant.

### 4.5 Hypothesis table

| # | Hypothesis | Supporting evidence | Contradicting evidence | Conclusion |
|---|---|---|---|---|
| 1 | Reversed score direction / wrong softmax axis / swapped classes | — | Class-axis softmax recompute max diff 5.6e-8; margin ranking 0/96 discordant; AUC 0.82 consistent with EER 0.23 | **Refuted** |
| 2 | Label misalignment in collation or shuffling | — | Collation self-checks pass; shuffled pairing verified; batch compositions sane; eval parity bit-exact | **Refuted** |
| 3 | Masking/pooling defect (padding affects logits) | — | Masked extra padding diff 0.0; batch-size invariance 3.6e-7; code uses packing + masked mean | **Refuted** |
| 4 | Optimizer/loss implementation defect | — | All params in optimizer once and updating; CE on raw logits with integer targets; example-weighted means | **Refuted** |
| 5 | Checkpoint save/restore defect | — | Reload reproduces saved predictions bit-exactly (0.0) | **Refuted** |
| 6 | Data leakage or split contamination | — | 0/0/0/0 speaker-key intersections across roles; 0 reference overlaps; 0 prepared-audio hash duplicates; 0 feature duplicates | **Refuted** |
| 7 | Feature cache corrupted / duplicated / stale | — | Fresh encoder extraction matches cache exactly (0.0); 0 duplicates; identity hashes match manifests | **Refuted** |
| 8 | Encoder final layer carries no usable signal | — | Linear probe on pooled features: train CE 7e-5/acc 1.000; **val EER 4.2 %, AUC 0.988, acc 92.7 %**; shuffled-label control 0.56 AUC | **Refuted** |
| 9 | Training dynamics/convergence failure on weak, ill-conditioned signal | Loss never reaches ln 2 (10 epochs); margins stay ~constant; 24-subset: flat ~150 steps then collapse (fits when given a smaller, cleaner problem); unscaled LBFGS diverges (1.3e9); scaled converges to EER 4.2 % | The 24-subset reached the target, so the pipeline *can* fit | **Supported — root cause** |
| 10 | Checkpoint selection noise (min-EER over 10 small-n epochs) | Per-epoch val EER 0.23–0.42 flickers with SE ≈ 0.06; selected epoch 2; ΔAUC to neighbours small | Selection is by design, not a bug | **Contributor** |
| 11 | Features degenerate across samples (all windows alike) | Template dominance (0.7 % cross-window variation) | Scaled linear probe reaches EER 4.2 % — per-sample differences are tiny but richly informative | **Partially supported as conditioning factor, not as signal absence** |

---

## 5. Canonical split audit (frozen manifests)

- Train 384 (12 languages × 16 genuine Kathbath + 16 synthetic IndicSynth), val 96 (12 × 4 + 4). Val generators: freevc24 25, xtts_v2 21, vits 2 (+48 genuine).
- Speaker-key intersections train↔val across all role pairs (source→source, source→target, target→source, target→target, within dataset+language namespaces): **0 in every pair**.
- Reference-recording string overlaps train↔val (per dataset): **0**. Prepared-audio SHA-256 duplicates: **0**. Numeric speaker-id collisions across datasets: **0** (flagged check; ids remain dataset-scoped namespaces).
- Cross-dataset person identity cannot be independently resolved beyond the recorded lineage evidence; this remains a stated identity limit (unchanged from dataset preparation).
- Per-language val subgroups have n=8 and vits has n=2 — subgroup claims are statistically thin by construction; no subgroup EER/AUC was computed on single-class subgroups.

---

## 6. Tiny balanced-subset overfit diagnostic (Stage 4)

- **Subset:** 24 distinct training windows (one genuine + one synthetic per core language), chosen deterministically with Python `random.Random(0).choice` over window_id-sorted candidates; ids in `stage4/subset_manifest.json`. No validation examples were used.
- **Setup:** fresh head (`1024→256`, 1 layer, dropout 0 verified), same optimizer (AdamW lr 1e-3, wd 0), full batch of 24, no scheduler, seed 0, 500-step cap. Predeclared diagnostic target: 100 % accuracy with CE < 0.05 for five consecutive evaluations.
- **Outcome — SUCCESS:** target reached at step 245 (final eval CE **0.0109**, accuracy **1.000**, 47.9 s). The lr 1e-4 fallback was not needed.

| step | train loss | eval loss | eval acc |
|---:|---:|---:|---:|
| 1 | 0.7113 | 1.5964 | 0.500 |
| 50 | 0.6917 | 0.6914 | 0.833 |
| 100 | 0.6590 | 0.6934 | 0.500 |
| 150 | 0.6280 | 1.9656 | 0.500 |
| 200 | 0.3326 | 0.3640 | 0.833 |
| 245 | 0.0112 | 0.0109 | 1.000 |

- **Interpretation:** the head, optimizer, collator and loop *can* fit these samples — this is memorization capacity, not spoof-generalization. Two features are notable: (a) a long flat phase (~150 steps) before runaway convergence, including transient overshoots (eval loss 1.60 → 1.97 spikes while accuracy was still 0.5); (b) the subset is a slightly "easier" problem than the full set (nearest-centroid train AUC 0.986 vs 0.934; d′ 1.98 vs 1.54), so success here does not imply the 384-window run should have converged within its 480-step budget.
- For scale: memorizing these 24 favourable examples required 245 full-batch steps (≈ 245 passes over the subset). The original run took 480 minibatch steps, which correspond to only ≈ 10 passes over the 384-window training set, and its losses at every epoch sit in the same slow phase the subset run showed during its first ~150 steps.

---

## 7. Pooled linear baseline (Stage 5) and GRU comparison

Specified protocol: mean-pool all valid frames (1024-d per window); per-dim mean/std scaler fitted on **train only** (epsilon 1e-6, 0 dims clamped); two-class logistic regression by PyTorch LBFGS (max 200 iterations), objective = cross-entropy + 1e-4·Σw² (bias unpenalized); final model = endpoint of that objective; no validation-based selection.

| variant | status | train CE / acc | val CE | val acc | val EER | val AUC | val TN/FP/FN/TP |
|---|---|---:|---:|---:|---:|---:|---|
| **scaled (specified default)** | converged, 65 iters (67 closures) | **7.3e-5 / 1.000** | 0.2241 | **0.927** | **0.0417** | **0.9883** | 47/1/6/42 |
| unscaled (comparison row) | diverged, ran full 200 iters | 1.27e9 / 0.500 | 1.27e9 | 0.500 | 0.500 | 0.500 | 48/0/48/0 |

- The unscaled variant is an optimizer-conditioning artefact (step sizes vs unbalanced per-dim scales; objective 1.3e9), recorded for completeness — it also independently demonstrates how badly scaled this feature space is.
- **Controls:** shuffled-label refit → val AUC 0.560, EER 0.479 (no leakage in the pipeline); nearest-centroid on scaled space (no fitting beyond means) already reaches val AUC 0.925 / EER 0.146; per-language val AUC = 1.000 in 10/12 languages (Telugu 0.875; Hindi / Malayalam / Urdu accuracies 0.75 / 0.88 / 0.88 stay at AUC 1.000 — threshold artefacts at n=8).
- **GRU vs baseline (same 96 val windows, same positive class, same EER/AUC definitions, same 0.5 threshold):** val EER 0.2292 vs **0.0417**; val AUC 0.8247 vs **0.9883**; val accuracy 0.500 vs **0.927**; train CE 0.703 vs **7e-5**. The GRU underperforms even the nearest-centroid rule in whitened space.
- **Comparison caveats (explicit):** the baseline uses static mean pooling (no temporal modelling), its own scaling, and a quasi-Newton solver; the GRU's LayerNorm is a different operation; this is a diagnostic reference, not a leaderboard. A linear model beating this GRU *setup* identifies a direction (input conditioning/optimization), and does not establish that GRUs are generally worse; conversely the baseline's success does not prove the encoder is spoof-robust — it may exploit any stable genuine-vs-converted difference, including recording/conversion artefacts.

---

## 8. Decision, fixes, and the one recommended next experiment

- **Bugs found and fixed in the training path: none.** Every suspected implementation item listed in the hypothesis table was tested and refuted; the original artifacts, checkpoints and metrics were reproduced exactly (bit-exact parity). Per the task's own rule, **no corrected full-core rerun was launched**, and no code in the training path was modified (nothing was patched merely to force a narrative).
- **Do not "continue until validation improves":** no further full-core runs are authorized or needed for this diagnostic.

**Recommended next controlled experiment (one):** rerun the *same* GRU head, same seed 0, same 10-epoch budget, same checkpoint-selection rule and metrics, with **one change: per-dimension standardization of the cached features fitted on the training split only** (save the scaler with the run). Rationale chain from evidence: (i) the class signal is already linearly reachable on the *same frozen features* at val EER 4.2 % once per-dim scale conditioning is fixed; (ii) the failing GRU path shows a slow-start/convergence pattern, and its unscaled-input sibling (LBFGS) diverges outright; (iii) the head itself is not the capacity bottleneck (24-example memorization succeeds). This isolates input conditioning as the single manipulated variable.

- If the whitened rerun still underperforms the linear probe, the *secondary* levers, in order, are: (a) a larger step budget at the same lr (flat-phase evidence), (b) selection stability (record per-epoch margins/scores, not only EER at n=96). 
- **Not** recommended now: data expansion (the linear ceiling already far exceeds the GRU result; more data would not fix optimization scale), encoder fine-tuning (prohibited and unnecessary), architecture replacement (no capacity defect demonstrated), calibration of the current checkpoint (its scores are bias-dominated; there is nothing worth calibrating yet).

---

## 9. User-facing explanation: why every window was classified synthetic

The selected model was not saying "this audio is fake" per window. It had learned to emit an almost constant score — values between 0.57047 and 0.57158 — for every window it saw, genuine or synthetic. Because the deployed rule predicts "synthetic" whenever the score is ≥ 0.5, all 96 windows were labelled synthetic. The tiny variations inside that band did carry a weak ranking signal (AUC 0.82; class means differ by ~1.3σ), which is why the EER statistic (22.9 %) is not 50 % — but nothing ever moved a single window across the 0.5 line, so the deployed accuracy was a coin flip and F1 0.667 was the trivial arithmetic of always predicting one class. By the end of the same run (epoch 10), the score band had drifted to the other side of the threshold (mean 0.4801) and the model classified everything genuine. Both outcomes are the same phenomenon: a near-constant output whose bias wandered around the decision threshold during an optimization that never truly separated the classes.

---

## 10. Evidence index and reproducibility

Primary evidence (all paths relative to the repository root):

- Original: `artifacts/runs/gru-core-v1/` (`settings.json`, `history.{json,csv}`, `metrics.json`, `val_predictions.jsonl`, `run_status.json`, checkpoints).
- Diagnosis: `artifacts/runs/gru-core-v1-diagnosis/`
  - `stage2/`: recomputed scores/metrics (`stage2_scores.json`), metric self-test, reload table + probes (`stage2_reload.json`), per-sample CSV, bucket tables.
  - `stage3/`: `stage3_features.json`, `stage3_batches.json`, `stage3_optimizer.json`, `stage3_activations.json`, `stage3_identity.json`, `stage3_fisher.json`.
  - `stage4/`: subset manifest (24 ids), learning history, model snapshot, summary.
  - `stage5/`: summary (both variants), fitted weights + scaler (`model_scaled.pt`), predictions CSVs.
  - `stage6_decision.json`, `commands.txt`, `artifact_hashes.txt`, copied diagnostic scripts.
- Dataset/caches (read-only): `artifacts/datasets/manifests/…`, `artifacts/datasets/features/…`.

Reproduction commands: `artifacts/runs/gru-core-v1-diagnosis/commands.txt`. Script hashes and original-checkpoint hashes: `artifact_hashes.txt`. The 96 validation windows are **development data already used for checkpoint selection** — none of the numbers above are independent-test claims.

Recorded failures and incomplete investigations: no plotting stack available (exact tables/CSV instead); one mis-designed padding probe replaced by the correct masked version (both documented); cross-dataset speaker identity resolvable only within namespaces; no second encoder checkpoint available for layer-depth comparison; per-generator val subgroups as small as n=2 prevent per-generator conclusions.
