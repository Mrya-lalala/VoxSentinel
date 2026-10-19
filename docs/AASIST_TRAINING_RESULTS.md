# Embedding-adapted AASIST v3 training results (2026-09-22)

This is the training and acceptance record for the **embedding-adapted AASIST**
head on the v3 Indic research split. It is not a reproduction of the published
waveform AASIST and no improvement over the GRU baseline is claimed — the
measured development result is worse than the GRU's and is reported as such.

Scope: branch `feature/aasist`; head introduced in `fbe1c8f`, training runner, release packaging and inference support added subsequently.

---

## 1. The blocker: what was found, what was implemented, what failed locally

### 1.1 Teammate evidence is absent (unknown, not resolved)

No teammate AASIST training script, configuration, logs, checkpoint or error
trace exists in this checkout, in `artifacts/**` (runs, handoffs, evaluations,
releases), in git stashes, or in the parent project folder. The user confirmed
that no evidence can be supplied and authorized independent diagnosis. The
teammate's original failure therefore remains **unreproduced and unidentified**
— it is recorded here as an open external unknown, not as a solved issue.

### 1.2 What independent setup found and fixed

1. **No AASIST training runner existed.** The Gate-4 script
   `scripts/train_aasist_v3_from_cache.py` planned in
   [AASIST_IMPLEMENTATION_PLAN.md](AASIST_IMPLEMENTATION_PLAN.md) was absent, so
   no authorized training path existed at all. Implemented as a strict sibling
   of the GRU v3 runner: same fail-closed cache audit, same dataset identity,
   same evidence capture, same controlled-recipe guard (`INITIAL_COMPARISON`),
   CPU + four threads enforced.
2. **v3 feature caches were absent locally** (the four asset-dependent test
   skips). The teammate's `voxsentinel-v3-features` bundle could not be
   obtained, so the caches were rebuilt from the verified manifests and
   prepared audio with the pinned frozen encoder (authorized fallback):
   - `artifacts/datasets-v3-coverage/features/train.pt` —
     sha256 `75a812faff67206e9ea2ded63f906d7c5ef701d14ef070b85c88d9e2fe113525`,
     458 windows (266 genuine / 192 synthetic), 91,117 valid frames;
   - `artifacts/datasets-v3-coverage/features/dev.pt` —
     sha256 `3d15508f10ce90b516276ca47bbc635a79e3386e47ee5764d66d69a2eb223913`,
     135 windows (87 / 48), 26,853 valid frames;
   - all items are `extracted_v3` (no historical v1 cache was available for
     reuse); per-language distributions and frame totals match the GRU
     release's recorded cache identity exactly;
   - `check` audit: ready, 0 failures (including prepared-audio hashes).
3. **`run_parity` was unverifiable on this platform (reproduced local failure).**
   The pre-existing contract required freshly extracted items to reproduce the
   cache **bit-exactly**, but the cache builder extracts in grouped batches
   while parity re-extracts one item at a time. Measured on this x86 Windows
   machine: the same window extracted alone is bit-exact across repeats (0.0),
   but differs from a grouped-batch extraction by 1.2–1.9e-06 because CPU
   kernel selection depends on the GEMM shape. The check is now: repeat
   single-item extraction must be bit-exact, and the cache comparison uses an
   absolute-error tolerance of 1e-5 (the same bound already applied to reused
   items; an empirical drift budget, not a ULP bound), and `--threads` flows
   into parity so the thread count can be held fixed. Evidence: `artifacts/datasets-v3-coverage/features/reports/parity.json`
   (`passed: true`, `max_abs_error: 1.9e-06`, `repeat_bit_exact: true`).

Previously fixed on this branch by the intervening agents and retained: the
AASIST input-validation defects and the release `head.model_name` dispatch.

---

## 2. Training inputs (verified contract)

| Item | Value |
|---|---|
| Train / dev windows | 458 / 135 |
| Development role | checkpoint selection only; never supplies gradients |
| Frozen encoder | IndicWav2Vec, `output_layer = null`, 1024-d, cpu |
| Encoder checkpoint sha256 | `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59` |
| Preprocessing | `voxsentinel-prep-2` |
| Labels | 0 = genuine, 1 = spoof |
| Standardizer | fitted on 91,117 TRAIN valid frames only; cache signature bound; stored in checkpoint |

No demonstration clips were added to training or development selection.

---

## 3. Controlled experiment

Recipe (mirrors the agreed GRU v3 comparison, enforced by the runner guard):
seed 0, batch size 8, 10 epochs, unweighted cross-entropy, AdamW lr 0.001
(weight decay 0), fixed threshold 0.5, checkpoint selection = minimum
development EER with earliest-epoch tie-break, CPU float32, 4 threads.

Documented deviations: **head architecture only** (the AASIST embedding
adapter + ported graph backend per the Gate-1 plan, `channels=32`,
`latent_nodes=16`, `pooled_bins=32`); caches freshly extracted because no
historical v1 cache was available for per-example reuse. No other protocol
change, no hyperparameter search.

Command (completed 2026-09-22 06:02 UTC, 23.6 s wall on cache):

```powershell
.\.venv\Scripts\python.exe -m scripts.train_aasist_v3_from_cache `
  --run-dir artifacts/runs/aasist-v3-expanded-standardized
```

### 3.1 Per-epoch development metrics

| Epoch | Train CE | Dev CE | Dev EER | Dev acc | FP | FN |
|---|---|---|---|---|---|---|
| 1 | 0.83358 | 0.71545 | 0.3125 | 0.60000 | 43 | 11 |
| 2 | 0.55812 | 0.34262 | 0.14583 | 0.86667 | 9 | 9 |
| 3 | 0.23008 | 0.32633 | 0.14583 | 0.87407 | 9 | 8 |
| 4 | 0.20307 | 0.33100 | 0.16667 | 0.88148 | 2 | 14 |
| 5 | 0.12397 | 0.25635 | 0.12644 | 0.88148 | 3 | 13 |
| 6 | 0.08283 | 0.27541 | 0.12644 | 0.88889 | 3 | 12 |
| 7 | 0.21674 | 0.23386 | 0.12500 | 0.90370 | 7 | 6 |
| 8 | 0.20437 | 0.22905 | 0.10417 | 0.91852 | 4 | 7 |
| 9 | 0.13417 | 0.20666 | 0.10345 | 0.92593 | 3 | 7 |
| **10 (selected)** | **0.10537** | **0.23737** | **0.08046** | **0.91852** | **1** | **10** |

Selected checkpoint: **epoch 10**, development EER `0.08046`, accuracy
`0.91852`, balanced accuracy `0.89009`, genuine false-alarm rate `0.0115`,
spoof miss rate `0.2083` (threshold 0.5). Best and final live/reload logit
error: exactly `0.0`.

**The AASIST head does not beat the GRU.** The GRU release's recorded
development result is 97.04% accuracy / 2.30% EER on the same split; the
embedding-adapted AASIST reaches 91.85% / 8.05%. This unfavorable result is
preserved here and in the release manifest.

### 3.2 Artifacts and hashes

| Artifact | Path | SHA-256 |
|---|---|---|
| Run directory | `artifacts/runs/aasist-v3-expanded-standardized/` | — |
| Best checkpoint | `.../aasist_best.pt` | `009228206c43a290fd1cd12b5eb96ed35426c20867998b0b25f9add43f1c224e` |
| Final checkpoint | `.../aasist_final.pt` | `7255aeb327783bb1d07f6ef2d37cc95652b576ff14a59694377ff2ba13705738` |
| Release head | `artifacts/releases/indic-aasist-v3-research/aasist_best.pt` | same as best |
| Release manifest | `artifacts/releases/indic-aasist-v3-research/release.json` | `8f18da5f94f5c8ff297c710d820c08a1dd61dd0b763c1b0b9df66a4fbf90a0f7` |
| Release notice | `artifacts/releases/indic-aasist-v3-research/RESEARCH_USE.txt` | `5d5ff26644f646ca489e8ec99ffa0b40559f47e72bccc068ad5586bb84197f03` |
| Dev predictions | `artifacts/runs/aasist-v3-expanded-standardized/dev_predictions.jsonl` | (per-window scores that establish selection) |
| Cache build report | `artifacts/datasets-v3-coverage/features/reports/cache_build_report.json` | — |
| Cache parity report | `artifacts/datasets-v3-coverage/features/reports/parity.json` | — |
| Check-only evidence | `artifacts/datasets-v3-coverage/reports/aasist_check_only.json` | — |

The run directory additionally carries `settings.json`, `dataset_identity.json`
(manifest and bundle hashes, ledger observation), `cache_audit.json`,
`class_ratios.json`, `environment.json`, `code_state.json`
(code snapshot + diff), `live_epoch_evidence.json`, `history.{json,csv}`,
`metrics.json`, `breakdown.json`, `coverage_breakdown.json`,
`standardizer.{pt,json}`, and `run_status.json`.

The GRU release (`artifacts/releases/indic-gru-v3-research/`) was not modified,
renamed or repackaged.

---

## 4. Reproduction and inference

Weights, prepared audio, feature caches and run evidence are separate local artifacts,
not included in this source branch. The numerical results above describe the recorded
Windows run; they are development results, not independent benchmark accuracy.

From the repository root (PowerShell, Python 3.10 environment):

```powershell
.\.venv\Scripts\python.exe -m scripts.train_aasist_v3_from_cache --check-only
.\.venv\Scripts\python.exe -m scripts.train_aasist_v3_from_cache --run-dir artifacts/runs/aasist-v3-new-run
.\.venv\Scripts\python.exe -m scripts.package_aasist_release --run-dir artifacts/runs/aasist-v3-new-run --out-dir artifacts/releases/aasist-v3-new-release
```

The cache build/check/parity commands are in [Data and training](DATA_AND_TRAINING.md).
Preserve original caches and use a new output directory for any rebuild.

For a supplied compatible release, the existing C1 API supports:

```powershell
$env:INFERENCE_RELEASE = 'artifacts/releases/indic-aasist-v3-research/release.json'
$env:ENCODER_CHECKPOINT = 'models/indicwav2vec_large.pt'
$env:DEVICE = 'cpu'
.\.venv\Scripts\python.exe -m uvicorn src.service.api:app --host 127.0.0.1 --port 8000
```

Use `/health` and multipart `file` uploads to `/detect`; stop with Ctrl+C.
This branch does not include the local demo webpage or whole-recording endpoint.
The released inference contract scores one selected window.

## 5. Limitations

The teammate's original training failure remains unidentified. The local training
runner and cache parity fixes address independently reproduced setup issues.
AASIST underperforms GRU on the shared development split. Fresh Windows caches have
different hashes from the original GRU training caches, despite matching data and
encoder identities. No Indian-English, streaming, GPU or production validation is
claimed. Keep research-use and upstream attribution with any shared release.

## 6. Source-only branch verification (2026-10-19)

The exact staged source tree was exported without the local demo changes or ignored
model/dataset assets and tested with the existing Python 3.10.11 environment:
`python -m pytest tests -q -rs` produced **302 passed, 8 skipped**. All eight skips
require v3 dataset/version/cache assets absent from the export. This checks the
published source independently of uncommitted demo code; it is not a fresh dependency
installation or a GitHub-hosted CI result. The broader local workspace test count
includes demo tests that are deliberately excluded from this branch update.
