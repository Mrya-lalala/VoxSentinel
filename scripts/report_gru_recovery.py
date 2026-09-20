"""Regenerate recovery tables/report from saved evidence; no training or extraction."""
from __future__ import annotations
import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

from scripts.gru_recovery import CORRECTED, ROOT, REFERENCE, dump, sha, preservation_inventory

RUN = Path('artifacts/runs/gru-core-v2-standardized')
REPORT = Path('docs/archive/history/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md')


def read(path): return json.loads(Path(path).read_text())

def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])

def main():
    original=read(CORRECTED/'original/evaluation.json'); new=read(RUN/'evaluation/evaluation.json')
    linear=read(CORRECTED/'linear/summary.json');ready=read(RUN/'readiness.json')
    status=read(RUN/'run_status.json');history=read(RUN/'eval_history.json')
    identity=ready['canonical_identity']; checks=read(CORRECTED/'standardization_checks.json')
    transform=read(RUN/'standardizer.json');env=read(RUN/'environment.json')
    old_linear=read('artifacts/runs/gru-core-v1-diagnosis/stage5/summary.json')
    frozen=read(CORRECTED/'preserved_originals_sha256.json');current=preservation_inventory()
    changed=[p for p,h in frozen.items() if current.get(p)!=h]
    if changed: raise AssertionError(f'Original evidence changed: {changed}')
    preservation={'checked_files':len(frozen),'changed_files':changed,'all_original_paths_present':True}
    dump(CORRECTED/'preservation_check.json',preservation)
    inventories=[]
    for p in sorted((ROOT/'staging/joint').glob('kathbath_inventory.*.jsonl')):
        counter=Counter(json.loads(line).get('gender','unknown') for line in p.open() if line.strip())
        inventories.append({'path':str(p),'sha256':sha(p),'counts':dict(counter)})
    dump(CORRECTED/'gender_inventory_evidence.json',inventories)
    result_rows=[]
    for family,report in [('Original GRU',original),('Standardized GRU',new)]:
        for key,m in report['models'].items():
            result_rows.append([family+' '+key,f"{m['cross_entropy']:.7f}",f"{m['accuracy']:.4f}",f"{m['eer']:.4f}",f"{m['auc']:.4f}",f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}",f"{m['genuine_false_alarm_rate']:.4f}",f"{m['spoof_miss_rate']:.4f}"])
    for split,m in linear['splits'].items():
        result_rows.append(['Corrected linear '+split,f"{m['cross_entropy']:.7f}",f"{m['accuracy']:.4f}",f"{m['eer']:.4f}",f"{m['auc']:.4f}",f"{m['tn']}/{m['fp']}/{m['fn']}/{m['tp']}",f"{m['genuine_false_alarm_rate']:.4f}",f"{m['spoof_miss_rate']:.4f}"])
    language_rows=[]
    coverage=identity['coverage'];languages=sorted({x['language'] for x in coverage})
    for language in languages:
        def count(split,label,generator=None):
            return sum(x['windows'] for x in coverage if x['language']==language and x['split']==split and x['label']==label and (generator is None or x['generator']==generator))
        language_rows.append([language,count('train',0),'/'.join(str(count('train',1,g)) for g in ['freevc24','xtts_v2','vits']),count('val',0),'/'.join(str(count('val',1,g)) for g in ['freevc24','xtts_v2','vits'])])
    gender_rows=[]
    for split in ('train','val'):
        for label,role in [(0,'speaker'),(1,'target'),(1,'source')]:
            d={x['gender']:x['windows'] for x in identity['gender_coverage'] if x['split']==split and x['label']==label and x['role']==role}
            gender_rows.append([split,'genuine' if label==0 else 'synthetic',role,d.get('f',0),d.get('m',0),d.get('unknown',0),d.get('not_applicable',0)])
    dist_rows=[]
    for family,report in [('Original',original),('Standardized',new)]:
        for key,m in report['models'].items():
            s,t=m['score_stats'],m['margin_stats']
            dist_rows.append([family+' '+key,f"{s['min']:.6g}–{s['max']:.6g}",f"{s['mean']:.6g}",f"{s['std']:.6g}",f"{t['min']:.6g}–{t['max']:.6g}",f"{t['mean']:.6g}"])
    changed_files=subprocess.run(['git','diff','--name-only'],capture_output=True,text=True,check=True).stdout.splitlines()
    changed_files+=subprocess.run(['git','ls-files','--others','--exclude-standard','src','scripts','configs','docs'],capture_output=True,text=True,check=True).stdout.splitlines()
    changed_files=sorted(set(changed_files+[str(REPORT)]))
    best=new['models']['best/val']
    text=f'''# Astra GRU recovery manager report

Prepared 2026-09-18. Local repository `/Users/mouryabs/VoxSentinel`.

## Verdict

The single controlled intervention succeeded on this frozen pilot: training-fitted **frame standardization before the existing LayerNorm** enabled the GRU to fit training data and improved development performance. Selected epoch **{status['selected_epoch']}**: development accuracy **{100*best['accuracy']:.2f}%**, EER **{100*best['eer']:.2f}%**, AUC **{best['auc']:.6f}**, CE **{best['cross_entropy']:.6f}**. The original selected GRU had 50% accuracy, 22.92% EER, AUC 0.824653, and CE 0.703088. The selected standardized model makes two genuine false alarms and three synthetic misses at the unchanged threshold 0.5.

The linear score-export defect and the incomplete identity audit were repaired. The linear weights/scaler were reloaded, never refitted. The canonical cross-class audit passed. Original run artifacts, diagnostics, manifests, and raw caches remain unchanged ({preservation['checked_files']} files verified by SHA-256).

This supports this intervention under seed 0 and the specified configuration. It does **not** establish input conditioning as the unique root cause, independent generalization, deployment readiness, calibrated confidence, Indian-English performance, or a reliable advantage over the linear reference. This 96-window set is development data and has informed checkpoint selection and subsequent decisions. No new dataset, encoder extraction/fine-tuning, sweep, threshold calibration, full test suite, commit, merge, or push was performed.

## 1. Scope and identities

Startup branch `feature/a1-datasets-manifests`, HEAD `884fcd7381b8409307c3b239fe9db5a22bc4798d`, clean working tree. It exactly matched the handoff's reviewed commit. The earlier report in the conversation inspected an older local state; its single-class-cache conclusion does not describe this frozen pilot.

The original run recorded `feature/indicVac2Wav` / `c250bbd0ac3b2af1ae345ce48960fcc63d200557` plus uncommitted code. Comparing all 39 recorded original-run file hashes to the reviewed commit found three differences: `.gitignore`, `README.md`, and `docs/archive/history/dataset-preparation-pilot.md`. None changes model training. This corrects the prior diagnostic report's count of two differences. All actual repairs are local and uncommitted; unrelated A1 placeholders, audio, routing, streaming and services remain untouched.

Environment: Python {env['python']}, PyTorch {env['torch']}, torchaudio {env['torchaudio']}, NumPy {env['numpy']}, Fairseq {env['fairseq']}; {env['platform']}; CPU FP32, four torch threads. Existing environment reused without installation. Encoder checkpoint identity: `26bb5ada18952fd7355f691d25927b34a0e46d6afda7658bf5c254621831ab59`; final `output_layer=None`, 1024 dimensions, 20 ms hop. `None` is not block 23. No encoder was loaded during recovery.

Frozen audio policy remains `voxsentinel-prep-2`: mono averaging, one SoXR HQ resampling pass when needed, 16 kHz FLOAT WAV, strict 1–4 second windows, recorded overflow attenuation, no denoising/loudness normalization/augmentation. The encoder's waveform normalization is unchanged and separate from downstream feature standardization.

'''
    text+=table(['Split','Manifest SHA-256','Cache bundle SHA-256'],[[s,ready['signatures'][s]['manifest_sha256'],ready['signatures'][s]['bundle_sha256']] for s in ('train','val')])
    text+='''

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

'''
    text+=f"SHA-256: `{linear['model_sha256']}`. Saved weights, bias, mean and scale were used unchanged; file hash before/after matches.\n\n"
    text+='''For raw response `m = X @ weight + bias`, logits are `[-m, m]`, logit margin is `2m`, and the correct score is `softmax(logits)[1] = sigmoid(2m)`. The old exporter used `sigmoid(m)`. The fitted objective already used the actual logits, so cross-entropy is unaffected.

'''
    ex=linear['splits']['val']['example']
    text+=f"Example `{ex['window_id']}`: old score {ex['old_score']:.9f} → corrected {ex['corrected_score']:.9f}, actual logit margin {ex['logit_margin']:.9f}. Zero decisions changed at 0.5 in either split. Training CE 0.000072797, dev CE 0.224134922, dev confusion 47/1/6/42, accuracy 92.71%, EER 4.17%, and AUC 0.988281 remain unchanged.\n\n"
    text+='''Primary ranking metrics are tie-aware AUC and interpolated EER on actual logit margins. Probability-based metrics are saved separately. Corrected float32 softmax has 296 unique training probabilities versus 384 unique margins, and 94 versus 96 on development; saturation creates ties, but does not alter these aggregate AUC/EER values here. Neither scores nor margins are calibrated confidence. No operating threshold is selected.

The linear scaler used one time-mean vector per training window and sample standard deviation (`ddof=1`). Its eight clamped dimensions are not comparable to the two clamped dimensions of the new frame-weighted population scaler. Linear pooling, LBFGS and L2 regularization also differ from the GRU; this is a diagnostic reference, not a controlled architecture comparison.

## 4. Canonical split and coverage audit

The frozen pilot has 384 training windows (192 genuine, 192 synthetic) and 96 development windows (48/48). All upstream rows are recorded as training material. Detector partitions are audited separately from upstream split membership.

Canonical speaker key = `(Kathbath origin, normalized language, normalized numeric speaker ID)`. Genuine filenames and synthetic verified parent evidence put the two classes into that common namespace. Synthetic source and target references are compared to genuine `source_file` recording identities across all roles, ignoring container extensions. Unrelated datasets are not merged by coincident numeric IDs. Cross-language person identity remains unresolved.

Results: **117 train / 37 development unique speaker keys; zero speaker overlaps; zero reference-recording overlaps across every one of the nine role pairs; zero unresolved required identities; zero within-split or cross-split prepared-audio hash duplicates.** There are 114 expected absent TTS source roles (91 train, 23 development), preserved as null/not-applicable. Known within-split genuine/synthetic sharing is expected and recorded in JSON. The audit uses recorded evidence; it does not re-query upstream datasets or rule out unknown speakers/near-duplicates.

Generator counts below use the order **FreeVC24 / XTTS-v2 / VITS**:

'''
    text+=table(['Language','Train genuine','Train synthetic F/X/V','Dev genuine','Dev synthetic F/X/V'],language_rows)
    text+='''

Totals: synthetic 127 FreeVC24, 108 XTTS-v2, 5 VITS; development 25/21/2. Every language has 16 examples per class in training and four per class in development. Eight development windows per language and two development VITS examples do not support strong subgroup conclusions. Detailed language/class/generator and language/class/gender/role counts are saved in canonical audit JSON and coverage CSVs.

Gender values describe **recorded participant identities**, not inferred or perceived generated-voice gender:

'''
    text+=table(['Split','Class','Role','Female','Male','Unknown/missing','Not applicable'],gender_rows)
    total_inventory=sum(sum(i['counts'].values()) for i in inventories)
    text+=f'''

A local scan of all 12 cached Kathbath candidate inventories found **{total_inventory:,} female / 0 male** metadata rows. This directly explains why that cached candidate pool cannot supply male genuine examples. `joint_core.py::_ensure_inventory` reads lexicographically sorted training shards progressively and stops based on speaker/recording sufficiency, with no gender-coverage criterion. `select_final_pairs` prefers speakers with genuine inventory material; `plan_language` prioritizes attached genuine material for development. These mechanisms can propagate a biased inventory into genuine and development synthetic coverage. They are a plausible selection explanation, not an upstream corpus-wide gender claim. No sampling code/data was changed.

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

N={transform['valid_frames']:,}, dimension={transform['dimension']}, training windows={transform['windows']}, clamped dimensions={transform['clamped_dimensions']}. Statistics use float64 parallel-Welford accumulation; population variance is retained in float64; applied mean/scale are serialized float32. This is frame weighting, not one-window-one-vote statistics. Development frames never fit/adjust the scaler. No per-clip or per-batch fitting occurs. Padded transformed positions are zero, and lengths/masks are unchanged.

`src/detectors/standardization.py::FrameStandardizer` stores mean, scale, variance, fitted flag and metadata with the head as nontrainable state. Config opt-in is `model.parameters.feature_standardization: true`. Historical defaults remain unchanged. Missing required transform state is rejected even with non-strict loading; mismatched enabled/disabled checkpoint construction is rejected. The generic `runner.run_training` and cache CLI both fit on training examples only. Raw caches stay immutable.

Checkpoint metadata records schema `{transform['schema']}`, placement, epsilon, weighting, variance convention, frame count, clamp count, training-cache hash and training-feature-content signature. Applied transform is also exported in `standardizer.pt` and `standardizer.json`.

The constructor's parameter tensors and post-construction RNG state exactly matched the reviewed original class under seed 0. Fitting did not consume RNG. Actual batch order and feature/label pairing were checked against all ten original seed permutations; epoch order hashes are saved in `standardization_checks.json`. Additional full-training eval-mode CE is measured after each epoch, without RNG consumption or optimizer updates, and is clearly separated from online minibatch CE.

Run status **{status['status']}**; selected epoch **{status['selected_epoch']}**; fit wall time **{status['head_training_seconds']:.3f} s**, including checkpoint writes and additional full train/dev epoch evaluations. Start {status['started']}; completion {status['finished']}. This timing is not a comparable speed benchmark against the original 43.9-second run. No technical retry or second scientific run occurred. A preliminary readiness gate flagged `.gitignore` hash divergence; inspection classified it as non-training metadata before the experiment started.

## 6. Results

CE is full-split evaluation-mode cross-entropy. Accuracy/FAR/miss/confusion use threshold 0.5; EER/AUC below use actual logit-margin ranking (probability versions also saved). Confusion order is TN/FP/FN/TP. “val” is development, not independent test data.

'''
    text+=table(['Model / split','Eval CE','Accuracy','EER','AUC','TN/FP/FN/TP','FAR','Miss'],result_rows)
    text+='''

The standardized head fits all training examples by the selected epoch. Development EER improves from 22.92% to 4.17%; accuracy improves by 44.79 percentage points at the fixed threshold. The final checkpoint further reduces training CE while increasing development CE and misses, so stronger training confidence does not imply better development behavior. Epoch 5 is selected because later epochs tie its EER; no threshold or epoch extension was tried. The linear reference has slightly higher AUC and fewer genuine false alarms, while selected GRU has fewer synthetic misses at 0.5; no significance or universal superiority claim is supported.

### Learning history

'''
    text+=table(['Epoch','Online train CE','Full train eval CE','Dev eval CE','Dev accuracy','Dev EER'],[[r['epoch'],f"{r['online_train_loss']:.6f}",f"{r['train_eval']['cross_entropy']:.6f}",f"{r['val_eval']['cross_entropy']:.6f}",f"{r['val_eval']['accuracy']:.4f}",f"{r['val_eval']['probability_eer']:.5f}"] for r in history])
    text+='''

### Score and margin distributions

'''
    text+=table(['Model / split','Score min–max','Score mean','Score std','Margin min–max','Margin mean'],dist_rows)
    text+='''

Full quantiles and per-class distributions are in `evaluation/evaluation.json` and `eval_history.json`. Per-window CSVs retain window IDs, labels, languages, generators, both logits, true logit margin and corrected score. `breakdown.json` contains selected-checkpoint language/generator errors with denominators; subgroup numbers remain descriptive only.

## 7. Bounded validation actually performed

- Original 384/96 cache/manifest identity, explicit per-item checks, all 480 prepared-file SHA-256 checks, canonical speaker/reference audit, and exact original artifact preservation checks passed.
- Injected in-memory genuine→synthetic-target leakage is detected across class/role/extension/numeric formatting. Missing required identities are reported. Within-split and cross-split duplicate scopes are distinguished. No manifest was modified for these checks.
- Corrected linear evaluation loaded original weights/scaler with no optimizer invocation; original CE/confusion reproduced and zero threshold decisions changed. Probability saturation ties were counted; margin-ranking metrics retained.
- Original best/final GRU checkpoints were reloaded and all four train/dev metric rows independently recomputed. Historical default forward behavior remains bit-exact under the checked batch.
- Unequal-length toy arithmetic checks population frame weighting and epsilon clamp. Real statistics match an independent two-pass float64 oracle on six representative dimensions. There are no standardizer parameters or buffer gradients. Best/final transform state matches the pre-training exported state exactly.
- Initialization/RNG and ten epoch batch orders match historical code. Generic runner setup was exercised with `fit` intercepted; no extra training was performed.
- Standardized best/final live-to-restored predictions match exactly. Repeated reconstruction is exact. Masked +77 padding changes logits by zero; transformed padding remains zero.
'''
    text+=f"- Batch-one versus batch-eight maximum logit differences: best {new['checks']['best']['batch1_vs_batch8_max_logit_error']:.9g}; final {new['checks']['final']['batch1_vs_batch8_max_logit_error']:.9g} (bounded tolerance 1e-5).\n"
    text+='''- Required missing transform state fails even with `strict=False`. Restoring historical checkpoints still succeeds.

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

- `docs/archive/history/ASTRA_GRU_RECOVERY_MANAGER_REPORT.md` — this report.
- `artifacts/runs/gru-core-v1-diagnosis-corrected/` — readiness, canonical audit, bounded check evidence, corrected linear predictions/metrics, re-evaluated original GRU, gender inventory evidence, preservation hashes and machine-readable recovery summary.
- `artifacts/runs/gru-core-v2-standardized/` — best/final checkpoints, transform, resolved settings, dataset/cache identities, environment, histories, selected-checkpoint predictions, complete best/final train/dev evaluation CSVs, status and log.
- New run `source_snapshot.tar.gz` — complete pre-run source/config/documentation and dependency-file contents, including untracked additions. `code_state.json` has per-file hashes and archive hash; `code_diff.patch`/`git_status.txt` preserve the dirty-tree context. `final-evidence/source_snapshot.tar.gz` additionally includes the report generator and final reporting changes; its own `code_state.json` records file/archive hashes. Hashes alone are not claimed to reconstruct missing uncommitted content.
- `artifact_hashes.json` — hashes of new supporting artifacts, excluding itself; original frozen evidence has a separate before/after hash inventory.

Checkpoint hashes:

'''
    text+=table(['Artifact','SHA-256'],[[f'gru_{n}.pt',new['checks'][n]['checkpoint_sha256']] for n in ('best','final')]+[['standardizer.pt',sha(RUN/'standardizer.pt')]])
    text+='\n\nChanged files (no unrelated code changes):\n\n'+ '\n'.join('- `'+p+'`' for p in changed_files)
    text+='''

## 9. One next recommendation

Freeze this standardized checkpoint as a **pilot reference**, then make the next phase a separately versioned coverage-and-independent-evaluation effort: build feasible gender coverage in both classes with the same participant/recording grouping rules, and reserve untouched speakers/lineage and synthesis systems before further tuning. Do not extend this experiment or sweep optimization settings: training fit has recovered, while independent generalization and source-condition shortcuts are now the larger evidence gaps. Any paired Indian-English expansion, including IFD, requires its own release/lineage audit and cannot be inferred from this Indic-language pilot.
'''
    REPORT.write_text(text)
    summary={'verdict':'training fit and development metrics improved with one controlled frame-standardization intervention',
             'original':original,'standardized':new,'linear':linear,'readiness':ready,'run_status':status,
             'transform':transform,'bounded_checks':checks,'identity_checks':read(CORRECTED/'identity_checks.json'),'final_checks':read(CORRECTED/'final_checks.json'),
             'preservation':preservation,'gender_candidate_inventories':inventories,'epoch_history':history,
             'changed_files':changed_files,'report':str(REPORT),'independent_test_performed':False}
    dump(CORRECTED/'recovery_summary.json',summary)
    for name,rows in [('coverage',identity['coverage']),('gender_by_language',identity['gender_by_language'])]:
        with (CORRECTED/f'{name}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(str(REPORT))

if __name__=='__main__': main()
