# VoxSentinel dataset-preparation pilot report

Generated: 2026-09-18T07:44:49.557467+00:00
Checkout: branch `feature/indicVac2Wav`, commit `c250bbd0ac3b2af1ae345ce48960fcc63d200557` (17 uncommitted files preserved)
Dataset root: `artifacts/datasets` (config `configs/datasets.yaml`)
Seed: 42

## Pools (actual window counts)

| Pool | Windows | File |
| --- | ---: | --- |
| core_train | 384 | `manifests/windows.core_train.jsonl` |
| core_val | 96 | `manifests/windows.core_val.jsonl` |
| supplementary | 10 | `manifests/windows.supplementary.jsonl` |
| external_eval | 20 | `manifests/windows.external_eval.jsonl` |
| fallback_baseline | 50 | `manifests/windows.fallback_baseline.jsonl` |
| unpaired_candidate | 40 | `manifests/windows.unpaired_candidate.jsonl` |
| excluded | 8853 | `manifests/windows.excluded.jsonl` |
| missing_coverage | 4 | `manifests/windows.missing_coverage.jsonl` |

## Core windows by dataset / language / split

| Group | Windows | Minutes | Speakers | Generators | Labels |
| --- | ---: | ---: | ---: | --- | --- |
| indicsynth|Bengali|train | 16 | 1.067 | 8 | freevc24:6, vits:3, xtts_v2:7 | spoof:16 |
| indicsynth|Bengali|val | 4 | 0.267 | 3 | freevc24:2, vits:2 | spoof:4 |
| indicsynth|Gujarati|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Gujarati|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Hindi|train | 16 | 1.064 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Hindi|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Kannada|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Kannada|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Malayalam|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Malayalam|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Marathi|train | 16 | 1.061 | 12 | freevc24:16 | spoof:16 |
| indicsynth|Marathi|val | 4 | 0.267 | 4 | freevc24:4 | spoof:4 |
| indicsynth|Odia|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Odia|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Punjabi|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Punjabi|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Sanskrit|train | 16 | 1.067 | 9 | freevc24:7, xtts_v2:9 | spoof:16 |
| indicsynth|Sanskrit|val | 4 | 0.267 | 3 | freevc24:1, xtts_v2:3 | spoof:4 |
| indicsynth|Tamil|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Tamil|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Telugu|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Telugu|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| indicsynth|Urdu|train | 16 | 1.067 | 9 | freevc24:8, xtts_v2:8 | spoof:16 |
| indicsynth|Urdu|val | 4 | 0.267 | 3 | freevc24:2, xtts_v2:2 | spoof:4 |
| kathbath|Bengali|train | 16 | 1.067 | 5 | - | genuine:16 |
| kathbath|Bengali|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Gujarati|train | 16 | 1.067 | 5 | - | genuine:16 |
| kathbath|Gujarati|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Hindi|train | 16 | 1.067 | 6 | - | genuine:16 |
| kathbath|Hindi|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Kannada|train | 16 | 1.067 | 4 | - | genuine:16 |
| kathbath|Kannada|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Malayalam|train | 16 | 1.067 | 4 | - | genuine:16 |
| kathbath|Malayalam|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Marathi|train | 16 | 1.067 | 6 | - | genuine:16 |
| kathbath|Marathi|val | 4 | 0.267 | 4 | - | genuine:4 |
| kathbath|Odia|train | 16 | 1.067 | 9 | - | genuine:16 |
| kathbath|Odia|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Punjabi|train | 16 | 1.067 | 5 | - | genuine:16 |
| kathbath|Punjabi|val | 4 | 0.263 | 3 | - | genuine:4 |
| kathbath|Sanskrit|train | 16 | 1.067 | 7 | - | genuine:16 |
| kathbath|Sanskrit|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Tamil|train | 16 | 1.067 | 4 | - | genuine:16 |
| kathbath|Tamil|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Telugu|train | 16 | 1.067 | 5 | - | genuine:16 |
| kathbath|Telugu|val | 4 | 0.267 | 3 | - | genuine:4 |
| kathbath|Urdu|train | 16 | 1.067 | 4 | - | genuine:16 |
| kathbath|Urdu|val | 4 | 0.267 | 3 | - | genuine:4 |

## Core windows by generator / split

| Group | Windows | Minutes | Speakers | Generators | Labels |
| --- | ---: | ---: | ---: | --- | --- |
| indicsynth|freevc24|train | 101 | 6.727 | 76 | freevc24:101 | spoof:101 |
| indicsynth|freevc24|val | 25 | 1.667 | 26 | freevc24:25 | spoof:25 |
| indicsynth|vits|train | 3 | 0.2 | 1 | vits:3 | spoof:3 |
| indicsynth|vits|val | 2 | 0.133 | 1 | vits:2 | spoof:2 |
| indicsynth|xtts_v2|train | 88 | 5.864 | 33 | xtts_v2:88 | spoof:88 |
| indicsynth|xtts_v2|val | 21 | 1.4 | 10 | xtts_v2:21 | spoof:21 |
| kathbath|None|train | 192 | 12.8 | 64 | - | genuine:192 |
| kathbath|None|val | 48 | 3.196 | 37 | - | genuine:48 |

## Unpaired candidate windows by native-language group / split

| Group | Windows | Minutes | Speakers | Generators | Labels |
| --- | ---: | ---: | ---: | --- | --- |
| nisp|Hindi|train | 6 | 0.4 | 3 | - | genuine:6 |
| nisp|Hindi|val | 2 | 0.133 | 2 | - | genuine:2 |
| nisp|Kannada|train | 6 | 0.4 | 3 | - | genuine:6 |
| nisp|Kannada|val | 2 | 0.133 | 2 | - | genuine:2 |
| nisp|Malayalam|train | 6 | 0.4 | 3 | - | genuine:6 |
| nisp|Malayalam|val | 2 | 0.133 | 2 | - | genuine:2 |
| nisp|Tamil|train | 6 | 0.4 | 3 | - | genuine:6 |
| nisp|Tamil|val | 2 | 0.133 | 2 | - | genuine:2 |
| nisp|Telugu|train | 6 | 0.382 | 3 | - | genuine:6 |
| nisp|Telugu|val | 2 | 0.133 | 2 | - | genuine:2 |

## Fallback/baseline windows by official split / label

| Group | Windows | Minutes | Speakers | Generators | Labels |
| --- | ---: | ---: | ---: | --- | --- |
| dev|genuine | 5 | 0.329 | 5 | - | genuine:5 |
| dev|spoof | 5 | 0.237 | 5 | - | spoof:5 |
| train|genuine | 20 | 1.032 | 20 | - | genuine:20 |
| train|spoof | 20 | 1.048 | 20 | - | spoof:20 |

## Download usage

Total new downloads: 3.37 GiB of 5.00 GiB cap
- asvspoof2019: 15.52 MiB
- indicsynth: 236.27 MiB
- kathbath: 1.61 GiB
- nisp: 1.21 GiB
- nptel: 180.07 MiB
- svarah: 135.87 MiB

## Source access states

| Source | Status | Role |
| --- | --- | --- |
| indicsynth | accessible | core synthetic windows (12 Indic languages) |
| nisp | accessible | genuine Indian-English candidate windows (unpaired without synthetic English) |
| asvspoof2019 | accessible_selective | separate fallback/baseline pool (20+20 train, 5+5 dev) |
| nptel | accessible_bounded | supplementary genuine English windows (lecturer identity unresolved) |
| kathbath | accessible | core genuine windows (12 Indic languages) - PROCESSED |
| svarah | accessible | external evaluation only (20 genuine English windows PROCESSED) |
| indicvoices | accessible_not_processed | external evaluation only (genuine Indic languages) |
| spire_sies | awaiting_user_links | supplementary genuine English |
| indic_timit | awaiting_user_links | supplementary genuine English |
| synthetic_english | missing_source | required pairing for NISP genuine English - NOT PREPARED |

## Readiness

- `core_indic_training_ready`: ready=True (ready=True, languages=12, train_windows=384, val_windows=96)
- `indian_english_training_ready`: ready=False (ready=False)
  - reason: No traceable synthetic Indian-English source exists and no user assets were supplied.
  - user action: Provide user-authorized synthetic assets following configs/synthetic_english_template.json.
- `external_audio_ready`: ready=True (ready=True, windows=20, sources=['svarah'])

## Core composition (identity & cross-split disjointness)

- windows: train 384, val 96
- unique speakers: train 174, val 74
- unique upstream originals: train 224, val 61
- cross-split speaker overlap: 0; reference overlap: 0
  - Bengali: train 16G/16S, val 4G/4S
  - Gujarati: train 16G/16S, val 4G/4S
  - Hindi: train 16G/16S, val 4G/4S
  - Kannada: train 16G/16S, val 4G/4S
  - Malayalam: train 16G/16S, val 4G/4S
  - Marathi: train 16G/16S, val 4G/4S
  - Odia: train 16G/16S, val 4G/4S
  - Punjabi: train 16G/16S, val 4G/4S
  - Sanskrit: train 16G/16S, val 4G/4S
  - Tamil: train 16G/16S, val 4G/4S
  - Telugu: train 16G/16S, val 4G/4S
  - Urdu: train 16G/16S, val 4G/4S

## Dataset revisions (immutable references)

- asvspoof2019: `as-published...` x50
- indicsynth: `c0a10386b723717a...` x240
- kathbath: `5b9e92849222026d...` x240
- nisp: `master...` x40
- nptel: `v0.1...` x10
- svarah: `ebbf7777fe771490...` x20

## Lineage resolution (full train-shard scan)

- synthetic windows: 240
- verification statuses: {"not_applicable_tts": 114, "verified_train": 366}
- evidence methods: {"full_train_shard_scan": 236, "none": 114, "scanned_inventory": 130}

## Preprocessing audit (voxsentinel-prep-2)

- windows at the strict 4 s cap: 540; shorter (whole recordings under the cap): 60
- active fraction: min 0.358491, mean 0.9175
- decoders: libsndfile:600
- overflow attenation applied (bounded, ≤ 0 dB): spoof:51, genuine:6

## Feature cache

- `train`: 384 items, status built, encoder 26bb5ada1895..., layer None, dim 1024
- `val`: 96 items, status built, encoder 26bb5ada1895..., layer None, dim 1024
- training-cache audit: ready=True; skipped: train:deep_identity, val:deep_identity

## Missing coverage (not concealed)

- spire_sies (all languages/classes): awaiting_user_links - SPIRE portal is request-based; the user reported the archive as unavailable for this session.
  - user action: Provide a local SPIRE-SIES archive path in a follow-up run (portal: https://spiredatasets.ee.iisc.ac.in/).
- indic_timit (all languages/classes): awaiting_user_links - SPIRE portal is request-based; the user reported the archive as unavailable for this session.
  - user action: Provide a local Indic TIMIT release path in a follow-up run (portal: https://spiredatasets.ee.iisc.ac.in/indictimitcorpus).
- indicvoices (all languages/classes): accessible_not_processed - The HF gate was accepted during this run and file reads succeed, but the bounded scope selected Svarah for new external-evaluation material.
  - user action: Optional: extend the external evaluation pool with IndicVoices shards in a follow-up run.
- synthetic_english (all languages/classes): missing_source - No traceable synthetic Indian-English source exists and no user assets were supplied.
  - user action: Provide user-authorized synthetic assets following configs/synthetic_english_template.json.

## Exclusions

- candidate_pool_not_selected: 8768
- reference_unresolvable_precheck: source:not_found: 40
- reference_unresolvable_precheck: target:not_found: 27
- lineage_replacement_required: not_found: 12
- reference_unresolvable_precheck: source:not_found, target:not_found: 5
- audio_fetch_missing: 1

## Commands actually executed

```
python -m scripts.prepare_datasets catalog
python -m scripts.prepare_datasets fetch --source indicsynth --languages Malayalam
python -m scripts.prepare_datasets fetch --source indicsynth --languages Malayalam
python -m scripts.prepare_datasets fetch --source indicsynth --languages Malayalam
python -m scripts.prepare_datasets fetch --source indicsynth --languages Bengali Gujarati Hindi Kannada Marathi Odia Punjabi Sanskrit Tamil Telugu Urdu
python -m scripts.prepare_datasets fetch --source indicsynth --languages Telugu
python -m scripts.prepare_datasets fetch --source nisp
python -m scripts.prepare_datasets fetch --source nisp
python -m scripts.prepare_datasets fetch --source asvspoof2019
python -m scripts.prepare_datasets fetch --source nptel
python -m scripts.prepare_datasets fetch --source synthetic_english
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets features --threads 4
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets status
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets features --threads 4 --force
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets status
python -m scripts.prepare_datasets core --plan-only --languages Malayalam
python -m scripts.prepare_datasets core --plan-only --languages Malayalam
python -m scripts.prepare_datasets core --plan-only --languages Malayalam
python -m scripts.prepare_datasets core --plan-only --languages Malayalam
python -m scripts.prepare_datasets core --languages Malayalam
python -m scripts.prepare_datasets core --languages Malayalam
python -m scripts.prepare_datasets fetch --source svarah
python -m scripts.prepare_datasets core --languages Bengali Gujarati Hindi Kannada Marathi Odia Punjabi Sanskrit Tamil Telugu Urdu
python -m scripts.prepare_datasets core --languages Malayalam Sanskrit
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets catalog
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets features --force --threads 4
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets core --plan-only --languages Bengali Gujarati Kannada Marathi Odia Sanskrit
python -m scripts.prepare_datasets core --plan-only --languages Sanskrit
python -m scripts.prepare_datasets core --languages Bengali Gujarati Kannada Marathi Odia Sanskrit
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets core --languages Bengali Kannada Sanskrit
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets core --languages Sanskrit
python -m scripts.prepare_datasets promote
python -m scripts.prepare_datasets split
python -m scripts.prepare_datasets report
python -m scripts.prepare_datasets features --force --threads 4
```

No tests, detector training or evaluation were run.
