# Regional frozen encoder training

Authorized bounded experiment: existing standardized GRU, frozen final-layer
IndicWav2Vec, 10 epochs, seed 0, AdamW lr .001, weight decay 0, batch 32.
Select earliest minimum dev EER at fixed threshold .5. Calibration/test reserved.
Use all frozen train/dev rows without oversampling. Report per-language dev metrics.

1. Verify version/manifests, available disk and CUDA; validate each waveform hash
   and geometry while extracting. Cache float32 embeddings on disk, resumably.
2. Fit existing standardizer on streamed unpadded TRAIN frames only; train existing
   GRU from scratch. Save compatible best/last checkpoints and provenance.
3. Verify checkpoint reload and dev predictions. Record live status and failures.

Training-only smoke verifies padding, finite gradients and checkpoint roundtrip.
No source dataset edits, downloads, encoder tuning or test scoring.
