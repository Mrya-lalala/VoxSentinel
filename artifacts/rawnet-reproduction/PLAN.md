# RawNet3 performance reproduction

The user explicitly set the IFD-only target to at least 94% accuracy and
at most 6.1% EER on 2026-11-25, with further improvement if feasible.
The Indian-language target is at least 95% accuracy and EER below 5%.

1. Use official Jungjee/RawNet source pinned in upstream-provenance.json.
   Keep its MIT license. Install asteroid-filterbanks 0.4.0 locally.
2. Build a versioned five-second mono 16 kHz waveform cache from the
   existing IFD archive index's training folders. Trim long recordings from
   the beginning and repeat short recordings as described by the IFD paper.
   Use the existing 200-clip development manifest. Hash inputs and exclude
   duplicate originals/PCM across train and development.
3. Use official RawNet3 defaults (C=1024, Res2Net scale 8, contextual ECA
   pooling, summed blocks, learned sinc filterbank). Set nOut=1 for binary
   logits. The paper does not specify this final projection detail.
4. Train from scratch for 50 epochs with Adam lr=1e-3, weight_decay=5e-5,
   BCEWithLogitsLoss. Attempt batch 64, using smaller microbatches and
   accumulation if VRAM requires it. Record that BatchNorm statistics use
   the microbatch size when accumulation is used.
5. Select by development EER and report fixed-threshold accuracy and counts.
   Verify checkpoint reload scores. Report comparison to 94% / 6.1% with
   split differences. Keep the existing separate calibration/test folders.

The published paper's exact split files and detector implementation were
not found. Our archive's folder/class counts differ from its test-set table.
This run reproduces the documented model family and optimization recipe
on an explicitly recorded speaker-disjoint partition.

Sources

- https://arxiv.org/html/2506.19014v2
- https://github.com/Jungjee/RawNet

## Post-run operating-point check

The 50-epoch fixed-threshold reproduction reached 6% EER and 93.5%
accuracy. Check one prior-shift adjustment after training. Derive the raw
sigmoid threshold from the retained training spoof fraction, targeting the
benchmark's predeclared 50/50 class mixture. This uses training counts to
choose the threshold. Keep the original threshold-0.5 result in the report.
