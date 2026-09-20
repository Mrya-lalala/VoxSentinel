# Feature-branch release checks

20 September 2026. Research handoff only; no production promotion or merge to main.

- Full local suite: 98 passed, three existing upstream weight-norm deprecation warnings.
- Clean source copy, with no training datasets/caches/models present: 93 passed, five expected artifact-dependent skips.
- Dependency check: no broken requirements. Setup script syntax valid; inference resampler is now included by setup.
- Clean-source CLI integration: extracted the exact handoff bundle into the clean source copy, verified its checksums, supplied the existing external encoder, and predicted one known development recording. Maximum logit difference from the saved development prediction: 3.5763e-6, within 1e-5 tolerance. No benchmark was scored.
- The clean-source tests used the existing pinned Python environment. A fresh dependency installation and other operating systems were not tested in this release check.
- Weights and audio are not included in the commit. The separate 7.3 MB research archive and its checksum are documented in `TEAMMATE_RESEARCH_BASELINE.md`; the repository owner must share it with authorized teammates. No public weights release was created.
- Staged code/docs were checked for obvious token/private-key/signed-URL patterns; no matches found in the checked source paths. Existing historical tracked acquisition metadata remains historical provenance.

Current model: epoch-three v3 research head; development accuracy 97.04%, EER 2.30%. V3 benchmark performance remains unmeasured. The historical v2 benchmark result must not be relabeled as a v3 result.
