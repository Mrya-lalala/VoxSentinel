"""Bounded dataset-preparation pilot for the frozen IndicWav2Vec -> GRU pipeline.

Stages (see ``python -m scripts.prepare_datasets --help``):

* ``catalog``  - write the source-access catalog and missing-coverage states;
* ``fetch``    - inventory/acquisition/materialization per accessible source;
* ``split``    - assemble pool manifests (core train/val, supplementary, ...);
* ``features`` - frozen encoder feature cache for core windows (encoder env);
* ``report``   - build the preparation report from the manifests.

Nothing in this package runs detector training, evaluation or benchmarks.
"""

from __future__ import annotations

__all__: list[str] = []
