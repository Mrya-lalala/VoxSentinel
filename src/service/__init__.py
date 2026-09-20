"""
C1 inference service: a FastAPI wrapper around the A2 -> B1 -> B2 pipeline.

Install the service dependencies (``requirements-service.txt``) and run::

    ENCODER_CHECKPOINT=models/indicwav2vec_large.pt \
    DETECTOR_CHECKPOINT=artifacts/releases/indic-gru-pilot-v0.1/gru_best.pt \
    .venv/bin/python -m uvicorn src.service.api:app --host 0.0.0.0 --port 8000

Environment variables:
    ENCODER_CHECKPOINT   required; B1 IndicWav2Vec checkpoint file.
    DETECTOR_CHECKPOINT  optional; without it ``/detect`` answers 503.
    DETECTION_THRESHOLD  optional operator override of the decision threshold.
                         When unset, the threshold recorded in the detector
                         checkpoint metadata is used, falling back to 0.5.
    MAX_UPLOAD_BYTES     optional upload cap in bytes (default 25 MB).
    AGGREGATION_METHOD   optional chunk aggregation method (default ``mean``).
    AGGREGATION_TOP_K    optional chunk count for the ``top_k`` method
                         (default 3).
    MODEL_ID, ENCODER_ID, DEVICE
                         optional reported identifiers and inference device.
"""
