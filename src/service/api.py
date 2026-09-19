from __future__ import annotations

import os
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from .aggregator import aggregate_scores
from .engine import (
    ServiceEngine,
    load_detector,
    load_encoder,
)
from .schemas import DetectResponse


ALLOWED_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/wave",
}

MAX_UPLOAD_BYTES = int(
    os.getenv(
        "MAX_UPLOAD_BYTES",
        str(25 * 1024 * 1024),
    )
)

AGGREGATION_METHOD = os.getenv(
    "AGGREGATION_METHOD",
    "mean",
)

TOP_K = int(
    os.getenv(
        "AGGREGATION_TOP_K",
        "3",
    )
)

THRESHOLD = float(
    os.getenv(
        "DETECTION_THRESHOLD",
        "0.5",
    )
)

MODEL_ID = os.getenv(
    "MODEL_ID",
    "voxsentinel-gru",
)

ENCODER_ID = os.getenv(
    "ENCODER_ID",
    "indicwav2vec",
)

DEVICE = os.getenv(
    "DEVICE",
    "cpu",
)

ENCODER_CHECKPOINT = os.getenv(
    "ENCODER_CHECKPOINT",
)

DETECTOR_CHECKPOINT = os.getenv(
    "DETECTOR_CHECKPOINT",
)


engine: ServiceEngine | None = None
engine_error: str | None = None


def create_engine() -> ServiceEngine:
    """
    Construct the C1 inference pipeline.

    B1 is loaded immediately.

    B2 is optional until the trained detector checkpoint becomes
    available. Once B2 provides the checkpoint, only the environment
    variable DETECTOR_CHECKPOINT needs to be supplied.
    """

    if not ENCODER_CHECKPOINT:
        raise RuntimeError(
            "ENCODER_CHECKPOINT environment variable is not set."
        )

    encoder = load_encoder(
        ENCODER_CHECKPOINT,
        device=DEVICE,
    )

    detector = None

    if DETECTOR_CHECKPOINT:
        detector = load_detector(
            DETECTOR_CHECKPOINT,
            device=DEVICE,
        )

    return ServiceEngine(
        encoder=encoder,
        detector=detector,
        model_id=MODEL_ID,
        encoder_id=ENCODER_ID,
        device=DEVICE,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    global engine_error

    try:
        engine = create_engine()
        engine_error = None

    except Exception as exc:
        engine = None
        engine_error = str(exc)

    yield

    engine = None


app = FastAPI(
    title="VoxSentinel API",
    description=(
        "Indian-language voice spoof detection service."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    """
    Lightweight service health endpoint.
    """

    if engine is None:
        return {
            "status": "degraded",
            "model_id": MODEL_ID,
            "encoder_id": ENCODER_ID,
            "detector_ready": False,
        }

    return {
        "status": "ok",
        "model_id": engine.model_id,
        "encoder_id": engine.encoder_id,
        "detector_ready": engine.detector_ready,
    }


@app.post(
    "/detect",
    response_model=DetectResponse,
)
async def detect(
    file: UploadFile = File(...),
) -> DetectResponse:

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Inference engine is not initialized."
            ),
        )

    if not engine.detector_ready:
        raise HTTPException(
            status_code=503,
            detail=(
                "B2 detector checkpoint is not available yet."
            ),
        )

    content_type = (
        file.content_type or ""
    ).lower()

    filename = file.filename or ""

    if not filename.lower().endswith(".wav"):
        raise HTTPException(
            status_code=400,
            detail=(
                "Only .wav files are currently supported."
            ),
        )

    if (
        content_type
        and content_type not in ALLOWED_CONTENT_TYPES
    ):
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported audio content type. "
                "Please upload a WAV file."
            ),
        )

    audio_bytes = await file.read()

    if not audio_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    if len(audio_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                "Uploaded file exceeds the "
                f"{MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
            ),
        )

    source_id = uuid.uuid4().hex

    try:
        chunk_results, metadata = engine.predict(
            audio_bytes=audio_bytes,
            source_id=source_id,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except RuntimeError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        # Do not expose internal implementation details.
        raise HTTPException(
            status_code=500,
            detail="Inference failed.",
        ) from exc

    chunks_total = metadata[
        "chunks_total"
    ]

    chunks_used = metadata[
        "chunks_used"
    ]

    coverage_ratio = (
        chunks_used / chunks_total
        if chunks_total > 0
        else 0.0
    )

    coverage_notes = list(
        metadata.get(
            "coverage_notes",
            [],
        )
    )

    # -------------------------------------------------------------
    # No usable speech
    # -------------------------------------------------------------

    if chunks_used == 0:
        return DetectResponse(
            status="insufficient_speech",
            score=None,
            decision=None,
            model_id=engine.model_id,
            encoder_id=engine.encoder_id,
            aggregation=AGGREGATION_METHOD,
            threshold=THRESHOLD,
            chunks_total=chunks_total,
            chunks_used=0,
            coverage_ratio=0.0,
            processing_time_ms=metadata[
                "processing_time_ms"
            ],
            coverage_notes=coverage_notes + [
                "No usable speech was available "
                "for inference."
            ],
        )

    # -------------------------------------------------------------
    # C1 aggregation
    # -------------------------------------------------------------

    scores = [
        result.score
        for result in chunk_results
    ]

    try:
        aggregation = aggregate_scores(
            scores=scores,
            method=AGGREGATION_METHOD,
            top_k=TOP_K,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Invalid aggregation configuration."
            ),
        ) from exc

    file_score = aggregation.score

    # -------------------------------------------------------------
    # C1 threshold decision
    # -------------------------------------------------------------

    decision = (
        "spoof"
        if file_score >= THRESHOLD
        else "genuine"
    )

    return DetectResponse(
        status="success",
        score=file_score,
        decision=decision,
        model_id=engine.model_id,
        encoder_id=engine.encoder_id,
        aggregation=aggregation.method,
        threshold=THRESHOLD,
        chunks_total=chunks_total,
        chunks_used=chunks_used,
        coverage_ratio=coverage_ratio,
        processing_time_ms=metadata[
            "processing_time_ms"
        ],
        coverage_notes=coverage_notes,
    )


__all__ = [
    "app",
]