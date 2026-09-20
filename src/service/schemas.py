from typing import Literal, Optional

from pydantic import BaseModel, Field


class DetectResponse(BaseModel):
    """
    Frozen response contract for the VoxSentinel frontend.

    Score convention:
        0.0 -> genuine-like
        1.0 -> spoof-like

    The score is NOT a calibrated probability of fraud.
    """

    status: Literal["success", "insufficient_speech", "error"]

    score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="File-level spoof score. Higher means more spoof-like.",
    )

    decision: Optional[Literal["genuine", "spoof"]] = None

    model_id: str = Field(
        description="Identifier of the detector checkpoint/model."
    )

    encoder_id: str = Field(
        description="Identifier of the acoustic encoder."
    )

    aggregation: str = Field(
        description="Chunk aggregation method used."
    )

    threshold: Optional[float] = Field(
        default=None,
        description="Operating threshold; the research baseline uses a fixed uncalibrated 0.5.",
    )

    threshold_source: Optional[
        Literal["configured", "release", "checkpoint", "default"]
    ] = Field(
        default=None,
        description=(
            "Where the effective threshold came from: an operator override "
            "(DETECTION_THRESHOLD), the release manifest, detector checkpoint metadata, or the "
            "service default."
        ),
    )

    chunks_total: int = Field(
        ge=0,
        description="Total chunks generated from the uploaded file."
    )

    chunks_used: int = Field(
        ge=0,
        description="Chunks actually used for inference."
    )

    coverage_ratio: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of generated chunks used successfully."
    )

    processing_time_ms: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Server-side inference/processing time."
    )

    coverage_notes: list[str] = Field(
        default_factory=list,
        description="Notes about coverage, silence, skipped chunks, etc."
    )