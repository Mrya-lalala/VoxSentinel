from __future__ import annotations

import math
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.audio.prepare import AudioDecodeError, decode_mono_16k
from src.audio.errors import AudioLoadError
from src.audio.identity import TRAINED_PREPROCESSING_VERSION, assert_trained_preprocessing
from src.dataset_prep.config import WindowPolicy
from src.dataset_prep.windows import select_window
from src.backbones.indic_wav2vec import (
    IndicWav2VecConfig,
    IndicWav2VecExtractor,
)


@dataclass(frozen=True)
class ChunkInference:
    """Result produced for one successfully processed audio chunk."""

    score: float
    start_sample: int
    end_sample: int


@dataclass(frozen=True)
class LoadedDetector:
    """A trained B2 detector together with its checkpoint metadata."""

    detector: torch.nn.Module
    threshold: float | None


DEFAULT_THRESHOLD = 0.5
"""Fallback decision threshold, matching EvaluationConfig.threshold."""


def resolve_threshold(
    *,
    configured: float | None = None,
    checkpoint: float | None = None,
) -> tuple[float, str]:
    """
    Resolve the effective decision threshold and its provenance.

    Precedence: an explicitly configured threshold (for example the
    ``DETECTION_THRESHOLD`` environment variable) wins over the value recorded
    in the detector checkpoint, which in turn wins over the default.
    """

    if configured is not None:
        return float(configured), "configured"

    if checkpoint is not None:
        return float(checkpoint), "checkpoint"

    return DEFAULT_THRESHOLD, "default"


class ServiceEngine:
   
    def __init__(
        self,
        *,
        encoder: IndicWav2VecExtractor,
        detector: torch.nn.Module | None,
        model_id: str,
        encoder_id: str,
        device: str | torch.device = "cpu",
        threshold: float = DEFAULT_THRESHOLD,
        threshold_source: str = "default",
        window_policy: WindowPolicy | None = None,
    ) -> None:

        self.window_policy = window_policy or WindowPolicy()
        self.preprocessing_version = TRAINED_PREPROCESSING_VERSION
        self.window_contract = "one-selected-window-v1"
        self.encoder = encoder
        self.detector = detector

        self.model_id = model_id
        self.encoder_id = encoder_id
        self.device = torch.device(device)

        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Threshold must be finite and within [0,1]")
        self.threshold = float(threshold)
        self.threshold_source = threshold_source

        if self.detector is not None:
            self.detector.to(self.device)
            self.detector.eval()

    @property
    def detector_ready(self) -> bool:
        """Whether a trained B2 detector is available."""
        return self.detector is not None

    @torch.inference_mode()
    def predict(
        self,
        audio_bytes: bytes,
        source_id: str,
    ) -> tuple[list[ChunkInference], dict[str, Any]]:
        """
        Run the complete A2 → B1 → B2 inference pipeline.

        Returns
        -------
        chunk_results:
            One spoof score for every successfully inferred chunk.

        metadata:
            Processing and coverage information used by the API.
        """

        started = time.perf_counter()

        # ---------------------------------------------------------
        # 0. Detector availability
        # ---------------------------------------------------------

        if self.detector is None:
            raise RuntimeError(
                "B2 detector checkpoint is not loaded yet."
            )

        # ---------------------------------------------------------
        # 1. A2 preprocessing
        # ---------------------------------------------------------

        with tempfile.NamedTemporaryFile(suffix=".wav") as temporary_file:
            temporary_file.write(audio_bytes)
            temporary_file.flush()
            try:
                audio = decode_mono_16k(temporary_file.name)
            except AudioDecodeError as exc:
                raise AudioLoadError("Uploaded audio could not be decoded") from exc

        window = select_window(audio.samples, audio.sample_rate, self.window_policy)
        coverage_notes = [
            f"{self.window_contract}: one highest-energy window; not whole-file coverage",
            f"preprocessing_version={self.preprocessing_version}",
            f"selected_samples=[{window.start_sample},{window.end_sample}) at 16000 Hz",
        ]
        if not window.usable:
            return [], {
                "chunks_total": 0, "chunks_used": 0,
                "processing_time_ms": self._elapsed_ms(started),
                "coverage_notes": coverage_notes + [str(window.reason)],
            }
        chunks = [window]
        chunks_total = 1
        selected = audio.samples[window.start_sample:window.end_sample].copy()
        waveforms = torch.from_numpy(selected[None]).to(self.device)
        valid_lengths_tensor = torch.tensor([selected.size], dtype=torch.int64, device=self.device)

        # ---------------------------------------------------------
        # 4. B1 IndicWav2Vec
        # ---------------------------------------------------------

        encoded = self.encoder.extract_batch(
            waveforms,
            valid_lengths_tensor,
            sample_rate=16000,
        )

        # ---------------------------------------------------------
        # 5. B1 → B2
        # ---------------------------------------------------------

        detector_batch = {
            "features": encoded.features.to(self.device),
            "valid_lengths": encoded.valid_lengths.to(
                self.device
            ),
            "padding_mask": encoded.padding_mask.to(
                self.device
            ),
        }

        output = self.detector(detector_batch)

        logits = output.logits

        # B2 has two classes:
        #
        #   class 0 = genuine
        #   class 1 = spoof
        #
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(
                "B2 detector must return logits with shape "
                f"[B, 2]. Received {tuple(logits.shape)}."
            )

        scores = torch.softmax(
            logits,
            dim=-1,
        )[:, 1]

        scores = scores.detach().float().cpu()

        if len(scores) != chunks_total:
            raise RuntimeError(
                "B2 detector returned a different number of "
                "scores than input chunks."
            )

        # ---------------------------------------------------------
        # 6. Build C1 chunk results
        # ---------------------------------------------------------

        chunk_results: list[ChunkInference] = []

        for chunk, score in zip(chunks, scores):

            score_value = float(score.item())

            if not 0.0 <= score_value <= 1.0:
                raise RuntimeError(
                    "B2 detector returned a score outside [0, 1]: "
                    f"{score_value}"
                )

            chunk_results.append(
                ChunkInference(
                    score=score_value,
                    start_sample=chunk.start_sample,
                    end_sample=chunk.end_sample,
                )
            )

        return chunk_results, {
            "chunks_total": chunks_total,
            "chunks_used": len(chunk_results),
            "processing_time_ms": self._elapsed_ms(started),
            "coverage_notes": coverage_notes,
        }

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return (
            time.perf_counter() - started
        ) * 1000.0


# ------------------------------------------------------------------
# B1 loader
# ------------------------------------------------------------------

def load_encoder(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> IndicWav2VecExtractor:
    """
    Construct and load the real B1 IndicWav2Vec encoder.

    This function is deliberately isolated so B1 configuration can
    evolve without affecting the rest of C1.
    """

    config = IndicWav2VecConfig(
        checkpoint_path=Path(checkpoint_path),
        device=str(device),
    )

    encoder = IndicWav2VecExtractor(config)

    encoder.load()

    return encoder


# ------------------------------------------------------------------
# B2 checkpoint loader
# ------------------------------------------------------------------

def _checkpoint_threshold(checkpoint: dict[str, Any]) -> float | None:
    """Read a usable operating threshold from the checkpoint, if recorded."""

    metadata = checkpoint.get("metadata")

    if isinstance(metadata, dict) and "threshold" in metadata:
        value = metadata["threshold"]
    else:
        value = checkpoint.get("threshold")

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None

    value = float(value)

    return value if math.isfinite(value) and 0.0 <= value <= 1.0 else None


def load_detector(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> LoadedDetector:
    """
    Load the trained B2 GRU detector and its checkpoint metadata.

    The detector is returned together with the decision threshold recorded in
    ``metadata["threshold"]`` (``None`` when the checkpoint stores none), so
    the service can honour the operating point selected during training.

    Legacy checkpoints without preprocessing identity emit a warning.
    Explicitly incompatible preprocessing identities fail before loading weights.
    Prefer INFERENCE_RELEASE to also verify hashes, encoder settings and windows.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"B2 detector checkpoint not found: {checkpoint_path}"
        )

    # Import B2 only when the detector is actually required.
    #
    # Keep this import isolated so the rest of C1 does not depend
    # on checkpoint-loading details.
    from src.detectors.gru import GruDetector
    

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected B2 checkpoint format. "
            "Expected a checkpoint dictionary."
        )

    assert_trained_preprocessing(
        checkpoint.get("metadata", {}).get("encoder", {}).get("preprocessing_version"),
        allow_missing=True,
    )

    # -------------------------------------------------------------
    # Locate model configuration
    # -------------------------------------------------------------

    model_config = (
        checkpoint.get("model_config")
        or checkpoint.get("config")
        or {}
    )

    if not isinstance(model_config, dict):
        raise RuntimeError(
            "B2 checkpoint contains an invalid model configuration."
        )

    # -------------------------------------------------------------
    # Construct B2 detector
    # -------------------------------------------------------------

    detector_config = type(
        "DetectorConfig",
        (),
        {
            "name": "gru",
            "parameters": model_config,
        },
    )()

    detector = GruDetector.from_config(
        detector_config
    )

    # -------------------------------------------------------------
    # Locate state dictionary
    # -------------------------------------------------------------

    state_dict = (
        checkpoint.get("model_state_dict")
        or checkpoint.get("state_dict")
        or checkpoint.get("model")
    )

    if state_dict is None:
        raise RuntimeError(
            "Could not find model weights in the B2 checkpoint."
        )

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            "B2 checkpoint model weights are not a state dictionary."
        )

    detector.load_state_dict(
        state_dict,
        strict=True,
    )

    detector.to(device)
    detector.eval()

    return LoadedDetector(
        detector=detector,
        threshold=_checkpoint_threshold(checkpoint),
    )


__all__ = [
    "ChunkInference",
    "DEFAULT_THRESHOLD",
    "LoadedDetector",
    "ServiceEngine",
    "load_encoder",
    "load_detector",
    "resolve_threshold",
]
