from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.detectors.gru import GruDetector
from src.audio.chunking import chunk_audio, collate_chunks
from src.audio.preprocessing import preprocess_file
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


class ServiceEngine:
   
    def __init__(
        self,
        *,
        encoder: IndicWav2VecExtractor,
        detector: torch.nn.Module | None,
        model_id: str,
        encoder_id: str,
        device: str | torch.device = "cpu",
    ) -> None:

        self.encoder = encoder
        self.detector = detector

        self.model_id = model_id
        self.encoder_id = encoder_id
        self.device = torch.device(device)

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

        # A2 currently exposes preprocess_file(), so C1 writes the
        # uploaded WAV bytes to a temporary file.
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=True,
        ) as temporary_file:

            temporary_file.write(audio_bytes)
            temporary_file.flush()

            audio = preprocess_file(
                temporary_file.name,
                source_id=source_id,
            )

        # ---------------------------------------------------------
        # 2. A2 chunking
        # ---------------------------------------------------------

        chunk_result = chunk_audio(audio)

        chunks = chunk_result.chunks
        chunks_total = len(chunks)

        coverage_notes: list[str] = []

        if chunk_result.message:
            coverage_notes.append(
                str(chunk_result.message)
            )

        if chunk_result.skipped:
            coverage_notes.extend(
                str(item)
                for item in chunk_result.skipped
            )

        if chunks_total == 0:
            return [], {
                "chunks_total": 0,
                "chunks_used": 0,
                "processing_time_ms": self._elapsed_ms(started),
                "coverage_notes": coverage_notes + [
                    "No usable speech chunks were produced."
                ],
            }

        # ---------------------------------------------------------
        # 3. A2 → tensor batch
        # ---------------------------------------------------------

        samples, valid_lengths = collate_chunks(chunks)

        waveforms = torch.from_numpy(
            np.asarray(samples, dtype=np.float32)
        ).to(self.device)

        valid_lengths_tensor = torch.from_numpy(
            np.asarray(valid_lengths, dtype=np.int64)
        ).to(self.device)

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

def load_detector(
    checkpoint_path: str | os.PathLike[str],
    *,
    device: str | torch.device = "cpu",
) -> torch.nn.Module:
    """
    Load the trained B2 GRU detector.

    IMPORTANT:
    B2 is still producing the final trained checkpoint. Therefore
    this function is intentionally isolated from the rest of C1.

    When the checkpoint becomes available, only this loader should
    need adjustment if B2's checkpoint dictionary format differs.
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
    )

    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            "Unexpected B2 checkpoint format. "
            "Expected a checkpoint dictionary."
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

    return detector


__all__ = [
    "ChunkInference",
    "ServiceEngine",
    "load_encoder",
    "load_detector",
]
