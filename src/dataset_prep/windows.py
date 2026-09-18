"""Deterministic, class-independent window selection.

Policy (recorded in the pilot config, contract version ``voxsentinel-prep-2``):

* clips shorter than ``min_seconds`` are unusable and excluded;
* clips of at most ``max_seconds`` (exactly 64,000 samples at 16 kHz) are used
  whole — there is **no grace margin**;
* longer clips contribute one contiguous ``max_seconds`` window, chosen by
  highest mean energy on a fixed hop, ties broken by the earliest offset;
* every selected window is checked for usable speech activity with a
  documented short-time frame heuristic (fraction of 30 ms frames whose RMS
  exceeds an absolute floor).  High energy alone is not proof of speech, so a
  window with too few active frames is excluded rather than silently kept.

The same policy applies to genuine and synthetic audio.  Windows are never
concatenated across gaps and padding is never written as signal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import WindowPolicy


@dataclass(frozen=True)
class WindowSelection:
    start_sample: int
    end_sample: int  # exclusive
    usable: bool
    reason: str | None
    rms: float
    peak: float
    silence_flag: bool
    active_fraction: float


def window_rms(samples: np.ndarray) -> float:
    """Root-mean-square of a float waveform (0.0 for empty)."""
    if samples.size == 0:
        return 0.0
    values = samples.astype(np.float64, copy=False)
    return float(np.sqrt(np.mean(np.square(values))))


def active_fraction(samples: np.ndarray, sample_rate: int, *, frame_ms: float, rms_threshold: float) -> float:
    """Fraction of short-time frames whose RMS exceeds ``rms_threshold``.

    A documented, class-independent activity heuristic — not a speech
    detector.  Frames are non-overlapping windows of ``frame_ms``.
    """
    frame = max(1, int(round(frame_ms * sample_rate / 1000.0)))
    total = int(samples.shape[0])
    if total < frame:
        return 0.0
    count = total // frame
    trimmed = samples[: count * frame].astype(np.float64, copy=False).reshape(count, frame)
    rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return float(np.count_nonzero(rms > rms_threshold)) / float(count)


def select_window(samples: np.ndarray, sample_rate: int, policy: WindowPolicy) -> WindowSelection:
    """Choose the prepared window within one decoded recording."""
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")
    data = np.asarray(samples, dtype=np.float32)
    total = int(data.shape[0])
    min_samples = int(round(policy.min_seconds * sample_rate))
    max_samples = int(round(policy.max_seconds * sample_rate))
    if total < min_samples:
        return WindowSelection(0, total, False, "shorter_than_min_window", 0.0, 0.0, True, 0.0)

    if total <= max_samples:
        window = data
        start = 0
    else:
        hop = max(1, int(round(policy.scan_hop_seconds * sample_rate)))
        best_start, best_energy = 0, -1.0
        for candidate in range(0, total - max_samples + 1, hop):
            segment = data[candidate : candidate + max_samples]
            energy = float(np.mean(np.square(segment.astype(np.float64, copy=False))))
            if energy > best_energy:
                best_start, best_energy = candidate, energy
        # Include the final flush window when it does not align with the hop grid.
        tail_start = total - max_samples
        if tail_start % hop != 0:
            segment = data[tail_start:]
            energy = float(np.mean(np.square(segment.astype(np.float64, copy=False))))
            if energy > best_energy:
                best_start, best_energy = tail_start, energy
        window = data[best_start : best_start + max_samples]
        start = best_start

    rms = window_rms(window)
    peak = float(np.max(np.abs(window))) if window.size else 0.0
    silence = rms < policy.silence_rms_threshold
    active = active_fraction(
        window,
        sample_rate,
        frame_ms=policy.activity_frame_ms,
        rms_threshold=policy.activity_rms_threshold,
    )
    if active < policy.min_active_fraction:
        return WindowSelection(
            start, start + int(window.shape[0]), False, "insufficient_speech_activity", rms, peak, silence, active
        )
    return WindowSelection(start, start + int(window.shape[0]), True, None, rms, peak, silence, active)


__all__ = ["WindowSelection", "active_fraction", "select_window", "window_rms"]
