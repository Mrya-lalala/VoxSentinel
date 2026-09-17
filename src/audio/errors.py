"""Structured errors raised by the A2 audio preprocessing boundary."""


class AudioError(RuntimeError):
    """Base error raised by the audio preprocessing layer."""


class AudioValidationError(AudioError, ValueError):
    """Audio violates the A2 input contract (rank, dtype, rate, range, ...)."""


class AudioLoadError(AudioError):
    """A file could not be decoded or its contents are unsupported."""
