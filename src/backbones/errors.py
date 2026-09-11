"""Structured errors for optional backbone integrations."""


class BackboneError(RuntimeError):
    """Base error raised by the backbone layer."""


class BackboneInputError(BackboneError, ValueError):
    """The audio sent to a backbone violates the common input contract."""


class BackboneLoadError(BackboneError):
    """A configured checkpoint or its runtime dependency could not be loaded."""
