class TogiError(Exception):
    """User-facing error. cli.main converts these to stderr + exit 1."""


class FullyTransparentError(TogiError):
    """Raised by crop_bbox when no opaque pixels remain."""
