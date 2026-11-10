"""Errors the API turns into a clear "try again" reply instead of a 500."""


class ServiceUnavailable(Exception):
    """Every provider for a required step failed (primary and fallback). The API answers
    503 with Retry-After and this message, which the page shows next to a retry button."""
