"""Text helpers for rendering summaries."""


def truncate(text: str, limit: int) -> str:
    """Cut ``text`` down to at most ``limit`` characters.

    Known defect: it slices mid-word and gives the reader no sign that
    anything was removed.
    """
    return text[:limit]
