"""The questions every trace importer has to ask untyped JSON.

Six importers read six different wire formats, and none of them can share a
mapping — that is the point of having six. But before any of them can map
anything they all have to ask the same handful of questions about the payload
in front of them: is this a dict, is this a number, is this string short enough
to keep. Each importer had written its own answers, and by the sixth there were
five copies of ``_as_dict`` (four of them character-for-character identical),
four of ``_truncate``, two each of ``_opt_float``, ``_int`` and ``_write_op``.

What is deliberately **not** here is anything a format decides for itself.
Three importers define a ``_parse_iso`` and all three mean something different
by it — one keeps the offset, one drops it, one converts to local time to fix a
real eight-hour latency bug. A name they happen to share is not a duplication;
merging those would be a regression wearing the costume of a cleanup. They stay
apart, renamed for what each actually returns.
"""

from __future__ import annotations

from typing import Any

#: How much of one reasoning step or error message a transcript keeps.
#: Every importer had its own copy of this, all four of them 4000.
MAX_STEP_CHARS = 4000


def as_dict(value: Any) -> dict[str, Any]:
    """Coerce a tool's arguments into a dict.

    Every format records them as an object, but a malformed trace can put a
    scalar there, and a ToolCall's ``input`` is typed as a mapping.
    """
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def opt_float(value: Any) -> float | None:
    """A float, or ``None`` when the field is absent or unparseable.

    ``None`` and "0.0" are different claims — the first says the trace did not
    report a cost, the second says it reported free.
    """
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def token_count(value: Any) -> int:
    """A token count: non-negative, and 0 for anything unreadable.

    Clamped rather than passed through, because these are summed into
    :class:`~compass.core.transcript.TokenUsage` and a negative would quietly
    subtract from a run's total.
    """
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def truncate(text: str, limit: int = MAX_STEP_CHARS) -> str:
    """Cut *text* to *limit* characters, marking that it was cut."""
    return text if len(text) <= limit else text[:limit] + "…"


def write_op(result_text: str) -> str:
    """``create`` or ``update`` for a whole-file write, read off its result.

    Write is the one editing tool that does both, and its input cannot tell
    them apart — only the result says whether the file already existed. The
    check is best-effort on the CLI's wording and defaults to ``update``, which
    is true either way (the contents changed). Match on ``target`` rather than
    ``op`` when you need certainty.
    """
    return "create" if "created" in (result_text or "").lower() else "update"
