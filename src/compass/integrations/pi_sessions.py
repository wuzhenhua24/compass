"""Import a pi (``@earendil-works/pi-*``) JSONL *session* file into a Compass Transcript.

pi persists an agent run as a **JSONL session tree**: the first line is a session
header, and every subsequent line is a ``SessionTreeEntry`` linked to its parent by
``parentId`` (forking/branching produces a tree). This module reads such a file
*offline* and reconstructs a Compass
:class:`~compass.core.transcript.Transcript`, so a pi run can be evaluated with
Compass's transcript-scope graders **without writing a per-agent adapter**::

    from compass.integrations import import_pi_session

    transcript = import_pi_session("~/.pi/sessions/2026-01-01-weather.jsonl")
    # transcript.tool_calls / .reasoning_steps / .outcome are ready to grade

Unlike the OpenAI Agents SDK bridge (a live ``TracingProcessor``), pi sessions are
files on disk, so this is a pure importer.

Mapping (pi -> Compass):
- session header ``{id, cwd, timestamp}``     -> ``trial_id`` / metadata / timing
- assistant message ``usage``                 -> an ``llm.generation`` ``ToolCall``
  (``TokenUsage`` + native ``CostInfo``; falls back to Compass pricing if pi did
  not record a cost)
- assistant ``toolCall`` content block         -> a ``ToolCall`` (matched to its
  ``toolResult`` message by ``toolCallId`` to fill output / status / duration)
- assistant ``thinking`` block                 -> a reasoning step
- ``user`` message                             -> ``input_prompt`` (first) / reasoning
- ``compaction`` / ``branch_summary`` / ``model_change`` / ``custom`` … -> metadata /
  reasoning

By default only the **active branch** (the path from the session's current leaf back
to the root) is reconstructed, so abandoned forks do not pollute the evaluation. For
the common linear session this is identical to file order.

Zero dependency on any pi package: the format is plain JSON, parsed directly.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from compass.adapters import calculate_cost
from compass.core.transcript import CostInfo, TokenUsage, ToolCall, Transcript

logger = logging.getLogger(__name__)

# Max characters kept for a single reasoning step (thinking blocks can be huge).
_MAX_STEP_CHARS = 4000


class PiSessionError(ValueError):
    """Raised when a file is not a recognizable pi JSONL session."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_pi_session(
    path: str | Path,
    *,
    active_branch_only: bool = True,
) -> Transcript:
    """Import a single pi JSONL session file into a Compass :class:`Transcript`.

    Args:
        path: Path to a pi ``*.jsonl`` session file.
        active_branch_only: If True (default), reconstruct only the path from the
            session's current leaf back to the root (abandoned forks are dropped).
            If False, use every entry in file (append) order.

    Returns:
        A reconstructed :class:`~compass.core.transcript.Transcript`.

    Raises:
        PiSessionError: If the file's first line is not a valid session header.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).expanduser()
    content = path.read_text(encoding="utf-8")
    header, entries = _parse_session(content, str(path))
    ordered = _active_path(entries) if active_branch_only else entries
    return _reconstruct(header, ordered, source=str(path))


def load_pi_session(
    content: str,
    *,
    source: str = "<string>",
    active_branch_only: bool = True,
) -> Transcript:
    """Reconstruct a Transcript from the raw text of a pi session file.

    Same as :func:`import_pi_session` but takes the file *content* directly, which
    is convenient for tests and in-memory pipelines.
    """
    header, entries = _parse_session(content, source)
    ordered = _active_path(entries) if active_branch_only else entries
    return _reconstruct(header, ordered, source=source)


def import_pi_sessions(
    directory: str | Path,
    *,
    pattern: str = "*.jsonl",
    active_branch_only: bool = True,
) -> list[Transcript]:
    """Import every pi session file under *directory* (non-recursive by default).

    Files that fail to parse are skipped with a logged warning rather than
    aborting the whole batch.

    Args:
        directory: Directory to scan.
        pattern: Glob pattern for session files (default ``"*.jsonl"``). Use
            ``"**/*.jsonl"`` to recurse.
        active_branch_only: Forwarded to :func:`import_pi_session`.

    Returns:
        Transcripts in sorted filename order.
    """
    directory = Path(directory).expanduser()
    recursive = "**" in pattern
    paths = sorted(glob.glob(str(directory / pattern), recursive=recursive))
    transcripts: list[Transcript] = []
    for p in paths:
        if os.path.isdir(p):
            continue
        try:
            transcripts.append(import_pi_session(p, active_branch_only=active_branch_only))
        except (PiSessionError, OSError, ValueError) as exc:
            logger.warning("Skipping unparseable pi session %s: %s", p, exc)
    return transcripts


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def _parse_session(content: str, source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split a pi session file into (header, entries).

    The header line is validated; malformed entry lines are skipped with a warning
    so a single corrupt line does not sink the whole import.
    """
    lines = [ln for ln in content.split("\n") if ln.strip()]
    if not lines:
        raise PiSessionError(f"{source}: empty session file")

    try:
        header = json.loads(lines[0])
    except (ValueError, TypeError) as exc:
        raise PiSessionError(f"{source}: first line is not valid JSON") from exc
    if not isinstance(header, dict) or header.get("type") != "session":
        raise PiSessionError(f"{source}: first line is not a pi session header")

    entries: list[dict[str, Any]] = []
    for i, line in enumerate(lines[1:], start=2):
        try:
            entry = json.loads(line)
        except (ValueError, TypeError):
            logger.warning("%s: line %d is not valid JSON, skipping", source, i)
            continue
        if not isinstance(entry, dict) or "type" not in entry or "id" not in entry:
            logger.warning("%s: line %d is not a valid session entry, skipping", source, i)
            continue
        entries.append(entry)
    return header, entries


def _leaf_id_after(entry: dict[str, Any]) -> str | None:
    """Mirror pi's ``leafIdAfterEntry``: a leaf entry redirects, others become the leaf."""
    if entry.get("type") == "leaf":
        return entry.get("targetId")
    return entry.get("id")


def _active_path(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return entries along the active branch (root -> current leaf), in order.

    Falls back to file order if the leaf pointer is missing/broken or the walk
    surfaces no message entries.
    """
    if not entries:
        return entries

    by_id = {e["id"]: e for e in entries}
    leaf_id = _leaf_id_after(entries[-1])
    if not leaf_id or leaf_id not in by_id:
        return entries

    path: list[dict[str, Any]] = []
    seen: set[str] = set()
    current: dict[str, Any] | None = by_id.get(leaf_id)
    while current is not None and current["id"] not in seen:
        seen.add(current["id"])
        path.append(current)
        parent_id = current.get("parentId")
        current = by_id.get(parent_id) if parent_id else None
    path.reverse()

    # Guard: a broken chain that dropped all messages -> prefer full file order.
    if not any(e.get("type") == "message" for e in path):
        return entries
    return path


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct(
    header: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    source: str,
) -> Transcript:
    session_id = str(header.get("id") or "pi_session")
    cwd = header.get("cwd", "")

    transcript = Transcript(task_id=session_id, trial_id=session_id)
    transcript.input_params = {"cwd": cwd, "source": source}
    transcript.metadata = {
        "importer": "pi",
        "session_id": session_id,
        "cwd": cwd,
        "session_version": header.get("version"),
    }
    if header.get("parentSession"):
        transcript.metadata["parent_session"] = header["parentSession"]

    # toolCallId -> the ToolCall awaiting its result.
    pending: dict[str, ToolCall] = {}
    last_assistant_text: str | None = None
    last_model: str | None = None
    turn = 0  # each assistant message starts a new turn

    for entry in entries:
        etype = entry.get("type")
        if etype == "message":
            message = entry.get("message") or {}
            if message.get("role") == "assistant":
                turn += 1
            text, model = _handle_message(transcript, message, pending, turn)
            if text is not None:
                last_assistant_text = text
            if model:
                last_model = model
        elif etype == "model_change":
            transcript.metadata.setdefault("model_changes", []).append(
                {"provider": entry.get("provider"), "modelId": entry.get("modelId")}
            )
        elif etype == "active_tools_change":
            transcript.metadata["active_tools"] = entry.get("activeToolNames")
        elif etype == "thinking_level_change":
            transcript.metadata["thinking_level"] = entry.get("thinkingLevel")
        elif etype == "compaction":
            summary = _truncate(str(entry.get("summary", "")))
            transcript.add_reasoning_step(f"[compaction] {summary}")
            transcript.metadata.setdefault("compactions", []).append(
                {"summary": summary, "tokens_before": entry.get("tokensBefore")}
            )
        elif etype == "branch_summary":
            summary = _truncate(str(entry.get("summary", "")))
            transcript.add_reasoning_step(f"[branch_summary] {summary}")
        elif etype in ("custom", "custom_message"):
            ctype = entry.get("customType", "custom")
            body = _truncate(_content_to_text(entry.get("content", "")))
            transcript.add_reasoning_step(f"[{ctype}] {body}")
        elif etype == "session_info":
            name = entry.get("name")
            if name:
                transcript.task_id = str(name)
                transcript.metadata["session_name"] = name
        # "label" / "leaf" are structural — ignored.

    _finalize(transcript, header, entries, last_assistant_text, last_model)
    return transcript


def _handle_message(
    transcript: Transcript,
    message: dict[str, Any],
    pending: dict[str, ToolCall],
    turn_index: int,
) -> tuple[str | None, str | None]:
    """Map one pi AgentMessage onto the transcript.

    Returns ``(assistant_text, model)`` for outcome/environment bookkeeping; both
    are ``None`` for non-assistant messages.
    """
    role = message.get("role")
    if role == "user":
        text = _content_to_text(message.get("content", ""))
        if not transcript.input_prompt:
            transcript.input_prompt = text
        elif text:
            transcript.add_reasoning_step(f"[user] {_truncate(text)}")
        return None, None

    if role == "assistant":
        return _handle_assistant(transcript, message, pending, turn_index)

    if role == "toolResult":
        _handle_tool_result(transcript, message, pending)
        return None, None

    return None, None


def _handle_assistant(
    transcript: Transcript,
    message: dict[str, Any],
    pending: dict[str, ToolCall],
    turn_index: int,
) -> tuple[str | None, str | None]:
    ts = _ms_to_epoch(message.get("timestamp"))
    model = message.get("model")
    provider = message.get("provider")
    stop_reason = message.get("stopReason")
    is_error = stop_reason in ("error", "aborted")

    error: dict[str, Any] | None = None
    if is_error:
        error = {"message": message.get("errorMessage") or stop_reason, "stop_reason": stop_reason}

    # 1) The model turn itself -> an llm.generation ToolCall (tokens + cost).
    tokens, cost = _usage_to_tokens_cost(message.get("usage"), model)
    text_parts: list[str] = []
    tool_blocks: list[dict[str, Any]] = []
    for block in message.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(block.get("text", ""))
        elif btype == "thinking":
            thinking = block.get("thinking", "")
            if thinking:
                transcript.add_reasoning_step(f"[thinking] {_truncate(thinking)}")
        elif btype == "toolCall":
            tool_blocks.append(block)

    assistant_text = "\n".join(p for p in text_parts if p).strip() or None

    llm_meta: dict[str, Any] = {}
    if model:
        llm_meta["model"] = model
    if provider:
        llm_meta["provider"] = provider
    if message.get("api"):
        llm_meta["api"] = message["api"]

    transcript.tool_calls.append(
        ToolCall(
            tool_name="llm.generation",
            input=_llm_input(message),
            output=assistant_text,
            status="error" if is_error else "ok",
            timestamp=ts,
            error=error,
            cost=cost,
            tokens=tokens,
            tool_type="llm",
            metadata=llm_meta,
            turn_index=turn_index,
        )
    )

    # 2) Each requested tool call -> a ToolCall awaiting its result.
    for block in tool_blocks:
        call_id = str(block.get("id") or "")
        kwargs: dict[str, Any] = dict(
            tool_name=block.get("name") or "unknown_tool",
            input=_as_dict(block.get("arguments")),
            status="ok",
            timestamp=ts,
            tool_type="function",
            metadata={"call_ts": ts},
            turn_index=turn_index,
        )
        if call_id:  # else let ToolCall generate one
            kwargs["call_id"] = call_id
        tc = ToolCall(**kwargs)
        transcript.tool_calls.append(tc)
        if call_id:
            pending[call_id] = tc

    return assistant_text, model


def _handle_tool_result(
    transcript: Transcript,
    message: dict[str, Any],
    pending: dict[str, ToolCall],
) -> None:
    call_id = str(message.get("toolCallId") or "")
    result_text = _content_to_text(message.get("content", ""))
    is_error = bool(message.get("isError"))
    result_ts = _ms_to_epoch(message.get("timestamp"))
    details = message.get("details")

    tc = pending.pop(call_id, None)
    if tc is None:
        # Orphan result (no matching call in this branch) — keep it anyway.
        kwargs: dict[str, Any] = dict(
            tool_name=message.get("toolName") or "unknown_tool",
            output=result_text,
            status="error" if is_error else "ok",
            timestamp=result_ts,
            error={"message": _truncate(result_text, 500)} if is_error else None,
            tool_type="function",
        )
        if call_id:
            kwargs["call_id"] = call_id
        transcript.tool_calls.append(ToolCall(**kwargs))
        return

    tc.output = result_text
    tc.status = "error" if is_error else "ok"
    if is_error:
        tc.error = {"message": _truncate(result_text, 500)}
    if details is not None:
        tc.metadata["details"] = details
    call_ts = tc.metadata.pop("call_ts", None)
    if call_ts and result_ts:
        tc.duration_ms = max(0.0, (result_ts - call_ts) * 1000.0)


def _finalize(
    transcript: Transcript,
    header: dict[str, Any],
    entries: list[dict[str, Any]],
    last_assistant_text: str | None,
    last_model: str | None,
) -> None:
    if last_model:
        transcript.environment.model_version = last_model

    # Timing: prefer the session header start + last message timestamp.
    start = _parse_iso(header.get("timestamp"))
    end_ts = _last_message_epoch(entries)
    if start:
        transcript.start_time = start
        if end_ts:
            end = datetime.fromtimestamp(end_ts)
            if end >= start:
                transcript.end_time = end
                transcript.total_duration_ms = (end - start).total_seconds() * 1000.0

    # Best-effort outcome: the final assistant text.
    if last_assistant_text:
        transcript.set_outcome(
            output_data={"final_output": last_assistant_text},
            metadata={"session_id": transcript.metadata.get("session_id")},
        )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _usage_to_tokens_cost(
    usage: Any, model: str | None
) -> tuple[TokenUsage | None, CostInfo | None]:
    """Map pi ``Usage`` to Compass ``TokenUsage`` + ``CostInfo``.

    pi records native dollar costs; use them when present, otherwise fall back to
    Compass's configurable pricing table.
    """
    if not isinstance(usage, dict):
        return None, None

    in_tok = int(usage.get("input", 0) or 0)
    out_tok = int(usage.get("output", 0) or 0)
    total = int(usage.get("totalTokens", 0) or 0) or (in_tok + out_tok)
    token_meta: dict[str, Any] = {}
    for key in ("cacheRead", "cacheWrite", "reasoning"):
        if usage.get(key):
            token_meta[key] = usage[key]
    tokens = TokenUsage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=total,
        metadata=token_meta,
    )

    cost: CostInfo | None = None
    native = usage.get("cost")
    if isinstance(native, dict) and native.get("total"):
        cost = CostInfo(
            total_usd=float(native.get("total", 0.0)),
            input_cost_usd=_opt_float(native.get("input")),
            output_cost_usd=_opt_float(native.get("output")),
            metadata={"model": model, "source": "pi"} if model else {"source": "pi"},
        )
    elif model:
        cost = calculate_cost(model, in_tok, out_tok)

    return tokens, cost


def _llm_input(message: dict[str, Any]) -> dict[str, Any]:
    """Compact description of the model turn's request context."""
    data: dict[str, Any] = {}
    if message.get("model"):
        data["model"] = message["model"]
    if message.get("provider"):
        data["provider"] = message["provider"]
    if message.get("stopReason"):
        data["stop_reason"] = message["stopReason"]
    return data


def _content_to_text(content: Any) -> str:
    """Flatten a pi message content (str or content-block list) to plain text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
                elif block.get("type") == "image":
                    parts.append("[image]")
                elif "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return ""


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce tool arguments into a dict (pi records them as an object)."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def _ms_to_epoch(ms: Any) -> float:
    """Convert a pi Unix-millisecond timestamp to epoch seconds."""
    try:
        return float(ms) / 1000.0
    except (TypeError, ValueError):
        return 0.0


def _last_message_epoch(entries: list[dict[str, Any]]) -> float:
    for entry in reversed(entries):
        if entry.get("type") == "message":
            ts = _ms_to_epoch((entry.get("message") or {}).get("timestamp"))
            if ts:
                return ts
    return 0.0


def _parse_iso(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truncate(text: str, limit: int = _MAX_STEP_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
