"""Reconstruct a Compass Transcript from a claude-agent-sdk ``Message`` stream.

The Claude Agent SDK (``claude_agent_sdk``) has a distinctive architecture: it
does **not** run the agent loop in-process — it drives the Claude Code CLI as a
subprocess and hands back a typed, Anthropic-shaped ``Message`` stream
(``query()`` / ``ClaudeSDKClient.receive_response()``). That stream is the SDK's
stable public contract, so it's the right thing to ingest (rather than the CLI's
deliberately-internal on-disk JSONL).

This module turns a collected list — or an async stream — of those messages into
a Compass :class:`~compass.core.transcript.Transcript`, so an agent built on the
SDK can be graded with Compass's transcript-scope graders **without a per-agent
adapter**::

    from claude_agent_sdk import query
    from compass.integrations import reconstruct_transcript

    messages = [m async for m in query(prompt="...")]
    transcript = reconstruct_transcript(messages)   # ready to grade

There is also an **offline** path for the CLI's ``--output-format stream-json``
output (one JSON wire-dict per line), which needs no SDK import at all::

    from compass.integrations import import_claude_stream_json

    transcript = import_claude_stream_json("run.stream.jsonl")

Mapping (SDK message -> Compass):
- ``AssistantMessage``            -> an ``llm.generation`` ``ToolCall``
  (``TokenUsage`` from ``usage``; ``turn_index`` per assistant message)
- ``ToolUseBlock``                -> a ``ToolCall`` (matched to its result by
  ``tool_use_id``)
- Claude Code returns tool *results* as a ``UserMessage`` carrying a
  ``ToolResultBlock`` — those fill the pending call's output/status (the
  SDK's characteristic "tool result is a user-role message" convention)
- ``ThinkingBlock``               -> a reasoning step
- plain-string ``UserMessage``    -> ``input_prompt`` (first) / reasoning
- ``ServerToolUseBlock`` / ``ServerToolResultBlock`` (web_search, web_fetch,
  advisor …) -> a server-side ``ToolCall`` (executed by the API)
- ``ResultMessage``               -> outcome (``result``), duration, and the
  **CLI-computed** ``total_cost_usd`` (attached to the final LLM call, since the
  CLI reports one authoritative total rather than per-call dollars)
- ``parent_tool_use_id`` + ``Task`` calls -> sub-agent attribution
  (``agent_name`` first-class field)

Zero hard dependency on ``claude_agent_sdk``: messages are read by duck-typing
their attributes, so this works on collected SDK objects without importing them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from compass.core.transcript import CostInfo, TokenUsage, ToolCall, Transcript

logger = logging.getLogger(__name__)

# Server-side tool names whose calls map to a "search" tool_type.
_SEARCH_TOOLS = frozenset({"web_search", "web_fetch"})
_CODE_TOOLS = frozenset(
    {"code_execution", "bash_code_execution", "text_editor_code_execution"}
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reconstruct_transcript(
    messages: Iterable[Any],
    *,
    task_id: str | None = None,
) -> Transcript:
    """Reconstruct a Transcript from collected claude-agent-sdk messages.

    Args:
        messages: An iterable of SDK ``Message`` objects (e.g. the list you get
            from ``[m async for m in query(...)]``).
        task_id: Optional task id; defaults to the run's session id.

    Returns:
        A reconstructed :class:`~compass.core.transcript.Transcript`.
    """
    builder = _Builder(task_id=task_id)
    for msg in messages:
        builder.consume(msg)
    return builder.finish()


async def reconstruct_transcript_from_stream(
    stream: AsyncIterable[Any],
    *,
    task_id: str | None = None,
) -> Transcript:
    """Like :func:`reconstruct_transcript`, but consumes an async message stream.

    Convenient to wrap ``query(...)`` / ``ClaudeSDKClient.receive_response()``
    directly without collecting the messages yourself.
    """
    builder = _Builder(task_id=task_id)
    async for msg in stream:
        builder.consume(msg)
    return builder.finish()


def reconstruct_transcript_from_wire(
    events: Iterable[dict[str, Any]],
    *,
    task_id: str | None = None,
) -> Transcript:
    """Reconstruct a Transcript from Claude Code **stream-json** wire dicts.

    Each event is a raw dict as emitted by
    ``claude -p ... --output-format stream-json`` (one JSON object per line):
    ``{"type": "assistant", "message": {...}, "session_id": ...}`` etc. This is
    the *offline* counterpart to :func:`reconstruct_transcript` — the wire dicts
    are adapted to the same shape and fed through the same mapping.
    """
    builder = _Builder(task_id=task_id)
    for event in events:
        obj = _wire_to_obj(event)
        if obj is not None:
            builder.consume(obj)
    return builder.finish()


def import_claude_stream_json(
    path: str | Path,
    *,
    task_id: str | None = None,
) -> Transcript:
    """Import a Claude Code ``--output-format stream-json`` file into a Transcript.

    Accepts JSON-lines (the stream-json default), or a single JSON array/object
    (the ``--output-format json`` shape). Malformed lines are skipped.

    Args:
        path: Path to the saved stream-json output.
        task_id: Optional task id; defaults to the run's session id.
    """
    path = Path(path).expanduser()
    events = _load_wire_events(path.read_text(encoding="utf-8"))
    return reconstruct_transcript_from_wire(events, task_id=task_id)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


class _Builder:
    def __init__(self, task_id: str | None) -> None:
        self._task_id = task_id
        self._session_id: str | None = None
        self.transcript = Transcript(task_id=task_id or "claude_agent", trial_id="")
        self.transcript.metadata = {"importer": "claude_agent"}
        # tool_use_id -> pending ToolCall awaiting its result
        self._pending: dict[str, ToolCall] = {}
        # tool_use_id of a Task call -> the sub-agent's display name
        self._subagents: dict[str, str] = {}
        # llm.generation calls, in order (cost is attached to the last one)
        self._llm_calls: list[ToolCall] = []
        self._last_assistant_text: str | None = None
        self._last_model: str | None = None
        self._turn = 0

    # -- dispatch ----------------------------------------------------------

    def consume(self, msg: Any) -> None:
        kind = _classify(msg)
        if kind == "assistant":
            self._on_assistant(msg)
        elif kind == "user":
            self._on_user(msg)
        elif kind == "result":
            self._on_result(msg)
        elif kind == "system":
            self._on_system(msg)
        # "stream" (partial deltas) / "ratelimit" / "other" carry no
        # reconstruction value and are ignored.

    # -- assistant ---------------------------------------------------------

    def _on_assistant(self, msg: Any) -> None:
        self._note_session(getattr(msg, "session_id", None))
        self._turn += 1
        agent_name = self._agent_of(msg)
        parent = getattr(msg, "parent_tool_use_id", None)
        model = getattr(msg, "model", None)
        if model:
            self._last_model = model
        status = "error" if getattr(msg, "error", None) else "ok"

        text_parts: list[str] = []
        for block in getattr(msg, "content", None) or []:
            bkind = _block_kind(block)
            if bkind == "text":
                text_parts.append(getattr(block, "text", "") or "")
            elif bkind == "thinking":
                thinking = getattr(block, "thinking", "") or ""
                if thinking:
                    self.transcript.add_reasoning_step(f"[thinking] {_truncate(thinking)}")
            elif bkind == "tool_use":
                self._open_tool_call(block, agent_name, parent)
                self._track_subagent(block)
            elif bkind == "tool_result":
                # Server-side tool results can appear inline in the assistant msg.
                self._close_tool_call(block)

        assistant_text = "\n".join(p for p in text_parts if p).strip() or None
        if assistant_text:
            self._last_assistant_text = assistant_text

        tokens = _tokens(getattr(msg, "usage", None))
        meta: dict[str, Any] = {}
        if model:
            meta["model"] = model
        if getattr(msg, "message_id", None):
            meta["message_id"] = msg.message_id
        if parent:
            meta["parent_tool_use_id"] = parent

        call = ToolCall(
            tool_name="llm.generation",
            input=_llm_input(msg),
            output=assistant_text,
            status=status,
            error={"message": msg.error} if getattr(msg, "error", None) else None,
            tokens=tokens,
            tool_type="llm",
            metadata=meta,
            turn_index=self._turn,
            agent_name=agent_name,
        )
        self.transcript.tool_calls.append(call)
        self._llm_calls.append(call)

    def _open_tool_call(self, block: Any, agent_name: str | None, parent: str | None) -> None:
        call_id = str(getattr(block, "id", "") or "")
        name = getattr(block, "name", "") or "unknown_tool"
        kwargs: dict[str, Any] = dict(
            tool_name=str(name),
            input=_as_dict(getattr(block, "input", None)),
            status="ok",
            tool_type=_tool_type(str(name)),
            metadata={"parent_tool_use_id": parent} if parent else {},
            turn_index=self._turn,
            agent_name=agent_name,
        )
        if call_id:
            kwargs["call_id"] = call_id
        call = ToolCall(**kwargs)
        self.transcript.tool_calls.append(call)
        if call_id:
            self._pending[call_id] = call

    def _track_subagent(self, block: Any) -> None:
        """A ``Task`` tool call spawns a sub-agent; remember its name by id."""
        if getattr(block, "name", None) != "Task":
            return
        tool_use_id = str(getattr(block, "id", "") or "")
        tinput = getattr(block, "input", None) or {}
        if isinstance(tinput, dict):
            label = (
                tinput.get("subagent_type")
                or tinput.get("description")
                or "subagent"
            )
            if tool_use_id:
                self._subagents[tool_use_id] = str(label)

    # -- user (tool results + human prompts) -------------------------------

    def _on_user(self, msg: Any) -> None:
        content = getattr(msg, "content", None)
        if isinstance(content, str):
            text = content
            if not self.transcript.input_prompt:
                self.transcript.input_prompt = text
            elif text:
                self.transcript.add_reasoning_step(f"[user] {_truncate(text)}")
            return
        for block in content or []:
            if _block_kind(block) == "tool_result":
                self._close_tool_call(block)

    def _close_tool_call(self, block: Any) -> None:
        call_id = str(getattr(block, "tool_use_id", "") or "")
        result_text = _content_to_text(getattr(block, "content", None))
        is_error = bool(getattr(block, "is_error", False))
        call = self._pending.pop(call_id, None)
        if call is None:
            return  # server result with no tracked call, or already filled
        call.output = result_text
        call.status = "error" if is_error else "ok"
        if is_error:
            call.error = {"message": _truncate(result_text, 500)}

    # -- result ------------------------------------------------------------

    def _on_result(self, msg: Any) -> None:
        self._note_session(getattr(msg, "session_id", None))
        result_text = getattr(msg, "result", None)
        if isinstance(result_text, str) and result_text:
            self._last_assistant_text = result_text

        duration = getattr(msg, "duration_ms", None)
        if isinstance(duration, (int, float)):
            self.transcript.total_duration_ms = float(duration)

        # The CLI reports one authoritative total; attach it to the final LLM
        # call so Transcript.sum_cost() / cost_budget see the true run cost.
        total = getattr(msg, "total_cost_usd", None)
        if isinstance(total, (int, float)) and self._llm_calls:
            self._llm_calls[-1].cost = CostInfo(
                total_usd=float(total),
                metadata={"source": "claude_agent_cli"},
            )
        if total is not None:
            self.transcript.metadata["total_cost_usd"] = total

        for key in ("num_turns", "model_usage", "permission_denials"):
            val = getattr(msg, key, None)
            if val:
                self.transcript.metadata[key] = val

    # -- system (task lifecycle) ------------------------------------------

    def _on_system(self, msg: Any) -> None:
        # TaskStarted carries a tool_use_id + human description for a sub-agent.
        tool_use_id = getattr(msg, "tool_use_id", None)
        description = getattr(msg, "description", None) or getattr(msg, "task_type", None)
        if tool_use_id and description:
            self._subagents.setdefault(str(tool_use_id), str(description))

    # -- finish ------------------------------------------------------------

    def finish(self) -> Transcript:
        if self._last_model:
            self.transcript.environment.model_version = self._last_model
        if self._last_assistant_text:
            self.transcript.set_outcome(
                output_data={"final_output": self._last_assistant_text}
            )
        return self.transcript

    # -- helpers -----------------------------------------------------------

    def _note_session(self, session_id: str | None) -> None:
        if session_id and not self._session_id:
            self._session_id = session_id
            self.transcript.trial_id = session_id
            self.transcript.metadata["session_id"] = session_id
            if self._task_id is None:
                self.transcript.task_id = session_id

    def _agent_of(self, msg: Any) -> str | None:
        parent = getattr(msg, "parent_tool_use_id", None)
        if parent:
            return self._subagents.get(str(parent), "subagent")
        return None


# ---------------------------------------------------------------------------
# Message / block classification (duck-typed)
# ---------------------------------------------------------------------------


def _classify(msg: Any) -> str:
    if hasattr(msg, "num_turns") or hasattr(msg, "total_cost_usd"):
        return "result"
    if hasattr(msg, "rate_limit_info"):
        return "ratelimit"
    if hasattr(msg, "event") and not hasattr(msg, "content"):
        return "stream"
    if hasattr(msg, "content"):
        # AssistantMessage carries a model; UserMessage does not.
        return "assistant" if hasattr(msg, "model") else "user"
    if hasattr(msg, "subtype") and hasattr(msg, "data"):
        return "system"
    return "other"


def _block_kind(block: Any) -> str:
    if hasattr(block, "thinking"):
        return "thinking"
    if hasattr(block, "tool_use_id"):
        return "tool_result"
    if hasattr(block, "name") and hasattr(block, "input"):
        return "tool_use"
    if hasattr(block, "text"):
        return "text"
    return "other"


def _tool_type(name: str) -> str:
    if name.startswith("mcp__"):
        return "mcp"
    if name in _SEARCH_TOOLS:
        return "search"
    if name in _CODE_TOOLS:
        return "code"
    return "function"


# ---------------------------------------------------------------------------
# stream-json wire adapter (dict -> the duck-typed shape _Builder reads)
# ---------------------------------------------------------------------------


def _load_wire_events(content: str) -> list[dict[str, Any]]:
    """Load stream-json events: JSON-lines, or a single JSON array/object."""
    stripped = content.strip()
    if not stripped:
        return []
    try:
        whole = json.loads(stripped)
    except (ValueError, TypeError):
        whole = None
    if isinstance(whole, list):
        return [e for e in whole if isinstance(e, dict)]
    if isinstance(whole, dict):
        return [whole]
    events: list[dict[str, Any]] = []
    for line in stripped.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            logger.warning("claude stream-json: skipping non-JSON line")
            continue
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _wire_to_obj(data: Any) -> Any:
    """Adapt one stream-json wire dict to the attribute shape ``_Builder`` reads.

    Returns ``None`` for events with no reconstruction value (stream deltas,
    rate-limit events, unknown types).
    """
    if not isinstance(data, dict):
        return None
    kind = data.get("type")
    if kind == "assistant":
        m = data.get("message") or {}
        blocks = [_wire_block(b) for b in (m.get("content") or []) if isinstance(b, dict)]
        return SimpleNamespace(
            content=blocks,
            model=m.get("model"),
            usage=m.get("usage"),
            stop_reason=m.get("stop_reason"),
            message_id=m.get("id"),
            session_id=data.get("session_id"),
            parent_tool_use_id=data.get("parent_tool_use_id"),
            error=data.get("error"),
        )
    if kind == "user":
        m = data.get("message") or {}
        content = m.get("content")
        if isinstance(content, list):
            content = [_wire_block(b) for b in content if isinstance(b, dict)]
        # No ``model`` attribute -> classified as a user message.
        return SimpleNamespace(
            content=content,
            parent_tool_use_id=data.get("parent_tool_use_id"),
            tool_use_result=data.get("tool_use_result"),
        )
    if kind == "result":
        return SimpleNamespace(
            total_cost_usd=data.get("total_cost_usd"),
            num_turns=data.get("num_turns"),
            duration_ms=data.get("duration_ms"),
            result=data.get("result"),
            session_id=data.get("session_id"),
            usage=data.get("usage"),
            model_usage=data.get("modelUsage"),
            permission_denials=data.get("permission_denials"),
            subtype=data.get("subtype"),
            is_error=data.get("is_error"),
        )
    if kind == "system":
        return SimpleNamespace(
            subtype=data.get("subtype", ""),
            data=data,
            tool_use_id=data.get("tool_use_id"),
            description=data.get("description"),
            task_type=data.get("task_type"),
            session_id=data.get("session_id"),
        )
    return None  # stream_event / rate_limit_event / unknown


def _wire_block(block: dict[str, Any]) -> Any:
    kind = block.get("type")
    if kind == "text":
        return SimpleNamespace(text=block.get("text", ""))
    if kind == "thinking":
        return SimpleNamespace(
            thinking=block.get("thinking", ""), signature=block.get("signature", "")
        )
    if kind in ("tool_use", "server_tool_use"):
        return SimpleNamespace(
            id=block.get("id"), name=block.get("name"), input=block.get("input")
        )
    if kind in ("tool_result", "advisor_tool_result"):
        return SimpleNamespace(
            tool_use_id=block.get("tool_use_id"),
            content=block.get("content"),
            is_error=block.get("is_error"),
        )
    return SimpleNamespace()  # unknown block -> ignored by _block_kind


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _tokens(usage: Any) -> TokenUsage | None:
    if not isinstance(usage, dict):
        return None
    in_tok = int(usage.get("input_tokens", 0) or 0)
    out_tok = int(usage.get("output_tokens", 0) or 0)
    if not (in_tok or out_tok):
        return None
    meta: dict[str, Any] = {}
    for key in ("cache_read_input_tokens", "cache_creation_input_tokens"):
        if usage.get(key):
            meta[key] = usage[key]
    return TokenUsage(input_tokens=in_tok, output_tokens=out_tok, metadata=meta)


def _llm_input(msg: Any) -> dict[str, Any]:
    data: dict[str, Any] = {}
    if getattr(msg, "model", None):
        data["model"] = msg.model
    if getattr(msg, "stop_reason", None):
        data["stop_reason"] = msg.stop_reason
    return data


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text" or "text" in block:
                    parts.append(str(block.get("text", "")))
                elif block.get("type") == "image":
                    parts.append("[image]")
            elif isinstance(block, str):
                parts.append(block)
            else:
                text = getattr(block, "text", None)
                if text is not None:
                    parts.append(str(text))
        return "\n".join(p for p in parts if p)
    if content is None:
        return ""
    return str(content)


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def _truncate(text: str, limit: int = 4000) -> str:
    return text if len(text) <= limit else text[:limit] + "…"
