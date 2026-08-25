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
- a **successful** ``Write`` / ``Edit`` / ``MultiEdit`` / ``NotebookEdit`` ->
  a ``StateChange`` on that call's ``state_delta`` (see below)

Zero hard dependency on ``claude_agent_sdk``: messages are read by duck-typing
their attributes, so this works on collected SDK objects without importing them.

**State delta — what the agent changed, not what it said.** Compass defines the
``StateChange`` slot but leaves *capturing* deltas to importers, and until this
mapping existed no importer filled it — which left the ``state_delta`` grader
inert on Claude traces, passing vacuously. The file-editing tools carry their
target in the tool input, so the mapping is exact rather than inferred::

    graders:
      - name: state_delta
        config:
          forbid: [{kind: file, target: "tests/*"}]   # don't rewrite the tests
          require: [{kind: file, target: "src/api/*"}]

Three properties worth knowing:

- **Only successful calls count.** The delta is recorded when the tool *result*
  arrives; a rejected edit or a call whose result never came back records
  nothing, because nothing changed.
- **Targets are relative to the session cwd** when the run reports one (the
  ``system``/``init`` event). This is what makes globs portable: a
  ``claude_code`` trial works inside a throwaway git worktree, so an absolute
  target would be a different random path every run. The absolute path is kept
  on the change's ``metadata``.
- **Shell-mediated changes are not captured.** An agent that deletes a file with
  ``Bash(rm ...)`` produces no ``StateChange`` here — reliably parsing shell is
  not something this layer should pretend to do, and a wrong delta is worse than
  a missing one. Guard those with a domain grader over ``Bash`` calls (see
  ``examples/ops_qa``), and read an empty ``state_delta`` as "nothing recorded",
  never as "nothing changed".
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from compass.core.transcript import (
    CostInfo,
    StateChange,
    TokenUsage,
    ToolCall,
    Transcript,
)
from compass.integrations._common import as_dict, truncate, write_op

logger = logging.getLogger(__name__)

# Server-side tool names whose calls map to a "search" tool_type.
_SEARCH_TOOLS = frozenset({"web_search", "web_fetch"})
_CODE_TOOLS = frozenset(
    {"code_execution", "bash_code_execution", "text_editor_code_execution"}
)

# File-editing tools -> the tool-input key holding the path they change.
# Only these produce a StateChange; Read/Glob/Grep change nothing, and Bash is
# deliberately excluded (see the module docstring).
_FILE_EDIT_TOOLS: dict[str, str] = {
    "Write": "file_path",
    "Edit": "file_path",
    "MultiEdit": "file_path",
    "NotebookEdit": "notebook_path",
}

# Edit/MultiEdit/NotebookEdit require the file to already exist, so "update" is
# the tool contract rather than a guess. Write is the ambiguous one.
_FILE_EDIT_OPS: dict[str, str] = {
    "Edit": "update",
    "MultiEdit": "update",
    "NotebookEdit": "update",
}


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
    reconstructor = WireReconstructor(task_id=task_id)
    for event in events:
        reconstructor.feed(event)
    return reconstructor.finish()


class WireReconstructor:
    """Incremental form of :func:`reconstruct_transcript_from_wire`.

    Same mapping, fed one wire dict at a time, for callers that read the CLI's
    stdout as it arrives rather than after the process exits. Two things fall
    out of that: a run killed by a timeout still yields the steps it completed
    (:meth:`finish` is valid at any point), and a live progress view is possible
    without a second parser.

    ::

        r = WireReconstructor(task_id="add-rate-limit")
        for line in proc.stdout:
            r.feed(json.loads(line))
        transcript = r.finish()
    """

    def __init__(self, *, task_id: str | None = None) -> None:
        self._builder = _Builder(task_id=task_id)

    def feed(self, event: dict[str, Any]) -> None:
        """Consume one stream-json wire dict.

        Event types this mapping does not model are counted into
        ``transcript.metadata["unhandled_events"]`` rather than discarded — see
        :meth:`_Builder.note_unhandled`.
        """
        obj = _wire_to_obj(event)
        if obj is None:
            self._builder.note_unhandled(str(event.get("type") or "unknown"))
            return
        self._builder.consume(obj)

    def feed_line(self, line: str) -> bool:
        """Consume one raw stdout line. Returns False if it was not JSON.

        The CLI interleaves the occasional non-JSON line into stdout, and a
        partial trace beats a crashed harness — so a bad line is reported, not
        raised.
        """
        line = line.strip()
        if not line:
            return False
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            return False
        if not isinstance(obj, dict):
            return False
        self.feed(obj)
        return True

    def finish(self) -> Transcript:
        """The Transcript reconstructed so far. Safe to call mid-stream."""
        return self._builder.finish()


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
        # Session cwd, reported by the system/init event; used to make file
        # targets relative (and therefore matchable by a stable glob).
        self._cwd: str | None = None
        # event type -> count, for everything this mapping does not consume
        self._unhandled: dict[str, int] = {}

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
        elif kind == "ratelimit":
            self._on_rate_limit(msg)
        else:
            # Partial deltas, rate-limit notices, anything the CLI grows next:
            # nothing to reconstruct, but counted rather than dropped.
            self.note_unhandled(
                type(msg).__name__ if kind == "other" else kind
            )

    def note_unhandled(self, kind: str) -> None:
        """Record that an event of this type went by without being mapped.

        Only a count — no payload, because the shape is by definition unknown
        and a stream can carry thousands of partial deltas. A count is enough
        to answer the question that matters: *did something happen during this
        run that the transcript does not explain?* A run throttled three times
        and a run that simply took a while look identical in the trace
        otherwise, and the first is a harness problem being read as a slow
        model.
        """
        self._unhandled[kind] = self._unhandled.get(kind, 0) + 1

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
                    self.transcript.add_reasoning_step(f"[thinking] {truncate(thinking)}")
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
            input=as_dict(getattr(block, "input", None)),
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
                self.transcript.add_reasoning_step(f"[user] {truncate(text)}")
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
            call.error = {"message": truncate(result_text, 500)}
            return  # a failed edit changed nothing
        self._record_state_delta(call, result_text)

    def _record_state_delta(self, call: ToolCall, result_text: str) -> None:
        """Attach the file change a successful editing tool just made.

        Recorded here, at result time, rather than when the call was opened:
        until the result arrives the change has not happened, and it may never
        (a denied permission, a truncated run). ``state_delta`` should record
        what the environment *did*, not what the model asked for.
        """
        path_key = _FILE_EDIT_TOOLS.get(call.tool_name)
        if path_key is None:
            return
        raw_path = (call.input or {}).get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            return

        target, absolute = self._relativize(raw_path)
        metadata: dict[str, Any] = {"tool": call.tool_name}
        if absolute and absolute != target:
            metadata["absolute_path"] = absolute

        call.state_delta.append(
            StateChange(
                kind="file",
                op=_FILE_EDIT_OPS.get(call.tool_name) or write_op(result_text),
                target=target,
                metadata=metadata,
            )
        )

    def _relativize(self, raw_path: str) -> tuple[str, str]:
        """(target, absolute) — target relative to the session cwd when possible.

        A ``claude_code`` trial runs in a throwaway worktree, so the absolute
        path differs every run and no glob a user writes could ever match it.
        """
        absolute = raw_path
        if not self._cwd:
            return raw_path, absolute
        try:
            relative = str(Path(raw_path).relative_to(self._cwd))
        except ValueError:
            return raw_path, absolute  # outside the workspace: keep it absolute
        return relative, absolute

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

    # -- rate limits -------------------------------------------------------

    def _on_rate_limit(self, msg: Any) -> None:
        """Record the CLI's quota notices verbatim, plus a tally by status.

        These matter to a comparison run on a subscription: hitting the window
        gets requests *rejected* (overage is commonly disabled at the org
        level), so a partial arm looks like a worse model rather than a run
        that ran out of quota. A run that was merely told "allowed" and one
        that was throttled are otherwise indistinguishable in the trace.

        Stored as the provider sent it. The status vocabulary is the CLI's, not
        something to second-guess here — a grader or a human can read
        ``status_counts`` and decide what counts as degraded.
        """
        info = getattr(msg, "rate_limit_info", None)
        if not isinstance(info, dict) or not info:
            return  # nothing was reported, so there is nothing to record
        record = self.transcript.metadata.setdefault(
            "rate_limit", {"events": 0, "status_counts": {}, "latest": {}}
        )
        record["events"] += 1
        status = str(info.get("status", "unknown"))
        record["status_counts"][status] = record["status_counts"].get(status, 0) + 1
        record["latest"] = dict(info)

    # -- system (task lifecycle) ------------------------------------------

    def _on_system(self, msg: Any) -> None:
        # The init event announces the session's working directory, which is
        # what makes file targets in the state delta relative and portable.
        data = getattr(msg, "data", None)
        if isinstance(data, dict):
            cwd = data.get("cwd")
            if isinstance(cwd, str) and cwd and not self._cwd:
                self._cwd = cwd
                self.transcript.metadata["cwd"] = cwd

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
        # Rewritten rather than merged: finish() is valid mid-stream and may be
        # called again later, and these are counts of the whole run so far.
        if self._unhandled:
            self.transcript.metadata["unhandled_events"] = dict(self._unhandled)
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
    if kind == "rate_limit_event":
        return SimpleNamespace(
            rate_limit_info=data.get("rate_limit_info") or {},
            session_id=data.get("session_id"),
        )
    return None  # stream_event / unknown


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
    # Cached traffic is first-class, not metadata: a coding agent's system
    # prompt and tool definitions are cached, so uncached input is a rounding
    # error next to it and a transcript that ignored it would report a fraction
    # of a percent of what the run actually processed.
    cache_read = int(usage.get("cache_read_input_tokens", 0) or 0)
    cache_creation = int(usage.get("cache_creation_input_tokens", 0) or 0)
    if not (in_tok or out_tok or cache_read or cache_creation):
        return None
    return TokenUsage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_creation,
    )


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




