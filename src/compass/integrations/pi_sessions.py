"""Import a pi (``@earendil-works/pi-*``) run into a Compass Transcript.

pi persists an agent run as a **JSONL session tree**: the first line is a session
header, and every subsequent line is a ``SessionTreeEntry`` linked to its parent by
``parentId`` (forking/branching produces a tree). This module reads such a file
*offline* and reconstructs a Compass
:class:`~compass.core.transcript.Transcript`, so a pi run can be evaluated with
Compass's transcript-scope graders **without writing a per-agent adapter**::

    from compass.integrations import import_pi_session

    transcript = import_pi_session("~/.pi/agent/sessions/<project>/run.jsonl")
    # transcript.tool_calls / .reasoning_steps / .outcome are ready to grade

The same mapping also runs **live**, over the event stream pi emits with
``--mode json``, through :class:`PiStreamReconstructor` — that is what the ``pi``
adapter uses to record a run it drives itself (see :mod:`compass.adapters.pi`).
The two formats share a first line (the session header) and share the message
payloads, so they share the mapping; :func:`import_pi_session` sniffs which one a
file holds, and a stream saved by ``save_stream_to`` imports as readily as a
session does.

Mapping (pi -> Compass):
- session header ``{id, cwd, timestamp}``     -> ``trial_id`` / metadata / timing
- assistant message ``usage``                 -> an ``llm.generation`` ``ToolCall``
  (``TokenUsage`` + native ``CostInfo``; falls back to Compass pricing if pi did
  not record a cost)
- assistant ``toolCall`` content block         -> a ``ToolCall`` (matched to its
  ``toolResult`` message by ``toolCallId`` to fill output / status / duration)
- a **successful** ``write`` / ``edit``        -> a ``StateChange`` on that call's
  ``state_delta`` (see below)
- assistant ``thinking`` block                 -> a reasoning step
- ``user`` message                             -> ``input_prompt`` (first) / reasoning
- ``compaction`` / ``branch_summary`` / ``model_change`` / ``custom`` … -> metadata /
  reasoning

By default only the **active branch** (the path from the session's current leaf back
to the root) is reconstructed, so abandoned forks do not pollute the evaluation. For
the common linear session this is identical to file order.

**State delta — what the agent changed, not what it said.** Compass defines the
``StateChange`` slot but leaves *capturing* deltas to importers. pi's file-editing
tools carry their target in the tool input, so the mapping is exact rather than
inferred::

    graders:
      - name: state_delta
        config:
          forbid: [{kind: file, target: "tests/*"}]   # don't rewrite the tests

Same three properties as the Claude mapping: only **successful** calls count (the
delta is recorded when the result arrives), targets are **relative to the session
cwd** so a glob still matches inside a throwaway worktree, and **shell-mediated
changes are not captured** — an agent that deletes a file with ``bash(rm …)``
records nothing here, so read an empty ``state_delta`` as "nothing recorded",
never as "nothing changed".

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

from compass.adapters.llm import calculate_cost  # submodule: see openai_agents.py
from compass.core.transcript import (
    CostInfo,
    StateChange,
    TokenUsage,
    ToolCall,
    Transcript,
)

logger = logging.getLogger(__name__)

# Max characters kept for a single reasoning step (thinking blocks can be huge).
_MAX_STEP_CHARS = 4000

# pi's file-editing tools -> the tool-input key holding the path they change.
# Only these produce a StateChange: `read`/`ls`/`glob` change nothing, and
# `bash` is deliberately excluded (see the module docstring).
_FILE_EDIT_TOOLS: dict[str, str] = {"write": "path", "edit": "path"}

# `edit` requires the file to already exist, so "update" is the tool contract
# rather than a guess. `write` does both and its result does not say which, so
# it falls back to _write_op — match on `target`, not `op`, when it matters.
_FILE_EDIT_OPS: dict[str, str] = {"edit": "update"}

# Event types that appear only in the ``--mode json`` stream, never in a saved
# session file. Used both to tell the two formats apart and to skip the events
# whose payload a message event already carries.
_STREAM_EVENT_TYPES = frozenset(
    {
        "agent_start",
        "agent_end",
        "agent_settled",
        "turn_start",
        "turn_end",
        "message_start",
        "message_update",
        "message_end",
        "tool_execution_start",
        "tool_execution_update",
        "tool_execution_end",
    }
)


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
    """Import a single pi JSONL file into a Compass :class:`Transcript`.

    Accepts either of pi's two on-disk shapes — a **session tree** (what pi
    persists under its session dir) or a saved **``--mode json`` event stream**
    (what the ``pi`` adapter writes with ``save_stream_to``). Both start with the
    same session header, so the format is sniffed from the lines after it.

    Args:
        path: Path to a pi ``*.jsonl`` session or stream file.
        active_branch_only: If True (default), reconstruct only the path from the
            session's current leaf back to the root (abandoned forks are dropped).
            If False, use every entry in file (append) order. Ignored for a
            stream, which is linear by construction.

    Returns:
        A reconstructed :class:`~compass.core.transcript.Transcript`.

    Raises:
        PiSessionError: If the file's first line is not a valid session header.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).expanduser()
    return load_pi_session(
        path.read_text(encoding="utf-8"),
        source=str(path),
        active_branch_only=active_branch_only,
    )


def load_pi_session(
    content: str,
    *,
    source: str = "<string>",
    active_branch_only: bool = True,
) -> Transcript:
    """Reconstruct a Transcript from the raw text of a pi session or stream file.

    Same as :func:`import_pi_session` but takes the file *content* directly, which
    is convenient for tests and in-memory pipelines.
    """
    header, objects = _parse_lines(content, source)
    if _looks_like_stream(objects):
        return _reconstruct_stream([header, *objects], source=source)
    entries = _valid_entries(objects, source)
    ordered = _active_path(entries) if active_branch_only else entries
    return _reconstruct(header, ordered, source=source)


def import_pi_stream_json(
    path: str | Path,
    *,
    task_id: str | None = None,
) -> Transcript:
    """Import a saved pi ``--mode json`` event stream into a Transcript.

    :func:`import_pi_session` already detects this shape; use this when you know
    what you have and want to name the task yourself.

    Args:
        path: Path to the saved stream (JSON lines).
        task_id: Optional task id; defaults to the run's session id.
    """
    path = Path(path).expanduser()
    reconstructor = PiStreamReconstructor(task_id=task_id, source=str(path))
    for line in path.read_text(encoding="utf-8").splitlines():
        reconstructor.feed_line(line)
    return reconstructor.finish()


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


def _parse_lines(content: str, source: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Split a pi JSONL file into (header, objects) without interpreting them.

    The header line is validated; malformed lines are skipped with a warning so a
    single corrupt line does not sink the whole import. Which *shape* the objects
    are — session-tree entries or stream events — is decided by the caller, since
    the two formats share this header.
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

    objects: list[dict[str, Any]] = []
    for i, line in enumerate(lines[1:], start=2):
        try:
            obj = json.loads(line)
        except (ValueError, TypeError):
            logger.warning("%s: line %d is not valid JSON, skipping", source, i)
            continue
        if not isinstance(obj, dict) or "type" not in obj:
            logger.warning("%s: line %d is not a valid session entry, skipping", source, i)
            continue
        objects.append(obj)
    return header, objects


def _valid_entries(objects: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    """Keep the objects that are session-tree entries (``type`` **and** ``id``)."""
    entries: list[dict[str, Any]] = []
    for obj in objects:
        if "id" not in obj:
            logger.warning("%s: entry %r has no id, skipping", source, obj.get("type"))
            continue
        entries.append(obj)
    return entries


def _looks_like_stream(objects: list[dict[str, Any]]) -> bool:
    """Whether these lines are ``--mode json`` events rather than session entries.

    Session-tree entries all carry an ``id`` (that is what ``parentId`` points at);
    stream events never do, and their type names are disjoint from the entry
    types. Checking both ways round means neither an unknown entry type nor an
    unknown event type can flip the answer on its own.
    """
    for obj in objects:
        if obj.get("type") in _STREAM_EVENT_TYPES:
            return True
        if "id" in obj:
            return False
    return False


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


def _new_transcript(
    header: dict[str, Any], *, source: str, task_id: str | None = None
) -> Transcript:
    """An empty Transcript carrying the session header's bookkeeping."""
    transcript = Transcript(task_id=task_id or "pi_session", trial_id="pi_session")
    transcript.input_params = {"cwd": "", "source": source}
    transcript.metadata = {"importer": "pi"}
    _apply_header(transcript, header, task_id=task_id)
    return transcript


def _apply_header(
    transcript: Transcript, header: dict[str, Any], *, task_id: str | None = None
) -> None:
    """Fold a session header into the transcript (also the stream's first event)."""
    session_id = str(header.get("id") or "pi_session")
    cwd = header.get("cwd", "") or transcript.input_params.get("cwd", "")
    transcript.trial_id = session_id
    if not task_id:
        transcript.task_id = session_id
    transcript.input_params["cwd"] = cwd
    transcript.metadata.update(
        {
            "session_id": session_id,
            "cwd": cwd,
            "session_version": header.get("version"),
        }
    )
    if header.get("parentSession"):
        transcript.metadata["parent_session"] = header["parentSession"]


def _reconstruct(
    header: dict[str, Any],
    entries: list[dict[str, Any]],
    *,
    source: str,
) -> Transcript:
    transcript = _new_transcript(header, source=source)

    # toolCallId -> the ToolCall awaiting its result.
    pending: dict[str, ToolCall] = {}
    last_assistant_text: str | None = None
    last_model: str | None = None
    turn = 0  # each assistant message starts a new turn

    for entry in entries:
        if entry.get("type") == "message":
            message = entry.get("message") or {}
            if message.get("role") == "assistant":
                turn += 1
            text, model = _handle_message(transcript, message, pending, turn)
            if text is not None:
                last_assistant_text = text
            if model:
                last_model = model
        else:
            _handle_meta_entry(transcript, entry)

    _finalize(
        transcript,
        header,
        _last_message_epoch(entries),
        last_assistant_text,
        last_model,
    )
    return transcript


def _handle_meta_entry(transcript: Transcript, entry: dict[str, Any]) -> bool:
    """Map a non-message session entry. False if the type is unknown here.

    These entries also show up in the ``--mode json`` stream (a mid-run model
    switch, a compaction), so both paths route through this.
    """
    etype = entry.get("type")
    if etype == "model_change":
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
    elif etype in ("label", "leaf"):
        pass  # structural — the branch walk already used them
    else:
        return False
    return True


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
    if not is_error:
        _record_state_delta(transcript, tc, result_text)


def _record_state_delta(
    transcript: Transcript, call: ToolCall, result_text: str
) -> None:
    """Attach the file change a successful editing tool just made.

    Recorded here, at result time, rather than when the call was opened: until
    the result arrives the change has not happened, and it may never (a denied
    permission, a killed run). ``state_delta`` records what the environment
    *did*, not what the model asked for.
    """
    path_key = _FILE_EDIT_TOOLS.get(call.tool_name)
    if path_key is None:
        return
    raw_path = (call.input or {}).get(path_key)
    if not isinstance(raw_path, str) or not raw_path:
        return

    target, absolute = _relativize(raw_path, str(transcript.metadata.get("cwd") or ""))
    metadata: dict[str, Any] = {"tool": call.tool_name}
    if absolute and absolute != target:
        metadata["absolute_path"] = absolute

    call.state_delta.append(
        StateChange(
            kind="file",
            op=_FILE_EDIT_OPS.get(call.tool_name) or _write_op(result_text),
            target=target,
            metadata=metadata,
        )
    )


def _relativize(raw_path: str, cwd: str) -> tuple[str, str]:
    """(target, absolute) — target relative to the session cwd, normalized.

    Two things have to happen for a user's glob to work:

    - **Relative to the cwd.** A ``pi`` trial runs in a throwaway worktree, so an
      absolute target is a different random path every run and no glob could
      match it. The absolute form is kept on the change's metadata.
    - **One spelling per file.** pi passes the path through as the agent typed
      it, and models type both ``pricing.py`` and ``./pricing.py`` for the same
      file — observed in a single two-model run. Left alone, ``target:
      "pricing.py"`` matches one run and silently misses the other, which is the
      worst possible failure for an integrity gate.
    """
    path = Path(raw_path)
    if not path.is_absolute():
        target = os.path.normpath(raw_path)
        return target, (os.path.normpath(os.path.join(cwd, raw_path)) if cwd else target)
    if cwd:
        try:
            return str(path.relative_to(cwd)), str(path)
        except ValueError:
            pass  # outside the session cwd — the absolute path *is* the target
    return str(path), str(path)


def _write_op(result_text: str) -> str:
    """``create`` or ``update`` for a ``write``, read off the tool's own result.

    ``write`` is the one editing tool that does both, and its input cannot tell
    them apart. pi's success message ("Successfully wrote N bytes to X") does not
    distinguish either, so this defaults to ``update`` — true either way, since
    the contents changed. Match on ``target`` rather than ``op`` when you need
    certainty.
    """
    return "create" if "created" in (result_text or "").lower() else "update"


def _finalize(
    transcript: Transcript,
    header: dict[str, Any],
    end_ts: float,
    last_assistant_text: str | None,
    last_model: str | None,
) -> None:
    """Timing, model and outcome. Safe to call repeatedly (the stream does)."""
    if last_model:
        transcript.environment.model_version = last_model

    # Timing: prefer the session header start + last message timestamp.
    start = _parse_iso(header.get("timestamp"))
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
# Streaming (pi --mode json)
# ---------------------------------------------------------------------------


class PiStreamReconstructor:
    """Incremental form of the mapping, fed pi's ``--mode json`` event stream.

    Same mapping as the offline importer — pi emits the *same message objects*
    live that it later persists — so a driven run and an imported session grade
    identically. Two things fall out of consuming it as it arrives: a run killed
    by a timeout still yields the steps it completed (:meth:`finish` is valid at
    any point), and a live progress view is possible without a second parser.

    ::

        r = PiStreamReconstructor(task_id="add-rate-limit")
        for line in proc.stdout:
            r.feed_line(line)      # non-JSON lines return False, never raise
        transcript = r.finish()

    Only ``message_end`` is mapped, never ``message_start`` / ``message_update``:
    a message is streamed as a start frame, a run of deltas and an end frame, and
    only the end frame carries the finished content and the turn's ``usage``.
    Mapping any of the others would double-count every turn.
    """

    def __init__(
        self, *, task_id: str | None = None, source: str = "<stream>"
    ) -> None:
        self._task_id = task_id
        self._transcript = _new_transcript({}, source=source, task_id=task_id)
        self._header: dict[str, Any] = {}
        self._pending: dict[str, ToolCall] = {}
        self._turn = 0
        self._last_text: str | None = None
        self._last_model: str | None = None
        self._last_ts = 0.0
        self._unhandled: dict[str, int] = {}

    def feed(self, event: dict[str, Any]) -> None:
        """Consume one stream event.

        Event types this mapping does not model are counted into
        ``transcript.metadata["unhandled_events"]`` rather than discarded — a
        count is enough to answer the question that matters: did something happen
        during this run that the transcript does not explain?
        """
        etype = event.get("type")
        if etype == "session":
            self._header = event
            _apply_header(self._transcript, event, task_id=self._task_id)
            return
        if etype == "message_end":
            self._consume_message(event.get("message") or {})
            return
        if etype in _STREAM_EVENT_TYPES:
            return  # bracketing frames; message_end already carried the payload
        if not _handle_meta_entry(self._transcript, event):
            self._unhandled[str(etype or "unknown")] = (
                self._unhandled.get(str(etype or "unknown"), 0) + 1
            )

    def feed_line(self, line: str) -> bool:
        """Consume one raw stdout line. Returns False if it was not JSON.

        pi prints the occasional non-JSON line to stdout, and a partial trace
        beats a crashed harness — so a bad line is reported, not raised.
        """
        line = line.strip()
        if not line:
            return False
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return False
        if not isinstance(event, dict):
            return False
        self.feed(event)
        return True

    def finish(self) -> Transcript:
        """The Transcript reconstructed so far. Safe to call mid-stream."""
        if self._unhandled:
            self._transcript.metadata["unhandled_events"] = dict(self._unhandled)
        _finalize(
            self._transcript,
            self._header,
            self._last_ts,
            self._last_text,
            self._last_model,
        )
        return self._transcript

    def _consume_message(self, message: dict[str, Any]) -> None:
        if message.get("role") == "assistant":
            self._turn += 1
        text, model = _handle_message(
            self._transcript, message, self._pending, self._turn
        )
        if text is not None:
            self._last_text = text
        if model:
            self._last_model = model
        ts = _ms_to_epoch(message.get("timestamp"))
        if ts:
            self._last_ts = ts


def _reconstruct_stream(
    events: list[dict[str, Any]], *, source: str
) -> Transcript:
    """Offline form: replay a saved event stream through the reconstructor."""
    reconstructor = PiStreamReconstructor(source=source)
    for event in events:
        reconstructor.feed(event)
    return reconstructor.finish()


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
    """A pi ISO timestamp as a *local* naive datetime.

    Local, not UTC, because the other end of the same subtraction comes from
    :func:`datetime.fromtimestamp`, which is local. Dropping the offset instead
    (the obvious ``replace(tzinfo=None)``) made every session's duration off by
    the host's UTC offset — eight hours of "latency" in Shanghai, none in London.
    """
    if not value or not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    if parsed.tzinfo is None:
        return parsed
    return parsed.astimezone().replace(tzinfo=None)


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _truncate(text: str, limit: int = _MAX_STEP_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "…"
