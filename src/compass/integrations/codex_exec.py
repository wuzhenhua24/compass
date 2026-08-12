"""Import a Codex CLI (``codex exec --json``) run into a Compass Transcript.

``codex exec --json`` prints one JSON object per line — a small, stable event
vocabulary rather than a provider-shaped message dump::

    {"type": "thread.started", "thread_id": "019ff67f-…"}
    {"type": "turn.started"}
    {"type": "item.completed", "item": {"id": "item_0", "type": "reasoning", "text": "…"}}
    {"type": "item.started",   "item": {"id": "item_1", "type": "command_execution",
                                        "command": "/bin/zsh -lc 'pytest -q'",
                                        "aggregated_output": "", "exit_code": null,
                                        "status": "in_progress"}}
    {"type": "item.completed", "item": {"id": "item_1", "type": "command_execution",
                                        "aggregated_output": "…", "exit_code": 0,
                                        "status": "completed"}}
    {"type": "turn.completed", "usage": {"input_tokens": 49951, "cached_input_tokens": 39424,
                                         "cache_write_input_tokens": 0,
                                         "output_tokens": 655, "reasoning_output_tokens": 50}}

This module maps that stream onto a Compass
:class:`~compass.core.transcript.Transcript`, live through
:class:`CodexStreamReconstructor` (what the ``codex`` adapter uses) or offline
through :func:`import_codex_stream_json` over a stream saved with the adapter's
``save_stream_to`` — so a run Compass drove and a stream a user captured by hand
grade identically.

Mapping (codex -> Compass):
- ``thread.started``            -> ``trial_id`` / ``metadata["thread_id"]``
- ``turn.completed``            -> an ``llm.generation`` ``ToolCall`` carrying the
  turn's ``TokenUsage`` (+ cost from Compass's pricing table; see below) and the
  turn's closing message as its output
- ``turn.failed``               -> the same call with ``status="error"``, plus
  ``metadata["turn_failures"]``
- ``command_execution`` item    -> a ``shell`` ``ToolCall`` (``exit_code != 0``
  reads as ``status="error"``, the way a failed ``Bash`` does on a Claude trace)
- ``file_change`` item          -> an ``apply_patch`` ``ToolCall`` + a
  ``StateChange`` per changed path (see below)
- ``mcp_tool_call`` item        -> a ``<server>.<tool>`` ``ToolCall``
- ``web_search`` / ``todo_list`` -> a ``web_search`` / ``update_plan`` ``ToolCall``
- ``reasoning`` item            -> a reasoning step
- ``agent_message`` item        -> a reasoning step, and (the last one of a turn)
  that turn's output — codex narrates *while* it works, so a run has several and
  only the last is the answer
- ``error`` item / ``error`` event -> ``metadata["errors"]`` + a reasoning step

**Three things this stream does not carry**, each of which shows up in the
transcript as an explicit gap rather than a plausible-looking number:

- **No timestamps.** Nothing in the event vocabulary is dated, so durations can
  only be measured by whoever is watching the stream arrive. Live, that is
  exactly right (``item.started`` -> ``item.completed`` is a real wall-clock
  span). Offline it would time the *replay*, so the importer records no
  durations at all and says so in ``metadata["timings"]``.
- **No model id.** ``codex exec`` never echoes which model ran, so the adapter
  passes the ``model`` it asked for. An imported stream has no model unless you
  name it — and with no model there is no cost, because there is nothing to
  price.
- **No dollar cost.** Codex reports tokens, never money (a ChatGPT-plan run is
  not billed per token at all). Cost is therefore *computed* from
  :func:`~compass.adapters.llm.calculate_cost`, which needs a pricing entry for
  the model: no entry means ``cost=None``, ``metadata["cost_unpriced_model"]``
  set, and a ``cost_budget`` grader that passes on $0.00 having measured
  nothing. Register a rate before reading that grader's verdict::

      from compass.adapters import register_pricing
      register_pricing("gpt-5.4-mini", {"input": 0.25, "output": 2.0, "cached": 0.025})

  or point ``COMPASS_PRICING_FILE`` at a JSON/YAML rate card.

**One ``llm.generation`` per turn, not per model round-trip.** Codex aggregates
usage over a whole turn and never reports the individual requests inside it, and
``codex exec`` is one turn by construction (one prompt, one completion). So a
codex transcript has exactly one ``llm.generation`` call holding the run's whole
token bill. It is honest, but it means ``turn_count`` with
``count_filter: "llm"`` always reads 1 on a codex trace: the comparable
efficiency number across stacks is the *tool-call* total, not the turn count.

**State delta — what the agent changed, not what it said.** Compass defines the
``StateChange`` slot and leaves capturing deltas to importers. ``file_change``
items name the paths they touch and label each one ``add`` / ``update`` /
``delete``, so the mapping is exact rather than inferred::

    graders:
      - name: state_delta
        config:
          forbid: [{kind: file, target: "tests/*"}]   # don't rewrite the tests

Same three properties as the Claude and pi mappings: only **completed** changes
count, targets are made **relative to the run's cwd** so a glob still matches
inside a throwaway worktree, and **shell-mediated changes are not captured** —
codex edits files through ``apply_patch``, but it can also ``rm`` one from the
shell, and that records nothing here. Read an empty ``state_delta`` as "nothing
recorded", never as "nothing changed".

Zero dependency on any codex package: the format is plain JSON, parsed directly.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable
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

# Max characters kept for a single reasoning step.
_MAX_STEP_CHARS = 4000

# Codex item type -> (tool name, tool_type). The tool names are codex's own —
# what the model actually calls — so a transcript reads the way the run did.
_ITEM_TOOLS: dict[str, tuple[str, str]] = {
    "command_execution": ("shell", "code"),
    "file_change": ("apply_patch", "file"),
    "mcp_tool_call": ("mcp", "mcp"),
    "web_search": ("web_search", "search"),
    "todo_list": ("update_plan", "function"),
}

# Item types that are narration, not action.
_TEXT_ITEMS = frozenset({"agent_message", "reasoning", "error"})

# codex ``file_change`` kinds -> StateChange ops.
_CHANGE_OPS: dict[str, str] = {"add": "create", "update": "update", "delete": "delete"}

# Item statuses that mean the step did not do what it set out to do.
_FAILED_STATUSES = frozenset({"failed", "errored", "aborted", "interrupted"})


class CodexStreamError(ValueError):
    """Raised when a file is not a recognizable ``codex exec --json`` stream."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_codex_stream_json(
    path: str | Path,
    *,
    task_id: str | None = None,
    model: str | None = None,
    cwd: str = "",
) -> Transcript:
    """Import a saved ``codex exec --json`` stream into a Transcript.

    Args:
        path: Path to the saved stream (JSON lines).
        task_id: Optional task id; defaults to the run's thread id.
        model: The model that ran. The stream does not record it, and without it
            there is no cost — see the module docstring.
        cwd: The directory codex ran in, used to make ``file_change`` targets
            relative. Defaults to the ``cwd`` recorded in the stream if the
            producer wrote one, else absolute paths are kept as-is.

    Returns:
        A reconstructed :class:`~compass.core.transcript.Transcript`.

    Raises:
        CodexStreamError: If no line in the file is a codex event.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).expanduser()
    return load_codex_stream(
        path.read_text(encoding="utf-8"),
        source=str(path),
        task_id=task_id,
        model=model,
        cwd=cwd,
    )


def load_codex_stream(
    content: str,
    *,
    source: str = "<string>",
    task_id: str | None = None,
    model: str | None = None,
    cwd: str = "",
) -> Transcript:
    """Reconstruct a Transcript from the raw text of a codex event stream.

    Same as :func:`import_codex_stream_json` but takes the file *content*
    directly, which is convenient for tests and in-memory pipelines.
    """
    reconstructor = CodexStreamReconstructor(
        task_id=task_id,
        source=source,
        model=model,
        cwd=cwd,
        record_timings=False,  # replaying would time the replay; see the docstring
    )
    seen = 0
    for line in content.splitlines():
        if reconstructor.feed_line(line):
            seen += 1
    if not seen:
        raise CodexStreamError(
            f"{source}: no `codex exec --json` events found. Expected JSON lines "
            "with a 'type' of thread.started / turn.* / item.*."
        )
    return reconstructor.finish()


def looks_like_codex_stream(obj: Any) -> bool:
    """Whether a parsed JSON line is a codex event (used for format sniffing).

    Codex's event names are namespaced (``thread.``/``turn.``/``item.``), which
    no other trace format in Compass uses — so one line is enough to tell.
    """
    if not isinstance(obj, dict):
        return False
    etype = obj.get("type")
    if not isinstance(etype, str):
        return False
    return etype.startswith(("thread.", "turn.", "item."))


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------


class CodexStreamReconstructor:
    """Incremental mapping, fed ``codex exec --json`` lines as they arrive.

    ::

        r = CodexStreamReconstructor(task_id="add-rate-limit", model="gpt-5.4-mini")
        for line in proc.stdout:
            r.feed_line(line)      # non-JSON lines return False, never raise
        transcript = r.finish()

    :meth:`finish` is valid at any point, so a run killed by a timeout still
    yields the steps it completed.

    Args:
        task_id: Task id for the transcript; defaults to the thread id.
        source: Where the stream came from, for diagnostics.
        model: The model that ran (the stream never says).
        cwd: The run's working directory, used to relativize changed paths.
        record_timings: Measure durations from the wall clock as events arrive.
            True for a live run; False when replaying a saved stream, where the
            only thing there would be to measure is the replay itself.
    """

    def __init__(
        self,
        *,
        task_id: str | None = None,
        source: str = "<stream>",
        model: str | None = None,
        cwd: str = "",
        record_timings: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._task_id = task_id
        self._model = model or ""
        self._cwd = cwd
        self._record_timings = record_timings
        self._clock = clock

        transcript = Transcript(
            task_id=task_id or "codex_run", trial_id="codex_run"
        )
        transcript.input_params = {"cwd": cwd, "source": source}
        transcript.metadata = {"importer": "codex", "cwd": cwd}
        if model:
            transcript.metadata["model"] = model
        if not record_timings:
            transcript.metadata["timings"] = (
                "not recorded — codex --json events carry no timestamps"
            )
        self._transcript = transcript

        self._pending: dict[str, ToolCall] = {}
        self._turn = 0
        self._turn_message: str | None = None
        self._last_text: str | None = None
        self._start_ts: float = 0.0
        self._last_ts: float = 0.0
        self._unhandled: dict[str, int] = {}

    # -- feeding -------------------------------------------------------

    def feed_line(self, line: str) -> bool:
        """Consume one raw stdout line. False if it was not a codex event.

        Codex prints the occasional non-JSON line to stdout, and a partial trace
        beats a crashed harness — so a bad line is reported, not raised.
        """
        line = line.strip()
        if not line:
            return False
        try:
            event = json.loads(line)
        except (ValueError, TypeError):
            return False
        if not looks_like_codex_stream(event):
            # A top-level {"type": "error", ...} is codex's, and worth keeping.
            if isinstance(event, dict) and event.get("type") == "error":
                self._record_error("stream", event.get("message"))
                return True
            return False
        self.feed(event)
        return True

    def feed(self, event: dict[str, Any]) -> None:
        """Consume one stream event.

        Event types this mapping does not model are counted into
        ``metadata["unhandled_events"]`` rather than discarded — a count is
        enough to answer the question that matters: did something happen during
        this run that the transcript does not explain?
        """
        etype = str(event.get("type") or "")
        self._tick()

        if etype == "thread.started":
            self._on_thread_started(event)
        elif etype == "turn.started":
            self._turn += 1
        elif etype == "turn.completed":
            self._on_turn_end(event, failed=False)
        elif etype == "turn.failed":
            self._on_turn_end(event, failed=True)
        elif etype in ("item.started", "item.updated", "item.completed"):
            self._on_item(etype, event.get("item") or {})
        else:
            self._unhandled[etype or "unknown"] = (
                self._unhandled.get(etype or "unknown", 0) + 1
            )

    def finish(self) -> Transcript:
        """The Transcript reconstructed so far. Safe to call mid-stream."""
        t = self._transcript

        # A turn that never closed (a killed run) still had a last word.
        if self._turn_message and not self._last_text:
            self._last_text = self._turn_message

        incomplete = sorted(self._pending)
        if incomplete:
            for call in self._pending.values():
                call.metadata["incomplete"] = True
            t.metadata["incomplete_items"] = incomplete

        t.metadata["turns"] = self._turn
        if self._unhandled:
            t.metadata["unhandled_events"] = dict(self._unhandled)
        if self._model:
            t.environment.model_version = self._model

        if self._record_timings and self._start_ts:
            start = datetime.fromtimestamp(self._start_ts)
            t.start_time = start
            if self._last_ts >= self._start_ts:
                t.end_time = datetime.fromtimestamp(self._last_ts)
                t.total_duration_ms = (self._last_ts - self._start_ts) * 1000.0

        if self._last_text:
            t.set_outcome(
                output_data={"final_output": self._last_text},
                metadata={"thread_id": t.metadata.get("thread_id")},
            )
        return t

    # -- events --------------------------------------------------------

    def _on_thread_started(self, event: dict[str, Any]) -> None:
        thread_id = str(event.get("thread_id") or "")
        if not thread_id:
            return
        self._transcript.trial_id = thread_id
        if not self._task_id:
            self._transcript.task_id = thread_id
        self._transcript.metadata["thread_id"] = thread_id

    def _on_turn_end(self, event: dict[str, Any], *, failed: bool) -> None:
        """Close a turn: its token bill, its closing message, its verdict."""
        if self._turn == 0:  # a stream that never announced turn.started
            self._turn = 1

        error: dict[str, Any] | None = None
        if failed:
            message = _error_message(event.get("error"))
            error = {"message": message}
            self._transcript.metadata.setdefault("turn_failures", []).append(message)

        tokens, cost = _usage_to_tokens_cost(event.get("usage"), self._model)
        if cost is None and tokens is not None and self._model:
            self._transcript.metadata["cost_unpriced_model"] = self._model

        self._transcript.tool_calls.append(
            ToolCall(
                tool_name="llm.generation",
                input={"model": self._model} if self._model else {},
                output=self._turn_message,
                status="error" if failed else "ok",
                timestamp=self._last_ts or self._clock(),
                error=error,
                cost=cost,
                tokens=tokens,
                tool_type="llm",
                metadata={"model": self._model} if self._model else {},
                turn_index=self._turn,
            )
        )

        if self._turn_message:
            self._last_text = self._turn_message
        self._turn_message = None

    def _on_item(self, etype: str, item: dict[str, Any]) -> None:
        kind = str(item.get("type") or "")
        if kind in _TEXT_ITEMS:
            if etype == "item.completed":
                self._on_text_item(kind, item)
            return
        if kind not in _ITEM_TOOLS:
            self._unhandled[f"item:{kind or 'unknown'}"] = (
                self._unhandled.get(f"item:{kind or 'unknown'}", 0) + 1
            )
            return

        item_id = str(item.get("id") or "")
        call = self._pending.get(item_id)
        if call is None:
            call = self._open_call(kind, item, item_id)
        if etype == "item.started":
            return

        _apply_item(call, kind, item)
        if etype == "item.updated":
            call.metadata["updates"] = int(call.metadata.get("updates", 0)) + 1
            return

        # item.completed — the step is over.
        self._close_call(call, kind, item, item_id)

    def _on_text_item(self, kind: str, item: dict[str, Any]) -> None:
        if kind == "reasoning":
            text = str(item.get("text") or "")
            if text:
                self._transcript.add_reasoning_step(f"[reasoning] {_truncate(text)}")
            return
        if kind == "error":
            self._record_error("item", item.get("message"))
            return

        # agent_message: codex narrates *while* it works, so a run has several
        # of these and only the last is the answer. They all land in the
        # trajectory, in the order they were said; the last one is additionally
        # the turn's output and the run's final answer.
        text = str(item.get("text") or "")
        if text:
            self._transcript.add_reasoning_step(f"[message] {_truncate(text)}")
            self._turn_message = text

    def _record_error(self, where: str, message: Any) -> None:
        text = _truncate(str(message or "unknown error"), 500)
        self._transcript.metadata.setdefault("errors", []).append(
            {"source": where, "message": text}
        )
        self._transcript.add_reasoning_step(f"[error] {text}")

    # -- tool calls ----------------------------------------------------

    def _open_call(self, kind: str, item: dict[str, Any], item_id: str) -> ToolCall:
        tool_name, tool_type = _ITEM_TOOLS[kind]
        if kind == "mcp_tool_call":
            tool_name = _mcp_tool_name(item)
        kwargs: dict[str, Any] = dict(
            tool_name=tool_name,
            input={},
            status="ok",
            timestamp=self._last_ts or self._clock(),
            tool_type=tool_type,
            metadata={"item_type": kind},
            turn_index=self._turn or 1,
        )
        if item_id:
            kwargs["call_id"] = item_id
        call = ToolCall(**kwargs)
        if self._record_timings:
            call.metadata["call_ts"] = call.timestamp
        self._transcript.tool_calls.append(call)
        if item_id:
            self._pending[item_id] = call
        return call

    def _close_call(
        self, call: ToolCall, kind: str, item: dict[str, Any], item_id: str
    ) -> None:
        self._pending.pop(item_id, None)
        call_ts = call.metadata.pop("call_ts", None)
        if call_ts and self._record_timings:
            call.duration_ms = max(0.0, (self._clock() - float(call_ts)) * 1000.0)
        if call.status != "error":
            self._record_state_delta(call, kind, item)

    def _record_state_delta(
        self, call: ToolCall, kind: str, item: dict[str, Any]
    ) -> None:
        """Attach the file changes a completed ``apply_patch`` just made.

        Recorded at completion rather than when the item opened: until then the
        change has not happened, and it may never (a rejected patch, a killed
        run). ``state_delta`` records what the environment *did*.
        """
        if kind != "file_change":
            return
        for change in item.get("changes") or []:
            if not isinstance(change, dict):
                continue
            raw_path = change.get("path")
            if not isinstance(raw_path, str) or not raw_path:
                continue
            target, absolute = _relativize(raw_path, self._cwd)
            metadata: dict[str, Any] = {"tool": call.tool_name}
            if absolute and absolute != target:
                metadata["absolute_path"] = absolute
            call.state_delta.append(
                StateChange(
                    kind="file",
                    op=_CHANGE_OPS.get(str(change.get("kind") or ""), "update"),
                    target=target,
                    metadata=metadata,
                )
            )

    # -- clock ---------------------------------------------------------

    def _tick(self) -> None:
        if not self._record_timings:
            return
        now = self._clock()
        if not self._start_ts:
            self._start_ts = now
        self._last_ts = now


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _apply_item(call: ToolCall, kind: str, item: dict[str, Any]) -> None:
    """Fold an item's payload into its ToolCall (each event carries the whole item)."""
    if kind == "command_execution":
        call.input = {"command": item.get("command", "")}
        if item.get("cwd"):
            call.input["cwd"] = item["cwd"]
        call.output = item.get("aggregated_output", "")
        exit_code = item.get("exit_code")
        if exit_code is not None:
            call.metadata["exit_code"] = exit_code
        failed = _item_failed(item) or (
            isinstance(exit_code, int) and exit_code != 0
        )
        if failed:
            call.status = "error"
            call.error = {
                "message": _truncate(str(call.output or ""), 500) or "command failed",
                "exit_code": exit_code,
            }
        return

    if kind == "file_change":
        call.input = {
            "changes": [
                {"path": c.get("path", ""), "kind": c.get("kind", "")}
                for c in (item.get("changes") or [])
                if isinstance(c, dict)
            ]
        }
        _mark_failed(call, item)
        return

    if kind == "mcp_tool_call":
        call.tool_name = _mcp_tool_name(item)
        call.input = _as_dict(item.get("arguments"))
        if item.get("server"):
            call.metadata["server"] = item["server"]
        call.output = item.get("result")
        _mark_failed(call, item)
        return

    if kind == "web_search":
        call.input = {"query": item.get("query", "")}
        _mark_failed(call, item)
        return

    if kind == "todo_list":
        call.input = {"items": item.get("items") or []}
        _mark_failed(call, item)
        return


def _mark_failed(call: ToolCall, item: dict[str, Any]) -> None:
    if _item_failed(item):
        call.status = "error"
        call.error = {"message": str(item.get("status") or "failed")}


def _item_failed(item: dict[str, Any]) -> bool:
    return str(item.get("status") or "") in _FAILED_STATUSES


def _mcp_tool_name(item: dict[str, Any]) -> str:
    """``<server>.<tool>`` for an MCP call, with whatever the item actually names."""
    server = str(item.get("server") or "").strip()
    tool = str(item.get("tool") or item.get("name") or item.get("tool_name") or "").strip()
    if server and tool:
        return f"{server}.{tool}"
    return tool or server or "mcp"


def _usage_to_tokens_cost(
    usage: Any, model: str
) -> tuple[TokenUsage | None, CostInfo | None]:
    """Map a codex turn's ``usage`` to Compass ``TokenUsage`` + ``CostInfo``.

    Codex's ``input_tokens`` is the whole prompt bill, cached traffic included;
    Compass keeps cached traffic out of ``input_tokens`` and reports it
    separately, because it is priced differently. So the fresh input is what is
    left after the cache reads and writes are taken out.

    ``reasoning_output_tokens`` is a *subset* of ``output_tokens`` (it is how
    much of the output was thinking), so it goes in metadata rather than being
    added to anything.
    """
    if not isinstance(usage, dict):
        return None, None

    prompt = int(usage.get("input_tokens", 0) or 0)
    cache_read = int(usage.get("cached_input_tokens", 0) or 0)
    cache_write = int(usage.get("cache_write_input_tokens", 0) or 0)
    output = int(usage.get("output_tokens", 0) or 0)
    fresh_input = max(0, prompt - cache_read - cache_write)

    token_meta: dict[str, Any] = {"prompt_tokens_including_cache": prompt}
    if usage.get("reasoning_output_tokens"):
        token_meta["reasoning_output_tokens"] = usage["reasoning_output_tokens"]

    tokens = TokenUsage(
        input_tokens=fresh_input,
        output_tokens=output,
        total_tokens=prompt + output,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
        metadata=token_meta,
    )

    # Codex reports no dollar cost at all (see the module docstring), so this is
    # Compass's own pricing table or nothing.
    cost = calculate_cost(model, fresh_input, output, cache_read) if model else None
    return tokens, cost


def _error_message(error: Any) -> str:
    if isinstance(error, dict):
        return _truncate(str(error.get("message") or error), 500)
    return _truncate(str(error or "turn failed"), 500)


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce tool arguments into a dict (codex records them as an object)."""
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def _relativize(raw_path: str, cwd: str) -> tuple[str, str]:
    """(target, absolute) — target relative to the run's cwd, normalized.

    Codex reports ``file_change`` paths absolute, and a codex trial runs in a
    throwaway worktree: left alone, every run's target is a different random
    path and no glob a user writes could match it. The absolute form is kept on
    the change's metadata.

    Paths are normalized first, so one file has one spelling. Without it
    ``/wt/repo/../repo/app.py`` relativizes *lexically* to ``../repo/app.py`` —
    a target that points outside a workspace the change never left.
    """
    normalized = os.path.normpath(raw_path)
    path = Path(normalized)
    if not path.is_absolute():
        return normalized, (
            os.path.normpath(os.path.join(cwd, normalized)) if cwd else normalized
        )
    if cwd:
        base = Path(cwd)
        # Compare resolved forms too. The two paths come from different places —
        # the workspace from ``tempfile``, the changed path from codex's own
        # process — and on macOS those disagree by a symlink: ``/var/folders/…``
        # against ``/private/var/folders/…`` for the same directory. Left
        # unresolved, every target stays absolute and every ``state_delta`` glob
        # silently stops matching.
        for candidate, root in (
            (path, base),
            (path, _resolved(base)),
            (_resolved(path), _resolved(base)),
        ):
            try:
                return str(candidate.relative_to(root)), str(path)
            except ValueError:
                continue  # outside the run's cwd — the absolute path *is* the target
    return str(path), str(path)


def _resolved(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:  # a broken symlink or a path that no longer exists
        return path


def _truncate(text: str, limit: int = _MAX_STEP_CHARS) -> str:
    return text if len(text) <= limit else text[:limit] + "…"
