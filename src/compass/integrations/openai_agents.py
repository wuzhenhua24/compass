"""Bridge the OpenAI Agents SDK's native tracing into a Compass Transcript.

The OpenAI Agents SDK (``agents`` package) records execution as typed *spans*
(``function`` / ``generation`` / ``response`` / ``agent`` / ``turn`` /
``handoff`` / ``guardrail`` …) and lets you register a ``TracingProcessor`` that
receives span lifecycle callbacks. This module implements such a processor that
reconstructs a Compass :class:`~compass.core.transcript.Transcript` — so any
agent built on the SDK can be evaluated with Compass's transcript-scope graders
**without writing a per-agent adapter**::

    from agents import Agent, Runner
    from compass.integrations import install_openai_agents_processor

    proc = install_openai_agents_processor()   # one line, then run your agent
    await Runner.run(agent, "...")
    transcript = proc.latest                    # a Compass Transcript, ready to grade

Design notes:
- Zero hard dependency on ``agents``: the processor is duck-typed and only reads
  attributes off the span/trace objects the SDK hands it. ``agents`` is imported
  lazily, and only inside :func:`install_openai_agents_processor`.
- Span → ToolCall mapping: ``function`` (tool/MCP call) and ``generation`` /
  ``response`` (LLM call) become ``ToolCall`` entries. ``agent`` / ``turn`` are
  used to tag each call with ``agent_name`` / ``turn_index`` (via parent walk).
  ``handoff`` / ``guardrail`` are recorded in transcript metadata.
- Token usage → ``TokenUsage``; cost is computed with Compass's configurable
  pricing (``calculate_cost``), since SDK spans carry tokens but not dollars.
"""

from __future__ import annotations

import json
import logging
import threading
from collections.abc import Callable
from datetime import datetime
from typing import Any

# Imported from the submodule, not the ``compass.adapters`` package: an adapter
# that consumes an integration (claude_code) makes the package-level import a
# cycle, while the submodule only depends on compass.core.
from compass.adapters.llm import calculate_cost
from compass.core.transcript import TokenUsage, ToolCall, Transcript

logger = logging.getLogger(__name__)

# Span types that represent an actual unit of work we record as a ToolCall.
_LLM_SPAN_TYPES = frozenset({"generation", "response"})
_TOOL_SPAN_TYPES = frozenset({"function"})
# Span types used only to resolve enclosing agent / turn context.
_CONTEXT_SPAN_TYPES = frozenset({"agent", "turn"})


def _parse_iso(value: Any) -> datetime | None:
    """Parse an ISO-8601 timestamp string (tolerating a trailing 'Z')."""
    if not value or not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None


def _as_input_dict(value: Any) -> dict[str, Any]:
    """Coerce a span's input (dict, JSON string, list, …) into a dict."""
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return {"input": value}
        return parsed if isinstance(parsed, dict) else {"input": parsed}
    if value is None:
        return {}
    return {"input": value}


class CompassTraceProcessor:
    """An OpenAI Agents SDK ``TracingProcessor`` that builds Compass Transcripts.

    Register it via :func:`install_openai_agents_processor` (recommended) or
    ``agents.add_trace_processor(CompassTraceProcessor())``. After a run, read
    the reconstructed transcripts from :attr:`transcripts` / :attr:`latest`.

    Args:
        on_trace_complete: Optional callback invoked with each finished
            ``Transcript`` (e.g. to grade or persist it immediately).
    """

    def __init__(
        self,
        on_trace_complete: Callable[[Transcript], None] | None = None,
    ) -> None:
        self._on_complete = on_trace_complete
        self._lock = threading.Lock()
        # trace_id -> in-progress Transcript
        self._active: dict[str, Transcript] = {}
        # span_id -> span object (kept for parent-context resolution)
        self._spans: dict[str, Any] = {}
        # completed transcripts, in completion order
        self.transcripts: list[Transcript] = []

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def latest(self) -> Transcript | None:
        """The most recently completed transcript, or None."""
        return self.transcripts[-1] if self.transcripts else None

    def get_transcript(self, trace_id: str) -> Transcript | None:
        """Return a completed transcript by its trace id, or None."""
        for t in self.transcripts:
            if t.trial_id == trace_id:
                return t
        return None

    # ------------------------------------------------------------------
    # TracingProcessor interface (duck-typed — called by the SDK)
    # ------------------------------------------------------------------

    def on_trace_start(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", "") or ""
        name = getattr(trace, "name", None) or trace_id or "openai_agents"
        transcript = Transcript(task_id=name, trial_id=trace_id)
        # Best-effort trace-level metadata.
        meta = _safe_getattr_dict(trace, "metadata")
        if meta:
            transcript.input_params = dict(meta)
        with self._lock:
            self._active[trace_id] = transcript

    def on_span_start(self, span: Any) -> None:
        span_id = getattr(span, "span_id", None)
        if span_id is not None:
            with self._lock:
                self._spans[span_id] = span

    def on_span_end(self, span: Any) -> None:
        trace_id = getattr(span, "trace_id", "") or ""
        with self._lock:
            transcript = self._active.get(trace_id)
        if transcript is None:
            return
        try:
            self._record_span(transcript, span)
        except Exception as exc:  # never disrupt the agent run
            logger.warning("Failed to map OpenAI Agents span: %s", exc)

    def on_trace_end(self, trace: Any) -> None:
        trace_id = getattr(trace, "trace_id", "") or ""
        with self._lock:
            transcript = self._active.pop(trace_id, None)
            # Drop this trace's spans from the parent-walk cache.
            self._spans = {
                sid: sp
                for sid, sp in self._spans.items()
                if getattr(sp, "trace_id", None) != trace_id
            }
        if transcript is None:
            return

        self._finalize(transcript)
        with self._lock:
            self.transcripts.append(transcript)
        if self._on_complete is not None:
            try:
                self._on_complete(transcript)
            except Exception as exc:  # pragma: no cover - user callback
                logger.warning("on_trace_complete callback raised: %s", exc)

    def shutdown(self) -> None:
        return None

    def force_flush(self) -> None:
        return None

    # ------------------------------------------------------------------
    # Mapping internals
    # ------------------------------------------------------------------

    def _record_span(self, transcript: Transcript, span: Any) -> None:
        data = getattr(span, "span_data", None)
        span_type = getattr(data, "type", None)
        if span_type is None:
            return

        if span_type in _CONTEXT_SPAN_TYPES:
            return  # used only for parent context, not recorded directly

        if span_type == "handoff":
            frm = getattr(data, "from_agent", None)
            to = getattr(data, "to_agent", None)
            transcript.metadata.setdefault("handoffs", []).append(
                {"from": frm, "to": to}
            )
            transcript.add_reasoning_step(f"Handoff: {frm} -> {to}")
            return

        if span_type == "guardrail":
            transcript.metadata.setdefault("guardrails", []).append(
                {
                    "name": getattr(data, "name", None),
                    "triggered": getattr(data, "triggered", False),
                }
            )
            return

        if span_type in _TOOL_SPAN_TYPES:
            self._record_tool_span(transcript, span, data)
        elif span_type in _LLM_SPAN_TYPES:
            self._record_llm_span(transcript, span, data, span_type)
        # other span types (task/custom/mcp_list_tools/speech/…) are ignored in v1

    def _record_tool_span(self, transcript: Transcript, span: Any, data: Any) -> None:
        duration_ms, timestamp = _timing(span)
        status, error = _status(span)
        tool_type = "mcp" if getattr(data, "mcp_data", None) else "function"
        turn_index, agent_name = self._context(span)
        metadata: dict[str, Any] = {}
        mcp = getattr(data, "mcp_data", None)
        if mcp:
            metadata["mcp_data"] = mcp

        transcript.tool_calls.append(
            ToolCall(
                tool_name=getattr(data, "name", "") or "unknown_tool",
                input=_as_input_dict(getattr(data, "input", None)),
                output=getattr(data, "output", None),
                status=status,
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=error,
                tool_type=tool_type,
                metadata=metadata,
                turn_index=turn_index,
                agent_name=agent_name,
            )
        )

    def _record_llm_span(
        self, transcript: Transcript, span: Any, data: Any, span_type: str
    ) -> None:
        duration_ms, timestamp = _timing(span)
        status, error = _status(span)

        model = getattr(data, "model", None)
        if model is None:  # ResponseSpanData keeps the model on the Response
            model = getattr(getattr(data, "response", None), "model", None)

        usage = _safe_getattr_dict(data, "usage")
        tokens = None
        cost = None
        if usage:
            in_tok = int(usage.get("input_tokens", 0) or 0)
            out_tok = int(usage.get("output_tokens", 0) or 0)
            tokens = TokenUsage(input_tokens=in_tok, output_tokens=out_tok)
            if model:
                cost = calculate_cost(model, in_tok, out_tok)

        turn_index, agent_name = self._context(span)
        metadata: dict[str, Any] = {}
        if model:
            metadata["model"] = model

        transcript.tool_calls.append(
            ToolCall(
                tool_name=f"llm.{span_type}",
                input=_as_input_dict(getattr(data, "input", None)),
                output=getattr(data, "output", None),
                status=status,
                duration_ms=duration_ms,
                timestamp=timestamp,
                error=error,
                cost=cost,
                tokens=tokens,
                tool_type="llm",
                metadata=metadata,
                turn_index=turn_index,
                agent_name=agent_name,
            )
        )

    def _context(self, span: Any) -> tuple[int | None, str | None]:
        """Walk parent spans to resolve the enclosing turn_index / agent_name."""
        turn_index: int | None = None
        agent_name: str | None = None
        seen: set[str] = set()
        parent_id = getattr(span, "parent_id", None)
        while parent_id and parent_id not in seen:
            seen.add(parent_id)
            parent = self._spans.get(parent_id)
            if parent is None:
                break
            pdata = getattr(parent, "span_data", None)
            ptype = getattr(pdata, "type", None)
            if ptype == "turn" and turn_index is None:
                turn_index = getattr(pdata, "turn", None)
                if agent_name is None:
                    agent_name = getattr(pdata, "agent_name", None)
            elif ptype == "agent" and agent_name is None:
                agent_name = getattr(pdata, "name", None)
            parent_id = getattr(parent, "parent_id", None)
        return turn_index, agent_name

    def _finalize(self, transcript: Transcript) -> None:
        """Compute wall-clock timing and a best-effort outcome."""
        starts = [tc.timestamp for tc in transcript.tool_calls if tc.timestamp]
        if starts:
            ends = [
                tc.timestamp + (tc.duration_ms / 1000.0)
                for tc in transcript.tool_calls
            ]
            transcript.total_duration_ms = (max(ends) - min(starts)) * 1000.0
            transcript.start_time = datetime.fromtimestamp(min(starts))
            transcript.end_time = datetime.fromtimestamp(max(ends))

        # Best-effort outcome: the last LLM output produced in the run.
        final_output = None
        for tc in reversed(transcript.tool_calls):
            if tc.tool_type == "llm" and tc.output is not None:
                final_output = tc.output
                break
        if final_output is not None:
            transcript.set_outcome(output_data={"final_output": final_output})


def _timing(span: Any) -> tuple[float, float]:
    """Return (duration_ms, start_timestamp_epoch) from a span."""
    start = _parse_iso(getattr(span, "started_at", None))
    end = _parse_iso(getattr(span, "ended_at", None))
    timestamp = start.timestamp() if start else 0.0
    if start and end:
        duration_ms = (end - start).total_seconds() * 1000.0
    else:
        duration_ms = 0.0
    return duration_ms, timestamp


def _status(span: Any) -> tuple[str, dict[str, Any] | None]:
    """Return (status, error_dict) from a span's error field."""
    err = getattr(span, "error", None)
    if err:
        if isinstance(err, dict):
            return "error", err
        # SpanError-like object
        return "error", {
            "message": getattr(err, "message", str(err)),
            "data": getattr(err, "data", None),
        }
    return "ok", None


def _safe_getattr_dict(obj: Any, name: str) -> dict[str, Any]:
    """Return obj.<name> if it is a dict, else {}."""
    value = getattr(obj, name, None)
    return value if isinstance(value, dict) else {}


def install_openai_agents_processor(
    on_trace_complete: Callable[[Transcript], None] | None = None,
    *,
    replace: bool = False,
) -> CompassTraceProcessor:
    """Register a :class:`CompassTraceProcessor` with the OpenAI Agents SDK.

    Args:
        on_trace_complete: Optional callback invoked with each finished Transcript.
        replace: If True, replace the SDK's default processors (traces will no
            longer be sent to OpenAI's backend). If False (default), add ours
            alongside the existing processors.

    Returns:
        The registered CompassTraceProcessor. Read ``.latest`` / ``.transcripts``
        after your agent run.

    Raises:
        ImportError: If the ``openai-agents`` package is not installed.
    """
    try:
        from agents import (  # type: ignore[import-not-found]
            add_trace_processor,
            set_trace_processors,
        )
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise ImportError(
            "install_openai_agents_processor requires the 'openai-agents' "
            "package. Install it with: pip install openai-agents"
        ) from exc

    processor = CompassTraceProcessor(on_trace_complete=on_trace_complete)
    if replace:
        set_trace_processors([processor])
    else:
        add_trace_processor(processor)
    return processor
