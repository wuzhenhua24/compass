"""Import OTLP / OpenInference trace JSON into Compass Transcripts (offline).

Most agent frameworks that don't ship their own Compass-friendly trace
(LangChain, LlamaIndex, CrewAI, Haystack, DSPy, …) *can* be instrumented with
**OpenInference** and exported over **OpenTelemetry (OTLP)** — e.g. through Arize
Phoenix. This module reads such an exported trace file and reconstructs one
Compass :class:`~compass.core.transcript.Transcript` per trace, so those agents
can be graded with Compass's transcript-scope graders **without a per-framework
adapter**::

    from compass.integrations import import_otlp_file

    transcripts = import_otlp_file("phoenix_export.json")   # one per trace
    t = transcripts[0]
    # t.tool_calls / t.reasoning_steps / t.outcome are ready to grade

Accepted shapes (auto-detected):
- OTLP/JSON envelope: ``{"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}``
  with attributes as ``[{"key", "value": {"stringValue": …}}]``.
- A flat list of spans, or the OpenTelemetry SDK's ``ReadableSpan.to_json()``
  shape (``{"context": {"trace_id", "span_id"}, "attributes": {flat dict}, …}``).
- JSON *or* JSON-lines (one span/envelope per line).

Span kind (``openinference.span.kind``) drives the mapping:
- ``LLM`` / ``EMBEDDING``  -> ``llm``-type ``ToolCall`` (tokens + cost)
- ``TOOL``                 -> ``function``-type ``ToolCall``
- ``RETRIEVER``            -> ``search``-type ``ToolCall`` (documents as output)
- ``AGENT`` / ``CHAIN``    -> structural (agent-name tagging + root input/output)
- ``GUARDRAIL`` / ``RERANKER`` / ``EVALUATOR`` -> ``Transcript.metadata`` / reasoning

Cost uses native ``llm.cost.*`` attributes when present, else Compass's
configurable pricing table. Zero dependency on any OpenTelemetry package — the
export is plain JSON, parsed directly.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from compass.adapters import calculate_cost
from compass.core.transcript import CostInfo, TokenUsage, ToolCall, Transcript

logger = logging.getLogger(__name__)

# --- OpenInference semantic-convention keys (verified against the spec) -------
_KIND = "openinference.span.kind"
_INPUT = "input.value"
_OUTPUT = "output.value"
_LLM_MODEL = "llm.model_name"
_LLM_PROVIDER = "llm.provider"
_LLM_SYSTEM = "llm.system"
_TOK_PROMPT = "llm.token_count.prompt"
_TOK_COMPLETION = "llm.token_count.completion"
_TOK_TOTAL = "llm.token_count.total"
_TOK_CACHE_READ = "llm.token_count.prompt_details.cache_read"
_COST_PROMPT = "llm.cost.prompt"
_COST_COMPLETION = "llm.cost.completion"
_COST_TOTAL = "llm.cost.total"
_TOOL_NAME = "tool.name"
_TOOL_PARAMS = "tool.parameters"
_EMBED_MODEL = "embedding.model_name"
_SESSION_ID = "session.id"


class OTLPImportError(ValueError):
    """Raised when a payload is not recognizable OTLP / OpenInference trace JSON."""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def import_otlp_file(path: str | Path) -> list[Transcript]:
    """Import an OTLP / OpenInference trace file, one Transcript per trace.

    Args:
        path: Path to a ``.json`` (or ``.jsonl``) trace export.

    Returns:
        Transcripts ordered by trace start time (may be empty if the file holds
        no spans).

    Raises:
        OTLPImportError: If the payload is not valid trace JSON.
        FileNotFoundError: If ``path`` does not exist.
    """
    path = Path(path).expanduser()
    return load_otlp(path.read_text(encoding="utf-8"))


def load_otlp(content: str) -> list[Transcript]:
    """Reconstruct Transcripts from the raw text of a trace export (JSON or JSONL)."""
    data = _parse_payload(content)
    return otlp_to_transcripts(data)


def otlp_to_transcripts(data: Any) -> list[Transcript]:
    """Convert already-parsed OTLP/OpenInference data into Transcripts.

    Accepts an OTLP envelope dict, a single span dict, or a list of either.
    """
    raw_spans = _collect_raw_spans(data)
    spans = [s for s in (_normalize_span(r) for r in raw_spans) if s is not None]
    if not spans:
        return []

    # Group by trace, preserving first-seen order; sort groups by start time.
    by_trace: dict[str, list[_Span]] = {}
    for span in spans:
        by_trace.setdefault(span.trace_id, []).append(span)

    transcripts = [_build_transcript(tid, group) for tid, group in by_trace.items()]
    transcripts.sort(key=lambda t: t.start_time)
    return transcripts


# ---------------------------------------------------------------------------
# Payload parsing / span collection
# ---------------------------------------------------------------------------


def _parse_payload(content: str) -> Any:
    """Parse a trace file that is either one JSON document or JSON-lines."""
    stripped = content.strip()
    if not stripped:
        raise OTLPImportError("empty trace payload")
    try:
        return json.loads(stripped)
    except (ValueError, TypeError):
        pass
    # Fall back to JSON-lines: collect every parseable line.
    items: list[Any] = []
    for i, line in enumerate(stripped.split("\n"), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            items.append(json.loads(line))
        except (ValueError, TypeError):
            logger.warning("OTLP import: line %d is not valid JSON, skipping", i)
    if not items:
        raise OTLPImportError("payload is neither valid JSON nor JSON-lines")
    return items


def _collect_raw_spans(data: Any) -> list[dict[str, Any]]:
    """Flatten any accepted shape into a list of raw span dicts."""
    spans: list[dict[str, Any]] = []

    def visit(node: Any) -> None:
        if isinstance(node, list):
            for item in node:
                visit(item)
            return
        if not isinstance(node, dict):
            return
        # OTLP envelope(s).
        if "resourceSpans" in node:
            for rs in node.get("resourceSpans") or []:
                scope_spans = rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []
                for ss in scope_spans:
                    for sp in ss.get("spans") or []:
                        spans.append(sp)
            return
        if "scopeSpans" in node:
            for ss in node.get("scopeSpans") or []:
                for sp in ss.get("spans") or []:
                    spans.append(sp)
            return
        if "spans" in node and isinstance(node["spans"], list):
            for sp in node["spans"]:
                spans.append(sp)
            return
        # Otherwise assume this dict is itself a span.
        spans.append(node)

    visit(data)
    return spans


# ---------------------------------------------------------------------------
# Span normalization
# ---------------------------------------------------------------------------


@dataclass
class _Span:
    trace_id: str
    span_id: str
    parent_id: str | None
    name: str
    kind: str
    attrs: dict[str, Any]
    start_s: float
    end_s: float
    is_error: bool
    status_msg: str | None = field(default=None)


def _normalize_span(raw: dict[str, Any]) -> _Span | None:
    if not isinstance(raw, dict):
        return None
    raw_ctx = raw.get("context")
    ctx: dict[str, Any] = raw_ctx if isinstance(raw_ctx, dict) else {}
    trace_id = raw.get("traceId") or raw.get("trace_id") or ctx.get("trace_id")
    span_id = raw.get("spanId") or raw.get("span_id") or ctx.get("span_id")
    if not trace_id or not span_id:
        return None
    parent = (
        raw.get("parentSpanId")
        or raw.get("parent_span_id")
        or raw.get("parentId")
        or raw.get("parent_id")
    )
    parent_id = str(parent) if parent else None

    attrs = _attributes_to_dict(raw.get("attributes"))
    kind = str(attrs.get(_KIND, "") or "").upper()
    start_s = _span_time(raw, "startTimeUnixNano", "start_time")
    end_s = _span_time(raw, "endTimeUnixNano", "end_time")
    is_error, status_msg = _status(raw.get("status"))

    return _Span(
        trace_id=str(trace_id),
        span_id=str(span_id),
        parent_id=parent_id,
        name=str(raw.get("name", "") or ""),
        kind=kind,
        attrs=attrs,
        start_s=start_s,
        end_s=end_s,
        is_error=is_error,
        status_msg=status_msg,
    )


def _attributes_to_dict(attributes: Any) -> dict[str, Any]:
    """Normalize both OTLP list-form and flat-dict attributes to a flat dict."""
    if isinstance(attributes, dict):
        return dict(attributes)
    result: dict[str, Any] = {}
    if isinstance(attributes, list):
        for item in attributes:
            if isinstance(item, dict) and "key" in item:
                result[item["key"]] = _otlp_value(item.get("value"))
    return result


def _otlp_value(value: Any) -> Any:
    """Unwrap an OTLP AnyValue ``{"stringValue": …}`` into a plain Python value."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "intValue" in value:
        try:
            return int(value["intValue"])
        except (TypeError, ValueError):
            return value["intValue"]
    if "doubleValue" in value:
        return value["doubleValue"]
    if "boolValue" in value:
        return value["boolValue"]
    if "bytesValue" in value:
        return value["bytesValue"]
    if "arrayValue" in value:
        return [_otlp_value(v) for v in value["arrayValue"].get("values", [])]
    if "kvlistValue" in value:
        return {
            kv["key"]: _otlp_value(kv.get("value"))
            for kv in value["kvlistValue"].get("values", [])
            if isinstance(kv, dict) and "key" in kv
        }
    return value


def _span_time(raw: dict[str, Any], nano_key: str, iso_key: str) -> float:
    if nano_key in raw and raw[nano_key] is not None:
        return _num_to_seconds(raw[nano_key])
    if iso_key in raw and raw[iso_key] is not None:
        val = raw[iso_key]
        if isinstance(val, str):
            dt = _parse_iso(val)
            return dt.timestamp() if dt else 0.0
        return _num_to_seconds(val)
    return 0.0


def _num_to_seconds(value: Any) -> float:
    """Interpret an OTLP timestamp: nanoseconds by spec, tolerate ms/seconds."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0.0
    if n <= 0:
        return 0.0
    if n > 1e17:      # nanoseconds (OTLP UnixNano)
        return n / 1e9
    if n > 1e14:      # microseconds
        return n / 1e6
    if n > 1e11:      # milliseconds
        return n / 1e3
    return n          # seconds


def _status(status: Any) -> tuple[bool, str | None]:
    if not isinstance(status, dict):
        return False, None
    code = status.get("code", status.get("status_code"))
    msg = status.get("message") or status.get("description")
    is_error = code in (2, "2", "STATUS_CODE_ERROR", "ERROR")
    return is_error, msg


# ---------------------------------------------------------------------------
# Transcript construction
# ---------------------------------------------------------------------------


def _build_transcript(trace_id: str, spans: list[_Span]) -> Transcript:
    spans.sort(key=lambda s: (s.start_s, s.span_id))
    by_id = {s.span_id: s for s in spans}

    # Root = earliest span with no in-trace parent.
    roots = [s for s in spans if not s.parent_id or s.parent_id not in by_id]
    root = roots[0] if roots else spans[0]

    task_id = root.name or trace_id
    transcript = Transcript(task_id=task_id, trial_id=trace_id)
    transcript.metadata = {"importer": "otlp", "trace_id": trace_id}
    session_id = next((s.attrs.get(_SESSION_ID) for s in spans if s.attrs.get(_SESSION_ID)), None)
    if session_id:
        transcript.metadata["session_id"] = session_id

    root_input = _coerce(root.attrs.get(_INPUT))
    if root_input is not None:
        transcript.input_prompt = _to_text(root_input)
        transcript.input_params = {"root_input": root_input}

    for span in spans:
        _map_span(transcript, span, by_id)

    _finalize(transcript, spans, root)
    return transcript


def _map_span(transcript: Transcript, span: _Span, by_id: dict[str, _Span]) -> None:
    kind = span.kind
    if kind in ("LLM", "EMBEDDING"):
        _add_llm_call(transcript, span, by_id, embedding=(kind == "EMBEDDING"))
    elif kind == "TOOL":
        _add_tool_call(transcript, span, by_id)
    elif kind == "RETRIEVER":
        _add_retriever_call(transcript, span, by_id)
    elif kind == "GUARDRAIL":
        transcript.metadata.setdefault("guardrails", []).append(
            {"name": span.name, "triggered": span.is_error}
        )
        transcript.add_reasoning_step(f"[guardrail] {span.name}")
    elif kind == "RERANKER":
        transcript.metadata.setdefault("rerankers", []).append({"name": span.name})
    elif kind == "EVALUATOR":
        transcript.metadata.setdefault("evaluators", []).append(
            {"name": span.name, "output": _to_text(_coerce(span.attrs.get(_OUTPUT)))}
        )
    # AGENT / CHAIN / PROMPT / UNKNOWN are structural (see _agent_name / root I/O).


def _add_llm_call(
    transcript: Transcript, span: _Span, by_id: dict[str, _Span], *, embedding: bool
) -> None:
    model = span.attrs.get(_EMBED_MODEL if embedding else _LLM_MODEL) or span.attrs.get(_LLM_MODEL)
    prompt_tok = _int(span.attrs.get(_TOK_PROMPT))
    completion_tok = _int(span.attrs.get(_TOK_COMPLETION))
    total_tok = _int(span.attrs.get(_TOK_TOTAL)) or (prompt_tok + completion_tok)

    tokens = None
    if prompt_tok or completion_tok or total_tok:
        token_meta = {}
        cache_read = _int(span.attrs.get(_TOK_CACHE_READ))
        if cache_read:
            token_meta["cache_read"] = cache_read
        tokens = TokenUsage(
            input_tokens=prompt_tok,
            output_tokens=completion_tok,
            total_tokens=total_tok,
            metadata=token_meta,
        )

    cost = _cost(span.attrs, model, prompt_tok, completion_tok)

    meta = _context_meta(span, by_id)
    if model:
        meta["model"] = model
    if span.attrs.get(_LLM_PROVIDER):
        meta["provider"] = span.attrs[_LLM_PROVIDER]
    if span.attrs.get(_LLM_SYSTEM):
        meta["system"] = span.attrs[_LLM_SYSTEM]
    requested = _requested_tool_names(span.attrs)
    if requested:
        meta["requested_tools"] = requested

    transcript.tool_calls.append(
        ToolCall(
            tool_name="llm.embedding" if embedding else "llm.generation",
            input=_as_dict(_coerce(span.attrs.get(_INPUT)) or _input_messages(span.attrs)),
            output=_to_text(_coerce(span.attrs.get(_OUTPUT))) or None,
            status="error" if span.is_error else "ok",
            duration_ms=_duration_ms(span),
            timestamp=span.start_s,
            error={"message": span.status_msg or "error"} if span.is_error else None,
            cost=cost,
            tokens=tokens,
            tool_type="llm",
            metadata=meta,
        )
    )


def _add_tool_call(transcript: Transcript, span: _Span, by_id: dict[str, _Span]) -> None:
    name = span.attrs.get(_TOOL_NAME) or span.name or "unknown_tool"
    tool_input = _coerce(span.attrs.get(_TOOL_PARAMS))
    if tool_input is None:
        tool_input = _coerce(span.attrs.get(_INPUT))
    transcript.tool_calls.append(
        ToolCall(
            tool_name=str(name),
            input=_as_dict(tool_input),
            output=_coerce(span.attrs.get(_OUTPUT)),
            status="error" if span.is_error else "ok",
            duration_ms=_duration_ms(span),
            timestamp=span.start_s,
            error={"message": span.status_msg or "error"} if span.is_error else None,
            tool_type="function",
            metadata=_context_meta(span, by_id),
        )
    )


def _add_retriever_call(transcript: Transcript, span: _Span, by_id: dict[str, _Span]) -> None:
    documents = _retrieval_documents(span.attrs)
    transcript.tool_calls.append(
        ToolCall(
            tool_name=span.name or "retriever",
            input=_as_dict(_coerce(span.attrs.get(_INPUT))),
            output=documents if documents else _coerce(span.attrs.get(_OUTPUT)),
            status="error" if span.is_error else "ok",
            duration_ms=_duration_ms(span),
            timestamp=span.start_s,
            error={"message": span.status_msg or "error"} if span.is_error else None,
            tool_type="search",
            metadata={**_context_meta(span, by_id), "document_count": len(documents)},
        )
    )


def _finalize(transcript: Transcript, spans: list[_Span], root: _Span) -> None:
    starts = [s.start_s for s in spans if s.start_s]
    ends = [s.end_s for s in spans if s.end_s]
    if starts:
        transcript.start_time = datetime.fromtimestamp(min(starts))
    if starts and ends:
        end = max(ends)
        transcript.end_time = datetime.fromtimestamp(end)
        transcript.total_duration_ms = (end - min(starts)) * 1000.0

    # Best-effort outcome: the root chain's output, else the last LLM output.
    final = _to_text(_coerce(root.attrs.get(_OUTPUT)))
    if not final:
        for tc in reversed(transcript.tool_calls):
            if tc.tool_type == "llm" and tc.output:
                final = _to_text(tc.output)
                break
    if final:
        transcript.set_outcome(output_data={"final_output": final})


# ---------------------------------------------------------------------------
# OpenInference flattened-attribute helpers
# ---------------------------------------------------------------------------


def _input_messages(attrs: dict[str, Any]) -> Any:
    """Reconstruct ``llm.input_messages.{i}.message.{role,content}`` into a list."""
    msgs = _indexed_messages(attrs, "llm.input_messages")
    return {"messages": msgs} if msgs else None


def _indexed_messages(attrs: dict[str, Any], prefix: str) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    role_key = f"{prefix}."
    for key, value in attrs.items():
        if not key.startswith(role_key):
            continue
        rest = key[len(role_key):]  # e.g. "0.message.role"
        head, _, tail = rest.partition(".")
        if not head.isdigit():
            continue
        idx = int(head)
        slot = by_index.setdefault(idx, {})
        if tail == "message.role":
            slot["role"] = value
        elif tail == "message.content":
            slot["content"] = value
    return [by_index[i] for i in sorted(by_index)]


def _requested_tool_names(attrs: dict[str, Any]) -> list[str]:
    """Tool-call function names the model requested (from output messages)."""
    names: list[str] = []
    for key, value in attrs.items():
        if key.startswith("llm.output_messages.") and key.endswith(
            "tool_call.function.name"
        ):
            names.append(str(value))
    return names


def _retrieval_documents(attrs: dict[str, Any]) -> list[dict[str, Any]]:
    """Reconstruct ``retrieval.documents.{i}.document.{content,id,score}``."""
    by_index: dict[int, dict[str, Any]] = {}
    prefix = "retrieval.documents."
    for key, value in attrs.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix):]
        head, _, tail = rest.partition(".")
        if not head.isdigit():
            continue
        slot = by_index.setdefault(int(head), {})
        if tail == "document.content":
            slot["content"] = value
        elif tail == "document.id":
            slot["id"] = value
        elif tail == "document.score":
            slot["score"] = value
    return [by_index[i] for i in sorted(by_index)]


def _context_meta(span: _Span, by_id: dict[str, _Span]) -> dict[str, Any]:
    """Walk ancestors to tag the call with the enclosing agent name."""
    meta: dict[str, Any] = {}
    agent = _agent_name(span, by_id)
    if agent:
        meta["agent_name"] = agent
    return meta


def _agent_name(span: _Span, by_id: dict[str, _Span]) -> str | None:
    seen: set[str] = set()
    parent_id = span.parent_id
    while parent_id and parent_id not in seen and parent_id in by_id:
        seen.add(parent_id)
        parent = by_id[parent_id]
        if parent.kind == "AGENT":
            return parent.name or None
        parent_id = parent.parent_id
    return None


# ---------------------------------------------------------------------------
# Small value helpers
# ---------------------------------------------------------------------------


def _cost(
    attrs: dict[str, Any], model: str | None, prompt_tok: int, completion_tok: int
) -> CostInfo | None:
    total = attrs.get(_COST_TOTAL)
    if total is not None:
        meta = {"source": "openinference"}
        if model:
            meta["model"] = model
        return CostInfo(
            total_usd=_float(total),
            input_cost_usd=_opt_float(attrs.get(_COST_PROMPT)),
            output_cost_usd=_opt_float(attrs.get(_COST_COMPLETION)),
            metadata=meta,
        )
    if model:
        return calculate_cost(model, prompt_tok, completion_tok)
    return None


def _duration_ms(span: _Span) -> float:
    if span.start_s and span.end_s and span.end_s >= span.start_s:
        return (span.end_s - span.start_s) * 1000.0
    return 0.0


def _coerce(value: Any) -> Any:
    """Parse a JSON string into a dict/list when it looks like one, else pass through."""
    if isinstance(value, str):
        s = value.strip()
        if s[:1] in ("{", "[") and s[-1:] in ("}", "]"):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return value
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    return {"value": value}


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        # Common OpenAI/Phoenix output shapes.
        for key in ("content", "output", "text", "final_output", "answer"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                return candidate
        try:
            return json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(value)
    if isinstance(value, list):
        parts = [_to_text(v) for v in value]
        return "\n".join(p for p in parts if p)
    return str(value)


def _int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value) if value is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _parse_iso(value: str) -> datetime | None:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None
