"""Tests for the OTLP / OpenInference trace importer.

No OpenTelemetry package is required: we hand-build trace payloads in both the
OTLP protobuf-JSON shape (list-form attributes, ``resourceSpans`` envelope) and
the flat-dict shape (OTel SDK ``ReadableSpan.to_json()`` / Phoenix export).
"""

from __future__ import annotations

import json

import pytest

from compass.adapters import register_pricing, reset_pricing
from compass.graders import GradeContext, get_grader
from compass.integrations import (
    OTLPImportError,
    import_otlp_file,
    load_otlp,
    otlp_to_transcripts,
)

_NS = 1_700_000_000_000_000_000  # base start (nanoseconds ~ 2023)


@pytest.fixture(autouse=True)
def _pricing_defaults():
    reset_pricing()
    register_pricing("gpt-4o", (2.50, 10.00))
    yield
    reset_pricing()


# ---------------------------------------------------------------------------
# Builders: OTLP list-form attributes + resourceSpans envelope
# ---------------------------------------------------------------------------


def _attr(key, value):
    if isinstance(value, bool):
        v = {"boolValue": value}
    elif isinstance(value, int):
        v = {"intValue": str(value)}          # OTLP encodes ints as strings
    elif isinstance(value, float):
        v = {"doubleValue": value}
    else:
        v = {"stringValue": str(value)}
    return {"key": key, "value": v}


def _span(span_id, parent, kind, attrs, *, trace="tr-1", name="",
          start=0, dur=1, error=False):
    a = dict(attrs)
    a["openinference.span.kind"] = kind
    span = {
        "traceId": trace,
        "spanId": span_id,
        "name": name or kind.lower(),
        "startTimeUnixNano": str(_NS + start * 1_000_000_000),
        "endTimeUnixNano": str(_NS + (start + dur) * 1_000_000_000),
        "attributes": [_attr(k, v) for k, v in a.items()],
        "status": {"code": 2 if error else 1},
    }
    if parent:
        span["parentSpanId"] = parent
    return span


def _envelope(*spans):
    return {"resourceSpans": [{"scopeSpans": [{"spans": list(spans)}]}]}


def _weather_trace() -> dict:
    """CHAIN -> AGENT -> (LLM, TOOL, LLM, RETRIEVER) + GUARDRAIL."""
    return _envelope(
        _span("c1", None, "CHAIN", {
            "input.value": "What's the weather in SF?",
            "output.value": "It's sunny, 22C.",
            "session.id": "sess-xyz",
        }, name="agent_run", start=0, dur=5),
        _span("a1", "c1", "AGENT", {}, name="WeatherAgent", start=0, dur=5),
        _span("l1", "a1", "LLM", {
            "llm.model_name": "gpt-4o",
            "llm.provider": "openai",
            "llm.token_count.prompt": 1_000_000,   # -> $2.50 via pricing fallback
            "llm.token_count.completion": 0,
            "output.value": "calling tool",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.tool_calls.0.tool_call.function.name": "get_weather",
        }, name="chat", start=0, dur=1),
        _span("t1", "a1", "TOOL", {
            "tool.name": "get_weather",
            "tool.parameters": '{"city": "SF"}',
            "output.value": "sunny, 22C",
        }, name="get_weather", start=1, dur=1),
        _span("l2", "a1", "LLM", {
            "llm.model_name": "gpt-4o",
            "llm.token_count.prompt": 10,
            "llm.token_count.completion": 20,
            "llm.cost.total": 0.123,               # native cost -> used verbatim
            "llm.cost.prompt": 0.1,
            "llm.cost.completion": 0.023,
            "output.value": "It's sunny, 22C.",
        }, name="chat", start=2, dur=1),
        _span("r1", "a1", "RETRIEVER", {
            "input.value": "weather docs",
            "retrieval.documents.0.document.content": "SF weather is mild.",
            "retrieval.documents.0.document.id": "d1",
            "retrieval.documents.0.document.score": 0.9,
        }, name="retrieve", start=1, dur=1),
        _span("g1", "c1", "GUARDRAIL", {}, name="safety", start=0, dur=1),
    )


# ---------------------------------------------------------------------------
# Core mapping (OTLP list-form)
# ---------------------------------------------------------------------------


class TestOtlpMapping:
    def test_one_transcript_per_trace(self):
        ts = otlp_to_transcripts(_weather_trace())
        assert len(ts) == 1
        t = ts[0]
        assert t.trial_id == "tr-1"
        assert t.task_id == "agent_run"          # root CHAIN name
        assert t.metadata["session_id"] == "sess-xyz"

    def test_root_input_and_outcome(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        assert t.input_prompt == "What's the weather in SF?"
        assert t.outcome.output_data.get("final_output") == "It's sunny, 22C."

    def test_tool_span_becomes_function_call(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        tool = next(tc for tc in t.tool_calls if tc.tool_name == "get_weather")
        assert tool.tool_type == "function"
        assert tool.input == {"city": "SF"}      # tool.parameters JSON parsed
        assert tool.output == "sunny, 22C"

    def test_llm_spans_tokens_and_cost(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert len(llm) == 2
        assert llm[0].tokens.input_tokens == 1_000_000
        # l1: no native cost -> computed from pricing ($2.50)
        assert llm[0].cost.total_usd == pytest.approx(2.50)
        assert llm[0].cost.metadata.get("source") != "openinference"
        # l2: native OpenInference cost -> used verbatim
        assert llm[1].cost.total_usd == pytest.approx(0.123)
        assert llm[1].cost.metadata.get("source") == "openinference"
        assert t.sum_cost().total_usd == pytest.approx(2.623)

    def test_requested_tool_names_captured(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        l1 = next(tc for tc in t.tool_calls if tc.output == "calling tool")
        assert l1.metadata["requested_tools"] == ["get_weather"]

    def test_retriever_span(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        retr = next(tc for tc in t.tool_calls if tc.tool_type == "search")
        assert retr.metadata["document_count"] == 1
        assert retr.output[0]["content"] == "SF weather is mild."
        assert retr.output[0]["score"] == pytest.approx(0.9)

    def test_agent_name_tagging_via_parent_walk(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        # LLM/TOOL/RETRIEVER are children of the AGENT span "WeatherAgent"
        tagged = [tc for tc in t.tool_calls if tc.metadata.get("agent_name")]
        assert tagged and all(tc.metadata["agent_name"] == "WeatherAgent" for tc in tagged)

    def test_guardrail_recorded_in_metadata_not_tool_calls(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        assert any(g["name"] == "safety" for g in t.metadata["guardrails"])
        assert all(tc.tool_name != "safety" for tc in t.tool_calls)

    def test_duration_from_span_times(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        tool = next(tc for tc in t.tool_calls if tc.tool_name == "get_weather")
        assert tool.duration_ms == pytest.approx(1000.0)  # 1s span


# ---------------------------------------------------------------------------
# Flat-dict shape (OTel SDK to_json / Phoenix)
# ---------------------------------------------------------------------------


class TestFlatDictShape:
    def _flat_trace(self):
        return [
            {
                "name": "root",
                "context": {"trace_id": "tr-2", "span_id": "s0"},
                "parent_id": None,
                "start_time": "2026-01-01T00:00:00.000000Z",
                "end_time": "2026-01-01T00:00:02.000000Z",
                "status": {"status_code": "OK"},
                "attributes": {
                    "openinference.span.kind": "CHAIN",
                    "input.value": "hello",
                    "output.value": "world",
                },
            },
            {
                "name": "bad_tool",
                "context": {"trace_id": "tr-2", "span_id": "s1"},
                "parent_id": "s0",
                "start_time": "2026-01-01T00:00:00.500000Z",
                "end_time": "2026-01-01T00:00:01.000000Z",
                "status": {"status_code": "ERROR", "description": "boom"},
                "attributes": {
                    "openinference.span.kind": "TOOL",
                    "tool.name": "search",
                    "output.value": "failed",
                },
            },
        ]

    def test_flat_dict_attributes_and_iso_times(self):
        t = otlp_to_transcripts(self._flat_trace())[0]
        assert t.trial_id == "tr-2"
        assert t.input_prompt == "hello"
        assert t.outcome.output_data["final_output"] == "world"

    def test_error_status_maps(self):
        t = otlp_to_transcripts(self._flat_trace())[0]
        tool = next(tc for tc in t.tool_calls if tc.tool_name == "search")
        assert tool.status == "error"
        assert tool.error["message"] == "boom"


# ---------------------------------------------------------------------------
# Multi-trace / JSONL / IO
# ---------------------------------------------------------------------------


class TestMultiTraceAndIO:
    def test_multiple_traces_yield_sorted_transcripts(self):
        early = _span("x", None, "CHAIN", {"input.value": "a"}, trace="early", start=0, dur=1)
        late = _span("y", None, "CHAIN", {"input.value": "b"}, trace="late", start=100, dur=1)
        ts = otlp_to_transcripts(_envelope(late, early))  # supplied out of order
        assert [t.trial_id for t in ts] == ["early", "late"]

    def test_jsonl_payload(self):
        # One span per line (flat-dict) + a header line.
        lines = [
            json.dumps({"resourceSpans": []}),  # empty envelope, ignored
            json.dumps(_span("s1", None, "TOOL", {"tool.name": "t", "output.value": "ok"},
                             trace="trj")),
        ]
        ts = load_otlp("\n".join(lines))
        assert len(ts) == 1
        assert ts[0].tool_calls[0].tool_name == "t"

    def test_import_from_file(self, tmp_path):
        p = tmp_path / "trace.json"
        p.write_text(json.dumps(_weather_trace()), encoding="utf-8")
        ts = import_otlp_file(p)
        assert ts[0].trial_id == "tr-1"


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    def test_empty_payload_raises(self):
        with pytest.raises(OTLPImportError):
            load_otlp("   ")

    def test_garbage_raises(self):
        with pytest.raises(OTLPImportError):
            load_otlp("not json at all\n%%%")

    def test_spans_without_ids_are_skipped(self):
        good = _span("s1", None, "TOOL", {"tool.name": "ok"}, trace="trg")
        bad = {"attributes": [], "name": "no-ids"}  # missing trace/span id
        ts = otlp_to_transcripts(_envelope(good, bad))
        assert len(ts) == 1
        assert len(ts[0].tool_calls) == 1

    def test_no_spans_returns_empty_list(self):
        assert otlp_to_transcripts({"resourceSpans": []}) == []


# ---------------------------------------------------------------------------
# Grading integration
# ---------------------------------------------------------------------------


class TestGradingIntegration:
    async def test_tool_usage_grader(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        grader = get_grader("tool_usage")({"required_tools": ["get_weather"]})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.passed is True

    async def test_cost_budget_grader(self):
        t = otlp_to_transcripts(_weather_trace())[0]
        grader = get_grader("cost_budget")({"max_cost_usd": 5.0, "max_tokens": 2_000_000})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.details["total_cost_usd"] == pytest.approx(2.623)
        assert result.passed is True
