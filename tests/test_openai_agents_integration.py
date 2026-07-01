"""Tests for the OpenAI Agents SDK -> Compass Transcript bridge.

The SDK is not a test dependency, so these tests feed the processor fake span /
trace objects that mimic the SDK's attribute shape (duck-typed).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from compass.graders import GradeContext, get_grader
from compass.integrations import (
    CompassTraceProcessor,
    install_openai_agents_processor,
)

# ---------------------------------------------------------------------------
# Fake SDK objects (mirror agents.tracing span/trace attribute surface)
# ---------------------------------------------------------------------------


def _span(span_type, *, span_id, parent_id=None, trace_id="tr1",
          started="2026-01-01T00:00:00", ended="2026-01-01T00:00:01",
          error=None, **data_fields):
    data = SimpleNamespace(type=span_type, **data_fields)
    return SimpleNamespace(
        trace_id=trace_id, span_id=span_id, parent_id=parent_id,
        started_at=started, ended_at=ended, error=error, span_data=data,
    )


def _trace(trace_id="tr1", name="weather-workflow", metadata=None):
    return SimpleNamespace(trace_id=trace_id, name=name, metadata=metadata or {})


def _run_agent_trace(proc: CompassTraceProcessor) -> None:
    """Drive one full trace: agent -> turn -> (generation, function) -> handoff."""
    trace = _trace()
    proc.on_trace_start(trace)

    agent = _span("agent", span_id="agent1", name="WeatherAgent")
    turn = _span("turn", span_id="turn1", parent_id="agent1", turn=2,
                 agent_name="WeatherAgent")
    proc.on_span_start(agent)
    proc.on_span_start(turn)

    gen = _span("generation", span_id="gen1", parent_id="turn1",
                model="gpt-4o", input=[{"role": "user", "content": "weather?"}],
                output=[{"role": "assistant", "content": "calling tool"}],
                usage={"input_tokens": 1_000_000, "output_tokens": 0})
    proc.on_span_start(gen)
    proc.on_span_end(gen)

    fn = _span("function", span_id="fn1", parent_id="turn1",
               name="get_weather", input='{"city": "SF"}', output="sunny",
               mcp_data=None)
    proc.on_span_start(fn)
    proc.on_span_end(fn)

    handoff = _span("handoff", span_id="ho1", parent_id="agent1",
                    from_agent="WeatherAgent", to_agent="SummaryAgent")
    proc.on_span_start(handoff)
    proc.on_span_end(handoff)

    proc.on_span_end(turn)
    proc.on_span_end(agent)
    proc.on_trace_end(trace)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSpanMapping:
    def test_produces_one_transcript(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        assert len(proc.transcripts) == 1
        assert proc.latest is proc.transcripts[0]
        assert proc.latest.trial_id == "tr1"
        assert proc.latest.task_id == "weather-workflow"

    def test_function_span_becomes_tool_call(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        calls = {tc.tool_name: tc for tc in proc.latest.tool_calls}
        assert "get_weather" in calls
        fn = calls["get_weather"]
        assert fn.tool_type == "function"
        assert fn.input == {"city": "SF"}   # JSON string parsed to dict
        assert fn.output == "sunny"
        assert fn.status == "ok"
        assert fn.duration_ms == pytest.approx(1000.0)

    def test_generation_span_becomes_llm_call_with_tokens_and_cost(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        gen = next(tc for tc in proc.latest.tool_calls if tc.tool_type == "llm")
        assert gen.tool_name == "llm.generation"
        assert gen.tokens is not None
        assert gen.tokens.input_tokens == 1_000_000
        # Cost is computed from Compass's configurable pricing table (gpt-4o=$2.50/1M in).
        assert gen.cost is not None
        assert gen.cost.total_usd == pytest.approx(2.50)
        assert gen.metadata.get("model") == "gpt-4o"

    def test_parent_walk_tags_turn_and_agent(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        for tc in proc.latest.tool_calls:
            assert tc.metadata.get("turn_index") == 2
            assert tc.metadata.get("agent_name") == "WeatherAgent"

    def test_handoff_recorded_in_metadata_not_tool_calls(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        t = proc.latest
        assert {"from": "WeatherAgent", "to": "SummaryAgent"} in t.metadata["handoffs"]
        assert any("Handoff" in step for step in t.reasoning_steps)
        # Handoff must NOT pollute the tool-call list.
        assert all(tc.tool_name != "handoff" for tc in t.tool_calls)

    def test_error_span_maps_to_error_status(self):
        proc = CompassTraceProcessor()
        proc.on_trace_start(_trace())
        fn = _span("function", span_id="fn1", name="bad_tool", input="{}",
                   output=None, error={"message": "boom", "data": None})
        proc.on_span_start(fn)
        proc.on_span_end(fn)
        proc.on_trace_end(_trace())
        tc = proc.latest.tool_calls[0]
        assert tc.status == "error"
        assert tc.error == {"message": "boom", "data": None}

    def test_finalize_sets_outcome_from_last_llm_output(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        assert proc.latest.outcome.output_data.get("final_output") is not None


class TestGradingIntegration:
    """The reconstructed Transcript must flow into Compass transcript graders."""

    async def test_tool_usage_grader_on_reconstructed_transcript(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        grader = get_grader("tool_usage")({"required_tools": ["get_weather"]})
        result = await grader.grade(
            GradeContext(transcript=proc.latest, outcome=proc.latest.outcome)
        )
        assert result.passed is True

    async def test_cost_budget_grader_sees_llm_cost(self):
        proc = CompassTraceProcessor()
        _run_agent_trace(proc)
        grader = get_grader("cost_budget")({"max_cost_usd": 5.0, "max_tokens": 2_000_000})
        result = await grader.grade(
            GradeContext(transcript=proc.latest, outcome=proc.latest.outcome)
        )
        assert result.details["total_cost_usd"] == pytest.approx(2.50)
        assert result.passed is True


class TestInstall:
    def test_install_requires_agents_package(self):
        try:
            import agents  # noqa: F401
            pytest.skip("openai-agents is installed; ImportError path not testable")
        except ImportError:
            pass
        with pytest.raises(ImportError):
            install_openai_agents_processor()
