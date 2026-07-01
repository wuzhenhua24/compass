"""Tests for the pi (@earendil-works/pi-*) JSONL session -> Compass Transcript importer.

No pi package is required: we hand-build session JSONL that mirrors pi's on-disk
shape (session header line + SessionTreeEntry lines linked by ``parentId``).
"""

from __future__ import annotations

import json

import pytest

from compass.adapters import register_pricing, reset_pricing
from compass.graders import GradeContext, get_grader
from compass.integrations import (
    PiSessionError,
    import_pi_session,
    import_pi_sessions,
    load_pi_session,
)


@pytest.fixture(autouse=True)
def _pricing_defaults():
    """Pin gpt-4o to the built-in $2.50/$10 rate so cost asserts are deterministic."""
    reset_pricing()
    register_pricing("gpt-4o", (2.50, 10.00))
    yield
    reset_pricing()


# ---------------------------------------------------------------------------
# Fixtures: build pi session JSONL by hand
# ---------------------------------------------------------------------------


def _line(obj) -> str:
    return json.dumps(obj)


def _session(*entries: dict, session_id: str = "sess-1", cwd: str = "/repo") -> str:
    header = {
        "type": "session",
        "version": 3,
        "id": session_id,
        "timestamp": "2026-01-01T00:00:00.000Z",
        "cwd": cwd,
    }
    return "\n".join([_line(header), *[_line(e) for e in entries]]) + "\n"


def _msg(entry_id, parent_id, message, ts="2026-01-01T00:00:01.000Z") -> dict:
    return {
        "type": "message", "id": entry_id, "parentId": parent_id,
        "timestamp": ts, "message": message,
    }


def _assistant(content, *, model="gpt-4o", provider="openai", usage=None,
               stop_reason="stop", ts_ms=1735689601000, error=None) -> dict:
    msg = {
        "role": "assistant",
        "content": content,
        "api": "openai-completions",
        "provider": provider,
        "model": model,
        "usage": usage or _usage(),
        "stopReason": stop_reason,
        "timestamp": ts_ms,
    }
    if error is not None:
        msg["errorMessage"] = error
    return msg


def _usage(input_=10, output=20, total=None, cost=None) -> dict:
    return {
        "input": input_,
        "output": output,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": total if total is not None else input_ + output,
        "cost": cost or {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": 0},
    }


def _standard_session() -> str:
    """user -> assistant(thinking + tool call) -> tool result -> assistant(final text)."""
    return _session(
        {"type": "session_info", "id": "e0", "parentId": None,
         "timestamp": "2026-01-01T00:00:00.100Z", "name": "weather-run"},
        _msg("e1", "e0", {"role": "user", "content": "Weather in SF?", "timestamp": 1735689600000}),
        _msg("e2", "e1", _assistant(
            [
                {"type": "thinking", "thinking": "I should call the weather tool."},
                {"type": "toolCall", "id": "call-1", "name": "get_weather",
                 "arguments": {"city": "SF"}},
            ],
            # 1M input, native cost 0 -> falls back to Compass pricing ($2.50).
            usage=_usage(input_=1_000_000, output=0),
            stop_reason="toolUse",
            ts_ms=1735689601000,
        )),
        _msg("e3", "e2", {
            "role": "toolResult", "toolCallId": "call-1", "toolName": "get_weather",
            "content": [{"type": "text", "text": "sunny, 22C"}],
            "isError": False, "timestamp": 1735689603000,
        }),
        _msg("e4", "e3", _assistant(
            [{"type": "text", "text": "It's sunny and 22C in SF."}],
            # native cost recorded by pi -> used verbatim ($0.123).
            usage=_usage(
                input_=10, output=20,
                cost={"input": 0.1, "output": 0.023, "cacheRead": 0,
                      "cacheWrite": 0, "total": 0.123},
            ),
            stop_reason="stop",
            ts_ms=1735689604000,
        )),
    )


# ---------------------------------------------------------------------------
# Core mapping
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_basic_shape(self):
        t = load_pi_session(_standard_session())
        assert t.trial_id == "sess-1"
        assert t.task_id == "weather-run"          # from session_info name
        assert t.input_prompt == "Weather in SF?"
        assert t.metadata["cwd"] == "/repo"
        assert t.metadata["importer"] == "pi"

    def test_tool_call_matched_to_result(self):
        t = load_pi_session(_standard_session())
        weather = next(tc for tc in t.tool_calls if tc.tool_name == "get_weather")
        assert weather.input == {"city": "SF"}
        assert weather.output == "sunny, 22C"
        assert weather.status == "ok"
        assert weather.call_id == "call-1"
        # duration = result_ts (…603) - call_ts (…601) = 2000 ms
        assert weather.duration_ms == pytest.approx(2000.0)

    def test_assistant_turns_become_llm_calls(self):
        t = load_pi_session(_standard_session())
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert len(llm) == 2
        assert all(tc.tool_name == "llm.generation" for tc in llm)
        assert llm[0].tokens.input_tokens == 1_000_000
        assert llm[0].metadata["model"] == "gpt-4o"

    def test_native_cost_used_when_present_else_computed(self):
        t = load_pi_session(_standard_session())
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        # Turn 1: no native cost -> computed from pricing (1M input * $2.50/1M).
        assert llm[0].cost.total_usd == pytest.approx(2.50)
        assert llm[0].cost.metadata.get("source") != "pi"
        # Turn 2: native pi cost used verbatim.
        assert llm[1].cost.total_usd == pytest.approx(0.123)
        assert llm[1].cost.metadata.get("source") == "pi"
        # Aggregate flows through Transcript.sum_cost().
        assert t.sum_cost().total_usd == pytest.approx(2.623)

    def test_thinking_becomes_reasoning_step(self):
        t = load_pi_session(_standard_session())
        assert any("I should call the weather tool" in s for s in t.reasoning_steps)
        assert any(s.startswith("[thinking]") for s in t.reasoning_steps)

    def test_outcome_is_final_assistant_text(self):
        t = load_pi_session(_standard_session())
        assert t.outcome.output_data.get("final_output") == "It's sunny and 22C in SF."

    def test_ordering_llm_before_its_tool_call(self):
        t = load_pi_session(_standard_session())
        names = [tc.tool_name for tc in t.tool_calls]
        # first turn's llm call precedes the get_weather it requested
        assert names.index("llm.generation") < names.index("get_weather")


class TestBranching:
    def test_active_branch_only_drops_abandoned_fork(self):
        # e1 forks into e2 (stock, abandoned) and e3 (weather, active leaf).
        session = _session(
            _msg("e1", None, {"role": "user", "content": "hi", "timestamp": 1735689600000}),
            _msg("e2", "e1", _assistant(
                [{"type": "toolCall", "id": "c-stock", "name": "get_stock", "arguments": {}}],
                stop_reason="toolUse")),
            _msg("e3", "e1", _assistant(
                [{"type": "toolCall", "id": "c-weather", "name": "get_weather", "arguments": {}}],
                stop_reason="toolUse")),
            {"type": "leaf", "id": "L1", "parentId": "e3",
             "timestamp": "2026-01-01T00:00:05.000Z", "targetId": "e3"},
        )
        active = load_pi_session(session)  # active_branch_only=True by default
        names = {tc.tool_name for tc in active.tool_calls}
        assert "get_weather" in names
        assert "get_stock" not in names

        full = load_pi_session(session, active_branch_only=False)
        names_full = {tc.tool_name for tc in full.tool_calls}
        assert {"get_weather", "get_stock"} <= names_full


class TestRobustness:
    def test_error_turn_marks_llm_call(self):
        session = _session(
            _msg("e1", None, {"role": "user", "content": "x", "timestamp": 1735689600000}),
            _msg("e2", "e1", _assistant(
                [{"type": "text", "text": ""}], stop_reason="error", error="rate limited")),
        )
        t = load_pi_session(session)
        llm = next(tc for tc in t.tool_calls if tc.tool_type == "llm")
        assert llm.status == "error"
        assert llm.error["message"] == "rate limited"

    def test_orphan_tool_result_is_kept(self):
        session = _session(
            _msg("e1", None, {"role": "user", "content": "x", "timestamp": 1735689600000}),
            _msg("e2", "e1", {
                "role": "toolResult", "toolCallId": "no-such-call", "toolName": "ghost",
                "content": [{"type": "text", "text": "boom"}], "isError": True,
                "timestamp": 1735689602000,
            }),
        )
        t = load_pi_session(session)
        ghost = next(tc for tc in t.tool_calls if tc.tool_name == "ghost")
        assert ghost.status == "error"
        assert ghost.output == "boom"

    def test_malformed_entry_line_is_skipped(self):
        good = _standard_session().rstrip("\n")
        session = good + "\n" + "{ this is not json }" + "\n"
        t = load_pi_session(session)  # should not raise
        assert any(tc.tool_name == "get_weather" for tc in t.tool_calls)

    def test_non_session_header_raises(self):
        with pytest.raises(PiSessionError):
            load_pi_session('{"type":"not-a-session"}\n')

    def test_empty_file_raises(self):
        with pytest.raises(PiSessionError):
            load_pi_session("   \n  \n")


class TestFileIO:
    def test_import_from_file(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(_standard_session(), encoding="utf-8")
        t = import_pi_session(p)
        assert t.trial_id == "sess-1"

    def test_import_directory(self, tmp_path):
        (tmp_path / "a.jsonl").write_text(_standard_session(), encoding="utf-8")
        (tmp_path / "b.jsonl").write_text(
            _session(_msg("e1", None, {"role": "user", "content": "hi", "timestamp": 1}),
                     session_id="sess-2"),
            encoding="utf-8",
        )
        (tmp_path / "junk.jsonl").write_text("not a session\n", encoding="utf-8")
        transcripts = import_pi_sessions(tmp_path)
        # junk.jsonl is skipped, the two valid sessions import.
        ids = {t.trial_id for t in transcripts}
        assert ids == {"sess-1", "sess-2"}


class TestGradingIntegration:
    async def test_tool_usage_grader(self):
        t = load_pi_session(_standard_session())
        grader = get_grader("tool_usage")({"required_tools": ["get_weather"]})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.passed is True

    async def test_cost_budget_grader(self):
        t = load_pi_session(_standard_session())
        grader = get_grader("cost_budget")({"max_cost_usd": 5.0, "max_tokens": 2_000_000})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.details["total_cost_usd"] == pytest.approx(2.623)
        assert result.passed is True
