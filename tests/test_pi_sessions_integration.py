"""Tests for the pi (@earendil-works/pi-*) trace -> Compass Transcript importer.

No pi package is required: we hand-build both of pi's shapes — the on-disk
session JSONL (session header line + SessionTreeEntry lines linked by
``parentId``) and the ``--mode json`` event stream — from the real formats.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from compass.adapters import register_pricing, reset_pricing
from compass.graders import GradeContext, get_grader
from compass.integrations import (
    PiSessionError,
    PiStreamReconstructor,
    import_pi_session,
    import_pi_sessions,
    import_pi_stream_json,
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

    def test_turn_index_is_first_class_field(self):
        t = load_pi_session(_standard_session())
        # Turn 1: llm.generation + get_weather; turn 2: final llm.generation.
        weather = next(tc for tc in t.tool_calls if tc.tool_name == "get_weather")
        assert weather.turn_index == 1
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert llm[0].turn_index == 1
        assert llm[1].turn_index == 2
        # Promoted to a field, not buried in metadata.
        assert "turn_index" not in weather.metadata


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


class TestTiming:
    def test_duration_is_not_offset_by_the_host_timezone(self):
        """Header start is ISO+Z, message end is epoch-ms: mixing the two frames
        put the host's UTC offset into every duration (8h of "latency" in
        Shanghai, 0 in London)."""
        start_ms = 1735689600000
        session = _session(
            _msg("e1", None, {"role": "user", "content": "hi", "timestamp": start_ms}),
            _msg("e2", "e1", _assistant(
                [{"type": "text", "text": "done"}], ts_ms=start_ms + 5000
            )),
        )
        # Rewrite the header timestamp to the same instant as the first message.
        header, rest = session.split("\n", 1)
        head = json.loads(header)
        head["timestamp"] = (
            datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        t = load_pi_session(json.dumps(head) + "\n" + rest)

        assert t.total_duration_ms == pytest.approx(5000.0)


# ---------------------------------------------------------------------------
# State delta
# ---------------------------------------------------------------------------


def _edit_session(tool: str, args: dict, *, is_error: bool = False, cwd="/repo") -> str:
    return _session(
        _msg("e1", None, {"role": "user", "content": "fix it", "timestamp": 1735689600000}),
        _msg("e2", "e1", _assistant(
            [{"type": "toolCall", "id": "c1", "name": tool, "arguments": args}],
            stop_reason="toolUse",
        )),
        _msg("e3", "e2", {
            "role": "toolResult", "toolCallId": "c1", "toolName": tool,
            "content": [{"type": "text", "text": "Successfully wrote 5 bytes to x"}],
            "isError": is_error, "timestamp": 1735689603000,
        }),
        cwd=cwd,
    )


class TestStateDelta:
    def test_write_and_edit_record_the_file_they_changed(self):
        for tool in ("write", "edit"):
            t = load_pi_session(_edit_session(tool, {"path": "src/app.py"}))
            call = next(tc for tc in t.tool_calls if tc.tool_name == tool)
            assert [(c.kind, c.op, c.target) for c in call.state_delta] == [
                ("file", "update", "src/app.py")
            ]
            assert call.state_delta[0].metadata["absolute_path"] == "/repo/src/app.py"

    def test_one_file_gets_one_spelling(self):
        """Models type both `pricing.py` and `./pricing.py` for the same file —
        both spellings showed up in one two-model run. Left alone, a
        `target: "pricing.py"` rule matches one and silently misses the other,
        which is the worst possible failure for an integrity gate."""
        plain = load_pi_session(_edit_session("edit", {"path": "pricing.py"}))
        dotted = load_pi_session(_edit_session("edit", {"path": "./pricing.py"}))

        def target(t):
            return next(c for c in t.tool_calls if c.state_delta).state_delta[0].target

        assert target(plain) == target(dotted) == "pricing.py"

    def test_a_path_outside_the_cwd_keeps_its_absolute_form(self):
        t = load_pi_session(_edit_session("write", {"path": "/etc/hosts"}))
        call = next(tc for tc in t.tool_calls if tc.tool_name == "write")
        assert call.state_delta[0].target == "/etc/hosts"

    def test_an_absolute_path_is_relativized_to_the_session_cwd(self):
        """A pi trial runs in a throwaway worktree, so an absolute target is a
        different random path every run and no user glob could match it."""
        t = load_pi_session(_edit_session("edit", {"path": "/repo/tests/test_app.py"}))
        call = next(tc for tc in t.tool_calls if tc.tool_name == "edit")
        assert call.state_delta[0].target == "tests/test_app.py"
        assert call.state_delta[0].metadata["absolute_path"] == "/repo/tests/test_app.py"

    def test_a_failed_edit_changed_nothing(self):
        t = load_pi_session(_edit_session("edit", {"path": "src/app.py"}, is_error=True))
        call = next(tc for tc in t.tool_calls if tc.tool_name == "edit")
        assert call.state_delta == []

    def test_shell_mediated_changes_are_not_captured(self):
        """Parsing shell is not something this layer should pretend to do — a
        wrong delta is worse than a missing one."""
        t = load_pi_session(_edit_session("bash", {"command": "rm -rf src/"}))
        call = next(tc for tc in t.tool_calls if tc.tool_name == "bash")
        assert call.state_delta == []

    async def test_state_delta_grader_catches_a_rewritten_test(self):
        t = load_pi_session(_edit_session("edit", {"path": "tests/test_app.py"}))
        grader = get_grader("state_delta")({"forbid": [{"kind": "file", "target": "tests/*"}]})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.passed is False
        assert result.details["violations"][0]["call_id"] == "c1"


# ---------------------------------------------------------------------------
# Streaming (pi --mode json)
# ---------------------------------------------------------------------------


def _stream_events() -> list[dict]:
    """A two-turn run in pi's live wire shape: one `write`, then a final answer.

    Each message is bracketed by start/update/end frames, exactly as pi emits
    them — the point of the test is that only the end frame gets mapped.
    """
    user = {"role": "user", "content": [{"type": "text", "text": "Add a limiter"}],
            "timestamp": 1735689600000}
    call = {"type": "toolCall", "id": "c1", "name": "write",
            "arguments": {"path": "app.py", "content": "x"}}
    first = _assistant(
        [{"type": "thinking", "thinking": "Write the limiter."}, call],
        usage=_usage(input_=1000, output=50, cost={"total": 0.004}),
        stop_reason="toolUse", ts_ms=1735689601000,
    )
    result = {"role": "toolResult", "toolCallId": "c1", "toolName": "write",
              "content": [{"type": "text", "text": "Successfully wrote 1 bytes to app.py"}],
              "isError": False, "timestamp": 1735689603000}
    final = _assistant(
        [{"type": "text", "text": "Added the limiter."}],
        usage=_usage(input_=1100, output=20, cost={"total": 0.002}),
        stop_reason="stop", ts_ms=1735689604000,
    )
    return [
        # Same instant as the first message (epoch 1735689600000), in the other
        # of pi's two time formats — the header is ISO, messages are epoch-ms.
        {"type": "session", "version": 3, "id": "stream-1",
         "timestamp": "2025-01-01T00:00:00.000Z", "cwd": "/repo"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_start", "message": user},
        {"type": "message_end", "message": user},
        {"type": "message_start", "message": {**first, "content": [], "usage": _usage(0, 0),
                                              "stopReason": "pending"}},
        {"type": "message_update",
         "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "W"}},
        {"type": "message_end", "message": first},
        {"type": "tool_execution_start", "toolCallId": "c1", "toolName": "write"},
        {"type": "tool_execution_end", "toolCallId": "c1", "toolName": "write",
         "isError": False},
        {"type": "message_start", "message": result},
        {"type": "message_end", "message": result},
        {"type": "turn_end", "message": first, "toolResults": [result]},
        {"type": "turn_start"},
        {"type": "message_end", "message": final},
        {"type": "turn_end", "message": final, "toolResults": []},
        {"type": "agent_end", "messages": [user, first, result, final], "willRetry": False},
        {"type": "agent_settled"},
    ]


def _stream_text(events=None) -> str:
    return "\n".join(json.dumps(e) for e in (events or _stream_events())) + "\n"


class TestStreamReconstruction:
    def _feed(self, events=None):
        r = PiStreamReconstructor()
        for line in _stream_text(events).splitlines():
            r.feed_line(line)
        return r.finish()

    def test_the_bracketing_frames_do_not_double_count_a_turn(self):
        """A message arrives as start + N deltas + end; only the end frame is
        complete (and carries the turn's usage)."""
        t = self._feed()

        assert [tc.tool_name for tc in t.tool_calls] == [
            "llm.generation", "write", "llm.generation"
        ]
        assert t.sum_cost().total_usd == pytest.approx(0.006)
        assert t.sum_tokens().total_tokens == 2170
        assert [tc.turn_index for tc in t.tool_calls] == [1, 1, 2]

    def test_the_run_is_gradeable(self):
        t = self._feed()

        assert t.input_prompt == "Add a limiter"
        assert t.outcome.output_data["final_output"] == "Added the limiter."
        assert t.environment.model_version == "gpt-4o"
        assert any("Write the limiter" in s for s in t.reasoning_steps)
        assert t.total_duration_ms == pytest.approx(4000.0)

        write = next(tc for tc in t.tool_calls if tc.tool_name == "write")
        assert write.call_id == "c1"
        assert write.duration_ms == pytest.approx(2000.0)
        assert [(c.kind, c.target) for c in write.state_delta] == [("file", "app.py")]

    def test_finish_is_valid_mid_stream(self):
        """A run killed by a timeout still yields the steps it completed."""
        events = _stream_events()
        r = PiStreamReconstructor(task_id="add-limiter")
        for event in events[:8]:  # cut the stream after the first message_end
            r.feed(event)
        t = r.finish()

        assert t.task_id == "add-limiter"          # the caller's id wins
        assert t.trial_id == "stream-1"            # ... over the session's
        assert [tc.tool_name for tc in t.tool_calls] == ["llm.generation", "write"]
        assert next(tc for tc in t.tool_calls if tc.tool_name == "write").output is None

    def test_an_unmodelled_event_is_counted_not_dropped(self):
        events = _stream_events()
        events.insert(-1, {"type": "rate_limit_backoff", "seconds": 30})
        t = self._feed(events)

        assert t.metadata["unhandled_events"] == {"rate_limit_backoff": 1}

    def test_a_non_json_line_is_reported_not_raised(self):
        r = PiStreamReconstructor()
        assert r.feed_line("Loading extensions…\n") is False
        assert r.feed_line("") is False
        assert r.feed_line(json.dumps(_stream_events()[0])) is True

    def test_a_mid_stream_model_switch_lands_in_metadata(self):
        events = _stream_events()
        events.insert(2, {"type": "model_change", "provider": "google",
                          "modelId": "gemini-3.6-flash"})
        t = self._feed(events)

        assert t.metadata["model_changes"] == [
            {"provider": "google", "modelId": "gemini-3.6-flash"}
        ]


class TestStreamFileIO:
    def test_a_saved_stream_imports_without_being_told_the_format(self, tmp_path):
        """Both shapes open with the same session header, so `compass import`
        cannot tell them apart from the first line alone."""
        p = tmp_path / "run.stream.jsonl"
        p.write_text(_stream_text(), encoding="utf-8")

        t = import_pi_session(p)
        assert [tc.tool_name for tc in t.tool_calls] == [
            "llm.generation", "write", "llm.generation"
        ]

    def test_import_pi_stream_json_names_the_task(self, tmp_path):
        p = tmp_path / "run.stream.jsonl"
        p.write_text(_stream_text(), encoding="utf-8")

        t = import_pi_stream_json(p, task_id="add_rate_limit")
        assert t.task_id == "add_rate_limit"
        assert t.metadata["session_id"] == "stream-1"

    def test_a_session_file_is_still_read_as_a_session(self, tmp_path):
        p = tmp_path / "s.jsonl"
        p.write_text(_standard_session(), encoding="utf-8")

        t = import_pi_session(p)
        assert t.task_id == "weather-run"          # session_info entry, not a stream
        assert any(tc.tool_name == "get_weather" for tc in t.tool_calls)


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
