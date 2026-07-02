"""Tests for the claude-agent-sdk Message stream -> Compass Transcript reconstructor.

The SDK is not a test dependency: we feed the reconstructor duck-typed fakes
(``SimpleNamespace``) that mirror the SDK message/block attribute shape. Claude
Code returns tool *results* as a UserMessage carrying a ToolResultBlock, which
these fakes reproduce.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from compass.graders import GradeContext, get_grader
from compass.integrations import (
    import_claude_stream_json,
    reconstruct_transcript,
    reconstruct_transcript_from_stream,
)
from compass.integrations.claude_agent import reconstruct_transcript_from_wire

# ---------------------------------------------------------------------------
# Fake SDK messages / blocks (attribute-compatible with claude_agent_sdk)
# ---------------------------------------------------------------------------


def _text(t):
    return SimpleNamespace(text=t)


def _thinking(t):
    return SimpleNamespace(thinking=t, signature="sig")


def _tool_use(id, name, inp):
    return SimpleNamespace(id=id, name=name, input=inp)


def _tool_result(tool_use_id, content, is_error=False):
    return SimpleNamespace(tool_use_id=tool_use_id, content=content, is_error=is_error)


def _server_tool_result(tool_use_id, content):
    return SimpleNamespace(tool_use_id=tool_use_id, content=content)


def _assistant(content, *, model="claude-opus-4-8", usage=None, stop_reason="end_turn",
               session_id="sess-1", message_id="m1", parent_tool_use_id=None, error=None):
    return SimpleNamespace(
        content=content, model=model,
        usage=usage if usage is not None else {"input_tokens": 10, "output_tokens": 5},
        stop_reason=stop_reason, session_id=session_id, message_id=message_id,
        parent_tool_use_id=parent_tool_use_id, error=error,
    )


def _user(content, *, parent_tool_use_id=None):
    return SimpleNamespace(content=content, parent_tool_use_id=parent_tool_use_id,
                           tool_use_result=None)


def _result(*, result="done", total_cost_usd=0.0123, duration_ms=4200, num_turns=2,
            session_id="sess-1"):
    return SimpleNamespace(
        subtype="success", duration_ms=duration_ms, duration_api_ms=4000,
        is_error=False, num_turns=num_turns, session_id=session_id,
        total_cost_usd=total_cost_usd, usage={"input_tokens": 1010, "output_tokens": 25},
        result=result, model_usage={"claude-opus-4-8": {"input_tokens": 1010}},
        permission_denials=[], stop_reason="end_turn",
    )


def _basic_run():
    """prompt -> assistant(think + Bash) -> tool result -> assistant(text) -> result."""
    return [
        _user("What's 2+2? Use bash."),
        _assistant(
            [_thinking("I'll run echo."),
             _tool_use("tu1", "Bash", {"command": "echo 4"})],
            usage={"input_tokens": 1000, "output_tokens": 20},
            stop_reason="tool_use", message_id="a1",
        ),
        _user([_tool_result("tu1", "4", is_error=False)]),
        _assistant([_text("The answer is 4.")], usage={"input_tokens": 10, "output_tokens": 5},
                   message_id="a2"),
        _result(result="The answer is 4.", total_cost_usd=0.0123),
    ]


# ---------------------------------------------------------------------------
# Core reconstruction
# ---------------------------------------------------------------------------


class TestReconstruction:
    def test_basic_shape(self):
        t = reconstruct_transcript(_basic_run())
        assert t.trial_id == "sess-1"
        assert t.task_id == "sess-1"       # defaults to session id
        assert t.input_prompt == "What's 2+2? Use bash."
        assert t.metadata["importer"] == "claude_agent"

    def test_tool_use_matched_to_user_message_result(self):
        t = reconstruct_transcript(_basic_run())
        bash = next(tc for tc in t.tool_calls if tc.tool_name == "Bash")
        assert bash.input == {"command": "echo 4"}
        assert bash.output == "4"          # from the UserMessage ToolResultBlock
        assert bash.status == "ok"
        assert bash.tool_type == "function"
        assert bash.call_id == "tu1"

    def test_assistant_messages_become_llm_calls_with_turns(self):
        t = reconstruct_transcript(_basic_run())
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert len(llm) == 2
        assert all(tc.tool_name == "llm.generation" for tc in llm)
        assert llm[0].turn_index == 1
        assert llm[1].turn_index == 2
        assert llm[0].tokens.input_tokens == 1000

    def test_cli_total_cost_attached_to_final_llm_call(self):
        t = reconstruct_transcript(_basic_run())
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert llm[0].cost is None
        assert llm[1].cost.total_usd == pytest.approx(0.0123)
        assert llm[1].cost.metadata.get("source") == "claude_agent_cli"
        # Whole-run cost is the CLI total, not a per-call sum.
        assert t.sum_cost().total_usd == pytest.approx(0.0123)
        assert t.metadata["total_cost_usd"] == pytest.approx(0.0123)

    def test_thinking_becomes_reasoning_step(self):
        t = reconstruct_transcript(_basic_run())
        assert any("I'll run echo" in s for s in t.reasoning_steps)

    def test_outcome_from_result_message(self):
        t = reconstruct_transcript(_basic_run())
        assert t.outcome.output_data["final_output"] == "The answer is 4."

    def test_duration_from_result(self):
        t = reconstruct_transcript(_basic_run())
        assert t.total_duration_ms == pytest.approx(4200.0)

    def test_error_tool_result_marks_error(self):
        msgs = [
            _assistant([_tool_use("t1", "Bash", {"command": "bad"})], stop_reason="tool_use"),
            _user([_tool_result("t1", "command failed", is_error=True)]),
        ]
        t = reconstruct_transcript(msgs)
        bash = next(tc for tc in t.tool_calls if tc.tool_name == "Bash")
        assert bash.status == "error"
        assert bash.error["message"] == "command failed"


class TestSubagents:
    def test_parent_tool_use_id_tags_agent_name(self):
        msgs = [
            _user("review the code"),
            # main thread spawns a Task sub-agent
            _assistant([_tool_use("task1", "Task",
                                  {"subagent_type": "code-reviewer", "prompt": "review"})],
                       stop_reason="tool_use", message_id="a1"),
            # sub-agent's own assistant message + tool call carry parent_tool_use_id
            _assistant([_tool_use("sub1", "Read", {"path": "x.py"})],
                       stop_reason="tool_use", parent_tool_use_id="task1", message_id="a2"),
            _user([_tool_result("sub1", "file contents")], parent_tool_use_id="task1"),
            _user([_tool_result("task1", "review done")]),
            _result(result="LGTM"),
        ]
        t = reconstruct_transcript(msgs)
        read = next(tc for tc in t.tool_calls if tc.tool_name == "Read")
        assert read.agent_name == "code-reviewer"
        # main-thread Task call has no sub-agent parent
        task = next(tc for tc in t.tool_calls if tc.tool_name == "Task")
        assert task.agent_name is None


class TestServerTools:
    def test_server_tool_use_and_inline_result(self):
        msgs = [
            _assistant([
                _tool_use("s1", "web_search", {"query": "weather SF"}),
                _server_tool_result("s1", {"type": "web_search_result", "count": 3}),
                _text("It is sunny."),
            ]),
            _result(result="It is sunny."),
        ]
        t = reconstruct_transcript(msgs)
        search = next(tc for tc in t.tool_calls if tc.tool_name == "web_search")
        assert search.tool_type == "search"
        assert search.output == {"type": "web_search_result", "count": 3} or \
            "web_search_result" in str(search.output)

    def test_mcp_tool_type(self):
        msgs = [_assistant([_tool_use("m1", "mcp__github__list_prs", {})],
                           stop_reason="tool_use")]
        t = reconstruct_transcript(msgs)
        tc = next(tc for tc in t.tool_calls if tc.tool_name.startswith("mcp__"))
        assert tc.tool_type == "mcp"


class TestRobustness:
    def test_stream_and_ratelimit_events_ignored(self):
        msgs = [
            SimpleNamespace(uuid="u1", session_id="s", event={"type": "delta"},
                            parent_tool_use_id=None),                       # StreamEvent
            SimpleNamespace(rate_limit_info=SimpleNamespace(status="allowed"),
                            uuid="u2", session_id="s"),                     # RateLimitEvent
            _assistant([_text("hi")]),
        ]
        t = reconstruct_transcript(msgs)
        assert len(t.tool_calls) == 1  # only the assistant llm call

    def test_empty_messages(self):
        t = reconstruct_transcript([])
        assert t.tool_calls == []
        assert t.trial_id == ""

    def test_orphan_tool_result_ignored(self):
        t = reconstruct_transcript([_user([_tool_result("nope", "x")])])
        assert t.tool_calls == []


class TestAsyncStream:
    async def test_from_async_stream(self):
        async def gen():
            for m in _basic_run():
                yield m
        t = await reconstruct_transcript_from_stream(gen())
        assert t.trial_id == "sess-1"
        assert any(tc.tool_name == "Bash" for tc in t.tool_calls)


class TestGradingIntegration:
    async def test_tool_usage_grader(self):
        t = reconstruct_transcript(_basic_run())
        grader = get_grader("tool_usage")({"required_tools": ["Bash"]})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.passed is True

    async def test_cost_budget_grader_sees_cli_total(self):
        t = reconstruct_transcript(_basic_run())
        grader = get_grader("cost_budget")({"max_cost_usd": 1.0, "max_tokens": 100_000})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.details["total_cost_usd"] == pytest.approx(0.0123)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Offline stream-json (wire dict) path
# ---------------------------------------------------------------------------


def _wire_run():
    """stream-json wire dicts: init -> prompt -> assistant(think+Bash) -> result -> assistant."""
    return [
        {"type": "system", "subtype": "init", "session_id": "cc-1", "cwd": "/repo"},
        {"type": "user", "session_id": "cc-1",
         "message": {"role": "user", "content": "What's 2+2? use bash"}},
        {"type": "assistant", "session_id": "cc-1",
         "message": {"role": "assistant", "id": "msg_1", "model": "claude-opus-4-8",
                     "stop_reason": "tool_use",
                     "usage": {"input_tokens": 1200, "output_tokens": 30},
                     "content": [
                         {"type": "thinking", "thinking": "I'll run echo.", "signature": "s"},
                         {"type": "tool_use", "id": "tu1", "name": "Bash",
                          "input": {"command": "echo 4"}},
                     ]}},
        {"type": "user", "session_id": "cc-1",
         "message": {"role": "user", "content": [
             {"type": "tool_result", "tool_use_id": "tu1", "content": "4", "is_error": False}]}},
        {"type": "assistant", "session_id": "cc-1",
         "message": {"role": "assistant", "id": "msg_2", "model": "claude-opus-4-8",
                     "stop_reason": "end_turn",
                     "usage": {"input_tokens": 15, "output_tokens": 8},
                     "content": [{"type": "text", "text": "The answer is 4."}]}},
        {"type": "result", "subtype": "success", "session_id": "cc-1", "is_error": False,
         "num_turns": 2, "duration_ms": 5300, "duration_api_ms": 5000,
         "total_cost_usd": 0.0456, "result": "The answer is 4.",
         "modelUsage": {"claude-opus-4-8": {"input_tokens": 1215}}, "permission_denials": []},
    ]


class TestWireStreamJson:
    def test_from_wire_dicts(self):
        t = reconstruct_transcript_from_wire(_wire_run())
        assert t.trial_id == "cc-1"
        bash = next(tc for tc in t.tool_calls if tc.tool_name == "Bash")
        assert bash.input == {"command": "echo 4"}      # nested under message.content
        assert bash.output == "4"                        # stitched from user tool_result
        assert bash.turn_index == 1
        llm = [tc for tc in t.tool_calls if tc.tool_type == "llm"]
        assert llm[0].tokens.input_tokens == 1200
        # CLI total attached to final llm call.
        assert llm[-1].cost.total_usd == pytest.approx(0.0456)
        assert t.sum_cost().total_usd == pytest.approx(0.0456)
        assert t.outcome.output_data["final_output"] == "The answer is 4."
        assert t.total_duration_ms == pytest.approx(5300.0)

    def test_wire_thinking_becomes_reasoning(self):
        t = reconstruct_transcript_from_wire(_wire_run())
        assert any("I'll run echo" in s for s in t.reasoning_steps)

    def test_import_from_jsonl_file(self, tmp_path):
        p = tmp_path / "run.stream.jsonl"
        p.write_text("\n".join(json.dumps(e) for e in _wire_run()) + "\n", encoding="utf-8")
        t = import_claude_stream_json(p)
        assert t.trial_id == "cc-1"
        assert any(tc.tool_name == "Bash" for tc in t.tool_calls)

    def test_import_from_json_array(self, tmp_path):
        p = tmp_path / "run.json"
        p.write_text(json.dumps(_wire_run()), encoding="utf-8")
        t = import_claude_stream_json(p)
        assert t.trial_id == "cc-1"

    def test_malformed_line_skipped(self, tmp_path):
        good = "\n".join(json.dumps(e) for e in _wire_run())
        p = tmp_path / "run.stream.jsonl"
        p.write_text(good + "\n{ not json }\n", encoding="utf-8")
        t = import_claude_stream_json(p)  # must not raise
        assert any(tc.tool_name == "Bash" for tc in t.tool_calls)

    def test_stream_and_ratelimit_wire_events_skipped(self):
        events = [
            {"type": "stream_event", "uuid": "u", "session_id": "cc-2", "event": {"x": 1}},
            {"type": "rate_limit_event", "uuid": "u2", "session_id": "cc-2",
             "rate_limit_info": {"status": "allowed"}},
            {"type": "assistant", "session_id": "cc-2",
             "message": {"role": "assistant", "model": "claude-opus-4-8",
                         "content": [{"type": "text", "text": "hi"}]}},
        ]
        t = reconstruct_transcript_from_wire(events)
        assert len(t.tool_calls) == 1  # only the assistant llm call

    def test_empty_content_returns_empty(self):
        assert reconstruct_transcript_from_wire([]).tool_calls == []

    async def test_grades_on_wire_transcript(self):
        t = reconstruct_transcript_from_wire(_wire_run())
        grader = get_grader("tool_usage")({"required_tools": ["Bash"]})
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))
        assert result.passed is True
