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
from compass.integrations.claude_agent import (
    WireReconstructor,
    reconstruct_transcript_from_wire,
)

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


class TestWireReconstructorStreaming:
    """The incremental form, used by the claude_code adapter to parse the CLI's
    stdout as it arrives rather than after the process exits."""

    def test_incremental_matches_batch(self):
        events = _wire_run()

        r = WireReconstructor()
        for event in events:
            r.feed(event)
        streamed = r.finish()
        batched = reconstruct_transcript_from_wire(events)

        assert [tc.tool_name for tc in streamed.tool_calls] == [
            tc.tool_name for tc in batched.tool_calls
        ]
        assert streamed.sum_cost().total_usd == batched.sum_cost().total_usd
        assert streamed.sum_tokens().total_tokens == batched.sum_tokens().total_tokens
        assert streamed.outcome.output_data == batched.outcome.output_data

    def test_feed_line_parses_raw_stdout(self):
        r = WireReconstructor(task_id="t")
        for event in _wire_run():
            assert r.feed_line(json.dumps(event) + "\n") is True
        assert len(r.finish().tool_calls) == 3

    def test_feed_line_reports_non_json_instead_of_raising(self):
        """The CLI interleaves the odd non-JSON line; a partial trace beats a
        crashed harness."""
        r = WireReconstructor()
        assert r.feed_line("Loading plugins...\n") is False
        assert r.feed_line("\n") is False
        assert r.feed_line('"just a string"') is False
        assert r.feed_line(json.dumps(_wire_run()[2])) is True  # the assistant turn
        assert len(r.finish().tool_calls) == 2  # tool_use + llm.generation

    def test_finish_is_valid_mid_stream(self):
        """A run killed by a timeout still yields the steps it completed."""
        events = _wire_run()
        r = WireReconstructor()
        for event in events[:3]:      # init + prompt + the first assistant turn
            r.feed(event)

        partial = r.finish()
        assert len(partial.tool_calls) == 2   # tool_use + its llm.generation
        assert partial.sum_cost().total_usd == 0.0   # no result event, no total

        # ... and feeding the rest afterwards still completes it.
        for event in events[3:]:
            r.feed(event)
        assert r.finish().sum_cost().total_usd == pytest.approx(0.0456)


class TestStateDeltaMapping:
    """Successful file edits become StateChanges, so the state_delta grader has
    something to grade. Before this mapping it passed vacuously on every Claude
    trace, because no importer ever filled the slot."""

    @staticmethod
    def _use(tid, name, inp):
        return {"type": "tool_use", "id": tid, "name": name, "input": inp}

    @staticmethod
    def _result(tid, text, err=False):
        return {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": tid, "content": text,
             "is_error": err}]}}

    def _run(self, uses, results, *, cwd="/wt"):
        events = []
        if cwd is not None:
            events.append(
                {"type": "system", "subtype": "init", "session_id": "s", "cwd": cwd}
            )
        events.append({"type": "assistant", "session_id": "s", "message": {
            "role": "assistant", "id": "m1", "model": "x", "usage": {},
            "content": uses}})
        events.extend(results)
        return reconstruct_transcript_from_wire(events)

    def test_edit_records_an_update(self):
        t = self._run(
            [self._use("t1", "Edit", {"file_path": "/wt/src/api.py"})],
            [self._result("t1", "ok")],
        )
        (change,) = t.all_state_changes()
        assert change.kind == "file"
        assert change.op == "update"
        assert change.target == "src/api.py"
        assert change.metadata["tool"] == "Edit"

    def test_every_editing_tool_is_mapped(self):
        t = self._run(
            [
                self._use("t1", "Edit", {"file_path": "/wt/a.py"}),
                self._use("t2", "MultiEdit", {"file_path": "/wt/b.py"}),
                self._use("t3", "NotebookEdit", {"notebook_path": "/wt/c.ipynb"}),
                self._use("t4", "Write", {"file_path": "/wt/d.py"}),
            ],
            [self._result(f"t{i}", "ok") for i in range(1, 5)],
        )
        assert [c.target for c in t.all_state_changes()] == [
            "a.py", "b.py", "c.ipynb", "d.py"
        ]

    def test_read_only_tools_record_nothing(self):
        t = self._run(
            [
                self._use("t1", "Read", {"file_path": "/wt/a.py"}),
                self._use("t2", "Grep", {"pattern": "x"}),
                self._use("t3", "Glob", {"pattern": "**/*.py"}),
            ],
            [self._result(f"t{i}", "ok") for i in range(1, 4)],
        )
        assert t.all_state_changes() == []

    def test_bash_records_nothing_even_when_it_deletes(self):
        """Parsing shell is not something this layer should guess at — a wrong
        delta is worse than a missing one."""
        t = self._run(
            [self._use("t1", "Bash", {"command": "rm -rf tests/"})],
            [self._result("t1", "")],
        )
        assert t.all_state_changes() == []

    def test_a_failed_edit_changes_nothing(self):
        t = self._run(
            [self._use("t1", "Edit", {"file_path": "/wt/src/api.py"})],
            [self._result("t1", "Permission denied", err=True)],
        )
        assert t.all_state_changes() == []

    def test_an_unanswered_call_changes_nothing(self):
        """A run cut short mid-edit did not necessarily complete that edit."""
        t = self._run([self._use("t1", "Edit", {"file_path": "/wt/a.py"})], [])
        assert t.all_state_changes() == []

    def test_write_op_comes_from_the_result(self):
        t = self._run(
            [
                self._use("t1", "Write", {"file_path": "/wt/new.py"}),
                self._use("t2", "Write", {"file_path": "/wt/old.py"}),
            ],
            [
                self._result("t1", "File created successfully at: /wt/new.py"),
                self._result("t2", "The file /wt/old.py has been updated."),
            ],
        )
        assert [(c.target, c.op) for c in t.all_state_changes()] == [
            ("new.py", "create"),
            ("old.py", "update"),
        ]

    def test_targets_are_relative_to_the_session_cwd(self):
        """A claude_code trial runs in a throwaway worktree, so an absolute
        target would be a different random path every run and no user glob
        could ever match it."""
        t = self._run(
            [self._use("t1", "Edit", {"file_path": "/wt/pkg/mod.py"})],
            [self._result("t1", "ok")],
            cwd="/wt",
        )
        (change,) = t.all_state_changes()
        assert change.target == "pkg/mod.py"
        assert change.metadata["absolute_path"] == "/wt/pkg/mod.py"

    def test_paths_outside_the_workspace_stay_absolute(self):
        t = self._run(
            [self._use("t1", "Edit", {"file_path": "/etc/hosts"})],
            [self._result("t1", "ok")],
            cwd="/wt",
        )
        (change,) = t.all_state_changes()
        assert change.target == "/etc/hosts"
        assert "absolute_path" not in change.metadata

    def test_no_cwd_reported_keeps_the_path_as_given(self):
        t = self._run(
            [self._use("t1", "Edit", {"file_path": "/wt/a.py"})],
            [self._result("t1", "ok")],
            cwd=None,
        )
        (change,) = t.all_state_changes()
        assert change.target == "/wt/a.py"

    def test_a_pathless_edit_is_skipped(self):
        t = self._run(
            [self._use("t1", "Edit", {"old_string": "x"})],
            [self._result("t1", "ok")],
        )
        assert t.all_state_changes() == []

    def test_changes_attribute_to_the_call_that_made_them(self):
        """Per-call attribution is the point — an outcome-level file listing
        cannot say which step touched what."""
        t = self._run(
            [
                self._use("t1", "Edit", {"file_path": "/wt/a.py"}),
                self._use("t2", "Edit", {"file_path": "/wt/b.py"}),
            ],
            [self._result("t1", "ok"), self._result("t2", "ok")],
        )
        by_call = {
            tc.call_id: [sc.target for sc in tc.state_delta]
            for tc in t.tool_calls
            if tc.state_delta
        }
        assert by_call == {"t1": ["a.py"], "t2": ["b.py"]}

    async def test_state_delta_grader_now_catches_an_out_of_remit_write(self):
        t = self._run(
            [
                self._use("t1", "Edit", {"file_path": "/wt/src/api.py"}),
                self._use("t2", "Edit", {"file_path": "/wt/tests/test_api.py"}),
            ],
            [self._result("t1", "ok"), self._result("t2", "ok")],
        )
        grader = get_grader("state_delta")({
            "require": [{"kind": "file", "target": "src/*"}],
            "forbid": [{"kind": "file", "target": "tests/*"}],
        })
        result = await grader.grade(GradeContext(transcript=t, outcome=t.outcome))

        assert result.passed is False
        assert any("tests/*" in f for f in result.details["failures"])
        # ... and the violation names the step responsible.
        assert result.details["violations"][0]["call_id"] == "t2"

    async def test_readonly_guard_is_no_longer_vacuous(self):
        edited = self._run(
            [self._use("t1", "Write", {"file_path": "/wt/a.py"})],
            [self._result("t1", "ok")],
        )
        looked = self._run(
            [self._use("t1", "Read", {"file_path": "/wt/a.py"})],
            [self._result("t1", "contents")],
        )
        grader = get_grader("state_delta")({"readonly": True})

        assert (await grader.grade(
            GradeContext(transcript=looked, outcome=looked.outcome)
        )).passed is True
        assert (await grader.grade(
            GradeContext(transcript=edited, outcome=edited.outcome)
        )).passed is False


class TestCachedTokens:
    """Regression: a run's token count was a rounding error of the truth.

    A coding agent's system prompt and tool definitions are cached, so
    Anthropic reports a handful of uncached ``input_tokens`` next to tens of
    thousands of cached ones. Treating the cache as metadata made a real
    26,000-token turn report 63, while the cost column reported the full
    charge — two numbers about the same call that could not both be right.
    """

    # Verbatim from a real `claude -p "say hi"` result event.
    REAL_USAGE = {
        "input_tokens": 10,
        "cache_creation_input_tokens": 7820,
        "cache_read_input_tokens": 18178,
        "output_tokens": 53,
    }

    def _transcript(self, usage):
        return reconstruct_transcript_from_wire([
            {"type": "assistant", "session_id": "s", "message": {
                "role": "assistant", "id": "m", "model": "claude-haiku-4-5",
                "usage": usage, "content": [{"type": "text", "text": "Hi!"}]}},
            {"type": "result", "subtype": "success", "session_id": "s",
             "is_error": False, "num_turns": 1, "duration_ms": 5116,
             "total_cost_usd": 0.0177328, "result": "Hi!",
             "permission_denials": []},
        ])

    def test_total_accounts_for_every_token_the_model_processed(self):
        tokens = self._transcript(self.REAL_USAGE).sum_tokens()
        assert tokens.total_tokens == sum(self.REAL_USAGE.values()) == 26061

    def test_cached_traffic_is_first_class_not_metadata(self):
        tokens = self._transcript(self.REAL_USAGE).sum_tokens()
        assert tokens.cache_read_tokens == 18178
        assert tokens.cache_creation_tokens == 7820

    def test_input_tokens_still_mean_what_the_provider_means(self):
        """Uncached prompt tokens. Folding the cache in here would misreport
        what the API said, and the two are priced differently."""
        tokens = self._transcript(self.REAL_USAGE).sum_tokens()
        assert tokens.input_tokens == 10
        assert tokens.billable_input_tokens == 26008

    def test_a_turn_that_is_only_cache_reads_is_still_recorded(self):
        """Previously dropped: the guard required input or output to be set."""
        tokens = self._transcript(
            {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 9000}
        ).sum_tokens()
        assert tokens.total_tokens == 9000
        assert tokens.cache_read_tokens == 9000

    def test_an_uncached_run_is_unchanged(self):
        tokens = self._transcript(
            {"input_tokens": 1200, "output_tokens": 30}
        ).sum_tokens()
        assert tokens.total_tokens == 1230
        assert tokens.cache_read_tokens == 0

    def test_cache_survives_a_save_load_round_trip(self, tmp_path):
        from compass.core.transcript import Transcript

        path = tmp_path / "t.json"
        self._transcript(self.REAL_USAGE).save(path)
        tokens = Transcript.load(path).sum_tokens()
        assert tokens.total_tokens == 26061
        assert tokens.cache_read_tokens == 18178


class TestUnhandledEvents:
    """Events this mapping does not model are counted, not dropped.

    The motivating case is ``rate_limit_event``: a throttled run and a slow one
    are indistinguishable in the trace otherwise, so a harness problem reads as
    a slow model — and on a subscription, where a long comparison run really
    can hit a usage window, that is the difference between a result and an
    artefact.
    """

    BASE = [
        {"type": "system", "subtype": "init", "session_id": "s", "cwd": "/w"},
        {"type": "assistant", "session_id": "s", "message": {
            "role": "assistant", "id": "m", "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [{"type": "text", "text": "working"}]}},
        {"type": "result", "subtype": "success", "session_id": "s",
         "is_error": False, "num_turns": 1, "duration_ms": 900,
         "total_cost_usd": 0.01, "result": "done", "permission_denials": []},
    ]

    def test_repeats_of_one_unknown_type_accumulate(self):
        events = self.BASE[:2] + [
            {"type": "mystery_event", "whatever": 1},
            {"type": "mystery_event", "whatever": 2},
        ] + self.BASE[2:]
        t = reconstruct_transcript_from_wire(events)
        assert t.metadata["unhandled_events"]["mystery_event"] == 2

    def test_an_event_type_that_does_not_exist_yet_is_still_counted(self):
        """Counting by name needs no schema, so the CLI can grow without this
        importer silently swallowing the new thing."""
        events = self.BASE[:2] + [
            {"type": "some_future_event", "payload": {"whatever": 1}}
        ] + self.BASE[2:]
        t = reconstruct_transcript_from_wire(events)
        assert t.metadata["unhandled_events"] == {"some_future_event": 1}

    def test_a_typeless_event_is_counted_as_unknown(self):
        t = reconstruct_transcript_from_wire(self.BASE[:2] + [{"nope": 1}] + self.BASE[2:])
        assert t.metadata["unhandled_events"] == {"unknown": 1}

    def test_partial_deltas_are_counted_not_stored(self):
        """A stream can carry thousands; the count is the whole record."""
        events = self.BASE[:2] + [
            {"type": "stream_event", "event": {"type": "content_block_delta"}}
            for _ in range(200)
        ] + self.BASE[2:]
        t = reconstruct_transcript_from_wire(events)
        assert t.metadata["unhandled_events"] == {"stream_event": 200}
        assert len(t.tool_calls) == 1  # nothing leaked into the trace

    def test_a_clean_run_carries_no_such_key(self):
        """Absence is the signal; an empty dict everywhere would be noise."""
        t = reconstruct_transcript_from_wire(self.BASE)
        assert "unhandled_events" not in t.metadata

    def test_counting_works_on_the_sdk_object_path_too(self):
        partial_delta = SimpleNamespace(event={"type": "content_block_delta"})
        t = reconstruct_transcript([*_basic_run(), partial_delta])
        assert t.metadata["unhandled_events"] == {"stream": 1}

    def test_an_unrecognised_object_is_counted_under_its_class_name(self):
        class SomeNewMessage:
            pass

        t = reconstruct_transcript([*_basic_run(), SomeNewMessage()])
        assert t.metadata["unhandled_events"] == {"SomeNewMessage": 1}

    def test_counts_are_current_when_finish_is_called_mid_stream(self):
        r = WireReconstructor()
        for event in self.BASE[:2]:
            r.feed(event)
        r.feed({"type": "mystery_event"})
        assert r.finish().metadata["unhandled_events"] == {"mystery_event": 1}

        r.feed({"type": "mystery_event"})
        assert r.finish().metadata["unhandled_events"] == {"mystery_event": 2}


class TestRateLimitEvents:
    """Parsed from a real event captured by save_stream_to during a live run.

    It matters for a comparison on a subscription: overage is commonly disabled
    at the org level, so hitting the five-hour window gets requests *rejected*.
    A run that ran out of quota then looks like a worse model rather than an
    incomplete arm.
    """

    # Verbatim from coding-eval-run/streams/, `claude -p` on a subscription.
    REAL_EVENT = {
        "type": "rate_limit_event",
        "rate_limit_info": {
            "status": "allowed",
            "resetsAt": 1786448400,
            "rateLimitType": "five_hour",
            "overageStatus": "rejected",
            "overageDisabledReason": "org_level_disabled",
            "isUsingOverage": False,
        },
        "uuid": "b3a63c28-6192-46b7-8bf1-0166fa1aa804",
        "session_id": "8af748af",
    }

    BASE = [
        {"type": "assistant", "session_id": "s", "message": {
            "role": "assistant", "id": "m", "model": "claude-haiku-4-5",
            "usage": {"input_tokens": 10, "output_tokens": 5},
            "content": [{"type": "text", "text": "done"}]}},
    ]

    def test_the_provider_payload_is_kept_verbatim(self):
        """No second-guessing the status vocabulary — it is the CLI's."""
        t = reconstruct_transcript_from_wire([*self.BASE, self.REAL_EVENT])
        assert t.metadata["rate_limit"]["latest"] == self.REAL_EVENT["rate_limit_info"]

    def test_statuses_are_tallied(self):
        throttled = {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour"},
        }
        t = reconstruct_transcript_from_wire(
            [*self.BASE, self.REAL_EVENT, throttled, throttled]
        )
        record = t.metadata["rate_limit"]
        assert record["events"] == 3
        assert record["status_counts"] == {"allowed": 1, "rejected": 2}
        assert record["latest"]["status"] == "rejected"

    def test_it_is_no_longer_filed_as_unhandled(self):
        t = reconstruct_transcript_from_wire([*self.BASE, self.REAL_EVENT])
        assert "unhandled_events" not in t.metadata

    def test_a_run_with_no_quota_notices_carries_no_key(self):
        t = reconstruct_transcript_from_wire(self.BASE)
        assert "rate_limit" not in t.metadata

    def test_an_event_with_no_info_is_ignored(self):
        t = reconstruct_transcript_from_wire(
            [*self.BASE, {"type": "rate_limit_event"}]
        )
        assert "rate_limit" not in t.metadata

    def test_the_sdk_object_path_records_it_too(self):
        msg = SimpleNamespace(rate_limit_info=self.REAL_EVENT["rate_limit_info"])
        t = reconstruct_transcript([*_basic_run(), msg])
        assert t.metadata["rate_limit"]["status_counts"] == {"allowed": 1}
