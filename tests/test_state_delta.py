"""Tests for StateChange / ToolCall.state_delta (protocol 1.3) and StateDeltaGrader."""

from __future__ import annotations

import pytest

from compass.core.transcript import (
    Outcome,
    StateChange,
    ToolCall,
    Transcript,
)
from compass.graders.base import GradeContext
from compass.graders.code.common.transcript_graders import StateDeltaGrader
from compass.graders.registry import get_grader

# ===================================================================
# StateChange model
# ===================================================================


class TestStateChange:
    def test_to_from_dict_roundtrip(self):
        sc = StateChange(
            kind="file",
            op="delete",
            target="/etc/nginx/nginx.conf",
            before="sha256:aaa",
            after=None,
            metadata={"bytes": 1024},
        )
        restored = StateChange.from_dict(sc.to_dict())
        assert restored == sc

    def test_from_dict_defaults(self):
        sc = StateChange.from_dict({"kind": "db", "op": "update"})
        assert sc.target == ""
        assert sc.before is None
        assert sc.after is None
        assert sc.metadata == {}


# ===================================================================
# ToolCall.state_delta (protocol field)
# ===================================================================


class TestToolCallStateDelta:
    def test_default_empty(self):
        tc = ToolCall(tool_name="sandbox.bash")
        assert tc.state_delta == []

    def test_post_init_converts_dicts(self):
        tc = ToolCall(
            tool_name="sandbox.bash",
            state_delta=[{"kind": "file", "op": "create", "target": "/tmp/x"}],
        )
        assert isinstance(tc.state_delta[0], StateChange)
        assert tc.state_delta[0].target == "/tmp/x"

    def test_dict_roundtrip(self):
        tc = ToolCall(
            tool_name="sandbox.bash",
            state_delta=[
                StateChange(kind="file", op="create", target="/tmp/a"),
                StateChange(kind="env", op="update", target="PATH"),
            ],
        )
        restored = ToolCall.from_dict(tc.to_dict())
        assert restored.state_delta == tc.state_delta

    def test_from_dict_legacy_without_state_delta(self):
        """Pre-1.3 transcripts load with an empty state_delta."""
        tc = ToolCall.from_dict({"tool_name": "sandbox.bash", "args": {}})
        assert tc.state_delta == []

    def test_add_tool_call_accepts_state_delta(self):
        transcript = Transcript(task_id="t", trial_id="1")
        transcript.add_tool_call(
            tool_name="sandbox.bash",
            input={"command": "rm -f /tmp/x"},
            state_delta=[{"kind": "file", "op": "delete", "target": "/tmp/x"}],
        )
        assert transcript.tool_calls[0].state_delta[0].op == "delete"

    def test_all_state_changes_flattens_chronologically(self):
        transcript = Transcript(task_id="t", trial_id="1")
        transcript.add_tool_call(
            tool_name="a",
            state_delta=[StateChange(kind="file", op="create", target="/1")],
        )
        transcript.add_tool_call(tool_name="b")  # no delta
        transcript.add_tool_call(
            tool_name="c",
            state_delta=[
                StateChange(kind="file", op="update", target="/2"),
                StateChange(kind="db", op="update", target="users"),
            ],
        )
        targets = [sc.target for sc in transcript.all_state_changes()]
        assert targets == ["/1", "/2", "users"]

    def test_jsonl_roundtrip_preserves_state_delta(self):
        transcript = Transcript(task_id="t", trial_id="1")
        transcript.add_tool_call(
            tool_name="sandbox.bash",
            state_delta=[
                StateChange(
                    kind="git",
                    op="update",
                    target="refs/heads/main",
                    before="abc",
                    after="def",
                )
            ],
        )
        restored = Transcript.from_jsonl(transcript.to_jsonl())
        assert restored.tool_calls[0].state_delta == transcript.tool_calls[0].state_delta

    def test_grade_context_state_changes(self):
        transcript = Transcript(task_id="t", trial_id="1")
        transcript.add_tool_call(
            tool_name="a",
            state_delta=[StateChange(kind="file", op="create", target="/1")],
        )
        context = GradeContext(transcript=transcript, outcome=Outcome())
        assert [sc.target for sc in context.state_changes] == ["/1"]

    def test_grade_context_state_changes_without_transcript(self):
        context = GradeContext(transcript=None, outcome=Outcome())
        assert context.state_changes == []


# ===================================================================
# StateDeltaGrader
# ===================================================================


def make_context(*deltas: list[StateChange]) -> GradeContext:
    """Build a context with one tool call per delta list."""
    transcript = Transcript(task_id="t", trial_id="1")
    for delta in deltas:
        transcript.add_tool_call(tool_name="sandbox.bash", state_delta=list(delta))
    return GradeContext(transcript=transcript, outcome=Outcome())


class TestStateDeltaGrader:
    def test_registered(self):
        assert get_grader("state_delta") is StateDeltaGrader

    @pytest.mark.asyncio
    async def test_no_config_passes_trivially(self):
        context = make_context([StateChange(kind="file", op="delete", target="/x")])
        result = await StateDeltaGrader().grade(context)
        assert result.passed is True
        assert result.score == 1.0
        assert result.details["total_changes"] == 1

    @pytest.mark.asyncio
    async def test_readonly_passes_when_no_changes(self):
        context = make_context([])
        result = await StateDeltaGrader({"readonly": True}).grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_readonly_fails_on_any_change(self):
        context = make_context([StateChange(kind="file", op="update", target="/x")])
        result = await StateDeltaGrader({"readonly": True}).grade(context)
        assert result.passed is False
        assert "readonly" in result.failure_tags
        assert result.details["violations"][0]["target"] == "/x"
        # Violation attributes to the causing call for failure attribution
        assert result.details["violations"][0]["call_id"]

    @pytest.mark.asyncio
    async def test_forbid_matcher_with_target_glob(self):
        context = make_context(
            [StateChange(kind="file", op="delete", target="/etc/nginx/nginx.conf")]
        )
        grader = StateDeltaGrader(
            {"forbid": [{"kind": "file", "op": "delete", "target": "/etc/*"}]}
        )
        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_forbid_matcher_no_match_passes(self):
        context = make_context(
            [StateChange(kind="file", op="create", target="/tmp/out.txt")]
        )
        grader = StateDeltaGrader(
            {"forbid": [{"kind": "file", "op": "delete", "target": "/etc/*"}]}
        )
        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_require_matcher_found(self):
        context = make_context(
            [StateChange(kind="db", op="update", target="tickets")]
        )
        grader = StateDeltaGrader({"require": [{"kind": "db", "op": "update"}]})
        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_require_matcher_missing_fails(self):
        """require doubles as a capture check: unrecorded expected change fails."""
        context = make_context([])
        grader = StateDeltaGrader({"require": [{"kind": "db", "op": "update"}]})
        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_max_changes(self):
        context = make_context(
            [
                StateChange(kind="file", op="create", target="/1"),
                StateChange(kind="file", op="create", target="/2"),
            ]
        )
        assert (await StateDeltaGrader({"max_changes": 2}).grade(context)).passed
        assert not (await StateDeltaGrader({"max_changes": 1}).grade(context)).passed

    @pytest.mark.asyncio
    async def test_partial_score(self):
        context = make_context(
            [StateChange(kind="file", op="delete", target="/etc/passwd")]
        )
        grader = StateDeltaGrader(
            {
                "forbid": [{"op": "delete"}],  # fails
                "require": [{"kind": "file"}],  # passes
            }
        )
        result = await grader.grade(context)
        assert result.passed is False
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_missing_transcript_errors(self):
        context = GradeContext(transcript=None, outcome=Outcome())
        result = await StateDeltaGrader({"readonly": True}).grade(context)
        assert result.passed is False
        assert result.error
