"""Tests for transcript-focused graders."""

from __future__ import annotations

import pytest

from compass.core.transcript import CostInfo, Outcome, TokenUsage, ToolCall, Transcript
from compass.graders.base import GradeContext
from compass.graders.code.common.transcript_graders import (
    CostBudgetGrader,
    LoopDetectionGrader,
)

# ===================================================================
# CostBudgetGrader tests
# ===================================================================


class TestCostBudgetGraderProtocol:
    """Tests for CostBudgetGrader reading protocol fields."""

    @pytest.fixture
    def make_context(self):
        """Factory to create GradeContext with tool calls."""

        def _make(tool_calls: list[ToolCall]) -> GradeContext:
            transcript = Transcript(task_id="test-task", trial_id="test-trial")
            transcript.tool_calls = tool_calls
            return GradeContext(
                transcript=transcript,
                outcome=Outcome(),
            )

        return _make

    @pytest.mark.asyncio
    async def test_cost_budget_reads_tc_cost(self, make_context):
        """CostBudgetGrader reads cost from tc.cost.total_usd."""
        tool_calls = [
            ToolCall(
                tool_name="llm_call",
                cost=CostInfo(total_usd=0.10),
            ),
            ToolCall(
                tool_name="llm_call",
                cost=CostInfo(total_usd=0.15),
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_cost_usd": 1.0})
        result = await grader.grade(context)

        assert result.details["total_cost_usd"] == 0.25
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_fallback_to_config(self, make_context):
        """CostBudgetGrader falls back to config when tc.cost is None."""
        tool_calls = [
            ToolCall(tool_name="search"),
            ToolCall(tool_name="search"),
            ToolCall(tool_name="execute"),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({
            "max_cost_usd": 1.0,
            "cost_per_tool": {"search": 0.05, "execute": 0.10},
        })
        result = await grader.grade(context)

        # 2 * 0.05 + 1 * 0.10 = 0.20
        assert result.details["total_cost_usd"] == 0.20
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_mixed_protocol_and_fallback(self, make_context):
        """CostBudgetGrader handles mix of protocol cost and config fallback."""
        tool_calls = [
            ToolCall(
                tool_name="llm_call",
                cost=CostInfo(total_usd=0.30),
            ),
            ToolCall(tool_name="search"),  # No cost, uses config
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({
            "max_cost_usd": 1.0,
            "cost_per_tool": {"search": 0.02},
        })
        result = await grader.grade(context)

        # 0.30 + 0.02 = 0.32
        assert result.details["total_cost_usd"] == 0.32
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_reads_tc_tokens(self, make_context):
        """CostBudgetGrader reads tokens from tc.tokens.total_tokens."""
        tool_calls = [
            ToolCall(
                tool_name="llm_call",
                tokens=TokenUsage(input_tokens=500, output_tokens=200),
            ),
            ToolCall(
                tool_name="llm_call",
                tokens=TokenUsage(input_tokens=300, output_tokens=100),
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_tokens": 10000})
        result = await grader.grade(context)

        # 700 + 400 = 1100
        assert result.details["total_tokens"] == 1100
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_tokens_fallback_to_result(self, make_context):
        """CostBudgetGrader falls back to result.tokens_used when tc.tokens is None."""
        tool_calls = [
            ToolCall(
                tool_name="old_tool",
                output={"tokens_used": 500},
            ),
            ToolCall(
                tool_name="old_tool",
                output={"tokens_used": 300},
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_tokens": 10000})
        result = await grader.grade(context)

        assert result.details["total_tokens"] == 800
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_tokens_mixed_protocol_and_fallback(self, make_context):
        """CostBudgetGrader handles mix of protocol tokens and legacy fallback."""
        tool_calls = [
            ToolCall(
                tool_name="new_tool",
                tokens=TokenUsage(input_tokens=1000, output_tokens=500),
            ),
            ToolCall(
                tool_name="old_tool",
                output={"tokens_used": 200},
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_tokens": 10000})
        result = await grader.grade(context)

        # 1500 + 200 = 1700
        assert result.details["total_tokens"] == 1700
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_cost_budget_exceeds_limit(self, make_context):
        """CostBudgetGrader fails when cost exceeds budget."""
        tool_calls = [
            ToolCall(
                tool_name="expensive",
                cost=CostInfo(total_usd=0.60),
            ),
            ToolCall(
                tool_name="expensive",
                cost=CostInfo(total_usd=0.50),
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_cost_usd": 1.0})
        result = await grader.grade(context)

        assert result.details["total_cost_usd"] == 1.10
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_cost_budget_exceeds_tokens(self, make_context):
        """CostBudgetGrader fails when tokens exceed limit."""
        tool_calls = [
            ToolCall(
                tool_name="heavy",
                tokens=TokenUsage(total_tokens=60000),
            ),
            ToolCall(
                tool_name="heavy",
                tokens=TokenUsage(total_tokens=50000),
            ),
        ]
        context = make_context(tool_calls)

        grader = CostBudgetGrader({"max_tokens": 100000})
        result = await grader.grade(context)

        assert result.details["total_tokens"] == 110000
        assert result.passed is False


# ===================================================================
# LoopDetectionGrader tests
# ===================================================================


class TestLoopDetectionGrader:
    """Tests for LoopDetectionGrader detecting loops and wasteful patterns."""

    @pytest.fixture
    def make_context(self):
        """Factory to create GradeContext with tool calls."""

        def _make(tool_calls: list[ToolCall]) -> GradeContext:
            transcript = Transcript(task_id="test-task", trial_id="test-trial")
            transcript.tool_calls = tool_calls
            return GradeContext(
                transcript=transcript,
                outcome=Outcome(),
            )

        return _make

    @pytest.mark.asyncio
    async def test_no_loops_passes(self, make_context):
        """Normal execution without loops should pass."""
        tool_calls = [
            ToolCall(tool_name="read_file", input={"path": "a.py"}),
            ToolCall(tool_name="edit_file", input={"path": "a.py"}),
            ToolCall(tool_name="run_tests", input={}),
        ]
        context = make_context(tool_calls)

        grader = LoopDetectionGrader()
        result = await grader.grade(context)

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["issues_found"] == 0

    @pytest.mark.asyncio
    async def test_empty_tool_calls_passes(self, make_context):
        """Empty tool calls should pass."""
        context = make_context([])

        grader = LoopDetectionGrader()
        result = await grader.grade(context)

        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_single_tool_repetition_detected(self, make_context):
        """Detect when same tool is called too many times."""
        tool_calls = [
            ToolCall(tool_name="search", input={"q": f"query_{i}"})
            for i in range(15)  # 15 > default max of 10
        ]
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"max_single_tool_calls": 10})
        result = await grader.grade(context)

        assert result.passed is False
        assert any(
            issue["type"] == "single_tool_repetition"
            for issue in result.details["issues"]
        )
        assert result.details["tool_distribution"]["search"] == 15

    @pytest.mark.asyncio
    async def test_exact_call_repetition_detected(self, make_context):
        """Detect identical calls (same tool + same input) repeated."""
        tool_calls = [
            ToolCall(tool_name="read_file", input={"path": "same.py"}),
            ToolCall(tool_name="read_file", input={"path": "same.py"}),
            ToolCall(tool_name="read_file", input={"path": "same.py"}),
            ToolCall(tool_name="read_file", input={"path": "same.py"}),
            ToolCall(tool_name="read_file", input={"path": "same.py"}),
        ]
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"max_exact_repetitions": 3})
        result = await grader.grade(context)

        assert result.passed is False
        assert any(
            issue["type"] == "exact_call_repetition"
            for issue in result.details["issues"]
        )

    @pytest.mark.asyncio
    async def test_sequence_pattern_loop_detected(self, make_context):
        """Detect repeating sequence patterns like A→B→A→B→A→B."""
        # Create pattern: read → edit → read → edit → read → edit → read → edit
        tool_calls = []
        for _ in range(5):  # 5 repeats > default max of 3
            tool_calls.append(ToolCall(tool_name="read_file", input={"path": "x.py"}))
            tool_calls.append(ToolCall(tool_name="edit_file", input={"path": "x.py"}))
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"max_sequence_repeats": 3})
        result = await grader.grade(context)

        assert result.passed is False
        assert any(
            issue["type"] == "sequence_pattern_loop"
            for issue in result.details["issues"]
        )
        # Check that pattern was detected
        pattern_issue = next(
            i for i in result.details["issues"]
            if i["type"] == "sequence_pattern_loop"
        )
        assert "read_file→edit_file" in pattern_issue["details"][0]["pattern"]

    @pytest.mark.asyncio
    async def test_longer_sequence_pattern_detected(self, make_context):
        """Detect longer repeating patterns like A→B→C→A→B→C."""
        tool_calls = []
        for _ in range(4):  # 4 repeats
            tool_calls.append(ToolCall(tool_name="fetch", input={}))
            tool_calls.append(ToolCall(tool_name="parse", input={}))
            tool_calls.append(ToolCall(tool_name="save", input={}))
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"max_sequence_repeats": 2})
        result = await grader.grade(context)

        assert result.passed is False
        pattern_issue = next(
            i for i in result.details["issues"]
            if i["type"] == "sequence_pattern_loop"
        )
        assert pattern_issue["details"][0]["length"] == 3  # Pattern length is 3

    @pytest.mark.asyncio
    async def test_output_repetition_detected(self, make_context):
        """Detect when same output is produced multiple times (wasted work)."""
        tool_calls = [
            ToolCall(tool_name="query", input={"sql": "SELECT 1"}, output={"result": [1]}),
            ToolCall(tool_name="query", input={"sql": "SELECT 1"}, output={"result": [1]}),
            ToolCall(tool_name="query", input={"sql": "SELECT 1"}, output={"result": [1]}),
        ]
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"check_output_repetition": True})
        result = await grader.grade(context)

        assert any(
            issue["type"] == "output_repetition"
            for issue in result.details["issues"]
        )

    @pytest.mark.asyncio
    async def test_ignore_tools_config(self, make_context):
        """Tools in ignore_tools should not be checked."""
        tool_calls = [
            ToolCall(tool_name="log", input={"msg": "step 1"}),
            ToolCall(tool_name="log", input={"msg": "step 2"}),
            ToolCall(tool_name="log", input={"msg": "step 3"}),
        ] * 5  # 15 log calls, but should be ignored
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({
            "max_single_tool_calls": 5,
            "ignore_tools": ["log"],
        })
        result = await grader.grade(context)

        assert result.passed is True
        assert result.details["total_tool_calls"] == 0  # All ignored

    @pytest.mark.asyncio
    async def test_custom_thresholds(self, make_context):
        """Custom thresholds should be respected."""
        tool_calls = [
            ToolCall(tool_name="api_call", input={"id": i})
            for i in range(20)  # 20 calls
        ]
        context = make_context(tool_calls)

        # With high threshold for all checks, should pass
        grader = LoopDetectionGrader({
            "max_single_tool_calls": 25,
            "max_sequence_repeats": 100,  # Disable sequence detection for this test
        })
        result = await grader.grade(context)
        assert result.passed is True

        # With low single tool threshold, should fail
        grader = LoopDetectionGrader({
            "max_single_tool_calls": 10,
            "max_sequence_repeats": 100,  # Focus on single tool check
        })
        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_mixed_issues_severity(self, make_context):
        """Multiple issues should be detected and reported."""
        tool_calls = []
        # Add exact repetition issue
        for _ in range(5):
            tool_calls.append(ToolCall(tool_name="read", input={"f": "same"}))
        # Add sequence pattern
        for _ in range(4):
            tool_calls.append(ToolCall(tool_name="a", input={}))
            tool_calls.append(ToolCall(tool_name="b", input={}))
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({
            "max_exact_repetitions": 2,
            "max_sequence_repeats": 2,
        })
        result = await grader.grade(context)

        assert result.passed is False
        assert result.details["issues_found"] >= 2

    @pytest.mark.asyncio
    async def test_reasoning_includes_issue_summary(self, make_context):
        """Reasoning should summarize detected issues."""
        tool_calls = [
            ToolCall(tool_name="loop_tool", input={})
            for _ in range(15)
        ]
        context = make_context(tool_calls)

        grader = LoopDetectionGrader({"max_single_tool_calls": 10})
        result = await grader.grade(context)

        assert "issue" in result.reasoning.lower()
        assert "loop_tool" in result.reasoning

    @pytest.mark.asyncio
    async def test_score_degrades_with_severity(self, make_context):
        """Score should decrease as issues become more severe."""
        # Mild case: just over threshold
        tool_calls_mild = [
            ToolCall(tool_name="tool", input={"i": i})
            for i in range(12)  # 12 > 10
        ]
        context_mild = make_context(tool_calls_mild)

        # Severe case: way over threshold
        tool_calls_severe = [
            ToolCall(tool_name="tool", input={"i": i})
            for i in range(30)  # 30 >> 10
        ]
        context_severe = make_context(tool_calls_severe)

        grader = LoopDetectionGrader({"max_single_tool_calls": 10})

        result_mild = await grader.grade(context_mild)
        result_severe = await grader.grade(context_severe)

        assert result_mild.score > result_severe.score
