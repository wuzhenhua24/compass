"""Tests for the TurnCountGrader."""

from __future__ import annotations

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders.base import GradeContext, GraderScope, GraderType
from compass.graders.code.common.transcript_graders import TurnCountGrader
from compass.graders.registry import _grader_registry, get_grader

# ===================================================================
# Helpers
# ===================================================================


def _tc(name: str = "tool", error: dict | None = None,
        tool_type: str | None = None) -> ToolCall:
    """Create a minimal ToolCall."""
    return ToolCall(
        tool_name=name,
        input={"x": 1},
        output="ok",
        status="error" if error else "ok",
        error=error,
        tool_type=tool_type,
    )


def _ctx(tool_calls: list[ToolCall] | None = None) -> GradeContext:
    """Create a GradeContext with a Transcript containing the given tool calls."""
    transcript = Transcript(
        task_id="test",
        trial_id="t1",
        tool_calls=tool_calls or [],
    )
    return GradeContext(
        prompt="test",
        transcript=transcript,
        outcome=Outcome(),
    )


# ===================================================================
# Registry
# ===================================================================


class TestTurnCountRegistry:
    def test_registered(self):
        assert "turn_count" in _grader_registry

    def test_get_grader(self):
        assert get_grader("turn_count") is TurnCountGrader

    def test_attributes(self):
        g = TurnCountGrader()
        assert g.name == "turn_count"
        assert g.grader_type == GraderType.CODE
        assert g.grader_scope == GraderScope.TRANSCRIPT


# ===================================================================
# Budget scoring (default)
# ===================================================================


class TestBudgetScoring:
    @pytest.mark.asyncio
    async def test_within_budget(self):
        g = TurnCountGrader({"max_turns": 10})
        result = await g.grade(_ctx([_tc() for _ in range(5)]))
        assert result.passed
        assert result.score == 1.0
        assert result.details["turn_count"] == 5

    @pytest.mark.asyncio
    async def test_at_limit(self):
        g = TurnCountGrader({"max_turns": 5})
        result = await g.grade(_ctx([_tc() for _ in range(5)]))
        assert result.passed
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_over_budget_graceful(self):
        g = TurnCountGrader({"max_turns": 10})
        result = await g.grade(_ctx([_tc() for _ in range(20)]))
        # 10/20 = 0.5, below default pass_threshold of 0.7
        assert not result.passed
        assert result.score == 0.5

    @pytest.mark.asyncio
    async def test_over_budget_slightly(self):
        g = TurnCountGrader({"max_turns": 10, "pass_threshold": 0.5})
        result = await g.grade(_ctx([_tc() for _ in range(15)]))
        # 10/15 ≈ 0.667 >= 0.5
        assert result.passed
        assert abs(result.score - 10 / 15) < 0.01

    @pytest.mark.asyncio
    async def test_zero_turns(self):
        g = TurnCountGrader({"max_turns": 10})
        result = await g.grade(_ctx([]))
        assert not result.passed
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_no_transcript(self):
        ctx = GradeContext(prompt="test")
        g = TurnCountGrader()
        result = await g.grade(ctx)
        assert not result.passed
        assert result.error is not None


# ===================================================================
# Linear scoring
# ===================================================================


class TestLinearScoring:
    @pytest.mark.asyncio
    async def test_at_min(self):
        g = TurnCountGrader({
            "scoring": "linear", "min_turns": 5, "max_turns": 25,
        })
        result = await g.grade(_ctx([_tc() for _ in range(5)]))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_below_min(self):
        g = TurnCountGrader({
            "scoring": "linear", "min_turns": 5, "max_turns": 25,
        })
        result = await g.grade(_ctx([_tc() for _ in range(3)]))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_at_max(self):
        g = TurnCountGrader({
            "scoring": "linear", "min_turns": 5, "max_turns": 25,
        })
        result = await g.grade(_ctx([_tc() for _ in range(25)]))
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_midpoint(self):
        g = TurnCountGrader({
            "scoring": "linear", "min_turns": 10, "max_turns": 30,
        })
        result = await g.grade(_ctx([_tc() for _ in range(20)]))
        assert abs(result.score - 0.5) < 0.01

    @pytest.mark.asyncio
    async def test_above_max(self):
        g = TurnCountGrader({
            "scoring": "linear", "min_turns": 5, "max_turns": 25,
        })
        result = await g.grade(_ctx([_tc() for _ in range(50)]))
        assert result.score == 0.0


# ===================================================================
# Logarithmic scoring
# ===================================================================


class TestLogarithmicScoring:
    @pytest.mark.asyncio
    async def test_at_ideal(self):
        g = TurnCountGrader({
            "scoring": "logarithmic", "min_turns": 10, "max_turns": 100,
        })
        result = await g.grade(_ctx([_tc() for _ in range(10)]))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_below_ideal(self):
        g = TurnCountGrader({
            "scoring": "logarithmic", "min_turns": 10, "max_turns": 100,
        })
        result = await g.grade(_ctx([_tc() for _ in range(5)]))
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_at_max(self):
        g = TurnCountGrader({
            "scoring": "logarithmic", "min_turns": 10, "max_turns": 100,
        })
        result = await g.grade(_ctx([_tc() for _ in range(100)]))
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_harsh_on_extreme_overrun(self):
        """Log scoring should be harsher than linear for large overruns."""
        g_log = TurnCountGrader({
            "scoring": "logarithmic", "min_turns": 10, "max_turns": 100,
        })
        g_lin = TurnCountGrader({
            "scoring": "linear", "min_turns": 10, "max_turns": 100,
        })
        # At 80 turns (close to max), log should score lower than linear
        ctx = _ctx([_tc() for _ in range(80)])
        r_log = await g_log.grade(ctx)
        r_lin = await g_lin.grade(ctx)
        assert r_log.score < r_lin.score

    @pytest.mark.asyncio
    async def test_above_max(self):
        g = TurnCountGrader({
            "scoring": "logarithmic", "min_turns": 10, "max_turns": 100,
        })
        result = await g.grade(_ctx([_tc() for _ in range(200)]))
        assert result.score == 0.0


# ===================================================================
# Count filters
# ===================================================================


class TestCountFilter:
    @pytest.mark.asyncio
    async def test_filter_all(self):
        g = TurnCountGrader({"max_turns": 10, "count_filter": "all"})
        calls = [_tc(), _tc(error={"msg": "fail"}), _tc(tool_type="llm")]
        result = await g.grade(_ctx(calls))
        assert result.details["turn_count"] == 3

    @pytest.mark.asyncio
    async def test_filter_llm_only(self):
        g = TurnCountGrader({"max_turns": 10, "count_filter": "llm"})
        calls = [
            _tc(tool_type="llm"),
            _tc(tool_type="llm"),
            _tc(tool_type="code"),
            _tc(),  # no tool_type
        ]
        result = await g.grade(_ctx(calls))
        assert result.details["turn_count"] == 2

    @pytest.mark.asyncio
    async def test_filter_non_error(self):
        g = TurnCountGrader({"max_turns": 10, "count_filter": "non-error"})
        calls = [
            _tc(),
            _tc(error={"msg": "timeout"}),
            _tc(),
            _tc(error={"msg": "fail"}),
        ]
        result = await g.grade(_ctx(calls))
        assert result.details["turn_count"] == 2


# ===================================================================
# Expected turns and efficiency ratio
# ===================================================================


class TestExpectedTurns:
    @pytest.mark.asyncio
    async def test_efficiency_ratio(self):
        g = TurnCountGrader({"max_turns": 50, "expected_turns": 10})
        result = await g.grade(_ctx([_tc() for _ in range(20)]))
        assert result.details["expected_turns"] == 10
        assert result.details["efficiency_ratio"] == 2.0

    @pytest.mark.asyncio
    async def test_efficiency_ratio_exact(self):
        g = TurnCountGrader({"max_turns": 50, "expected_turns": 15})
        result = await g.grade(_ctx([_tc() for _ in range(15)]))
        assert result.details["efficiency_ratio"] == 1.0


# ===================================================================
# Pass threshold
# ===================================================================


class TestPassThreshold:
    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        g = TurnCountGrader({
            "max_turns": 10, "pass_threshold": 0.5,
        })
        # 15 turns → score = 10/15 = 0.667 >= 0.5 → pass
        result = await g.grade(_ctx([_tc() for _ in range(15)]))
        assert result.passed

    @pytest.mark.asyncio
    async def test_strict_threshold(self):
        g = TurnCountGrader({
            "max_turns": 10, "pass_threshold": 1.0,
        })
        # 10 turns → score = 1.0 → pass
        result = await g.grade(_ctx([_tc() for _ in range(10)]))
        assert result.passed
        # 11 turns → score = 10/11 < 1.0 → fail
        result = await g.grade(_ctx([_tc() for _ in range(11)]))
        assert not result.passed


# ===================================================================
# Reasoning output
# ===================================================================


class TestReasoning:
    @pytest.mark.asyncio
    async def test_reasoning_contains_info(self):
        g = TurnCountGrader({"max_turns": 20, "expected_turns": 10})
        result = await g.grade(_ctx([_tc() for _ in range(15)]))
        assert "15" in result.reasoning
        assert "expected 10" in result.reasoning
        assert "budget" in result.reasoning

    @pytest.mark.asyncio
    async def test_reasoning_scoring_mode(self):
        g = TurnCountGrader({"scoring": "linear", "max_turns": 50})
        result = await g.grade(_ctx([_tc() for _ in range(10)]))
        assert "linear" in result.reasoning


# ===================================================================
# YAML integration test
# ===================================================================


class TestYAMLIntegration:
    """Test that grader works as specified in YAML config style."""

    @pytest.mark.asyncio
    async def test_stripe_style_config(self):
        """Simulate a Stripe-like config: expected ~30 turns, max 100."""
        g = TurnCountGrader({
            "scoring": "logarithmic",
            "min_turns": 17,
            "max_turns": 216,
            "expected_turns": 50,
            "pass_threshold": 0.5,
        })
        # Efficient agent: 25 turns (between min=17 and max=216)
        r1 = await g.grade(_ctx([_tc() for _ in range(25)]))
        assert r1.passed
        assert r1.score > 0.8

        # Moderate agent: 50 turns (at expected)
        r2 = await g.grade(_ctx([_tc() for _ in range(50)]))
        assert r2.passed  # should pass with 0.5 threshold
        assert r2.score > 0.5

        # Slow agent: 200 turns
        r3 = await g.grade(_ctx([_tc() for _ in range(200)]))
        assert r3.score < 0.1
