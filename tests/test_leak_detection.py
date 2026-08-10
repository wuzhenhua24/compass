"""Tests for leak detection — GradeContext-level and standalone grader."""

from __future__ import annotations

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders.base import (
    GradeContext,
    GraderScope,
    LeakCheckResult,
    generate_leak_marker,
)
from compass.graders.registry import get_grader

# ===================================================================
# generate_leak_marker helper
# ===================================================================


class TestGenerateLeakMarker:
    def test_default_prefix(self):
        marker = generate_leak_marker()
        assert marker.startswith("COMPASS_LEAK_")
        assert len(marker) == len("COMPASS_LEAK_") + 12

    def test_custom_prefix(self):
        marker = generate_leak_marker(prefix="EVAL_UUID")
        assert marker.startswith("EVAL_UUID_")

    def test_unique_each_call(self):
        markers = {generate_leak_marker() for _ in range(100)}
        assert len(markers) == 100


# ===================================================================
# LeakCheckResult
# ===================================================================


class TestLeakCheckResult:
    def test_no_leaks(self):
        r = LeakCheckResult(leaked=[], searched_patterns=["A", "B"])
        assert not r.has_leaks
        d = r.to_dict()
        assert d["has_leaks"] is False
        assert d["total_patterns_checked"] == 2

    def test_has_leaks(self):
        r = LeakCheckResult(leaked=["A"], searched_patterns=["A", "B"])
        assert r.has_leaks
        d = r.to_dict()
        assert d["has_leaks"] is True
        assert d["leaked_markers"] == ["A"]


# ===================================================================
# GradeContext.check_leaks
# ===================================================================


def _make_context(
    markers: list[str] | None = None,
    tool_outputs: list[str] | None = None,
    tool_inputs: list[dict] | None = None,
    reasoning: list[str] | None = None,
) -> GradeContext:
    """Build a GradeContext with minimal transcript data."""
    tool_calls = []
    if tool_outputs:
        for out in tool_outputs:
            tool_calls.append(ToolCall(tool_name="read_file", output=out))
    if tool_inputs:
        for inp in tool_inputs:
            tool_calls.append(ToolCall(tool_name="write_file", input=inp))

    transcript = Transcript(
        task_id="test",
        trial_id="t1",
        tool_calls=tool_calls,
        reasoning_steps=reasoning or [],
    )
    return GradeContext(
        transcript=transcript,
        outcome=Outcome(),
        leak_markers=markers or [],
    )


class TestGradeContextCheckLeaks:
    def test_no_markers_no_leaks(self):
        ctx = _make_context(markers=[], tool_outputs=["some text"])
        result = ctx.check_leaks()
        assert not result.has_leaks

    def test_marker_in_tool_output(self):
        ctx = _make_context(
            markers=["SECRET_UUID_123"],
            tool_outputs=["file content with SECRET_UUID_123 here"],
        )
        result = ctx.check_leaks()
        assert result.has_leaks
        assert "SECRET_UUID_123" in result.leaked

    def test_marker_in_tool_input(self):
        ctx = _make_context(
            markers=["LEAK_MARKER"],
            tool_inputs=[{"path": "/grader/LEAK_MARKER.sh"}],
        )
        result = ctx.check_leaks()
        assert result.has_leaks

    def test_marker_in_reasoning(self):
        ctx = _make_context(
            markers=["UUID_IN_REASONING"],
            reasoning=["I found UUID_IN_REASONING in the solution file"],
        )
        result = ctx.check_leaks()
        assert result.has_leaks

    def test_no_leak_clean_transcript(self):
        ctx = _make_context(
            markers=["SECRET_UUID_123"],
            tool_outputs=["clean output with no markers"],
            reasoning=["normal reasoning"],
        )
        result = ctx.check_leaks()
        assert not result.has_leaks

    def test_extra_patterns_merged(self):
        ctx = _make_context(
            markers=["MARKER_A"],
            tool_outputs=["contains EXTRA_PATTERN here"],
        )
        result = ctx.check_leaks(extra_patterns=["EXTRA_PATTERN"])
        assert result.has_leaks
        assert "EXTRA_PATTERN" in result.leaked
        assert "MARKER_A" not in result.leaked

    def test_multiple_leaks(self):
        ctx = _make_context(
            markers=["UUID_1", "UUID_2", "UUID_3"],
            tool_outputs=["UUID_1 and UUID_3 found here"],
        )
        result = ctx.check_leaks()
        assert len(result.leaked) == 2
        assert "UUID_1" in result.leaked
        assert "UUID_3" in result.leaked
        assert "UUID_2" not in result.leaked

    def test_no_transcript_no_crash(self):
        ctx = GradeContext(
            outcome=Outcome(),
            leak_markers=["MARKER"],
        )
        result = ctx.check_leaks()
        assert not result.has_leaks

    def test_empty_markers_with_extra(self):
        ctx = _make_context(
            markers=[],
            tool_outputs=["EXTRA here"],
        )
        result = ctx.check_leaks(extra_patterns=["EXTRA"])
        assert result.has_leaks


# ===================================================================
# LeakDetectionGrader — registration and basic grading
# ===================================================================


class TestLeakDetectionGraderRegistry:
    def test_registered(self):
        cls = get_grader("leak_detection")
        assert cls is not None
        assert cls.name == "leak_detection"

    def test_scope_is_transcript(self):
        cls = get_grader("leak_detection")
        assert cls.grader_scope == GraderScope.TRANSCRIPT


class TestLeakDetectionGraderGrade:
    @pytest.mark.asyncio
    async def test_clean_transcript_passes(self):
        cls = get_grader("leak_detection")
        grader = cls({})
        ctx = _make_context(
            markers=["SECRET_UUID"],
            tool_outputs=["clean output"],
        )
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_leaked_marker_fails(self):
        cls = get_grader("leak_detection")
        grader = cls({})
        ctx = _make_context(
            markers=["SECRET_UUID_ABC"],
            tool_outputs=["I read SECRET_UUID_ABC from grader file"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score == 0.0
        assert "answer_leak" in result.failure_tags

    @pytest.mark.asyncio
    async def test_grader_config_patterns(self):
        """Patterns from grader config are merged with context markers."""
        cls = get_grader("leak_detection")
        grader = cls({"leak_patterns": ["GRADER_LEVEL_UUID"]})
        ctx = _make_context(
            markers=[],
            tool_outputs=["output has GRADER_LEVEL_UUID"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert "answer_leak" in result.failure_tags

    @pytest.mark.asyncio
    async def test_both_sources_merged(self):
        """Context markers + config patterns are both checked."""
        cls = get_grader("leak_detection")
        grader = cls({"leak_patterns": ["CONFIG_UUID"]})
        ctx = _make_context(
            markers=["CONTEXT_UUID"],
            tool_outputs=["text with CONFIG_UUID and CONTEXT_UUID"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert len(result.details["leaked_markers"]) == 2

    @pytest.mark.asyncio
    async def test_no_transcript_passes(self):
        cls = get_grader("leak_detection")
        grader = cls({"leak_patterns": ["UUID"]})
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalidate_false_proportional_score(self):
        """When invalidate=False, score is reduced proportionally."""
        cls = get_grader("leak_detection")
        grader = cls({
            "leak_patterns": ["A", "B", "C", "D"],
            "invalidate": False,
        })
        ctx = _make_context(
            markers=[],
            tool_outputs=["contains A only"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        # 1 of 4 leaked → score = 1 - 1/4 = 0.75
        assert result.score == pytest.approx(0.75)

    @pytest.mark.asyncio
    async def test_invalidate_true_zero_score(self):
        """Default invalidate=True gives score 0 on any leak."""
        cls = get_grader("leak_detection")
        grader = cls({"leak_patterns": ["A", "B", "C", "D"]})
        ctx = _make_context(markers=[], tool_outputs=["contains A only"])
        result = await grader.grade(ctx)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_reasoning_contains_details(self):
        cls = get_grader("leak_detection")
        grader = cls({})
        ctx = _make_context(
            markers=["LEAK_1", "LEAK_2"],
            tool_outputs=["LEAK_1 found"],
        )
        result = await grader.grade(ctx)
        assert "LEAK_1" in result.reasoning
        assert "1 of 2" in result.reasoning


# ===================================================================
# Scenario / TestCase leak_markers
# ===================================================================


class TestScenarioLeakMarkers:
    def test_case_level_markers(self):
        from compass.core.scenario import (
            AgentConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        case = TestCase(
            id="t1",
            input=InputConfig(prompt="test"),
            leak_markers=["CASE_UUID"],
        )
        scenario = Scenario(
            name="test",
            agent=AgentConfig(adapter="coding"),
            cases=[case],
        )
        markers = scenario.get_leak_markers_for_case(case)
        assert markers == ["CASE_UUID"]

    def test_scenario_level_inherited(self):
        from compass.core.scenario import (
            AgentConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        case = TestCase(id="t1", input=InputConfig(prompt="test"))
        scenario = Scenario(
            name="test",
            agent=AgentConfig(adapter="coding"),
            leak_markers=["SCENARIO_UUID"],
            cases=[case],
        )
        markers = scenario.get_leak_markers_for_case(case)
        assert markers == ["SCENARIO_UUID"]

    def test_merged_deduped(self):
        from compass.core.scenario import (
            AgentConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        case = TestCase(
            id="t1",
            input=InputConfig(prompt="test"),
            leak_markers=["SHARED", "CASE_ONLY"],
        )
        scenario = Scenario(
            name="test",
            agent=AgentConfig(adapter="coding"),
            leak_markers=["SHARED", "SCENARIO_ONLY"],
            cases=[case],
        )
        markers = scenario.get_leak_markers_for_case(case)
        assert "SHARED" in markers
        assert "CASE_ONLY" in markers
        assert "SCENARIO_ONLY" in markers
        # No duplicates
        assert markers.count("SHARED") == 1

    def test_yaml_round_trip(self):
        import yaml

        yaml_str = """
name: leak_test
agent:
  adapter: coding
leak_markers:
  - "SCENARIO_LEAK_UUID"
cases:
  - id: t1
    input:
      prompt: "test"
    leak_markers:
      - "CASE_LEAK_UUID"
"""
        from compass.core.scenario import Scenario

        data = yaml.safe_load(yaml_str)
        scenario = Scenario.model_validate(data)
        assert scenario.leak_markers == ["SCENARIO_LEAK_UUID"]
        assert scenario.cases[0].leak_markers == ["CASE_LEAK_UUID"]
        markers = scenario.get_leak_markers_for_case(scenario.cases[0])
        assert len(markers) == 2


# ===================================================================
# IntegrationGrader — refactored leak detection via GradeContext
# ===================================================================


class TestIntegrationGraderLeakRefactor:
    @pytest.mark.asyncio
    async def test_context_markers_detected(self):
        """IntegrationGrader picks up leak_markers from GradeContext."""
        from compass.graders.code.coding.integration import IntegrationGrader

        grader = IntegrationGrader({"script": "echo ok"})
        ctx = _make_context(
            markers=["CONTEXT_LEVEL_UUID"],
            tool_outputs=["output has CONTEXT_LEVEL_UUID"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert "answer_leak" in result.failure_tags

    @pytest.mark.asyncio
    async def test_config_patterns_still_work(self):
        """IntegrationGrader's own leak_patterns still detected."""
        from compass.graders.code.coding.integration import IntegrationGrader

        grader = IntegrationGrader({
            "script": "echo ok",
            "leak_patterns": ["CONFIG_UUID"],
        })
        ctx = _make_context(
            markers=[],
            tool_outputs=["CONFIG_UUID found"],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert "answer_leak" in result.failure_tags

    @pytest.mark.asyncio
    async def test_clean_runs_script(self):
        """No leak → proceeds to run the script."""
        from compass.graders.code.coding.integration import IntegrationGrader

        grader = IntegrationGrader({"script": "echo '1 passed'"})
        ctx = _make_context(
            markers=["SECRET"],
            tool_outputs=["clean output"],
        )
        # Should proceed past leak check (may fail on script execution,
        # but should NOT fail on leak detection)
        result = await grader.grade(ctx)
        assert "answer_leak" not in result.failure_tags
