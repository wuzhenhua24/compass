"""Tests for Code-First short-circuit execution mode."""

from __future__ import annotations

import pytest

from compass.core.result import EvaluatorResult
from compass.core.runner import Compass
from compass.core.scenario import (
    AggregationConfig,
    GraderConfig,
    ShortCircuitMode,
)
from compass.core.scenario import (
    GraderType as ScenarioGraderType,
)
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
    ModelGrader,
)
from compass.graders.registry import register_grader

# ===================================================================
# Test Graders (track execution order)
# ===================================================================

# Track execution order globally
execution_log: list[str] = []


def reset_execution_log():
    """Reset the execution log before each test."""
    global execution_log
    execution_log = []


@register_grader("_test_code_pass")
class TestCodePassGrader(CodeGrader):
    """A Code Grader that always passes."""

    name = "_test_code_pass"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        execution_log.append("code_pass")
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=True,
            score=1.0,
        )


@register_grader("_test_code_fail")
class TestCodeFailGrader(CodeGrader):
    """A Code Grader that always fails."""

    name = "_test_code_fail"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        execution_log.append("code_fail")
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
        )


@register_grader("_test_model_pass")
class TestModelPassGrader(ModelGrader):
    """A Model Grader that always passes."""

    name = "_test_model_pass"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        execution_log.append("model_pass")
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=True,
            score=1.0,
        )


@register_grader("_test_model_fail")
class TestModelFailGrader(ModelGrader):
    """A Model Grader that always fails."""

    name = "_test_model_fail"
    grader_type = GraderType.MODEL
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        execution_log.append("model_fail")
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
        )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture(autouse=True)
def reset_log():
    """Reset execution log before each test."""
    reset_execution_log()
    yield


@pytest.fixture
def compass():
    """Create a Compass runner instance."""
    return Compass()


# ===================================================================
# Helper functions
# ===================================================================


def make_grader_config(
    name: str, grader_type: ScenarioGraderType, required: bool = False
) -> GraderConfig:
    """Create a GraderConfig for testing."""
    return GraderConfig(
        type=grader_type,
        name=name,
        required=required,
    )


async def run_graders_with_mode(
    compass: Compass,
    grader_configs: list[GraderConfig],
    short_circuit: ShortCircuitMode,
    required_graders: list[str] | None = None,
) -> list[EvaluatorResult]:
    """Run graders with specified short-circuit mode."""
    from compass.core.transcript import Outcome, Transcript

    context = GradeContext(
        prompt="test",
        transcript=Transcript(task_id="test", trial_id="test"),
        outcome=Outcome(output_data="test output"),
    )

    aggregation = AggregationConfig(
        short_circuit=short_circuit,
        required_graders=required_graders or [],
    )

    return await compass._run_graders(grader_configs, context, aggregation)


# ===================================================================
# Tests: Short-circuit disabled (default behavior)
# ===================================================================


class TestShortCircuitDisabled:
    """Tests for disabled short-circuit mode (default behavior)."""

    @pytest.mark.asyncio
    async def test_all_graders_run_sequentially(self, compass):
        """All graders run in config order when short-circuit is disabled."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.DISABLED
        )

        # All graders should run in order
        assert execution_log == ["code_pass", "model_pass", "code_fail"]
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_failing_code_grader_does_not_skip_model(self, compass):
        """Failing Code Grader does not skip Model Graders when disabled."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.DISABLED
        )

        # Both graders should run
        assert execution_log == ["code_fail", "model_pass"]
        assert len(results) == 2
        assert not any(r.metadata.get("skipped") for r in results)


# ===================================================================
# Tests: Short-circuit on CODE_FAIL
# ===================================================================


class TestShortCircuitCodeFail:
    """Tests for CODE_FAIL short-circuit mode."""

    @pytest.mark.asyncio
    async def test_code_pass_runs_model_graders(self, compass):
        """When all Code Graders pass, Model Graders run."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # Both should run
        assert execution_log == ["code_pass", "model_pass"]
        assert len(results) == 2
        assert not any(r.metadata.get("skipped") for r in results)

    @pytest.mark.asyncio
    async def test_code_fail_skips_model_graders(self, compass):
        """When any Code Grader fails, Model Graders are skipped."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # Only Code graders should run
        assert execution_log == ["code_pass", "code_fail"]
        assert len(results) == 3

        # Model grader should be marked as skipped
        model_result = next(r for r in results if r.name == "_test_model_pass")
        assert model_result.skipped is True
        assert model_result.skip_reason == "short_circuit"
        assert model_result.metadata.get("skipped") is True
        assert model_result.metadata.get("reason") == "short_circuit"
        assert not model_result.passed
        # It never ran, so it measured nothing — unscored, not zero
        assert model_result.score is None
        assert model_result.scored is False

    @pytest.mark.asyncio
    async def test_multiple_model_graders_all_skipped(self, compass):
        """Multiple Model Graders are all skipped when Code fails."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_model_fail", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # Only Code grader runs
        assert execution_log == ["code_fail"]

        # Both Model graders should be skipped
        model_results = [r for r in results if r.grader_type == "model"]
        assert len(model_results) == 2
        assert all(r.metadata.get("skipped") for r in model_results)

    @pytest.mark.asyncio
    async def test_code_graders_run_in_order(self, compass):
        """Code Graders run in their config order before Model Graders."""
        configs = [
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # Code graders should run first regardless of config order
        assert execution_log == ["code_pass", "code_fail"]

        # Model grader skipped
        model_result = next(r for r in results if r.name == "_test_model_pass")
        assert model_result.metadata.get("skipped") is True


# ===================================================================
# Tests: Short-circuit on CODE_PASS (Model runs only on Code failure)
# ===================================================================


class TestShortCircuitCodePass:
    """Tests for CODE_PASS short-circuit mode.

    This mode is useful when:
    - Code rules are sufficient to confirm quality (skip expensive Model)
    - Code failure indicates edge cases needing Model's intelligent judgment
    """

    @pytest.mark.asyncio
    async def test_code_pass_skips_model_graders(self, compass):
        """When all Code Graders pass, Model Graders are skipped."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_PASS
        )

        # Only Code grader should run
        assert execution_log == ["code_pass"]
        assert len(results) == 2

        # Model grader should be marked as skipped
        model_result = next(r for r in results if r.name == "_test_model_pass")
        assert model_result.metadata.get("skipped") is True
        assert model_result.metadata.get("reason") == "short_circuit"

    @pytest.mark.asyncio
    async def test_code_fail_runs_model_graders(self, compass):
        """When any Code Grader fails, Model Graders run for intelligent judgment."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_PASS
        )

        # Both should run - Code failed, need Model evaluation
        assert execution_log == ["code_fail", "model_pass"]
        assert len(results) == 2
        assert not any(r.metadata.get("skipped") for r in results)

    @pytest.mark.asyncio
    async def test_mixed_code_results_runs_model(self, compass):
        """When any Code Grader fails (mixed results), Model Graders run."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_PASS
        )

        # All graders should run - not all Code passed
        assert execution_log == ["code_pass", "code_fail", "model_pass"]
        assert len(results) == 3
        assert not any(r.metadata.get("skipped") for r in results)

    @pytest.mark.asyncio
    async def test_multiple_code_pass_skips_model(self, compass):
        """Multiple passing Code Graders skip Model Graders."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_model_fail", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_PASS
        )

        # Only Code graders run
        assert execution_log == ["code_pass", "code_pass"]

        # Both Model graders skipped
        model_results = [r for r in results if r.grader_type == "model"]
        assert len(model_results) == 2
        assert all(r.metadata.get("skipped") for r in model_results)

    @pytest.mark.asyncio
    async def test_no_code_graders_runs_model(self, compass):
        """When no Code Graders exist, Model Graders run."""
        configs = [
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_PASS
        )

        # Model grader should run (no Code graders to trigger short-circuit)
        assert execution_log == ["model_pass"]
        assert len(results) == 1
        assert not results[0].metadata.get("skipped")

    @pytest.mark.asyncio
    async def test_code_pass_preserves_order(self, compass):
        """Code Graders run first, then Model if Code fails."""
        configs = [
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_fail", ScenarioGraderType.MODEL),
        ]

        await run_graders_with_mode(compass, configs, ShortCircuitMode.CODE_PASS)

        # Code runs first, then Model (because Code failed)
        assert execution_log == ["code_fail", "model_pass", "model_fail"]


# ===================================================================
# Tests: Short-circuit on REQUIRED_FAIL
# ===================================================================


class TestShortCircuitRequiredFail:
    """Tests for REQUIRED_FAIL short-circuit mode."""

    @pytest.mark.asyncio
    async def test_non_required_fail_runs_model_graders(self, compass):
        """When non-required Code Grader fails, Model Graders still run."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE, required=False),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.REQUIRED_FAIL
        )

        # Both should run
        assert execution_log == ["code_fail", "model_pass"]
        assert len(results) == 2
        assert not any(r.metadata.get("skipped") for r in results)

    @pytest.mark.asyncio
    async def test_required_fail_skips_model_graders(self, compass):
        """When required Code Grader fails, Model Graders are skipped."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE, required=True),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.REQUIRED_FAIL
        )

        # Only Code grader runs
        assert execution_log == ["code_fail"]

        # Model grader skipped
        model_result = next(r for r in results if r.name == "_test_model_pass")
        assert model_result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_required_by_name_fail_skips_model_graders(self, compass):
        """Required grader by name (in aggregation config) triggers short-circuit."""
        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE, required=False),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass,
            configs,
            ShortCircuitMode.REQUIRED_FAIL,
            required_graders=["_test_code_fail"],  # Required by name
        )

        # Only Code grader runs
        assert execution_log == ["code_fail"]

        # Model grader skipped
        model_result = next(r for r in results if r.name == "_test_model_pass")
        assert model_result.metadata.get("skipped") is True

    @pytest.mark.asyncio
    async def test_required_pass_runs_model_graders(self, compass):
        """When required Code Grader passes, Model Graders run."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE, required=True),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.REQUIRED_FAIL
        )

        # Both should run
        assert execution_log == ["code_pass", "model_pass"]
        assert len(results) == 2
        assert not any(r.metadata.get("skipped") for r in results)

    @pytest.mark.asyncio
    async def test_mix_required_and_non_required(self, compass):
        """Mix of required and non-required graders."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE, required=True),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE, required=False),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.REQUIRED_FAIL
        )

        # Required passed, non-required failed, but still runs Model
        assert execution_log == ["code_pass", "code_fail", "model_pass"]
        assert len(results) == 3


# ===================================================================
# Tests: Edge cases
# ===================================================================


class TestShortCircuitEdgeCases:
    """Edge case tests for short-circuit mode."""

    @pytest.mark.asyncio
    async def test_no_code_graders(self, compass):
        """When there are no Code Graders, Model Graders run normally."""
        configs = [
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
            make_grader_config("_test_model_fail", ScenarioGraderType.MODEL),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # All Model graders should run (no Code grader to trigger short-circuit)
        assert execution_log == ["model_pass", "model_fail"]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_no_model_graders(self, compass):
        """When there are no Model Graders, Code Graders run normally."""
        configs = [
            make_grader_config("_test_code_pass", ScenarioGraderType.CODE),
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
        ]

        results = await run_graders_with_mode(
            compass, configs, ShortCircuitMode.CODE_FAIL
        )

        # All Code graders should run
        assert execution_log == ["code_pass", "code_fail"]
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_grader_list(self, compass):
        """Empty grader list should return empty results."""
        results = await run_graders_with_mode(
            compass, [], ShortCircuitMode.CODE_FAIL
        )

        assert execution_log == []
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_aggregation_none_uses_disabled(self, compass):
        """When aggregation is None, behaves as disabled."""
        from compass.core.transcript import Outcome, Transcript

        configs = [
            make_grader_config("_test_code_fail", ScenarioGraderType.CODE),
            make_grader_config("_test_model_pass", ScenarioGraderType.MODEL),
        ]

        context = GradeContext(
            prompt="test",
            transcript=Transcript(task_id="test", trial_id="test"),
            outcome=Outcome(output_data="test output"),
        )

        # Pass None for aggregation
        results = await compass._run_graders(configs, context, None)

        # Both should run (disabled mode)
        assert execution_log == ["code_fail", "model_pass"]
        assert len(results) == 2


# ===================================================================
# Tests: YAML configuration
# ===================================================================


class TestShortCircuitYAMLConfig:
    """Tests for short-circuit configuration via YAML."""

    def test_parse_short_circuit_mode_from_dict(self):
        """ShortCircuitMode can be parsed from string in config."""
        config = AggregationConfig(
            short_circuit=ShortCircuitMode.REQUIRED_FAIL,
        )
        assert config.short_circuit == ShortCircuitMode.REQUIRED_FAIL

    def test_parse_short_circuit_mode_from_string(self):
        """ShortCircuitMode can be parsed from string value."""
        config = AggregationConfig.model_validate({
            "short_circuit": "code_fail",
        })
        assert config.short_circuit == ShortCircuitMode.CODE_FAIL

    def test_default_short_circuit_is_disabled(self):
        """Default short_circuit mode is disabled."""
        config = AggregationConfig()
        assert config.short_circuit == ShortCircuitMode.DISABLED
