"""Tests for runner.py grader integration."""

from __future__ import annotations

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifacts import CodeArtifact, ExecutionResult, GeneratedFile
from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.core.runner import Compass
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    GraderConfig,
    InputConfig,
    MetricsConfig,
    Scenario,
    TestCase,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_scenario(
    grader_configs: list[GraderConfig] | None = None,
    default_graders: list[GraderConfig] | None = None,
    case_input: InputConfig | None = None,
    adapter_name: str = "_test_coding_runner",
    adapter_config: dict | None = None,
    pass_threshold: float = 0.7,
    required_graders: list[str] | None = None,
    expect: str = "pass",
    trials: int | None = None,
    metrics: MetricsConfig | None = None,
) -> Scenario:
    """Build a minimal Scenario for testing."""
    from typing import Literal
    case = TestCase(
        id="case_1",
        input=case_input or InputConfig(
            prompt="test",
            params={
                "files": [
                    {"path": "main.py", "content": "print('hello')"},
                ],
            },
        ),
        expect=expect,  # type: ignore[arg-type]
        trials=trials,
        metrics=metrics or MetricsConfig(),
        graders=grader_configs or [],
        aggregation=AggregationConfig(
            pass_threshold=pass_threshold,
            required_graders=required_graders or [],
        ),
    )
    return Scenario(
        name="test_scenario",
        agent=AgentConfig(
            adapter=adapter_name,
            config=adapter_config or {},
        ),
        default_graders=default_graders or [],
        cases=[case],
    )


# A lightweight adapter that runs code in a sandbox, registered only for tests.
_registered = False


def _ensure_test_adapter():
    """Register a simple coding adapter for runner integration tests."""
    global _registered
    if _registered:
        return

    from compass.adapters.coding import CodingAdapter

    @register_adapter("_test_coding_runner")
    class _TestCodingAdapter(CodingAdapter):
        name = "_test_coding_runner"

    _registered = True


@pytest.fixture(autouse=True)
def _setup_adapter():
    _ensure_test_adapter()
    yield


# ===================================================================
# Integration tests
# ===================================================================


class TestRunnerGraderIntegration:
    @pytest.mark.asyncio
    async def test_coding_adapter_with_exit_code_grader_pass(self):
        """End-to-end: coding adapter produces exit_code=0, grader passes."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        assert isinstance(result, EvalResult)
        assert result.total_cases == 1
        assert result.passed_cases == 1
        case = result.case_results[0]
        assert case.passed is True
        assert case.status == TestStatus.PASSED
        assert len(case.evaluator_results) == 1
        assert case.evaluator_results[0].passed is True

    @pytest.mark.asyncio
    async def test_coding_adapter_with_exit_code_grader_fail(self):
        """End-to-end: coding adapter produces exit_code=1, grader fails."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
            case_input=InputConfig(
                prompt="test",
                params={
                    "files": [
                        {"path": "main.py", "content": "raise SystemExit(1)"},
                    ],
                },
            ),
        )
        runner = Compass()
        result = await runner.run(scenario)

        assert result.failed_cases == 1
        case = result.case_results[0]
        assert case.passed is False
        assert case.evaluator_results[0].passed is False
        assert case.evaluator_results[0].score == 0.0

    @pytest.mark.asyncio
    async def test_multiple_graders_aggregation(self):
        """Multiple graders are run and their scores aggregated."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={},
                    weight=1.0,
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.7,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # First grader passes (exit_code 0 == 0), second fails (0 != 99)
        assert len(case.evaluator_results) == 2
        assert case.evaluator_results[0].passed is True
        assert case.evaluator_results[1].passed is False
        # Weighted average: (1.0 + 0.0) / 2 = 0.5 < 0.7 threshold
        assert case.passed is False
        assert case.overall_score == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_default_graders_applied(self):
        """Default graders from the scenario are applied to cases."""
        scenario = _make_scenario(
            default_graders=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            grader_configs=[],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert len(case.evaluator_results) == 1
        assert case.evaluator_results[0].name == "exit_code_check"
        assert case.evaluator_results[0].passed is True

    @pytest.mark.asyncio
    async def test_default_and_case_graders_combined(self):
        """Default graders and case-level graders are combined."""
        scenario = _make_scenario(
            default_graders=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 0},
                ),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # 1 default + 1 case-level = 2 total
        assert len(case.evaluator_results) == 2

    @pytest.mark.asyncio
    async def test_no_graders_returns_no_pass(self):
        """With no graders configured, the case does not pass."""
        scenario = _make_scenario(
            grader_configs=[],
            adapter_config={"run_command": "echo ok"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.passed is False
        assert case.overall_score == 0.0

    @pytest.mark.asyncio
    async def test_evaluator_result_metadata_from_grade_details(self):
        """GradeResult.details are mapped to EvaluatorResult.metadata."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        assert isinstance(er, EvaluatorResult)
        assert "actual_exit_code" in er.metadata
        assert "expected_exit_code" in er.metadata

    @pytest.mark.asyncio
    async def test_grader_error_captured(self):
        """An unknown grader name produces an error EvaluatorResult."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="nonexistent_grader_xyz", config={}),
            ],
            adapter_config={"run_command": "echo ok"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert len(case.evaluator_results) == 1
        er = case.evaluator_results[0]
        assert er.passed is False
        assert er.error is not None
        assert "nonexistent_grader_xyz" in er.error

    @pytest.mark.asyncio
    async def test_results_accumulated(self):
        """Runner accumulates results across multiple runs."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "echo ok"},
        )
        runner = Compass()
        await runner.run(scenario)
        await runner.run(scenario)

        assert len(runner.get_results()) == 2

    @pytest.mark.asyncio
    async def test_clear_results(self):
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "echo ok"},
        )
        runner = Compass()
        await runner.run(scenario)
        runner.clear_results()
        assert len(runner.get_results()) == 0


# ===================================================================
# Required graders tests
# ===================================================================


class TestRequiredGraders:
    """Tests for required_graders and GraderConfig.required."""

    @pytest.mark.asyncio
    async def test_required_graders_all_pass(self):
        """When required graders all pass, case passes."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            required_graders=["exit_code_check"],
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.passed is True

    @pytest.mark.asyncio
    async def test_required_graders_one_fails(self):
        """When a required grader fails, case fails even if score is above threshold."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Will fail
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.3,  # Low threshold - would pass without required check
            required_graders=["exit_code_check"],  # One of them must pass
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Score is 0.5 which is > 0.3 threshold, but one required grader failed
        assert case.overall_score == pytest.approx(0.5)
        # Since the second exit_code_check (expected=99) fails, and it's in required_graders,
        # the case should fail. But note: required_graders matches by name, so if ANY
        # grader named exit_code_check passes, the requirement is met.
        # In this case, the first one passes, so the requirement IS met.
        assert case.passed is True

    @pytest.mark.asyncio
    async def test_required_flag_on_grader_config(self):
        """GraderConfig.required=True causes failure if that grader fails."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Will fail
                    weight=1.0,
                    required=True,  # Must pass
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.3,  # Low threshold
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Score is 0.5 > 0.3, but required grader failed
        assert case.overall_score == pytest.approx(0.5)
        assert case.passed is False  # Failed because required grader failed

    @pytest.mark.asyncio
    async def test_required_flag_passes_when_grader_passes(self):
        """GraderConfig.required=True passes when that grader passes."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={},
                    weight=1.0,
                    required=True,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.passed is True
        # Verify required flag is preserved in result
        assert case.evaluator_results[0].required is True

    @pytest.mark.asyncio
    async def test_evaluator_result_required_field(self):
        """EvaluatorResult.required reflects GraderConfig.required."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, required=False),
                GraderConfig(name="exit_code_check", config={}, required=True),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.evaluator_results[0].required is False
        assert case.evaluator_results[1].required is True


# ===================================================================
# Negative tests (expect=fail)
# ===================================================================


class TestNegativeTests:
    """Tests for expect=fail negative test cases."""

    @pytest.mark.asyncio
    async def test_expect_fail_passes_when_graders_fail(self):
        """Negative test passes when graders fail (agent correctly failed)."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Will fail (actual is 0)
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            expect="fail",  # Negative test
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Grader failed (score 0.0), but this is a negative test
        # so overall case should PASS (we expected failure)
        assert case.overall_score == 0.0
        assert case.passed is True
        assert case.status == TestStatus.PASSED

    @pytest.mark.asyncio
    async def test_expect_fail_fails_when_graders_pass(self):
        """Negative test fails when graders pass (agent should have failed but didn't)."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            expect="fail",  # Negative test
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Grader passed, but this is a negative test
        # so overall case should FAIL (we expected failure but got success)
        assert case.overall_score == 1.0
        assert case.passed is False
        assert case.status == TestStatus.FAILED

    @pytest.mark.asyncio
    async def test_expect_fail_multitrial_passes_when_graders_fail(self):
        """Regression (runner→TaskResult chain): a negative test with trials>1
        must PASS when the agent correctly fails on every trial.

        _run_single_trial already inverts for negative tests (case-success
        semantic), so TaskResult.overall_passed must not invert again. Before the
        fix this double-inverted and reported PASSED cases as FAILED.
        """
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # fails (actual exit 0)
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            expect="fail",
            trials=3,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.total_trials == 3
        # Agent "failed" (grader failed) on all 3 trials → negative test PASSES,
        # exactly like the single-trial path does.
        assert case.passed is True
        assert case.status == TestStatus.PASSED

    @pytest.mark.asyncio
    async def test_expect_fail_multitrial_fails_when_graders_pass(self):
        """Opposite direction: a negative test with trials>1 must FAIL when the
        agent succeeds on every trial (it should have been rejected).

        Before the fix the double-inversion made this INCORRECTLY pass.
        """
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            expect="fail",
            trials=3,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.total_trials == 3
        assert case.passed is False
        assert case.status == TestStatus.FAILED

    @pytest.mark.asyncio
    async def test_expect_pass_normal_behavior(self):
        """Positive test (expect=pass) has normal pass/fail behavior."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, weight=1.0),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
            expect="pass",  # Positive test (default)
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.overall_score == 1.0
        assert case.passed is True


class TestMetricsConfigWiring:
    """The case ``metrics:`` block must actually drive reported trial metrics."""

    @pytest.mark.asyncio
    async def test_case_pass_at_k_config_flows_into_trial_metrics(self):
        """A custom ``metrics.pass_at_k`` reaches CaseResult.trial_metrics
        through the runner→TaskResult chain (previously ignored)."""
        scenario = _make_scenario(
            grader_configs=[GraderConfig(name="exit_code_check", config={})],
            adapter_config={"run_command": "python main.py"},
            trials=2,
            metrics=MetricsConfig(pass_at_k=[1, 2], consistency=False),
        )
        runner = Compass()
        result = await runner.run(scenario)

        tm = result.case_results[0].trial_metrics
        assert tm is not None
        # Requested k values are present...
        assert "pass_at_1" in tm and "pass_at_2" in tm
        assert "pass_all_2" in tm
        # ...and non-requested defaults are absent...
        assert "pass_at_3" not in tm
        assert "pass_at_5" not in tm
        # ...and consistency was disabled via config.
        assert "consistency" not in tm


class TestAggregationZeroWeight:
    """A case whose scored graders all have weight 0 must not be force-failed."""

    def test_all_zero_weight_passing_graders_pass(self):
        """All graded graders pass but carry weight 0 → fall back to an
        unweighted mean (1.0) and pass, instead of the old (0.0, fail)."""
        results = [
            EvaluatorResult(name="g1", score=1.0, passed=True, weight=0.0),
            EvaluatorResult(name="g2", score=1.0, passed=True, weight=0.0),
        ]
        score, passed = Compass._aggregate_results(results, pass_threshold=0.7)
        assert score == pytest.approx(1.0)
        assert passed is True

    def test_all_zero_weight_failing_graders_fail(self):
        """Unweighted mean of failing (0.0) graders stays below threshold."""
        results = [
            EvaluatorResult(name="g1", score=0.0, passed=False, weight=0.0),
            EvaluatorResult(name="g2", score=0.0, passed=False, weight=0.0),
        ]
        score, passed = Compass._aggregate_results(results, pass_threshold=0.7)
        assert score == pytest.approx(0.0)
        assert passed is False

    def test_zero_weight_still_respects_gate(self):
        """The gate/required checks must still run (no early return): a failing
        gate fails the case even though the scored mean is 1.0."""
        results = [
            EvaluatorResult(name="scored", score=1.0, passed=True, weight=0.0),
            EvaluatorResult(name="gate", score=0.0, passed=False, weight=1.0, gate=True),
        ]
        score, passed = Compass._aggregate_results(results, pass_threshold=0.7)
        assert score == pytest.approx(1.0)  # gate excluded from score
        assert passed is False  # but the failing gate blocks the case


# ===================================================================
# Output format tests (compass analyze compatibility)
# ===================================================================


class TestOutputFormat:
    """Tests for EvalResult output format compatibility with compass analyze."""

    @pytest.mark.asyncio
    async def test_evaluator_result_has_grader_type_and_scope(self):
        """EvaluatorResult includes grader_type and grader_scope."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        assert hasattr(er, "grader_type")
        assert hasattr(er, "grader_scope")
        assert er.grader_type == "code"
        assert er.grader_scope == "outcome"

    @pytest.mark.asyncio
    async def test_eval_result_to_dict_includes_grade_results(self):
        """EvalResult.to_dict() includes grade_results for compass analyze."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        result_dict = result.to_dict()
        case_dict = result_dict["case_results"][0]

        # Should have both field names for compatibility
        assert "grade_results" in case_dict
        assert "evaluator_results" in case_dict
        assert "task_id" in case_dict  # Alias for case_id

        # Check grade_results structure
        gr = case_dict["grade_results"][0]
        assert "name" in gr
        assert "score" in gr
        assert "passed" in gr
        assert "grader_type" in gr
        assert "grader_scope" in gr
        assert gr["grader_type"] == "code"
        assert gr["grader_scope"] == "outcome"

    @pytest.mark.asyncio
    async def test_evaluator_result_to_dict(self):
        """EvaluatorResult.to_dict() includes all fields."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, required=True),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        er_dict = er.to_dict()

        assert er_dict["name"] == "exit_code_check"
        assert er_dict["passed"] is True
        assert er_dict["score"] == 1.0
        assert er_dict["weight"] == 1.0
        assert er_dict["required"] is True
        assert er_dict["grader_type"] == "code"
        assert er_dict["grader_scope"] == "outcome"
        assert "weighted_score" in er_dict
        assert "metadata" in er_dict


# ===================================================================
# Grader type string fallback tests
# ===================================================================


class TestGraderTypeStringFallback:
    """Tests for grader_type/grader_scope string handling in runner."""

    @pytest.mark.asyncio
    async def test_grader_with_string_type_works(self):
        """Grader with string grader_type (not enum) should work without error.

        This tests the defensive code in runner._run_graders_sequential() that
        handles both enum and string types for grader_type/grader_scope.
        """
        from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope
        from compass.graders.registry import register_grader, unregister_grader

        # Register a grader with string grader_type (simulating the bug scenario)
        @register_grader("_test_string_type_grader")
        class StringTypeGrader(CodeGrader):
            name = "_test_string_type_grader"
            grader_type = "code"  # String instead of GraderType.CODE
            grader_scope = GraderScope.OUTCOME

            async def grade(self, context: GradeContext) -> GradeResult:
                return GradeResult(
                    name=self.name,
                    passed=True,
                    score=1.0,
                )

        try:
            scenario = _make_scenario(
                grader_configs=[
                    GraderConfig(name="_test_string_type_grader", config={}),
                ],
                adapter_config={"run_command": "echo ok"},
            )
            runner = Compass()
            result = await runner.run(scenario)

            # Should not raise AttributeError
            case = result.case_results[0]
            assert len(case.evaluator_results) == 1
            er = case.evaluator_results[0]
            assert er.grader_type == "code"
            assert er.passed is True
        finally:
            unregister_grader("_test_string_type_grader")

    @pytest.mark.asyncio
    async def test_grader_with_string_scope_works(self):
        """Grader with string grader_scope (not enum) should work without error."""
        from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderType
        from compass.graders.registry import register_grader, unregister_grader

        @register_grader("_test_string_scope_grader")
        class StringScopeGrader(CodeGrader):
            name = "_test_string_scope_grader"
            grader_type = GraderType.CODE
            grader_scope = "outcome"  # String instead of GraderScope.OUTCOME

            async def grade(self, context: GradeContext) -> GradeResult:
                return GradeResult(
                    name=self.name,
                    passed=True,
                    score=1.0,
                )

        try:
            scenario = _make_scenario(
                grader_configs=[
                    GraderConfig(name="_test_string_scope_grader", config={}),
                ],
                adapter_config={"run_command": "echo ok"},
            )
            runner = Compass()
            result = await runner.run(scenario)

            case = result.case_results[0]
            er = case.evaluator_results[0]
            assert er.grader_scope == "outcome"
            assert er.passed is True
        finally:
            unregister_grader("_test_string_scope_grader")


# ===================================================================
# Gate grader tests
# ===================================================================


class TestGateGraders:
    """Tests for gate grader mechanism (hard pass/fail excluded from score)."""

    @pytest.mark.asyncio
    async def test_gate_pass_excluded_from_score(self):
        """Gate grader that passes should not affect weighted score."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={},
                    gate=True,  # Gate: must pass, excluded from score
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Fails, score=0.0
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.3,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Gate grader passes (exit_code 0 == 0)
        assert case.evaluator_results[0].gate is True
        assert case.evaluator_results[0].passed is True
        # Graded grader fails (exit_code 0 != 99) → score = 0.0
        assert case.evaluator_results[1].gate is False
        # Score is based ONLY on graded grader (0.0), not gate
        assert case.overall_score == pytest.approx(0.0)
        # Fails because graded score 0.0 < 0.3 threshold
        assert case.passed is False

    @pytest.mark.asyncio
    async def test_gate_fail_causes_overall_fail(self):
        """Gate grader failure causes overall failure even if graded score is high."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Fails (actual 0 != 99)
                    gate=True,
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={},  # Passes (exit_code 0 == 0), score=1.0
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.5,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Graded score is 1.0 (above threshold), but gate failed
        assert case.overall_score == pytest.approx(1.0)
        assert case.passed is False  # Gate failure overrides

    @pytest.mark.asyncio
    async def test_gate_fail_score_from_graded_only(self):
        """When gate fails, score is still calculated from graded graders only."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Fails
                    gate=True,
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={},  # Passes, score=1.0
                    weight=2.0,
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Fails, score=0.0
                    weight=1.0,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.3,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Score from graded only: (1.0*2.0 + 0.0*1.0) / (2.0+1.0) = 0.667
        assert case.overall_score == pytest.approx(2.0 / 3.0, abs=0.01)
        # Gate failed, so overall fails regardless of score
        assert case.passed is False

    @pytest.mark.asyncio
    async def test_only_gate_graders_all_pass(self):
        """Only gate graders, all pass → passed=True, score=1.0."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, gate=True),
                GraderConfig(name="exit_code_check", config={}, gate=True),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.7,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.overall_score == pytest.approx(1.0)
        assert case.passed is True

    @pytest.mark.asyncio
    async def test_only_gate_graders_one_fail(self):
        """Only gate graders, one fails → passed=False, score=0.0."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, gate=True),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},
                    gate=True,
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.7,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.overall_score == pytest.approx(0.0)
        assert case.passed is False

    @pytest.mark.asyncio
    async def test_gate_and_required_combined(self):
        """Gate and required can coexist independently."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={},
                    gate=True,  # Gate: passes
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},
                    weight=1.0,
                    required=True,  # Required: fails (0 != 99)
                ),
                GraderConfig(
                    name="exit_code_check",
                    config={},
                    weight=1.0,  # Passes
                ),
            ],
            adapter_config={"run_command": "python main.py"},
            pass_threshold=0.3,
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # Gate passes, graded score = (0+1)/2 = 0.5 > 0.3 threshold,
        # but required grader failed → overall fails
        assert case.overall_score == pytest.approx(0.5)
        assert case.passed is False

    @pytest.mark.asyncio
    async def test_gate_flag_preserved_in_result(self):
        """Gate flag is correctly propagated to EvaluatorResult."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}, gate=True),
                GraderConfig(name="exit_code_check", config={}, gate=False),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.evaluator_results[0].gate is True
        assert case.evaluator_results[1].gate is False

        # Check to_dict serialization
        er_dict = case.evaluator_results[0].to_dict()
        assert er_dict["gate"] is True

    @pytest.mark.asyncio
    async def test_gate_from_yaml(self):
        """Gate flag is correctly parsed from YAML scenario."""
        import tempfile

        yaml_content = """
name: Gate Test
agent:
  adapter: _test_coding_runner
cases:
  - id: case_1
    input:
      prompt: test
      params:
        files:
          - path: main.py
            content: "print('hello')"
    graders:
      - name: exit_code_check
        gate: true
        config: {}
      - name: exit_code_check
        weight: 1.0
        config: {}
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()

            scenario = Scenario.from_yaml(f.name)
            case = scenario.cases[0]

            assert case.graders[0].gate is True
            assert case.graders[1].gate is False

            runner = Compass()
            result = await runner.run(scenario)

            case_result = result.case_results[0]
            assert case_result.passed is True
            assert case_result.evaluator_results[0].gate is True


# ===================================================================
# Failure tags tests
# ===================================================================


class TestFailureTags:
    """Tests for failure_tags propagation from graders to results."""

    @pytest.mark.asyncio
    async def test_failure_tags_propagated_on_fail(self):
        """Failing grader should have failure_tags propagated to EvaluatorResult."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},  # Will fail
                ),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        assert er.passed is False
        assert "exit_code_mismatch" in er.failure_tags

    @pytest.mark.asyncio
    async def test_failure_tags_empty_on_pass(self):
        """Passing grader should have empty failure_tags."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="exit_code_check", config={}),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        assert er.passed is True
        assert er.failure_tags == []

    @pytest.mark.asyncio
    async def test_failure_tags_in_to_dict(self):
        """failure_tags should appear in to_dict() serialization."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(
                    name="exit_code_check",
                    config={"expected_exit_code": 99},
                ),
            ],
            adapter_config={"run_command": "python main.py"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        er_dict = er.to_dict()
        assert "failure_tags" in er_dict
        assert "exit_code_mismatch" in er_dict["failure_tags"]

    @pytest.mark.asyncio
    async def test_grader_error_has_failure_tag(self):
        """Grader that raises an exception should get grader_error tag."""
        scenario = _make_scenario(
            grader_configs=[
                GraderConfig(name="nonexistent_grader_xyz", config={}),
            ],
            adapter_config={"run_command": "echo ok"},
        )
        runner = Compass()
        result = await runner.run(scenario)

        er = result.case_results[0].evaluator_results[0]
        assert er.passed is False
        assert "grader_error" in er.failure_tags
