"""Tests for evaluation hygiene: what counts in the numbers, and what doesn't.

Four distinctions the aggregate must keep straight:

1. A harness failure is not the agent failing  -> excluded from the denominator
2. "Not measured" is not "measured as zero"    -> unscored, not 0.0
3. Core control flags are not user data        -> a grader can't clobber them
4. A score is comparable only under the same grading contract -> fingerprints

Plus the sampling discipline that makes pass^k mean what it claims: trials run
round-major and top up to a target instead of re-running from scratch.
"""

from __future__ import annotations

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifacts import TextArtifact
from compass.core.checkpoint import (
    _dict_to_trial_result,
    _trial_result_to_dict,
)
from compass.core.regrade import grader_fingerprint
from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.core.runner import Compass
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    DefaultsConfig,
    GraderConfig,
    InputConfig,
    Scenario,
    ShortCircuitMode,
    TestCase,
)
from compass.core.trial import TaskResult, TrialResult
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    ModelGrader,
)
from compass.graders.registry import register_grader

# ===================================================================
# Test doubles
# ===================================================================


@register_grader("_hyg_always_pass")
class _AlwaysPass(CodeGrader):
    name = "_hyg_always_pass"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=True,
            score=self.config.get("score", 1.0),
        )


@register_grader("_hyg_crashes")
class _Crashes(ModelGrader):
    """Stands in for an LLM judge that times out."""

    name = "_hyg_crashes"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        raise RuntimeError("judge unavailable")


@register_grader("_hyg_details_say_skipped")
class _DetailsSaySkipped(CodeGrader):
    """A grader whose own `details` payload collides with a core control flag."""

    name = "_hyg_details_say_skipped"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=False, score=0.0,
            details={"skipped": True, "reason": "user payload, not a control flag"},
        )


class _CountingAdapter(Adapter):
    """Records call order; can be told to fail on specific attempts."""

    name = "_hyg_counting"
    calls: list[str] = []
    fail_first_n: dict[str, int] = {}

    async def run(self, input: AgentInput) -> AgentOutput:
        case_id = input.params.get("case_id", "?")
        type(self).calls.append(case_id)
        remaining = type(self).fail_first_n.get(case_id, 0)
        if remaining > 0:
            type(self).fail_first_n[case_id] = remaining - 1
            raise RuntimeError(f"adapter exploded for {case_id}")
        return AgentOutput(artifacts=[TextArtifact(content="ok")])


@pytest.fixture(autouse=True)
def _adapter():
    _CountingAdapter.calls = []
    _CountingAdapter.fail_first_n = {}
    register_adapter("_hyg_counting")(_CountingAdapter)
    yield _CountingAdapter
    unregister_adapter("_hyg_counting")


def _scenario(
    *,
    graders: list[GraderConfig] | None = None,
    case_ids: tuple[str, ...] = ("case_a",),
    trials: int = 1,
    threshold: float = 0.7,
    short_circuit: ShortCircuitMode = ShortCircuitMode.DISABLED,
) -> Scenario:
    return Scenario(
        name="hygiene",
        agent=AgentConfig(adapter="_hyg_counting"),
        defaults=DefaultsConfig(trials=trials),
        cases=[
            TestCase(
                id=cid,
                input=InputConfig(prompt="go", params={"case_id": cid}),
                graders=list(graders or [GraderConfig(name="_hyg_always_pass")]),
                aggregation=AggregationConfig(
                    pass_threshold=threshold, short_circuit=short_circuit
                ),
            )
            for cid in case_ids
        ],
    )


def _case(status: TestStatus, passed: bool, score: float) -> CaseResult:
    return CaseResult(
        case_id=f"c{score}", status=status, passed=passed, overall_score=score
    )


# ===================================================================
# 1. A harness failure is not the agent failing
# ===================================================================


class TestHarnessErrorsExcluded:
    def test_pass_rate_denominator_excludes_errors(self):
        """One pass + one infra error is 100%, not 50%."""
        result = EvalResult(
            scenario_name="s", total_cases=2, passed_cases=1,
            failed_cases=0, error_cases=1,
            case_results=[
                _case(TestStatus.PASSED, True, 1.0),
                _case(TestStatus.ERROR, False, 0.0),
            ],
        )
        assert result.evaluated_cases == 1
        assert result.pass_rate == 1.0

    def test_real_failure_still_counts(self):
        """Excluding errors must not also excuse genuine failures."""
        result = EvalResult(
            scenario_name="s", total_cases=2, passed_cases=1,
            failed_cases=1, error_cases=0,
            case_results=[
                _case(TestStatus.PASSED, True, 1.0),
                _case(TestStatus.FAILED, False, 0.0),
            ],
        )
        assert result.evaluated_cases == 2
        assert result.pass_rate == 0.5

    def test_average_score_excludes_errors(self):
        """An errored case contributes no 0.0 to the mean."""
        result = EvalResult(
            scenario_name="s", total_cases=2, passed_cases=1,
            failed_cases=0, error_cases=1,
            case_results=[
                _case(TestStatus.PASSED, True, 0.8),
                _case(TestStatus.ERROR, False, 0.0),
            ],
        )
        assert result.average_score == pytest.approx(0.8)
        assert result.best_of_k_score == pytest.approx(0.8)

    def test_all_errors_is_zero_not_a_crash(self):
        result = EvalResult(
            scenario_name="s", total_cases=1, passed_cases=0,
            failed_cases=0, error_cases=1,
            case_results=[_case(TestStatus.ERROR, False, 0.0)],
        )
        assert result.evaluated_cases == 0
        assert result.pass_rate == 0.0
        assert result.average_score == 0.0

    def test_to_dict_exposes_the_denominator(self):
        result = EvalResult(
            scenario_name="s", total_cases=2, passed_cases=1,
            failed_cases=0, error_cases=1,
            case_results=[
                _case(TestStatus.PASSED, True, 1.0),
                _case(TestStatus.ERROR, False, 0.0),
            ],
        )
        d = result.to_dict()
        assert d["evaluated_cases"] == 1
        assert d["total_cases"] == 2
        assert d["pass_rate"] == 1.0

    async def test_adapter_failure_produces_an_error_case(self):
        scenario = _scenario(case_ids=("case_a", "case_b"))
        _CountingAdapter.fail_first_n = {"case_b": 1}

        result = await Compass().run(scenario)

        assert result.error_cases == 1
        assert result.evaluated_cases == 1
        # The one case that actually ran passed — infra noise didn't halve it
        assert result.pass_rate == 1.0


class TestErroredTrialsExcluded:
    def _task(self, *specs: tuple[bool, float | None, str | None]) -> TaskResult:
        return TaskResult(
            task_id="t",
            expect="pass",
            trials=[
                TrialResult(
                    trial_id=f"t{i}", trial_number=i + 1,
                    passed=p, score=s, error=e,
                )
                for i, (p, s, e) in enumerate(specs)
            ],
        )

    def test_errored_trial_is_a_missing_sample_not_a_miss(self):
        """2 passes + 1 harness error is pass_rate 1.0, not 2/3."""
        task = self._task(
            (True, 1.0, None), (True, 1.0, None), (False, None, "network down")
        )
        assert task.total_trials == 3
        assert task.error_trials == 1
        assert task.passed_trials == 2
        assert task.metrics.pass_rate == 1.0
        assert task.overall_passed is True

    def test_genuine_trial_failure_still_counts(self):
        task = self._task((True, 1.0, None), (False, 0.0, None))
        assert task.error_trials == 0
        assert task.metrics.pass_rate == 0.5

    def test_all_trials_errored_fails_closed(self):
        """No evidence either way must not read as a vacuous pass."""
        task = self._task((False, None, "boom"), (False, None, "boom"))
        assert task.evaluated_trials == []
        assert task.overall_passed is False


# ===================================================================
# 2. "Not measured" is not "measured as zero"
# ===================================================================


class TestUnscoredIsNotZero:
    async def test_crashed_grader_is_unscored(self):
        scenario = _scenario(graders=[GraderConfig(name="_hyg_crashes")])
        result = await Compass().run(scenario)

        grader = result.case_results[0].evaluator_results[0]
        assert grader.score is None
        assert grader.scored is False
        assert "grader_error" in grader.failure_tags

    async def test_crashed_grader_does_not_drag_the_mean_to_zero(self):
        """The surviving grader's 1.0 stays 1.0 — it is not averaged with a phantom 0."""
        scenario = _scenario(
            graders=[
                GraderConfig(name="_hyg_always_pass"),
                GraderConfig(name="_hyg_crashes"),
            ]
        )
        result = await Compass().run(scenario)
        case = result.case_results[0]

        assert case.overall_score == pytest.approx(1.0)

    async def test_but_an_incomplete_evaluation_cannot_certify_a_pass(self):
        """Score 1.0 and still failed: we did not measure everything we promised."""
        scenario = _scenario(
            graders=[
                GraderConfig(name="_hyg_always_pass"),
                GraderConfig(name="_hyg_crashes"),
            ]
        )
        result = await Compass().run(scenario)
        case = result.case_results[0]

        assert case.overall_score == pytest.approx(1.0)
        assert case.passed is False
        assert case.status == TestStatus.FAILED

    def test_aggregate_ignores_unscored_in_the_denominator(self):
        scored = EvaluatorResult(name="a", score=0.6, passed=True, weight=1.0)
        unscored = EvaluatorResult(name="b", score=None, passed=False, weight=1.0)

        score, passed = Compass._aggregate_results([scored], pass_threshold=0.5)
        assert score == pytest.approx(0.6) and passed

        score, passed = Compass._aggregate_results(
            [scored, unscored], pass_threshold=0.5
        )
        # Same score (0.6, not 0.3) — but the case can no longer pass
        assert score == pytest.approx(0.6)
        assert passed is False

    def test_breakdown_omits_unscored(self):
        case = CaseResult(
            case_id="c", status=TestStatus.FAILED, passed=False, overall_score=0.6,
            evaluator_results=[
                EvaluatorResult(name="a", score=0.6, passed=True),
                EvaluatorResult(name="b", score=None, passed=False),
            ],
        )
        assert case.breakdown == {"a": pytest.approx(0.6)}

    async def test_short_circuit_skip_is_unscored_but_forgiven(self):
        """Skipped is a decision, not a gap: it must not fail the case."""
        scenario = _scenario(
            graders=[
                GraderConfig(name="_hyg_always_pass"),
                GraderConfig(name="_hyg_crashes", type="model"),
            ],
            short_circuit=ShortCircuitMode.CODE_PASS,
        )
        result = await Compass().run(scenario)
        case = result.case_results[0]

        skipped = next(r for r in case.evaluator_results if r.name == "_hyg_crashes")
        assert skipped.skipped is True
        assert skipped.score is None
        # Skipping is deliberate, so the case still passes on the code grader
        assert case.passed is True

    def test_weighted_score_of_an_unscored_result_is_zero_not_a_crash(self):
        assert EvaluatorResult(name="a", score=None, passed=False).weighted_score == 0.0


# ===================================================================
# 3. Core control flags are not user data
# ===================================================================


class TestNamespaceIsolation:
    async def test_user_details_cannot_fake_a_skip(self):
        """A grader returning details={"skipped": True} is still scored.

        The aggregate reads the core-owned `skipped` field; `metadata` carries
        the grader's own payload and is never trusted for control flow.
        """
        scenario = _scenario(
            graders=[
                GraderConfig(name="_hyg_always_pass"),
                GraderConfig(name="_hyg_details_say_skipped"),
            ]
        )
        result = await Compass().run(scenario)
        case = result.case_results[0]

        sneaky = next(
            r for r in case.evaluator_results if r.name == "_hyg_details_say_skipped"
        )
        assert sneaky.metadata.get("skipped") is True  # user payload survives
        assert sneaky.skipped is False  # ...but the core flag is untouched
        # It scored 0.0 and was counted: mean of 1.0 and 0.0
        assert case.overall_score == pytest.approx(0.5)
        assert case.passed is False

    def test_aggregate_reads_the_field_not_the_dict(self):
        counted = EvaluatorResult(
            name="a", score=0.0, passed=False, metadata={"skipped": True}
        )
        other = EvaluatorResult(name="b", score=1.0, passed=True)
        score, _ = Compass._aggregate_results([counted, other], pass_threshold=0.5)
        assert score == pytest.approx(0.5)

        really_skipped = EvaluatorResult(
            name="a", score=None, passed=False, skipped=True
        )
        score, _ = Compass._aggregate_results(
            [really_skipped, other], pass_threshold=0.5
        )
        assert score == pytest.approx(1.0)


# ===================================================================
# 4. A score is comparable only under the same grading contract
# ===================================================================


class TestGraderFingerprint:
    async def test_live_run_stamps_the_fingerprint(self):
        scenario = _scenario()
        result = await Compass().run(scenario)
        assert result.case_results[0].grader_fingerprint

    async def test_live_and_offline_fingerprints_agree(self):
        """The same contract must hash the same whether run or re-graded."""
        scenario = _scenario()
        result = await Compass().run(scenario)
        expected = grader_fingerprint(scenario, scenario.cases[0])
        assert result.case_results[0].grader_fingerprint == expected

    async def test_editing_a_threshold_changes_it(self):
        a = await Compass().run(_scenario(threshold=0.7))
        b = await Compass().run(_scenario(threshold=0.9))
        assert (
            a.case_results[0].grader_fingerprint
            != b.case_results[0].grader_fingerprint
        )

    def test_compare_flags_cases_graded_under_different_contracts(self):
        from compass.report.compare import CaseRecord, compare_results

        a = {"c1": CaseRecord("c1", True, 1.0, 1.0, grader_fingerprint="aaa")}
        b = {"c1": CaseRecord("c1", False, 0.0, 0.0, grader_fingerprint="bbb")}
        report = compare_results(a, b)

        assert report.regraded == ["c1"]
        assert report.to_dict()["regraded"] == ["c1"]

    def test_compare_stays_quiet_when_contracts_match(self):
        from compass.report.compare import CaseRecord, compare_results

        a = {"c1": CaseRecord("c1", True, 1.0, 1.0, grader_fingerprint="aaa")}
        b = {"c1": CaseRecord("c1", False, 0.0, 0.0, grader_fingerprint="aaa")}
        assert compare_results(a, b).regraded == []

    def test_compare_cannot_claim_a_mismatch_it_does_not_know_about(self):
        """Older results have no fingerprint — absence is not a mismatch."""
        from compass.report.compare import CaseRecord, compare_results

        a = {"c1": CaseRecord("c1", True, 1.0, 1.0, grader_fingerprint="")}
        b = {"c1": CaseRecord("c1", False, 0.0, 0.0, grader_fingerprint="bbb")}
        assert compare_results(a, b).regraded == []

    def test_pass_fraction_excludes_errored_trials(self):
        from compass.report.compare import _extract_case_records

        record = _extract_case_records(
            {
                "case_id": "c1", "passed": True, "overall_score": 1.0,
                "total_trials": 3, "passed_trials": 2, "error_trials": 1,
            }
        )
        # 2 of 2 evaluated, not 2 of 3
        assert record.pass_fraction == pytest.approx(1.0)


# ===================================================================
# 5. Sampling: round-major, and top up rather than re-run
# ===================================================================


class TestTrialSampling:
    async def test_trials_run_round_major(self):
        """Full passes over every case, so an interruption leaves it balanced."""
        scenario = _scenario(case_ids=("case_a", "case_b"), trials=3)
        await Compass().run(scenario)

        assert _CountingAdapter.calls == [
            "case_a", "case_b",  # round 1
            "case_a", "case_b",  # round 2
            "case_a", "case_b",  # round 3
        ]

    async def test_resume_after_completion_is_a_no_op(self, tmp_path):
        """Target already met: re-running the same command executes nothing."""
        scenario = _scenario(case_ids=("case_a",), trials=3)
        runner = Compass()

        await runner.run(scenario, trace_dir=tmp_path)
        assert len(_CountingAdapter.calls) == 3

        _CountingAdapter.calls = []
        result = await runner.run(scenario, trace_dir=tmp_path, resume=True)

        assert _CountingAdapter.calls == []
        assert result.case_results[0].total_trials == 3

    async def test_partial_trials_are_topped_up_not_restarted(self, tmp_path):
        from compass.core.checkpoint import CheckpointStore, scenario_fingerprint

        scenario = _scenario(case_ids=("case_a",), trials=3)
        # Seed a checkpoint that already holds one good trial
        store = CheckpointStore.create(
            tmp_path, "run1", scenario_fingerprint(scenario), scenario.name, ["case_a"]
        )
        store.save_trial_result(
            "case_a",
            TrialResult(trial_id="x", trial_number=1, passed=True, score=1.0),
        )

        result = await Compass().run(scenario, trace_dir=tmp_path, resume=True)

        # Only the 2-trial shortfall was executed
        assert _CountingAdapter.calls == ["case_a", "case_a"]
        assert result.case_results[0].total_trials == 3

    async def test_errored_trials_do_not_count_toward_the_target(self):
        """A lost sample is not a sample — but the shortfall is attempted once."""
        scenario = _scenario(case_ids=("case_a",), trials=2)
        _CountingAdapter.fail_first_n = {"case_a": 1}

        result = await Compass().run(scenario)
        case = result.case_results[0]

        # Two attempts made this invocation; one died in the harness
        assert len(_CountingAdapter.calls) == 2
        assert case.error_trials == 1
        assert case.total_trials == 2
        # No retry loop: the replacement is left for the next invocation
        assert case.trial_metrics["total_trials"] == 1

    async def test_a_persistently_broken_adapter_does_not_spin(self):
        scenario = _scenario(case_ids=("case_a",), trials=3)
        _CountingAdapter.fail_first_n = {"case_a": 99}

        result = await Compass().run(scenario)

        assert len(_CountingAdapter.calls) == 3  # exactly the shortfall, once
        assert result.case_results[0].status == TestStatus.ERROR
        assert result.error_cases == 1

    async def test_parallel_rounds_stay_balanced(self):
        scenario = _scenario(case_ids=("case_a", "case_b", "case_c"), trials=2)
        await Compass().run(scenario, parallel=True, max_workers=3)

        # Order within a round is unconstrained, but each round is a full pass
        assert sorted(_CountingAdapter.calls[:3]) == ["case_a", "case_b", "case_c"]
        assert sorted(_CountingAdapter.calls[3:]) == ["case_a", "case_b", "case_c"]


class TestTrialCheckpointRoundTrip:
    def test_trial_result_survives_serialization(self):
        original = TrialResult(
            trial_id="t1", trial_number=2, passed=True, score=0.75,
            grader_results=[EvaluatorResult(name="g", score=0.75, passed=True)],
            duration_ms=12.5, outcome={"final_output": "hi"},
            transcript_id="tr-1",
        )
        restored = _dict_to_trial_result(_trial_result_to_dict(original))

        assert restored.trial_number == 2
        assert restored.score == pytest.approx(0.75)
        assert restored.grader_results[0].name == "g"
        assert restored.outcome == {"final_output": "hi"}
        assert restored.transcript_id == "tr-1"

    def test_unscored_trial_stays_unscored(self):
        original = TrialResult(
            trial_id="t1", trial_number=1, passed=False, score=None,
            error="network down",
        )
        restored = _dict_to_trial_result(_trial_result_to_dict(original))
        assert restored.score is None
        assert restored.error == "network down"

    def test_evaluator_result_control_flags_survive(self):
        original = EvaluatorResult(
            name="g", score=None, passed=False, skipped=True,
            skip_reason="short_circuit",
        )
        restored = EvaluatorResult.from_dict(original.to_dict())
        assert restored.score is None
        assert restored.skipped is True
        assert restored.skip_reason == "short_circuit"
