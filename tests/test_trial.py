"""Tests for trial.py - TrialManager and multi-trial flow."""

from __future__ import annotations

import pytest

from compass.core.trial import TaskResult, TrialManager, TrialResult


# ===================================================================
# TrialResult tests
# ===================================================================


class TestTrialResult:
    """Tests for TrialResult dataclass."""

    def test_trial_result_basic(self):
        """TrialResult stores basic trial data."""
        result = TrialResult(
            trial_id="task_1_trial_1_abc123",
            trial_number=1,
            passed=True,
            score=0.95,
        )
        assert result.trial_id == "task_1_trial_1_abc123"
        assert result.trial_number == 1
        assert result.passed is True
        assert result.score == 0.95
        assert result.error is None

    def test_trial_result_with_error(self):
        """TrialResult captures error information."""
        result = TrialResult(
            trial_id="task_1_trial_1_abc123",
            trial_number=1,
            passed=False,
            score=0.0,
            error="Something went wrong",
        )
        assert result.passed is False
        assert result.error == "Something went wrong"

    def test_trial_result_to_dict(self):
        """TrialResult.to_dict() includes all fields."""
        result = TrialResult(
            trial_id="task_1_trial_1_abc123",
            trial_number=1,
            passed=True,
            score=0.85,
            grader_results=[{"name": "test", "passed": True}],
            duration_ms=150.5,
            outcome={"output": "test"},
            environment={"var": "value"},
            transcript_id="transcript_xyz789",
        )
        d = result.to_dict()
        assert d["trial_id"] == "task_1_trial_1_abc123"
        assert d["trial_number"] == 1
        assert d["passed"] is True
        assert d["score"] == 0.85
        assert d["duration_ms"] == 150.5
        assert d["outcome"] == {"output": "test"}
        assert d["environment"] == {"var": "value"}
        assert d["transcript_id"] == "transcript_xyz789"


# ===================================================================
# TaskResult tests
# ===================================================================


class TestTaskResult:
    """Tests for TaskResult dataclass."""

    def test_task_result_metrics(self):
        """TaskResult computes metrics from trials."""
        task = TaskResult(task_id="task_1", expect="pass")
        task.trials = [
            TrialResult(trial_id="t1", trial_number=1, passed=True, score=1.0),
            TrialResult(trial_id="t2", trial_number=2, passed=False, score=0.5),
            TrialResult(trial_id="t3", trial_number=3, passed=True, score=0.8),
        ]
        assert task.total_trials == 3
        assert task.passed_trials == 2
        assert task.metrics.pass_rate == pytest.approx(2 / 3)

    def test_task_result_overall_passed_positive_test(self):
        """Positive test passes if majority of trials pass."""
        task = TaskResult(task_id="task_1", expect="pass")
        task.trials = [
            TrialResult(trial_id="t1", trial_number=1, passed=True, score=1.0),
            TrialResult(trial_id="t2", trial_number=2, passed=True, score=1.0),
            TrialResult(trial_id="t3", trial_number=3, passed=False, score=0.0),
        ]
        # 2/3 pass rate >= 0.5 threshold
        assert task.overall_passed is True
        assert task.is_positive_test is True

    def test_task_result_overall_passed_negative_test(self):
        """Negative test: TrialResult.passed already means 'the case met its
        expectation' (agent correctly rejected). Aggregation must NOT re-invert.

        Here the agent is correctly blocked on every trial, so each trial is a
        case-success (passed=True) and the negative test passes overall.
        """
        task = TaskResult(task_id="task_1", expect="fail")
        task.trials = [
            TrialResult(trial_id="t1", trial_number=1, passed=True, score=0.0),
            TrialResult(trial_id="t2", trial_number=2, passed=True, score=0.0),
            TrialResult(trial_id="t3", trial_number=3, passed=True, score=0.0),
        ]
        assert task.metrics.pass_rate == pytest.approx(1.0)
        assert task.overall_passed is True
        assert task.is_negative_test is True

    def test_task_result_overall_failed_negative_test(self):
        """Negative test fails when the agent was NOT rejected on the majority
        of trials (i.e. most trials are case-failures)."""
        task = TaskResult(task_id="task_1", expect="fail")
        task.trials = [
            TrialResult(trial_id="t1", trial_number=1, passed=False, score=1.0),
            TrialResult(trial_id="t2", trial_number=2, passed=False, score=1.0),
            TrialResult(trial_id="t3", trial_number=3, passed=True, score=0.0),
        ]
        # 1/3 case-success rate < 0.5 → negative test fails overall.
        assert task.overall_passed is False
        assert task.is_negative_test is True

    def test_task_result_to_dict(self):
        """TaskResult.to_dict() includes all fields."""
        task = TaskResult(
            task_id="task_1",
            expect="pass",
            expect_reason="Should work",
            input_data={"prompt": "test"},
        )
        task.trials = [
            TrialResult(trial_id="t1", trial_number=1, passed=True, score=1.0),
        ]
        d = task.to_dict()
        assert d["task_id"] == "task_1"
        assert d["expect"] == "pass"
        assert d["expect_reason"] == "Should work"
        assert d["overall_passed"] is True
        assert d["total_trials"] == 1
        assert d["passed_trials"] == 1
        assert "metrics" in d
        assert "trials" in d


# ===================================================================
# TrialManager unit tests
# ===================================================================


class TestTrialManager:
    """Tests for TrialManager."""

    @pytest.mark.asyncio
    async def test_single_trial_5_tuple(self):
        """TrialManager handles 5-tuple return from run_fn."""

        async def run_fn():
            return (True, 0.9, [{"name": "g1"}], {"result": "ok"}, {"env": "test"})

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 1
        assert result.trials[0].passed is True
        assert result.trials[0].score == 0.9
        assert result.trials[0].outcome == {"result": "ok"}
        assert result.trials[0].environment == {"env": "test"}

    @pytest.mark.asyncio
    async def test_single_trial_6_tuple(self):
        """TrialManager handles 6-tuple return (with transcript) from run_fn.

        This test would have caught the ValueError before the fix.
        """

        class MockTranscript:
            trial_id = "transcript_trial_abc123"

        async def run_fn():
            # Simulates _run_single_trial() return value
            return (
                True,
                0.85,
                [{"name": "grader1"}],
                {"output": "data"},
                {"env_key": "env_value"},
                MockTranscript(),  # 6th element with trial_id
            )

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 1
        assert result.trials[0].passed is True
        assert result.trials[0].score == 0.85
        assert result.trials[0].error is None
        assert result.trials[0].transcript_id == "transcript_trial_abc123"

    @pytest.mark.asyncio
    async def test_single_trial_6_tuple_no_transcript(self):
        """TrialManager handles 6-tuple with None transcript."""

        async def run_fn():
            return (True, 0.9, [], {}, {}, None)

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.trials[0].passed is True
        assert result.trials[0].transcript_id is None

    @pytest.mark.asyncio
    async def test_single_trial_7_tuple(self):
        """TrialManager handles tuples with more than 6 elements gracefully."""

        async def run_fn():
            return (True, 0.8, [], {}, {}, None, "extra1", "extra2")

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 1
        assert result.trials[0].passed is True
        assert result.trials[0].score == 0.8

    @pytest.mark.asyncio
    async def test_multiple_trials_sequential(self):
        """TrialManager runs multiple trials sequentially."""
        call_count = 0

        async def run_fn():
            nonlocal call_count
            call_count += 1
            passed = call_count % 2 == 1  # Alternate pass/fail
            return (passed, 1.0 if passed else 0.0, [], {}, {})

        manager = TrialManager(num_trials=3, parallel=False)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert call_count == 3
        assert result.total_trials == 3
        assert result.passed_trials == 2  # trials 1 and 3 pass
        assert result.trials[0].passed is True
        assert result.trials[1].passed is False
        assert result.trials[2].passed is True

    @pytest.mark.asyncio
    async def test_multiple_trials_parallel(self):
        """TrialManager runs multiple trials in parallel."""
        import asyncio

        call_times = []

        async def run_fn():
            call_times.append(asyncio.get_event_loop().time())
            await asyncio.sleep(0.01)  # Small delay
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=3, parallel=True, max_workers=3)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 3
        # All trials should have started at roughly the same time
        if len(call_times) == 3:
            time_spread = max(call_times) - min(call_times)
            assert time_spread < 0.05  # Should start within 50ms of each other

    @pytest.mark.asyncio
    async def test_trial_error_handling(self):
        """TrialManager captures exceptions as trial errors."""

        async def run_fn():
            raise ValueError("Test error")

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 1
        assert result.trials[0].passed is False
        assert result.trials[0].score == 0.0
        assert "Test error" in result.trials[0].error

    @pytest.mark.asyncio
    async def test_mixed_success_and_errors(self):
        """TrialManager handles mix of successful and failed trials."""
        call_count = 0

        async def run_fn():
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("Trial 2 failed")
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=3, parallel=False)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.total_trials == 3
        assert result.passed_trials == 2
        assert result.trials[0].passed is True
        assert result.trials[0].error is None
        assert result.trials[1].passed is False
        assert "Trial 2 failed" in result.trials[1].error
        assert result.trials[2].passed is True

    @pytest.mark.asyncio
    async def test_trial_ids_are_unique(self):
        """Each trial gets a unique trial_id."""

        async def run_fn():
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=5)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        trial_ids = [t.trial_id for t in result.trials]
        assert len(set(trial_ids)) == 5  # All unique

    @pytest.mark.asyncio
    async def test_trial_numbers_are_sequential(self):
        """Trial numbers are 1-indexed and sequential."""

        async def run_fn():
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=3)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        trial_numbers = [t.trial_number for t in result.trials]
        assert trial_numbers == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_object_result_handling(self):
        """TrialManager extracts fields from object results."""

        class MockResult:
            passed = True
            score = 0.75
            grader_results = [{"name": "mock"}]
            outcome = {"data": "test"}
            environment = {"key": "value"}
            error = None

        async def run_fn():
            return MockResult()

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        assert result.trials[0].passed is True
        assert result.trials[0].score == 0.75
        assert result.trials[0].outcome == {"data": "test"}

    @pytest.mark.asyncio
    async def test_duration_is_recorded(self):
        """Trial duration is recorded in milliseconds."""
        import asyncio

        async def run_fn():
            await asyncio.sleep(0.05)  # 50ms
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(task_id="task_1", run_fn=run_fn)

        # Should be at least 50ms
        assert result.trials[0].duration_ms >= 50

    @pytest.mark.asyncio
    async def test_expect_and_reason_stored(self):
        """TaskResult stores expect and expect_reason."""

        async def run_fn():
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(
            task_id="task_1",
            run_fn=run_fn,
            expect="fail",
            expect_reason="Should reject invalid input",
        )

        assert result.expect == "fail"
        assert result.expect_reason == "Should reject invalid input"

    @pytest.mark.asyncio
    async def test_input_data_stored(self):
        """TaskResult stores input_data."""

        async def run_fn():
            return (True, 1.0, [], {}, {})

        manager = TrialManager(num_trials=1)
        result = await manager.run_trials(
            task_id="task_1",
            run_fn=run_fn,
            input_data={"prompt": "test", "params": {"key": "value"}},
        )

        assert result.input_data == {"prompt": "test", "params": {"key": "value"}}


# ===================================================================
# Integration tests with runner (multi-trial flow)
# ===================================================================


class TestMultiTrialIntegration:
    """Integration tests for multi-trial flow with Compass runner."""

    @pytest.fixture(autouse=True)
    def setup_adapter(self):
        """Register test adapter."""
        from compass.adapters.coding import CodingAdapter
        from compass.adapters.registry import register_adapter, unregister_adapter

        @register_adapter("_test_trial_adapter")
        class _TestTrialAdapter(CodingAdapter):
            name = "_test_trial_adapter"

        yield
        try:
            unregister_adapter("_test_trial_adapter")
        except Exception:
            pass

    def _make_scenario(
        self,
        num_trials: int = 1,
        pass_threshold: float = 0.7,
        expect: str = "pass",
    ):
        """Create a test scenario."""
        from compass.core.scenario import (
            AgentConfig,
            AggregationConfig,
            DefaultsConfig,
            GraderConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        case = TestCase(
            id="case_1",
            input=InputConfig(
                prompt="test",
                params={
                    "files": [
                        {"path": "main.py", "content": "print('hello')"},
                    ],
                },
            ),
            expect=expect,
            graders=[GraderConfig(name="exit_code_check", config={})],
            aggregation=AggregationConfig(pass_threshold=pass_threshold),
        )
        return Scenario(
            name="test_scenario",
            agent=AgentConfig(
                adapter="_test_trial_adapter",
                config={"run_command": "python main.py"},
            ),
            cases=[case],
            defaults=DefaultsConfig(trials=num_trials),
        )

    @pytest.mark.asyncio
    async def test_multi_trial_returns_correct_structure(self):
        """Multi-trial run returns proper result structure."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_trials=3)
        runner = Compass()
        result = await runner.run(scenario)

        assert result.total_cases == 1
        case = result.case_results[0]
        assert case.total_trials == 3
        assert case.passed_trials == 3  # All should pass

    @pytest.mark.asyncio
    async def test_multi_trial_aggregates_results(self):
        """Multi-trial properly aggregates pass/fail across trials."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_trials=5)
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        # All trials should pass (exit_code 0)
        assert case.passed_trials == 5
        assert case.total_trials == 5

    @pytest.mark.asyncio
    async def test_single_trial_fallback(self):
        """Single trial (num_trials=1) works correctly."""
        from compass.core.runner import Compass

        scenario = self._make_scenario(num_trials=1)
        runner = Compass()
        result = await runner.run(scenario)

        case = result.case_results[0]
        assert case.total_trials == 1
        assert case.passed is True
