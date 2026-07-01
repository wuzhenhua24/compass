"""Trial management for multi-attempt testing."""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from compass.core.metrics import TrialMetrics, calculate_metrics


@dataclass
class TrialResult:
    """Result of a single trial (one attempt at a task)."""

    trial_id: str
    trial_number: int
    passed: bool
    score: float
    grader_results: list[Any] = field(default_factory=list)
    duration_ms: float = 0.0
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Outcome state
    outcome: dict[str, Any] = field(default_factory=dict)

    # Environment snapshot
    environment: dict[str, Any] = field(default_factory=dict)

    # Transcript reference (for trace correlation)
    transcript_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "trial_id": self.trial_id,
            "trial_number": self.trial_number,
            "passed": self.passed,
            "score": self.score,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "outcome": self.outcome,
            "environment": self.environment,
            "transcript_id": self.transcript_id,
        }


@dataclass
class TaskResult:
    """Aggregated result for a task across multiple trials."""

    task_id: str
    expect: str  # "pass" or "fail"
    expect_reason: str = ""

    trials: list[TrialResult] = field(default_factory=list)
    input_data: dict[str, Any] = field(default_factory=dict)

    # Computed after all trials complete
    _metrics: TrialMetrics | None = field(default=None, repr=False)

    @property
    def total_trials(self) -> int:
        """Number of trials executed."""
        return len(self.trials)

    @property
    def passed_trials(self) -> int:
        """Number of trials that passed."""
        return sum(1 for t in self.trials if t.passed)

    @property
    def metrics(self) -> TrialMetrics:
        """Get or calculate metrics."""
        if self._metrics is None:
            self._metrics = calculate_metrics(
                passed_list=[t.passed for t in self.trials],
                scores=[t.score for t in self.trials],
            )
        return self._metrics

    @property
    def overall_passed(self) -> bool:
        """Whether the task passed overall (majority of trials met the expectation).

        ``TrialResult.passed`` already encodes *case-level* success: for negative
        tests (``expect="fail"``) a trial is recorded as passed when the agent was
        correctly blocked or the graders correctly rejected the output — this
        inversion happens once, in ``Compass._run_single_trial``. Aggregation is
        therefore uniform across positive and negative tests; re-inverting here
        would double-negate and flip the verdict for negative tests.
        """
        return self.metrics.pass_rate >= 0.5

    @property
    def is_positive_test(self) -> bool:
        """Whether this is a positive test (expect pass)."""
        return self.expect == "pass"

    @property
    def is_negative_test(self) -> bool:
        """Whether this is a negative test (expect fail)."""
        return self.expect == "fail"

    def pass_at_k(self, k: int) -> float:
        """Get pass@k metric."""
        return self.metrics.pass_at_k(k)

    def pass_all_k(self, k: int) -> float:
        """Get pass^k metric."""
        return self.metrics.pass_all_k(k)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "expect": self.expect,
            "expect_reason": self.expect_reason,
            "overall_passed": self.overall_passed,
            "is_positive_test": self.is_positive_test,
            "total_trials": self.total_trials,
            "passed_trials": self.passed_trials,
            "metrics": self.metrics.to_dict(),
            "trials": [t.to_dict() for t in self.trials],
        }


class TrialManager:
    """Manages execution of multiple trials for a task."""

    def __init__(
        self,
        num_trials: int = 1,
        parallel: bool = False,
        max_workers: int = 4,
    ):
        """Initialize trial manager.

        Args:
            num_trials: Number of trials to run per task.
            parallel: Whether to run trials in parallel.
            max_workers: Maximum parallel workers.
        """
        self.num_trials = num_trials
        self.parallel = parallel
        self.max_workers = max_workers

    async def run_trials(
        self,
        task_id: str,
        run_fn,
        expect: str = "pass",
        expect_reason: str = "",
        input_data: dict[str, Any] | None = None,
    ) -> TaskResult:
        """Run multiple trials for a task.

        Args:
            task_id: Unique identifier for the task.
            run_fn: Async function that executes a single trial.
                    Should return (passed: bool, score: float, grader_results, outcome, error)
            expect: Expected outcome ("pass" or "fail").
            expect_reason: Reason for expected outcome.
            input_data: Input data for the task.

        Returns:
            TaskResult with all trial results and metrics.
        """
        task_result = TaskResult(
            task_id=task_id,
            expect=expect,
            expect_reason=expect_reason,
            input_data=input_data or {},
        )

        if self.parallel and self.num_trials > 1:
            trials = await self._run_parallel(task_id, run_fn)
        else:
            trials = await self._run_sequential(task_id, run_fn)

        task_result.trials = trials
        return task_result

    async def _run_sequential(self, task_id: str, run_fn) -> list[TrialResult]:
        """Run trials sequentially."""
        results = []
        for i in range(self.num_trials):
            trial = await self._execute_trial(task_id, i + 1, run_fn)
            results.append(trial)
        return results

    async def _run_parallel(self, task_id: str, run_fn) -> list[TrialResult]:
        """Run trials in parallel with semaphore."""
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_with_semaphore(trial_num: int) -> TrialResult:
            async with semaphore:
                return await self._execute_trial(task_id, trial_num, run_fn)

        tasks = [run_with_semaphore(i + 1) for i in range(self.num_trials)]
        return await asyncio.gather(*tasks)

    async def _execute_trial(
        self,
        task_id: str,
        trial_number: int,
        run_fn,
    ) -> TrialResult:
        """Execute a single trial.

        Args:
            task_id: Task identifier.
            trial_number: Trial number (1-indexed).
            run_fn: Function to execute.

        Returns:
            TrialResult for this trial.
        """
        trial_id = f"{task_id}_trial_{trial_number}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()

        try:
            result = await run_fn()

            # Unpack result - expecting tuple of (passed, score, grader_results, outcome, env, transcript?)
            transcript_id = None
            if isinstance(result, tuple):
                passed, score, grader_results, outcome, env, *rest = result
                error = None
                # Extract transcript_id if transcript is provided (6th element)
                if rest and rest[0] is not None:
                    transcript = rest[0]
                    transcript_id = getattr(transcript, "trial_id", None)
            else:
                # If result is just a dict or object, try to extract
                passed = getattr(result, "passed", False)
                score = getattr(result, "score", 0.0)
                grader_results = getattr(result, "grader_results", [])
                outcome = getattr(result, "outcome", {})
                env = getattr(result, "environment", {})
                error = getattr(result, "error", None)
                transcript_id = getattr(result, "transcript_id", None)

            duration_ms = (time.time() - start_time) * 1000

            return TrialResult(
                trial_id=trial_id,
                trial_number=trial_number,
                passed=passed,
                score=score,
                grader_results=grader_results,
                duration_ms=duration_ms,
                error=error,
                outcome=outcome,
                environment=env,
                transcript_id=transcript_id,
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000

            return TrialResult(
                trial_id=trial_id,
                trial_number=trial_number,
                passed=False,
                score=0.0,
                duration_ms=duration_ms,
                error=str(e),
            )
