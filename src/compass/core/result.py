"""Result data structures for test execution."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TestStatus(Enum):
    """Test execution status."""

    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


@dataclass
class EvaluatorResult:
    """Result from a single evaluator."""

    name: str
    score: float
    passed: bool
    weight: float = 1.0
    required: bool = False  # Whether this grader must pass for overall pass
    gate: bool = False  # Hard gate: must pass, excluded from score
    grader_type: str = "code"  # "code", "model", or "human"
    grader_scope: str = "outcome"  # "outcome", "transcript", or "both"
    grader_version: str = ""  # Verifier version that produced this score
    metadata: dict[str, Any] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)  # Structured failure labels
    error: str | None = None

    @property
    def weighted_score(self) -> float:
        """Calculate weighted score."""
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "required": self.required,
            "gate": self.gate,
            "grader_type": self.grader_type,
            "grader_scope": self.grader_scope,
            "grader_version": self.grader_version,
            "metadata": self.metadata,
            "failure_tags": self.failure_tags,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluatorResult":
        """Rebuild from a ``to_dict()`` payload.

        Unknown keys are ignored, so a stored record stays loadable after new
        fields are added — and ``weighted_score`` (a property, not a field)
        does not blow up the constructor.
        """
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class CaseResult:
    """Result for a single test case."""

    case_id: str
    status: TestStatus
    passed: bool
    overall_score: float
    evaluator_results: list[EvaluatorResult] = field(default_factory=list)
    input_data: dict[str, Any] = field(default_factory=dict)
    output_data: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    error: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Classification
    tags: list[str] = field(default_factory=list)
    category: str = ""

    # Trials-related fields
    total_trials: int = 1
    passed_trials: int = 0
    trial_metrics: dict[str, Any] | None = None  # pass@k, pass^k, pass_rate, etc.

    @property
    def best_score(self) -> float:
        """Best (max) score across trials, or overall_score for single trial."""
        if self.trial_metrics and "best_of_k" in self.trial_metrics:
            return self.trial_metrics["best_of_k"]
        if self.trial_metrics and "score_max" in self.trial_metrics:
            return self.trial_metrics["score_max"]
        return self.overall_score

    @property
    def has_trials(self) -> bool:
        """Whether this case was run with multiple trials."""
        return self.total_trials > 1

    @property
    def breakdown(self) -> dict[str, float]:
        """Get score breakdown by evaluator."""
        return {r.name: r.weighted_score for r in self.evaluator_results}


@dataclass
class EvalResult:
    """Aggregated result for a test scenario."""

    scenario_name: str
    total_cases: int
    passed_cases: int
    failed_cases: int
    error_cases: int
    case_results: list[CaseResult] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    # Audit provenance: the run that produced this result and the scenario
    # fingerprint it executed under (matches Transcript.run_id/config_hash).
    run_id: str = ""
    config_hash: str = ""

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases

    @property
    def average_score(self) -> float:
        """Calculate average score across all cases."""
        if not self.case_results:
            return 0.0
        return sum(r.overall_score for r in self.case_results) / len(self.case_results)

    @property
    def best_of_k_score(self) -> float:
        """Average of per-case best scores (best-of-k across trials)."""
        if not self.case_results:
            return 0.0
        return sum(r.best_score for r in self.case_results) / len(self.case_results)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scenario_name": self.scenario_name,
            "run_id": self.run_id,
            "config_hash": self.config_hash,
            "total_cases": self.total_cases,
            "passed_cases": self.passed_cases,
            "failed_cases": self.failed_cases,
            "error_cases": self.error_cases,
            "pass_rate": self.pass_rate,
            "average_score": self.average_score,
            "best_of_k_score": self.best_of_k_score,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "case_results": [
                {
                    "case_id": r.case_id,
                    "task_id": r.case_id,  # Alias for compass analyze compatibility
                    "status": r.status.value,
                    "passed": r.passed,
                    "overall_score": r.overall_score,
                    "best_score": r.best_score,
                    "breakdown": r.breakdown,
                    "duration_ms": r.duration_ms,
                    "tags": r.tags,
                    "category": r.category,
                    "error": r.error,
                    # Full evaluator results for compass analyze
                    "evaluator_results": [er.to_dict() for er in r.evaluator_results],
                    # Alias for compass analyze compatibility
                    "grade_results": [er.to_dict() for er in r.evaluator_results],
                    # Trials information
                    "total_trials": r.total_trials,
                    "passed_trials": r.passed_trials,
                    "trial_metrics": r.trial_metrics,
                }
                for r in self.case_results
            ],
        }
