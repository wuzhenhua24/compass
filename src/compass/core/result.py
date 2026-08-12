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
    """Result from a single evaluator.

    Two distinctions the aggregate depends on, both core-owned:

    - ``skipped``: the grader never ran (short-circuit). It contributes nothing
      and does not hold the case back.
    - ``score is None``: the grader ran but produced no measurement (it crashed,
      timed out, ...). "Not measured" is not "measured as zero", so an unscored
      grader is excluded from the score denominator — but it *does* fail the
      case, because a pass cannot be certified from an incomplete evaluation.

    Both live in dedicated fields rather than inside ``metadata``: ``metadata``
    carries the grader's own ``details`` payload, so anything the core reads
    from it could be clobbered by a user grader.
    """

    name: str
    score: float | None
    passed: bool
    # ``GraderConfig.label`` — a name for this *instance*, when one case runs
    # the same grader more than once. Empty when the scenario set none.
    label: str = ""
    weight: float = 1.0
    required: bool = False  # Whether this grader must pass for overall pass
    gate: bool = False  # Hard gate: must pass, excluded from score
    grader_type: str = "code"  # "code", "model", or "human"
    grader_scope: str = "outcome"  # "outcome", "transcript", or "both"
    grader_version: str = ""  # Verifier version that produced this score
    # Core-owned control flags — never read these from `metadata`
    skipped: bool = False
    skip_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    # Neutral observations about what was seen, emitted pass or fail.
    # Presence-only: an absent tag means "not observed", not "false".
    tags: list[str] = field(default_factory=list)
    # Quantitative observations: numbers aggregate as mean ± stderr,
    # booleans as rates. Core-owned, like tags — the aggregate reads them.
    metrics: dict[str, Any] = field(default_factory=dict)
    failure_tags: list[str] = field(default_factory=list)  # Structured failure labels
    error: str | None = None

    @property
    def scored(self) -> bool:
        """Whether this grader produced an actual measurement."""
        return self.score is not None

    @property
    def weighted_score(self) -> float:
        """Weighted score; 0.0 when unscored (callers should filter on ``scored``)."""
        if self.score is None:
            return 0.0
        return self.score * self.weight

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "label": self.label,
            "weight": self.weight,
            "weighted_score": self.weighted_score,
            "required": self.required,
            "gate": self.gate,
            "grader_type": self.grader_type,
            "grader_scope": self.grader_scope,
            "grader_version": self.grader_version,
            "scored": self.scored,
            "skipped": self.skipped,
            "skip_reason": self.skip_reason,
            "metadata": self.metadata,
            "tags": self.tags,
            "metrics": self.metrics,
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
    error_trials: int = 0  # Trials lost to harness errors, excluded from metrics
    trial_metrics: dict[str, Any] | None = None  # pass@k, pass^k, pass_rate, etc.

    #: Per-grader aggregate **across trials**, keyed by ``label or name``.
    #:
    #: ``evaluator_results`` deliberately holds one trial's detail — a merged
    #: ``metadata`` payload would be meaningless, and the last evaluated trial
    #: is the honest thing to show. But that makes it the wrong input for
    #: comparing runs: on a 3-trial case it samples one attempt, so a case
    #: whose trials took 38/27/26 turns reports 26 as though it were the
    #: figure. ``overall_score`` has always been the mean across trials; this
    #: gives every grader's score and metrics the same treatment.
    #:
    #: Each entry: ``{"score_mean", "pass_fraction", "trials", "metrics"}``.
    #: ``trials`` is the number of trials that actually measured that grader,
    #: which is its denominator — a grader that ran twice out of three is a
    #: mean over two, not a mean with a zero in it.
    grader_summary: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Audit provenance: content hash of the grading contract applied to this
    # case (graders + aggregation + leak markers + expect). Automatic, so a
    # score is only comparable across runs when this matches — no reliance on a
    # hand-maintained Grader.version.
    grader_fingerprint: str = ""

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
        """Score breakdown by evaluator (unscored/skipped graders omitted).

        Keyed on ``label`` when the scenario set one, else on the grader name.
        A case may legitimately run one grader twice — hidden acceptance tests
        and the repository's own suite are both ``integration_test`` — and a
        plain name-keyed dict silently kept only the last of them, which is
        data loss in the serialized result, not just a display quirk. Unlabelled
        repeats therefore get a ``#2``, ``#3`` suffix in declaration order; the
        first keeps the bare name, so output is unchanged for the overwhelmingly
        common case of one grader per name.

        Prefer a ``label`` to relying on the suffix: the suffix moves if the
        grader order changes, which makes it a poor key to compare runs on.
        """
        scored = [r for r in self.evaluator_results if r.scored and not r.skipped]
        seen: dict[str, int] = {}
        out: dict[str, float] = {}
        for r in scored:
            key = r.label or r.name
            if not r.label:
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > 1:
                    key = f"{key}#{seen[key]}"
            out[key] = r.weighted_score
        return out

    @property
    def observed_tags(self) -> list[str]:
        """Union of what this case's graders observed.

        Distinct from ``tags``, which is the static classification you wrote in
        the scenario YAML: these are emitted at grading time and describe what
        actually happened in *this* run. Derived rather than stored, so it can
        never drift from the grader results it summarizes.
        """
        return sorted({t for r in self.evaluator_results for t in r.tags})


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
    def evaluated_cases(self) -> int:
        """Cases that produced evidence about the agent.

        Excludes ERROR cases: a network drop or a crashed adapter is a harness
        failure, not the agent getting it wrong. Counting them as failures
        would let infrastructure flakiness masquerade as a quality regression.
        """
        return self.total_cases - self.error_cases

    @property
    def _evaluated_results(self) -> list[CaseResult]:
        return [r for r in self.case_results if r.status != TestStatus.ERROR]

    @property
    def pass_rate(self) -> float:
        """Pass rate over evaluated cases (harness errors excluded)."""
        if self.evaluated_cases <= 0:
            return 0.0
        return self.passed_cases / self.evaluated_cases

    @property
    def average_score(self) -> float:
        """Mean case score over evaluated cases (harness errors excluded)."""
        evaluated = self._evaluated_results
        if not evaluated:
            return 0.0
        return sum(r.overall_score for r in evaluated) / len(evaluated)

    @property
    def best_of_k_score(self) -> float:
        """Mean per-case best score over evaluated cases (best-of-k across trials)."""
        evaluated = self._evaluated_results
        if not evaluated:
            return 0.0
        return sum(r.best_score for r in evaluated) / len(evaluated)

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
            # Denominator behind pass_rate — harness errors are excluded
            "evaluated_cases": self.evaluated_cases,
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
                    "observed_tags": r.observed_tags,
                    "category": r.category,
                    "error": r.error,
                    "grader_fingerprint": r.grader_fingerprint,
                    # Full evaluator results for compass analyze
                    "evaluator_results": [er.to_dict() for er in r.evaluator_results],
                    # Alias for compass analyze compatibility
                    "grade_results": [er.to_dict() for er in r.evaluator_results],
                    # Trials information
                    "total_trials": r.total_trials,
                    "passed_trials": r.passed_trials,
                    "error_trials": r.error_trials,
                    "trial_metrics": r.trial_metrics,
                    "grader_summary": r.grader_summary,
                }
                for r in self.case_results
            ],
        }
