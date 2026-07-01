"""Score aggregation strategies."""

from enum import Enum
from typing import Any

from compass.core.result import EvaluatorResult
from compass.core.scenario import AggregationConfig


class AggregationStrategy(Enum):
    """Available aggregation strategies."""

    WEIGHTED_SUM = "weighted_sum"
    MIN_PASS = "min_pass"
    MAJORITY_VOTE = "majority_vote"
    CUSTOM = "custom"


class Aggregator:
    """Aggregates scores from multiple evaluators."""

    def __init__(self, config: AggregationConfig):
        """Initialize aggregator with config.

        Args:
            config: Aggregation configuration.
        """
        self.config = config
        self.strategy = AggregationStrategy(config.method)

    def aggregate(self, results: list[EvaluatorResult]) -> tuple[float, bool]:
        """Aggregate evaluator results.

        Args:
            results: List of evaluator results.

        Returns:
            Tuple of (overall_score, passed).
        """
        if not results:
            return 0.0, False

        # Check required evaluators first
        for result in results:
            if result.name in self.config.required_graders and not result.passed:
                return self._calculate_score(results), False

        match self.strategy:
            case AggregationStrategy.WEIGHTED_SUM:
                return self._weighted_sum(results)
            case AggregationStrategy.MIN_PASS:
                return self._min_pass(results)
            case AggregationStrategy.MAJORITY_VOTE:
                return self._majority_vote(results)
            case AggregationStrategy.CUSTOM:
                return self._custom(results)
            case _:
                return self._weighted_sum(results)

    def _calculate_score(self, results: list[EvaluatorResult]) -> float:
        """Calculate weighted score."""
        total_weight = sum(r.weight for r in results if r.weight > 0)
        if total_weight == 0:
            return 0.0
        return sum(r.weighted_score for r in results) / total_weight

    def _weighted_sum(self, results: list[EvaluatorResult]) -> tuple[float, bool]:
        """Weighted sum aggregation."""
        score = self._calculate_score(results)
        passed = score >= self.config.pass_threshold
        return score, passed

    def _min_pass(self, results: list[EvaluatorResult]) -> tuple[float, bool]:
        """All evaluators must pass."""
        score = self._calculate_score(results)
        passed = all(r.passed for r in results)
        return score, passed

    def _majority_vote(self, results: list[EvaluatorResult]) -> tuple[float, bool]:
        """Majority of evaluators must pass."""
        score = self._calculate_score(results)
        passed_count = sum(1 for r in results if r.passed)
        passed = passed_count > len(results) / 2
        return score, passed

    def _custom(self, results: list[EvaluatorResult]) -> tuple[float, bool]:
        """Custom aggregation - placeholder for user-defined logic."""
        # TODO: Support custom formula from config
        return self._weighted_sum(results)
