"""Metrics calculation for trial-based evaluation."""

import math
import statistics
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class TrialMetrics:
    """Metrics calculated from multiple trial results."""

    total_trials: int
    passed_trials: int
    scores: list[float] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        """Simple pass rate = passed / total."""
        if self.total_trials == 0:
            return 0.0
        return self.passed_trials / self.total_trials

    @property
    def score_mean(self) -> float:
        """Mean score across all trials."""
        if not self.scores:
            return 0.0
        return statistics.mean(self.scores)

    @property
    def score_std(self) -> float:
        """Standard deviation of scores."""
        if len(self.scores) < 2:
            return 0.0
        return statistics.stdev(self.scores)

    @property
    def score_min(self) -> float:
        """Minimum score."""
        if not self.scores:
            return 0.0
        return min(self.scores)

    @property
    def score_max(self) -> float:
        """Maximum score."""
        if not self.scores:
            return 0.0
        return max(self.scores)

    def pass_at_k(self, k: int) -> float:
        """Calculate pass@k: probability of at least 1 success in k attempts.

        For ``k <= total_trials`` this uses the unbiased Codex estimator
        (``estimate_pass_at_k``) over the observed samples, which is more
        accurate than the naive ``1 - (1 - p)^k``. For ``k > total_trials`` there
        aren't enough samples to estimate directly, so it extrapolates from the
        observed pass rate with ``1 - (1 - p)^k`` — an honest projection rather
        than silently clamping ``k`` down (which would report the pass@n value
        under a ``pass@k`` label).

        Args:
            k: Number of attempts.

        Returns:
            Probability of at least one success.
        """
        if k <= 0 or self.total_trials == 0:
            return 0.0
        if k <= self.total_trials:
            return estimate_pass_at_k(self.total_trials, self.passed_trials, k)

        p = self.pass_rate
        return 1.0 - math.pow(1.0 - p, k)

    def pass_all_k(self, k: int) -> float:
        """Calculate pass^k: probability of all k attempts succeeding.

        For ``k <= total_trials`` this uses the unbiased combinatorial estimator
        (``estimate_pass_all_k``) — the probability that a random k-subset of the
        observed trials is all passes. For ``k > total_trials`` it extrapolates
        from the observed pass rate with ``p^k`` instead of clamping ``k`` (which
        would mislabel the pass^n value as pass^k).

        Args:
            k: Number of attempts.

        Returns:
            Probability of all attempts succeeding.
        """
        if k <= 0 or self.total_trials == 0:
            return 0.0
        if k <= self.total_trials:
            return estimate_pass_all_k(self.total_trials, self.passed_trials, k)

        p = self.pass_rate
        return math.pow(p, k)

    def consistency_score(self) -> float:
        """Calculate consistency of results across trials.

        Returns a score from 0 to 1 where:
        - 1.0 = all trials had identical outcomes
        - 0.0 = maximum variance in outcomes

        Based on normalized inverse of score variance.
        """
        if len(self.scores) < 2:
            return 1.0  # Single trial is perfectly consistent

        std = self.score_std
        if std == 0:
            # All trials produced the identical score (including an all-fail run
            # of zeros): zero variance means perfectly consistent, not chaotic.
            return 1.0

        # Use coefficient of variation (CV) for consistency
        mean = self.score_mean
        if mean == 0:
            # Non-zero spread around a zero mean: CV is undefined → treat as
            # maximally inconsistent.
            return 0.0

        cv = std / mean

        # Convert CV to consistency score (lower CV = higher consistency)
        # Using exponential decay: consistency = e^(-cv)
        return math.exp(-cv)

    def best_of_k(self, k: int) -> float:
        """Best (maximum) score among the first k trials.

        Stripe's evaluation model: run k times, take the highest score.
        This rewards agents that can succeed at least once.

        Args:
            k: Number of attempts to consider.

        Returns:
            Maximum score among the first k trials.
        """
        if not self.scores or k <= 0:
            return 0.0
        k = min(k, len(self.scores))
        return max(self.scores[:k])

    def to_dict(
        self,
        pass_at_k: list[int] | None = None,
        include_consistency: bool = True,
    ) -> dict:
        """Convert to a metrics dictionary.

        Args:
            pass_at_k: Which ``k`` values to report ``pass@k``/``pass^k`` for.
                Driven by the case/scenario ``metrics.pass_at_k`` config;
                falls back to ``[1, 3, 5]`` when not provided.
            include_consistency: Whether to include the consistency score
                (honours the ``metrics.consistency`` config flag).
        """
        ks = pass_at_k if pass_at_k else [1, 3, 5]
        result = {
            "total_trials": self.total_trials,
            "passed_trials": self.passed_trials,
            "pass_rate": self.pass_rate,
            "score_mean": self.score_mean,
            "score_std": self.score_std,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "best_of_k": self.best_of_k(self.total_trials),
        }
        for k in ks:
            result[f"pass_at_{k}"] = self.pass_at_k(k)
            result[f"pass_all_{k}"] = self.pass_all_k(k)
        if include_consistency:
            result["consistency"] = self.consistency_score()
        return result


def calculate_metrics(
    passed_list: Sequence[bool],
    scores: Sequence[float],
) -> TrialMetrics:
    """Calculate metrics from trial results.

    Args:
        passed_list: List of pass/fail results for each trial.
        scores: List of scores for each trial.

    Returns:
        TrialMetrics object with calculated statistics.
    """
    return TrialMetrics(
        total_trials=len(passed_list),
        passed_trials=sum(passed_list),
        scores=list(scores),
    )


def estimate_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass@k.

    This is the unbiased estimator from the Codex paper:
    pass@k = 1 - C(n-c, k) / C(n, k)

    where n = total samples, c = correct samples, k = k value.

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k value for pass@k.

    Returns:
        Estimated pass@k probability.
    """
    if n - c < k:
        return 1.0

    # Calculate using log to avoid overflow
    # C(n-c, k) / C(n, k) = product((n-c-i)/(n-i) for i in range(k))
    result = 1.0
    for i in range(k):
        result *= (n - c - i) / (n - i)

    return 1.0 - result


def estimate_pass_all_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator for pass^k (all of a random k-subset pass).

    pass^k = C(c, k) / C(n, k)

    the probability that k samples drawn without replacement from n samples
    (of which c passed) are all passes. This mirrors ``estimate_pass_at_k`` and
    is only defined for ``k <= n``.

    Args:
        n: Total number of samples.
        c: Number of correct samples.
        k: k value for pass^k.

    Returns:
        Estimated pass^k probability (0.0 when fewer than k samples passed).
    """
    if k <= 0 or k > n:
        return 0.0
    if c < k:
        return 0.0

    # C(c, k) / C(n, k) = product((c - i) / (n - i) for i in range(k))
    result = 1.0
    for i in range(k):
        result *= (c - i) / (n - i)

    return result
