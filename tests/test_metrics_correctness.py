"""Regression coverage for low-severity metric correctness fixes.

- consistency_score: an all-identical run (including all-fail zeros) is perfectly
  consistent, not maximally inconsistent.
- pass_at_k / pass_all_k: use the unbiased estimator for k <= n and honestly
  extrapolate for k > n instead of silently clamping k (which mislabelled the
  pass@n value as pass@k).
"""

import math

import pytest

from compass.core.metrics import (
    TrialMetrics,
    estimate_pass_all_k,
    estimate_pass_at_k,
)


class TestConsistencyScore:
    def test_all_fail_identically_is_consistent(self):
        """Every trial scores 0.0 → zero variance → perfectly consistent (1.0),
        not 0.0 as the old mean==0 branch returned."""
        m = TrialMetrics(total_trials=3, passed_trials=0, scores=[0.0, 0.0, 0.0])
        assert m.consistency_score() == 1.0

    def test_all_pass_identically_is_consistent(self):
        m = TrialMetrics(total_trials=3, passed_trials=3, scores=[1.0, 1.0, 1.0])
        assert m.consistency_score() == 1.0

    def test_varied_scores_are_less_consistent(self):
        m = TrialMetrics(total_trials=2, passed_trials=1, scores=[1.0, 0.0])
        c = m.consistency_score()
        assert 0.0 < c < 1.0


class TestPassAtKUnbiased:
    """k <= n uses the unbiased estimator (previously the dead code path)."""

    def _m(self) -> TrialMetrics:
        # n = 4, c = 2, p = 0.5
        return TrialMetrics(total_trials=4, passed_trials=2, scores=[1.0, 0.0, 1.0, 0.0])

    def test_pass_at_k_matches_unbiased_estimator(self):
        m = self._m()
        assert m.pass_at_k(1) == pytest.approx(estimate_pass_at_k(4, 2, 1))
        assert m.pass_at_k(2) == pytest.approx(estimate_pass_at_k(4, 2, 2))
        assert m.pass_at_k(2) == pytest.approx(5 / 6)  # not the naive 0.75

    def test_pass_all_k_matches_unbiased_estimator(self):
        m = self._m()
        assert m.pass_all_k(2) == pytest.approx(estimate_pass_all_k(4, 2, 2))
        assert m.pass_all_k(2) == pytest.approx(1 / 6)  # not the naive 0.25


class TestPassAtKExtrapolationNoClamp:
    """k > n extrapolates from the observed rate rather than clamping to n."""

    def _m(self) -> TrialMetrics:
        # n = 4, c = 2, p = 0.5
        return TrialMetrics(total_trials=4, passed_trials=2, scores=[1.0, 0.0, 1.0, 0.0])

    def test_pass_at_k_extrapolates_beyond_n(self):
        m = self._m()
        # Honest projection 1 - (1 - p)^5, NOT the clamped pass@4 value (which is
        # 1.0 here because n - c < 4).
        assert m.pass_at_k(5) == pytest.approx(1 - math.pow(0.5, 5))
        assert m.pass_at_k(5) < 1.0
        assert m.pass_at_k(5) != pytest.approx(m.pass_at_k(4))

    def test_pass_all_k_extrapolates_beyond_n(self):
        m = self._m()
        # p^5, NOT the clamped pass^4 value (0.0 here because c < 4).
        assert m.pass_all_k(5) == pytest.approx(math.pow(0.5, 5))
        assert m.pass_all_k(5) > 0.0
        assert m.pass_all_k(4) == 0.0


class TestEstimatePassAllK:
    def test_basic(self):
        assert estimate_pass_all_k(4, 2, 2) == pytest.approx(1 / 6)
        assert estimate_pass_all_k(3, 3, 2) == pytest.approx(1.0)

    def test_fewer_correct_than_k_is_zero(self):
        assert estimate_pass_all_k(3, 1, 2) == 0.0

    def test_k_exceeds_n_is_zero(self):
        assert estimate_pass_all_k(2, 2, 5) == 0.0

    def test_nonpositive_k_is_zero(self):
        assert estimate_pass_all_k(3, 3, 0) == 0.0
