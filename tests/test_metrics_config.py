"""Tests that the case ``metrics:`` config actually drives metric reporting.

Regression coverage for the previously-dead ``MetricsConfig``: ``pass_at_k`` and
``consistency`` were defined on the scenario/case model but never consumed —
``TrialMetrics.to_dict`` hardcoded pass_at_1/3/5 and always emitted consistency.
"""

import pytest

from compass.core.metrics import TrialMetrics


class TestToDictDrivenByConfig:
    def _metrics(self) -> TrialMetrics:
        # pass_rate = 2/4 = 0.5
        return TrialMetrics(
            total_trials=4,
            passed_trials=2,
            scores=[1.0, 0.0, 1.0, 0.0],
        )

    def test_default_reports_1_3_5(self):
        """With no config, falls back to k = [1, 3, 5]."""
        d = self._metrics().to_dict()
        for k in (1, 3, 5):
            assert f"pass_at_{k}" in d
            assert f"pass_all_{k}" in d
        assert "consistency" in d

    def test_custom_pass_at_k_is_honoured(self):
        """A custom k list drives exactly which pass@k/pass^k keys appear.

        Values use the unbiased estimator (k <= n = 4, c = 2):
        pass@1 = 2/4, pass@2 = 1 - C(2,2)/C(4,2) = 5/6, pass^2 = C(2,2)/C(4,2) = 1/6.
        """
        d = self._metrics().to_dict(pass_at_k=[1, 2])

        assert d["pass_at_1"] == pytest.approx(0.5)
        assert d["pass_at_2"] == pytest.approx(5 / 6)
        assert d["pass_all_2"] == pytest.approx(1 / 6)

        # Keys outside the requested list must NOT be emitted.
        assert "pass_at_3" not in d
        assert "pass_at_5" not in d
        assert "pass_all_5" not in d

    def test_consistency_can_be_disabled(self):
        """``metrics.consistency: false`` omits the consistency score."""
        d = self._metrics().to_dict(include_consistency=False)
        assert "consistency" not in d
        # Core metrics are still present.
        assert "pass_rate" in d
        assert "best_of_k" in d
