"""Tests for inter-rater agreement module."""

import pytest

from compass.graders.human.agreement import (
    AgreementReport,
    RaterAgreement,
    compute_cohens_kappa,
    compute_krippendorffs_alpha,
)

# ---------------------------------------------------------------------------
# Cohen's Kappa
# ---------------------------------------------------------------------------


class TestCohensKappa:
    def test_perfect_agreement(self):
        scores = [1.0, 2.0, 3.0, 4.0, 5.0]
        kappa = compute_cohens_kappa(scores, scores)
        assert kappa == pytest.approx(1.0)

    def test_no_agreement(self):
        r1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        r2 = [5.0, 4.0, 3.0, 2.0, 1.0]
        kappa = compute_cohens_kappa(r1, r2)
        assert kappa is not None
        # Only item index 2 agrees (3.0 == 3.0)
        # p_observed = 1/5 = 0.2, kappa < 1
        assert kappa < 0.5

    def test_partial_agreement(self):
        r1 = [1.0, 2.0, 3.0, 4.0]
        r2 = [1.0, 2.0, 4.0, 3.0]
        kappa = compute_cohens_kappa(r1, r2)
        assert kappa is not None
        assert 0.0 < kappa < 1.0

    def test_tolerance(self):
        r1 = [1.0, 2.0, 3.0]
        r2 = [1.5, 2.5, 3.5]
        # Without tolerance: no agreement
        kappa_strict = compute_cohens_kappa(r1, r2, tolerance=0.0)
        # With tolerance=0.5: all agree
        kappa_tolerant = compute_cohens_kappa(r1, r2, tolerance=0.5)
        assert kappa_strict is not None
        assert kappa_tolerant is not None
        assert kappa_tolerant > kappa_strict

    def test_insufficient_data_empty(self):
        assert compute_cohens_kappa([], []) is None

    def test_insufficient_data_mismatched(self):
        assert compute_cohens_kappa([1.0, 2.0], [1.0]) is None

    def test_all_same_scores(self):
        r1 = [3.0, 3.0, 3.0]
        r2 = [3.0, 3.0, 3.0]
        kappa = compute_cohens_kappa(r1, r2)
        # All same → p_expected = 1.0, special case
        assert kappa == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Krippendorff's Alpha
# ---------------------------------------------------------------------------


class TestKrippendorffsAlpha:
    def test_perfect_agreement(self):
        data = {
            "r1": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
            "r2": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
        }
        alpha = compute_krippendorffs_alpha(data)
        assert alpha == pytest.approx(1.0)

    def test_no_agreement(self):
        data = {
            "r1": {"i1": 1.0, "i2": 5.0},
            "r2": {"i1": 5.0, "i2": 1.0},
        }
        alpha = compute_krippendorffs_alpha(data)
        assert alpha is not None
        assert alpha < 0.5

    def test_three_raters(self):
        data = {
            "r1": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
            "r2": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
            "r3": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
        }
        alpha = compute_krippendorffs_alpha(data)
        assert alpha == pytest.approx(1.0)

    def test_missing_data(self):
        """Some raters skip some items — alpha should still compute."""
        data = {
            "r1": {"i1": 4.0, "i2": 2.0, "i3": 5.0},
            "r2": {"i1": 4.0, "i3": 5.0},  # Skipped i2
            "r3": {"i2": 2.0, "i3": 5.0},  # Skipped i1
        }
        alpha = compute_krippendorffs_alpha(data)
        assert alpha is not None
        # High agreement on shared items
        assert alpha > 0.5

    def test_all_same_scores(self):
        data = {
            "r1": {"i1": 3.0, "i2": 3.0},
            "r2": {"i1": 3.0, "i2": 3.0},
        }
        alpha = compute_krippendorffs_alpha(data)
        # All scores identical → D_expected = 0, D_observed = 0 → alpha = 1.0
        assert alpha == pytest.approx(1.0)

    def test_insufficient_data_one_rater(self):
        data = {"r1": {"i1": 4.0}}
        assert compute_krippendorffs_alpha(data) is None

    def test_insufficient_data_no_overlap(self):
        data = {
            "r1": {"i1": 4.0},
            "r2": {"i2": 3.0},
        }
        # No items with 2+ ratings
        assert compute_krippendorffs_alpha(data) is None


# ---------------------------------------------------------------------------
# RaterAgreement
# ---------------------------------------------------------------------------


class TestRaterAgreement:
    def test_add_score(self):
        ra = RaterAgreement()
        ra.add_score("r1", "i1", 4.0)
        ra.add_score("r2", "i1", 3.0)
        report = ra.compute()
        assert report.num_raters == 2
        assert report.num_items == 1

    def test_compute_two_raters(self):
        ra = RaterAgreement()
        for item, s1, s2 in [("i1", 4.0, 4.0), ("i2", 2.0, 2.0), ("i3", 5.0, 5.0)]:
            ra.add_score("r1", item, s1)
            ra.add_score("r2", item, s2)
        report = ra.compute()
        assert report.cohens_kappa == pytest.approx(1.0)
        assert report.krippendorffs_alpha == pytest.approx(1.0)
        assert report.pairwise_agreement == pytest.approx(1.0)

    def test_compute_three_raters(self):
        ra = RaterAgreement()
        for item in ["i1", "i2", "i3"]:
            ra.add_score("r1", item, 4.0)
            ra.add_score("r2", item, 4.0)
            ra.add_score("r3", item, 4.0)
        report = ra.compute()
        # 3 raters → no Cohen's Kappa
        assert report.cohens_kappa is None
        assert report.krippendorffs_alpha == pytest.approx(1.0)

    def test_multi_criterion(self):
        ra = RaterAgreement()
        ra.add_score("r1", "i1", 4.0, criterion="quality")
        ra.add_score("r2", "i1", 4.0, criterion="quality")
        ra.add_score("r1", "i1", 3.0, criterion="style")
        ra.add_score("r2", "i1", 1.0, criterion="style")
        report = ra.compute()
        assert "quality" in report.per_criterion
        assert "style" in report.per_criterion

    def test_insufficient_raters(self):
        ra = RaterAgreement()
        ra.add_score("r1", "i1", 4.0)
        report = ra.compute()
        assert report.num_raters == 1
        assert report.cohens_kappa is None
        assert report.krippendorffs_alpha is None

    def test_tolerance(self):
        ra = RaterAgreement(tolerance=0.5)
        ra.add_score("r1", "i1", 4.0)
        ra.add_score("r2", "i1", 4.3)
        ra.add_score("r1", "i2", 2.0)
        ra.add_score("r2", "i2", 2.4)
        report = ra.compute()
        # Within tolerance → high agreement
        assert report.pairwise_agreement == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# AgreementReport
# ---------------------------------------------------------------------------


class TestAgreementReport:
    def test_to_dict(self):
        report = AgreementReport(
            num_raters=2,
            num_items=5,
            cohens_kappa=0.75,
            krippendorffs_alpha=0.8,
            pairwise_agreement=0.9,
        )
        d = report.to_dict()
        assert d["num_raters"] == 2
        assert d["cohens_kappa"] == 0.75
        assert "interpretation" in d

    def test_interpretation_poor(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=-0.1)
        assert report.interpretation == "poor"

    def test_interpretation_slight(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=0.1)
        assert report.interpretation == "slight"

    def test_interpretation_fair(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=0.3)
        assert report.interpretation == "fair"

    def test_interpretation_moderate(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=0.5)
        assert report.interpretation == "moderate"

    def test_interpretation_substantial(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=0.7)
        assert report.interpretation == "substantial"

    def test_interpretation_almost_perfect(self):
        report = AgreementReport(num_raters=2, num_items=3, krippendorffs_alpha=0.9)
        assert report.interpretation == "almost_perfect"

    def test_interpretation_insufficient_data(self):
        report = AgreementReport(num_raters=1, num_items=0)
        assert report.interpretation == "insufficient_data"

    def test_interpretation_uses_kappa_when_no_alpha(self):
        report = AgreementReport(
            num_raters=2, num_items=3,
            cohens_kappa=0.85, krippendorffs_alpha=None,
        )
        assert report.interpretation == "almost_perfect"
