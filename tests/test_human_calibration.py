"""Tests for calibration module."""

import pytest

from compass.graders.human.calibration import (
    AnchorSample,
    CalibrationConfig,
    CalibrationResult,
    CalibrationSession,
)

# ---------------------------------------------------------------------------
# AnchorSample
# ---------------------------------------------------------------------------


class TestAnchorSample:
    def test_create(self):
        anchor = AnchorSample(
            id="a1",
            description="High quality",
            expected_scores={"composition": 4.5, "color": 4.0},
        )
        assert anchor.id == "a1"
        assert anchor.expected_scores["composition"] == 4.5

    def test_defaults(self):
        anchor = AnchorSample(id="a2", expected_scores={"q": 3.0})
        assert anchor.description == ""
        assert anchor.output_reference == ""
        assert anchor.difficulty == "medium"
        assert anchor.metadata == {}

    def test_difficulty(self):
        anchor = AnchorSample(
            id="a3", expected_scores={"q": 1.0}, difficulty="hard"
        )
        assert anchor.difficulty == "hard"


# ---------------------------------------------------------------------------
# CalibrationConfig
# ---------------------------------------------------------------------------


class TestCalibrationConfig:
    def test_default_disabled(self):
        config = CalibrationConfig()
        assert config.enabled is False
        assert config.anchors == []
        assert config.max_drift == 1.0
        assert config.require_pass is True
        assert config.recalibration_interval == 0
        assert config.min_anchors_before_review == 0

    def test_from_dict(self):
        config = CalibrationConfig.model_validate({
            "enabled": True,
            "max_drift": 0.5,
            "require_pass": False,
        })
        assert config.enabled is True
        assert config.max_drift == 0.5
        assert config.require_pass is False

    def test_with_anchors(self):
        config = CalibrationConfig.model_validate({
            "enabled": True,
            "anchors": [
                {
                    "id": "a1",
                    "expected_scores": {"q": 4.0},
                    "description": "Good sample",
                },
            ],
        })
        assert len(config.anchors) == 1
        assert config.anchors[0].id == "a1"


# ---------------------------------------------------------------------------
# CalibrationSession
# ---------------------------------------------------------------------------


def _make_session(
    max_drift: float = 1.0,
    require_pass: bool = True,
    min_anchors: int = 0,
    recalibration_interval: int = 0,
) -> CalibrationSession:
    """Helper to create a session with two anchors."""
    config = CalibrationConfig(
        enabled=True,
        max_drift=max_drift,
        require_pass=require_pass,
        min_anchors_before_review=min_anchors,
        recalibration_interval=recalibration_interval,
        anchors=[
            AnchorSample(
                id="high",
                expected_scores={"composition": 4.0, "color": 4.0},
            ),
            AnchorSample(
                id="low",
                expected_scores={"composition": 2.0, "color": 2.0},
            ),
        ],
    )
    return CalibrationSession(config)


class TestCalibrationSession:
    def test_present_anchor_returns_first_unscored(self):
        session = _make_session()
        anchor = session.present_anchor("r1")
        assert anchor is not None
        assert anchor.id == "high"

    def test_present_anchor_skips_scored(self):
        session = _make_session()
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        anchor = session.present_anchor("r1")
        assert anchor is not None
        assert anchor.id == "low"

    def test_present_anchor_none_when_all_done(self):
        session = _make_session()
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        assert session.present_anchor("r1") is None

    def test_submit_returns_result_when_all_scored(self):
        session = _make_session()
        result = session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        assert result is None  # Not all anchors scored yet

        result = session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        assert result is not None
        assert result.passed is True
        assert result.overall_drift == 0.0

    def test_calibration_passes_low_drift(self):
        session = _make_session(max_drift=1.0)
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.5, "color": 3.5}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.5, "color": 1.5}
        )
        result = session.check_calibration("r1")
        assert result.passed is True
        # mean(|4.5-4|, |3.5-4|, |2.5-2|, |1.5-2|) = 0.5
        assert result.overall_drift == 0.5

    def test_calibration_fails_high_drift(self):
        session = _make_session(max_drift=0.5)
        session.submit_anchor_response(
            "r1", "high", {"composition": 1.0, "color": 1.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 5.0, "color": 5.0}
        )
        result = session.check_calibration("r1")
        assert result.passed is False
        assert result.overall_drift == 3.0  # mean(|1-4|, |1-4|, |5-2|, |5-2|) = 3.0

    def test_per_criterion_drift(self):
        session = _make_session()
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 3.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 1.0}
        )
        result = session.check_calibration("r1")
        assert result.per_criterion_drift["composition"] == 0.0  # exact match
        assert result.per_criterion_drift["color"] == pytest.approx(1.0)  # mean(|3-4|, |1-2|) = 1.0

    def test_min_anchors_before_review(self):
        session = _make_session(min_anchors=3)  # Only 2 anchors available
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        result = session.check_calibration("r1")
        # Only 2 responses but need 3 → fails
        assert result.passed is False

    def test_recalibration_interval(self):
        session = _make_session(recalibration_interval=5)
        assert session.should_recalibrate("r1") is False
        for _ in range(4):
            session.record_review("r1")
        assert session.should_recalibrate("r1") is False
        session.record_review("r1")
        assert session.should_recalibrate("r1") is True

    def test_multiple_reviewers_independent(self):
        session = _make_session()
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        # r2 should still see "high" as first anchor
        anchor = session.present_anchor("r2")
        assert anchor is not None
        assert anchor.id == "high"

    def test_cache_invalidation(self):
        session = _make_session()
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        result1 = session.check_calibration("r1")
        result2 = session.check_calibration("r1")
        # Should be cached (same object)
        assert result1 is result2

    def test_no_responses_fails(self):
        session = _make_session()
        result = session.check_calibration("r1")
        assert result.passed is False
        assert result.overall_drift == float("inf")

    def test_invalid_anchor_id_returns_none(self):
        session = _make_session()
        result = session.submit_anchor_response(
            "r1", "nonexistent", {"composition": 4.0}
        )
        assert result is None

    def test_is_reviewer_calibrated(self):
        session = _make_session(max_drift=1.0)
        assert session.is_reviewer_calibrated("r1") is False
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        assert session.is_reviewer_calibrated("r1") is False  # Not all scored
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        assert session.is_reviewer_calibrated("r1") is True

    def test_reset_recalibration(self):
        session = _make_session(recalibration_interval=2)
        session.submit_anchor_response(
            "r1", "high", {"composition": 4.0, "color": 4.0}
        )
        session.submit_anchor_response(
            "r1", "low", {"composition": 2.0, "color": 2.0}
        )
        assert session.is_reviewer_calibrated("r1") is True
        session.reset_recalibration("r1")
        assert session.is_reviewer_calibrated("r1") is False


# ---------------------------------------------------------------------------
# CalibrationResult
# ---------------------------------------------------------------------------


class TestCalibrationResult:
    def test_to_dict(self):
        result = CalibrationResult(
            reviewer_id="r1",
            passed=True,
            overall_drift=0.3,
            per_criterion_drift={"q": 0.3},
            per_anchor_drift={"a1": 0.3},
        )
        d = result.to_dict()
        assert d["reviewer_id"] == "r1"
        assert d["passed"] is True
        assert d["overall_drift"] == 0.3

    def test_failed_result(self):
        result = CalibrationResult(
            reviewer_id="r2",
            passed=False,
            overall_drift=2.5,
            per_criterion_drift={},
            per_anchor_drift={},
        )
        assert result.passed is False
        assert result.overall_drift == 2.5
