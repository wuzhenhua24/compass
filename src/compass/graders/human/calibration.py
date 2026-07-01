"""Calibration module for human review graders.

Provides anchor-based calibration to ensure reviewer consistency:
- AnchorSample: reference items with known expected scores
- CalibrationSession: manages per-reviewer calibration state
- Drift detection: measures deviation from expected scores
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel


class AnchorSample(BaseModel):
    """A reference sample with known expected scores for calibration."""

    id: str
    description: str = ""
    output_reference: str = ""
    expected_scores: dict[str, float]
    difficulty: str = "medium"
    metadata: dict[str, Any] = {}


class CalibrationConfig(BaseModel):
    """Configuration for reviewer calibration."""

    enabled: bool = False
    anchors: list[AnchorSample] = []
    max_drift: float = 1.0
    require_pass: bool = True
    recalibration_interval: int = 0
    min_anchors_before_review: int = 0


@dataclass
class AnchorResponse:
    """A reviewer's scores for a single anchor sample."""

    anchor_id: str
    reviewer_id: str
    scores: dict[str, float]
    drift: float = 0.0


@dataclass
class CalibrationResult:
    """Result of a calibration check for a reviewer."""

    reviewer_id: str
    passed: bool
    overall_drift: float
    per_criterion_drift: dict[str, float]
    per_anchor_drift: dict[str, float]
    anchor_responses: list[AnchorResponse] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reviewer_id": self.reviewer_id,
            "passed": self.passed,
            "overall_drift": self.overall_drift,
            "per_criterion_drift": self.per_criterion_drift,
            "per_anchor_drift": self.per_anchor_drift,
            "anchor_responses": [
                {
                    "anchor_id": r.anchor_id,
                    "reviewer_id": r.reviewer_id,
                    "scores": r.scores,
                    "drift": r.drift,
                }
                for r in self.anchor_responses
            ],
        }


class CalibrationSession:
    """Manages calibration state for multiple reviewers.

    Each reviewer must score anchor samples before starting formal reviews.
    The session tracks responses, computes drift, and determines calibration status.
    """

    def __init__(self, config: CalibrationConfig) -> None:
        self.config = config
        # reviewer_id -> list of AnchorResponse
        self._responses: dict[str, list[AnchorResponse]] = {}
        # reviewer_id -> CalibrationResult (cached)
        self._calibration_cache: dict[str, CalibrationResult] = {}
        # reviewer_id -> count of formal reviews since last calibration check
        self._review_counts: dict[str, int] = {}

    def present_anchor(self, reviewer_id: str) -> AnchorSample | None:
        """Return the next unscored anchor for this reviewer, or None if all done."""
        scored_ids = {
            r.anchor_id for r in self._responses.get(reviewer_id, [])
        }
        for anchor in self.config.anchors:
            if anchor.id not in scored_ids:
                return anchor
        return None

    def submit_anchor_response(
        self,
        reviewer_id: str,
        anchor_id: str,
        scores: dict[str, float],
    ) -> CalibrationResult | None:
        """Record scores for an anchor; return CalibrationResult when all done."""
        anchor = self._get_anchor(anchor_id)
        if anchor is None:
            return None

        # Compute per-anchor drift
        drifts = []
        for criterion, expected in anchor.expected_scores.items():
            actual = scores.get(criterion)
            if actual is not None:
                drifts.append(abs(actual - expected))
        anchor_drift = sum(drifts) / len(drifts) if drifts else 0.0

        response = AnchorResponse(
            anchor_id=anchor_id,
            reviewer_id=reviewer_id,
            scores=scores,
            drift=anchor_drift,
        )

        if reviewer_id not in self._responses:
            self._responses[reviewer_id] = []
        self._responses[reviewer_id].append(response)

        # Invalidate cache
        self._calibration_cache.pop(reviewer_id, None)

        # If all anchors scored, compute and return result
        if self._all_anchors_scored(reviewer_id):
            return self.check_calibration(reviewer_id)
        return None

    def check_calibration(self, reviewer_id: str) -> CalibrationResult:
        """Compute calibration result for a reviewer."""
        if reviewer_id in self._calibration_cache:
            return self._calibration_cache[reviewer_id]

        responses = self._responses.get(reviewer_id, [])
        if not responses:
            result = CalibrationResult(
                reviewer_id=reviewer_id,
                passed=False,
                overall_drift=float("inf"),
                per_criterion_drift={},
                per_anchor_drift={},
                anchor_responses=[],
            )
            return result

        # Per-anchor drift
        per_anchor_drift: dict[str, float] = {}
        for resp in responses:
            per_anchor_drift[resp.anchor_id] = resp.drift

        # Per-criterion drift
        criterion_drifts: dict[str, list[float]] = {}
        for resp in responses:
            anchor = self._get_anchor(resp.anchor_id)
            if anchor is None:
                continue
            for criterion, expected in anchor.expected_scores.items():
                actual = resp.scores.get(criterion)
                if actual is not None:
                    if criterion not in criterion_drifts:
                        criterion_drifts[criterion] = []
                    criterion_drifts[criterion].append(abs(actual - expected))

        per_criterion_drift = {
            c: sum(ds) / len(ds) for c, ds in criterion_drifts.items() if ds
        }

        # Overall drift = mean of all individual drifts
        all_drifts = [d for ds in criterion_drifts.values() for d in ds]
        overall_drift = sum(all_drifts) / len(all_drifts) if all_drifts else 0.0

        passed = overall_drift <= self.config.max_drift

        # Check min_anchors_before_review
        if len(responses) < self.config.min_anchors_before_review:
            passed = False

        result = CalibrationResult(
            reviewer_id=reviewer_id,
            passed=passed,
            overall_drift=overall_drift,
            per_criterion_drift=per_criterion_drift,
            per_anchor_drift=per_anchor_drift,
            anchor_responses=responses,
        )
        self._calibration_cache[reviewer_id] = result
        return result

    def is_reviewer_calibrated(self, reviewer_id: str) -> bool:
        """Quick check: is this reviewer calibrated?"""
        if reviewer_id not in self._responses:
            return False
        if not self._all_anchors_scored(reviewer_id):
            return False
        result = self.check_calibration(reviewer_id)
        return result.passed

    def should_recalibrate(self, reviewer_id: str) -> bool:
        """Check if reviewer needs recalibration based on review count."""
        if self.config.recalibration_interval <= 0:
            return False
        count = self._review_counts.get(reviewer_id, 0)
        return count >= self.config.recalibration_interval

    def record_review(self, reviewer_id: str) -> None:
        """Record that a reviewer completed a formal review."""
        self._review_counts[reviewer_id] = (
            self._review_counts.get(reviewer_id, 0) + 1
        )

    def reset_recalibration(self, reviewer_id: str) -> None:
        """Reset review count and clear calibration cache after recalibration."""
        self._review_counts[reviewer_id] = 0
        self._responses.pop(reviewer_id, None)
        self._calibration_cache.pop(reviewer_id, None)

    def _get_anchor(self, anchor_id: str) -> AnchorSample | None:
        for anchor in self.config.anchors:
            if anchor.id == anchor_id:
                return anchor
        return None

    def _all_anchors_scored(self, reviewer_id: str) -> bool:
        scored_ids = {
            r.anchor_id for r in self._responses.get(reviewer_id, [])
        }
        return all(a.id in scored_ids for a in self.config.anchors)
