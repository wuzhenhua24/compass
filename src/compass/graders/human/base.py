"""Human review grader base implementation."""

from __future__ import annotations

import uuid
from typing import Any

from compass.graders.base import GradeContext, GradeResult, GraderScope, GraderType, HumanGrader
from compass.graders.human.calibration import (
    AnchorSample,
    CalibrationConfig,
    CalibrationResult,
    CalibrationSession,
)
from compass.graders.registry import register_grader


@register_grader("human_review")
class HumanReviewGrader(HumanGrader):
    """Human review grader for manual evaluation.

    Scope: OUTCOME - creates review tasks for the final result.
    Supports optional calibration via CalibrationSession.
    """

    name = "human_review"
    grader_type = GraderType.HUMAN
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.review_criteria = self.config.get("criteria", [])
        self.instructions = self.config.get("instructions", "")
        self.scale = self.config.get("scale", {"min": 1, "max": 5})
        self.platform = self.config.get("platform", "manual")

        # Optional calibration
        cal_data = self.config.get("calibration")
        self._calibration_session: CalibrationSession | None = None
        if cal_data:
            cal_config = CalibrationConfig.model_validate(cal_data)
            if cal_config.enabled:
                self._calibration_session = CalibrationSession(cal_config)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Create human review task for the outcome."""
        if context.image is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No image in outcome",
            )

        task_info = await self._create_review_task(context)

        details: dict[str, Any] = {
            "status": "pending_human_review",
            "task_id": task_info.get("task_id"),
            "platform": self.platform,
            "criteria": self.review_criteria,
        }
        if self._calibration_session:
            details["calibration_required"] = True

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,  # Pending
            score=0.0,
            details=details,
            reasoning="Awaiting human evaluation",
        )

    async def _create_review_task(self, context: GradeContext) -> dict[str, Any]:
        """Create a review task on the configured platform."""
        # TODO: Implement platform-specific task creation
        return {
            "task_id": str(uuid.uuid4()),
            "platform": self.platform,
            "created": True,
        }

    async def submit_result(
        self,
        task_id: str,
        scores: dict[str, float],
        passed: bool,
        notes: str = "",
        reviewer_id: str = "",
    ) -> GradeResult:
        """Submit human evaluation result.

        If calibration is enabled and require_pass is True, the reviewer
        must be calibrated before submitting formal reviews.
        """
        # Calibration gate
        if self._calibration_session and self._calibration_session.config.require_pass:
            if not self._calibration_session.is_reviewer_calibrated(reviewer_id):
                return GradeResult(
                    name=self.name,
                    grader_type=self.grader_type,
                    grader_scope=self.grader_scope,
                    passed=False,
                    score=0.0,
                    error="Reviewer not calibrated",
                )

        if scores:
            overall_score = sum(scores.values()) / len(scores)
            scale_range = self.scale["max"] - self.scale["min"]
            overall_score = (overall_score - self.scale["min"]) / scale_range
        else:
            overall_score = 1.0 if passed else 0.0

        # Record review for recalibration tracking
        if self._calibration_session:
            self._calibration_session.record_review(reviewer_id)

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=overall_score,
            details={
                "status": "human_reviewed",
                "task_id": task_id,
                "criterion_scores": scores,
                "notes": notes,
                "reviewer_id": reviewer_id,
            },
            reasoning=notes or "Human evaluation completed",
        )

    # ------------------------------------------------------------------
    # Calibration API
    # ------------------------------------------------------------------

    def get_calibration_anchor(self, reviewer_id: str) -> AnchorSample | None:
        """Get the next calibration anchor for a reviewer."""
        if self._calibration_session is None:
            return None
        return self._calibration_session.present_anchor(reviewer_id)

    def submit_calibration_response(
        self,
        reviewer_id: str,
        anchor_id: str,
        scores: dict[str, float],
    ) -> CalibrationResult | None:
        """Submit calibration scores for an anchor sample."""
        if self._calibration_session is None:
            return None
        return self._calibration_session.submit_anchor_response(
            reviewer_id, anchor_id, scores
        )
