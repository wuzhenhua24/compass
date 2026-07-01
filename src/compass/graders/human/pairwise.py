"""Pairwise comparison grader for human evaluation.

Supports A vs B comparisons for subjective quality assessment.
Registered as 'pairwise_comparison' in the grader registry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from compass.graders.base import GradeContext, GradeResult, GraderScope, GraderType, HumanGrader
from compass.graders.registry import register_grader


class Preference(str, Enum):
    """Reviewer preference in a pairwise comparison."""

    A = "a"
    B = "b"
    TIE = "tie"


@dataclass
class PairwiseResult:
    """Result of a pairwise comparison submission."""

    task_id: str
    reviewer_id: str
    preference: Preference
    confidence: float
    criteria_preferences: dict[str, Preference] = field(default_factory=dict)
    reasoning: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "reviewer_id": self.reviewer_id,
            "preference": self.preference.value,
            "confidence": self.confidence,
            "criteria_preferences": {
                k: v.value for k, v in self.criteria_preferences.items()
            },
            "reasoning": self.reasoning,
        }


@register_grader("pairwise_comparison")
class PairwiseComparisonGrader(HumanGrader):
    """Pairwise comparison grader for A vs B human evaluation.

    Scope: OUTCOME - compares two output candidates.
    """

    name = "pairwise_comparison"
    grader_type = GraderType.HUMAN
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.criteria = self.config.get("criteria", [])
        self.candidate_source = self.config.get("candidate_source", "reference")
        self.allow_tie = self.config.get("allow_tie", True)
        self.randomize_order = self.config.get("randomize_order", True)
        self.target_preference = self.config.get("target_preference", "a")
        self._pending_tasks: dict[str, dict[str, Any]] = {}

    async def grade(self, context: GradeContext) -> GradeResult:
        """Create a pairwise comparison task.

        Candidate A is the outcome output.
        Candidate B comes from reference or metadata based on candidate_source.
        """
        # Check candidate A (the output — legacy image or artifact)
        has_candidate_a = context.image is not None or context.has_output
        if not has_candidate_a:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No output for candidate A",
            )

        # Check candidate B
        has_candidate_b = False
        if self.candidate_source == "reference":
            has_candidate_b = context.has_reference_images or context.reference_artifact is not None
        elif self.candidate_source == "metadata":
            has_candidate_b = bool(context.metadata.get("candidate_b"))

        if not has_candidate_b:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=f"No candidate B from source '{self.candidate_source}'",
            )

        task_id = str(uuid.uuid4())
        self._pending_tasks[task_id] = {
            "candidate_source": self.candidate_source,
            "criteria": self.criteria,
            "allow_tie": self.allow_tie,
            "target_preference": self.target_preference,
        }

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
            details={
                "status": "pending_pairwise_comparison",
                "task_id": task_id,
                "criteria": self.criteria,
                "allow_tie": self.allow_tie,
                "target_preference": self.target_preference,
            },
            reasoning="Awaiting pairwise comparison",
        )

    async def submit_comparison(
        self,
        task_id: str,
        preference: Preference,
        confidence: float = 1.0,
        criteria_preferences: dict[str, Preference] | None = None,
        reasoning: str = "",
        reviewer_id: str = "",
    ) -> GradeResult:
        """Submit a pairwise comparison result.

        Score computation:
        - target="a" and preference=A → score=confidence, passed=True
        - target="a" and preference=B → score=1-confidence, passed=False
        - target="a" and preference=TIE → score=0.5, passed depends on confidence>=0.5
        - target="any" → always passed=True, score=confidence
        """
        target = self.target_preference

        if target == "any":
            score = confidence
            passed = True
        elif preference == Preference.TIE:
            score = 0.5
            passed = target == "any"
        elif preference.value == target:
            score = confidence
            passed = True
        else:
            score = 1.0 - confidence
            passed = False

        pairwise_result = PairwiseResult(
            task_id=task_id,
            reviewer_id=reviewer_id,
            preference=preference,
            confidence=confidence,
            criteria_preferences=criteria_preferences or {},
            reasoning=reasoning,
        )

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={
                "status": "pairwise_compared",
                "task_id": task_id,
                "pairwise_result": pairwise_result.to_dict(),
            },
            reasoning=reasoning or "Pairwise comparison completed",
        )
