"""Tests for pairwise comparison grader."""

import pytest
from PIL import Image

from compass.core.transcript import Outcome
from compass.graders.base import GradeContext, GraderScope, GraderType
from compass.graders.human.pairwise import (
    PairwiseComparisonGrader,
    PairwiseResult,
    Preference,
)
from compass.graders.registry import get_grader


def _make_context(
    has_output: bool = True,
    has_reference: bool = True,
    metadata: dict | None = None,
) -> GradeContext:
    """Create a GradeContext for pairwise testing."""
    outcome = None
    if has_output:
        img = Image.new("RGB", (64, 64), "red")
        outcome = Outcome(image=img)

    reference_image = None
    if has_reference:
        reference_image = Image.new("RGB", (64, 64), "blue")

    return GradeContext(
        prompt="test prompt",
        outcome=outcome,
        reference_image=reference_image,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# PairwiseComparisonGrader
# ---------------------------------------------------------------------------


class TestPairwiseComparisonGrader:
    @pytest.mark.asyncio
    async def test_pending_result(self):
        grader = PairwiseComparisonGrader({"criteria": ["quality"]})
        ctx = _make_context()
        result = await grader.grade(ctx)
        assert result.details["status"] == "pending_pairwise_comparison"
        assert result.passed is False
        assert result.score == 0.0
        assert "task_id" in result.details

    @pytest.mark.asyncio
    async def test_no_candidate_a_error(self):
        grader = PairwiseComparisonGrader()
        ctx = _make_context(has_output=False)
        result = await grader.grade(ctx)
        assert result.error == "No output for candidate A"

    @pytest.mark.asyncio
    async def test_no_candidate_b_error(self):
        grader = PairwiseComparisonGrader({"candidate_source": "reference"})
        ctx = _make_context(has_reference=False)
        result = await grader.grade(ctx)
        assert "No candidate B" in result.error

    @pytest.mark.asyncio
    async def test_submit_preference_a_passes(self):
        grader = PairwiseComparisonGrader({"target_preference": "a"})
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        result = await grader.submit_comparison(
            task_id=task_id,
            preference=Preference.A,
            confidence=0.9,
        )
        assert result.passed is True
        assert result.score == 0.9

    @pytest.mark.asyncio
    async def test_submit_preference_b_fails_target_a(self):
        grader = PairwiseComparisonGrader({"target_preference": "a"})
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        result = await grader.submit_comparison(
            task_id=task_id,
            preference=Preference.B,
            confidence=0.8,
        )
        assert result.passed is False
        assert result.score == pytest.approx(0.2)

    @pytest.mark.asyncio
    async def test_tie_with_target_a(self):
        grader = PairwiseComparisonGrader({
            "target_preference": "a",
            "allow_tie": True,
        })
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        result = await grader.submit_comparison(
            task_id=task_id,
            preference=Preference.TIE,
            confidence=0.7,
        )
        assert result.score == 0.5
        assert result.passed is False  # TIE doesn't match target "a"

    @pytest.mark.asyncio
    async def test_target_any_always_passes(self):
        grader = PairwiseComparisonGrader({"target_preference": "any"})
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        for pref in [Preference.A, Preference.B, Preference.TIE]:
            result = await grader.submit_comparison(
                task_id=task_id,
                preference=pref,
                confidence=0.6,
            )
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_confidence_affects_score(self):
        grader = PairwiseComparisonGrader({"target_preference": "a"})
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        r1 = await grader.submit_comparison(
            task_id=task_id, preference=Preference.A, confidence=0.5
        )
        r2 = await grader.submit_comparison(
            task_id=task_id, preference=Preference.A, confidence=1.0
        )
        assert r1.score < r2.score

    @pytest.mark.asyncio
    async def test_criteria_preferences_in_result(self):
        grader = PairwiseComparisonGrader({
            "criteria": ["sharpness", "color"],
            "target_preference": "a",
        })
        ctx = _make_context()
        grade_result = await grader.grade(ctx)
        task_id = grade_result.details["task_id"]

        result = await grader.submit_comparison(
            task_id=task_id,
            preference=Preference.A,
            confidence=0.9,
            criteria_preferences={
                "sharpness": Preference.A,
                "color": Preference.B,
            },
        )
        pairwise = result.details["pairwise_result"]
        assert pairwise["criteria_preferences"]["sharpness"] == "a"
        assert pairwise["criteria_preferences"]["color"] == "b"

    @pytest.mark.asyncio
    async def test_metadata_candidate_source(self):
        grader = PairwiseComparisonGrader({"candidate_source": "metadata"})
        ctx = _make_context(
            has_reference=False,
            metadata={"candidate_b": "some_reference"},
        )
        result = await grader.grade(ctx)
        assert result.details["status"] == "pending_pairwise_comparison"


# ---------------------------------------------------------------------------
# PairwiseResult
# ---------------------------------------------------------------------------


class TestPairwiseResult:
    def test_to_dict(self):
        pr = PairwiseResult(
            task_id="t1",
            reviewer_id="r1",
            preference=Preference.A,
            confidence=0.9,
            criteria_preferences={"q": Preference.B},
            reasoning="A looks better",
        )
        d = pr.to_dict()
        assert d["preference"] == "a"
        assert d["criteria_preferences"]["q"] == "b"
        assert d["reasoning"] == "A looks better"

    def test_preference_enum(self):
        assert Preference.A.value == "a"
        assert Preference.B.value == "b"
        assert Preference.TIE.value == "tie"


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestPairwiseRegistration:
    def test_registered_name(self):
        cls = get_grader("pairwise_comparison")
        assert cls is PairwiseComparisonGrader

    def test_grader_type(self):
        grader = PairwiseComparisonGrader()
        assert grader.grader_type == GraderType.HUMAN

    def test_grader_scope(self):
        grader = PairwiseComparisonGrader()
        assert grader.grader_scope == GraderScope.OUTCOME
