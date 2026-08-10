"""Tests for how the agent's final text answer reaches graders.

`GradeContext.answer` used to read only `outcome.output_data["final_output"]`,
which every trace *importer* sets but no *adapter* does — AgentOutput.to_dict()
has no text field. So for any adapter-driven run the answer was empty and the
graders that read it judged nothing at all.

Second, related discipline: a model grader that cannot find its input has not
measured anything. Recording 0.0 there makes "no image was produced" look
identical to "the image is terrible".
"""

from __future__ import annotations

import pytest

from compass.core.artifacts import CodeArtifact, TextArtifact
from compass.core.transcript import Outcome
from compass.graders.base import GradeContext
from compass.graders.model.aesthetic import AestheticScoreGrader
from compass.graders.model.rubric import RubricGrader
from compass.graders.model.semantic import SemanticMatchGrader
from compass.graders.model.vlm import VLMJudgeGrader

# ===================================================================
# Answer resolution
# ===================================================================


class TestAnswerResolution:
    def test_explicit_final_output_wins(self):
        """The importer contract stays authoritative."""
        outcome = Outcome(
            output_data={"final_output": "from the importer"},
            artifacts=[TextArtifact(content="from the artifact")],
        )
        assert GradeContext(outcome=outcome).answer == "from the importer"

    def test_a_text_artifact_is_used_when_there_is_no_final_output(self):
        """The adapter path: returning a TextArtifact is how you return text."""
        outcome = Outcome(artifacts=[TextArtifact(content="the agent's reply")])
        assert GradeContext(outcome=outcome).answer == "the agent's reply"

    def test_an_empty_final_output_falls_through_to_the_artifact(self):
        outcome = Outcome(
            output_data={"final_output": ""},
            artifacts=[TextArtifact(content="the real answer")],
        )
        assert GradeContext(outcome=outcome).answer == "the real answer"

    def test_no_text_anywhere_is_still_empty(self):
        outcome = Outcome(artifacts=[CodeArtifact(files=[])])
        assert GradeContext(outcome=outcome).answer == ""

    def test_an_empty_text_artifact_is_not_an_answer(self):
        outcome = Outcome(artifacts=[TextArtifact(content="")])
        assert GradeContext(outcome=outcome).answer == ""

    def test_no_outcome_at_all(self):
        assert GradeContext().answer == ""

    def test_a_non_string_final_output_is_ignored(self):
        outcome = Outcome(
            output_data={"final_output": {"nested": "thing"}},
            artifacts=[TextArtifact(content="the text")],
        )
        assert GradeContext(outcome=outcome).answer == "the text"


class TestAnswerReachesItsConsumers:
    """The graders that read `answer` were the ones silently judging nothing."""

    async def test_external_checker_exports_the_answer(self, tmp_path):
        import stat

        checker = tmp_path / "echo-answer"
        checker.write_text(
            "#!/usr/bin/env python3\n"
            "import json, os\n"
            "print(json.dumps({'details': {'answer': os.environ.get('COMPASS_ANSWER')}}))\n"
        )
        checker.chmod(checker.stat().st_mode | stat.S_IXUSR)

        from compass.core.transcript import Transcript
        from compass.graders.code.common.external import ExternalCheckerGrader

        ws = tmp_path / "ws"
        ws.mkdir()
        context = GradeContext(
            transcript=Transcript(task_id="t", trial_id="t1"),
            outcome=Outcome(artifacts=[TextArtifact(content="42 is the answer")]),
            workspace=ws,
        )
        result = await ExternalCheckerGrader({"command": str(checker)}).grade(context)

        assert result.details["answer"] == "42 is the answer"

    def test_groundedness_can_now_see_an_adapter_answer(self):
        """It judges answer-vs-evidence; an empty answer means it judges nothing."""
        from compass.graders.model.groundedness import GroundednessGrader

        grader = GroundednessGrader({})
        context = GradeContext(
            outcome=Outcome(artifacts=[TextArtifact(content="redis uses port 6379")])
        )
        assert grader._extract_content(context) is not None


# ===================================================================
# A grader that cannot find its input has not measured anything
# ===================================================================


class TestUnmeasurableInputsAreUnscored:
    async def test_no_image_is_unscored_not_zero(self):
        """"There was no image" is not an aesthetic measurement of 0.0."""
        result = await AestheticScoreGrader({}).grade(
            GradeContext(prompt="x", outcome=Outcome())
        )
        assert result.score is None
        assert result.scored is False
        assert result.passed is False
        assert "No image" in result.error

    async def test_semantic_match_without_an_image_is_unscored(self):
        result = await SemanticMatchGrader({}).grade(
            GradeContext(prompt="x", outcome=Outcome())
        )
        assert result.score is None

    async def test_vlm_judge_without_an_image_is_unscored(self):
        result = await VLMJudgeGrader({"criteria": ["sharp"]}).grade(
            GradeContext(prompt="x", outcome=Outcome())
        )
        assert result.score is None

    async def test_a_misconfigured_rubric_is_unscored(self):
        """No criteria configured: the grader could not evaluate anything."""
        result = await RubricGrader({}).grade(
            GradeContext(prompt="x", outcome=Outcome())
        )
        assert result.score is None
        assert "No criteria" in result.error

    async def test_a_rubric_with_no_content_is_unscored(self):
        result = await RubricGrader(
            {"criteria": [{"name": "a", "description": "b"}]}
        ).grade(GradeContext(prompt="x", outcome=Outcome()))

        assert result.score is None
        assert "No content found" in result.error

    async def test_a_failed_llm_call_is_unscored(self, monkeypatch):
        """A judge outage must not be recorded as a bad answer."""
        grader = RubricGrader(
            {
                "criteria": [{"name": "a", "description": "b"}],
                "source": "text_artifact",
            }
        )

        async def boom(prompt):
            raise RuntimeError("judge unavailable")

        monkeypatch.setattr(grader, "_call_llm_structured", boom)
        result = await grader.grade(
            GradeContext(
                prompt="x", outcome=Outcome(artifacts=[TextArtifact(content="hi")])
            )
        )

        assert result.score is None
        assert "LLM evaluation failed" in result.error

    async def test_a_real_judgement_still_scores_normally(self, monkeypatch):
        """The change must only affect the paths that measured nothing."""
        grader = RubricGrader(
            {
                "criteria": [{"name": "a", "description": "b"}],
                "source": "text_artifact",
                "pass_threshold": 0.5,
            }
        )

        async def judged(prompt):
            return {
                "overall_score": 0.8,
                "criteria_scores": {"a": {"score": 0.8, "reasoning": "ok"}},
                "overall_reasoning": "fine",
            }

        monkeypatch.setattr(grader, "_call_llm_structured", judged)
        result = await grader.grade(
            GradeContext(
                prompt="x", outcome=Outcome(artifacts=[TextArtifact(content="hi")])
            )
        )

        assert result.score == pytest.approx(0.8)
        assert result.passed is True

    async def test_an_unscored_grader_stays_out_of_the_score_but_fails_the_case(self):
        """End to end: the P1 aggregation rule now applies to these paths too."""
        from compass.core.result import EvaluatorResult
        from compass.core.runner import Compass

        good = EvaluatorResult(name="ok", score=1.0, passed=True)
        unmeasured = await AestheticScoreGrader({}).grade(
            GradeContext(prompt="x", outcome=Outcome())
        )
        as_result = EvaluatorResult(
            name=unmeasured.name, score=unmeasured.score, passed=unmeasured.passed
        )

        score, passed = Compass._aggregate_results([good, as_result], 0.5)
        assert score == pytest.approx(1.0)  # not dragged to 0.5
        assert passed is False  # but the case cannot be certified
