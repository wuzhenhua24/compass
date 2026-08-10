"""Tests for observed tags: neutral, presence-only, aggregated as shares.

`failure_tags` explain why something failed. Observed tags describe *what was
seen*, on passing runs too — which is what turns a pile of scores into a
behavioural profile ("62% of answers cited a document", "9% showed no bicycle").

Three properties under test:

1. neutral — emitted pass or fail, and aggregated over every task
2. presence-only — an absent tag means "not observed", never "false"
3. normalized + vocabulary-checked — so they actually bucket together
"""

from __future__ import annotations

import json

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifacts import TextArtifact
from compass.core.regrade import grade_traces
from compass.core.result import CaseResult, EvaluatorResult, TestStatus
from compass.core.runner import Compass
from compass.core.scenario import (
    AgentConfig,
    AggregationConfig,
    GraderConfig,
    InputConfig,
    Scenario,
    TestCase,
)
from compass.core.transcript import Outcome, Transcript
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
    normalize_tag,
    normalize_tags,
)
from compass.graders.model.rubric import Criterion, _build_evaluation_schema
from compass.graders.registry import register_grader
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult

# ===================================================================
# Test doubles
# ===================================================================


@register_grader("_tag_observer")
class _Observer(CodeGrader):
    """Tags what it sees in the answer — and passes or fails independently."""

    name = "_tag_observer"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        answer = context.text_artifact.content if context.text_artifact else ""
        tags = []
        if "doc" in answer:
            tags.append("cited_a_document")
        if "sorry" in answer:
            tags.append("declined_to_answer")
        if not answer:
            tags.append("empty_response")
        passed = "doc" in answer
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=passed,
            score=1.0 if passed else 0.0, tags=tags,
        )


class _EchoAdapter(Adapter):
    name = "_tag_echo"

    async def run(self, input: AgentInput) -> AgentOutput:
        return AgentOutput(
            artifacts=[TextArtifact(content=input.params.get("answer", ""))]
        )


@pytest.fixture(autouse=True)
def _adapter():
    register_adapter("_tag_echo")(_EchoAdapter)
    yield
    unregister_adapter("_tag_echo")


def _scenario(answers: dict[str, str]) -> Scenario:
    return Scenario(
        name="tags",
        agent=AgentConfig(adapter="_tag_echo"),
        cases=[
            TestCase(
                id=cid,
                input=InputConfig(prompt="q", params={"answer": ans}),
                graders=[GraderConfig(name="_tag_observer", type="code")],
                aggregation=AggregationConfig(pass_threshold=0.5),
            )
            for cid, ans in answers.items()
        ],
    )


def _task(task_id: str, passed: bool, tags: list[str]) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=task_id,
        passed=passed,
        grade_results=[
            GradeResult(
                name="g", grader_type=GraderType.CODE,
                grader_scope=GraderScope.OUTCOME,
                passed=passed, score=1.0 if passed else 0.0, tags=tags,
            )
        ],
    )


# ===================================================================
# Normalization at the boundary
# ===================================================================


class TestNormalization:
    def test_tags_become_snake_case(self):
        assert normalize_tag("Correct Bicycle Shape!") == "correct_bicycle_shape"
        assert normalize_tag("no-svg") == "no_svg"
        assert normalize_tag("  Wearing A Hat  ") == "wearing_a_hat"

    def test_normalize_deduplicates_and_sorts(self):
        assert normalize_tags(["B tag", "b_tag", "a tag"]) == ["a_tag", "b_tag"]

    def test_empty_tags_are_dropped(self):
        assert normalize_tags(["", "   ", "!!!", "ok"]) == ["ok"]

    def test_grade_result_normalizes_on_construction(self):
        """Enforced at the boundary — no grader can leak a raw tag through."""
        result = GradeResult(name="g", tags=["Wearing A Hat", "wearing_a_hat", "!!"])
        assert result.tags == ["wearing_a_hat"]

    def test_no_tags_stays_empty(self):
        assert GradeResult(name="g").tags == []


# ===================================================================
# Neutral: tags flow from passing runs too
# ===================================================================


class TestNeutralSemantics:
    async def test_tags_reach_the_case_result_on_a_pass(self):
        result = await Compass().run(_scenario({"a": "see the doc"}))
        case = result.case_results[0]

        assert case.passed is True
        assert case.observed_tags == ["cited_a_document"]

    async def test_tags_reach_the_case_result_on_a_fail(self):
        result = await Compass().run(_scenario({"a": "sorry, no idea"}))
        case = result.case_results[0]

        assert case.passed is False
        assert case.observed_tags == ["declined_to_answer"]

    def test_observed_tags_is_the_union_across_graders(self):
        case = CaseResult(
            case_id="c", status=TestStatus.PASSED, passed=True, overall_score=1.0,
            evaluator_results=[
                EvaluatorResult(name="a", score=1.0, passed=True, tags=["x", "y"]),
                EvaluatorResult(name="b", score=1.0, passed=True, tags=["y", "z"]),
            ],
        )
        assert case.observed_tags == ["x", "y", "z"]

    def test_observed_tags_is_distinct_from_static_classification(self):
        """`tags` is what you wrote in YAML; `observed_tags` is what happened."""
        case = CaseResult(
            case_id="c", status=TestStatus.PASSED, passed=True, overall_score=1.0,
            tags=["backend", "smoke"],
            evaluator_results=[
                EvaluatorResult(name="a", score=1.0, passed=True, tags=["cited_a_doc"])
            ],
        )
        assert case.tags == ["backend", "smoke"]
        assert case.observed_tags == ["cited_a_doc"]

    async def test_tags_survive_serialization(self):
        result = await Compass().run(_scenario({"a": "see the doc"}))
        d = result.to_dict()["case_results"][0]

        assert d["observed_tags"] == ["cited_a_document"]
        assert d["grade_results"][0]["tags"] == ["cited_a_document"]

    def test_evaluator_result_tags_round_trip(self):
        original = EvaluatorResult(name="g", score=1.0, passed=True, tags=["a", "b"])
        assert EvaluatorResult.from_dict(original.to_dict()).tags == ["a", "b"]


# ===================================================================
# Aggregation: counts, shares, and per-tag pass rate
# ===================================================================


class TestAggregation:
    def test_shares_are_over_every_task_not_just_failures(self):
        results = [
            _task("t1", True, ["cited_a_document"]),
            _task("t2", True, ["cited_a_document"]),
            _task("t3", False, ["declined_to_answer"]),
            _task("t4", True, []),
        ]
        analysis = EvalResultAnalyzer(results).analyze().observed_tag_analysis

        assert analysis["cited_a_document"]["count"] == 2
        assert analysis["cited_a_document"]["share"] == pytest.approx(0.5)
        assert analysis["declined_to_answer"]["share"] == pytest.approx(0.25)

    def test_per_tag_pass_rate_is_the_actionable_column(self):
        """"The runs tagged X fail more" is the finding worth surfacing."""
        results = [
            _task("t1", True, ["short_answer"]),
            _task("t2", False, ["short_answer"]),
            _task("t3", False, ["short_answer"]),
            _task("t4", True, ["long_answer"]),
        ]
        analysis = EvalResultAnalyzer(results).analyze().observed_tag_analysis

        assert analysis["short_answer"]["pass_rate"] == pytest.approx(1 / 3)
        assert analysis["long_answer"]["pass_rate"] == pytest.approx(1.0)

    def test_a_tag_is_counted_once_per_task(self):
        """Two graders observing the same thing is one observation, not two."""
        task = TaskEvalResult(
            task_id="t1", passed=True,
            grade_results=[
                GradeResult(name="a", grader_type=GraderType.CODE,
                            grader_scope=GraderScope.OUTCOME,
                            passed=True, score=1.0, tags=["same_tag"]),
                GradeResult(name="b", grader_type=GraderType.CODE,
                            grader_scope=GraderScope.OUTCOME,
                            passed=True, score=1.0, tags=["same_tag"]),
            ],
        )
        analysis = EvalResultAnalyzer([task]).analyze().observed_tag_analysis
        assert analysis["same_tag"]["count"] == 1
        assert sorted(analysis["same_tag"]["graders"]) == ["a", "b"]

    def test_sorted_by_frequency(self):
        results = [
            _task("t1", True, ["rare"]),
            _task("t2", True, ["common"]),
            _task("t3", True, ["common"]),
        ]
        analysis = EvalResultAnalyzer(results).analyze().observed_tag_analysis
        assert list(analysis) == ["common", "rare"]

    def test_no_tags_yields_an_empty_section(self):
        analysis = EvalResultAnalyzer([_task("t1", True, [])]).analyze()
        assert analysis.observed_tag_analysis == {}
        assert analysis.to_dict()["observed_tag_analysis"] == {}

    def test_empty_result_set_does_not_divide_by_zero(self):
        assert EvalResultAnalyzer([]).analyze().observed_tag_analysis == {}


# ===================================================================
# Controlled vocabulary for LLM judges
# ===================================================================


class TestControlledVocabulary:
    def _criteria(self) -> list[Criterion]:
        return [Criterion(name="accuracy", description="is it right")]

    def test_configured_tags_become_a_schema_enum(self):
        """A judge that can only pick from a fixed list produces aggregatable tags."""
        schema = _build_evaluation_schema(
            self._criteria(), ["cited_a_doc", "hedged", "refused"]
        )
        tags = schema["properties"]["tags"]

        assert tags["type"] == "array"
        assert tags["items"]["enum"] == ["cited_a_doc", "hedged", "refused"]

    def test_without_a_vocabulary_no_tags_are_requested(self):
        """Opt-in: existing rubric configs are untouched and pay no extra tokens."""
        schema = _build_evaluation_schema(self._criteria())
        assert "tags" not in schema["properties"]
        schema = _build_evaluation_schema(self._criteria(), [])
        assert "tags" not in schema["properties"]

    def test_the_rest_of_the_schema_is_unchanged(self):
        schema = _build_evaluation_schema(self._criteria(), ["a"])
        assert schema["required"] == [
            "overall_score", "criteria_scores", "overall_reasoning",
        ]
        assert "accuracy" in schema["properties"]["criteria_scores"]["properties"]

    async def test_off_vocabulary_tags_from_the_judge_are_dropped(self, monkeypatch):
        """A judge ignoring the enum cannot smuggle in an unaggregatable tag."""
        from compass.graders.model.rubric import RubricGrader

        grader = RubricGrader(
            {
                "criteria": [{"name": "accuracy", "description": "is it right"}],
                "tags": ["cited_a_doc", "hedged"],
                "source": "text_artifact",
            }
        )

        async def fake_call(prompt):
            return {
                "overall_score": 0.9,
                "criteria_scores": {"accuracy": {"score": 0.9, "reasoning": "ok"}},
                "overall_reasoning": "fine",
                "tags": ["cited_a_doc", "invented_by_the_judge"],
            }

        monkeypatch.setattr(grader, "_call_llm_structured", fake_call)
        outcome = Outcome(artifacts=[TextArtifact(content="the answer")])
        result = await grader.grade(
            GradeContext(prompt="q", outcome=outcome, params={})
        )

        assert result.tags == ["cited_a_doc"]


# ===================================================================
# End to end through compass grade
# ===================================================================


class TestGradeRecord:
    async def test_grade_record_carries_the_tag_union(self, tmp_path):
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        transcript = Transcript(task_id="a", trial_id="t1")
        transcript.outcome = Outcome(
            artifacts=[TextArtifact(content="see the doc, sorry")]
        )
        transcript.finalize(grader_results=[], final_score=0.0, final_passed=False)
        transcript.save(trace_dir / "a.json")

        report = await grade_traces(
            _scenario({"a": "unused"}), trace_dir, grade_set="default"
        )

        record = json.loads((trace_dir / "grades" / "default" / "a.json").read_text())
        assert record["tags"] == ["cited_a_document", "declined_to_answer"]
        assert report.result.case_results[0].observed_tags == [
            "cited_a_document", "declined_to_answer",
        ]

    async def test_analyze_reads_tags_back_from_a_results_file(self, tmp_path):
        """The CLI path must reconstruct tags, or the aggregation is always empty."""
        from compass.report.analyzer import iter_case_dicts

        result = await Compass().run(
            _scenario({"a": "see the doc", "b": "sorry, no idea"})
        )
        path = tmp_path / "results.json"
        path.write_text(json.dumps(result.to_dict()))

        cases = iter_case_dicts(json.loads(path.read_text()))
        tags = {t for c in cases for gr in c["grade_results"] for t in gr["tags"]}
        assert tags == {"cited_a_document", "declined_to_answer"}
