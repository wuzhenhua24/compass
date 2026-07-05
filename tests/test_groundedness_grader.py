"""Tests for the groundedness grader (answer-vs-evidence LLM judge).

The LLM call is mocked, so these run offline with no provider / API key.
"""

from __future__ import annotations

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders import GradeContext, GraderScope, get_grader
from compass.graders.model.groundedness import _is_empty_result


def _transcript(tool_calls=None, answer=""):
    t = Transcript(task_id="t", trial_id="t")
    t.tool_calls = tool_calls or []
    t.outcome = Outcome(output_data={"final_output": answer} if answer else {})
    return t


def _ctx(transcript, prompt="How many open tickets?"):
    return GradeContext(prompt=prompt, transcript=transcript, outcome=transcript.outcome)


def _fake_llm(capture=None, *, overall=0.9, per=None):
    async def call(prompt):
        if capture is not None:
            capture["prompt"] = prompt
        return {"overall_score": overall, "criteria_scores": per or {},
                "overall_reasoning": "grounded"}
    return call


def _hallucination_case():
    """The canonical failure: empty tool result, fabricated number."""
    return _transcript(
        tool_calls=[
            ToolCall(tool_name="db.query", input={"sql": "SELECT * FROM tickets"},
                     output=[]),
        ],
        answer="There are 42 open tickets.",
    )


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_registered_as_both_scope(self):
        g = get_grader("groundedness")({})
        assert g.grader_scope == GraderScope.BOTH
        assert g.name == "groundedness"

    def test_defaults_to_three_groundedness_criteria(self):
        g = get_grader("groundedness")({})
        names = {c.name for c in g.criteria}
        assert names == {"claim_support", "no_fabrication", "faithful_use"}

    def test_custom_criteria_rebuild_schema(self):
        g = get_grader("groundedness")({"criteria": [{"name": "x", "description": "d"}]})
        assert [c.name for c in g.criteria] == ["x"]
        assert "x" in g._schema["properties"]["criteria_scores"]["properties"]

    def test_inherits_rubric_defaults(self):
        g = get_grader("groundedness")({})
        assert g.model == "gpt-4o" and g.provider == "openai"


class TestIsEmptyResult:
    @pytest.mark.parametrize("output", [None, "", "   ", [], {}, "[]", "{}", "null", "None"])
    def test_empty_shapes(self, output):
        assert _is_empty_result(output) is True

    @pytest.mark.parametrize("output", ["42", [1], {"rows": []}, 0, False, "no results found"])
    def test_non_empty_shapes(self, output):
        assert _is_empty_result(output) is False


# ---------------------------------------------------------------------------
# Serialization into the prompt
# ---------------------------------------------------------------------------


class TestSerialization:
    async def test_answer_and_evidence_in_prompt(self, monkeypatch):
        g = get_grader("groundedness")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        t = _transcript(
            tool_calls=[ToolCall(tool_name="db.query", input={"sql": "SELECT count(*)"},
                                 output="17")],
            answer="There are 17 open tickets.",
        )
        await g.grade(_ctx(t))
        p = cap["prompt"]
        assert "Final answer:" in p and "17 open tickets" in p
        assert "Evidence (tool observations):" in p
        assert "db.query(" in p and "SELECT count(*)" in p
        assert "User request: How many open tickets?" in p

    async def test_empty_result_flagged(self, monkeypatch):
        g = get_grader("groundedness")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_hallucination_case()))
        p = cap["prompt"]
        assert "[EMPTY RESULT: no evidence produced]" in p
        assert "1 returned EMPTY/ERROR" in p  # stats line warns the judge

    async def test_error_result_flagged(self, monkeypatch):
        g = get_grader("groundedness")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        t = _transcript(
            tool_calls=[ToolCall(tool_name="api.get", input={}, output="boom",
                                 status="error", error={"message": "boom"})],
            answer="The API says everything is healthy.",
        )
        await g.grade(_ctx(t))
        assert "[ERROR: no evidence produced]" in cap["prompt"]

    async def test_evidence_tools_filter(self, monkeypatch):
        g = get_grader("groundedness")({"evidence_tools": ["db.*"]})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        t = _transcript(
            tool_calls=[
                ToolCall(tool_name="db.query", input={}, output="17"),
                ToolCall(tool_name="scratchpad.write", input={}, output="noted"),
            ],
            answer="17.",
        )
        await g.grade(_ctx(t))
        assert "db.query(" in cap["prompt"]
        assert "scratchpad.write" not in cap["prompt"]

    async def test_llm_generation_excluded_by_default(self, monkeypatch):
        """Imported llm.generation spans carry the model's own text — an answer
        must not count as evidence for itself."""
        g = get_grader("groundedness")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        t = _transcript(
            tool_calls=[
                ToolCall(tool_name="llm.generation", input={},
                         output="There are 42 open tickets."),
            ],
            answer="There are 42 open tickets.",
        )
        await g.grade(_ctx(t))
        assert "llm.generation" not in cap["prompt"]
        assert "no evidence tool calls recorded" in cap["prompt"]

    async def test_no_evidence_note(self, monkeypatch):
        g = get_grader("groundedness")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_transcript(answer="Paris is the capital of France.")))
        assert "no evidence tool calls recorded" in cap["prompt"]

    async def test_caps_and_omission_note(self, monkeypatch):
        calls = [ToolCall(tool_name=f"t{i}", input={}, output="x") for i in range(10)]
        g = get_grader("groundedness")({"max_tool_calls": 3})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_transcript(tool_calls=calls, answer="a")))
        assert "7 more observations omitted" in cap["prompt"]

    async def test_truncates_long_observation(self, monkeypatch):
        calls = [ToolCall(tool_name="Read", input={}, output="A" * 5000)]
        g = get_grader("groundedness")({"max_output_chars": 100})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_transcript(tool_calls=calls, answer="a")))
        assert "A" * 5000 not in cap["prompt"]


# ---------------------------------------------------------------------------
# Scoring / result
# ---------------------------------------------------------------------------


class TestScoring:
    async def test_passes_at_threshold(self, monkeypatch):
        g = get_grader("groundedness")({})
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(overall=0.9))
        result = await g.grade(_ctx(_hallucination_case()))
        assert result.passed is True
        assert result.score == 0.9

    async def test_fails_below_threshold(self, monkeypatch):
        g = get_grader("groundedness")({})
        monkeypatch.setattr(
            g, "_call_llm_structured",
            _fake_llm(overall=0.2, per={"no_fabrication": {
                "score": 0.0, "reasoning": "fabricated 42 from empty result"}}),
        )
        result = await g.grade(_ctx(_hallucination_case()))
        assert result.passed is False
        assert result.score == 0.2

    async def test_no_answer_errors(self, monkeypatch):
        g = get_grader("groundedness")({})
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm())
        result = await g.grade(_ctx(_transcript(
            tool_calls=[ToolCall(tool_name="db.query", input={}, output="17")],
        )))
        assert result.passed is False
        assert result.error and "No content" in result.error

    async def test_llm_failure_is_error_not_crash(self, monkeypatch):
        g = get_grader("groundedness")({})

        async def boom(prompt):
            raise RuntimeError("provider down")

        monkeypatch.setattr(g, "_call_llm_structured", boom)
        result = await g.grade(_ctx(_hallucination_case()))
        assert result.passed is False
        assert "provider down" in (result.error or "")
