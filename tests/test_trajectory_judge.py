"""Tests for the trajectory_judge grader (LLM-as-judge over the process).

The LLM call is mocked, so these run offline with no provider / API key.
"""

from __future__ import annotations

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders import GradeContext, GraderScope, get_grader


def _transcript(tool_calls=None, reasoning=None, answer=""):
    t = Transcript(task_id="t", trial_id="t")
    t.tool_calls = tool_calls or []
    t.reasoning_steps = reasoning or []
    t.outcome = Outcome(output_data={"final_output": answer} if answer else {})
    return t


def _ctx(transcript, prompt="q"):
    return GradeContext(prompt=prompt, transcript=transcript, outcome=transcript.outcome)


def _fake_llm(capture=None, *, overall=0.85, per=None):
    async def call(prompt):
        if capture is not None:
            capture["prompt"] = prompt
        return {"overall_score": overall, "criteria_scores": per or {},
                "overall_reasoning": "reasonable plan"}
    return call


def _weather_trajectory():
    return _transcript(
        tool_calls=[
            ToolCall(tool_name="Read", input={"file_path": "docs/redis/x.md"},
                     output="INFO memory ... maxmemory-policy", turn_index=1),
            ToolCall(tool_name="Bash", input={"command": "redis-cli INFO memory"},
                     output="used_memory:5G", turn_index=2, status="ok"),
        ],
        reasoning=["[thinking] read the doc first"],
        answer="先看 INFO memory，确认 maxmemory-policy。",
    )


# ---------------------------------------------------------------------------
# Basics
# ---------------------------------------------------------------------------


class TestBasics:
    def test_registered_as_transcript_scope(self):
        g = get_grader("trajectory_judge")({})
        assert g.grader_scope == GraderScope.TRANSCRIPT
        assert g.name == "trajectory_judge"

    def test_defaults_to_four_process_criteria(self):
        g = get_grader("trajectory_judge")({})
        names = {c.name for c in g.criteria}
        assert names == {"plan_soundness", "completeness", "non_redundancy", "tool_choice"}

    def test_custom_criteria_honoured(self):
        g = get_grader("trajectory_judge")({"criteria": [
            {"name": "x", "description": "d"}]})
        assert [c.name for c in g.criteria] == ["x"]
        # schema rebuilt to match the custom criteria
        assert "x" in g._schema["properties"]["criteria_scores"]["properties"]

    def test_inherits_rubric_defaults(self):
        g = get_grader("trajectory_judge")({})
        assert g.model == "gpt-4o" and g.provider == "openai"


# ---------------------------------------------------------------------------
# Trajectory serialization into the prompt
# ---------------------------------------------------------------------------


class TestSerialization:
    async def test_tool_calls_and_reasoning_in_prompt(self, monkeypatch):
        g = get_grader("trajectory_judge")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_weather_trajectory()))
        p = cap["prompt"]
        assert "Tool call sequence" in p
        assert "Read(" in p and "docs/redis/x.md" in p
        assert "redis-cli INFO memory" in p
        assert "turn=1" in p
        assert "Reasoning steps" in p and "read the doc first" in p
        assert "Final answer:" in p and "maxmemory-policy" in p

    async def test_expected_key_steps_in_prompt(self, monkeypatch):
        g = get_grader("trajectory_judge")({"expected_key_steps": ["检索文档", "基于文档作答"]})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_weather_trajectory()))
        assert "Expected key steps" in cap["prompt"]
        assert "检索文档" in cap["prompt"]

    async def test_stats_in_prompt(self, monkeypatch):
        g = get_grader("trajectory_judge")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_weather_trajectory()))
        assert "Trajectory stats: 2 tool calls" in cap["prompt"]

    async def test_include_outcome_false_omits_answer(self, monkeypatch):
        g = get_grader("trajectory_judge")({"include_outcome": False})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_weather_trajectory()))
        assert "Final answer:" not in cap["prompt"]

    async def test_caps_tool_calls(self, monkeypatch):
        calls = [ToolCall(tool_name=f"T{i}", input={}, output="x") for i in range(10)]
        g = get_grader("trajectory_judge")({"max_tool_calls": 3})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_transcript(tool_calls=calls)))
        assert "7 more tool calls omitted" in cap["prompt"]

    async def test_truncates_long_output(self, monkeypatch):
        calls = [ToolCall(tool_name="Read", input={}, output="A" * 5000)]
        g = get_grader("trajectory_judge")({"max_output_chars": 100})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap))
        await g.grade(_ctx(_transcript(tool_calls=calls)))
        assert "…" in cap["prompt"]
        assert "A" * 5000 not in cap["prompt"]


# ---------------------------------------------------------------------------
# Scoring / result
# ---------------------------------------------------------------------------


class TestScoring:
    async def test_pass_uses_overall_score(self, monkeypatch):
        g = get_grader("trajectory_judge")({"pass_threshold": 0.7})
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(overall=0.9))
        r = await g.grade(_ctx(_weather_trajectory()))
        assert r.passed and r.score == pytest.approx(0.9)
        assert r.grader_scope == GraderScope.TRANSCRIPT

    async def test_fail_below_threshold(self, monkeypatch):
        g = get_grader("trajectory_judge")({"pass_threshold": 0.7})
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(overall=0.4))
        r = await g.grade(_ctx(_weather_trajectory()))
        assert not r.passed

    async def test_per_criterion_scores_in_details(self, monkeypatch):
        g = get_grader("trajectory_judge")({})
        per = {c.name: {"score": 0.8, "reasoning": f"{c.name} ok"} for c in g.criteria}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(overall=0.8, per=per))
        r = await g.grade(_ctx(_weather_trajectory()))
        crit = {c["name"] for c in r.details["criteria_results"]}
        assert "non_redundancy" in crit


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


class TestRobustness:
    async def test_empty_transcript_errors(self, monkeypatch):
        g = get_grader("trajectory_judge")({})
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm())
        r = await g.grade(_ctx(_transcript()))
        assert not r.passed and r.error

    async def test_llm_failure_becomes_error_result(self, monkeypatch):
        g = get_grader("trajectory_judge")({})

        async def boom(prompt):
            raise RuntimeError("api down")
        monkeypatch.setattr(g, "_call_llm_structured", boom)
        r = await g.grade(_ctx(_weather_trajectory()))
        assert not r.passed and "api down" in r.error


# ---------------------------------------------------------------------------
# Grading integration (as a normal grader through GradeContext)
# ---------------------------------------------------------------------------


class TestIntegration:
    async def test_works_on_imported_transcript(self, monkeypatch):
        # a transcript as produced by the trace importers
        from compass.integrations import reconstruct_transcript_from_wire
        wire = [
            {"type": "user", "session_id": "s", "message": {"role": "user", "content": "q"}},
            {"type": "assistant", "session_id": "s",
             "message": {"role": "assistant", "model": "claude-opus-4-8", "stop_reason": "tool_use",
                         "usage": {"input_tokens": 10, "output_tokens": 5},
                         "content": [{"type": "tool_use", "id": "t1", "name": "Read",
                                      "input": {"file_path": "docs/x.md"}}]}},
            {"type": "user", "session_id": "s",
             "message": {"role": "user", "content": [
                 {"type": "tool_result", "tool_use_id": "t1",
                  "content": "doc", "is_error": False}]}},
            {"type": "assistant", "session_id": "s",
             "message": {"role": "assistant", "model": "claude-opus-4-8", "stop_reason": "end_turn",
                         "usage": {"input_tokens": 5, "output_tokens": 3},
                         "content": [{"type": "text", "text": "answer"}]}},
            {"type": "result", "subtype": "success", "session_id": "s", "is_error": False,
             "num_turns": 2, "duration_ms": 100, "duration_api_ms": 90, "total_cost_usd": 0.01,
             "result": "answer"},
        ]
        t = reconstruct_transcript_from_wire(wire)
        g = get_grader("trajectory_judge")({})
        cap = {}
        monkeypatch.setattr(g, "_call_llm_structured", _fake_llm(cap, overall=0.9))
        r = await g.grade(GradeContext(prompt="q", transcript=t, outcome=t.outcome))
        assert r.passed
        assert "Read(" in cap["prompt"]
