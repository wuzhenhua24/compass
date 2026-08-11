"""Tests for regrade.py — offline grading of recorded transcripts.

The contract under test: a trace is immutable evidence, and grading it offline
must produce the score the live run produced from the same evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifacts import TextArtifact
from compass.core.regrade import (
    GradeStore,
    case_grader_spec,
    discover_traces,
    grade_traces,
    load_trace,
    spec_fingerprint,
)
from compass.core.result import TestStatus
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
)
from compass.graders.registry import register_grader

# ===================================================================
# Test doubles
# ===================================================================


@register_grader("_regrade_text_contains")
class _TextContainsGrader(CodeGrader):
    """OUTCOME grader: does the text artifact contain a configured phrase?"""

    name = "_regrade_text_contains"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME
    version = "1.0"

    async def grade(self, context: GradeContext) -> GradeResult:
        artifact = context.text_artifact
        content = artifact.content if artifact else ""
        needle = self.config.get("value", "")
        ok = needle in content
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=ok,
            score=1.0 if ok else 0.0,
            details={"content_length": len(content)},
        )


@register_grader("_regrade_min_tool_calls")
class _MinToolCallsGrader(CodeGrader):
    """TRANSCRIPT grader: did the agent make at least N tool calls?"""

    name = "_regrade_min_tool_calls"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT
    version = "1.0"

    async def grade(self, context: GradeContext) -> GradeResult:
        minimum = self.config.get("min_calls", 1)
        count = len(context.tool_calls)
        ok = count >= minimum
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=ok,
            score=1.0 if ok else 0.0,
            details={"tool_calls": count},
        )


class _EchoAdapter(Adapter):
    """Emits a fixed text artifact and records one tool call on the transcript."""

    name = "_regrade_echo"

    async def run(self, input: AgentInput) -> AgentOutput:
        transcript = input.context.get("transcript")
        if transcript is not None:
            transcript.add_tool_call(
                tool_name="search", input={"q": input.prompt}, output="ok"
            )
        text = self.config.get("text", "the answer is 42")
        return AgentOutput(
            metadata={"answer": text},
            artifacts=[TextArtifact(content=text, word_count=len(text.split()))],
        )


@pytest.fixture(autouse=True)
def _register_echo_adapter():
    register_adapter("_regrade_echo")(_EchoAdapter)
    yield
    unregister_adapter("_regrade_echo")


# ===================================================================
# Helpers
# ===================================================================


def _make_scenario(
    *,
    needle: str = "42",
    threshold: float = 0.7,
    expect: str = "pass",
    adapter_text: str = "the answer is 42",
    graders: list[GraderConfig] | None = None,
    case_ids: tuple[str, ...] = ("case_a",),
) -> Scenario:
    grader_configs = graders or [
        GraderConfig(name="_regrade_text_contains", config={"value": needle})
    ]
    return Scenario(
        name="regrade-scenario",
        agent=AgentConfig(adapter="_regrade_echo", config={"text": adapter_text}),
        cases=[
            TestCase(
                id=case_id,
                input=InputConfig(prompt=f"prompt for {case_id}"),
                expect=expect,  # type: ignore[arg-type]
                graders=list(grader_configs),
                aggregation=AggregationConfig(pass_threshold=threshold),
            )
            for case_id in case_ids
        ],
    )


def _write_transcript(
    trace_dir: Path,
    *,
    task_id: str = "case_a",
    text: str = "the answer is 42",
    tool_calls: int = 1,
    blocked: bool = False,
    fmt: str = "json",
) -> Path:
    """Write a trace file directly, without running an agent."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    transcript = Transcript(task_id=task_id, trial_id=f"{task_id}_t1")
    transcript.input_prompt = f"prompt for {task_id}"
    transcript.run_id = "run-123"
    transcript.config_hash = "cfg-abc"
    for i in range(tool_calls):
        transcript.add_tool_call(
            tool_name="search", input={"i": i}, output="ok", duration_ms=5.0
        )
    transcript.outcome = Outcome(
        blocked=blocked,
        output_data={"final_output": text},
        artifacts=[TextArtifact(content=text, word_count=len(text.split()))],
    )
    transcript.finalize(grader_results=[], final_score=0.0, final_passed=False)

    if fmt == "jsonl":
        path = trace_dir / f"{task_id}.jsonl"
        transcript.save_jsonl(path)
    else:
        path = trace_dir / f"{task_id}.json"
        transcript.save(path)
    return path


# ===================================================================
# The core promise: re-grading a trace reproduces the live verdict
# ===================================================================


async def test_regrade_reproduces_live_run_verdict(tmp_path):
    """`compass test` then `compass grade` must agree on the same evidence."""
    trace_dir = tmp_path / "traces"
    scenario = _make_scenario()

    live = await Compass().run(scenario, trace_dir=trace_dir)
    assert live.passed_cases == 1

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    assert report.graded == 1
    assert report.result.total_cases == 1
    assert report.result.passed_cases == 1
    assert report.result.case_results[0].overall_score == pytest.approx(
        live.case_results[0].overall_score
    )


async def test_transcript_scope_grader_regrades_from_trace(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, tool_calls=3)
    scenario = _make_scenario(
        graders=[
            GraderConfig(name="_regrade_min_tool_calls", config={"min_calls": 2})
        ]
    )

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    assert report.result.passed_cases == 1
    grader = report.result.case_results[0].evaluator_results[0]
    assert grader.metadata["tool_calls"] == 3


# ===================================================================
# Grades are additive, idempotent and staleness-aware
# ===================================================================


async def test_grades_land_in_named_directory_with_spec_snapshot(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)
    scenario = _make_scenario()

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    grade_dir = trace_dir / "grades" / "default"
    assert report.grade_dir == grade_dir
    assert (grade_dir / "case_a.json").exists()

    spec = json.loads((grade_dir / "_spec.json").read_text())
    assert spec["grade_set"] == "default"
    assert "case_a" in spec["cases"]
    assert spec["cases"]["case_a"]["fingerprint"]

    # The trace itself is untouched evidence
    assert (trace_dir / "case_a.json").exists()


async def test_second_run_reuses_grades_and_stays_idempotent(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)
    scenario = _make_scenario()

    first = await grade_traces(scenario, trace_dir, grade_set="default")
    second = await grade_traces(scenario, trace_dir, grade_set="default")

    assert first.graded == 1 and first.skipped == 0
    assert second.graded == 0 and second.skipped == 1
    assert second.stale == 0
    # Reused grades still report
    assert second.result.total_cases == 1
    assert second.result.passed_cases == 1


async def test_edited_grader_spec_is_reported_stale(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)

    await grade_traces(_make_scenario(needle="42"), trace_dir, grade_set="default")

    # Edit the grader config — the fingerprint changes even though no
    # `Grader.version` was bumped by hand.
    edited = _make_scenario(needle="something else")
    report = await grade_traces(edited, trace_dir, grade_set="default")

    assert report.stale == 1
    assert report.graded == 0
    assert report.has_stale


async def test_regrade_flag_discards_and_redoes(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)

    await grade_traces(_make_scenario(needle="42"), trace_dir, grade_set="default")
    edited = _make_scenario(needle="not present anywhere")
    report = await grade_traces(edited, trace_dir, grade_set="default", regrade=True)

    assert report.graded == 1
    assert report.stale == 0
    assert report.result.passed_cases == 0  # the edited grader now fails

    record = json.loads((trace_dir / "grades" / "default" / "case_a.json").read_text())
    assert record["passed"] is False


async def test_regrade_removes_orphaned_grade_records(tmp_path):
    """--regrade must not leave a stale record behind for a dropped trace."""
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)
    scenario = _make_scenario()
    await grade_traces(scenario, trace_dir, grade_set="default")

    orphan = trace_dir / "grades" / "default" / "deleted_case.json"
    orphan.write_text(json.dumps({"passed": True, "score": 1.0}))

    await grade_traces(scenario, trace_dir, grade_set="default", regrade=True)
    assert not orphan.exists()


async def test_multiple_grade_sets_coexist(tmp_path):
    """A cheap grader set and an expensive one score the same trace side by side."""
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, tool_calls=3)

    cheap = _make_scenario()
    judge = _make_scenario(
        graders=[GraderConfig(name="_regrade_min_tool_calls", config={"min_calls": 99})]
    )

    await grade_traces(cheap, trace_dir, grade_set="default")
    await grade_traces(judge, trace_dir, grade_set="judge")

    default_rec = json.loads(
        (trace_dir / "grades" / "default" / "case_a.json").read_text()
    )
    judge_rec = json.loads((trace_dir / "grades" / "judge" / "case_a.json").read_text())

    assert default_rec["passed"] is True
    assert judge_rec["passed"] is False
    assert default_rec["spec_fingerprint"] != judge_rec["spec_fingerprint"]


# ===================================================================
# Provenance
# ===================================================================


async def test_grade_record_carries_both_provenances(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)
    scenario = _make_scenario()

    await grade_traces(scenario, trace_dir, grade_set="default")
    record = json.loads((trace_dir / "grades" / "default" / "case_a.json").read_text())

    # Evidence provenance, carried over from the trace
    assert record["run_id"] == "run-123"
    assert record["config_hash"] == "cfg-abc"
    assert record["trace_file"] == "case_a.json"
    # Judgement provenance
    assert record["spec_fingerprint"]
    assert record["grade_results"][0]["grader_version"] == "1.0"


def test_spec_fingerprint_tracks_config_not_formatting():
    scenario = _make_scenario(needle="42")
    case = scenario.cases[0]
    base = spec_fingerprint(case_grader_spec(scenario, case))

    same = _make_scenario(needle="42")
    assert spec_fingerprint(case_grader_spec(same, same.cases[0])) == base

    different = _make_scenario(needle="43")
    assert spec_fingerprint(case_grader_spec(different, different.cases[0])) != base

    # Aggregation is part of the contract too
    moved_threshold = _make_scenario(needle="42", threshold=0.9)
    assert (
        spec_fingerprint(
            case_grader_spec(moved_threshold, moved_threshold.cases[0])
        )
        != base
    )


# ===================================================================
# Fidelity of the round-trip
# ===================================================================


def test_transcript_load_restores_blocked_flag(tmp_path):
    """Negative tests decide from outcome.blocked — it must survive the round-trip."""
    path = _write_transcript(tmp_path, blocked=True)
    loaded = Transcript.load(path)
    assert loaded.outcome.blocked is True


async def test_negative_test_regrades_from_blocked_outcome(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, blocked=True, text="refused")
    # A negative test passes when the agent was blocked, even though the
    # OUTCOME grader itself fails.
    scenario = _make_scenario(needle="42", expect="fail")

    report = await grade_traces(scenario, trace_dir, grade_set="default")
    assert report.result.passed_cases == 1


async def test_multi_trial_traces_produce_trial_metrics(tmp_path):
    trace_dir = tmp_path / "traces"
    trace_dir.mkdir()
    for i, text in enumerate(["the answer is 42", "wrong", "the answer is 42"], start=1):
        transcript = Transcript(task_id="case_a", trial_id=f"case_a_t{i}")
        transcript.outcome = Outcome(artifacts=[TextArtifact(content=text)])
        transcript.finalize(grader_results=[], final_score=0.0, final_passed=False)
        transcript.save(trace_dir / f"case_a_trial{i}.json")

    report = await grade_traces(_make_scenario(), trace_dir, grade_set="default")

    case_result = report.result.case_results[0]
    assert case_result.total_trials == 3
    assert case_result.passed_trials == 2
    assert case_result.trial_metrics is not None
    assert "pass_rate" in case_result.trial_metrics


# ===================================================================
# Honest reporting of what was NOT graded
# ===================================================================


async def test_jsonl_trace_refused_for_outcome_graders(tmp_path):
    """JSONL carries no outcome payload — refuse rather than score a phantom 0."""
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, fmt="jsonl")
    scenario = _make_scenario()  # OUTCOME-scoped grader

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    assert report.graded == 0
    assert len(report.unreadable) == 1
    assert "--trace-format json" in report.unreadable[0][1]
    # The case is reported as an error, not as a failing case
    assert report.result.case_results[0].status == TestStatus.ERROR


async def test_jsonl_trace_graded_for_transcript_scope(tmp_path):
    """TRANSCRIPT-scoped graders survive the JSONL round-trip, so they run."""
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, tool_calls=3, fmt="jsonl")
    scenario = _make_scenario(
        graders=[GraderConfig(name="_regrade_min_tool_calls", config={"min_calls": 2})]
    )

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    assert report.graded == 1
    assert report.result.passed_cases == 1
    assert not report.unreadable


async def test_unmatched_traces_and_missing_cases_are_reported(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, task_id="case_a")
    _write_transcript(trace_dir, task_id="stranger")
    scenario = _make_scenario(case_ids=("case_a", "case_b"))

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    assert report.unmatched == ["stranger"]
    assert report.missing_traces == ["case_b"]
    assert report.result.total_cases == 1  # only case_a was gradable


async def test_case_filter_limits_grading(tmp_path):
    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir, task_id="case_a")
    _write_transcript(trace_dir, task_id="case_b")
    scenario = _make_scenario(case_ids=("case_a", "case_b"))

    report = await grade_traces(
        scenario, trace_dir, grade_set="default", case_ids=["case_a"]
    )

    assert report.result.total_cases == 1
    assert report.result.case_results[0].case_id == "case_a"
    assert not (trace_dir / "grades" / "default" / "case_b.json").exists()
    # A filtered-out case is not reported as a missing trace
    assert report.missing_traces == []


# ===================================================================
# Trace discovery
# ===================================================================


def test_discover_traces_ignores_non_trace_files(tmp_path):
    _write_transcript(tmp_path, task_id="case_a")
    (tmp_path / "checkpoint.json").write_text("{}")
    (tmp_path / "notes.txt").write_text("hi")
    (tmp_path / "grades" / "default").mkdir(parents=True)
    (tmp_path / "grades" / "default" / "case_a.json").write_text("{}")
    (tmp_path / "case_a").mkdir()
    (tmp_path / "case_a" / "manifest.json").write_text("{}")

    found = discover_traces(tmp_path)
    assert [p.name for p in found] == ["case_a.json"]


def test_load_trace_returns_none_for_unrelated_json(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"scenario_name": "x", "total_cases": 1}))
    assert load_trace(path) is None


def test_load_trace_raises_on_corrupt_transcript(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text(json.dumps({"task_id": "a", "trial_id": "b"}))  # missing "input"
    with pytest.raises(ValueError):
        load_trace(path)


def test_grade_store_discards_unreadable_record(tmp_path):
    store = GradeStore(tmp_path, "default")
    store.dir.mkdir(parents=True)
    store.record_path("case_a").write_text("{not json")
    assert store.load_record("case_a") is None


async def test_single_trace_file_can_be_graded(tmp_path):
    """Pointing at one file grades it, writing grades beside it."""
    trace_dir = tmp_path / "traces"
    path = _write_transcript(trace_dir)

    report = await grade_traces(_make_scenario(), path, grade_set="default")

    assert report.graded == 1
    assert report.grade_dir == trace_dir / "grades" / "default"


async def test_grade_output_carries_the_same_fingerprint_as_a_live_run(tmp_path):
    """A re-graded CaseResult must be comparable with a freshly-run one."""
    from compass.core.regrade import grader_fingerprint

    trace_dir = tmp_path / "traces"
    scenario = _make_scenario()
    live = await Compass().run(scenario, trace_dir=trace_dir)

    report = await grade_traces(scenario, trace_dir, grade_set="default")

    expected = grader_fingerprint(scenario, scenario.cases[0])
    assert live.case_results[0].grader_fingerprint == expected
    assert report.result.case_results[0].grader_fingerprint == expected


async def test_compare_flags_a_regrade_under_a_changed_contract(tmp_path):
    """The payoff: two grade sets over one trace set are detectably different."""
    from compass.report.compare import compare_results, load_case_records

    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)

    strict = await grade_traces(
        _make_scenario(needle="42"), trace_dir, grade_set="strict"
    )
    relaxed = await grade_traces(
        _make_scenario(needle="the answer"), trace_dir, grade_set="relaxed"
    )
    (tmp_path / "a.json").write_text(json.dumps(strict.result.to_dict()))
    (tmp_path / "b.json").write_text(json.dumps(relaxed.result.to_dict()))

    report = compare_results(
        load_case_records(tmp_path / "a.json"),
        load_case_records(tmp_path / "b.json"),
    )
    assert report.regraded == ["case_a"]


async def test_a_grader_label_reaches_results_without_moving_the_fingerprint(tmp_path):
    """`label` names a grader instance so results can tell two runs of the same
    grader apart. It cannot move a score, so hashing it would report "regraded"
    — the signal that a diff is not attributable to the agent — for a purely
    cosmetic edit."""
    from compass.core.regrade import grader_fingerprint

    plain = _make_scenario(
        graders=[GraderConfig(name="_regrade_text_contains", config={"value": "42"})]
    )
    labelled = _make_scenario(
        graders=[
            GraderConfig(
                name="_regrade_text_contains",
                label="correctness",
                config={"value": "42"},
            )
        ]
    )
    assert grader_fingerprint(plain, plain.cases[0]) == grader_fingerprint(
        labelled, labelled.cases[0]
    )

    trace_dir = tmp_path / "traces"
    _write_transcript(trace_dir)
    report = await grade_traces(labelled, trace_dir, grade_set="labelled")

    case = report.result.case_results[0]
    assert [e.label for e in case.evaluator_results] == ["correctness"]
    assert "correctness" in case.breakdown
