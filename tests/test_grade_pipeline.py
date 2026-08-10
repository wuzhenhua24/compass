"""Tests for the grade pipeline: shared workspace, `creates:`, chain halting.

What turns a flat list of independent checks into a pipeline:

- graders run in declared order over one shared workspace, so a grader's
  output is the next one's input
- `creates:` is a verifiable promise — claiming success without producing the
  promised file is a silent failure, not a pass
- a failed `required` grader halts the chain: no point rendering an SVG that
  could not be extracted, or paying a VLM to judge an image never produced
- whatever the graders leave behind is kept as scoring evidence
"""

from __future__ import annotations

import json

import pytest

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, unregister_adapter
from compass.core.artifact_store import ArtifactStore
from compass.core.artifacts import TextArtifact
from compass.core.regrade import grade_traces
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
from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope
from compass.graders.registry import register_grader

# ===================================================================
# A four-stage pipeline, mirroring the extract -> validate -> render ->
# judge chain a real image eval needs.
# ===================================================================

ran: list[str] = []


@register_grader("_pipe_extract")
class _Extract(CodeGrader):
    """Stage 1: pull the payload out of the agent's answer into the workspace."""

    name = "_pipe_extract"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        ran.append(self.name)
        text = context.text_artifact.content if context.text_artifact else ""
        marker = self.config.get("marker", "<payload>")
        ok = marker in text
        if ok:
            context.workspace_file("extracted.txt").write_text(
                text.split(marker, 1)[1], encoding="utf-8"
            )
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=ok, score=1.0 if ok else 0.0,
        )


@register_grader("_pipe_transform")
class _Transform(CodeGrader):
    """Stage 2: consume stage 1's file and produce a derived one."""

    name = "_pipe_transform"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        ran.append(self.name)
        source = context.read_workspace_file(self.config.get("input", "extracted.txt"))
        if source is None:
            return GradeResult(
                name=self.name, grader_type=self.grader_type,
                grader_scope=self.grader_scope, passed=False, score=0.0,
                error="no input from the previous stage",
            )
        context.workspace_file("rendered.txt").write_text(
            source.upper(), encoding="utf-8"
        )
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=True, score=1.0,
            details={"source_length": len(source)},
        )


@register_grader("_pipe_judge")
class _Judge(CodeGrader):
    """Stage 3: score the derived artifact (stands in for a VLM judge)."""

    name = "_pipe_judge"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        ran.append(self.name)
        rendered = context.read_workspace_file("rendered.txt") or ""
        score = 1.0 if "GOOD" in rendered else 0.2
        context.workspace_file("judge-log.json").write_text(
            json.dumps({"saw": rendered}), encoding="utf-8"
        )
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=score >= 0.5, score=score,
        )


@register_grader("_pipe_liar")
class _Liar(CodeGrader):
    """Reports success but never writes the file it promised."""

    name = "_pipe_liar"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        ran.append(self.name)
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope, passed=True, score=1.0,
        )


@register_grader("_pipe_needs_workspace")
class _NeedsWorkspace(CodeGrader):
    name = "_pipe_needs_workspace"
    grader_scope = GraderScope.OUTCOME

    async def grade(self, context: GradeContext) -> GradeResult:
        return GradeResult(
            name=self.name, grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=context.workspace is not None, score=1.0,
        )


class _AnswerAdapter(Adapter):
    name = "_pipe_answer"

    async def run(self, input: AgentInput) -> AgentOutput:
        return AgentOutput(
            artifacts=[TextArtifact(content=self.config.get("text", "<payload>good"))]
        )


@pytest.fixture(autouse=True)
def _setup():
    ran.clear()
    register_adapter("_pipe_answer")(_AnswerAdapter)
    yield
    unregister_adapter("_pipe_answer")


def _scenario(
    graders: list[GraderConfig], *, text: str = "<payload>good", threshold: float = 0.5
) -> Scenario:
    return Scenario(
        name="pipeline",
        agent=AgentConfig(adapter="_pipe_answer", config={"text": text}),
        cases=[
            TestCase(
                id="case_a",
                input=InputConfig(prompt="go"),
                graders=graders,
                aggregation=AggregationConfig(pass_threshold=threshold),
            )
        ],
    )


def _full_pipeline() -> list[GraderConfig]:
    return [
        GraderConfig(
            name="_pipe_extract", type="code",
            required=True, creates="extracted.txt",
        ),
        GraderConfig(
            name="_pipe_transform", type="code",
            required=True, creates=["rendered.txt"],
        ),
        GraderConfig(name="_pipe_judge", type="code"),
    ]


# ===================================================================
# Artifacts flow between graders
# ===================================================================


class TestSharedWorkspace:
    async def test_graders_chain_through_the_workspace(self):
        """Stage 3 scores what stage 2 derived from what stage 1 extracted."""
        result = await Compass().run(_scenario(_full_pipeline(), text="<payload>good"))
        case = result.case_results[0]

        assert ran == ["_pipe_extract", "_pipe_transform", "_pipe_judge"]
        assert case.passed is True
        # The judge only sees "GOOD" if the chain actually carried the payload
        judge = next(r for r in case.evaluator_results if r.name == "_pipe_judge")
        assert judge.score == pytest.approx(1.0)

    async def test_a_grader_reads_what_the_previous_one_wrote(self):
        result = await Compass().run(
            _scenario(_full_pipeline(), text="<payload>abcdef")
        )
        transform = next(
            r for r in result.case_results[0].evaluator_results
            if r.name == "_pipe_transform"
        )
        assert transform.metadata["source_length"] == len("abcdef")

    async def test_workspace_exists_even_without_a_trace_dir(self):
        """No --trace-dir still gives a temp workspace, so graders can chain."""
        result = await Compass().run(
            _scenario([GraderConfig(name="_pipe_needs_workspace", type="code")])
        )
        assert result.case_results[0].passed is True

    def test_workspace_file_without_a_workspace_is_an_error_not_a_stray_write(self):
        """Better to fail loudly than to write into the current directory."""
        context = GradeContext(prompt="x")
        with pytest.raises(RuntimeError, match="No grade workspace"):
            context.workspace_file("out.txt")

    def test_read_workspace_file_returns_none_when_absent(self):
        assert GradeContext(prompt="x").read_workspace_file("nope.txt") is None


# ===================================================================
# `creates:` is a verifiable promise
# ===================================================================


class TestCreatesContract:
    async def test_a_grader_that_did_not_deliver_fails(self):
        """Success without the promised file is a silent failure, not a pass."""
        result = await Compass().run(
            _scenario(
                [
                    GraderConfig(
                        name="_pipe_liar", type="code", creates="never-written.txt"
                    )
                ]
            )
        )
        grader = result.case_results[0].evaluator_results[0]

        assert grader.passed is False
        assert "missing_promised_file" in grader.failure_tags
        assert "never-written.txt" in grader.error

    async def test_delivering_the_promise_passes(self):
        result = await Compass().run(
            _scenario(
                [
                    GraderConfig(
                        name="_pipe_extract", type="code", creates="extracted.txt"
                    )
                ]
            )
        )
        assert result.case_results[0].evaluator_results[0].passed is True

    async def test_a_list_of_promises_is_checked_in_full(self):
        result = await Compass().run(
            _scenario(
                [
                    GraderConfig(
                        name="_pipe_extract", type="code",
                        creates=["extracted.txt", "also-missing.txt"],
                    )
                ]
            )
        )
        grader = result.case_results[0].evaluator_results[0]
        assert grader.passed is False
        assert "also-missing.txt" in grader.error
        assert "extracted.txt" not in grader.error

    async def test_an_already_failing_grader_is_not_double_penalised(self):
        """A grader that failed on its own merits keeps its own error message."""
        result = await Compass().run(
            _scenario(
                [
                    GraderConfig(
                        name="_pipe_extract", type="code", creates="extracted.txt"
                    )
                ],
                text="no marker here",
            )
        )
        grader = result.case_results[0].evaluator_results[0]
        assert grader.passed is False
        assert "missing_promised_file" not in grader.failure_tags

    def test_creates_normalizes_to_a_list(self):
        assert GraderConfig(name="g").promised_files == []
        assert GraderConfig(name="g", creates="a.txt").promised_files == ["a.txt"]
        assert GraderConfig(name="g", creates=["a", "b"]).promised_files == ["a", "b"]
        assert GraderConfig(name="g", creates="").promised_files == []


# ===================================================================
# A failed prerequisite halts the chain
# ===================================================================


class TestChainHalting:
    async def test_required_failure_stops_the_expensive_stages(self):
        """No payload extracted -> don't transform, don't call the judge."""
        result = await Compass().run(
            _scenario(_full_pipeline(), text="nothing to extract")
        )
        case = result.case_results[0]

        assert ran == ["_pipe_extract"]  # stages 2 and 3 never ran
        assert case.passed is False

        skipped = [r for r in case.evaluator_results if r.skipped]
        assert [r.name for r in skipped] == ["_pipe_transform", "_pipe_judge"]
        assert all(r.skip_reason == "required_failed" for r in skipped)
        assert all("'_pipe_extract' failed" in r.error for r in skipped)

    async def test_halted_graders_are_unscored_not_zero(self):
        """The score is the failing prerequisite's own, not a diluted average."""
        result = await Compass().run(
            _scenario(_full_pipeline(), text="nothing to extract")
        )
        case = result.case_results[0]

        assert all(r.score is None for r in case.evaluator_results if r.skipped)
        assert case.overall_score == pytest.approx(0.0)

    async def test_a_non_required_failure_does_not_halt(self):
        """Only `required` gates the chain; an ordinary failure is just a score."""
        graders = [
            GraderConfig(name="_pipe_extract", type="code", creates="extracted.txt"),
            GraderConfig(name="_pipe_transform", type="code"),
            GraderConfig(name="_pipe_judge", type="code"),
        ]
        result = await Compass().run(_scenario(graders, text="nothing to extract"))

        # Stage 1 failed but was not required, so the chain ran to the end
        assert ran == ["_pipe_extract", "_pipe_transform", "_pipe_judge"]
        assert not any(r.skipped for r in result.case_results[0].evaluator_results)

    async def test_a_broken_promise_on_a_required_grader_halts_too(self):
        """`creates:` failure is a real failure, with the same halting power."""
        await Compass().run(
            _scenario(
                [
                    GraderConfig(
                        name="_pipe_liar", type="code",
                        required=True, creates="never-written.txt",
                    ),
                    GraderConfig(name="_pipe_judge", type="code"),
                ]
            )
        )
        assert ran == ["_pipe_liar"]

    async def test_a_crashed_required_grader_halts_too(self):
        """An unmet prerequisite is unmet whether it failed or exploded."""

        @register_grader("_pipe_boom")
        class _Boom(CodeGrader):
            name = "_pipe_boom"
            grader_scope = GraderScope.OUTCOME

            async def grade(self, context: GradeContext) -> GradeResult:
                raise RuntimeError("prerequisite exploded")

        await Compass().run(
            _scenario(
                [
                    GraderConfig(name="_pipe_boom", type="code", required=True),
                    GraderConfig(name="_pipe_judge", type="code"),
                ]
            )
        )
        assert ran == []  # the judge never ran

    async def test_the_verdict_is_unchanged_by_halting(self):
        """Halting saves money; it must not turn a fail into a pass."""
        result = await Compass().run(
            _scenario(_full_pipeline(), text="nothing to extract", threshold=0.0)
        )
        # Even with a threshold of 0.0, the failed required grader fails the case
        assert result.case_results[0].passed is False


# ===================================================================
# Evidence is retained
# ===================================================================


class TestEvidenceRetention:
    async def test_live_run_keeps_workspace_files_beside_the_trace(self, tmp_path):
        await Compass().run(_scenario(_full_pipeline()), trace_dir=tmp_path)

        evidence = ArtifactStore(tmp_path).list_grade_evidence("case_a")
        assert evidence == ["extracted.txt", "judge-log.json", "rendered.txt"]

    async def test_empty_workspaces_are_not_littered_around(self, tmp_path):
        """A case with no pipeline graders leaves no empty directory behind."""
        await Compass().run(
            _scenario([GraderConfig(name="_pipe_needs_workspace", type="code")]),
            trace_dir=tmp_path,
        )
        assert ArtifactStore(tmp_path).list_grade_evidence("case_a") == []
        assert not (tmp_path / "case_a" / "grade").exists()

    async def test_regrade_keeps_evidence_beside_the_grade_record(self, tmp_path):
        trace_dir = tmp_path / "traces"
        trace_dir.mkdir()
        transcript = Transcript(task_id="case_a", trial_id="t1")
        transcript.outcome = Outcome(
            artifacts=[TextArtifact(content="<payload>good")]
        )
        transcript.finalize(grader_results=[], final_score=0.0, final_passed=False)
        transcript.save(trace_dir / "case_a.json")

        report = await grade_traces(
            _scenario(_full_pipeline()), trace_dir, grade_set="default"
        )
        assert report.result.case_results[0].passed is True

        workspace = trace_dir / "grades" / "default" / "case_a"
        assert (workspace / "rendered.txt").read_text() == "GOOD"

        record = json.loads(
            (trace_dir / "grades" / "default" / "case_a.json").read_text()
        )
        assert record["evidence"] == [
            "extracted.txt", "judge-log.json", "rendered.txt",
        ]

    async def test_artifact_store_workspace_is_beside_output_artifacts(self, tmp_path):
        store = ArtifactStore(tmp_path)
        ws = store.grade_workspace("case_a")
        assert ws == tmp_path / "case_a" / "grade"
        assert ws.is_dir()
        assert store.list_grade_evidence("case_a") == []

        (ws / "x.png").write_text("data")
        assert store.list_grade_evidence("case_a") == ["x.png"]
