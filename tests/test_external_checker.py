"""Tests for the subprocess checker contract.

The point of this grader is that a checker needs to know nothing about Compass:
a shell script, a Go binary, `npm test`. So the tests are written the same way
— real executables on disk, driven through the real contract.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from compass.core.transcript import Outcome, Transcript
from compass.graders.base import GradeContext
from compass.graders.code.common.external import (
    ExternalCheckerGrader,
    _parse_output,
    _scalar_env_vars,
)

# ===================================================================
# Helpers
# ===================================================================


def _script(tmp_path: Path, name: str, body: str, executable: bool = True) -> Path:
    """Write a python script to disk and make it runnable."""
    path = tmp_path / name
    path.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
    if executable:
        path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _context(workspace: Path, **kwargs) -> GradeContext:
    transcript = Transcript(task_id="t", trial_id="t1")
    transcript.input_prompt = kwargs.pop("prompt", "draw a pelican")
    outcome = Outcome(output_data={"final_output": kwargs.pop("answer", "the answer")})
    workspace.mkdir(parents=True, exist_ok=True)
    return GradeContext(
        prompt=transcript.input_prompt,
        transcript=transcript,
        outcome=outcome,
        workspace=workspace,
        **kwargs,
    )


async def _grade(config: dict, workspace: Path, **ctx_kwargs):
    return await ExternalCheckerGrader(config).grade(_context(workspace, **ctx_kwargs))


# ===================================================================
# The core contract: exit code decides, stdout enriches
# ===================================================================


class TestExitCodeContract:
    async def test_exit_zero_passes(self, tmp_path):
        checker = _script(tmp_path, "ok", "pass\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.passed is True
        assert result.score == pytest.approx(1.0)
        assert result.error is None

    async def test_nonzero_exit_fails_with_stderr_as_the_message(self, tmp_path):
        checker = _script(tmp_path, "bad", "import sys; sys.exit('no svg found')\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.passed is False
        assert result.error == "no svg found"
        assert "checker_failed" in result.failure_tags

    async def test_a_failing_check_scores_zero_not_unscored(self, tmp_path):
        """Deliberate deviation from smevals: exit 1 IS a measurement.

        Compass reserves score=None for "we could not measure", which also
        fails the case. A checker saying "this check failed" measured something.
        """
        checker = _script(tmp_path, "bad", "raise SystemExit(1)\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score == pytest.approx(0.0)
        assert result.scored is True

    async def test_a_failing_check_may_still_report_partial_credit(self, tmp_path):
        checker = _script(
            tmp_path, "partial",
            "import json,sys; print(json.dumps({'score': 0.4})); sys.exit(1)\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.passed is False
        assert result.score == pytest.approx(0.4)

    async def test_exit_code_is_kept_in_details(self, tmp_path):
        checker = _script(tmp_path, "bad", "raise SystemExit(3)\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")
        assert result.details["exit_code"] == 3


# ===================================================================
# The JSON result contract
# ===================================================================


class TestResultContract:
    async def test_score_tags_metrics_notes_and_details(self, tmp_path):
        checker = _script(
            tmp_path, "rich",
            "import json; print(json.dumps({"
            "'score': 0.75, 'tags': ['Has Circle', 'wearing_a_hat'],"
            "'metrics': {'elements': 3}, 'notes': 'looks fine',"
            "'details': {'raw': 7}}))\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score == pytest.approx(0.75)
        # Normalized by the GradeResult boundary, as for any other grader
        assert result.tags == ["has_circle", "wearing_a_hat"]
        # metrics is a first-class field, not buried in the checker's details:
        # the aggregate reads it, so it cannot live in the user namespace.
        assert result.metrics == {"elements": 3.0}
        assert "metrics" not in result.details
        assert result.details["raw"] == 7
        assert result.reasoning == "looks fine"

    async def test_unknown_keys_are_folded_into_details(self, tmp_path):
        """A checker cannot reach in and set a field Compass reserves."""
        checker = _script(
            tmp_path, "sneaky",
            "import json; print(json.dumps({"
            "'passed': False, 'skipped': True, 'grader_version': 'lol',"
            "'whatever': 1}))\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        # Exit code said pass; the payload cannot override that
        assert result.passed is True
        assert result.grader_version == "1.0"
        assert result.details["skipped"] is True  # kept as data, not a control flag
        assert result.details["whatever"] == 1

    async def test_non_json_stdout_is_a_note_not_an_error(self, tmp_path):
        """Plenty of useful checkers just print a line and exit 0."""
        checker = _script(tmp_path, "chatty", "print('all good here')\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.passed is True
        assert result.reasoning == "all good here"

    async def test_silent_checkers_are_fine(self, tmp_path):
        checker = _script(tmp_path, "quiet", "pass\n")
        result = await _grade({"command": str(checker)}, tmp_path / "ws")
        assert result.passed is True
        assert result.reasoning == ""

    def test_parse_output_ignores_a_bool_score(self):
        assert "score" not in _parse_output(json.dumps({"score": True}))

    def test_parse_output_handles_a_json_scalar(self):
        assert _parse_output("42")["details"] == {"output": 42}


# ===================================================================
# The environment the checker sees
# ===================================================================


class TestCheckerEnvironment:
    async def test_cwd_is_the_shared_workspace(self, tmp_path):
        ws = tmp_path / "ws"
        checker = _script(
            tmp_path, "cwd",
            "import pathlib; pathlib.Path('made-here.txt').write_text('x')\n",
        )
        await _grade({"command": str(checker)}, ws)

        assert (ws / "made-here.txt").read_text() == "x"

    async def test_transcript_and_outcome_are_handed_over_as_files(self, tmp_path):
        checker = _script(
            tmp_path, "reads",
            "import json,os,pathlib\n"
            "t = json.loads(pathlib.Path(os.environ['COMPASS_TRANSCRIPT']).read_text())\n"
            "o = json.loads(pathlib.Path(os.environ['COMPASS_OUTCOME']).read_text())\n"
            "print(json.dumps({'details': {'task': t['task_id'],"
            " 'answer': o['output_data']['final_output']}}))\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.details["task"] == "t"
        assert result.details["answer"] == "the answer"

    async def test_those_files_do_not_pollute_the_evidence_workspace(self, tmp_path):
        """The workspace is scoring evidence — not a dumping ground for inputs."""
        ws = tmp_path / "ws"
        checker = _script(tmp_path, "noop", "pass\n")
        await _grade({"command": str(checker)}, ws)

        assert list(ws.iterdir()) == []

    async def test_scalar_config_keys_become_env_vars(self, tmp_path):
        checker = _script(
            tmp_path, "env",
            "import json,os; print(json.dumps({'details': {"
            "'input': os.environ.get('COMPASS_CONFIG_INPUT'),"
            "'max': os.environ.get('COMPASS_CONFIG_MAX_SIZE'),"
            "'strict': os.environ.get('COMPASS_CONFIG_STRICT'),"
            "'nested': os.environ.get('COMPASS_CONFIG_NESTED')}}))\n",
        )
        result = await _grade(
            {
                "command": str(checker),
                "input": "extracted.svg",
                "max_size": 100,
                "strict": True,
                "nested": {"not": "scalar"},
            },
            tmp_path / "ws",
        )

        assert result.details["input"] == "extracted.svg"
        assert result.details["max"] == "100"
        assert result.details["strict"] == "1"
        # Nested values are only available via COMPASS_CONFIG
        assert result.details["nested"] is None

    async def test_full_config_is_available_as_json(self, tmp_path):
        checker = _script(
            tmp_path, "cfg",
            "import json,os; c = json.loads(os.environ['COMPASS_CONFIG']);"
            " print(json.dumps({'details': {'tags': c['tags']}}))\n",
        )
        result = await _grade(
            {"command": str(checker), "tags": ["a", "b"]}, tmp_path / "ws"
        )
        assert result.details["tags"] == ["a", "b"]

    async def test_prompt_and_workspace_are_exported(self, tmp_path):
        ws = tmp_path / "ws"
        checker = _script(
            tmp_path, "env2",
            "import json,os; print(json.dumps({'details': {"
            "'prompt': os.environ['COMPASS_PROMPT'],"
            "'ws': os.environ['COMPASS_WORKSPACE']}}))\n",
        )
        result = await _grade({"command": str(checker)}, ws, prompt="hello there")

        assert result.details["prompt"] == "hello there"
        assert result.details["ws"] == str(ws)

    def test_scalar_env_vars_normalizes_key_names(self):
        assert _scalar_env_vars("P_", {"max-size": 1, "a.b": 2}) == {
            "P_MAX_SIZE": "1", "P_A_B": "2",
        }


# ===================================================================
# Chaining through the workspace
# ===================================================================


class TestChaining:
    async def test_one_checker_consumes_what_another_produced(self, tmp_path):
        ws = tmp_path / "ws"
        producer = _script(
            tmp_path, "produce",
            "import pathlib; pathlib.Path('extracted.svg').write_text('<svg/>')\n",
        )
        consumer = _script(
            tmp_path, "consume",
            "import json,os,pathlib\n"
            "text = pathlib.Path(os.environ['COMPASS_CONFIG_INPUT']).read_text()\n"
            "print(json.dumps({'details': {'seen': text}}))\n",
        )

        first = await _grade({"command": str(producer)}, ws)
        second = await _grade(
            {"command": str(consumer), "input": "extracted.svg"}, ws
        )

        assert first.passed and second.passed
        assert second.details["seen"] == "<svg/>"


# ===================================================================
# Failures Compass itself detects are unscored, not zero
# ===================================================================


class TestHarnessFailures:
    async def test_missing_command_config(self, tmp_path):
        result = await _grade({}, tmp_path / "ws")

        assert result.score is None
        assert "checker_misconfigured" in result.failure_tags

    async def test_a_nonexistent_path(self, tmp_path):
        result = await _grade(
            {"command": str(tmp_path / "nope")}, tmp_path / "ws"
        )
        assert result.score is None
        assert "checker_unavailable" in result.failure_tags

    async def test_a_file_that_is_not_executable(self, tmp_path):
        checker = _script(tmp_path, "noexec", "pass\n", executable=False)
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score is None
        assert "chmod +x" in result.error

    async def test_a_bare_name_not_on_path(self, tmp_path):
        result = await _grade(
            {"command": "definitely-not-a-real-program-xyz"}, tmp_path / "ws"
        )
        assert result.score is None
        assert "PATH" in result.error

    async def test_a_bare_name_is_looked_up_on_path(self, tmp_path):
        result = await _grade({"command": "true"}, tmp_path / "ws")
        assert result.passed is True

    async def test_argv_form_passes_arguments(self, tmp_path):
        checker = _script(
            tmp_path, "args",
            "import json,sys; print(json.dumps({'details': {'argv': sys.argv[1:]}}))\n",
        )
        result = await _grade(
            {"command": [str(checker), "--strict", "x"]}, tmp_path / "ws"
        )
        assert result.details["argv"] == ["--strict", "x"]

    async def test_a_timeout_is_unscored(self, tmp_path):
        checker = _script(tmp_path, "slow", "import time; time.sleep(30)\n")
        result = await _grade(
            {"command": str(checker), "timeout": 0.3}, tmp_path / "ws"
        )

        assert result.score is None
        assert result.passed is False
        assert "checker_timeout" in result.failure_tags

    async def test_a_checker_can_declare_it_could_not_measure(self, tmp_path):
        """The distinction `score is None` exists for, extended to checkers.

        An LLM-judge checker whose model is unreachable has failed at *its*
        job. Without a way to say so it exits non-zero and scores 0.0, which
        reads as "the agent did badly" — evidence about the agent invented out
        of a harness failure.
        """
        checker = _script(
            tmp_path, "cannot_judge",
            "import json,sys\n"
            "print(json.dumps({'unscored': True, 'notes': 'judge unreachable'}))\n"
            "sys.exit(1)\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score is None, "a declared non-measurement scored the agent"
        assert result.passed is False
        assert "checker_unavailable" in result.failure_tags
        assert "judge unreachable" in (result.error or "")

    async def test_a_declared_non_measurement_beats_a_score_in_the_same_payload(
        self, tmp_path
    ):
        """Claiming both a measurement and its absence is a checker bug. The
        safe reading is the one that does not invent evidence."""
        checker = _script(
            tmp_path, "confused",
            "import json\n"
            "print(json.dumps({'unscored': True, 'score': 1.0}))\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score is None
        assert result.passed is False

    async def test_unscored_false_is_an_ordinary_verdict(self, tmp_path):
        """Only the literal `true` opts in — otherwise every checker that
        happens to emit the key gets silently excluded from scoring."""
        checker = _script(
            tmp_path, "fine",
            "import json; print(json.dumps({'unscored': False, 'score': 0.5}))\n",
        )
        result = await _grade({"command": str(checker)}, tmp_path / "ws")

        assert result.score == 0.5
        assert result.passed is True

    async def test_no_workspace_is_reported_not_crashed(self):
        transcript = Transcript(task_id="t", trial_id="t1")
        context = GradeContext(prompt="x", transcript=transcript, outcome=Outcome())
        result = await ExternalCheckerGrader({"command": "true"}).grade(context)

        assert result.score is None
        assert "checker_unavailable" in result.failure_tags


# ===================================================================
# Integration with the rest of the grade pipeline
# ===================================================================


class TestPipelineIntegration:
    async def test_creates_contract_applies_to_external_checkers(self, tmp_path):
        """A checker that exits 0 without producing its promised file fails."""
        from compass.adapters.base import Adapter, AgentInput, AgentOutput
        from compass.adapters.registry import register_adapter, unregister_adapter
        from compass.core.runner import Compass
        from compass.core.scenario import (
            AgentConfig,
            AggregationConfig,
            GraderConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        class _Noop(Adapter):
            name = "_ext_noop"

            async def run(self, input: AgentInput) -> AgentOutput:
                return AgentOutput()

        register_adapter("_ext_noop")(_Noop)
        try:
            liar = _script(tmp_path, "liar", "pass\n")
            scenario = Scenario(
                name="ext",
                agent=AgentConfig(adapter="_ext_noop"),
                cases=[
                    TestCase(
                        id="c1",
                        input=InputConfig(prompt="go"),
                        graders=[
                            GraderConfig(
                                name="external_checker", type="code",
                                creates="promised.txt",
                                config={"command": str(liar)},
                            )
                        ],
                        aggregation=AggregationConfig(pass_threshold=0.5),
                    )
                ],
            )
            result = await Compass().run(scenario, trace_dir=tmp_path / "traces")
        finally:
            unregister_adapter("_ext_noop")

        grader = result.case_results[0].evaluator_results[0]
        assert grader.passed is False
        assert "missing_promised_file" in grader.failure_tags

    def test_the_grader_is_registered(self):
        from compass.graders.registry import get_grader

        assert get_grader("external_checker") is ExternalCheckerGrader

    def test_scope_is_both_so_lossy_traces_are_refused(self):
        """BOTH means a JSONL trace (no outcome payload) will not be graded."""
        from compass.graders.base import GraderScope

        assert ExternalCheckerGrader.grader_scope == GraderScope.BOTH
