"""Tests for functional code graders (ExitCodeGrader, TestRunnerGrader)."""

from __future__ import annotations

import pytest

from compass.core.artifacts import CodeArtifact, ExecutionResult, GeneratedFile
from compass.core.transcript import Outcome
from compass.graders.base import GradeContext, GradeResult, GraderType
from compass.graders.domains.coding.functional import ExitCodeGrader, TestRunnerGrader
from compass.graders.registry import get_grader


def _make_context(
    exit_code: int = 0,
    stdout: str = "",
    stderr: str = "",
    files: list[GeneratedFile] | None = None,
) -> GradeContext:
    """Helper to build a GradeContext with a CodeArtifact."""
    artifact = CodeArtifact(
        files=files or [],
        execution=ExecutionResult(
            exit_code=exit_code, stdout=stdout, stderr=stderr
        ),
    )
    outcome = Outcome(artifacts=[artifact])
    return GradeContext(outcome=outcome)


# ===================================================================
# ExitCodeGrader tests
# ===================================================================


class TestExitCodeGrader:
    def test_registered(self):
        assert get_grader("exit_code_check") is ExitCodeGrader

    @pytest.mark.asyncio
    async def test_exit_code_zero_passes(self):
        grader = ExitCodeGrader()
        ctx = _make_context(exit_code=0)
        result = await grader.grade(ctx)
        assert isinstance(result, GradeResult)
        assert result.passed is True
        assert result.score == 1.0
        assert result.grader_type == GraderType.CODE

    @pytest.mark.asyncio
    async def test_exit_code_nonzero_fails(self):
        grader = ExitCodeGrader()
        ctx = _make_context(exit_code=1)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_custom_expected_exit_code(self):
        grader = ExitCodeGrader({"expected_exit_code": 42})
        ctx = _make_context(exit_code=42)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_custom_expected_exit_code_mismatch(self):
        grader = ExitCodeGrader({"expected_exit_code": 42})
        ctx = _make_context(exit_code=0)
        result = await grader.grade(ctx)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_no_code_artifact_returns_error(self):
        grader = ExitCodeGrader()
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "CodeArtifact" in result.error

    @pytest.mark.asyncio
    async def test_no_execution_returns_error(self):
        grader = ExitCodeGrader()
        artifact = CodeArtifact(files=[], execution=None)
        ctx = GradeContext(outcome=Outcome(artifacts=[artifact]))
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "execution" in result.error.lower()

    @pytest.mark.asyncio
    async def test_details_contain_codes(self):
        grader = ExitCodeGrader()
        ctx = _make_context(exit_code=1)
        result = await grader.grade(ctx)
        assert result.details["actual_exit_code"] == 1
        assert result.details["expected_exit_code"] == 0


# ===================================================================
# TestRunnerGrader tests
# ===================================================================


class TestTestRunnerGrader:
    def test_registered(self):
        assert get_grader("test_runner") is TestRunnerGrader

    @pytest.mark.asyncio
    async def test_simple_test_passes(self):
        grader = TestRunnerGrader({"test_command": "python test.py"})
        files = [
            GeneratedFile(path="test.py", content="print('ok')"),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert isinstance(result, GradeResult)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_test_failure(self):
        grader = TestRunnerGrader({"test_command": "python test.py"})
        files = [
            GeneratedFile(path="test.py", content="raise SystemExit(1)"),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_timeout(self):
        grader = TestRunnerGrader({
            "test_command": "sleep 60",
            "timeout": 0.5,
        })
        files = [GeneratedFile(path="dummy.py", content="")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_pass_pattern_match(self):
        grader = TestRunnerGrader({
            "test_command": "python test.py",
            "pass_pattern": r"ALL TESTS PASSED",
        })
        files = [
            GeneratedFile(
                path="test.py",
                content="print('ALL TESTS PASSED')",
            ),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_pass_pattern_no_match(self):
        grader = TestRunnerGrader({
            "test_command": "python test.py",
            "pass_pattern": r"ALL TESTS PASSED",
        })
        files = [
            GeneratedFile(path="test.py", content="print('some output')"),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fail_pattern_match(self):
        grader = TestRunnerGrader({
            "test_command": "python test.py",
            "fail_pattern": r"FAILURE",
        })
        files = [
            GeneratedFile(
                path="test.py",
                content="print('FAILURE detected')",
            ),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fail_pattern_no_match_passes(self):
        grader = TestRunnerGrader({
            "test_command": "python test.py",
            "fail_pattern": r"FAILURE",
        })
        files = [
            GeneratedFile(path="test.py", content="print('success')"),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_files_written_to_sandbox(self):
        """Verify that the grader writes the artifact files and can read them."""
        grader = TestRunnerGrader({
            "test_command": "cat data.txt",
        })
        files = [
            GeneratedFile(path="data.txt", content="hello from artifact"),
        ]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert "hello from artifact" in result.details["stdout"]

    @pytest.mark.asyncio
    async def test_no_code_artifact_returns_error(self):
        grader = TestRunnerGrader()
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_returns_grade_result(self):
        grader = TestRunnerGrader({"test_command": "echo ok"})
        files = [GeneratedFile(path="f.py", content="")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert isinstance(result, GradeResult)
