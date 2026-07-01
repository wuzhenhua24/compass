"""Tests for code quality graders (LintGrader, TypeCheckGrader)."""

from __future__ import annotations

import pytest

from compass.core.artifacts import CodeArtifact, GeneratedFile
from compass.core.transcript import Outcome
from compass.graders.base import GradeContext, GradeResult, GraderType
from compass.graders.code.coding.quality import LintGrader, TypeCheckGrader
from compass.graders.registry import get_grader


def _make_context(
    files: list[GeneratedFile] | None = None,
    **kwargs,
) -> GradeContext:
    """Helper to build a GradeContext with a CodeArtifact."""
    artifact = CodeArtifact(files=files or [])
    outcome = Outcome(artifacts=[artifact])
    return GradeContext(outcome=outcome, **kwargs)


# ===================================================================
# LintGrader tests
# ===================================================================


class TestLintGrader:
    def test_registered(self):
        assert get_grader("lint") is LintGrader

    @pytest.mark.asyncio
    async def test_clean_code_passes(self):
        grader = LintGrader({"lint_command": "echo '{}'"})
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert isinstance(result, GradeResult)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_errors_fail(self):
        # Simulate flake8 text output with errors
        grader = LintGrader({
            "lint_command": (
                "echo 'main.py:1:1: E302 expected 2 blank lines\n"
                "main.py:2:1: E303 too many blank lines'"
            ),
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["error_count"] == 2

    @pytest.mark.asyncio
    async def test_custom_max_errors(self):
        # 2 errors, max_errors=2 should pass
        grader = LintGrader({
            "lint_command": (
                "echo 'main.py:1:1: E302 expected 2 blank lines\n"
                "main.py:2:1: E303 too many blank lines'"
            ),
            "max_errors": 2,
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_max_warnings(self):
        # 2 warnings, max_warnings=1 should fail
        grader = LintGrader({
            "lint_command": (
                "echo 'main.py:1:1: W291 trailing whitespace\n"
                "main.py:2:1: W292 no newline at end'"
            ),
            "max_warnings": 1,
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["warning_count"] == 2

    @pytest.mark.asyncio
    async def test_custom_command(self):
        grader = LintGrader({
            "lint_command": "echo 'custom lint output'",
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        # No parseable lint issues in plain text → clean
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_artifact_returns_error(self):
        grader = LintGrader()
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "CodeArtifact" in result.error

    @pytest.mark.asyncio
    async def test_details_structure(self):
        grader = LintGrader({
            "lint_command": "echo 'main.py:1:1: E302 expected 2 blank lines'",
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert "error_count" in result.details
        assert "warning_count" in result.details
        assert "total_issues" in result.details
        assert "issues_by_severity" in result.details
        assert "issues" in result.details
        assert "stdout" in result.details


# ===================================================================
# TypeCheckGrader tests
# ===================================================================


class TestTypeCheckGrader:
    def test_registered(self):
        assert get_grader("type_check") is TypeCheckGrader

    @pytest.mark.asyncio
    async def test_clean_code_passes(self):
        grader = TypeCheckGrader({
            "type_check_command": "echo 'Success: no issues found'",
        })
        files = [GeneratedFile(path="main.py", content="x: int = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert isinstance(result, GradeResult)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_no_artifact_returns_error(self):
        grader = TypeCheckGrader()
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "CodeArtifact" in result.error

    @pytest.mark.asyncio
    async def test_max_errors_threshold(self):
        # 2 errors, max_errors=2 → pass
        grader = TypeCheckGrader({
            "type_check_command": (
                "echo 'main.py:1: error: Incompatible type\n"
                "main.py:2: error: Missing return'"
            ),
            "max_errors": 2,
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_errors_exceed_max(self):
        # 2 errors, max_errors=0 → fail
        grader = TypeCheckGrader({
            "type_check_command": (
                "echo 'main.py:1: error: Incompatible type\n"
                "main.py:2: error: Missing return'"
            ),
            "max_errors": 0,
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["error_count"] == 2

    @pytest.mark.asyncio
    async def test_strict_mode(self):
        grader = TypeCheckGrader({
            # Use a command that echoes back args so we can check --strict
            "type_check_command": "echo mypy_args",
            "strict": True,
        })
        # Verify the strict flag is set
        assert grader.strict is True
        # The command should have --strict appended
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.details["strict"] is True

    @pytest.mark.asyncio
    async def test_details_structure(self):
        grader = TypeCheckGrader({
            "type_check_command": "echo 'main.py:1: error: Bad type'",
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert "error_count" in result.details
        assert "max_errors" in result.details
        assert "errors" in result.details
        assert "strict" in result.details
        assert "exit_code" in result.details
