"""Tests for security scanning grader (SecurityScanGrader)."""

from __future__ import annotations

import pytest

from compass.core.artifacts import CodeArtifact, GeneratedFile
from compass.core.transcript import Outcome
from compass.graders.base import GradeContext, GradeResult
from compass.graders.code.coding.security import SecurityScanGrader
from compass.graders.registry import get_grader


def _make_context(
    files: list[GeneratedFile] | None = None,
) -> GradeContext:
    """Helper to build a GradeContext with a CodeArtifact."""
    artifact = CodeArtifact(files=files or [])
    outcome = Outcome(artifacts=[artifact])
    return GradeContext(outcome=outcome)


class TestSecurityScanGrader:
    def test_registered(self):
        assert get_grader("security_scan") is SecurityScanGrader

    @pytest.mark.asyncio
    async def test_clean_code_passes(self):
        grader = SecurityScanGrader({"scan_command": None, "blocked_patterns": []})
        files = [GeneratedFile(path="main.py", content="x = 1\nprint(x)\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_blocked_pattern_detected(self):
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\("],
        })
        files = [GeneratedFile(path="main.py", content="result = eval('1+1')\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["pattern_matches"] >= 1

    @pytest.mark.asyncio
    async def test_multiple_patterns(self):
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\(", r"\bexec\s*\("],
        })
        files = [GeneratedFile(
            path="main.py",
            content="eval('1+1')\nexec('print(1)')\n",
        )]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details["pattern_matches"] >= 2

    @pytest.mark.asyncio
    async def test_severity_filtering(self):
        # Pattern matches are HIGH severity; set min_severity=HIGH to include them
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\("],
            "min_severity": "HIGH",
        })
        files = [GeneratedFile(path="main.py", content="eval('1')\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.details["filtered_issues"] >= 1

    @pytest.mark.asyncio
    async def test_max_issues_threshold(self):
        # 1 pattern match, max_issues=1 → pass
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\("],
            "max_issues": 1,
        })
        files = [GeneratedFile(path="main.py", content="eval('1')\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_artifact_returns_error(self):
        grader = SecurityScanGrader({"scan_command": None})
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "CodeArtifact" in result.error

    @pytest.mark.asyncio
    async def test_scan_command_none_pattern_only(self):
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\("],
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.details["pattern_matches"] == 0

    @pytest.mark.asyncio
    async def test_scanner_unavailable_degrades(self):
        # Use a command that will fail
        grader = SecurityScanGrader({
            "scan_command": "nonexistent_command_xyz",
            "blocked_patterns": [],
        })
        files = [GeneratedFile(path="main.py", content="x = 1\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        # Should still produce a result (degraded)
        assert isinstance(result, GradeResult)

    @pytest.mark.asyncio
    async def test_invalid_regex_skipped(self):
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": ["[invalid regex", r"\beval\s*\("],
        })
        files = [GeneratedFile(path="main.py", content="eval('1')\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        # Should still detect eval despite invalid first pattern
        assert result.details["pattern_matches"] >= 1

    @pytest.mark.asyncio
    async def test_details_structure(self):
        grader = SecurityScanGrader({
            "scan_command": None,
            "blocked_patterns": [r"\beval\s*\("],
        })
        files = [GeneratedFile(path="main.py", content="eval('1')\n")]
        ctx = _make_context(files=files)
        result = await grader.grade(ctx)
        assert "total_issues" in result.details
        assert "filtered_issues" in result.details
        assert "issues" in result.details
        assert "pattern_matches" in result.details
        assert "scanner_available" in result.details
