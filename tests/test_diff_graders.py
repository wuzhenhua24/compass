"""Tests for diff-based graders (DiffAccuracyGrader, DiffSizeGrader)."""

from __future__ import annotations

import pytest

from compass.core.artifacts import CodeArtifact, GeneratedFile
from compass.core.transcript import Outcome
from compass.graders.base import GradeContext
from compass.graders.code.coding.diff import DiffAccuracyGrader, DiffSizeGrader
from compass.graders.registry import get_grader


def _make_context(
    files: list[GeneratedFile] | None = None,
    reference_artifact=None,
    metadata: dict | None = None,
) -> GradeContext:
    """Helper to build a GradeContext with a CodeArtifact."""
    artifact = CodeArtifact(files=files or [])
    outcome = Outcome(artifacts=[artifact])
    return GradeContext(
        outcome=outcome,
        reference_artifact=reference_artifact,
        metadata=metadata or {},
    )


def _make_diff_context(
    files: list[GeneratedFile] | None = None,
    diff: str = "",
    reference_artifact=None,
) -> GradeContext:
    """Helper for DiffSizeGrader with optional diff field."""
    artifact = CodeArtifact(files=files or [], diff=diff)
    outcome = Outcome(artifacts=[artifact])
    return GradeContext(
        outcome=outcome,
        reference_artifact=reference_artifact,
    )


# ===================================================================
# DiffAccuracyGrader tests
# ===================================================================


class TestDiffAccuracyGrader:
    def test_registered(self):
        assert get_grader("diff_accuracy") is DiffAccuracyGrader

    @pytest.mark.asyncio
    async def test_identical_files_score_1(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\n")]
        )
        grader = DiffAccuracyGrader()
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_different_files_low_score(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\ny = 2\nz = 3\n")]
        )
        grader = DiffAccuracyGrader()
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="a = 100\nb = 200\nc = 300\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score < 0.9
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_normalized_ignores_whitespace(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\n\n\ny = 2\n")]
        )
        grader = DiffAccuracyGrader({"match_mode": "normalized", "ignore_whitespace": True})
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\ny = 2\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_exact_catches_whitespace(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\n\n\ny = 2\n")]
        )
        grader = DiffAccuracyGrader({"match_mode": "exact"})
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\ny = 2\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score == 0.0

    @pytest.mark.asyncio
    async def test_ignore_comments(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1  # original comment\n")]
        )
        grader = DiffAccuracyGrader({
            "match_mode": "normalized",
            "ignore_comments": True,
            "ignore_whitespace": True,
        })
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1  # different comment\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_file_weights(self):
        ref = CodeArtifact(files=[
            GeneratedFile(path="important.py", content="x = 1\n"),
            GeneratedFile(path="trivial.py", content="y = 2\n"),
        ])
        grader = DiffAccuracyGrader({
            "file_weights": {"important.py": 10.0, "trivial.py": 1.0},
        })
        # important.py matches, trivial.py doesn't
        ctx = _make_context(
            files=[
                GeneratedFile(path="important.py", content="x = 1\n"),
                GeneratedFile(path="trivial.py", content="completely different\n"),
            ],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        # The high-weight file matched, so overall score should be high
        assert result.score > 0.8

    @pytest.mark.asyncio
    async def test_missing_file_scores_zero(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\n")]
        )
        grader = DiffAccuracyGrader()
        # Generated artifact has no files
        ctx = _make_context(files=[], reference_artifact=ref)
        result = await grader.grade(ctx)
        assert result.details["per_file_scores"]["main.py"] < 0.5

    @pytest.mark.asyncio
    async def test_metadata_reference(self):
        grader = DiffAccuracyGrader()
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\n")],
            metadata={
                "expected_files": [
                    {"path": "main.py", "content": "x = 1\n"},
                ]
            },
        )
        result = await grader.grade(ctx)
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_reference_returns_error(self):
        grader = DiffAccuracyGrader()
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\n")],
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_no_artifact_returns_error(self):
        grader = DiffAccuracyGrader()
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "CodeArtifact" in result.error

    @pytest.mark.asyncio
    async def test_ast_mode(self):
        # Same AST but different formatting
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x=1\ny =  2\n")]
        )
        grader = DiffAccuracyGrader({"match_mode": "ast"})
        ctx = _make_context(
            files=[GeneratedFile(path="main.py", content="x = 1\ny = 2\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.score == 1.0


# ===================================================================
# DiffSizeGrader tests
# ===================================================================


class TestDiffSizeGrader:
    def test_registered(self):
        assert get_grader("diff_size") is DiffSizeGrader

    @pytest.mark.asyncio
    async def test_small_diff_passes(self):
        diff_text = (
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        grader = DiffSizeGrader({"max_total_changes": 10})
        ctx = _make_diff_context(diff=diff_text)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_large_diff_fails(self):
        # 5 additions, limit is 2
        lines = ["--- a/main.py\n", "+++ b/main.py\n", "@@ -1 +1 @@\n"]
        for i in range(5):
            lines.append(f"+line{i}\n")
        diff_text = "".join(lines)
        grader = DiffSizeGrader({"max_additions": 2})
        ctx = _make_diff_context(diff=diff_text)
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_additions_deletions_limits(self):
        diff_text = (
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old1\n"
            "-old2\n"
            "-old3\n"
            "+new1\n"
            "+new2\n"
            "+new3\n"
        )
        # 3 additions, 3 deletions
        grader = DiffSizeGrader({
            "max_additions": 5,
            "max_deletions": 2,
        })
        ctx = _make_diff_context(diff=diff_text)
        result = await grader.grade(ctx)
        # deletions=3 > max_deletions=2 → fail
        assert result.passed is False
        assert result.details["additions"] == 3
        assert result.details["deletions"] == 3

    @pytest.mark.asyncio
    async def test_from_artifact_diff(self):
        diff_text = (
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1 +1 @@\n"
            "-x = 1\n"
            "+x = 2\n"
        )
        grader = DiffSizeGrader({"max_total_changes": 10})
        ctx = _make_diff_context(diff=diff_text)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.details["additions"] == 1
        assert result.details["deletions"] == 1

    @pytest.mark.asyncio
    async def test_computed_from_reference(self):
        ref = CodeArtifact(
            files=[GeneratedFile(path="main.py", content="x = 1\n")]
        )
        grader = DiffSizeGrader({"max_total_changes": 10})
        ctx = _make_diff_context(
            files=[GeneratedFile(path="main.py", content="x = 2\n")],
            reference_artifact=ref,
        )
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.details["total_changes"] > 0

    @pytest.mark.asyncio
    async def test_no_diff_no_ref_returns_error(self):
        grader = DiffSizeGrader({"max_total_changes": 10})
        ctx = _make_diff_context(files=[GeneratedFile(path="main.py", content="x = 1\n")])
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_no_limits_passes(self):
        diff_text = (
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1 +1,10 @@\n"
        )
        diff_text += "".join(f"+line{i}\n" for i in range(10))
        grader = DiffSizeGrader()  # no limits
        ctx = _make_diff_context(diff=diff_text)
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0
