"""Diff-based graders: accuracy comparison and diff size evaluation.

DiffAccuracyGrader — compares generated code to a reference file-by-file.
DiffSizeGrader     — evaluates whether the diff stays within size limits.
"""

from __future__ import annotations

import ast
import difflib
import re
from typing import Any

from compass.core.artifacts import CodeArtifact
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader


@register_grader("diff_accuracy")
class DiffAccuracyGrader(CodeGrader):
    """Compare generated code against a reference, file by file.

    Scope: OUTCOME — reads CodeArtifact files and compares to reference.

    Config:
        match_mode: str — ``"exact"``, ``"normalized"``, or ``"ast"``
            (default ``"normalized"``).
        ignore_whitespace: bool (default ``True``).
        ignore_comments: bool (default ``False``).
        file_weights: dict[str, float] — per-file weight overrides.
        pass_threshold: float (default ``0.9``).
    """

    name = "diff_accuracy"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.match_mode: str = self.config.get("match_mode", "normalized")
        self.ignore_whitespace: bool = self.config.get("ignore_whitespace", True)
        self.ignore_comments: bool = self.config.get("ignore_comments", False)
        self.file_weights: dict[str, float] = self.config.get("file_weights", {})
        self.pass_threshold: float = self.config.get("pass_threshold", 0.9)

    async def grade(self, context: GradeContext) -> GradeResult:
        code_artifact = context.code_artifact
        if code_artifact is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No CodeArtifact in outcome",
            )

        # Resolve reference files
        ref_files = self._get_reference_files(context)
        if ref_files is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No reference available (need reference_artifact or metadata['expected_files'])",
            )

        # Build lookup from generated files
        generated: dict[str, str] = {
            f.path: f.content for f in code_artifact.files
        }

        per_file_scores: dict[str, float] = {}
        diff_snippets: dict[str, str] = {}

        for path, ref_content in ref_files.items():
            gen_content = generated.get(path, "")
            similarity = self._compare(ref_content, gen_content, path)
            per_file_scores[path] = similarity

            if similarity < 1.0:
                diff_lines = list(
                    difflib.unified_diff(
                        ref_content.splitlines(keepends=True),
                        gen_content.splitlines(keepends=True),
                        fromfile=f"expected/{path}",
                        tofile=f"generated/{path}",
                    )
                )
                diff_snippets[path] = "".join(diff_lines[:20])

        # Weighted average
        total_weight = 0.0
        weighted_sum = 0.0
        for path, sim in per_file_scores.items():
            w = self.file_weights.get(path, 1.0)
            weighted_sum += sim * w
            total_weight += w

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        passed = score >= self.pass_threshold

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={
                "per_file_scores": per_file_scores,
                "diff_snippets": diff_snippets,
                "match_mode": self.match_mode,
                "pass_threshold": self.pass_threshold,
            },
        )

    def _get_reference_files(
        self, context: GradeContext
    ) -> dict[str, str] | None:
        """Resolve reference files from context, in priority order."""
        # 1. reference_artifact (CodeArtifact)
        if context.reference_artifact is not None:
            if isinstance(context.reference_artifact, CodeArtifact):
                return {
                    f.path: f.content
                    for f in context.reference_artifact.files
                }

        # 2. metadata["expected_files"]
        expected = context.metadata.get("expected_files")
        if expected is not None and isinstance(expected, list):
            return {
                item["path"]: item["content"]
                for item in expected
                if isinstance(item, dict) and "path" in item and "content" in item
            }

        return None

    def _compare(self, reference: str, generated: str, path: str) -> float:
        """Compare two strings using the configured match mode."""
        if self.match_mode == "exact":
            return 1.0 if reference == generated else 0.0

        if self.match_mode == "ast" and path.endswith(".py"):
            try:
                ref_ast = ast.dump(ast.parse(reference))
                gen_ast = ast.dump(ast.parse(generated))
                return 1.0 if ref_ast == gen_ast else 0.0
            except SyntaxError:
                # Fall back to normalized
                pass

        # "normalized" mode (also fallback for ast failures)
        ref_norm = self._normalize(reference)
        gen_norm = self._normalize(generated)
        return difflib.SequenceMatcher(None, ref_norm, gen_norm).ratio()

    def _normalize(self, text: str) -> str:
        """Normalize text for comparison."""
        if self.ignore_comments:
            text = self._strip_comments(text)
        if self.ignore_whitespace:
            lines = [line.strip() for line in text.splitlines()]
            lines = [line for line in lines if line]
            text = "\n".join(lines)
        return text

    @staticmethod
    def _strip_comments(text: str) -> str:
        """Strip Python-style # comments from each line."""
        lines = []
        for line in text.splitlines():
            # Remove inline comments (simple heuristic — not inside strings)
            stripped = re.sub(r"#.*$", "", line)
            lines.append(stripped)
        return "\n".join(lines)


@register_grader("diff_size")
class DiffSizeGrader(CodeGrader):
    """Evaluate whether a diff stays within size limits.

    Scope: OUTCOME — reads CodeArtifact.diff or computes from reference.

    Config:
        max_additions: int | None (default ``None``).
        max_deletions: int | None (default ``None``).
        max_total_changes: int | None (default ``None``).
    """

    name = "diff_size"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.max_additions: int | None = self.config.get("max_additions", None)
        self.max_deletions: int | None = self.config.get("max_deletions", None)
        self.max_total_changes: int | None = self.config.get(
            "max_total_changes", None
        )

    async def grade(self, context: GradeContext) -> GradeResult:
        code_artifact = context.code_artifact
        if code_artifact is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No CodeArtifact in outcome",
            )

        diff_text = self._get_diff(code_artifact, context)
        if diff_text is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No diff available (need artifact.diff or reference_artifact)",
            )

        additions, deletions = self._count_changes(diff_text)
        total_changes = additions + deletions

        # Calculate score — min(limit/actual) across all limits
        ratios: list[float] = []
        if self.max_additions is not None and additions > 0:
            ratios.append(min(1.0, self.max_additions / additions))
        if self.max_deletions is not None and deletions > 0:
            ratios.append(min(1.0, self.max_deletions / deletions))
        if self.max_total_changes is not None and total_changes > 0:
            ratios.append(min(1.0, self.max_total_changes / total_changes))

        if not ratios:
            # No limits set, or no changes — pass
            score = 1.0
        else:
            score = min(ratios)

        passed = score >= 1.0

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={
                "additions": additions,
                "deletions": deletions,
                "total_changes": total_changes,
                "max_additions": self.max_additions,
                "max_deletions": self.max_deletions,
                "max_total_changes": self.max_total_changes,
            },
        )

    def _get_diff(
        self, artifact: CodeArtifact, context: GradeContext
    ) -> str | None:
        """Get diff text from artifact or by computing from reference."""
        # 1. artifact.diff field
        if artifact.diff:
            return artifact.diff

        # 2. Compute from reference_artifact
        if context.reference_artifact is not None:
            if isinstance(context.reference_artifact, CodeArtifact):
                return self._compute_diff(
                    context.reference_artifact, artifact
                )

        return None

    @staticmethod
    def _compute_diff(
        reference: CodeArtifact, generated: CodeArtifact
    ) -> str:
        """Compute a unified diff between reference and generated files."""
        ref_files = {f.path: f.content for f in reference.files}
        gen_files = {f.path: f.content for f in generated.files}

        all_paths = sorted(set(ref_files) | set(gen_files))
        diff_lines: list[str] = []

        for path in all_paths:
            ref = ref_files.get(path, "").splitlines(keepends=True)
            gen = gen_files.get(path, "").splitlines(keepends=True)
            diff_lines.extend(
                difflib.unified_diff(ref, gen, fromfile=path, tofile=path)
            )

        return "".join(diff_lines)

    @staticmethod
    def _count_changes(diff_text: str) -> tuple[int, int]:
        """Count additions and deletions in a unified diff."""
        additions = 0
        deletions = 0
        for line in diff_text.splitlines():
            if line.startswith("+") and not line.startswith("+++"):
                additions += 1
            elif line.startswith("-") and not line.startswith("---"):
                deletions += 1
        return additions, deletions
