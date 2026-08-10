"""Code quality graders: linting and type checking.

LintGrader   — runs a lint command (e.g. flake8) and counts issues.
TypeCheckGrader — runs a type checker (e.g. mypy) and counts errors.
"""

from __future__ import annotations

import json
import re
from typing import Any

from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader
from compass.sandbox import get_sandbox_class


@register_grader("lint")
class LintGrader(CodeGrader):
    """Run a lint command on generated code and evaluate the results.

    Scope: OUTCOME — reads the CodeArtifact's files, writes them into a
    sandbox, executes a lint command, and parses the output.

    Config:
        lint_command: str (default ``"python -m flake8 --format=json ."``).
        max_errors: int (default ``0``).
        max_warnings: int | None (default ``None`` — unlimited).
        severity_weights: dict mapping code prefix to weight.
        timeout: float (default ``60``).
        sandbox_type: str (default ``"local"``).
        sandbox_config: dict (default ``{}``).
    """

    name = "lint"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.lint_command: str = self.config.get(
            "lint_command", "python -m flake8 --format=json ."
        )
        self.max_errors: int = self.config.get("max_errors", 0)
        self.max_warnings: int | None = self.config.get("max_warnings", None)
        self.severity_weights: dict[str, float] = self.config.get(
            "severity_weights", {"E": 1.0, "W": 0.5, "F": 1.0, "C": 0.25}
        )
        self.timeout: float = self.config.get("timeout", 60)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        # If True, fail when the lint tool itself fails (e.g., not installed)
        self.fail_on_tool_error: bool = self.config.get("fail_on_tool_error", True)

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

        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        try:
            async with sandbox:
                for f in code_artifact.files:
                    await sandbox.write_file(f.path, f.content)

                result = await sandbox.exec(
                    self.lint_command, timeout=self.timeout
                )

            stdout = result.stdout
            stderr = result.stderr if hasattr(result, "stderr") else ""
            exit_code = result.exit_code

            # Parse output: try JSON first, then text fallback
            issues = self._parse_flake8_json(stdout)
            if issues is None:
                issues = self._parse_flake8_text(stdout)

            # Detect tool execution failure:
            # flake8 returns exit_code=1 when issues found, exit_code=0 when clean.
            # If exit_code != 0 AND no issues parsed AND stdout is empty/minimal,
            # the tool likely failed to run (not installed, crashed, etc.)
            tool_error = None
            if exit_code != 0 and len(issues) == 0 and len(stdout.strip()) == 0:
                tool_error = f"Lint tool failed (exit_code={exit_code})"
                if stderr:
                    tool_error += f": {stderr[:500]}"

            if tool_error and self.fail_on_tool_error:
                return GradeResult(
                    name=self.name,
                    grader_type=self.grader_type,
                    grader_scope=self.grader_scope,
                    passed=False,
                    score=0.0,
                    details={
                        "exit_code": exit_code,
                        "stderr": stderr[:2000] if stderr else None,
                        "stdout": stdout[:2000],
                    },
                    error=tool_error,
                )

            # Classify issues by severity (first char of code)
            issues_by_severity: dict[str, int] = {}
            for issue in issues:
                code = issue.get("code", "")
                prefix = code[0] if code else "E"
                issues_by_severity[prefix] = issues_by_severity.get(prefix, 0) + 1

            error_count = sum(
                count
                for prefix, count in issues_by_severity.items()
                if prefix in ("E", "F")
            )
            warning_count = sum(
                count
                for prefix, count in issues_by_severity.items()
                if prefix in ("W", "C")
            )
            total_issues = len(issues)

            # Determine pass/fail
            passed = error_count <= self.max_errors
            if passed and self.max_warnings is not None:
                passed = warning_count <= self.max_warnings

            # Calculate score
            if total_issues == 0:
                score = 1.0
            else:
                weighted_sum = sum(
                    self.severity_weights.get(
                        issue.get("code", "E")[0] if issue.get("code") else "E",
                        1.0,
                    )
                    for issue in issues
                )
                score = 1.0 - min(1.0, weighted_sum / (self.max_errors + 1))

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=score,
                details={
                    "error_count": error_count,
                    "warning_count": warning_count,
                    "total_issues": total_issues,
                    "issues_by_severity": issues_by_severity,
                    "issues": issues[:50],
                    "exit_code": exit_code,
                    "stderr": stderr[:2000] if stderr else None,
                    "stdout": stdout[:2000],
                    "tool_error": tool_error,
                },
            )

        except Exception as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=str(e),
            )

    @staticmethod
    def _parse_flake8_json(stdout: str) -> list[dict[str, Any]] | None:
        """Try to parse flake8 JSON output. Returns None on failure."""
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return None

        issues: list[dict[str, Any]] = []
        if isinstance(data, dict):
            # flake8 JSON format: {filename: [issue, ...]}
            for _file, file_issues in data.items():
                if isinstance(file_issues, list):
                    for item in file_issues:
                        issues.append({
                            "file": item.get("filename", _file),
                            "line": item.get("line_number", 0),
                            "col": item.get("column_number", 0),
                            "code": item.get("code", ""),
                            "message": item.get("text", ""),
                        })
        elif isinstance(data, list):
            for item in data:
                issues.append({
                    "file": item.get("filename", ""),
                    "line": item.get("line_number", 0),
                    "col": item.get("column_number", 0),
                    "code": item.get("code", ""),
                    "message": item.get("text", ""),
                })
        return issues

    @staticmethod
    def _parse_flake8_text(stdout: str) -> list[dict[str, Any]]:
        """Parse flake8 text output as fallback."""
        pattern = re.compile(r"(.+):(\d+):(\d+): (\w+) (.+)")
        issues: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            m = pattern.match(line.strip())
            if m:
                issues.append({
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "col": int(m.group(3)),
                    "code": m.group(4),
                    "message": m.group(5),
                })
        return issues


@register_grader("type_check")
class TypeCheckGrader(CodeGrader):
    """Run a type checker on generated code and evaluate the results.

    Scope: OUTCOME — reads the CodeArtifact's files, writes them into a
    sandbox, executes a type check command, and parses the output.

    Config:
        type_check_command: str (default ``"python -m mypy --no-error-summary ."``).
        max_errors: int (default ``0``).
        strict: bool (default ``False``).
        timeout: float (default ``60``).
        sandbox_type: str (default ``"local"``).
        sandbox_config: dict (default ``{}``).
    """

    name = "type_check"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.type_check_command: str = self.config.get(
            "type_check_command", "python -m mypy --no-error-summary ."
        )
        self.max_errors: int = self.config.get("max_errors", 0)
        self.strict: bool = self.config.get("strict", False)
        self.timeout: float = self.config.get("timeout", 60)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        # If True, fail when the type check tool itself fails (e.g., not installed)
        self.fail_on_tool_error: bool = self.config.get("fail_on_tool_error", True)

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

        command = self.type_check_command
        if self.strict and "--strict" not in command:
            command = command + " --strict"

        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        try:
            async with sandbox:
                for f in code_artifact.files:
                    await sandbox.write_file(f.path, f.content)

                result = await sandbox.exec(command, timeout=self.timeout)

            stdout = result.stdout
            stderr = result.stderr if hasattr(result, "stderr") else ""
            exit_code = result.exit_code

            errors = self._parse_mypy_output(stdout)
            error_count = len(errors)

            # Detect tool execution failure:
            # mypy returns exit_code=1 when errors found, exit_code=0 when clean.
            # exit_code=2 means mypy itself had an error (e.g., not installed, bad args).
            # If exit_code != 0 AND no errors parsed AND stdout is empty/minimal,
            # the tool likely failed to run.
            tool_error = None
            if exit_code != 0 and error_count == 0 and len(stdout.strip()) == 0:
                tool_error = f"Type check tool failed (exit_code={exit_code})"
                if stderr:
                    tool_error += f": {stderr[:500]}"

            if tool_error and self.fail_on_tool_error:
                return GradeResult(
                    name=self.name,
                    grader_type=self.grader_type,
                    grader_scope=self.grader_scope,
                    passed=False,
                    score=0.0,
                    details={
                        "exit_code": exit_code,
                        "stderr": stderr[:2000] if stderr else None,
                        "stdout": stdout[:2000],
                        "strict": self.strict,
                    },
                    error=tool_error,
                )

            # Determine pass/fail
            passed = error_count <= self.max_errors

            # Calculate score
            if error_count <= self.max_errors:
                score = 1.0
            else:
                score = max(0.0, 1.0 - (error_count - self.max_errors) / error_count)

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=score,
                details={
                    "error_count": error_count,
                    "max_errors": self.max_errors,
                    "errors": errors[:50],
                    "strict": self.strict,
                    "exit_code": exit_code,
                    "stderr": stderr[:2000] if stderr else None,
                    "tool_error": tool_error,
                },
            )

        except Exception as e:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error=str(e),
            )

    @staticmethod
    def _parse_mypy_output(stdout: str) -> list[dict[str, Any]]:
        """Parse mypy output for error lines."""
        pattern = re.compile(r"(.+):(\d+): error: (.+)")
        errors: list[dict[str, Any]] = []
        for line in stdout.splitlines():
            m = pattern.match(line.strip())
            if m:
                errors.append({
                    "file": m.group(1),
                    "line": int(m.group(2)),
                    "message": m.group(3),
                })
        return errors
