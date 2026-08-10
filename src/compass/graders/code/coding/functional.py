"""Functional code graders for evaluating execution outcomes.

ExitCodeGrader — checks that the process exit code matches an expected value.
TestRunnerGrader — runs a test command in a sandbox and checks results.
"""

from __future__ import annotations

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


@register_grader("exit_code_check")
class ExitCodeGrader(CodeGrader):
    """Check that the execution exit code matches an expected value.

    Scope: OUTCOME — reads the CodeArtifact's execution result.

    Config:
        expected_exit_code: int (default ``0``).
    """

    name = "exit_code_check"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.expected_exit_code: int = self.config.get("expected_exit_code", 0)

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

        execution = code_artifact.execution
        if execution is None:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                error="No execution result in CodeArtifact",
            )

        matched = execution.exit_code == self.expected_exit_code
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=matched,
            score=1.0 if matched else 0.0,
            details={
                "actual_exit_code": execution.exit_code,
                "expected_exit_code": self.expected_exit_code,
            },
            failure_tags=["exit_code_mismatch"] if not matched else [],
        )


@register_grader("test_runner")
class TestRunnerGrader(CodeGrader):
    """Run a test command in a sandbox and evaluate the results.

    Scope: OUTCOME — reads the CodeArtifact's files, writes them into a
    fresh sandbox, executes a test command, and evaluates the result.

    Config:
        test_command: str (default ``"python -m pytest"``).
        timeout: float (default ``60``).
        sandbox_type: str (default ``"local"``).
        sandbox_config: dict (default ``{}``).
        pass_pattern: str | None — regex that must match stdout for a pass.
        fail_pattern: str | None — regex that must *not* match stdout.
    """

    name = "test_runner"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.test_command: str = self.config.get("test_command", "python -m pytest")
        self.timeout: float = self.config.get("timeout", 60)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        self.pass_pattern: str | None = self.config.get("pass_pattern")
        self.fail_pattern: str | None = self.config.get("fail_pattern")

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

        files = code_artifact.files
        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        try:
            async with sandbox:
                # Write all files from the artifact into the sandbox
                for f in files:
                    await sandbox.write_file(f.path, f.content)

                # Execute the test command
                result = await sandbox.exec(
                    self.test_command, timeout=self.timeout
                )

            passed = result.exit_code == 0

            # Check optional pass_pattern
            if passed and self.pass_pattern is not None:
                if not re.search(self.pass_pattern, result.stdout):
                    passed = False

            # Check optional fail_pattern
            if passed and self.fail_pattern is not None:
                if re.search(self.fail_pattern, result.stdout):
                    passed = False

            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=1.0 if passed else 0.0,
                details={
                    "exit_code": result.exit_code,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration_ms": result.duration_ms,
                },
                failure_tags=["test_failure"] if not passed else [],
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
