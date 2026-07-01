"""Integration grader — run external test scripts and compute proportional scores.

Inspired by Stripe's agent benchmark approach: graders are standalone test
suites (rspec, pytest, shell scripts, etc.) that verify the final state of an
environment after an agent has modified it.  The score is the fraction of
individual checks that pass, giving partial credit rather than binary pass/fail.

Unlike ``TestRunnerGrader`` which writes ``CodeArtifact`` files into a sandbox,
``IntegrationGrader`` runs a grading script against an *existing* working
directory, Docker container, or running service — evaluating what the agent
actually did to the environment.
"""

from __future__ import annotations

import json
import re
from typing import Any

from compass.core.artifacts import ExecutionResult
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader
from compass.sandbox import get_sandbox_class

# ---------------------------------------------------------------------------
# Result parsers — extract (passed, total) from test-runner output
# ---------------------------------------------------------------------------

def _parse_pytest(stdout: str) -> tuple[int, int] | None:
    """Parse pytest summary line, e.g. ``5 passed, 2 failed``."""
    passed = len(re.findall(r"(\d+) passed", stdout))
    failed = len(re.findall(r"(\d+) failed", stdout))
    if passed or failed:
        m_passed = re.search(r"(\d+) passed", stdout)
        m_failed = re.search(r"(\d+) failed", stdout)
        p = int(m_passed.group(1)) if m_passed else 0
        f = int(m_failed.group(1)) if m_failed else 0
        total = p + f
        if total > 0:
            return p, total
    return None


def _parse_rspec(stdout: str) -> tuple[int, int] | None:
    """Parse rspec summary, e.g. ``10 examples, 2 failures``."""
    m = re.search(r"(\d+)\s+examples?,\s*(\d+)\s+failures?", stdout)
    if m:
        total = int(m.group(1))
        failures = int(m.group(2))
        if total > 0:
            return total - failures, total
    return None


def _parse_go_test(stdout: str) -> tuple[int, int] | None:
    """Parse ``go test`` output (ok / FAIL lines)."""
    ok = len(re.findall(r"^ok\s", stdout, re.MULTILINE))
    fail = len(re.findall(r"^FAIL\s", stdout, re.MULTILINE))
    total = ok + fail
    if total > 0:
        return ok, total
    return None


def _parse_json(stdout: str) -> tuple[int, int] | None:
    """Parse JSON output with ``passed``/``total`` or a ``tests`` array.

    Supports two shapes:

    1. ``{"passed": 3, "total": 5}``   (or ``passed_tests`` / ``total_tests``)
    2. ``{"tests": [{"status": "passed"}, ...]}``  (Stripe-style)
    """
    try:
        data = json.loads(stdout)
    except (json.JSONDecodeError, TypeError):
        return None

    if not isinstance(data, dict):
        return None

    # Shape 1: explicit counts
    for pk in ("passed", "passed_tests"):
        for tk in ("total", "total_tests"):
            if pk in data and tk in data:
                return int(data[pk]), int(data[tk])

    # Shape 2: tests array
    tests = data.get("tests")
    if isinstance(tests, list) and tests:
        total = len(tests)
        passed = sum(
            1 for t in tests
            if isinstance(t, dict) and t.get("status") == "passed"
        )
        return passed, total

    return None


def _parse_generic(stdout: str) -> tuple[int, int] | None:
    """Last-resort: look for ``X/Y passed`` or ``X of Y tests passed``."""
    m = re.search(r"(\d+)\s*/\s*(\d+)\s*passed", stdout, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)\s+of\s+(\d+)\s+tests?\s+passed", stdout, re.IGNORECASE)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None


_PARSERS: dict[str, Any] = {
    "pytest": _parse_pytest,
    "rspec": _parse_rspec,
    "go_test": _parse_go_test,
    "json": _parse_json,
    "generic": _parse_generic,
}

_AUTO_ORDER = ["json", "pytest", "rspec", "go_test", "generic"]


def parse_test_output(
    stdout: str,
    output_format: str = "auto",
) -> tuple[int, int] | None:
    """Extract (passed, total) from test command output.

    Args:
        stdout: Combined stdout of the test command.
        output_format: One of ``"auto"``, ``"pytest"``, ``"rspec"``,
            ``"go_test"``, ``"json"``, ``"generic"``.

    Returns:
        ``(passed, total)`` or ``None`` if parsing fails.
    """
    if output_format != "auto":
        parser = _PARSERS.get(output_format)
        if parser is None:
            return None
        return parser(stdout)

    for name in _AUTO_ORDER:
        result = _PARSERS[name](stdout)
        if result is not None:
            return result
    return None


# ---------------------------------------------------------------------------
# IntegrationGrader
# ---------------------------------------------------------------------------

@register_grader("integration_test")
class IntegrationGrader(CodeGrader):
    """Run an external test script and score by proportion of tests passed.

    Unlike ``TestRunnerGrader`` (which writes artifact files into a sandbox),
    this grader runs a grading script against an *existing* environment — a
    working directory the agent modified, a running service, or a Docker
    container.

    The score equals ``passed_tests / total_tests``, giving partial credit.

    Scope: OUTCOME — evaluates the final state produced by the agent.

    Config:
        script:         str   — shell command to execute (required).
        workdir:        str   — working directory for the script (default: sandbox cwd).
        timeout:        float — seconds before the script is killed (default: 120).
        env:            dict  — extra environment variables.
        output_format:  str   — ``"auto"`` | ``"pytest"`` | ``"rspec"`` |
                                ``"go_test"`` | ``"json"`` | ``"generic"``
                                (default: ``"auto"``).
        pass_threshold: float — minimum score to count as passed (default: 1.0).
        sandbox_type:   str   — sandbox backend (default: ``"local"``).
        sandbox_config: dict  — extra sandbox config (default: ``{}``).
        setup_commands: list[str] — commands to run before the grading script
                                    (e.g. install deps).
        leak_patterns:  list[str] — strings whose presence in transcript
                                    stdout invalidates the trial (Stripe-style
                                    leak detection).

    YAML example::

        graders:
          - name: integration_test
            config:
              script: "./grader/grade.sh"
              workdir: "/workdir"
              output_format: json
              pass_threshold: 0.8
              timeout: 300
              env:
                STRIPE_SECRET_KEY: "sk_test_..."
              setup_commands:
                - "bundle install"
              leak_patterns:
                - "EVAL_LEAK_CHECK_abc123"
    """

    name = "integration_test"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.script: str = self.config.get("script", "")
        self.workdir: str | None = self.config.get("workdir")
        self.timeout: float = self.config.get("timeout", 120)
        self.env: dict[str, str] = self.config.get("env", {})
        self.output_format: str = self.config.get("output_format", "auto")
        self.pass_threshold: float = self.config.get("pass_threshold", 1.0)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get("sandbox_config", {})
        self.setup_commands: list[str] = self.config.get("setup_commands", [])
        self.leak_patterns: list[str] = self.config.get("leak_patterns", [])

    def validate_config(self) -> bool:
        """Script must be specified."""
        return bool(self.script)

    async def grade(self, context: GradeContext) -> GradeResult:
        if not self.script:
            return self._error_result("Config 'script' is required")

        # --- Leak detection (check transcript for answer leakage) ---
        if (self.leak_patterns or context.leak_markers) and context.has_transcript:
            leak_result = context.check_leaks(extra_patterns=self.leak_patterns)
            if leak_result.has_leaks:
                return GradeResult(
                    name=self.name,
                    grader_type=self.grader_type,
                    grader_scope=self.grader_scope,
                    passed=False,
                    score=0.0,
                    details=leak_result.to_dict(),
                    failure_tags=["answer_leak"],
                    reasoning=(
                        "Trial invalidated: agent accessed grader/solution data "
                        f"(leaked: {leak_result.leaked})"
                    ),
                )

        # --- Run the grading script ---
        sandbox_cls = get_sandbox_class(self.sandbox_type)
        sandbox = sandbox_cls(self.sandbox_config)

        try:
            async with sandbox:
                # Copy environment workdir into sandbox if specified
                if self.workdir:
                    await sandbox.copy_in(self.workdir, ".")

                # Run setup commands
                for cmd in self.setup_commands:
                    setup_result = await sandbox.exec(
                        cmd,
                        timeout=self.timeout,
                        env=self.env or None,
                    )
                    if setup_result.exit_code != 0:
                        return GradeResult(
                            name=self.name,
                            grader_type=self.grader_type,
                            grader_scope=self.grader_scope,
                            passed=False,
                            score=0.0,
                            details={
                                "phase": "setup",
                                "command": cmd,
                                "exit_code": setup_result.exit_code,
                                "stdout": setup_result.stdout,
                                "stderr": setup_result.stderr,
                            },
                            failure_tags=["setup_failure"],
                            reasoning=f"Setup command failed: {cmd}",
                        )

                # Run the grading script
                result = await sandbox.exec(
                    self.script,
                    timeout=self.timeout,
                    env=self.env or None,
                )

            return self._build_result(result)

        except Exception as e:
            return self._error_result(str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_result(self, result: ExecutionResult) -> GradeResult:
        """Parse test output and build a GradeResult with proportional score."""
        combined_output = result.stdout
        if result.stderr:
            combined_output += "\n" + result.stderr

        parsed = parse_test_output(combined_output, self.output_format)

        if parsed is not None:
            passed_count, total_count = parsed
            score = passed_count / total_count if total_count > 0 else 0.0
            passed = score >= self.pass_threshold
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=passed,
                score=score,
                details={
                    "tests_passed": passed_count,
                    "tests_total": total_count,
                    "exit_code": result.exit_code,
                    "output_format": self.output_format,
                    "pass_threshold": self.pass_threshold,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration_ms": result.duration_ms,
                },
                failure_tags=self._failure_tags(passed, passed_count, total_count),
                reasoning=self._reasoning(passed_count, total_count, passed),
            )

        # Parsing failed — fall back to exit-code based scoring
        exit_passed = result.exit_code == 0
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=exit_passed,
            score=1.0 if exit_passed else 0.0,
            details={
                "tests_passed": None,
                "tests_total": None,
                "exit_code": result.exit_code,
                "output_format": self.output_format,
                "parse_fallback": True,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration_ms": result.duration_ms,
            },
            failure_tags=["test_failure"] if not exit_passed else [],
            reasoning=(
                "Could not parse test counts from output; "
                f"fell back to exit code ({result.exit_code})."
            ),
        )

    def _failure_tags(
        self, passed: bool, passed_count: int, total_count: int,
    ) -> list[str]:
        tags: list[str] = []
        if not passed:
            tags.append("test_failure")
            if passed_count == 0:
                tags.append("all_tests_failed")
            else:
                tags.append("partial_failure")
        return tags

    def _reasoning(
        self, passed_count: int, total_count: int, passed: bool,
    ) -> str:
        pct = (passed_count / total_count * 100) if total_count > 0 else 0
        status = "PASSED" if passed else "FAILED"
        return (
            f"{status}: {passed_count}/{total_count} tests passed "
            f"({pct:.0f}%, threshold {self.pass_threshold * 100:.0f}%)"
        )

    def _error_result(self, error: str) -> GradeResult:
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=0.0,
            error=error,
        )
