"""Security scanning grader.

SecurityScanGrader — pattern-based and external scanner (e.g. bandit) checks.
"""

from __future__ import annotations

import json
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
from compass.sandbox import get_sandbox_class

_SEVERITY_LEVELS = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


@register_grader("security_scan")
class SecurityScanGrader(CodeGrader):
    """Two-layer security scanning: regex patterns + external scanner.

    Scope: OUTCOME — reads CodeArtifact files and optionally runs an
    external scanner (e.g. bandit) in a sandbox.

    Config:
        scan_command: str | None (default ``"python -m bandit -r -f json ."``).
            Set to ``None`` to skip external scanning.
        max_issues: int (default ``0``).
        min_severity: str — ``"LOW"``, ``"MEDIUM"``, or ``"HIGH"``
            (default ``"LOW"``).
        blocked_patterns: list[str] — regex patterns to flag as HIGH severity.
        timeout: float (default ``60``).
        sandbox_type: str (default ``"local"``).
        sandbox_config: dict (default ``{}``).
    """

    name = "security_scan"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.scan_command: str | None = self.config.get(
            "scan_command", "python -m bandit -r -f json ."
        )
        self.max_issues: int = self.config.get("max_issues", 0)
        self.min_severity: str = self.config.get("min_severity", "LOW")
        self.blocked_patterns: list[str] = self.config.get(
            "blocked_patterns", []
        )
        self.timeout: float = self.config.get("timeout", 60)
        self.sandbox_type: str = self.config.get("sandbox_type", "local")
        self.sandbox_config: dict[str, Any] = self.config.get(
            "sandbox_config", {}
        )
        # If True, fail when the security scanner itself fails (e.g., not installed)
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

        all_issues: list[dict[str, Any]] = []
        pattern_match_count = 0
        scanner_available = True
        tool_error = None
        scanner_exit_code = None
        scanner_stderr = None

        # --- Layer 1: Pattern scanning (always runs) ---
        for f in code_artifact.files:
            for pat_str in self.blocked_patterns:
                try:
                    pat = re.compile(pat_str)
                except re.error:
                    continue  # skip invalid regex silently
                for i, line in enumerate(f.content.splitlines(), 1):
                    if pat.search(line):
                        all_issues.append({
                            "file": f.path,
                            "line": i,
                            "severity": "HIGH",
                            "message": f"Blocked pattern matched: {pat_str}",
                            "source": "pattern",
                        })
                        pattern_match_count += 1

        # --- Layer 2: External scanner (if configured) ---
        if self.scan_command is not None:
            try:
                sandbox_cls = get_sandbox_class(self.sandbox_type)
                sandbox = sandbox_cls(self.sandbox_config)

                async with sandbox:
                    for f in code_artifact.files:
                        await sandbox.write_file(f.path, f.content)

                    result = await sandbox.exec(
                        self.scan_command, timeout=self.timeout
                    )

                stdout = result.stdout
                scanner_stderr = result.stderr if hasattr(result, "stderr") else ""
                scanner_exit_code = result.exit_code

                scanner_issues = self._parse_bandit_json(stdout)
                all_issues.extend(scanner_issues)

                # Detect tool execution failure:
                # bandit returns exit_code=1 when issues found, exit_code=0 when clean.
                # If exit_code != 0 AND no issues parsed AND stdout doesn't look like
                # valid bandit JSON, the tool likely failed to run.
                if (
                    scanner_exit_code != 0
                    and len(scanner_issues) == 0
                    and not stdout.strip().startswith("{")
                ):
                    tool_error = f"Security scanner failed (exit_code={scanner_exit_code})"
                    if scanner_stderr:
                        tool_error += f": {scanner_stderr[:500]}"
                    scanner_available = False

            except Exception as e:
                scanner_available = False
                tool_error = f"Security scanner exception: {e}"

        # If scanner failed and fail_on_tool_error is True, return failure
        if tool_error and self.fail_on_tool_error:
            return GradeResult(
                name=self.name,
                grader_type=self.grader_type,
                grader_scope=self.grader_scope,
                passed=False,
                score=0.0,
                details={
                    "pattern_matches": pattern_match_count,
                    "scanner_available": scanner_available,
                    "exit_code": scanner_exit_code,
                    "stderr": scanner_stderr[:2000] if scanner_stderr else None,
                },
                error=tool_error,
            )

        # --- Filter by severity ---
        min_level = _SEVERITY_LEVELS.get(self.min_severity.upper(), 0)
        filtered = [
            issue
            for issue in all_issues
            if _SEVERITY_LEVELS.get(
                issue.get("severity", "LOW").upper(), 0
            )
            >= min_level
        ]
        filtered_count = len(filtered)

        # --- Scoring ---
        passed = filtered_count <= self.max_issues
        if filtered_count <= self.max_issues:
            score = 1.0
        else:
            score = max(
                0.0,
                1.0
                - (filtered_count - self.max_issues)
                / max(filtered_count, 1),
            )

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed,
            score=score,
            details={
                "total_issues": len(all_issues),
                "filtered_issues": filtered_count,
                "issues": filtered[:50],
                "pattern_matches": pattern_match_count,
                "scanner_available": scanner_available,
                "exit_code": scanner_exit_code,
                "stderr": scanner_stderr[:2000] if scanner_stderr else None,
                "tool_error": tool_error,
            },
        )

    @staticmethod
    def _parse_bandit_json(stdout: str) -> list[dict[str, Any]]:
        """Parse bandit JSON output into a list of issue dicts."""
        try:
            data = json.loads(stdout)
        except (json.JSONDecodeError, ValueError):
            return []

        results = data.get("results", [])
        issues: list[dict[str, Any]] = []
        for item in results:
            issues.append({
                "file": item.get("filename", ""),
                "line": item.get("line_number", 0),
                "severity": item.get("issue_severity", "LOW").upper(),
                "message": item.get("issue_text", ""),
                "source": "bandit",
            })
        return issues
