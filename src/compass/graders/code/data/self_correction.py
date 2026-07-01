"""Self Correction Grader for Data Agent evaluation.

Evaluates the agent's ability to detect and recover from errors.
"""

from __future__ import annotations

import json
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader


@register_grader("self_correction")
class SelfCorrectionGrader(CodeGrader):
    """Evaluates the agent's ability to detect and recover from errors.

    OpenAI's Kepler "evaluates its own progress. If an intermediate result
    looks wrong, the agent investigates what went wrong, adjusts its approach,
    and tries again."

    This grader checks:
    - Error detection capability
    - Recovery attempts
    - Final success after retry

    Config:
        source: str — Where to get execution trace (default from transcript).
        require_recovery: bool — Must successfully recover to pass (default True).
        max_retries: int — Maximum acceptable retry count (default 3).
        check_error_explanation: bool — Check if errors are explained (default True).

    Example YAML:
        graders:
          - name: self_correction
            config:
              require_recovery: true
              max_retries: 3
              check_error_explanation: true
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.BOTH

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.require_recovery = self.config.get("require_recovery", True)
        self.max_retries = self.config.get("max_retries", 3)
        self.check_error_explanation = self.config.get("check_error_explanation", True)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Evaluate self-correction capability."""
        findings = {
            "errors_detected": 0,
            "recoveries_attempted": 0,
            "final_success": False,
            "retry_count": 0,
            "error_explanations": [],
        }

        # Analyze tool calls for error patterns
        tool_calls = context.tool_calls
        error_sequence = []

        for i, tc in enumerate(tool_calls):
            if tc.status == "error" or (tc.error and tc.error.get("message")):
                findings["errors_detected"] += 1
                error_sequence.append(("error", i, tc.error))

                # Check for explanation in subsequent calls
                if self.check_error_explanation and i + 1 < len(tool_calls):
                    next_tc = tool_calls[i + 1]
                    if next_tc.input and isinstance(next_tc.input, dict):
                        input_str = json.dumps(next_tc.input).lower()
                        if any(word in input_str for word in ["error", "fix", "retry", "adjust"]):
                            findings["error_explanations"].append(True)
                        else:
                            findings["error_explanations"].append(False)
            else:
                if error_sequence and error_sequence[-1][0] == "error":
                    # Success after error = recovery
                    findings["recoveries_attempted"] += 1
                error_sequence.append(("success", i, None))

        # Check final status
        if tool_calls:
            last_tc = tool_calls[-1]
            findings["final_success"] = last_tc.status != "error"

        # Count retries (similar tool calls in sequence)
        if len(tool_calls) >= 2:
            for i in range(1, len(tool_calls)):
                if tool_calls[i].tool_name == tool_calls[i-1].tool_name:
                    findings["retry_count"] += 1

        # Calculate score
        score_components = []

        # Recovery score
        if findings["errors_detected"] > 0:
            recovery_rate = findings["recoveries_attempted"] / findings["errors_detected"]
            score_components.append(recovery_rate)
        else:
            # No errors = perfect (or no opportunity to test)
            score_components.append(1.0)

        # Final success score
        score_components.append(1.0 if findings["final_success"] else 0.0)

        # Retry efficiency score
        if findings["retry_count"] <= self.max_retries:
            retry_score = 1.0
        else:
            retry_score = max(0, 1 - (findings["retry_count"] - self.max_retries) / self.max_retries)
        score_components.append(retry_score)

        final_score = sum(score_components) / len(score_components)

        # Determine pass/fail
        if self.require_recovery and findings["errors_detected"] > 0:
            passed = findings["final_success"] and findings["recoveries_attempted"] > 0
        else:
            passed = final_score >= 0.6

        return GradeResult(
            score=final_score,
            passed=passed,
            details=findings,
            error=(
                "Failed to recover from errors"
                if not passed and findings["errors_detected"] > 0
                else None
            ),
        )
