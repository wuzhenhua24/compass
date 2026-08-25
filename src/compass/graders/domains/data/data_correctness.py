"""Data Correctness Grader for Data Agent evaluation.

Evaluates if query/analysis results match expected data.
"""

from __future__ import annotations

from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader

from ._utils import compare_rows, compare_values


@register_grader("data_correctness")
class DataCorrectnessGrader(CodeGrader):
    """Evaluates if query/analysis results match expected data.

    Unlike sql_equivalence which compares SQL results, this grader directly
    compares output data against expected values, useful for evaluating
    the final answer regardless of how it was computed.

    Config:
        expected: dict | list — Expected result data.
        source: str — Where to get actual results (default "output_data").
        comparison_mode: str — "exact", "subset", "superset" (default "exact").
        key_columns: list[str] — Columns for row matching.
        tolerance: float — Numeric tolerance (default 1e-6).
        required_fields: list[str] — Fields that must be present and correct.

    Example YAML:
        graders:
          - name: data_correctness
            config:
              expected:
                total_users: 1000
                active_rate: 0.75
              required_fields: [total_users]
              tolerance: 0.01
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.expected = self.config.get("expected", {})
        self.source = self.config.get("source", "output_data")
        self.comparison_mode = self.config.get("comparison_mode", "exact")
        self.key_columns = self.config.get("key_columns")
        self.tolerance = self.config.get("tolerance", 1e-6)
        self.required_fields = self.config.get("required_fields", [])

    async def grade(self, context: GradeContext) -> GradeResult:
        """Compare actual results with expected data."""
        # Extract actual data
        actual = self._extract_data(context)
        if actual is None:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"No data found at source: {self.source}",
            )

        # Handle different data structures
        if isinstance(self.expected, list) and isinstance(actual, list):
            return self._compare_list_results(actual, self.expected)
        elif isinstance(self.expected, dict) and isinstance(actual, dict):
            return self._compare_dict_results(actual, self.expected)
        else:
            # Direct value comparison
            if compare_values(actual, self.expected, self.tolerance):
                return GradeResult(score=1.0, passed=True)
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"Value mismatch: actual={actual}, expected={self.expected}",
            )

    def _extract_data(self, context: GradeContext) -> Any:
        """Extract data from context."""
        if not context.outcome:
            return None

        if self.source == "output_data":
            return context.outcome.output_data

        if self.source.startswith("output_data."):
            key = self.source[12:]
            data = context.outcome.output_data
            if isinstance(data, dict):
                return data.get(key)

        return None

    def _compare_dict_results(
        self, actual: dict[str, Any], expected: dict
    ) -> GradeResult:
        """Compare dictionary results."""
        errors = []
        correct_count = 0
        total_count = len(expected)

        # Check required fields first
        for field in self.required_fields:
            if field not in actual:
                errors.append(f"Missing required field: {field}")
            elif field in expected:
                if not compare_values(actual[field], expected[field], self.tolerance):
                    errors.append(
                        f"Required field '{field}' mismatch: "
                        f"actual={actual[field]}, expected={expected[field]}"
                    )

        # Compare all expected fields
        for key, expected_val in expected.items():
            if key not in actual:
                if self.comparison_mode == "exact":
                    errors.append(f"Missing field: {key}")
            else:
                if compare_values(actual[key], expected_val, self.tolerance):
                    correct_count += 1
                else:
                    errors.append(
                        f"Field '{key}' mismatch: "
                        f"actual={actual[key]}, expected={expected_val}"
                    )

        # Check for required field failures
        required_errors = [e for e in errors if "required" in e.lower()]
        if required_errors:
            return GradeResult(
                score=0.0,
                passed=False,
                details={"errors": errors},
                error="; ".join(required_errors),
            )

        score = correct_count / total_count if total_count > 0 else 1.0
        passed = len(errors) == 0

        return GradeResult(
            score=score,
            passed=passed,
            details={
                "correct_fields": correct_count,
                "total_fields": total_count,
                "errors": errors[:10] if errors else [],
            },
            error="; ".join(errors[:3]) if errors else None,
        )

    def _compare_list_results(
        self, actual: list[Any], expected: list
    ) -> GradeResult:
        """Compare list results."""
        match, differences = compare_rows(
            actual if isinstance(actual[0], dict) else [{"value": v} for v in actual],
            expected if isinstance(expected[0], dict) else [{"value": v} for v in expected],
            key_columns=self.key_columns,
            tolerance=self.tolerance,
            ignore_order=True,
            ignore_extra_columns=True,
        )

        if match:
            return GradeResult(
                score=1.0,
                passed=True,
                details={"row_count": len(actual)},
            )

        score = max(0, 1 - len(differences) / max(len(expected), 1))
        return GradeResult(
            score=score,
            passed=False,
            details={"differences": differences[:10]},
            error=f"Data mismatch: {len(differences)} differences",
        )
