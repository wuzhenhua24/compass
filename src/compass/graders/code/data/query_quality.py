"""Query Quality Grader for Data Agent evaluation.

Evaluates SQL query quality and common anti-patterns.
"""

from __future__ import annotations

import re
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader

from ._utils import extract_sql_from_text, normalize_sql


@register_grader("query_quality")
class QueryQualityGrader(CodeGrader):
    """Evaluates SQL query quality and common anti-patterns.

    Checks for common SQL issues that can lead to incorrect or inefficient results:
    - Many-to-many joins without aggregation
    - Missing GROUP BY for aggregations
    - SELECT * usage
    - Cartesian products
    - Missing WHERE clauses on large tables

    Config:
        source: str — Where to get SQL (default "output_data.sql").
        checks: list[str] — Which checks to run (default all).
            Options: "select_star", "cartesian_join", "missing_where",
                    "aggregation_without_group", "suboptimal_join"
        severity_weights: dict — Weight for each check type.
        table_sizes: dict — Known table sizes for optimization hints.

    Example YAML:
        graders:
          - name: query_quality
            config:
              checks: [select_star, cartesian_join, aggregation_without_group]
              severity_weights:
                cartesian_join: 1.0
                select_star: 0.3
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    DEFAULT_CHECKS = [
        "select_star",
        "cartesian_join",
        "missing_where",
        "aggregation_without_group",
        "suboptimal_join",
    ]

    DEFAULT_WEIGHTS = {
        "select_star": 0.3,
        "cartesian_join": 1.0,
        "missing_where": 0.5,
        "aggregation_without_group": 1.0,
        "suboptimal_join": 0.7,
    }

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.source = self.config.get("source", "output_data.sql")
        self.checks = self.config.get("checks", self.DEFAULT_CHECKS)
        self.severity_weights = {
            **self.DEFAULT_WEIGHTS,
            **self.config.get("severity_weights", {}),
        }
        self.table_sizes = self.config.get("table_sizes", {})

    async def grade(self, context: GradeContext) -> GradeResult:
        """Analyze SQL for quality issues."""
        sql = self._extract_sql(context)
        if not sql:
            return GradeResult(
                score=0.0,
                passed=False,
                error="No SQL found in output",
            )

        issues = []
        normalized = normalize_sql(sql).upper()

        # Run checks
        if "select_star" in self.checks:
            if re.search(r"SELECT\s+\*", normalized):
                issues.append({
                    "type": "select_star",
                    "message": "SELECT * usage - specify columns explicitly",
                    "severity": self.severity_weights["select_star"],
                })

        if "cartesian_join" in self.checks:
            # Detect JOIN without ON/USING
            if re.search(r"JOIN\s+\w+\s*(?:,|JOIN)", normalized):
                issues.append({
                    "type": "cartesian_join",
                    "message": "Possible Cartesian join detected",
                    "severity": self.severity_weights["cartesian_join"],
                })
            # Detect comma joins without WHERE
            if "," in normalized and "JOIN" not in normalized:
                if "WHERE" not in normalized:
                    issues.append({
                        "type": "cartesian_join",
                        "message": "Comma join without WHERE clause",
                        "severity": self.severity_weights["cartesian_join"],
                    })

        if "missing_where" in self.checks:
            # Check for UPDATE/DELETE without WHERE
            if re.search(r"(UPDATE|DELETE)\s+", normalized) and "WHERE" not in normalized:
                issues.append({
                    "type": "missing_where",
                    "message": "UPDATE/DELETE without WHERE clause",
                    "severity": self.severity_weights["missing_where"],
                })

        if "aggregation_without_group" in self.checks:
            # Check for aggregations without GROUP BY
            has_agg = re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", normalized)
            has_group = "GROUP BY" in normalized
            has_non_agg_select = re.search(
                r"SELECT\s+(?!.*\b(COUNT|SUM|AVG|MIN|MAX)\b)[^,]+,",
                normalized
            )
            if has_agg and not has_group and has_non_agg_select:
                issues.append({
                    "type": "aggregation_without_group",
                    "message": "Aggregation with non-aggregated columns but no GROUP BY",
                    "severity": self.severity_weights["aggregation_without_group"],
                })

        if "suboptimal_join" in self.checks:
            # Check for NOT IN with subquery (often slow)
            if re.search(r"NOT\s+IN\s*\(\s*SELECT", normalized):
                issues.append({
                    "type": "suboptimal_join",
                    "message": "NOT IN with subquery - consider NOT EXISTS or LEFT JOIN",
                    "severity": self.severity_weights["suboptimal_join"],
                })

        # Calculate score
        if not issues:
            return GradeResult(
                score=1.0,
                passed=True,
                details={"checks_passed": self.checks},
            )

        total_severity = sum(i["severity"] for i in issues)
        max_severity = sum(self.severity_weights.get(c, 1.0) for c in self.checks)
        score = max(0, 1 - total_severity / max_severity)

        # Fail if any critical issue (severity >= 1.0)
        has_critical = any(i["severity"] >= 1.0 for i in issues)

        return GradeResult(
            score=score,
            passed=not has_critical and score >= 0.7,
            details={
                "issues": issues,
                "issue_count": len(issues),
            },
            error=f"Found {len(issues)} quality issue(s)" if issues else None,
        )

    def _extract_sql(self, context: GradeContext) -> str | None:
        """Extract SQL from context."""
        if not context.outcome:
            return None

        if self.source.startswith("output_data."):
            key = self.source[12:]
            data = context.outcome.output_data
            if isinstance(data, dict):
                content = data.get(key)
                if isinstance(content, str):
                    return extract_sql_from_text(content) or content

        if self.source == "output_data":
            data = context.outcome.output_data
            if isinstance(data, dict) and "sql" in data:
                return data["sql"]
            if isinstance(data, str):
                return extract_sql_from_text(data)

        return None
