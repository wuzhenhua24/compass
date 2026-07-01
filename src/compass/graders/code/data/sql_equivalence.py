"""SQL Equivalence Grader for Data Agent evaluation.

Evaluates if generated SQL produces equivalent results to expected SQL.
"""

from __future__ import annotations

from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.registry import register_grader

from ._utils import compare_rows, extract_sql_from_text


@register_grader("sql_equivalence")
class SqlEquivalenceGrader(CodeGrader):
    """Evaluates if generated SQL produces equivalent results to expected SQL.

    This grader compares query results rather than SQL syntax, following
    OpenAI's approach of comparing "the SQL and the resulting data" rather
    than naive string matching.

    Config:
        expected_sql: str — The golden/expected SQL query.
        expected_result: list[dict] — Pre-computed expected results (optional).
        source: str — Where to get generated SQL (default "output_data.sql").
        key_columns: list[str] — Columns for row matching (optional).
        tolerance: float — Numeric comparison tolerance (default 1e-6).
        ignore_order: bool — Ignore row order (default True).
        ignore_extra_columns: bool — Allow extra columns (default True).
        executor: str — SQL executor type: "mock", "sqlite", "duckdb" (default "mock").
        executor_config: dict — Executor-specific configuration.

    Example YAML:
        graders:
          - name: sql_equivalence
            config:
              expected_sql: "SELECT name, COUNT(*) as cnt FROM users GROUP BY name"
              expected_result:
                - {name: "Alice", cnt: 10}
                - {name: "Bob", cnt: 5}
              key_columns: [name]
              ignore_order: true
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.expected_sql = self.config.get("expected_sql", "")
        self.expected_result = self.config.get("expected_result")
        self.source = self.config.get("source", "output_data.sql")
        self.key_columns = self.config.get("key_columns")
        self.tolerance = self.config.get("tolerance", 1e-6)
        self.ignore_order = self.config.get("ignore_order", True)
        self.ignore_extra_columns = self.config.get("ignore_extra_columns", True)
        self.executor = self.config.get("executor", "mock")
        self.executor_config = self.config.get("executor_config", {})

    async def grade(self, context: GradeContext) -> GradeResult:
        """Compare generated SQL results with expected results."""
        # Extract generated SQL
        generated_sql = self._extract_sql(context)
        if not generated_sql:
            return GradeResult(
                score=0.0,
                passed=False,
                error="No SQL found in output",
            )

        # Get expected results
        if self.expected_result is not None:
            expected_rows = self.expected_result
        elif self.expected_sql:
            # Execute expected SQL to get results
            expected_rows = await self._execute_sql(self.expected_sql)
            if expected_rows is None:
                return GradeResult(
                    score=0.0,
                    passed=False,
                    error="Failed to execute expected SQL",
                )
        else:
            return GradeResult(
                score=0.0,
                passed=False,
                error="No expected_sql or expected_result provided",
            )

        # Execute generated SQL
        actual_rows = await self._execute_sql(generated_sql)
        if actual_rows is None:
            return GradeResult(
                score=0.0,
                passed=False,
                details={"generated_sql": generated_sql},
                error="Failed to execute generated SQL",
            )

        # Compare results
        match, differences = compare_rows(
            actual_rows,
            expected_rows,
            key_columns=self.key_columns,
            tolerance=self.tolerance,
            ignore_order=self.ignore_order,
            ignore_extra_columns=self.ignore_extra_columns,
        )

        if match:
            return GradeResult(
                score=1.0,
                passed=True,
                details={
                    "row_count": len(actual_rows),
                    "generated_sql": generated_sql[:500],
                },
            )

        # Partial score based on matching percentage
        if expected_rows:
            match_ratio = max(0, 1 - len(differences) / len(expected_rows))
        else:
            match_ratio = 0.0

        return GradeResult(
            score=match_ratio,
            passed=False,
            details={
                "differences": differences[:10],
                "actual_row_count": len(actual_rows),
                "expected_row_count": len(expected_rows),
                "generated_sql": generated_sql[:500],
            },
            error=f"Result mismatch: {len(differences)} differences found",
        )

    def _extract_sql(self, context: GradeContext) -> str | None:
        """Extract SQL from context."""
        if not context.outcome:
            return None

        # Try source path
        if self.source.startswith("output_data."):
            key = self.source[12:]
            data = context.outcome.output_data
            if isinstance(data, dict):
                content = data.get(key)
                if isinstance(content, str):
                    return extract_sql_from_text(content) or content
                return content

        if self.source == "output_data":
            data = context.outcome.output_data
            if isinstance(data, dict) and "sql" in data:
                return data["sql"]
            if isinstance(data, str):
                return extract_sql_from_text(data)

        return None

    async def _execute_sql(self, sql: str) -> list[dict] | None:
        """Execute SQL and return results."""
        if self.executor == "mock":
            # Mock executor expects results in executor_config
            return self.executor_config.get("mock_results", [])

        elif self.executor == "sqlite":
            try:
                import sqlite3

                conn = sqlite3.connect(
                    self.executor_config.get("database", ":memory:")
                )
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(sql)
                rows = [dict(row) for row in cursor.fetchall()]
                conn.close()
                return rows
            except Exception:
                return None

        elif self.executor == "duckdb":
            try:
                import duckdb

                conn = duckdb.connect(
                    self.executor_config.get("database", ":memory:")
                )
                result = conn.execute(sql).fetchdf()
                conn.close()
                return result.to_dict(orient="records")
            except Exception:
                return None

        return None
