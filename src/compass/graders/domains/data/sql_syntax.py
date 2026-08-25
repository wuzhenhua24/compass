"""SQL syntax validation.

Lives with the other SQL graders rather than with the structural ones: it asks
whether a string is valid SQL, which is a question about a domain, not about a
format. ``sqlparse`` is an optional dependency (``compass[data]``) and the
grader reports its absence rather than raising.
"""

from __future__ import annotations

import re
from typing import Any

from compass.graders.base import CodeGrader, GradeContext, GradeResult, GraderScope, GraderType
from compass.graders.content import extract_content
from compass.graders.registry import register_grader


def _stmt_type(stmt: Any) -> str:
    """The statement kind, at the boundary where sqlparse stops being typed.

    ``sqlparse`` ships ``py.typed`` but leaves ``Statement.get_type`` without
    a return annotation, so calling it from strict code needs the crossing
    written down once rather than cast at each of the two call sites.
    """
    return str(stmt.get_type())


@register_grader("sql_syntax")
class SqlSyntaxGrader(CodeGrader):
    """Validates SQL syntax correctness.

    Config:
        source: str — Where to get SQL from (default "text_artifact").
        dialect: str — SQL dialect for validation (default "generic").
            Options: "generic", "mysql", "postgresql", "sqlite", "tsql".
        allowed_statements: list[str] — Allowed statement types (default all).
            Options: "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", etc.
        forbidden_keywords: list[str] — Keywords that should not appear.
        require_semicolon: bool — Require statements to end with semicolon.
        max_statements: int — Maximum number of statements allowed (0 = unlimited).

    Example YAML:
        graders:
          - name: sql_syntax
            config:
              source: text_artifact
              dialect: postgresql
              allowed_statements: [SELECT]
              forbidden_keywords: [DROP, TRUNCATE, DELETE]
    """

    grader_type = GraderType.CODE
    grader_scope = GraderScope.OUTCOME

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.source = self.config.get("source", "text_artifact")
        self.dialect = self.config.get("dialect", "generic")
        self.allowed_statements = self.config.get("allowed_statements", [])
        self.forbidden_keywords = [k.upper() for k in self.config.get("forbidden_keywords", [])]
        self.require_semicolon = self.config.get("require_semicolon", False)
        self.max_statements = self.config.get("max_statements", 0)

    async def grade(self, context: GradeContext) -> GradeResult:
        """Validate SQL syntax."""
        try:
            import sqlparse
        except ImportError:
            return GradeResult(
                score=0.0,
                passed=False,
                error="sqlparse library not installed. Run: pip install sqlparse",
            )

        # Extract content
        content = extract_content(context, self.source)
        if content is None:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"No content found at source: {self.source}",
            )

        if not isinstance(content, str):
            content = str(content)

        # Extract SQL from code blocks if present
        code_block_match = re.search(r"```(?:sql)?\s*([\s\S]*?)```", content)
        if code_block_match:
            content = code_block_match.group(1).strip()

        if not content.strip():
            return GradeResult(
                score=0.0,
                passed=False,
                error="Empty SQL content",
            )

        # Parse SQL
        try:
            statements = sqlparse.parse(content)
        except Exception as e:
            return GradeResult(
                score=0.0,
                passed=False,
                error=f"SQL parse error: {e}",
            )

        if not statements:
            return GradeResult(
                score=0.0,
                passed=False,
                error="No valid SQL statements found",
            )

        # Filter out empty statements
        kept = [s for s in statements if _stmt_type(s) != "UNKNOWN" or str(s).strip()]

        errors = []
        warnings = []
        statement_types = []

        # Check max statements
        if self.max_statements > 0 and len(kept) > self.max_statements:
            errors.append(f"Too many statements: {len(kept)} > {self.max_statements}")

        for i, stmt in enumerate(kept):
            stmt_str = str(stmt).strip()
            if not stmt_str:
                continue

            stmt_type = _stmt_type(stmt)
            statement_types.append(stmt_type)

            # Check allowed statements
            if self.allowed_statements:
                allowed_upper = [s.upper() for s in self.allowed_statements]
                if stmt_type.upper() not in allowed_upper and stmt_type != "UNKNOWN":
                    errors.append(f"Statement {i+1}: {stmt_type} not in allowed types")

            # Check forbidden keywords
            stmt_upper = stmt_str.upper()
            for keyword in self.forbidden_keywords:
                if re.search(rf"\b{keyword}\b", stmt_upper):
                    errors.append(f"Statement {i+1}: contains forbidden keyword '{keyword}'")

            # Check semicolon
            if self.require_semicolon and not stmt_str.rstrip().endswith(";"):
                warnings.append(f"Statement {i+1}: missing semicolon")

            # Basic syntax validation
            if stmt_type == "UNKNOWN" and not self._is_valid_statement(stmt_str):
                errors.append(f"Statement {i+1}: unrecognized or invalid syntax")

        # Calculate score
        if errors:
            score = 0.0
            passed = False
        elif warnings:
            score = 0.8
            passed = True
        else:
            score = 1.0
            passed = True

        return GradeResult(
            score=score,
            passed=passed,
            details={
                "statement_count": len(kept),
                "statement_types": statement_types,
                "errors": errors,
                "warnings": warnings,
            },
            error="; ".join(errors) if errors else None,
        )

    def _is_valid_statement(self, sql: str) -> bool:
        """Basic check if SQL looks valid."""
        sql = sql.strip().upper()
        # Check if it starts with a known keyword
        known_starts = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP",
            "TRUNCATE", "GRANT", "REVOKE", "BEGIN", "COMMIT", "ROLLBACK",
            "WITH", "EXPLAIN", "SHOW", "DESCRIBE", "USE", "SET",
        ]
        return any(sql.startswith(k) for k in known_starts)
