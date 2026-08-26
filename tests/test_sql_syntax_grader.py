"""Tests for the sql_syntax grader (compass.graders.domains.data)."""

import pytest

from compass.core.transcript import Outcome
from compass.graders import get_grader
from compass.graders.base import GradeContext
from compass.graders.domains.data.sql_syntax import SqlSyntaxGrader

# `sqlparse` is the `compass[data]` extra, not a hard dependency — the grader
# reports its absence instead of raising, so these tests skip rather than fail.
# `uv sync` installs it (dev group); a consumer install may not have it.
pytest.importorskip("sqlparse")


class TestSqlSyntaxGrader:
    """Tests for SqlSyntaxGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("sql_syntax")
        assert grader_cls is SqlSyntaxGrader

    @pytest.mark.asyncio
    async def test_valid_select_passes(self):
        """Test valid SELECT statement passes."""
        grader = SqlSyntaxGrader({"source": "output_data.sql"})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT * FROM users WHERE id = 1"}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_multiple_statements(self):
        """Test multiple SQL statements."""
        grader = SqlSyntaxGrader({"source": "output_data.sql"})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={"sql": "SELECT * FROM users; INSERT INTO logs VALUES (1);"}
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.details["statement_count"] >= 2

    @pytest.mark.asyncio
    async def test_allowed_statements_filter(self):
        """Test allowed_statements restricts statement types."""
        grader = SqlSyntaxGrader({
            "source": "output_data.sql",
            "allowed_statements": ["SELECT"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "DELETE FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "not in allowed types" in str(result.details.get("errors", []))

    @pytest.mark.asyncio
    async def test_forbidden_keywords(self):
        """Test forbidden keywords are detected."""
        grader = SqlSyntaxGrader({
            "source": "output_data.sql",
            "forbidden_keywords": ["DROP", "TRUNCATE"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "DROP TABLE users"}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert any("DROP" in str(e) for e in result.details.get("errors", []))

    @pytest.mark.asyncio
    async def test_max_statements(self):
        """Test max_statements limit."""
        grader = SqlSyntaxGrader({
            "source": "output_data.sql",
            "max_statements": 1,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT 1; SELECT 2; SELECT 3"}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "Too many statements" in str(result.details.get("errors", []))

    @pytest.mark.asyncio
    async def test_extract_from_code_block(self):
        """Test SQL extraction from code block."""
        grader = SqlSyntaxGrader({"source": "output_data.response"})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(
                output_data={
                    "response": "Here's the query:\n```sql\nSELECT * FROM users\n```"
                }
            ),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_empty_sql_fails(self):
        """Test empty SQL fails."""
        grader = SqlSyntaxGrader({"source": "output_data.sql"})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "   "}),
        )

        result = await grader.grade(context)
        assert result.passed is False
