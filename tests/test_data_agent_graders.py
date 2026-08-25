"""Tests for Data Agent evaluation graders."""

import pytest

from compass.core.transcript import Outcome, Transcript
from compass.graders import get_grader
from compass.graders.base import GradeContext
from compass.graders.domains.data import (
    DataCorrectnessGrader,
    QueryQualityGrader,
    ReasoningTraceGrader,
    SelfCorrectionGrader,
    SqlEquivalenceGrader,
)

# =============================================================================
# SqlEquivalenceGrader Tests
# =============================================================================


class TestSqlEquivalenceGrader:
    """Tests for SqlEquivalenceGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("sql_equivalence")
        assert grader_cls is SqlEquivalenceGrader

    @pytest.mark.asyncio
    async def test_exact_result_match(self):
        """Test exact result match passes."""
        grader = SqlEquivalenceGrader({
            "expected_result": [
                {"name": "Alice", "count": 10},
                {"name": "Bob", "count": 5},
            ],
            "executor": "mock",
            "executor_config": {
                "mock_results": [
                    {"name": "Alice", "count": 10},
                    {"name": "Bob", "count": 5},
                ]
            },
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT name, count FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_result_order_ignored(self):
        """Test row order is ignored by default."""
        grader = SqlEquivalenceGrader({
            "expected_result": [
                {"name": "Alice", "count": 10},
                {"name": "Bob", "count": 5},
            ],
            "executor": "mock",
            "executor_config": {
                "mock_results": [
                    {"name": "Bob", "count": 5},
                    {"name": "Alice", "count": 10},
                ]
            },
            "ignore_order": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT name, count FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_extra_columns_allowed(self):
        """Test extra columns are allowed by default."""
        grader = SqlEquivalenceGrader({
            "expected_result": [{"name": "Alice"}],
            "executor": "mock",
            "executor_config": {
                "mock_results": [{"name": "Alice", "extra": "ignored"}]
            },
            "ignore_extra_columns": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT * FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_result_mismatch_fails(self):
        """Test result mismatch fails."""
        grader = SqlEquivalenceGrader({
            "expected_result": [{"name": "Alice", "count": 10}],
            "executor": "mock",
            "executor_config": {
                "mock_results": [{"name": "Alice", "count": 99}]  # Wrong count
            },
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT name, count FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_numeric_tolerance(self):
        """Test numeric tolerance for floating point comparison."""
        grader = SqlEquivalenceGrader({
            "expected_result": [{"value": 1.0}],
            "executor": "mock",
            "executor_config": {
                "mock_results": [{"value": 1.0000001}]
            },
            "tolerance": 1e-5,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT value FROM data"}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_sql_fails(self):
        """Test missing SQL fails gracefully."""
        grader = SqlEquivalenceGrader({
            "expected_result": [{"name": "Alice"}],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "No SQL found" in result.error

    @pytest.mark.asyncio
    async def test_key_columns_matching(self):
        """Test matching by key columns."""
        grader = SqlEquivalenceGrader({
            "expected_result": [
                {"id": 1, "name": "Alice", "score": 100},
                {"id": 2, "name": "Bob", "score": 90},
            ],
            "key_columns": ["id"],
            "executor": "mock",
            "executor_config": {
                "mock_results": [
                    {"id": 2, "name": "Bob", "score": 90},
                    {"id": 1, "name": "Alice", "score": 100},
                ]
            },
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT * FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is True


# =============================================================================
# DataCorrectnessGrader Tests
# =============================================================================


class TestDataCorrectnessGrader:
    """Tests for DataCorrectnessGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("data_correctness")
        assert grader_cls is DataCorrectnessGrader

    @pytest.mark.asyncio
    async def test_dict_exact_match(self):
        """Test exact dictionary match passes."""
        grader = DataCorrectnessGrader({
            "expected": {"total": 100, "active": 75},
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"total": 100, "active": 75}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_partial_match_score(self):
        """Test partial match gives partial score."""
        grader = DataCorrectnessGrader({
            "expected": {"a": 1, "b": 2, "c": 3, "d": 4},
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"a": 1, "b": 2, "c": 99, "d": 99}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert result.score == 0.5  # 2 out of 4 correct

    @pytest.mark.asyncio
    async def test_required_fields(self):
        """Test required fields must be correct."""
        grader = DataCorrectnessGrader({
            "expected": {"critical": 100, "optional": 50},
            "required_fields": ["critical"],
        })

        # Wrong critical field
        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"critical": 999, "optional": 50}),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_numeric_tolerance(self):
        """Test numeric tolerance."""
        grader = DataCorrectnessGrader({
            "expected": {"rate": 0.75},
            "tolerance": 0.01,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"rate": 0.751}),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_list_comparison(self):
        """Test list comparison."""
        grader = DataCorrectnessGrader({
            "expected": [
                {"name": "A", "value": 1},
                {"name": "B", "value": 2},
            ],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data=[
                {"name": "B", "value": 2},
                {"name": "A", "value": 1},
            ]),
        )

        result = await grader.grade(context)
        assert result.passed is True


# =============================================================================
# QueryQualityGrader Tests
# =============================================================================


class TestQueryQualityGrader:
    """Tests for QueryQualityGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("query_quality")
        assert grader_cls is QueryQualityGrader

    @pytest.mark.asyncio
    async def test_good_query_passes(self):
        """Test well-formed query passes."""
        grader = QueryQualityGrader({})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "sql": "SELECT name, COUNT(*) FROM users WHERE active = 1 GROUP BY name"
            }),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_select_star_warning(self):
        """Test SELECT * is flagged."""
        grader = QueryQualityGrader({
            "checks": ["select_star"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "SELECT * FROM users"}),
        )

        result = await grader.grade(context)
        assert result.score < 1.0
        assert any(i["type"] == "select_star" for i in result.details["issues"])

    @pytest.mark.asyncio
    async def test_delete_without_where(self):
        """Test DELETE without WHERE is flagged."""
        grader = QueryQualityGrader({
            "checks": ["missing_where"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={"sql": "DELETE FROM users"}),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert any(i["type"] == "missing_where" for i in result.details["issues"])

    @pytest.mark.asyncio
    async def test_suboptimal_not_in(self):
        """Test NOT IN with subquery is flagged."""
        grader = QueryQualityGrader({
            "checks": ["suboptimal_join"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "sql": "SELECT * FROM users WHERE id NOT IN (SELECT user_id FROM banned)"
            }),
        )

        result = await grader.grade(context)
        assert any(i["type"] == "suboptimal_join" for i in result.details["issues"])

    @pytest.mark.asyncio
    async def test_no_sql_fails(self):
        """Test missing SQL fails."""
        grader = QueryQualityGrader({})

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={}),
        )

        result = await grader.grade(context)
        assert result.passed is False


# =============================================================================
# ReasoningTraceGrader Tests
# =============================================================================


class TestReasoningTraceGrader:
    """Tests for ReasoningTraceGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("reasoning_trace")
        assert grader_cls is ReasoningTraceGrader

    @pytest.mark.asyncio
    async def test_complete_reasoning_passes(self):
        """Test complete reasoning trace passes."""
        grader = ReasoningTraceGrader({
            "required_steps": ["understand_question", "identify_tables"],
            "expected_tables": ["users"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "reasoning": "I need to find the user count. Looking at the users table...",
                "sql": "SELECT COUNT(*) FROM users",
            }),
        )

        result = await grader.grade(context)
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_missing_tables_fails(self):
        """Test missing expected tables fails."""
        grader = ReasoningTraceGrader({
            "expected_tables": ["orders", "products"],
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "sql": "SELECT * FROM users",  # Missing orders and products
            }),
        )

        result = await grader.grade(context)
        assert result.passed is False
        assert "orders" in result.details["tables_missing"]

    @pytest.mark.asyncio
    async def test_self_correction_detection(self):
        """Test self-correction pattern detection."""
        grader = ReasoningTraceGrader({
            "check_self_correction": True,
        })

        context = GradeContext(
            prompt="test",
            outcome=Outcome(output_data={
                "reasoning": (
                    "The first query failed with an error. "
                    "Let me retry with a different approach."
                ),
                "sql": "SELECT * FROM users",
            }),
        )

        result = await grader.grade(context)
        assert result.details["self_correction_detected"] is True


# =============================================================================
# SelfCorrectionGrader Tests
# =============================================================================


class TestSelfCorrectionGrader:
    """Tests for SelfCorrectionGrader."""

    def test_registered(self):
        """Test grader is registered."""
        grader_cls = get_grader("self_correction")
        assert grader_cls is SelfCorrectionGrader

    @pytest.mark.asyncio
    async def test_successful_recovery(self):
        """Test successful error recovery passes."""
        grader = SelfCorrectionGrader({
            "require_recovery": True,
        })

        transcript = Transcript(task_id="test", trial_id="trial1")
        transcript.add_tool_call(
            tool_name="sql.execute",
            input={"query": "SELECT * FROM nonexistent"},
            output=None,
            status="error",
            error={"message": "Table not found"},
        )
        transcript.add_tool_call(
            tool_name="sql.execute",
            input={"query": "SELECT * FROM users"},
            output={"rows": []},
            status="ok",
        )

        context = GradeContext(
            prompt="test",
            transcript=transcript,
            outcome=Outcome(output_data={"result": "success"}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.details["errors_detected"] == 1
        assert result.details["final_success"] is True

    @pytest.mark.asyncio
    async def test_failed_recovery_fails(self):
        """Test failed recovery fails when required."""
        grader = SelfCorrectionGrader({
            "require_recovery": True,
        })

        transcript = Transcript(task_id="test", trial_id="trial1")
        transcript.add_tool_call(
            tool_name="sql.execute",
            input={"query": "SELECT * FROM nonexistent"},
            output=None,
            status="error",
            error={"message": "Table not found"},
        )
        # No successful retry

        context = GradeContext(
            prompt="test",
            transcript=transcript,
            outcome=Outcome(),
        )

        result = await grader.grade(context)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_no_errors_passes(self):
        """Test no errors scenario passes."""
        grader = SelfCorrectionGrader({})

        transcript = Transcript(task_id="test", trial_id="trial1")
        transcript.add_tool_call(
            tool_name="sql.execute",
            input={"query": "SELECT * FROM users"},
            output={"rows": [{"id": 1}]},
            status="ok",
        )

        context = GradeContext(
            prompt="test",
            transcript=transcript,
            outcome=Outcome(output_data={"result": "success"}),
        )

        result = await grader.grade(context)
        assert result.passed is True
        assert result.details["errors_detected"] == 0

    @pytest.mark.asyncio
    async def test_excessive_retries_penalized(self):
        """Test excessive retries reduce score."""
        grader = SelfCorrectionGrader({
            "max_retries": 2,
        })

        transcript = Transcript(task_id="test", trial_id="trial1")
        # 5 retries of the same operation
        for i in range(5):
            transcript.add_tool_call(
                tool_name="sql.execute",
                input={"query": f"SELECT {i}"},
                output={"rows": []},
                status="ok",
            )

        context = GradeContext(
            prompt="test",
            transcript=transcript,
            outcome=Outcome(),
        )

        result = await grader.grade(context)
        assert result.details["retry_count"] == 4  # 5 calls = 4 retries
        assert result.score < 1.0  # Penalized for excessive retries
