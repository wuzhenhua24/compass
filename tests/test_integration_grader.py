"""Tests for IntegrationGrader and output parsers."""

from __future__ import annotations

import json

import pytest

from compass.core.transcript import Outcome, ToolCall, Transcript
from compass.graders.base import GradeContext, GraderScope, GraderType
from compass.graders.code.coding.integration import (
    IntegrationGrader,
    _parse_generic,
    _parse_go_test,
    _parse_json,
    _parse_pytest,
    _parse_rspec,
    parse_test_output,
)
from compass.graders.registry import get_grader

# ===================================================================
# Parser unit tests
# ===================================================================


class TestParsePytest:
    def test_passed_and_failed(self):
        assert _parse_pytest("5 passed, 2 failed in 3.5s") == (5, 7)

    def test_all_passed(self):
        assert _parse_pytest("10 passed in 1.2s") == (10, 10)

    def test_all_failed(self):
        assert _parse_pytest("3 failed in 0.5s") == (0, 3)

    def test_no_match(self):
        assert _parse_pytest("no pytest output here") is None


class TestParseRspec:
    def test_basic(self):
        assert _parse_rspec("10 examples, 2 failures") == (8, 10)

    def test_singular(self):
        assert _parse_rspec("1 example, 0 failures") == (1, 1)

    def test_no_match(self):
        assert _parse_rspec("some other output") is None


class TestParseGoTest:
    def test_ok_and_fail(self):
        stdout = "ok  \tpkg/a\t0.5s\nFAIL\tpkg/b\t1.2s\nok  \tpkg/c\t0.3s"
        assert _parse_go_test(stdout) == (2, 3)

    def test_all_ok(self):
        assert _parse_go_test("ok  \tpkg/a\t0.5s\nok  \tpkg/b\t0.3s") == (2, 2)

    def test_no_match(self):
        assert _parse_go_test("nothing here") is None


class TestParseJson:
    def test_passed_total(self):
        data = json.dumps({"passed": 3, "total": 5})
        assert _parse_json(data) == (3, 5)

    def test_passed_tests_total_tests(self):
        data = json.dumps({"passed_tests": 7, "total_tests": 10})
        assert _parse_json(data) == (7, 10)

    def test_tests_array(self):
        data = json.dumps({
            "tests": [
                {"test_name": "a", "status": "passed"},
                {"test_name": "b", "status": "failed"},
                {"test_name": "c", "status": "passed"},
            ]
        })
        assert _parse_json(data) == (2, 3)

    def test_invalid_json(self):
        assert _parse_json("not json") is None

    def test_empty_object(self):
        assert _parse_json("{}") is None


class TestParseGeneric:
    def test_slash_format(self):
        assert _parse_generic("Tests: 8/10 passed") == (8, 10)

    def test_of_format(self):
        assert _parse_generic("3 of 5 tests passed") == (3, 5)

    def test_no_match(self):
        assert _parse_generic("no recognizable format") is None


class TestAutoDetection:
    def test_json_detected(self):
        data = json.dumps({"passed": 4, "total": 5})
        assert parse_test_output(data, "auto") == (4, 5)

    def test_pytest_detected(self):
        assert parse_test_output("3 passed, 1 failed", "auto") == (3, 4)

    def test_rspec_detected(self):
        assert parse_test_output("8 examples, 1 failure", "auto") == (7, 8)

    def test_explicit_format(self):
        assert parse_test_output("5 passed", "pytest") == (5, 5)

    def test_unknown_format(self):
        assert parse_test_output("anything", "unknown_format") is None


# ===================================================================
# IntegrationGrader tests
# ===================================================================


class TestIntegrationGraderRegistration:
    def test_registered(self):
        assert get_grader("integration_test") is IntegrationGrader

    def test_grader_type(self):
        grader = IntegrationGrader()
        assert grader.grader_type == GraderType.CODE
        assert grader.grader_scope == GraderScope.OUTCOME


class TestIntegrationGraderValidation:
    def test_validate_config_no_script(self):
        grader = IntegrationGrader({})
        assert grader.validate_config() is False

    def test_validate_config_with_script(self):
        grader = IntegrationGrader({"script": "./grade.sh"})
        assert grader.validate_config() is True


class TestIntegrationGraderGrade:
    @pytest.mark.asyncio
    async def test_missing_script_returns_error(self):
        grader = IntegrationGrader({})
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.error is not None
        assert "script" in result.error.lower()

    @pytest.mark.asyncio
    async def test_all_tests_pass(self):
        grader = IntegrationGrader({
            "script": "echo '5 passed in 1.0s'",
            "output_format": "pytest",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is True
        assert result.score == 1.0
        assert result.details["tests_passed"] == 5
        assert result.details["tests_total"] == 5

    @pytest.mark.asyncio
    async def test_partial_pass(self):
        grader = IntegrationGrader({
            "script": "echo '3 passed, 2 failed'",
            "output_format": "pytest",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False  # default threshold 1.0
        assert result.score == pytest.approx(3 / 5)
        assert result.details["tests_passed"] == 3
        assert result.details["tests_total"] == 5

    @pytest.mark.asyncio
    async def test_partial_pass_with_threshold(self):
        grader = IntegrationGrader({
            "script": "echo '3 passed, 2 failed'",
            "output_format": "pytest",
            "pass_threshold": 0.5,
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is True  # 0.6 >= 0.5
        assert result.score == pytest.approx(0.6)

    @pytest.mark.asyncio
    async def test_json_output(self):
        payload = json.dumps({"passed": 7, "total": 10})
        grader = IntegrationGrader({
            "script": f"echo '{payload}'",
            "output_format": "json",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.score == pytest.approx(0.7)
        assert result.details["tests_passed"] == 7
        assert result.details["tests_total"] == 10

    @pytest.mark.asyncio
    async def test_json_tests_array(self):
        payload = json.dumps({
            "tests": [
                {"test_name": "a", "status": "passed"},
                {"test_name": "b", "status": "passed"},
                {"test_name": "c", "status": "failed"},
            ]
        })
        grader = IntegrationGrader({
            "script": f"echo '{payload}'",
            "output_format": "json",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.score == pytest.approx(2 / 3)

    @pytest.mark.asyncio
    async def test_rspec_output(self):
        grader = IntegrationGrader({
            "script": "echo '10 examples, 3 failures'",
            "output_format": "rspec",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.score == pytest.approx(0.7)
        assert result.details["tests_passed"] == 7

    @pytest.mark.asyncio
    async def test_exit_code_fallback(self):
        """When output can't be parsed, fall back to exit code."""
        grader = IntegrationGrader({
            "script": "echo 'no parseable output'",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is True  # exit code 0
        assert result.score == 1.0
        assert result.details["parse_fallback"] is True

    @pytest.mark.asyncio
    async def test_exit_code_fallback_failure(self):
        grader = IntegrationGrader({
            "script": "exit 1",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score == 0.0
        assert result.details["parse_fallback"] is True

    @pytest.mark.asyncio
    async def test_script_timeout(self):
        grader = IntegrationGrader({
            "script": "sleep 60",
            "timeout": 0.5,
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_failure_tags_all_failed(self):
        grader = IntegrationGrader({
            "script": "echo '0 passed, 5 failed'",
            "output_format": "pytest",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert "all_tests_failed" in result.failure_tags
        assert "test_failure" in result.failure_tags

    @pytest.mark.asyncio
    async def test_failure_tags_partial(self):
        grader = IntegrationGrader({
            "script": "echo '2 passed, 3 failed'",
            "output_format": "pytest",
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert "partial_failure" in result.failure_tags

    @pytest.mark.asyncio
    async def test_reasoning_message(self):
        grader = IntegrationGrader({
            "script": "echo '8 passed, 2 failed'",
            "output_format": "pytest",
            "pass_threshold": 0.7,
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert "8/10" in result.reasoning
        assert "PASSED" in result.reasoning

    @pytest.mark.asyncio
    async def test_setup_command_failure(self):
        grader = IntegrationGrader({
            "script": "echo ok",
            "setup_commands": ["exit 1"],
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.details.get("phase") == "setup"
        assert "setup_failure" in result.failure_tags


class TestLeakDetection:
    @pytest.mark.asyncio
    async def test_leak_detected(self):
        grader = IntegrationGrader({
            "script": "echo 'ok'",
            "leak_patterns": ["SECRET_UUID_123"],
        })
        transcript = Transcript(
            task_id="test", trial_id="t1",
            tool_calls=[
                ToolCall(
                    tool_name="read_file",
                    input={"path": "/grader/grade.sh"},
                    output="SECRET_UUID_123 found here",
                )
            ],
        )
        ctx = GradeContext(
            outcome=Outcome(),
            transcript=transcript,
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert result.score == 0.0
        assert "answer_leak" in result.failure_tags
        assert "SECRET_UUID_123" in result.details["leaked_markers"]

    @pytest.mark.asyncio
    async def test_no_leak(self):
        grader = IntegrationGrader({
            "script": "echo '3 passed'",
            "output_format": "pytest",
            "leak_patterns": ["SECRET_UUID_123"],
        })
        transcript = Transcript(
            task_id="test", trial_id="t1",
            tool_calls=[
                ToolCall(
                    tool_name="edit_file",
                    input={"path": "/env/server.js"},
                    output="file saved",
                )
            ],
        )
        ctx = GradeContext(
            outcome=Outcome(),
            transcript=transcript,
        )
        result = await grader.grade(ctx)
        assert result.passed is True
        assert "answer_leak" not in result.failure_tags

    @pytest.mark.asyncio
    async def test_leak_in_reasoning_steps(self):
        grader = IntegrationGrader({
            "script": "echo 'ok'",
            "leak_patterns": ["LEAK_MARKER_ABC"],
        })
        transcript = Transcript(
            task_id="test", trial_id="t1",
            reasoning_steps=["Let me read LEAK_MARKER_ABC from grader"],
        )
        ctx = GradeContext(
            outcome=Outcome(),
            transcript=transcript,
        )
        result = await grader.grade(ctx)
        assert result.passed is False
        assert "answer_leak" in result.failure_tags

    @pytest.mark.asyncio
    async def test_no_transcript_skips_leak_check(self):
        """Leak detection is skipped when transcript is absent."""
        grader = IntegrationGrader({
            "script": "echo '5 passed'",
            "output_format": "pytest",
            "leak_patterns": ["SECRET"],
        })
        ctx = GradeContext(outcome=Outcome())
        result = await grader.grade(ctx)
        assert result.passed is True
