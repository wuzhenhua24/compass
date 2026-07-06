"""Tests for ToolCall Protocol v1 enhancements."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from compass.core.transcript import (
    CostInfo,
    TokenUsage,
    ToolCall,
    Transcript,
    TOOLCALL_PROTOCOL_VERSION,
    ToolNameInfo,
    parse_tool_name,
    format_tool_name,
    validate_tool_name,
)


# ===================================================================
# CostInfo tests
# ===================================================================


class TestCostInfo:
    """Tests for CostInfo dataclass."""

    def test_cost_info_defaults(self):
        """CostInfo has correct default values."""
        cost = CostInfo()
        assert cost.total_usd == 0.0
        assert cost.input_cost_usd is None
        assert cost.output_cost_usd is None
        assert cost.currency == "USD"
        assert cost.metadata == {}

    def test_cost_info_to_dict(self):
        """CostInfo serializes correctly."""
        cost = CostInfo(
            total_usd=0.05,
            input_cost_usd=0.02,
            output_cost_usd=0.03,
            currency="USD",
            metadata={"model": "gpt-4"},
        )
        d = cost.to_dict()
        assert d["total_usd"] == 0.05
        assert d["input_cost_usd"] == 0.02
        assert d["output_cost_usd"] == 0.03
        assert d["currency"] == "USD"
        assert d["metadata"] == {"model": "gpt-4"}

    def test_cost_info_from_dict(self):
        """CostInfo deserializes correctly."""
        data = {
            "total_usd": 0.10,
            "input_cost_usd": 0.04,
            "output_cost_usd": 0.06,
            "currency": "EUR",
            "metadata": {"provider": "anthropic"},
        }
        cost = CostInfo.from_dict(data)
        assert cost.total_usd == 0.10
        assert cost.input_cost_usd == 0.04
        assert cost.output_cost_usd == 0.06
        assert cost.currency == "EUR"
        assert cost.metadata == {"provider": "anthropic"}

    def test_cost_info_from_dict_missing_fields(self):
        """CostInfo handles missing fields with defaults."""
        cost = CostInfo.from_dict({})
        assert cost.total_usd == 0.0
        assert cost.input_cost_usd is None
        assert cost.output_cost_usd is None
        assert cost.currency == "USD"
        assert cost.metadata == {}

    def test_cost_info_roundtrip(self):
        """CostInfo survives to_dict -> from_dict roundtrip."""
        original = CostInfo(
            total_usd=0.15,
            input_cost_usd=0.05,
            output_cost_usd=0.10,
            metadata={"key": "value"},
        )
        restored = CostInfo.from_dict(original.to_dict())
        assert restored.total_usd == original.total_usd
        assert restored.input_cost_usd == original.input_cost_usd
        assert restored.output_cost_usd == original.output_cost_usd
        assert restored.currency == original.currency
        assert restored.metadata == original.metadata


# ===================================================================
# TokenUsage tests
# ===================================================================


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_token_usage_defaults(self):
        """TokenUsage has correct default values."""
        tokens = TokenUsage()
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.total_tokens == 0
        assert tokens.metadata == {}

    def test_token_usage_auto_total(self):
        """TokenUsage auto-calculates total_tokens when not provided."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50)
        assert tokens.total_tokens == 150

    def test_token_usage_explicit_total(self):
        """TokenUsage respects explicit total_tokens."""
        tokens = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=200)
        assert tokens.total_tokens == 200

    def test_token_usage_to_dict(self):
        """TokenUsage serializes correctly."""
        tokens = TokenUsage(
            input_tokens=500,
            output_tokens=200,
            total_tokens=700,
            metadata={"cached": True},
        )
        d = tokens.to_dict()
        assert d["input_tokens"] == 500
        assert d["output_tokens"] == 200
        assert d["total_tokens"] == 700
        assert d["metadata"] == {"cached": True}

    def test_token_usage_from_dict(self):
        """TokenUsage deserializes correctly."""
        data = {
            "input_tokens": 1000,
            "output_tokens": 500,
            "total_tokens": 1500,
            "metadata": {"model": "claude-3"},
        }
        tokens = TokenUsage.from_dict(data)
        assert tokens.input_tokens == 1000
        assert tokens.output_tokens == 500
        assert tokens.total_tokens == 1500
        assert tokens.metadata == {"model": "claude-3"}

    def test_token_usage_from_dict_missing_fields(self):
        """TokenUsage handles missing fields with defaults."""
        tokens = TokenUsage.from_dict({})
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.total_tokens == 0
        assert tokens.metadata == {}

    def test_token_usage_roundtrip(self):
        """TokenUsage survives to_dict -> from_dict roundtrip."""
        original = TokenUsage(
            input_tokens=250,
            output_tokens=125,
            metadata={"source": "test"},
        )
        restored = TokenUsage.from_dict(original.to_dict())
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.total_tokens == original.total_tokens
        assert restored.metadata == original.metadata


# ===================================================================
# ToolCall tests
# ===================================================================


class TestToolCall:
    """Tests for ToolCall dataclass enhancements."""

    def test_call_id_auto_generated(self):
        """ToolCall generates call_id automatically."""
        tc = ToolCall(tool_name="test_tool")
        assert tc.call_id is not None
        assert len(tc.call_id) == 12

    def test_call_id_unique(self):
        """ToolCall generates unique call_ids."""
        tc1 = ToolCall(tool_name="tool1")
        tc2 = ToolCall(tool_name="tool2")
        assert tc1.call_id != tc2.call_id

    def test_call_id_in_to_dict(self):
        """call_id is included in to_dict output."""
        tc = ToolCall(tool_name="test_tool")
        d = tc.to_dict()
        assert "call_id" in d
        assert d["call_id"] == tc.call_id

    def test_from_dict_protocol_fields(self):
        """from_dict correctly handles protocol fields."""
        data = {
            "call_id": "abc123def456",
            "tool_name": "run_code",
            "input": {"code": "print('hello')"},
            "output": "hello",
            "status": "ok",
            "duration_ms": 150.5,
            "timestamp": 1234567890.0,
        }
        tc = ToolCall.from_dict(data)
        assert tc.call_id == "abc123def456"
        assert tc.tool_name == "run_code"
        assert tc.input == {"code": "print('hello')"}
        assert tc.output == "hello"
        assert tc.status == "ok"
        assert tc.duration_ms == 150.5
        assert tc.timestamp == 1234567890.0

    def test_from_dict_legacy_fields(self):
        """from_dict correctly handles legacy fields (tool/args/result)."""
        data = {
            "tool": "old_tool",
            "args": {"arg1": "value1"},
            "result": "old result",
        }
        tc = ToolCall.from_dict(data)
        assert tc.tool_name == "old_tool"
        assert tc.input == {"arg1": "value1"}
        assert tc.output == "old result"
        # Backward compat properties
        assert tc.tool == "old_tool"
        assert tc.args == {"arg1": "value1"}
        assert tc.result == "old result"

    def test_from_dict_missing_call_id(self):
        """from_dict auto-generates call_id when missing."""
        data = {"tool_name": "some_tool"}
        tc = ToolCall.from_dict(data)
        assert tc.call_id is not None
        assert len(tc.call_id) == 12

    def test_from_dict_with_cost(self):
        """from_dict correctly reconstructs CostInfo."""
        data = {
            "tool_name": "expensive_tool",
            "cost": {
                "total_usd": 0.25,
                "input_cost_usd": 0.10,
                "output_cost_usd": 0.15,
            },
        }
        tc = ToolCall.from_dict(data)
        assert tc.cost is not None
        assert isinstance(tc.cost, CostInfo)
        assert tc.cost.total_usd == 0.25
        assert tc.cost.input_cost_usd == 0.10
        assert tc.cost.output_cost_usd == 0.15

    def test_from_dict_with_tokens(self):
        """from_dict correctly reconstructs TokenUsage."""
        data = {
            "tool_name": "token_heavy_tool",
            "tokens": {
                "input_tokens": 2000,
                "output_tokens": 1000,
                "total_tokens": 3000,
            },
        }
        tc = ToolCall.from_dict(data)
        assert tc.tokens is not None
        assert isinstance(tc.tokens, TokenUsage)
        assert tc.tokens.input_tokens == 2000
        assert tc.tokens.output_tokens == 1000
        assert tc.tokens.total_tokens == 3000

    def test_from_dict_status_inference(self):
        """from_dict infers status from error when status is missing."""
        # No error -> ok
        tc_ok = ToolCall.from_dict({"tool_name": "test"})
        assert tc_ok.status == "ok"

        # With error -> error
        tc_err = ToolCall.from_dict({"tool_name": "test", "error": {"message": "fail"}})
        assert tc_err.status == "error"

    def test_cost_dict_auto_convert(self):
        """Passing dict for cost auto-converts to CostInfo."""
        tc = ToolCall(
            tool_name="test",
            cost={"total_usd": 0.05, "input_cost_usd": 0.02},  # type: ignore
        )
        assert isinstance(tc.cost, CostInfo)
        assert tc.cost.total_usd == 0.05
        assert tc.cost.input_cost_usd == 0.02

    def test_tokens_dict_auto_convert(self):
        """Passing dict for tokens auto-converts to TokenUsage."""
        tc = ToolCall(
            tool_name="test",
            tokens={"input_tokens": 100, "output_tokens": 50},  # type: ignore
        )
        assert isinstance(tc.tokens, TokenUsage)
        assert tc.tokens.input_tokens == 100
        assert tc.tokens.output_tokens == 50
        assert tc.tokens.total_tokens == 150

    def test_backward_compat_properties(self):
        """tc.tool, tc.args, tc.result work as expected."""
        tc = ToolCall(
            tool_name="my_tool",
            input={"key": "val"},
            output="result_data",
        )
        assert tc.tool == "my_tool"
        assert tc.args == {"key": "val"}
        assert tc.result == "result_data"

        # Setters
        tc.tool = "new_tool"
        tc.args = {"new_key": "new_val"}
        tc.result = "new_result"

        assert tc.tool_name == "new_tool"
        assert tc.input == {"new_key": "new_val"}
        assert tc.output == "new_result"

    def test_to_dict_roundtrip(self):
        """ToolCall survives to_dict -> from_dict roundtrip."""
        original = ToolCall(
            tool_name="complete_tool",
            input={"param": "value"},
            output={"result": "data"},
            status="ok",
            duration_ms=250.0,
            error=None,
            cost=CostInfo(total_usd=0.03, input_cost_usd=0.01, output_cost_usd=0.02),
            tokens=TokenUsage(input_tokens=500, output_tokens=250),
            tool_type="llm",
            retry_count=1,
            metadata={"custom": "field"},
            redacted=False,
        )
        d = original.to_dict()
        restored = ToolCall.from_dict(d)

        assert restored.call_id == original.call_id
        assert restored.tool_name == original.tool_name
        assert restored.input == original.input
        assert restored.output == original.output
        assert restored.status == original.status
        assert restored.duration_ms == original.duration_ms
        assert restored.cost.total_usd == original.cost.total_usd
        assert restored.cost.input_cost_usd == original.cost.input_cost_usd
        assert restored.tokens.input_tokens == original.tokens.input_tokens
        assert restored.tokens.total_tokens == original.tokens.total_tokens
        assert restored.tool_type == original.tool_type
        assert restored.retry_count == original.retry_count
        assert restored.metadata == original.metadata
        assert restored.redacted == original.redacted

    def test_to_dict_cost_tokens_serialized(self):
        """to_dict serializes CostInfo and TokenUsage to dicts."""
        tc = ToolCall(
            tool_name="test",
            cost=CostInfo(total_usd=0.01),
            tokens=TokenUsage(input_tokens=10),
        )
        d = tc.to_dict()
        assert isinstance(d["cost"], dict)
        assert d["cost"]["total_usd"] == 0.01
        assert isinstance(d["tokens"], dict)
        assert d["tokens"]["input_tokens"] == 10

    def test_to_dict_cost_tokens_none(self):
        """to_dict handles None cost/tokens."""
        tc = ToolCall(tool_name="test")
        d = tc.to_dict()
        assert d["cost"] is None
        assert d["tokens"] is None

    def test_turn_index_agent_name_defaults(self):
        """New context fields default to None."""
        tc = ToolCall(tool_name="test")
        assert tc.turn_index is None
        assert tc.agent_name is None

    def test_turn_index_agent_name_roundtrip(self):
        """turn_index / agent_name survive to_dict -> from_dict as first-class fields."""
        original = ToolCall(tool_name="t", turn_index=3, agent_name="Planner")
        d = original.to_dict()
        assert d["turn_index"] == 3
        assert d["agent_name"] == "Planner"
        restored = ToolCall.from_dict(d)
        assert restored.turn_index == 3
        assert restored.agent_name == "Planner"

    def test_from_dict_legacy_metadata_location(self):
        """Old transcripts stored these in metadata -> still populate the fields."""
        legacy = {
            "tool_name": "t",
            "metadata": {"turn_index": 2, "agent_name": "Weather"},
        }
        tc = ToolCall.from_dict(legacy)
        assert tc.turn_index == 2
        assert tc.agent_name == "Weather"

    def test_from_dict_top_level_wins_over_metadata(self):
        """First-class value takes precedence over a stale metadata copy."""
        data = {
            "tool_name": "t",
            "turn_index": 5,
            "agent_name": "New",
            "metadata": {"turn_index": 2, "agent_name": "Old"},
        }
        tc = ToolCall.from_dict(data)
        assert tc.turn_index == 5
        assert tc.agent_name == "New"


# ===================================================================
# Transcript integration tests
# ===================================================================


class TestTranscriptIntegration:
    """Tests for Transcript load/save with enhanced ToolCall."""

    def test_transcript_load_uses_from_dict(self):
        """Transcript.load() correctly reconstructs ToolCall with from_dict."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_transcript.json"
            data = {
                "task_id": "task-001",
                "trial_id": "trial-001",
                "input": {"prompt": "test prompt", "params": {}},
                "tool_calls": [
                    {
                        "call_id": "call123",
                        "tool_name": "test_tool",
                        "input": {"arg": "value"},
                        "output": "result",
                        "status": "ok",
                        "cost": {"total_usd": 0.05},
                        "tokens": {"input_tokens": 100, "output_tokens": 50},
                    }
                ],
                "outcome": {},
                "grading": {},
            }
            with open(path, "w") as f:
                json.dump(data, f)

            transcript = Transcript.load(path)

            assert len(transcript.tool_calls) == 1
            tc = transcript.tool_calls[0]
            assert tc.call_id == "call123"
            assert tc.tool_name == "test_tool"
            assert isinstance(tc.cost, CostInfo)
            assert tc.cost.total_usd == 0.05
            assert isinstance(tc.tokens, TokenUsage)
            assert tc.tokens.input_tokens == 100
            assert tc.tokens.output_tokens == 50

    def test_transcript_save_load_with_cost_tokens(self):
        """Transcript save/load preserves cost and tokens."""
        transcript = Transcript(task_id="task-002", trial_id="trial-002")
        transcript.add_tool_call(
            tool_name="expensive_tool",
            input={"param": "value"},
            output="result",
            cost={"total_usd": 0.10, "input_cost_usd": 0.04},
            tokens={"input_tokens": 500, "output_tokens": 200},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transcript.json"
            transcript.save(path)
            loaded = Transcript.load(path)

            assert len(loaded.tool_calls) == 1
            tc = loaded.tool_calls[0]
            assert isinstance(tc.cost, CostInfo)
            assert tc.cost.total_usd == 0.10
            assert tc.cost.input_cost_usd == 0.04
            assert isinstance(tc.tokens, TokenUsage)
            assert tc.tokens.input_tokens == 500
            assert tc.tokens.output_tokens == 200
            assert tc.tokens.total_tokens == 700

    def test_transcript_load_legacy_format(self):
        """Transcript.load() handles legacy tool/args/result format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            data = {
                "task_id": "task-legacy",
                "trial_id": "trial-legacy",
                "input": {"prompt": "legacy prompt", "params": {}},
                "tool_calls": [
                    {
                        "tool": "legacy_tool",
                        "args": {"old_arg": "old_value"},
                        "result": "old_result",
                    }
                ],
                "outcome": {},
                "grading": {},
            }
            with open(path, "w") as f:
                json.dump(data, f)

            transcript = Transcript.load(path)
            tc = transcript.tool_calls[0]

            assert tc.tool_name == "legacy_tool"
            assert tc.input == {"old_arg": "old_value"}
            assert tc.output == "old_result"
            assert tc.call_id is not None  # Auto-generated


# ===================================================================
# Error normalization tests
# ===================================================================


class TestErrorNormalization:
    """Tests for error field normalization."""

    def test_error_string_normalized_to_dict(self):
        """String error is normalized to dict with message key."""
        tc = ToolCall(tool_name="test", error="Something went wrong")
        assert tc.error == {"message": "Something went wrong"}

    def test_error_dict_preserved(self):
        """Dict error is preserved as-is."""
        tc = ToolCall(
            tool_name="test",
            error={"message": "Error", "code": 500, "details": {"foo": "bar"}},
        )
        assert tc.error == {"message": "Error", "code": 500, "details": {"foo": "bar"}}

    def test_error_none_preserved(self):
        """None error stays None."""
        tc = ToolCall(tool_name="test", error=None)
        assert tc.error is None

    def test_error_normalized_in_to_dict(self):
        """to_dict returns normalized error."""
        tc = ToolCall(tool_name="test", error="fail")
        d = tc.to_dict()
        assert d["error"] == {"message": "fail"}

    def test_error_from_dict_preserved(self):
        """from_dict preserves error format."""
        data = {"tool_name": "test", "error": {"message": "error", "code": 123}}
        tc = ToolCall.from_dict(data)
        assert tc.error == {"message": "error", "code": 123}

    def test_error_string_in_from_dict_normalized(self):
        """from_dict normalizes string error."""
        data = {"tool_name": "test", "error": "string error"}
        tc = ToolCall.from_dict(data)
        assert tc.error == {"message": "string error"}


# ===================================================================
# Protocol version tests
# ===================================================================


class TestProtocolVersion:
    """Tests for protocol_version field."""

    def test_default_protocol_version(self):
        """Transcript has default protocol version."""
        transcript = Transcript(task_id="task", trial_id="trial")
        assert transcript.protocol_version == TOOLCALL_PROTOCOL_VERSION

    def test_protocol_version_in_to_dict(self):
        """protocol_version is included in to_dict."""
        transcript = Transcript(task_id="task", trial_id="trial")
        d = transcript.to_dict()
        assert "protocol_version" in d
        assert d["protocol_version"] == TOOLCALL_PROTOCOL_VERSION

    def test_protocol_version_save_load(self):
        """protocol_version survives save/load."""
        transcript = Transcript(task_id="task", trial_id="trial")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "transcript.json"
            transcript.save(path)
            loaded = Transcript.load(path)

            assert loaded.protocol_version == TOOLCALL_PROTOCOL_VERSION

    def test_load_legacy_without_version(self):
        """Loading file without protocol_version defaults to 1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "legacy.json"
            data = {
                "task_id": "task",
                "trial_id": "trial",
                "input": {"prompt": "", "params": {}},
                "outcome": {},
                "grading": {},
            }
            with open(path, "w") as f:
                json.dump(data, f)

            transcript = Transcript.load(path)
            assert transcript.protocol_version == "1.0"


# ===================================================================
# Aggregation methods tests
# ===================================================================


class TestAggregationMethods:
    """Tests for sum_cost, sum_tokens, sum_duration methods."""

    def test_sum_cost_empty(self):
        """sum_cost returns zero CostInfo for empty tool_calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        cost = transcript.sum_cost()
        assert cost.total_usd == 0.0
        assert cost.input_cost_usd is None
        assert cost.output_cost_usd is None

    def test_sum_cost_single(self):
        """sum_cost with single tool call."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            cost={"total_usd": 0.10, "input_cost_usd": 0.04, "output_cost_usd": 0.06},
        )
        cost = transcript.sum_cost()
        assert cost.total_usd == 0.10
        assert cost.input_cost_usd == 0.04
        assert cost.output_cost_usd == 0.06

    def test_sum_cost_multiple(self):
        """sum_cost aggregates across multiple tool calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            cost={"total_usd": 0.10, "input_cost_usd": 0.04, "output_cost_usd": 0.06},
        )
        transcript.add_tool_call(
            tool_name="tool2",
            cost={"total_usd": 0.20, "input_cost_usd": 0.08, "output_cost_usd": 0.12},
        )
        cost = transcript.sum_cost()
        assert cost.total_usd == pytest.approx(0.30)
        assert cost.input_cost_usd == pytest.approx(0.12)
        assert cost.output_cost_usd == pytest.approx(0.18)

    def test_sum_cost_skips_none(self):
        """sum_cost skips tool calls without cost."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(tool_name="tool1", cost={"total_usd": 0.10})
        transcript.add_tool_call(tool_name="tool2")  # No cost
        transcript.add_tool_call(tool_name="tool3", cost={"total_usd": 0.20})
        cost = transcript.sum_cost()
        assert cost.total_usd == pytest.approx(0.30)

    def test_sum_cost_partial_breakdown(self):
        """sum_cost handles partial input/output breakdowns."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            cost={"total_usd": 0.10, "input_cost_usd": 0.04},  # No output
        )
        transcript.add_tool_call(
            tool_name="tool2",
            cost={"total_usd": 0.20},  # No breakdown
        )
        cost = transcript.sum_cost()
        assert cost.total_usd == pytest.approx(0.30)
        assert cost.input_cost_usd == pytest.approx(0.04)
        assert cost.output_cost_usd is None

    def test_sum_tokens_empty(self):
        """sum_tokens returns zero TokenUsage for empty tool_calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        tokens = transcript.sum_tokens()
        assert tokens.input_tokens == 0
        assert tokens.output_tokens == 0
        assert tokens.total_tokens == 0

    def test_sum_tokens_single(self):
        """sum_tokens with single tool call."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            tokens={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        tokens = transcript.sum_tokens()
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50
        assert tokens.total_tokens == 150

    def test_sum_tokens_multiple(self):
        """sum_tokens aggregates across multiple tool calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            tokens={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
        )
        transcript.add_tool_call(
            tool_name="tool2",
            tokens={"input_tokens": 200, "output_tokens": 100, "total_tokens": 300},
        )
        tokens = transcript.sum_tokens()
        assert tokens.input_tokens == 300
        assert tokens.output_tokens == 150
        assert tokens.total_tokens == 450

    def test_sum_tokens_skips_none(self):
        """sum_tokens skips tool calls without tokens."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(
            tool_name="tool1",
            tokens={"input_tokens": 100, "output_tokens": 50},
        )
        transcript.add_tool_call(tool_name="tool2")  # No tokens
        tokens = transcript.sum_tokens()
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50

    def test_sum_duration_empty(self):
        """sum_duration returns 0 for empty tool_calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        assert transcript.sum_duration() == 0.0

    def test_sum_duration_single(self):
        """sum_duration with single tool call."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(tool_name="tool1", duration_ms=150.5)
        assert transcript.sum_duration() == 150.5

    def test_sum_duration_multiple(self):
        """sum_duration aggregates across multiple tool calls."""
        transcript = Transcript(task_id="task", trial_id="trial")
        transcript.add_tool_call(tool_name="tool1", duration_ms=100.0)
        transcript.add_tool_call(tool_name="tool2", duration_ms=200.5)
        transcript.add_tool_call(tool_name="tool3", duration_ms=50.0)
        assert transcript.sum_duration() == 350.5


# ===================================================================
# tool_name naming convention tests
# ===================================================================


class TestToolNameParsing:
    """Tests for tool_name parsing and formatting."""

    def test_parse_two_parts(self):
        """Parse simple adapter.action format."""
        info = parse_tool_name("sandbox.exec")
        assert info.adapter == "sandbox"
        assert info.action == "exec"
        assert info.resource is None
        assert info.full_name == "sandbox.exec"

    def test_parse_three_parts(self):
        """Parse adapter.resource.action format."""
        info = parse_tool_name("openai.chat.completion")
        assert info.adapter == "openai"
        assert info.resource == "chat"
        assert info.action == "completion"
        assert info.full_name == "openai.chat.completion"

    def test_parse_single_part(self):
        """Parse single part defaults to custom adapter."""
        info = parse_tool_name("execute")
        assert info.adapter == "custom"
        assert info.action == "execute"
        assert info.resource is None

    def test_parse_four_parts(self):
        """Parse four+ parts joins extra as action."""
        info = parse_tool_name("mcp.server.tool.invoke")
        assert info.adapter == "mcp"
        assert info.resource == "server"
        assert info.action == "tool.invoke"

    def test_format_two_parts(self):
        """Format simple adapter.action."""
        name = format_tool_name("sandbox", "exec")
        assert name == "sandbox.exec"

    def test_format_three_parts(self):
        """Format adapter.resource.action."""
        name = format_tool_name("openai", "completion", "chat")
        assert name == "openai.chat.completion"

    def test_matches_adapter(self):
        """ToolNameInfo.matches_adapter works correctly."""
        info = parse_tool_name("openai.chat.completion")
        assert info.matches_adapter("openai")
        assert info.matches_adapter("OpenAI")  # Case insensitive
        assert not info.matches_adapter("anthropic")

    def test_matches_pattern_exact(self):
        """Pattern matching with exact match."""
        info = parse_tool_name("sandbox.exec")
        assert info.matches_pattern("sandbox.exec")
        assert not info.matches_pattern("sandbox.run")

    def test_matches_pattern_wildcard_suffix(self):
        """Pattern matching with *.action."""
        info = parse_tool_name("sandbox.exec")
        assert info.matches_pattern("*.exec")
        assert not info.matches_pattern("*.run")

    def test_matches_pattern_wildcard_prefix(self):
        """Pattern matching with adapter.*."""
        info = parse_tool_name("openai.chat.completion")
        assert info.matches_pattern("openai.*")
        assert not info.matches_pattern("anthropic.*")

    def test_matches_pattern_star(self):
        """Pattern matching with * matches everything."""
        info = parse_tool_name("anything.here")
        assert info.matches_pattern("*")


class TestToolNameValidation:
    """Tests for tool_name validation."""

    def test_validate_valid_name(self):
        """Valid tool names pass validation."""
        assert validate_tool_name("openai.chat.completion") == []
        assert validate_tool_name("sandbox.exec") == []
        assert validate_tool_name("custom.my_tool") == []

    def test_validate_empty(self):
        """Empty tool name fails validation."""
        warnings = validate_tool_name("")
        assert len(warnings) == 1
        assert "empty" in warnings[0].lower()

    def test_validate_single_part(self):
        """Single part tool name warns about format."""
        warnings = validate_tool_name("execute")
        assert len(warnings) == 1
        assert "format" in warnings[0].lower()

    def test_validate_empty_segment(self):
        """Tool name with empty segment warns."""
        warnings = validate_tool_name("openai..completion")
        assert any("empty segments" in w for w in warnings)

    def test_validate_spaces(self):
        """Tool name with spaces warns."""
        warnings = validate_tool_name("openai.chat completion")
        assert any("spaces" in w for w in warnings)

    def test_validate_strict_unknown_adapter(self):
        """Strict mode warns about unknown adapters."""
        warnings = validate_tool_name("unknown_provider.action", strict=True)
        assert any("Unknown adapter" in w for w in warnings)

    def test_validate_strict_known_adapter(self):
        """Strict mode passes known adapters."""
        warnings = validate_tool_name("openai.chat.completion", strict=True)
        assert len(warnings) == 0


# ===================================================================
# JSONL Export/Import tests
# ===================================================================


class TestTranscriptJsonl:
    """Tests for JSONL event stream export/import."""

    def test_to_jsonl_basic_structure(self):
        """to_jsonl produces valid JSONL with expected event types."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.input_prompt = "Test prompt"
        transcript.input_params = {"key": "value"}

        jsonl = transcript.to_jsonl()
        lines = jsonl.strip().split("\n")
        events = [json.loads(line) for line in lines]

        # Should have at least started and completed events
        event_types = [e["type"] for e in events]
        assert "transcript.started" in event_types
        assert "transcript.completed" in event_types

    def test_to_jsonl_with_tool_calls(self):
        """to_jsonl includes tool_call.started and tool_call.completed events."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="sandbox.exec",
            tool_type="code",
            input={"command": "echo hello"},
            output={"stdout": "hello"},
            status="ok",
            duration_ms=150.0,
            cost={"total_usd": 0.01},
            tokens={"input_tokens": 100, "output_tokens": 50},
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        # Find tool call events
        started_events = [e for e in events if e["type"] == "tool_call.started"]
        completed_events = [e for e in events if e["type"] == "tool_call.completed"]

        assert len(started_events) == 1
        assert len(completed_events) == 1

        # Check started event
        started = started_events[0]
        assert started["tool_name"] == "sandbox.exec"
        assert started["tool_type"] == "code"
        assert started["input"] == {"command": "echo hello"}
        assert "call_id" in started

        # Check completed event
        completed = completed_events[0]
        assert completed["tool_name"] == "sandbox.exec"
        assert completed["status"] == "ok"
        assert completed["duration_ms"] == 150.0
        assert completed["output"] == {"stdout": "hello"}
        assert completed["cost"]["total_usd"] == 0.01
        assert completed["tokens"]["input_tokens"] == 100

        # call_id should match
        assert started["call_id"] == completed["call_id"]

    def test_to_jsonl_with_error(self):
        """to_jsonl includes error in completed event."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="sandbox.exec",
            input={"command": "invalid"},
            status="error",
            error={"message": "Command failed", "code": 1},
            duration_ms=50.0,
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        completed = next(e for e in events if e["type"] == "tool_call.completed")
        assert completed["status"] == "error"
        assert completed["error"]["message"] == "Command failed"

    def test_to_jsonl_with_reasoning_steps(self):
        """to_jsonl includes reasoning.step events."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_reasoning_step("Analyzing the prompt")
        transcript.add_reasoning_step("Selecting the best approach")

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        reasoning_events = [e for e in events if e["type"] == "reasoning.step"]
        assert len(reasoning_events) == 2
        assert reasoning_events[0]["step"] == "Analyzing the prompt"
        assert reasoning_events[0]["index"] == 0
        assert reasoning_events[1]["step"] == "Selecting the best approach"
        assert reasoning_events[1]["index"] == 1

    def test_to_jsonl_with_outcome(self):
        """to_jsonl includes outcome.set event."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.set_outcome(
            blocked=False,
            image_path="/outputs/result.png",
            output_data={"success": True},
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        outcome_event = next(e for e in events if e["type"] == "outcome.set")
        assert outcome_event["blocked"] is False
        assert outcome_event["has_image"] is True
        assert outcome_event["image_path"] == "/outputs/result.png"

    def test_to_jsonl_with_blocked_outcome(self):
        """to_jsonl handles blocked outcome."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.set_outcome(
            blocked=True,
            blocked_reason="Safety filter triggered",
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        outcome_event = next(e for e in events if e["type"] == "outcome.set")
        assert outcome_event["blocked"] is True
        assert outcome_event["blocked_reason"] == "Safety filter triggered"

    def test_to_jsonl_completed_event(self):
        """to_jsonl completed event includes summary stats."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="tool1",
            cost={"total_usd": 0.05},
            tokens={"input_tokens": 100, "output_tokens": 50},
            duration_ms=100.0,
        )
        transcript.add_tool_call(
            tool_name="tool2",
            cost={"total_usd": 0.10},
            tokens={"input_tokens": 200, "output_tokens": 100},
            duration_ms=200.0,
        )
        transcript.finalize(
            grader_results=[],
            final_score=0.85,
            final_passed=True,
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        completed = next(e for e in events if e["type"] == "transcript.completed")
        assert completed["final_score"] == 0.85
        assert completed["final_passed"] is True
        assert completed["tool_call_count"] == 2
        assert completed["total_cost_usd"] == pytest.approx(0.15)
        assert completed["total_tokens"] == 450

    def test_to_jsonl_events_chronological(self):
        """to_jsonl events are in chronological order."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(tool_name="tool1", duration_ms=100.0)
        transcript.add_tool_call(tool_name="tool2", duration_ms=200.0)

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        timestamps = [e["ts"] for e in events]
        assert timestamps == sorted(timestamps)

    def test_save_jsonl(self):
        """save_jsonl writes to file correctly."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(tool_name="test_tool", duration_ms=100.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            result_path = transcript.save_jsonl(path)

            assert result_path == path
            assert path.exists()

            content = path.read_text()
            lines = content.strip().split("\n")
            assert len(lines) >= 3  # started, tool events, completed

            # Each line should be valid JSON
            for line in lines:
                json.loads(line)

    def test_from_jsonl_basic(self):
        """from_jsonl reconstructs transcript from JSONL."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.input_prompt = "Test prompt"
        transcript.input_params = {"key": "value"}
        transcript.add_tool_call(
            tool_name="sandbox.exec",
            tool_type="code",
            input={"command": "echo hello"},
            output={"stdout": "hello"},
            status="ok",
            duration_ms=150.0,
            cost={"total_usd": 0.01},
            tokens={"input_tokens": 100, "output_tokens": 50},
        )
        transcript.finalize(
            grader_results=[],
            final_score=0.9,
            final_passed=True,
        )

        # Export and reimport
        jsonl = transcript.to_jsonl()
        restored = Transcript.from_jsonl(jsonl)

        # Check basic fields
        assert restored.task_id == "test-task"
        assert restored.trial_id == "test-trial"
        assert restored.input_prompt == "Test prompt"
        assert restored.input_params == {"key": "value"}

        # Check tool calls
        assert len(restored.tool_calls) == 1
        tc = restored.tool_calls[0]
        assert tc.tool_name == "sandbox.exec"
        assert tc.tool_type == "code"
        assert tc.input == {"command": "echo hello"}
        assert tc.output == {"stdout": "hello"}
        assert tc.status == "ok"
        assert tc.duration_ms == 150.0
        assert tc.cost.total_usd == 0.01
        assert tc.tokens.input_tokens == 100

        # Check grading
        assert restored.final_score == 0.9
        assert restored.final_passed is True

    def test_from_jsonl_with_reasoning(self):
        """from_jsonl restores reasoning steps."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_reasoning_step("Step 1")
        transcript.add_reasoning_step("Step 2")

        jsonl = transcript.to_jsonl()
        restored = Transcript.from_jsonl(jsonl)

        assert restored.reasoning_steps == ["Step 1", "Step 2"]

    def test_from_jsonl_with_outcome(self):
        """from_jsonl restores outcome."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.set_outcome(
            blocked=True,
            blocked_reason="Test reason",
            image_path="/test/path.png",
        )

        jsonl = transcript.to_jsonl()
        restored = Transcript.from_jsonl(jsonl)

        assert restored.outcome.blocked is True
        assert restored.outcome.blocked_reason == "Test reason"
        assert restored.outcome.image_path == "/test/path.png"

    def test_load_jsonl(self):
        """load_jsonl reads from file correctly."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="test_tool",
            input={"arg": "value"},
            output="result",
            duration_ms=100.0,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.jsonl"
            transcript.save_jsonl(path)

            loaded = Transcript.load_jsonl(path)

            assert loaded.task_id == "test-task"
            assert loaded.trial_id == "test-trial"
            assert len(loaded.tool_calls) == 1
            assert loaded.tool_calls[0].tool_name == "test_tool"

    def test_jsonl_roundtrip(self):
        """JSONL export/import roundtrip preserves key data."""
        original = Transcript(task_id="roundtrip-task", trial_id="roundtrip-trial")
        original.input_prompt = "Complex prompt"
        original.input_params = {"width": 1024, "height": 768}

        # Add multiple tool calls
        original.add_tool_call(
            tool_name="llm.chat.completion",
            tool_type="llm",
            input={"messages": [{"role": "user", "content": "Hello"}]},
            output={"content": "Hi there!"},
            status="ok",
            duration_ms=500.0,
            cost={"total_usd": 0.02, "input_cost_usd": 0.01, "output_cost_usd": 0.01},
            tokens={"input_tokens": 50, "output_tokens": 30},
            turn_index=1,
            agent_name="Planner",
        )
        original.add_tool_call(
            tool_name="sandbox.exec",
            tool_type="code",
            input={"command": "python script.py"},
            output={"exit_code": 0},
            status="ok",
            duration_ms=1000.0,
        )

        # Add reasoning
        original.add_reasoning_step("Analyzed the request")
        original.add_reasoning_step("Generated response")

        # Set outcome
        original.set_outcome(
            blocked=False,
            output_data={"success": True},
        )

        # Finalize
        original.finalize(
            grader_results=[{"name": "test", "passed": True}],
            final_score=0.95,
            final_passed=True,
        )

        # Roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "roundtrip.jsonl"
            original.save_jsonl(path)
            restored = Transcript.load_jsonl(path)

        # Verify
        assert restored.task_id == original.task_id
        assert restored.trial_id == original.trial_id
        assert restored.input_prompt == original.input_prompt
        assert restored.input_params == original.input_params
        assert len(restored.tool_calls) == 2
        assert restored.tool_calls[0].tool_name == "llm.chat.completion"
        assert restored.tool_calls[0].turn_index == 1
        assert restored.tool_calls[0].agent_name == "Planner"
        assert restored.tool_calls[1].tool_name == "sandbox.exec"
        assert restored.reasoning_steps == original.reasoning_steps
        assert restored.final_score == original.final_score
        assert restored.final_passed == original.final_passed

    def test_from_jsonl_empty_raises(self):
        """from_jsonl raises on empty content."""
        with pytest.raises(ValueError, match="Empty JSONL"):
            Transcript.from_jsonl("")

    def test_from_jsonl_missing_started_raises(self):
        """from_jsonl raises when transcript.started is missing."""
        jsonl = '{"type":"transcript.completed","ts":123}'
        with pytest.raises(ValueError, match="Missing transcript.started"):
            Transcript.from_jsonl(jsonl)

    def test_to_jsonl_retry_count(self):
        """to_jsonl includes retry_count when non-zero."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="flaky_tool",
            status="ok",
            duration_ms=100.0,
            retry_count=2,
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        completed = next(e for e in events if e["type"] == "tool_call.completed")
        assert completed["retry_count"] == 2

    def test_to_jsonl_no_retry_count_when_zero(self):
        """to_jsonl omits retry_count when zero."""
        transcript = Transcript(task_id="test-task", trial_id="test-trial")
        transcript.add_tool_call(
            tool_name="stable_tool",
            status="ok",
            duration_ms=100.0,
            retry_count=0,
        )

        jsonl = transcript.to_jsonl()
        events = [json.loads(line) for line in jsonl.strip().split("\n")]

        completed = next(e for e in events if e["type"] == "tool_call.completed")
        assert "retry_count" not in completed


# ===================================================================
# JSON save/load fidelity: timing + environment
# ===================================================================


class TestJsonLoadFidelity:
    """Regression tests: Transcript.load() must restore timing and environment.

    Before the fix, a loaded transcript reported the *load* time as start_time
    and 0ms duration, and dropped environment.model_version filled by trace
    importers — breaking `compass trace` and TRANSCRIPT graders re-run on
    saved traces.
    """

    def test_load_restores_timing(self, tmp_path):
        transcript = Transcript(task_id="t", trial_id="tr")
        transcript.add_tool_call(tool_name="a.b", input={}, output=1, duration_ms=5.0)
        transcript.finalize([], 0.9, True)

        loaded = Transcript.load(transcript.save(tmp_path / "trace.json"))

        assert loaded.start_time == transcript.start_time
        assert loaded.end_time == transcript.end_time
        assert loaded.total_duration_ms == transcript.total_duration_ms
        assert loaded.total_duration_ms > 0

    def test_load_restores_environment(self, tmp_path):
        transcript = Transcript(task_id="t", trial_id="tr")
        transcript.environment.model_version = "claude-sonnet-5"
        transcript.environment.adapter_version = "1.2.3"
        transcript.environment.random_seed = 42
        transcript.environment.attributes = {"region": "us-east-1"}

        loaded = Transcript.load(transcript.save(tmp_path / "trace.json"))

        assert loaded.environment.model_version == "claude-sonnet-5"
        assert loaded.environment.adapter_version == "1.2.3"
        assert loaded.environment.random_seed == 42
        assert loaded.environment.attributes == {"region": "us-east-1"}

    def test_load_tolerates_missing_timing_and_environment(self, tmp_path):
        """Old/minimal trace files without timing/environment still load."""
        transcript = Transcript(task_id="t", trial_id="tr")
        path = transcript.save(tmp_path / "trace.json")
        data = json.loads(path.read_text())
        del data["timing"]
        del data["environment"]
        path.write_text(json.dumps(data))

        loaded = Transcript.load(path)
        assert loaded.total_duration_ms == 0.0
        assert loaded.environment.model_version == ""

    def test_save_serializes_non_json_types(self, tmp_path):
        """save() degrades non-JSON values (datetime, Path) to strings."""
        from datetime import datetime as dt

        transcript = Transcript(task_id="t", trial_id="tr")
        transcript.set_outcome(
            output_data={"when": dt(2026, 1, 1), "where": Path("/tmp/x")},
        )
        path = transcript.save(tmp_path / "trace.json")
        data = json.loads(path.read_text())
        assert "2026" in data["outcome"]["output_data"]["when"]
