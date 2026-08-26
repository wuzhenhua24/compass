"""Tests for LLMToolCallMixin — the adapter-side recorder for an LLM call.

The pricing table and the provider usage extractors it leans on moved to
``compass.llm`` and are tested in test_llm_pricing.py; what is left here is the
part that only makes sense while Compass is driving the agent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from compass.adapters.llm import LLMToolCallMixin
from compass.core.transcript import CostInfo, TokenUsage, ToolCall
from compass.llm import reset_pricing


@pytest.fixture(autouse=True)
def _pricing_defaults():
    """Isolate every test from machine-level pricing files (COMPASS_PRICING_FILE
    or ~/.compass/pricing.*) that ``load_default_pricing()`` applies at import."""
    reset_pricing()
    yield
    reset_pricing()


class MockAdapter:
    """Mock adapter for testing the mixin."""

    def __init__(self):
        self.recorded_calls: list[ToolCall] = []

    def _record_tool_call(self, agent_input, **kwargs):
        """Mock recording that captures the call."""
        self.recorded_calls.append(ToolCall(**kwargs))


class TestLLMAdapter(MockAdapter, LLMToolCallMixin):
    """Test adapter combining mock and mixin."""

    pass


@dataclass
class MockAgentInput:
    """Mock AgentInput for testing."""

    prompt: str = ""
    context: dict = None

    def __post_init__(self):
        if self.context is None:
            self.context = {}


class TestLLMToolCallMixin:
    """Tests for LLMToolCallMixin."""

    def test_record_llm_call_basic(self):
        """Basic LLM call recording."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            duration_ms=150.0,
        )

        assert len(adapter.recorded_calls) == 1
        tc = adapter.recorded_calls[0]
        assert tc.tool_name == "openai.chat.completion"
        assert tc.tool_type == "llm"
        assert tc.duration_ms == 150.0
        assert tc.metadata["provider"] == "openai"
        assert tc.metadata["model"] == "gpt-4o"

    def test_record_llm_call_with_response(self):
        """LLM call with response extracts tokens and calculates cost."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        response = {
            "usage": {
                "prompt_tokens": 1000,
                "completion_tokens": 500,
                "total_tokens": 1500,
            }
        }

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            response=response,
            duration_ms=200.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.tokens is not None
        assert tc.tokens.input_tokens == 1000
        assert tc.tokens.output_tokens == 500

        assert tc.cost is not None
        assert tc.cost.total_usd > 0

    def test_record_llm_call_with_messages(self):
        """LLM call with messages records message count."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello!"},
        ]

        adapter._record_llm_call(
            agent_input,
            provider="anthropic",
            model="claude-3-opus",
            messages=messages,
            duration_ms=100.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.input["message_count"] == 2
        assert tc.input["first_role"] == "system"

    def test_record_llm_call_with_error(self):
        """LLM call with error status."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            status="error",
            error={"type": "RateLimitError", "message": "Rate limit exceeded"},
            duration_ms=50.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.status == "error"
        assert tc.error is not None
        assert tc.error["type"] == "RateLimitError"

    def test_record_llm_call_override_tokens(self):
        """Override auto-extracted tokens."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        custom_tokens = TokenUsage(input_tokens=999, output_tokens=111)

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            tokens=custom_tokens,
            duration_ms=100.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.tokens.input_tokens == 999
        assert tc.tokens.output_tokens == 111

    def test_record_llm_call_override_cost(self):
        """Override auto-calculated cost."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        custom_cost = CostInfo(total_usd=0.99)

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            cost=custom_cost,
            duration_ms=100.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.cost.total_usd == 0.99

    def test_record_llm_call_custom_metadata(self):
        """Custom metadata is merged."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        adapter._record_llm_call(
            agent_input,
            provider="openai",
            model="gpt-4o",
            metadata={"custom_key": "custom_value", "request_id": "abc123"},
            duration_ms=100.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.metadata["provider"] == "openai"
        assert tc.metadata["model"] == "gpt-4o"
        assert tc.metadata["custom_key"] == "custom_value"
        assert tc.metadata["request_id"] == "abc123"

    def test_record_llm_call_anthropic_format(self):
        """Anthropic response format works correctly."""
        adapter = TestLLMAdapter()
        agent_input = MockAgentInput()

        response = {
            "usage": {
                "input_tokens": 500,
                "output_tokens": 250,
            }
        }

        adapter._record_llm_call(
            agent_input,
            provider="anthropic",
            model="claude-3-sonnet",
            response=response,
            duration_ms=180.0,
        )

        tc = adapter.recorded_calls[0]
        assert tc.tool_name == "anthropic.chat.completion"
        assert tc.tokens.input_tokens == 500
        assert tc.tokens.output_tokens == 250
