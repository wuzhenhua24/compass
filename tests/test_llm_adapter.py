"""Tests for LLM adapter utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from compass.adapters.llm import (
    LLMToolCallMixin,
    ModelPricing,
    MODEL_PRICING,
    get_model_pricing,
    calculate_cost,
    extract_usage,
    extract_openai_usage,
    extract_anthropic_usage,
    extract_google_usage,
)
from compass.core.transcript import CostInfo, TokenUsage, ToolCall


# ===================================================================
# Model pricing tests
# ===================================================================


class TestModelPricing:
    """Tests for model pricing utilities."""

    def test_known_models_have_pricing(self):
        """Common models are in the pricing table."""
        assert "gpt-4o" in MODEL_PRICING
        assert "claude-3-opus" in MODEL_PRICING
        assert "gemini-1.5-pro" in MODEL_PRICING

    def test_get_model_pricing_exact(self):
        """Exact model name lookup."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing is not None
        assert pricing.input_per_1m > 0
        assert pricing.output_per_1m > 0

    def test_get_model_pricing_normalized(self):
        """Normalized model name lookup (underscores to dashes)."""
        pricing = get_model_pricing("gpt_4o")
        assert pricing is not None

    def test_get_model_pricing_prefix_match(self):
        """Prefix matching for versioned models."""
        pricing = get_model_pricing("gpt-4o-2024-08-06")
        assert pricing is not None
        # Should match gpt-4o
        assert pricing == MODEL_PRICING["gpt-4o"]

    def test_get_model_pricing_unknown(self):
        """Unknown model returns None."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing is None

    def test_calculate_cost_known_model(self):
        """Calculate cost for known model."""
        cost = calculate_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert cost is not None
        assert cost.total_usd > 0
        assert cost.input_cost_usd is not None
        assert cost.output_cost_usd is not None
        # Verify calculation (gpt-4o: $2.50/1M input, $10/1M output)
        expected_input = (1000 / 1_000_000) * 2.50
        expected_output = (500 / 1_000_000) * 10.00
        assert cost.input_cost_usd == pytest.approx(expected_input)
        assert cost.output_cost_usd == pytest.approx(expected_output)
        assert cost.total_usd == pytest.approx(expected_input + expected_output)

    def test_calculate_cost_unknown_model(self):
        """Calculate cost returns None for unknown model."""
        cost = calculate_cost("unknown-model", input_tokens=1000, output_tokens=500)
        assert cost is None

    def test_calculate_cost_metadata_includes_model(self):
        """Cost metadata includes model name."""
        cost = calculate_cost("gpt-4o", input_tokens=100, output_tokens=50)
        assert cost is not None
        assert cost.metadata.get("model") == "gpt-4o"


# ===================================================================
# Token extraction tests
# ===================================================================


class TestTokenExtraction:
    """Tests for token extraction from API responses."""

    def test_extract_openai_usage_object(self):
        """Extract from OpenAI response object."""
        # Mock OpenAI response object
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.prompt_tokens = 100
        response.usage.completion_tokens = 50
        response.usage.total_tokens = 150

        tokens = extract_openai_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 100
        assert tokens.output_tokens == 50
        assert tokens.total_tokens == 150

    def test_extract_openai_usage_dict(self):
        """Extract from OpenAI response dict."""
        response = {
            "usage": {
                "prompt_tokens": 200,
                "completion_tokens": 100,
                "total_tokens": 300,
            }
        }

        tokens = extract_openai_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 200
        assert tokens.output_tokens == 100
        assert tokens.total_tokens == 300

    def test_extract_anthropic_usage_object(self):
        """Extract from Anthropic response object."""
        response = MagicMock()
        response.usage = MagicMock()
        response.usage.input_tokens = 150
        response.usage.output_tokens = 75

        tokens = extract_anthropic_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 150
        assert tokens.output_tokens == 75

    def test_extract_anthropic_usage_dict(self):
        """Extract from Anthropic response dict."""
        response = {
            "usage": {
                "input_tokens": 250,
                "output_tokens": 125,
            }
        }

        tokens = extract_anthropic_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 250
        assert tokens.output_tokens == 125

    def test_extract_google_usage_object(self):
        """Extract from Google Gemini response object."""
        response = MagicMock()
        response.usage_metadata = MagicMock()
        response.usage_metadata.prompt_token_count = 300
        response.usage_metadata.candidates_token_count = 150
        response.usage_metadata.total_token_count = 450

        tokens = extract_google_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 300
        assert tokens.output_tokens == 150
        assert tokens.total_tokens == 450

    def test_extract_google_usage_dict(self):
        """Extract from Google Gemini response dict."""
        response = {
            "usageMetadata": {
                "promptTokenCount": 400,
                "candidatesTokenCount": 200,
                "totalTokenCount": 600,
            }
        }

        tokens = extract_google_usage(response)
        assert tokens is not None
        assert tokens.input_tokens == 400
        assert tokens.output_tokens == 200
        assert tokens.total_tokens == 600

    def test_extract_usage_dispatches_by_provider(self):
        """extract_usage dispatches to correct extractor."""
        openai_response = {"usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}
        anthropic_response = {"usage": {"input_tokens": 20, "output_tokens": 10}}

        openai_tokens = extract_usage("openai", openai_response)
        assert openai_tokens is not None
        assert openai_tokens.input_tokens == 10

        anthropic_tokens = extract_usage("anthropic", anthropic_response)
        assert anthropic_tokens is not None
        assert anthropic_tokens.input_tokens == 20

    def test_extract_usage_azure_uses_openai_format(self):
        """Azure uses OpenAI format."""
        response = {"usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75}}
        tokens = extract_usage("azure", response)
        assert tokens is not None
        assert tokens.input_tokens == 50

    def test_extract_usage_no_usage_returns_none(self):
        """Response without usage returns None."""
        response = {"choices": [{"message": {"content": "Hello"}}]}
        tokens = extract_openai_usage(response)
        assert tokens is None


# ===================================================================
# LLM Tool Call Mixin tests
# ===================================================================


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
