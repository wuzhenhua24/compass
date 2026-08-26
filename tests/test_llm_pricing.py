"""Tests for compass.llm — model pricing and provider usage extraction.

These cover the control-plane half of what used to be ``compass.adapters.llm``:
pricing a run someone else recorded, and reading token counts out of a raw
provider response. The adapter-side recorder is tested in test_llm_adapter.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from compass.llm import (
    MODEL_PRICING,
    ModelPricing,
    calculate_cost,
    extract_anthropic_usage,
    extract_google_usage,
    extract_openai_usage,
    extract_usage,
    get_model_pricing,
    load_default_pricing,
    load_pricing_file,
    register_pricing,
    reset_pricing,
)


@pytest.fixture(autouse=True)
def _pricing_defaults():
    """Isolate every test from machine-level pricing files (COMPASS_PRICING_FILE
    or ~/.compass/pricing.*) that ``load_default_pricing()`` applies at import."""
    reset_pricing()
    yield
    reset_pricing()


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

    def test_current_anthropic_models_present(self):
        """Current-generation Anthropic models are priced (stale-table fix)."""
        assert MODEL_PRICING["claude-opus-4-8"].input_per_1m == 5.00
        assert MODEL_PRICING["claude-opus-4-8"].output_per_1m == 25.00
        assert MODEL_PRICING["claude-sonnet-5"].input_per_1m == 3.00
        assert MODEL_PRICING["claude-haiku-4-5"].input_per_1m == 1.00
        # Cached-input reflects the ~0.1x cache-read rate.
        assert MODEL_PRICING["claude-opus-4-8"].cached_input_per_1m == 0.50

    def test_get_model_pricing_longest_prefix_wins(self):
        """A dated ID must resolve to the longest matching key, not a shorter
        prefix that happens to iterate first (order-independent matching).

        'claude-opus-4-8-...' must map to claude-opus-4-8 ($5/$25), NOT the
        legacy 'claude-opus-4' ($15/$75) which is inserted earlier.
        """
        pricing = get_model_pricing("claude-opus-4-8-20260101")
        assert pricing == MODEL_PRICING["claude-opus-4-8"]
        assert pricing != MODEL_PRICING["claude-opus-4"]

    def test_calculate_cost_with_cached_tokens(self):
        """Cached input tokens are billed at the cached rate on top of input."""
        cost = calculate_cost(
            "claude-opus-4-8",
            input_tokens=1_000_000,
            output_tokens=0,
            cached_tokens=1_000_000,
        )
        assert cost is not None
        # 1M fresh input @ $5 + 1M cached input @ $0.50 = $5.50
        assert cost.input_cost_usd == pytest.approx(5.50)

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


class TestPricingOverrides:
    """User-configurable pricing: register_pricing / load_pricing_file / env."""

    def test_register_overrides_existing_model(self):
        register_pricing("gpt-4o", {"input": 9.99, "output": 19.99})
        pricing = get_model_pricing("gpt-4o")
        assert pricing.input_per_1m == 9.99
        assert pricing.output_per_1m == 19.99
        # And it flows through calculate_cost.
        cost = calculate_cost("gpt-4o", input_tokens=1_000_000, output_tokens=0)
        assert cost.total_usd == pytest.approx(9.99)

    def test_register_new_model_via_tuple(self):
        register_pricing("my-private-model", (1.5, 6.0))
        pricing = get_model_pricing("my-private-model")
        assert pricing.input_per_1m == 1.5
        assert pricing.output_per_1m == 6.0

    def test_register_normalizes_key(self):
        register_pricing("GPT_5", {"input": 2.0, "output": 8.0, "cached": 0.2})
        # Lookup with any casing / separator resolves to the same entry.
        assert get_model_pricing("gpt-5").input_per_1m == 2.0
        assert get_model_pricing("gpt_5").cached_input_per_1m == 0.2

    def test_register_accepts_model_pricing_instance(self):
        register_pricing("x-model", ModelPricing(3.0, 9.0, 0.3))
        assert get_model_pricing("x-model").cached_input_per_1m == 0.3

    def test_register_rejects_incomplete_dict(self):
        with pytest.raises(ValueError):
            register_pricing("bad", {"input": 1.0})  # missing output

    def test_load_pricing_file_json(self, tmp_path):
        import json

        f = tmp_path / "pricing.json"
        f.write_text(json.dumps({
            "gpt-4o": {"input": 1.0, "output": 2.0},
            "brand-new": {"input": 4.0, "output": 8.0, "cached": 0.4},
        }))
        n = load_pricing_file(f)
        assert n == 2
        assert get_model_pricing("gpt-4o").input_per_1m == 1.0
        assert get_model_pricing("brand-new").cached_input_per_1m == 0.4

    def test_load_pricing_file_yaml(self, tmp_path):
        f = tmp_path / "pricing.yaml"
        f.write_text("my-model:\n  input: 5.0\n  output: 10.0\n")
        assert load_pricing_file(f) == 1
        assert get_model_pricing("my-model").output_per_1m == 10.0

    def test_load_default_pricing_from_env(self, tmp_path, monkeypatch):
        import json

        f = tmp_path / "custom.json"
        f.write_text(json.dumps({"env-model": {"input": 7.0, "output": 14.0}}))
        monkeypatch.setenv("COMPASS_PRICING_FILE", str(f))
        count = load_default_pricing()
        assert count == 1
        assert get_model_pricing("env-model").input_per_1m == 7.0

    def test_load_default_pricing_missing_env_file_is_safe(self, tmp_path, monkeypatch):
        monkeypatch.setenv("COMPASS_PRICING_FILE", str(tmp_path / "nope.json"))
        assert load_default_pricing() == 0  # warns, doesn't raise

    def test_reset_pricing_restores_defaults(self):
        register_pricing("gpt-4o", {"input": 0.01, "output": 0.01})
        assert get_model_pricing("gpt-4o").input_per_1m == 0.01
        reset_pricing()
        assert get_model_pricing("gpt-4o").input_per_1m == 2.50  # built-in default


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
        openai_response = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
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
