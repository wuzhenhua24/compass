"""Token usage extraction from raw provider responses.

Control plane. Every provider reports the same two numbers under different
names, and each one's SDK object disagrees with its own raw dict. These four
functions are the single place that knows the difference, so everything
downstream sees one :class:`~compass.core.transcript.TokenUsage`.
"""

from __future__ import annotations

from typing import Any

from compass.core.transcript import TokenUsage


def extract_openai_usage(response: Any) -> TokenUsage | None:
    """Extract token usage from OpenAI API response.

    Works with both openai library response objects and raw dicts.
    """
    usage = None

    # Try attribute access (openai library objects)
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        return TokenUsage(
            input_tokens=getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "completion_tokens", 0),
            total_tokens=getattr(usage, "total_tokens", 0),
            metadata={"provider": "openai"},
        )

    # Try dict access
    if isinstance(response, dict) and "usage" in response:
        usage = response["usage"]
        return TokenUsage(
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            metadata={"provider": "openai"},
        )

    return None


def extract_anthropic_usage(response: Any) -> TokenUsage | None:
    """Extract token usage from Anthropic API response.

    Works with both anthropic library response objects and raw dicts.
    """
    usage = None

    # Try attribute access (anthropic library objects)
    if hasattr(response, "usage") and response.usage is not None:
        usage = response.usage
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            metadata={"provider": "anthropic"},
        )

    # Try dict access
    if isinstance(response, dict) and "usage" in response:
        usage = response["usage"]
        return TokenUsage(
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            metadata={"provider": "anthropic"},
        )

    return None


def extract_google_usage(response: Any) -> TokenUsage | None:
    """Extract token usage from Google Gemini API response."""
    # Try attribute access
    if hasattr(response, "usage_metadata"):
        meta = response.usage_metadata
        return TokenUsage(
            input_tokens=getattr(meta, "prompt_token_count", 0),
            output_tokens=getattr(meta, "candidates_token_count", 0),
            total_tokens=getattr(meta, "total_token_count", 0),
            metadata={"provider": "google"},
        )

    # Try dict access
    if isinstance(response, dict) and "usageMetadata" in response:
        meta = response["usageMetadata"]
        return TokenUsage(
            input_tokens=meta.get("promptTokenCount", 0),
            output_tokens=meta.get("candidatesTokenCount", 0),
            total_tokens=meta.get("totalTokenCount", 0),
            metadata={"provider": "google"},
        )

    return None


def extract_usage(provider: str, response: Any) -> TokenUsage | None:
    """Extract token usage from an LLM API response.

    Args:
        provider: LLM provider name ("openai", "anthropic", "google", etc.)
        response: Raw API response object or dict.

    Returns:
        TokenUsage if extraction successful, None otherwise.
    """
    extractors = {
        "openai": extract_openai_usage,
        "azure": extract_openai_usage,  # Azure uses OpenAI format
        "anthropic": extract_anthropic_usage,
        "google": extract_google_usage,
        "gemini": extract_google_usage,
    }

    extractor = extractors.get(provider.lower())
    if extractor:
        return extractor(response)

    # Fallback: try common patterns
    if hasattr(response, "usage"):
        usage = response.usage
        return TokenUsage(
            input_tokens=getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0)
            or getattr(usage, "completion_tokens", 0),
            metadata={"provider": provider},
        )

    return None
