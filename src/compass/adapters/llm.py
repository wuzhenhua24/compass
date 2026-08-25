"""LLM adapter utilities for standardized tool call recording.

This module provides a mixin class and utilities for recording LLM API calls
in a standardized format, enabling cross-agent cost and token comparison.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from compass.core.transcript import (
    CostInfo,
    TokenUsage,
    format_tool_name,
)

if TYPE_CHECKING:
    from compass.adapters.base import AgentInput

logger = logging.getLogger(__name__)

# Environment variable pointing to a user pricing-override file (JSON or YAML).
PRICING_FILE_ENV = "COMPASS_PRICING_FILE"


# ============================================================================
# Model pricing (USD per 1M tokens)
# ============================================================================

@dataclass
class ModelPricing:
    """Pricing information for a model."""

    input_per_1m: float  # USD per 1M input tokens
    output_per_1m: float  # USD per 1M output tokens
    cached_input_per_1m: float | None = None  # USD per 1M cached input tokens


# Built-in default pricing table: model_id -> ModelPricing (USD per 1M tokens).
# Anthropic prices verified against platform.claude.com (2026-06); cached-input
# reflects the ~0.1x cache-read rate. Other providers are approximate and should
# be refreshed periodically. These are DEFAULTS — users override per-model rates
# at runtime via register_pricing() / load_pricing_file(), or by pointing
# COMPASS_PRICING_FILE at a JSON/YAML file. See MODEL_PRICING below.
_BUILTIN_PRICING: dict[str, ModelPricing] = {
    # OpenAI
    "gpt-4o": ModelPricing(2.50, 10.00),
    "gpt-4o-mini": ModelPricing(0.15, 0.60),
    "gpt-4-turbo": ModelPricing(10.00, 30.00),
    "gpt-4": ModelPricing(30.00, 60.00),
    "gpt-3.5-turbo": ModelPricing(0.50, 1.50),
    "o1": ModelPricing(15.00, 60.00),
    "o1-mini": ModelPricing(3.00, 12.00),
    "o1-preview": ModelPricing(15.00, 60.00),
    # Anthropic — legacy (kept for historical cost analysis)
    "claude-3-opus": ModelPricing(15.00, 75.00),
    "claude-3-sonnet": ModelPricing(3.00, 15.00),
    "claude-3-haiku": ModelPricing(0.25, 1.25),
    "claude-3-5-sonnet": ModelPricing(3.00, 15.00),
    "claude-3-5-haiku": ModelPricing(0.80, 4.00),
    "claude-sonnet-4": ModelPricing(3.00, 15.00),
    "claude-opus-4": ModelPricing(15.00, 75.00),
    # Anthropic — current (input, output, cached-input per 1M tokens)
    "claude-fable-5": ModelPricing(10.00, 50.00, 1.00),
    "claude-opus-4-8": ModelPricing(5.00, 25.00, 0.50),
    "claude-opus-4-7": ModelPricing(5.00, 25.00, 0.50),
    "claude-opus-4-6": ModelPricing(5.00, 25.00, 0.50),
    "claude-opus-4-5": ModelPricing(5.00, 25.00, 0.50),
    "claude-sonnet-5": ModelPricing(3.00, 15.00, 0.30),
    "claude-sonnet-4-6": ModelPricing(3.00, 15.00, 0.30),
    "claude-sonnet-4-5": ModelPricing(3.00, 15.00, 0.30),
    "claude-haiku-4-5": ModelPricing(1.00, 5.00, 0.10),
    # Google
    "gemini-1.5-pro": ModelPricing(1.25, 5.00),
    "gemini-1.5-flash": ModelPricing(0.075, 0.30),
    "gemini-2.0-flash": ModelPricing(0.10, 0.40),
    # Mistral
    "mistral-large": ModelPricing(2.00, 6.00),
    "mistral-small": ModelPricing(0.20, 0.60),
    "codestral": ModelPricing(0.20, 0.60),
    # Cohere
    "command-r-plus": ModelPricing(2.50, 10.00),
    "command-r": ModelPricing(0.15, 0.60),
}

# Effective pricing table = built-in defaults overlaid with user overrides.
# get_model_pricing()/calculate_cost() read this. Mutate it through
# register_pricing() / load_pricing_file(); restore defaults with reset_pricing().
MODEL_PRICING: dict[str, ModelPricing] = dict(_BUILTIN_PRICING)


def _normalize_key(model: str) -> str:
    """Normalise a model id to the table's key convention (lower, dashed)."""
    return model.strip().lower().replace("_", "-")


def _to_pricing(
    spec: ModelPricing | dict[str, Any] | tuple[float, ...] | list[float],
) -> ModelPricing:
    """Coerce a user-supplied pricing spec into a ModelPricing.

    Accepts a ModelPricing, a ``(input, output[, cached])`` tuple/list, or a
    dict using either friendly keys (``input``/``output``/``cached``) or the
    explicit field names (``input_per_1m``/``output_per_1m``/``cached_input_per_1m``).
    """
    if isinstance(spec, ModelPricing):
        return spec
    if isinstance(spec, (tuple, list)):
        return ModelPricing(*(float(x) for x in spec))
    if isinstance(spec, dict):
        inp = spec.get("input", spec.get("input_per_1m"))
        out = spec.get("output", spec.get("output_per_1m"))
        cached = spec.get("cached", spec.get("cached_input_per_1m"))
        if inp is None or out is None:
            raise ValueError(
                f"Pricing entry must define 'input' and 'output': {spec!r}"
            )
        return ModelPricing(
            float(inp), float(out), None if cached is None else float(cached)
        )
    raise TypeError(f"Unsupported pricing spec: {spec!r}")


def register_pricing(
    model: str,
    pricing: ModelPricing | dict[str, Any] | tuple[float, ...] | list[float],
) -> None:
    """Add or override the price for a single model at runtime.

    Overrides take precedence over the built-in defaults for all subsequent
    ``get_model_pricing`` / ``calculate_cost`` lookups.

    Example::

        register_pricing("gpt-5", {"input": 2.0, "output": 8.0, "cached": 0.2})
        register_pricing("my-model", (1.5, 6.0))
    """
    MODEL_PRICING[_normalize_key(model)] = _to_pricing(pricing)


def load_pricing_file(path: str | Path) -> int:
    """Merge model price overrides from a JSON or YAML file into MODEL_PRICING.

    The file is a mapping of model id -> pricing spec, e.g.::

        {"gpt-5": {"input": 2.0, "output": 8.0, "cached": 0.2},
         "my-model": {"input": 1.5, "output": 6.0}}

    Format is chosen by extension (``.json`` -> JSON, otherwise YAML; YAML also
    parses JSON). Returns the number of models loaded.
    """
    path = Path(path)
    raw = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(raw)
    else:
        import yaml

        data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ValueError(
            f"Pricing file {path} must contain a mapping of model -> pricing"
        )
    for model, spec in data.items():
        register_pricing(model, spec)
    return len(data)


def reset_pricing() -> None:
    """Restore the built-in default pricing table, discarding all overrides."""
    MODEL_PRICING.clear()
    MODEL_PRICING.update(_BUILTIN_PRICING)


def load_default_pricing() -> int:
    """Load user pricing overrides from the environment or default location.

    Precedence: the file named by ``COMPASS_PRICING_FILE`` if set; otherwise the
    first of ``~/.compass/pricing.{json,yaml,yml}`` that exists. Failures are
    logged and swallowed so a bad file never breaks ``import compass``.
    Returns the number of overrides loaded (0 if none found).
    """
    env = os.environ.get(PRICING_FILE_ENV)
    if env:
        path = Path(env)
        if not path.is_file():
            logger.warning("%s points to a missing file: %s", PRICING_FILE_ENV, path)
            return 0
        candidates = [path]
    else:
        base = Path.home() / ".compass"
        candidates = [base / "pricing.json", base / "pricing.yaml", base / "pricing.yml"]

    for path in candidates:
        if not path.is_file():
            continue
        try:
            count = load_pricing_file(path)
            logger.info("Loaded %d model price override(s) from %s", count, path)
            return count
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load pricing file %s: %s", path, exc)
            return 0
    return 0


def get_model_pricing(model: str) -> ModelPricing | None:
    """Get pricing for a model, with fuzzy matching.

    Args:
        model: Model name or ID.

    Returns:
        ModelPricing if found, None otherwise.
    """
    # Exact match
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Normalize and try again
    normalized = model.lower().replace("_", "-")
    if normalized in MODEL_PRICING:
        return MODEL_PRICING[normalized]

    # Longest-prefix match (e.g. "gpt-4o-2024-08-06" -> "gpt-4o",
    # "claude-opus-4-8-20260101" -> "claude-opus-4-8", not the shorter
    # "claude-opus-4"). Picking the longest match makes the result independent
    # of dict insertion order.
    best_key: str | None = None
    for key in MODEL_PRICING:
        if normalized.startswith(key) and (best_key is None or len(key) > len(best_key)):
            best_key = key
    if best_key is not None:
        return MODEL_PRICING[best_key]

    return None


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> CostInfo | None:
    """Calculate cost for a model invocation.

    Args:
        model: Model name or ID.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        cached_tokens: Number of cached input tokens (if applicable).

    Returns:
        CostInfo with calculated costs, or None if pricing unknown.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing.input_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m

    # Handle cached tokens if pricing available
    if cached_tokens > 0 and pricing.cached_input_per_1m is not None:
        cached_cost = (cached_tokens / 1_000_000) * pricing.cached_input_per_1m
        input_cost += cached_cost

    total_cost = input_cost + output_cost

    return CostInfo(
        total_usd=total_cost,
        input_cost_usd=input_cost,
        output_cost_usd=output_cost,
        metadata={"model": model},
    )


# Apply user pricing overrides at import (env var or ~/.compass/pricing.*).
load_default_pricing()


# ============================================================================
# Token extraction from different provider responses
# ============================================================================


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


# ============================================================================
# LLM Tool Call Mixin
# ============================================================================


class LLMToolCallMixin:
    """Mixin for adapters that make LLM API calls.

    Provides standardized recording of LLM tool calls with proper
    cost and token extraction.

    Usage:
        class MyLLMAdapter(Adapter, LLMToolCallMixin):
            async def run(self, input: AgentInput) -> AgentOutput:
                response = await self._call_openai(...)
                self._record_llm_call(
                    input,
                    provider="openai",
                    model="gpt-4o",
                    messages=[...],
                    response=response,
                    duration_ms=150.0,
                )
    """

    def _record_llm_call(
        self,
        agent_input: AgentInput,
        *,
        provider: str,
        model: str,
        messages: list[dict[str, Any]] | None = None,
        response: Any = None,
        duration_ms: float = 0.0,
        status: str = "ok",
        error: dict[str, Any] | None = None,
        # Override auto-extracted values
        tokens: TokenUsage | None = None,
        cost: CostInfo | None = None,
        # Additional metadata
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an LLM API call with standardized format.

        Args:
            agent_input: The AgentInput for context access.
            provider: LLM provider ("openai", "anthropic", "google", etc.)
            model: Model name or ID.
            messages: Input messages (optional, for logging).
            response: Raw API response for token extraction.
            duration_ms: Call duration in milliseconds.
            status: Call status ("ok", "error", "blocked").
            error: Error details if status is "error".
            tokens: Override auto-extracted token usage.
            cost: Override auto-calculated cost.
            metadata: Additional metadata to include.
        """
        # Extract tokens from response if not provided
        if tokens is None and response is not None:
            tokens = extract_usage(provider, response)

        # Calculate cost if not provided
        if cost is None and tokens is not None:
            cost = calculate_cost(
                model,
                tokens.input_tokens,
                tokens.output_tokens,
            )

        # Build tool name
        tool_name = format_tool_name(provider, "completion", "chat")

        # Build input summary (avoid logging full messages in production)
        input_summary: dict[str, Any] = {"model": model}
        if messages:
            input_summary["message_count"] = len(messages)
            # Optionally include first message for debugging
            if messages and len(messages) > 0:
                first_msg = messages[0]
                input_summary["first_role"] = first_msg.get("role", "unknown")

        # Build output summary
        output_summary: dict[str, Any] | None = None
        if response is not None:
            output_summary = {}
            # Extract content length if available
            if hasattr(response, "choices") and response.choices:
                choice = response.choices[0]
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content
                    if content:
                        output_summary["content_length"] = len(content)
            elif isinstance(response, dict) and "choices" in response:
                choices = response["choices"]
                if choices and "message" in choices[0]:
                    content = choices[0]["message"].get("content", "")
                    if content:
                        output_summary["content_length"] = len(content)

        # Merge metadata
        full_metadata = {
            "provider": provider,
            "model": model,
        }
        if metadata:
            full_metadata.update(metadata)

        # Record the tool call using parent class method
        # This assumes the class also inherits from Adapter
        self._record_tool_call(  # type: ignore[attr-defined]
            agent_input,
            tool_name=tool_name,
            tool_type="llm",
            input=input_summary,
            output=output_summary,
            status=status,
            duration_ms=duration_ms,
            error=error,
            cost=cost,
            tokens=tokens,
            metadata=full_metadata,
        )


# ============================================================================
# Structured completions
# ============================================================================
#
# One place that knows how to ask a provider for JSON matching a schema. The
# two providers disagree about how you ask — OpenAI takes a `response_format`,
# Anthropic takes a single-tool `tool_choice` — and that difference is not
# something every caller should have to carry.


async def structured_completion(
    prompt: str,
    *,
    schema: dict[str, Any],
    model: str,
    provider: str = "openai",
    system: str = "",
    schema_name: str = "structured_output",
    max_tokens: int = 4096,
    temperature: float = 0.1,
) -> dict[str, Any]:
    """Ask *model* for JSON matching *schema*, and return it parsed.

    Args:
        prompt: The user message.
        schema: JSON Schema the reply must match.
        model: Model id.
        provider: ``"openai"`` or ``"anthropic"``.
        system: System message. OpenAI takes it as a system role; Anthropic
            has no system role in this call shape, so it is prepended.
        schema_name: Name the provider attaches to the schema.
        max_tokens: Reply cap (Anthropic requires one).
        temperature: Low by default — a judge that rephrases itself run to run
            is harder to trust than one that repeats itself.

    Raises:
        ValueError: On an unsupported provider, or a reply that carried no
            structured content.
        ImportError: If the provider's package is not installed.
    """
    if provider == "openai":
        return await _openai_structured(
            prompt,
            schema=schema,
            model=model,
            system=system,
            schema_name=schema_name,
            temperature=temperature,
        )
    if provider == "anthropic":
        return await _anthropic_structured(
            prompt,
            schema=schema,
            model=model,
            system=system,
            schema_name=schema_name,
            max_tokens=max_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


async def _openai_structured(
    prompt: str,
    *,
    schema: dict[str, Any],
    model: str,
    system: str,
    schema_name: str,
    temperature: float,
) -> dict[str, Any]:
    try:
        from openai import AsyncOpenAI
    except ImportError as exc:
        raise ImportError(
            "openai package required for OpenAI provider. "
            "Install with: pip install openai"
        ) from exc

    client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": schema_name, "strict": True, "schema": schema},
        },
        temperature=temperature,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("empty reply from OpenAI")
    parsed: dict[str, Any] = json.loads(content)
    return parsed


async def _anthropic_structured(
    prompt: str,
    *,
    schema: dict[str, Any],
    model: str,
    system: str,
    schema_name: str,
    max_tokens: int,
) -> dict[str, Any]:
    try:
        from anthropic import AsyncAnthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required for Anthropic provider. "
            "Install with: pip install anthropic"
        ) from exc

    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    content = f"{system}\n\n{prompt}" if system else prompt

    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        tools=[
            {
                "name": schema_name,
                "description": "Submit the structured result",
                "input_schema": schema,
            }
        ],
        tool_choice={"type": "tool", "name": schema_name},
        messages=[{"role": "user", "content": content}],
    )
    for block in response.content:
        if block.type == "tool_use" and block.name == schema_name:
            result: dict[str, Any] = dict(block.input)
            return result
    raise ValueError(f"no {schema_name} tool use in the reply from Anthropic")
