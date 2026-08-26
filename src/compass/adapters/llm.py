"""Recording an LLM call that an adapter is making.

Data plane. This is the one piece of the old, larger ``compass.adapters.llm``
that is genuinely an adapter concern: it needs an
:class:`~compass.adapters.base.AgentInput` and the ``_record_tool_call`` its
host Adapter provides, so it only makes sense while Compass is driving the
agent itself.

The pricing table, the usage extractors and the structured-completion client
moved to :mod:`compass.llm` — the scoring side needs those to price an imported
trajectory or run a judge, and should not have to register every agent driver
to get at them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from compass.core.transcript import (
    CostInfo,
    TokenUsage,
    format_tool_name,
)
from compass.llm import calculate_cost, extract_usage

if TYPE_CHECKING:
    from compass.adapters.base import AgentInput


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
