"""Structured completions — ask a provider for JSON matching a schema.

Control plane: this is how the model graders and ``compass insights`` reach an
LLM. It is deliberately the *only* such path, because the two providers
disagree about how you ask — OpenAI takes a `response_format`, Anthropic takes
a single-tool `tool_choice` — and that difference is not something every judge
should have to carry.
"""

from __future__ import annotations

import json
import os
from typing import Any


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
