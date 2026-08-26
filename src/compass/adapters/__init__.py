"""Agent adapters module."""

# Import built-in adapters to trigger registration
from compass.adapters import (
    claude_code,  # noqa: F401
    codex,  # noqa: F401
    coding,  # noqa: F401
    environment,  # noqa: F401
    image,  # noqa: F401
    pi,  # noqa: F401
)
from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.llm import LLMToolCallMixin
from compass.adapters.registry import get_adapter, list_adapters, register_adapter

__all__ = [
    # Base
    "Adapter",
    "AgentInput",
    "AgentOutput",
    # Registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
    # LLM call recording (pricing and provider access live in compass.llm)
    "LLMToolCallMixin",
]
