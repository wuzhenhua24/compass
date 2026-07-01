"""Agent adapters module."""

from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.registry import register_adapter, get_adapter, list_adapters
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

# Import built-in adapters to trigger registration
from compass.adapters import image  # noqa: F401
from compass.adapters import coding  # noqa: F401
from compass.adapters import environment  # noqa: F401

__all__ = [
    # Base
    "Adapter",
    "AgentInput",
    "AgentOutput",
    # Registry
    "register_adapter",
    "get_adapter",
    "list_adapters",
    # LLM utilities
    "LLMToolCallMixin",
    "ModelPricing",
    "MODEL_PRICING",
    "get_model_pricing",
    "calculate_cost",
    "extract_usage",
    "extract_openai_usage",
    "extract_anthropic_usage",
    "extract_google_usage",
]
