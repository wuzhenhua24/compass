"""Agent adapters module."""

# Import built-in adapters to trigger registration
from compass.adapters import (
    coding,  # noqa: F401
    environment,  # noqa: F401
    image,  # noqa: F401
)
from compass.adapters.base import Adapter, AgentInput, AgentOutput
from compass.adapters.llm import (
    MODEL_PRICING,
    LLMToolCallMixin,
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
    # LLM utilities
    "LLMToolCallMixin",
    "ModelPricing",
    "MODEL_PRICING",
    "get_model_pricing",
    "calculate_cost",
    "register_pricing",
    "load_pricing_file",
    "load_default_pricing",
    "reset_pricing",
    "extract_usage",
    "extract_openai_usage",
    "extract_anthropic_usage",
    "extract_google_usage",
]
