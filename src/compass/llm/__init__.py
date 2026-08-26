"""LLM plumbing for the control plane: pricing, usage extraction, judges.

Nothing here drives an agent. These are the three things the *scoring* side
needs from a provider: what a recorded run cost (:mod:`~compass.llm.pricing`),
how to read token counts out of a raw response (:mod:`~compass.llm.usage`), and
how to ask for JSON matching a schema (:mod:`~compass.llm.completion`).

They used to live in ``compass.adapters.llm``, which meant a trace importer or
an LLM judge could not be imported without registering every agent driver
behind it. The drivers are data plane and belong to the harness that runs the
agent; these do not. Only :class:`~compass.adapters.llm.LLMToolCallMixin`,
which records a call an adapter is making, stayed behind.

Importing this package is cheap — the provider SDKs are imported inside the
call that needs them, so a deployment that only imports traces never pays for
``openai`` or ``anthropic``.
"""

from compass.llm.completion import structured_completion
from compass.llm.pricing import (
    MODEL_PRICING,
    ModelPricing,
    calculate_cost,
    get_model_pricing,
    load_default_pricing,
    load_pricing_file,
    register_pricing,
    reset_pricing,
)
from compass.llm.usage import (
    extract_anthropic_usage,
    extract_google_usage,
    extract_openai_usage,
    extract_usage,
)

__all__ = [
    # Pricing
    "ModelPricing",
    "MODEL_PRICING",
    "get_model_pricing",
    "calculate_cost",
    "register_pricing",
    "load_pricing_file",
    "load_default_pricing",
    "reset_pricing",
    # Usage extraction
    "extract_usage",
    "extract_openai_usage",
    "extract_anthropic_usage",
    "extract_google_usage",
    # Structured completions
    "structured_completion",
]
