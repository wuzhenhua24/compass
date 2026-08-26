"""Model pricing — what a recorded run cost.

Control plane. A trace importer reads token counts out of a trajectory someone
else produced and prices them here, long after the agent has exited; nothing in
this module talks to a provider or drives anything.

The built-in table is a *default*, not a claim. Prices move, so users override
per model at runtime via :func:`register_pricing` / :func:`load_pricing_file`,
or by pointing ``COMPASS_PRICING_FILE`` at a JSON/YAML file — overrides win.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from compass.core.transcript import CostInfo

logger = logging.getLogger(__name__)

# Environment variable pointing to a user pricing-override file (JSON or YAML).
PRICING_FILE_ENV = "COMPASS_PRICING_FILE"


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
