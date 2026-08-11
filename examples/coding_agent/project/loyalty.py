"""Loyalty tiers.

The thresholds are a product decision, pinned by ``tests/test_loyalty.py``
including the boundary: a customer who lands exactly on a threshold is in that
tier. Changing a number here changes who gets what, so it is the repository's
own suite that guards them rather than a comment asking nicely.
"""

from __future__ import annotations

DEFAULT_TIER = "basic"

# Highest first — ``tier_for`` returns the first one the customer reaches.
TIERS: tuple[tuple[float, str], ...] = (
    (1000.0, "gold"),
    (400.0, "silver"),
)


def tier_for(lifetime_spend: float) -> str:
    """The tier a customer's lifetime spend earns.

    The comparison is inclusive: spending exactly 1000.00 is gold.
    """
    for threshold, name in TIERS:
        if lifetime_spend >= threshold:
            return name
    return DEFAULT_TIER


def next_tier(lifetime_spend: float) -> tuple[str, float] | None:
    """``(tier, amount still needed)``, or None once the top tier is reached."""
    for threshold, name in reversed(TIERS):
        if lifetime_spend < threshold:
            return name, round(threshold - lifetime_spend, 2)
    return None
