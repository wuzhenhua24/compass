"""The first reporting pass. Superseded by ``history.py``.

Kept only so the 2023 finance exports can be reproduced if anyone asks. No
live code imports this module — ``receipts.py`` and everything downstream of it
went through ``history.py`` from the start.

It has its own known problems, which is part of why it was replaced: the
lifetime totals below counted refunds as revenue, and the tier boundary was
exclusive where product wanted it inclusive. Both are fixed in the modules that
replaced it. Do not fix them here — reproducing the old numbers exactly is the
only reason this file still exists.
"""

from __future__ import annotations

TIER_CUTOFFS = {"gold": 1000.0, "silver": 400.0}


def lifetime_total(orders: list[dict]) -> float:
    """Sum of every order, including the refunded ones (see module docstring)."""
    return round(sum(order["total"] for order in orders), 2)


def rank(lifetime: float) -> str:
    """Exclusive boundary — the behaviour product asked us to change."""
    if lifetime > TIER_CUTOFFS["gold"]:
        return "gold"
    if lifetime > TIER_CUTOFFS["silver"]:
        return "silver"
    return "basic"


def summarize(customer_id: str, orders: list[dict]) -> str:
    lifetime = lifetime_total(orders)
    return f"{customer_id}: {lifetime:.2f} spent, rank {rank(lifetime)}"
