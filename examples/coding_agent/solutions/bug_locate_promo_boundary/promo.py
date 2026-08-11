"""Promo codes.

Rules live in one table so the eval's tasks can be about *behaviour* rather
than about finding where a value is hidden.
"""

from __future__ import annotations

PROMOS: dict[str, dict[str, float]] = {
    # percent off, the order subtotal it needs, and how many times one
    # customer may use it
    "WELCOME10": {"percent": 10.0, "min_total": 0.0, "max_uses": 1},
    "BULK20": {"percent": 20.0, "min_total": 200.0, "max_uses": 5},
    "LOYAL5": {"percent": 5.0, "min_total": 0.0, "max_uses": 12},
}


def promo_percent(code: str, subtotal: float, uses_so_far: int = 0) -> float:
    """Percent off for ``code``, or 0.0 when it does not apply.

    An order landing exactly on the threshold qualifies.
    """
    promo = PROMOS.get(code)
    if promo is None:
        return 0.0
    if subtotal < promo["min_total"]:
        return 0.0
    if uses_so_far >= promo["max_uses"]:
        return 0.0
    return promo["percent"]
