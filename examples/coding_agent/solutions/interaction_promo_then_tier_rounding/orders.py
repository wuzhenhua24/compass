"""The order pipeline — where pricing, promos and stock meet.

This is the module most tasks touch, and the one where a change made in the
wrong place shows up as a broken invariant somewhere else.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

from inventory import reserve
from pricing import order_total, shipping_fee
from promo import promo_percent

# Loyalty tiers. Anything not listed here — including "" — pays full price.
TIER_DISCOUNT: dict[str, float] = {"gold": 0.15, "silver": 0.10}


def _to_cents(value: float) -> float:
    """Round to cents, half away from zero (not banker's rounding)."""
    return float(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def place_order(
    lines: list[dict],
    stock: dict[str, int],
    *,
    promo_code: str = "",
    promo_uses: int = 0,
    tier: str = "",
) -> dict:
    """Price an order and reserve its stock.

    Returns a breakdown with, in order of computation:
        ``subtotal``        the sum of the lines, before any discount
        ``discount``        currency taken off by the promo code
        ``tier_discount``   currency taken off by the loyalty tier, applied to
                            the post-promo amount
        ``shipping``        the shipping fee
        ``total``           what the customer pays
        ``stock``           the stock map after reservation

    Nothing is reserved unless the whole order can be priced, and the caller's
    ``stock`` map is never modified in place.
    """
    subtotal = order_total(lines)
    percent = promo_percent(promo_code, subtotal, promo_uses)
    discount = _to_cents(subtotal * percent / 100)

    # The tier comes off what is left after the promo, not off the subtotal.
    after_promo = _to_cents(subtotal - discount)
    tier_discount = _to_cents(after_promo * TIER_DISCOUNT.get(tier, 0.0))

    # Shipping is judged on the subtotal, before either discount comes off.
    shipping = shipping_fee(subtotal)

    remaining = stock
    for line in lines:
        remaining = reserve(remaining, line["sku"], line["quantity"])

    return {
        "subtotal": subtotal,
        "discount": discount,
        "tier_discount": tier_discount,
        "shipping": shipping,
        "total": _to_cents(subtotal - discount - tier_discount + shipping),
        "stock": remaining,
    }
