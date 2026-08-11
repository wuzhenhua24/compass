"""The order pipeline — where pricing, promos and stock meet.

This is the module most tasks touch, and the one where a change made in the
wrong place shows up as a broken invariant somewhere else.
"""

from __future__ import annotations

from inventory import reserve
from pricing import order_total, shipping_fee
from promo import promo_percent


def place_order(
    lines: list[dict],
    stock: dict[str, int],
    *,
    promo_code: str = "",
    promo_uses: int = 0,
) -> dict:
    """Price an order and reserve its stock.

    Returns a breakdown with, in order of computation:
        ``subtotal``  the sum of the lines, before any discount
        ``discount``  currency taken off by the promo code
        ``shipping``  the shipping fee
        ``total``     what the customer pays
        ``stock``     the stock map after reservation

    Nothing is reserved unless the whole order can be priced, and the caller's
    ``stock`` map is never modified in place.
    """
    subtotal = order_total(lines)
    percent = promo_percent(promo_code, subtotal, promo_uses)
    discount = round(subtotal * percent / 100, 2)

    # Shipping is judged on the subtotal, before the promo comes off.
    shipping = shipping_fee(subtotal)

    remaining = stock
    for line in lines:
        remaining = reserve(remaining, line["sku"], line["quantity"])

    return {
        "subtotal": subtotal,
        "discount": discount,
        "shipping": shipping,
        "total": round(subtotal - discount + shipping, 2),
        "stock": remaining,
    }
