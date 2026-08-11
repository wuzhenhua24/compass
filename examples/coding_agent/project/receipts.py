"""Receipts — the outermost layer, and the one support screenshots.

Nothing is computed here that is not already computed somewhere else: this
module looks things up and lays them out. That is deliberate, and it is why a
number being wrong on a receipt is almost never a bug in this file.
"""

from __future__ import annotations

from catalog import display_name
from customers import customer
from history import order_count, orders_for, total_spent
from loyalty import next_tier, tier_for
from pricing import order_total
from promo import promo_percent


def build_receipt(
    customer_id: str,
    lines: list[dict],
    *,
    promo_code: str = "",
    promo_uses: int = 0,
) -> dict:
    """Everything that goes on a customer's receipt.

    The loyalty tier shown here is derived at print time from the order
    history, not read from the customer record — a stored tier goes stale the
    moment an order is cancelled.
    """
    record = customer(customer_id)
    past = orders_for(customer_id)
    spend = total_spent(past)

    subtotal = order_total(lines)
    percent = promo_percent(promo_code, subtotal, promo_uses)
    discount = round(subtotal * percent / 100, 2)

    return {
        "customer": record["name"],
        "region": record["region"],
        "tier": tier_for(spend),
        "lifetime_spend": spend,
        "orders_placed": order_count(past),
        "items": [display_name(line["sku"]) for line in lines],
        "subtotal": subtotal,
        "discount": discount,
        "total": round(subtotal - discount, 2),
    }


def tier_progress(customer_id: str) -> str:
    """The 'spend X more to reach Y' line at the bottom of a receipt."""
    spend = total_spent(orders_for(customer_id))
    upcoming = next_tier(spend)
    if upcoming is None:
        return "You are at our highest tier."
    name, remaining = upcoming
    return f"Spend {remaining:.2f} more to reach {name}."
