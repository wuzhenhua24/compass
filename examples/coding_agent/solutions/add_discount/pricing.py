"""Order pricing for the storefront.

The starting state of the fixture codebase: enough real structure for a coding
agent to have to read before it edits, and two deliberate gaps that the
evaluation's business requirements ask it to close.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def line_total(unit_price: float, quantity: int) -> float:
    """Total for a single order line.

    Known defect (see the ``fix_rounding`` requirement): float multiplication
    leaves sub-cent dust, so 19.99 x 3 comes out as 59.969999999999999.
    """
    return unit_price * quantity


def order_total(lines: list[dict]) -> float:
    """Sum every line in an order."""
    return sum(line_total(line["unit_price"], line["quantity"]) for line in lines)


TIER_DISCOUNT = {"gold": 0.15, "silver": 0.10}


def apply_discount(total: float, tier: str) -> float:
    """Apply the loyalty-tier discount to an order total."""
    if total < 0:
        raise ValueError(f"order total cannot be negative: {total}")
    rate = TIER_DISCOUNT.get(tier, 0.0)
    net = Decimal(str(total)) * (Decimal("1") - Decimal(str(rate)))
    return float(net.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def shipping_fee(total: float) -> float:
    """Flat shipping, waived above the free-shipping threshold."""
    from config import FREE_SHIPPING_THRESHOLD, SHIPPING_FEE

    return 0.0 if total >= FREE_SHIPPING_THRESHOLD else SHIPPING_FEE
