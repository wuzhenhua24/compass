"""Order pricing for the storefront.

The starting state of the fixture codebase: enough real structure for a coding
agent to have to read before it edits, and two deliberate gaps that the
evaluation's business requirements ask it to close.
"""

from __future__ import annotations


def line_total(unit_price: float, quantity: int) -> float:
    """Total for a single order line.

    Known defect (see the ``fix_rounding`` requirement): float multiplication
    leaves sub-cent dust, so 19.99 x 3 comes out as 59.969999999999999.
    """
    return unit_price * quantity


def order_total(lines: list[dict]) -> float:
    """Sum every line in an order."""
    return sum(line_total(line["unit_price"], line["quantity"]) for line in lines)


def shipping_fee(total: float) -> float:
    """Flat shipping, waived above the free-shipping threshold."""
    from config import FREE_SHIPPING_THRESHOLD, SHIPPING_FEE

    return 0.0 if total >= FREE_SHIPPING_THRESHOLD else SHIPPING_FEE
