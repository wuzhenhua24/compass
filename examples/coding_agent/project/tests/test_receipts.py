"""The repository's own suite for receipts.

Built against customers whose order history is straightforward, which is how
the reporting migration was signed off.
"""

import pytest
from customers import UnknownCustomer
from receipts import build_receipt, tier_progress

LINES = [
    {"sku": "mug", "unit_price": 12.0, "quantity": 2},
    {"sku": "tee", "unit_price": 26.0, "quantity": 1},
]


def test_a_receipt_names_the_customer_and_the_items():
    receipt = build_receipt("c_1001", LINES)
    assert receipt["customer"] == "Ada Okonkwo"
    assert receipt["region"] == "us"
    assert receipt["items"] == ["Enamel mug", "Cotton tee"]


def test_the_tier_is_derived_from_the_order_history():
    assert build_receipt("c_1001", LINES)["tier"] == "gold"
    assert build_receipt("c_1001", LINES)["lifetime_spend"] == 1250.00
    assert build_receipt("c_1002", LINES)["tier"] == "silver"


def test_the_money_on_the_receipt_adds_up():
    receipt = build_receipt("c_1001", LINES, promo_code="WELCOME10")
    assert receipt["subtotal"] == 50.0
    assert receipt["discount"] == 5.0
    assert receipt["total"] == 45.0


def test_orders_placed_counts_completed_orders_only():
    assert build_receipt("c_1003", LINES)["orders_placed"] == 2


def test_an_unknown_customer_cannot_be_billed():
    with pytest.raises(UnknownCustomer):
        build_receipt("c_9999", LINES)


def test_tier_progress_reads_off_the_same_number():
    assert tier_progress("c_1002") == "Spend 480.00 more to reach gold."
    assert tier_progress("c_1001") == "You are at our highest tier."
