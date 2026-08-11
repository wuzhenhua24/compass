"""Hidden acceptance tests — ``bug_locate_promo_boundary``.

The bug is reported against the order total; the defect is a boundary check in
``promo.py``. These assert the fix at both levels, so an agent that patches the
symptom in ``orders.py`` instead of the cause still fails.
"""

from orders import place_order
from promo import promo_percent

# 200.00 exactly — the threshold BULK20 declares.
LINES_AT_THRESHOLD = [{"sku": "mug", "unit_price": 50.0, "quantity": 4}]
STOCK = {"mug": 10}


def test_the_threshold_itself_qualifies():
    assert promo_percent("BULK20", 200.0) == 20.0


def test_just_under_the_threshold_still_does_not():
    assert promo_percent("BULK20", 199.99) == 0.0


def test_comfortably_over_is_unchanged():
    assert promo_percent("BULK20", 300.0) == 20.0


def test_the_order_total_reflects_it():
    breakdown = place_order(LINES_AT_THRESHOLD, STOCK, promo_code="BULK20")
    assert breakdown["discount"] == 40.0


def test_a_zero_minimum_code_still_applies_at_zero():
    # WELCOME10 declares min_total 0.0, so a 0.00 subtotal must qualify —
    # the same off-by-one, at the other end of the table.
    assert promo_percent("WELCOME10", 0.0) == 10.0
