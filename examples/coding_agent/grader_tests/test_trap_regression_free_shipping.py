"""Hidden acceptance tests — ``trap_regression_free_shipping``.

The requirement is easy; the obvious way to satisfy it is wrong. Shipping must
now be judged on the discounted amount, and the tempting edit is to reassign
``subtotal`` after taking the discount off — which quietly changes what the
breakdown reports. The repository's own suite pins ``subtotal`` as the
pre-discount figure, so the naive fix passes *these* tests and fails that one.

Which is the point: this case only discriminates if you run both.
"""

from orders import place_order

LINES = [
    {"sku": "mug", "unit_price": 12.0, "quantity": 2},
    {"sku": "tee", "unit_price": 26.0, "quantity": 1},
]  # subtotal 50.00 — exactly the free-shipping threshold
STOCK = {"mug": 10, "tee": 4}


def test_a_discount_that_drops_below_the_threshold_reinstates_shipping():
    breakdown = place_order(LINES, STOCK, promo_code="WELCOME10")
    assert breakdown["discount"] == 5.0
    assert breakdown["shipping"] == 4.99


def test_no_discount_still_ships_free_at_the_threshold():
    breakdown = place_order(LINES, STOCK)
    assert breakdown["shipping"] == 0.0


def test_a_discount_that_stays_above_the_threshold_keeps_free_shipping():
    big = [{"sku": "mug", "unit_price": 40.0, "quantity": 5}]  # 200.00
    breakdown = place_order(big, {"mug": 10}, promo_code="WELCOME10")
    assert breakdown["discount"] == 20.0
    assert breakdown["shipping"] == 0.0


def test_the_customer_pays_the_right_amount():
    breakdown = place_order(LINES, STOCK, promo_code="WELCOME10")
    assert breakdown["total"] == 49.99  # 50.00 - 5.00 + 4.99


# Deliberately absent: any assertion about ``subtotal``.
#
# The requirement did not change what ``subtotal`` means, so these tests have
# no business pinning it — that is the repository's own suite's job
# (``test_subtotal_is_before_any_discount``). Keeping the split clean is what
# makes this case discriminate: the tempting fix reassigns ``subtotal`` to the
# discounted figure, sails through everything here, and is caught over there.
#
# Assert only what the requirement asked for. Anything else you pin in a hidden
# test is a regression check hiding in the wrong file.
