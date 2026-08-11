"""The repository's own test suite — the one the agent can see and edit.

Kept deliberately thin: it covers what already works, not the requirements
under evaluation. The acceptance tests for those live outside the repo, in
``grader_tests/``, so the agent cannot read them (or edit them into passing).
"""

from pricing import order_total, shipping_fee


def test_order_total_sums_lines():
    lines = [
        {"unit_price": 10.0, "quantity": 2},
        {"unit_price": 5.0, "quantity": 1},
    ]
    assert order_total(lines) == 25.0


def test_shipping_is_free_above_the_threshold():
    assert shipping_fee(50.0) == 0.0
    assert shipping_fee(120.0) == 0.0


def test_shipping_is_charged_below_the_threshold():
    assert shipping_fee(49.99) == 4.99
