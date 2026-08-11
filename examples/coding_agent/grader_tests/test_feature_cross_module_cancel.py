"""Hidden acceptance tests — ``feature_cross_module_cancel``.

Two modules have to change together: ``inventory`` gains the inverse of
``reserve``, and ``orders`` gains the pipeline-level cancel that uses it. The
invariant that makes this more than a two-liner is the one ``inventory`` has
declared all along — nothing is mutated in place, and a rejected operation
changes nothing at all.

The requirement text names both signatures. That is deliberate: a test can only
assert an interface the task actually specified, and "guess what I called it"
measures luck.
"""

import pytest
from inventory import release, reserve
from orders import cancel_order, place_order

LINES = [
    {"sku": "mug", "unit_price": 12.0, "quantity": 2},
    {"sku": "tee", "unit_price": 26.0, "quantity": 1},
]
STOCK = {"mug": 10, "tee": 4}


# --- inventory.release -----------------------------------------------------


def test_release_adds_stock_back():
    assert release({"mug": 7}, "mug", 3) == {"mug": 10}


def test_release_knows_about_a_sku_it_has_never_seen():
    assert release({}, "new", 2) == {"new": 2}


def test_release_leaves_the_original_alone():
    stock = {"mug": 7}
    release(stock, "mug", 3)
    assert stock == {"mug": 7}


def test_release_refuses_a_non_positive_quantity():
    with pytest.raises(ValueError):
        release({"mug": 7}, "mug", 0)


def test_release_is_the_inverse_of_reserve():
    after = reserve(STOCK, "mug", 3)
    assert release(after, "mug", 3) == STOCK


# --- orders.cancel_order ---------------------------------------------------


def test_cancel_restores_every_line():
    placed = place_order(LINES, STOCK)
    assert cancel_order(LINES, placed["stock"]) == STOCK


def test_cancel_does_not_touch_the_callers_stock():
    placed = place_order(LINES, STOCK)
    after_order = dict(placed["stock"])
    cancel_order(LINES, placed["stock"])
    assert placed["stock"] == after_order


def test_a_rejected_cancel_restores_nothing():
    """Atomicity: one bad line and the whole cancel is off."""
    bad = [
        {"sku": "mug", "unit_price": 12.0, "quantity": 2},
        {"sku": "tee", "unit_price": 26.0, "quantity": 0},  # invalid
    ]
    stock = {"mug": 8, "tee": 3}
    with pytest.raises(ValueError):
        cancel_order(bad, stock)
    assert stock == {"mug": 8, "tee": 3}
