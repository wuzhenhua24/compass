"""Hidden acceptance tests for the ``add_discount`` requirement.

These live **outside** the project directory on purpose. They are the scoring
signal, so an agent that could read them could write to the test rather than to
the requirement, and an agent that could edit them could make any
implementation pass. ``integration_test`` copies the agent's workspace into a
sandbox and runs this file against it from here.
"""

import pytest
from pricing import apply_discount


def test_gold_tier_takes_fifteen_percent():
    assert apply_discount(200.0, "gold") == 170.0


def test_silver_tier_takes_ten_percent():
    assert apply_discount(200.0, "silver") == 180.0


def test_unknown_tier_gets_no_discount():
    assert apply_discount(200.0, "bronze") == 200.0
    assert apply_discount(200.0, "") == 200.0


def test_result_is_rounded_to_cents():
    # 15% off 19.99 is 16.9915 — money has two decimal places.
    assert apply_discount(19.99, "gold") == 16.99


def test_negative_total_is_rejected():
    with pytest.raises(ValueError):
        apply_discount(-1.0, "gold")
