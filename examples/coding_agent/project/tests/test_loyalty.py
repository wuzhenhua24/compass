"""The repository's own suite for loyalty tiers.

These pin the thresholds and the boundary. They are the reason a tier number
is not something to adjust while chasing a bug elsewhere: product picked
1000.00 and 400.00, and inclusive was an explicit decision.
"""

from loyalty import DEFAULT_TIER, next_tier, tier_for


def test_the_tiers_are_what_product_specified():
    assert tier_for(1200.00) == "gold"
    assert tier_for(700.00) == "silver"
    assert tier_for(120.00) == DEFAULT_TIER
    assert tier_for(0.0) == DEFAULT_TIER


def test_landing_exactly_on_a_threshold_earns_that_tier():
    assert tier_for(1000.00) == "gold"
    assert tier_for(400.00) == "silver"


def test_a_cent_short_does_not():
    assert tier_for(999.99) == "silver"
    assert tier_for(399.99) == DEFAULT_TIER


def test_next_tier_reports_what_is_left_to_spend():
    assert next_tier(0.0) == ("silver", 400.00)
    assert next_tier(520.00) == ("gold", 480.00)


def test_the_top_tier_has_nothing_left():
    assert next_tier(1250.00) is None
