"""Hidden acceptance tests — ``interaction_promo_then_tier_rounding``.

Two discount rules that are each trivial on their own. All the difficulty is in
how they compose, and there are five distinct ways to compose them wrongly:

    tier applied to the subtotal instead of the post-promo amount
    the two percentages summed into one (10% + 15% != 10% then 15%)
    the tier applied first and the promo to the remainder
    the tier folded into the existing ``discount`` field
    the tier applied after shipping was added in

Every one of those is a plausible reading of a requirement written in a hurry,
and every one lands on a different number here. That is the point: this file is
eight independent assertions rather than one, so a run that gets the structure
half right scores 5/8 instead of 0 — and a per-case score in (0, 1) is worth
several extra cases when you are trying to tell two models apart.

Deliberately *not* an axis: the rounding rule. Every figure below comes out the
same whether you use ``round()`` or ``Decimal`` half-up, checked numerically.
The case measures composition, and mixing a second failure mode into the same
assertions would only blur what a failure means.
"""

from inventory import OutOfStock
from orders import place_order

# 200.00 — above the free-shipping threshold, so shipping stays out of the way.
L200 = [{"sku": "mug", "unit_price": 40.0, "quantity": 5}]
S200 = {"mug": 10}

# 40.00 — below the threshold, so shipping is charged and can be discounted by
# an implementation that adds it in too early.
L40 = [{"sku": "mug", "unit_price": 8.0, "quantity": 5}]
S40 = {"mug": 10}

# 133.45 — deliberately not a round number, and below BULK20's 200.00 minimum.
L133 = [
    {"sku": "mug", "unit_price": 33.25, "quantity": 4},
    {"sku": "tee", "unit_price": 0.45, "quantity": 1},
]
S133 = {"mug": 10, "tee": 4}

# 50.00 — exactly the free-shipping threshold, the same order the repository's
# own suite uses to pin ``subtotal``/``discount``.
L50 = [
    {"sku": "mug", "unit_price": 12.0, "quantity": 2},
    {"sku": "tee", "unit_price": 26.0, "quantity": 1},
]
S50 = {"mug": 10, "tee": 4}


def test_the_tier_comes_off_the_post_promo_amount():
    # 200.00 - 10% = 180.00, and gold's 15% is 27.00 of *that*, not 30.00.
    breakdown = place_order(L200, S200, promo_code="WELCOME10", tier="gold")
    assert breakdown["discount"] == 20.0
    assert breakdown["tier_discount"] == 27.0
    assert breakdown["total"] == 153.0


def test_the_promo_is_applied_first():
    # Order matters even where the total does not: taking silver's 10% first
    # would leave the promo only 120.11 to work on, reporting discount 12.01.
    breakdown = place_order(L133, S133, promo_code="WELCOME10", tier="silver")
    assert breakdown["subtotal"] == 133.45
    assert breakdown["discount"] == 13.35
    assert breakdown["tier_discount"] == 12.01
    assert breakdown["total"] == 108.09


def test_discount_still_means_the_promo_alone():
    # The requirement did not redefine ``discount``. Folding the tier into it
    # is the one wrong turn the repository's own suite does *not* catch — that
    # test calls ``place_order`` without a tier, where the two agree.
    breakdown = place_order(L50, S50, promo_code="WELCOME10", tier="gold")
    assert breakdown["discount"] == 5.0
    assert breakdown["tier_discount"] == 6.75


def test_a_tier_applies_when_the_promo_code_does_not():
    # BULK20 needs a 200.00 subtotal, so it contributes nothing here. The two
    # rules have to be independent of each other.
    breakdown = place_order(L133, S133, promo_code="BULK20", tier="gold")
    assert breakdown["discount"] == 0.0
    assert breakdown["tier_discount"] == 20.02
    assert breakdown["total"] == 113.43


def test_shipping_is_not_discounted():
    # 40.00 subtotal, gold: 6.00 off, then 4.99 shipping on top. Adding the fee
    # in before the tier is computed would take 6.75 and land on 38.24.
    breakdown = place_order(L40, S40, tier="gold")
    assert breakdown["shipping"] == 4.99
    assert breakdown["tier_discount"] == 6.0
    assert breakdown["total"] == 38.99


def test_an_order_with_no_tier_keeps_todays_numbers():
    breakdown = place_order(L50, S50, promo_code="WELCOME10")
    assert breakdown["subtotal"] == 50.0
    assert breakdown["discount"] == 5.0
    assert breakdown["tier_discount"] == 0.0
    assert breakdown["shipping"] == 0.0
    assert breakdown["total"] == 45.0


def test_an_unknown_tier_pays_full_price():
    breakdown = place_order(L200, S200, promo_code="WELCOME10", tier="bronze")
    assert breakdown["tier_discount"] == 0.0
    assert breakdown["total"] == 180.0


def test_pricing_changes_did_not_disturb_the_stock_contract():
    # Pinned here as well as in the repository's suite because this is the
    # invariant a rewrite of ``place_order`` is most likely to drop on the way
    # past. Asserted with a tier set so the check cannot pass on the base
    # project — a hidden assertion that is already green is a free point.
    before = dict(S50)
    breakdown = place_order(L50, S50, promo_code="WELCOME10", tier="gold")
    assert breakdown["tier_discount"] == 6.75
    assert breakdown["stock"] == {"mug": 8, "tee": 3}
    assert S50 == before

    try:
        place_order(
            [{"sku": "tee", "unit_price": 26.0, "quantity": 99}], S50, tier="gold"
        )
    except OutOfStock:
        pass
    else:
        raise AssertionError("an unfillable order should raise OutOfStock")
    assert S50 == before
