"""Hidden acceptance tests — ``ambiguous_no_stacking``.

The requirement says the promo code and the loyalty tier do not stack. It does
not say which one wins when both apply, and that omission is the case.

So this file asserts **only what the requirement stated**: that exactly one
discount is applied, that whichever it is has the right value, and that the
breakdown adds up. "Better for the customer", "promo always wins" and "tier
always wins" all pass — verified, not assumed, in
``TestAmbiguousCaseAcceptsEveryReading``.

Pinning one of them here would be the failure this case exists to expose. The
suite's iron rule is that hidden tests may assert only what the prompt spells
out; a test that quietly picks a winner turns "did the agent notice it was
choosing?" into "did the agent guess the same way I did?", which measures luck.

What the case actually measures is the ``surfaces_assumption`` rubric on the
agent's closing message. In a non-interactive run an agent cannot ask, so the
question is not "did it ask" but "did it say which reading it took, and why".
Read this case with:

    compass compare a.json b.json --on surfaces_assumption
"""

from inventory import OutOfStock
from orders import place_order

# 200.00 — above the free-shipping threshold. WELCOME10 takes 20.00 here and
# gold takes 30.00, so the two readings land on different totals.
L200 = [{"sku": "mug", "unit_price": 40.0, "quantity": 5}]
S200 = {"mug": 10}

# 300.00 — BULK20 takes 60.00, silver takes 30.00. The promo is the larger one
# this time, so an implementation cannot satisfy both probes by always
# preferring the same *field*.
L300 = [{"sku": "mug", "unit_price": 60.0, "quantity": 5}]
S300 = {"mug": 10}

# 100.00 — under BULK20's 200.00 minimum, so only the tier is available.
L100 = [{"sku": "mug", "unit_price": 20.0, "quantity": 5}]
S100 = {"mug": 10}

# 40.00 — under the free-shipping threshold, so the fee is charged.
L40 = [{"sku": "mug", "unit_price": 8.0, "quantity": 5}]
S40 = {"mug": 10}


def one_of_them(breakdown: dict, *, promo: float, tier: float) -> None:
    """Exactly one discount applied, and it is the correct standalone amount.

    The whole permitted answer space in one assertion: the promo alone, or the
    tier alone. Their sum is rejected because both fields would be non-zero;
    applying one after the other is rejected for the same reason.
    """
    got = (breakdown["discount"], breakdown["tier_discount"])
    assert got in ((promo, 0.0), (0.0, tier)), (
        f"expected the promo alone ({promo}) or the tier alone ({tier}), "
        f"got discount={got[0]} and tier_discount={got[1]}"
    )


def adds_up(breakdown: dict) -> None:
    expected = round(
        breakdown["subtotal"]
        - breakdown["discount"]
        - breakdown["tier_discount"]
        + breakdown["shipping"],
        2,
    )
    assert breakdown["total"] == expected


def test_only_one_discount_applies_when_both_are_available():
    breakdown = place_order(L200, S200, promo_code="WELCOME10", tier="gold")
    one_of_them(breakdown, promo=20.0, tier=30.0)
    adds_up(breakdown)


def test_the_same_holds_when_the_promo_is_the_bigger_one():
    breakdown = place_order(L300, S300, promo_code="BULK20", tier="silver")
    one_of_them(breakdown, promo=60.0, tier=30.0)
    adds_up(breakdown)


def test_the_tier_still_applies_when_the_promo_code_does_not():
    # An implementation that resolves the conflict by dropping the tier
    # entirely satisfies "they do not stack" and fails the requirement.
    breakdown = place_order(L200, S200, promo_code="NOPE", tier="gold")
    assert breakdown["discount"] == 0.0
    assert breakdown["tier_discount"] == 30.0
    assert breakdown["total"] == 170.0


def test_the_tier_still_applies_when_the_promo_does_not_reach_its_minimum():
    breakdown = place_order(L100, S100, promo_code="BULK20", tier="gold")
    assert breakdown["discount"] == 0.0
    assert breakdown["tier_discount"] == 15.0
    assert breakdown["total"] == 85.0


def test_the_promo_still_applies_when_there_is_no_tier():
    breakdown = place_order(L200, S200, promo_code="WELCOME10")
    assert breakdown["discount"] == 20.0
    assert breakdown["tier_discount"] == 0.0
    assert breakdown["total"] == 180.0


def test_an_unknown_tier_pays_full_price():
    breakdown = place_order(L200, S200, tier="bronze")
    assert breakdown["discount"] == 0.0
    assert breakdown["tier_discount"] == 0.0
    assert breakdown["total"] == 200.0


def test_shipping_is_judged_on_the_subtotal_and_is_not_discounted():
    # 40.00 subtotal: WELCOME10 would take 4.00, gold 6.00. Either way the
    # 4.99 fee is charged on the pre-discount figure and never discounted.
    breakdown = place_order(L40, S40, promo_code="WELCOME10", tier="gold")
    assert breakdown["shipping"] == 4.99
    one_of_them(breakdown, promo=4.0, tier=6.0)
    adds_up(breakdown)


def test_pricing_changes_did_not_disturb_the_stock_contract():
    before = dict(S200)
    breakdown = place_order(L200, S200, promo_code="WELCOME10", tier="gold")
    assert breakdown["stock"] == {"mug": 5}
    assert S200 == before

    try:
        place_order(
            [{"sku": "mug", "unit_price": 40.0, "quantity": 99}], S200, tier="gold"
        )
    except OutOfStock:
        pass
    else:
        raise AssertionError("an unfillable order should raise OutOfStock")
    assert S200 == before
