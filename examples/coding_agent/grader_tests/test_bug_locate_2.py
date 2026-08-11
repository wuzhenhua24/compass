"""Hidden acceptance tests — ``bug_locate_2``.

The symptom is on a receipt. The cause is two hops away, in how lifetime spend
is summed, and there are two wrong places to land on the way:

  - ``loyalty.py``'s threshold table, one hop from the symptom and the obvious
    suspect for "the tier is wrong". It is correct, and the repository's own
    suite pins it — an agent that adjusts a threshold to make this customer
    gold turns ``tests/test_loyalty.py`` red and fails the regression gate.
  - ``legacy/reporting_v1.py``, which has a lifetime-total function with real
    bugs in it and is imported by nothing.

Most of what follows exercises ``total_spent`` with histories built here rather
than the ones in the repository. That is on purpose: deleting the cancelled
order out of ``history.ORDERS`` would make the reported customer's receipt
right and fix nothing, and against the shipped data alone that edit is
indistinguishable from a real fix.
"""

from history import ORDERS, order_count, orders_for, total_spent
from receipts import build_receipt

LINES = [{"sku": "mug", "unit_price": 12.0, "quantity": 1}]


def history(*orders: tuple[str, float]) -> list[dict]:
    """A synthetic order history: ``("completed", 100.0), ("cancelled", 50.0)``."""
    return [
        {"id": f"o_{i}", "status": status, "total": total}
        for i, (status, total) in enumerate(orders)
    ]


def test_a_cancellation_does_not_hide_the_orders_after_it():
    assert total_spent(
        history(("completed", 100.0), ("cancelled", 50.0), ("completed", 200.0))
    ) == 300.0


def test_a_cancellation_first_does_not_hide_the_whole_history():
    assert total_spent(history(("cancelled", 50.0), ("completed", 100.0))) == 100.0


def test_several_cancellations_are_all_skipped():
    assert total_spent(
        history(
            ("cancelled", 10.0),
            ("completed", 100.0),
            ("cancelled", 20.0),
            ("completed", 250.0),
            ("cancelled", 30.0),
        )
    ) == 350.0


def test_cancelled_orders_still_do_not_count():
    # The fix is to stop skipping the rest, not to stop skipping cancellations.
    assert total_spent(history(("completed", 100.0), ("cancelled", 50.0))) == 100.0
    assert total_spent(history(("cancelled", 50.0), ("cancelled", 60.0))) == 0.0
    assert total_spent([]) == 0.0


def test_the_reported_customer_gets_the_tier_they_earned():
    receipt = build_receipt("c_1004", LINES)
    assert receipt["lifetime_spend"] == 1462.50
    assert receipt["tier"] == "gold"


def test_the_other_customer_with_an_early_cancellation_is_fixed_too():
    # Not named in the ticket. A fix aimed at the one reported account leaves
    # this one wrong.
    receipt = build_receipt("c_1005", LINES)
    assert receipt["lifetime_spend"] == 700.00
    assert receipt["tier"] == "silver"


def test_customers_who_were_already_right_are_untouched():
    assert build_receipt("c_1001", LINES)["lifetime_spend"] == 1250.00
    assert build_receipt("c_1002", LINES)["tier"] == "silver"
    assert build_receipt("c_1003", LINES)["lifetime_spend"] == 510.00


def test_the_cancelled_orders_are_still_on_file():
    # Support needs to see that a cancellation happened, so the history is
    # append-only — removing the record would make the number above come out
    # right without repairing anything.
    cancelled = [o for o in ORDERS["c_1004"] if o["status"] == "cancelled"]
    assert [o["id"] for o in cancelled] == ["o_5031"]
    assert order_count(orders_for("c_1004")) == 3
