"""The repository's own suite for order history.

Like the rest of ``tests/``, this covers what already works rather than what is
under evaluation. Note which histories it reaches for: customers whose orders
all went through, and one whose cancellation is the most recent thing they did.
Those are the shapes the reporting migration was tested against.
"""

from history import completed, order_count, orders_for, total_spent


def test_orders_come_back_oldest_first():
    ids = [o["id"] for o in orders_for("c_1001")]
    assert ids == ["o_5001", "o_5002", "o_5003"]


def test_an_unknown_customer_has_no_history():
    assert orders_for("c_9999") == []


def test_the_caller_cannot_mutate_the_stored_history():
    orders_for("c_1001").append({"id": "nope", "status": "completed", "total": 1.0})
    assert len(orders_for("c_1001")) == 3


def test_completed_filters_out_cancellations():
    assert [o["id"] for o in completed(orders_for("c_1003"))] == ["o_5020", "o_5021"]


def test_lifetime_spend_of_a_clean_history():
    assert total_spent(orders_for("c_1001")) == 1250.00
    assert total_spent(orders_for("c_1002")) == 520.00


def test_a_cancelled_order_does_not_count_towards_lifetime_spend():
    # c_1003 cancelled their most recent order: 450.00 + 60.00 counts, 190.00
    # does not.
    assert total_spent(orders_for("c_1003")) == 510.00


def test_an_empty_history_spends_nothing():
    assert total_spent([]) == 0.0


def test_order_count_ignores_cancellations():
    assert order_count(orders_for("c_1001")) == 3
    assert order_count(orders_for("c_1003")) == 2
