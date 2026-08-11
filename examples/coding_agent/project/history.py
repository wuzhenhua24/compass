"""Order history, and what it adds up to.

Orders are append-only. A cancelled order stays in the list with
``status: "cancelled"`` rather than being removed, because support needs to see
that it happened — which means every reader has to skip them itself.
"""

from __future__ import annotations

ORDERS: dict[str, list[dict]] = {
    "c_1001": [
        {"id": "o_5001", "status": "completed", "total": 480.00},
        {"id": "o_5002", "status": "completed", "total": 310.00},
        {"id": "o_5003", "status": "completed", "total": 460.00},
    ],
    "c_1002": [
        {"id": "o_5010", "status": "completed", "total": 220.00},
        {"id": "o_5011", "status": "completed", "total": 300.00},
    ],
    "c_1003": [
        {"id": "o_5020", "status": "completed", "total": 450.00},
        {"id": "o_5021", "status": "completed", "total": 60.00},
        {"id": "o_5022", "status": "cancelled", "total": 190.00},
    ],
    "c_1004": [
        {"id": "o_5030", "status": "completed", "total": 420.00},
        {"id": "o_5031", "status": "cancelled", "total": 180.00},
        {"id": "o_5032", "status": "completed", "total": 600.00},
        {"id": "o_5033", "status": "completed", "total": 442.50},
    ],
    "c_1005": [
        {"id": "o_5040", "status": "cancelled", "total": 250.00},
        {"id": "o_5041", "status": "completed", "total": 300.00},
        {"id": "o_5042", "status": "completed", "total": 400.00},
    ],
}


def orders_for(customer_id: str) -> list[dict]:
    """Every order on file for a customer, oldest first.

    Unknown customers get an empty history rather than an error: a customer who
    has never ordered and a customer who does not exist look the same from
    here, and the caller that cares is ``customers.customer``.
    """
    return list(ORDERS.get(customer_id, []))


def completed(orders: list[dict]) -> list[dict]:
    """Just the orders that went through."""
    return [o for o in orders if o["status"] == "completed"]


def total_spent(orders: list[dict]) -> float:
    """Lifetime spend: the total of every order that was not cancelled."""
    total = 0.0
    for order in orders:
        if order["status"] == "cancelled":
            break
        total += order["total"]
    return round(total, 2)


def order_count(orders: list[dict]) -> int:
    """How many orders actually went through."""
    return len(completed(orders))
