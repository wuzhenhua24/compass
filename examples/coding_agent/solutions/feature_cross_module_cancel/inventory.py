"""Stock reservation.

Stock maps are treated as immutable: every operation returns a new map and
leaves its input alone. Callers rely on that — see
``test_reserve_leaves_the_original_stock_alone`` — because a half-applied
reservation is much harder to recover from than a rejected one.
"""

from __future__ import annotations


class OutOfStock(Exception):
    """Raised when a reservation asks for more than is on hand."""

    def __init__(self, sku: str, wanted: int, available: int) -> None:
        super().__init__(
            f"{sku}: wanted {wanted}, only {available} available"
        )
        self.sku = sku
        self.wanted = wanted
        self.available = available


def reserve(stock: dict[str, int], sku: str, quantity: int) -> dict[str, int]:
    """Reserve ``quantity`` of ``sku``; return the updated stock map.

    Raises:
        ValueError: quantity is not positive.
        OutOfStock: not enough on hand.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    available = stock.get(sku, 0)
    if quantity > available:
        raise OutOfStock(sku, quantity, available)
    return {**stock, sku: available - quantity}


def release(stock: dict[str, int], sku: str, quantity: int) -> dict[str, int]:
    """Give ``quantity`` of ``sku`` back; return the updated stock map.

    The inverse of :func:`reserve`, and immutable in the same way.

    Raises:
        ValueError: quantity is not positive.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    return {**stock, sku: stock.get(sku, 0) + quantity}
