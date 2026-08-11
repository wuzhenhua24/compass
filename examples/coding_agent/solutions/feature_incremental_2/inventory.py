"""Stock reservation.

Stock maps are treated as immutable: every operation returns a new map and
leaves its input alone. Callers rely on that — see
``test_reserve_leaves_the_original_stock_alone`` — because a half-applied
reservation is much harder to recover from than a rejected one.
"""

from __future__ import annotations

from catalog import product


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


def transfer(
    stock_from: dict[str, int],
    stock_to: dict[str, int],
    sku: str,
    quantity: int,
) -> tuple[dict[str, int], dict[str, int]]:
    """Move ``quantity`` of ``sku`` between two warehouses.

    Returns the two updated stock maps, source first. Neither input is
    modified — same contract as ``reserve``, and doubly load-bearing here:
    a transfer that half-applied would invent or destroy stock rather than
    merely mislaying it.

    Raises:
        ValueError: quantity is not positive.
        UnknownSKU: the product is not in the catalogue.
        OutOfStock: the source warehouse does not have that many.
    """
    if quantity <= 0:
        raise ValueError(f"quantity must be positive, got {quantity}")
    product(sku)  # archived products stay where they are
    available = stock_from.get(sku, 0)
    if quantity > available:
        raise OutOfStock(sku, quantity, available)
    return (
        {**stock_from, sku: available - quantity},
        {**stock_to, sku: stock_to.get(sku, 0) + quantity},
    )
