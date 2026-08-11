"""The product catalogue.

Lookup only — no pricing logic lives here. ``pricing.py`` takes the unit price
from the order line rather than from this table, because a line records what
the customer was charged at the time and the catalogue records what a SKU costs
today. Conflating the two is how historical orders start changing value.
"""

from __future__ import annotations


class UnknownSKU(KeyError):
    """Raised for a SKU that is not in the catalogue."""


PRODUCTS: dict[str, dict[str, object]] = {
    "mug": {"name": "Enamel mug", "category": "drinkware", "weight_g": 320},
    "tee": {"name": "Cotton tee", "category": "apparel", "weight_g": 180},
    "cap": {"name": "Six-panel cap", "category": "apparel", "weight_g": 95},
    "tote": {"name": "Canvas tote", "category": "bags", "weight_g": 240},
    "poster": {"name": "A2 poster", "category": "print", "weight_g": 60},
    "sticker": {"name": "Vinyl sticker", "category": "print", "weight_g": 5},
}


def product(sku: str) -> dict[str, object]:
    """The catalogue entry for ``sku``.

    Raises:
        UnknownSKU: no such product.
    """
    try:
        return PRODUCTS[sku]
    except KeyError:
        raise UnknownSKU(sku) from None


def display_name(sku: str) -> str:
    """Human-readable name, falling back to the SKU itself.

    Deliberately forgiving where ``product()`` is not: a receipt for an
    archived SKU should still print, even though pricing one should fail.
    """
    entry = PRODUCTS.get(sku)
    return str(entry["name"]) if entry else sku


def category(sku: str) -> str:
    return str(product(sku)["category"])


def shipping_weight(lines: list[dict]) -> int:
    """Total weight of an order in grams."""
    return sum(
        int(PRODUCTS.get(line["sku"], {}).get("weight_g", 0)) * line["quantity"]
        for line in lines
    )
