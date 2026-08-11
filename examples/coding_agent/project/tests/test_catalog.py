"""The repository's own suite for the product catalogue."""

import pytest
from catalog import UnknownSKU, category, display_name, product, shipping_weight


def test_a_known_sku_resolves():
    assert product("mug")["name"] == "Enamel mug"
    assert category("tee") == "apparel"


def test_an_unknown_sku_is_an_error_when_pricing_it():
    with pytest.raises(UnknownSKU):
        product("nosuch")


def test_display_name_falls_back_to_the_sku():
    # A receipt for an archived product still has to print.
    assert display_name("mug") == "Enamel mug"
    assert display_name("archived-2019") == "archived-2019"


def test_shipping_weight_sums_the_lines():
    lines = [
        {"sku": "mug", "quantity": 2},   # 640
        {"sku": "sticker", "quantity": 4},  # 20
    ]
    assert shipping_weight(lines) == 660


def test_an_unknown_sku_weighs_nothing_rather_than_exploding():
    assert shipping_weight([{"sku": "archived-2019", "quantity": 3}]) == 0
