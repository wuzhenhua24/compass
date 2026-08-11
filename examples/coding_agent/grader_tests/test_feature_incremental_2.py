"""Hidden acceptance tests — ``feature_incremental_2``.

One function in one module, with the whole signature given in the prompt. The
work is not writing it; it is honouring three contracts the module already has
and the requirement only names:

  - stock maps are immutable (``reserve`` returns a new map and leaves its
    input alone — ``tests/test_orders.py`` pins that for the existing code)
  - a rejected operation changes nothing, and with two maps in play that now
    means *neither* of them
  - the existing error vocabulary is reused rather than reinvented:
    ``ValueError`` / ``OutOfStock`` / ``UnknownSKU``, not a bare ``Exception``
    or a ``False`` return

Deliberately not asserted: the order the three checks run in. Nothing in the
requirement fixes it, so no probe below supplies two bad arguments at once —
pinning an order the prompt never stated is how a hidden test starts measuring
the author's habits.
"""

import pytest
from catalog import UnknownSKU
from inventory import OutOfStock, transfer

WAREHOUSE = {"mug": 10, "tee": 4}
OUTLET = {"mug": 2}


def test_stock_moves_from_one_warehouse_to_the_other():
    source, destination = transfer(WAREHOUSE, OUTLET, "mug", 3)
    assert source == {"mug": 7, "tee": 4}
    assert destination == {"mug": 5}


def test_the_destination_gains_a_sku_it_did_not_stock():
    source, destination = transfer(WAREHOUSE, OUTLET, "tee", 4)
    assert source == {"mug": 10, "tee": 0}
    assert destination == {"mug": 2, "tee": 4}


def test_neither_input_map_is_modified():
    before_from, before_to = dict(WAREHOUSE), dict(OUTLET)
    transfer(WAREHOUSE, OUTLET, "mug", 3)
    assert WAREHOUSE == before_from
    assert OUTLET == before_to


def test_moving_everything_on_hand_is_allowed():
    source, destination = transfer(WAREHOUSE, OUTLET, "mug", 10)
    assert source["mug"] == 0
    assert destination["mug"] == 12


def test_moving_more_than_is_on_hand_is_refused():
    with pytest.raises(OutOfStock) as excinfo:
        transfer(WAREHOUSE, OUTLET, "tee", 5)
    # The existing exception carries what support needs; a bare raise loses it.
    assert excinfo.value.sku == "tee"
    assert excinfo.value.wanted == 5
    assert excinfo.value.available == 4


def test_a_sku_the_source_does_not_stock_at_all_is_refused():
    with pytest.raises(OutOfStock) as excinfo:
        transfer(OUTLET, WAREHOUSE, "tote", 1)
    assert excinfo.value.available == 0


def test_a_non_positive_quantity_is_refused():
    for bad in (0, -1):
        with pytest.raises(ValueError):
            transfer(WAREHOUSE, OUTLET, "mug", bad)


def test_an_archived_product_cannot_be_moved():
    with pytest.raises(UnknownSKU):
        transfer({"archived-2019": 5}, OUTLET, "archived-2019", 1)


def test_a_refused_transfer_leaves_both_warehouses_alone():
    before_from, before_to = dict(WAREHOUSE), dict(OUTLET)
    refusals = (
        (lambda: transfer(WAREHOUSE, OUTLET, "tee", 99), OutOfStock),
        (lambda: transfer(WAREHOUSE, OUTLET, "mug", 0), ValueError),
        # WAREHOUSE stocks no archived product either, so both refusals are
        # correct here — which of them wins depends on the order the checks
        # run in, and the requirement does not fix that.
        (
            lambda: transfer(WAREHOUSE, OUTLET, "archived-2019", 1),
            (UnknownSKU, OutOfStock),
        ),
    )
    for call, expected in refusals:
        with pytest.raises(expected):
            call()
        assert WAREHOUSE == before_from
        assert OUTLET == before_to


def test_the_existing_reservation_path_still_works():
    from inventory import reserve

    assert reserve(WAREHOUSE, "mug", 3) == {"mug": 7, "tee": 4}
    assert WAREHOUSE == {"mug": 10, "tee": 4}
