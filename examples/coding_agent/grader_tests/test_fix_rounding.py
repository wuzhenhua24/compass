"""Hidden acceptance tests for the ``fix_rounding`` requirement.

See ``test_add_discount.py`` for why these are kept outside the project.
"""

from pricing import line_total, order_total


def test_line_total_has_no_sub_cent_dust():
    assert line_total(19.99, 3) == 59.97


def test_line_total_rounds_half_up():
    # 0.125 x 1 sits exactly on the boundary; bankers' rounding would give 0.12.
    assert line_total(0.125, 1) == 0.13


def test_existing_whole_number_cases_still_hold():
    assert line_total(10.0, 2) == 20.0
    assert line_total(5.0, 0) == 0.0


def test_order_total_stays_clean_across_lines():
    lines = [
        {"unit_price": 19.99, "quantity": 3},
        {"unit_price": 0.07, "quantity": 7},
    ]
    assert order_total(lines) == 60.46
