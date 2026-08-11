"""Customer records.

A flat table stood in for the customer service years ago and never moved. It
holds identity only — anything derived (lifetime spend, loyalty tier) is
computed from the order history rather than stored here, so there is one place
a stale number can come from instead of two.
"""

from __future__ import annotations


class UnknownCustomer(KeyError):
    """Raised for a customer id that is not on file."""


CUSTOMERS: dict[str, dict[str, str]] = {
    "c_1001": {"name": "Ada Okonkwo", "region": "us", "joined": "2021-03-04"},
    "c_1002": {"name": "Bo Lindqvist", "region": "eu", "joined": "2022-07-19"},
    "c_1003": {"name": "Chen Wei", "region": "apac", "joined": "2020-11-30"},
    "c_1004": {"name": "Dara Ferreira", "region": "us", "joined": "2019-05-12"},
    "c_1005": {"name": "Emil Novak", "region": "eu", "joined": "2023-01-08"},
}


def customer(customer_id: str) -> dict[str, str]:
    """The record for ``customer_id``.

    Raises:
        UnknownCustomer: no such customer.
    """
    try:
        return CUSTOMERS[customer_id]
    except KeyError:
        raise UnknownCustomer(customer_id) from None


def region(customer_id: str) -> str:
    return customer(customer_id)["region"]
