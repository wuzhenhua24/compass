"""Sandbox module for isolated code execution environments."""

# Import built-in sandboxes to trigger registration
from compass.sandbox import local  # noqa: F401
from compass.sandbox.base import Sandbox
from compass.sandbox.registry import (
    get_sandbox_class,
    list_sandboxes,
    register_sandbox,
)

__all__ = [
    "Sandbox",
    "register_sandbox",
    "get_sandbox_class",
    "list_sandboxes",
]
