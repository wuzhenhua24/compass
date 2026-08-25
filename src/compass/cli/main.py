"""CLI entry point for Compass.

The command surface lives in :mod:`compass.cli.commands` — one module per
command. This module exists to import them all, which is what registers them,
and to be the stable name ``[project.scripts]`` points at.
"""

from compass.cli import commands as _commands  # noqa: F401  (registers the commands)
from compass.cli.app import cli, console

__all__ = ["cli", "console"]
