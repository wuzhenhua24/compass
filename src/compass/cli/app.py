"""The Click group and the shared console.

Command modules import ``cli`` from here and hang themselves off it. Keeping
the group in its own module is what lets them do that: :mod:`compass.cli.main`
imports every command module, so a command importing ``main`` back would close
the loop.
"""

import click
from rich.console import Console

from compass import __version__

#: The one console every command prints through.
console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="compass")
def cli() -> None:
    """Compass - Agent Evaluation Substrate.

    The reusable spine for evaluating AI agents: a standard trajectory model,
    trace ingestion, process graders and reliability metrics. Domain
    correctness graders are yours to write.
    """
    pass
