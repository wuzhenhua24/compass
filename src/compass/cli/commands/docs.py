"""``compass docs`` — print Compass's own documentation."""

import sys
from typing import TYPE_CHECKING

import click
from rich.table import Table

from compass.cli.app import cli, console

if TYPE_CHECKING:
    pass


@cli.command()
@click.argument("topic", required=False)
@click.option("--all", "show_all", is_flag=True, help="Print every topic, concatenated")
def docs(topic: str | None, show_all: bool) -> None:
    """Print Compass's documentation.

    So the terminal is enough — for a person, or for an agent driving the CLI.
    Output is raw markdown, so it pipes.

    \b
      compass docs                 # list the topics
      compass docs graders         # print one
      compass docs --all > all.md  # everything
    """
    from compass.docs_index import TOPICS, get_topic, read_topic, topic_names

    if show_all:
        chunks = []
        for entry in TOPICS:
            text = read_topic(entry)
            if text is not None:
                chunks.append(text.rstrip())
        if not chunks:
            console.print("[red]No documentation found in this installation[/red]")
            sys.exit(1)
        click.echo("\n\n---\n\n".join(chunks))
        return

    if topic is None:
        table = Table(title="compass docs <topic>")
        table.add_column("Topic", style="bold cyan")
        table.add_column("Contents")
        for entry in TOPICS:
            missing = read_topic(entry) is None
            table.add_row(
                entry.name + (" [dim](missing)[/dim]" if missing else ""),
                entry.summary,
            )
        console.print(table)
        console.print("[dim]raw markdown — `compass docs graders | less`[/dim]")
        return

    selected = get_topic(topic)
    if selected is None:
        console.print(
            f"[red]No such topic: {topic}[/red]\n"
            f"Available: {', '.join(topic_names())}"
        )
        sys.exit(1)

    text = read_topic(selected)
    if text is None:
        console.print(
            f"[red]{selected.source} is not available in this installation[/red]"
        )
        sys.exit(1)
    click.echo(text)
