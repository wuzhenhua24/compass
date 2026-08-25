"""``compass list`` — the registered graders and adapters."""

from typing import TYPE_CHECKING

from compass.adapters import list_adapters
from compass.cli.app import cli, console

if TYPE_CHECKING:
    pass


@cli.command("list")
def list_registered() -> None:
    """List available graders and adapters.

    Named ``list_registered`` rather than ``list``: a module-level function
    called ``list`` shadows the builtin for every other command in this file,
    which silently turned ``list(...)`` calls into invocations of this command.
    """
    from compass.graders import GraderType, list_graders

    console.print("\n[bold]Available Graders:[/bold]")

    # Group by type
    code_graders = list_graders(GraderType.CODE)
    model_graders = list_graders(GraderType.MODEL)
    human_graders = list_graders(GraderType.HUMAN)

    if code_graders:
        console.print("  [cyan]Code graders:[/cyan]")
        for name in code_graders:
            console.print(f"    - {name}")

    if model_graders:
        console.print("  [cyan]Model graders:[/cyan]")
        for name in model_graders:
            console.print(f"    - {name}")

    if human_graders:
        console.print("  [cyan]Human graders:[/cyan]")
        for name in human_graders:
            console.print(f"    - {name}")

    console.print("\n[bold]Available Adapters:[/bold]")
    for name in list_adapters():
        console.print(f"  - {name}")
