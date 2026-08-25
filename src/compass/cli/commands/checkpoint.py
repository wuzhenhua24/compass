"""``compass checkpoint`` — the resume points under a trace directory."""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.table import Table

from compass.cli.app import cli, console

if TYPE_CHECKING:
    pass


@cli.group()
def checkpoint() -> None:
    """Manage evaluation checkpoints."""
    pass


@checkpoint.command("list")
@click.argument("search_dir", type=click.Path(exists=True), default=".")
@click.option("--no-recursive", is_flag=True,
              help="Only search the given directory, not subdirectories")
def checkpoint_list(search_dir: str, no_recursive: bool) -> None:
    """List checkpoints found under SEARCH_DIR.

    Searches recursively by default. Shows run ID, scenario name, status,
    progress, and timestamps for each checkpoint.

    Examples:

        compass checkpoint list results/

        compass checkpoint list . --no-recursive
    """
    from compass.core.checkpoint import find_checkpoints

    stores = find_checkpoints(Path(search_dir), recursive=not no_recursive)

    if not stores:
        console.print("[yellow]No checkpoints found[/yellow]")
        return

    table = Table(title="Checkpoints", show_lines=True)
    table.add_column("Directory", style="cyan", max_width=50)
    table.add_column("Run ID", style="bold")
    table.add_column("Scenario")
    table.add_column("Status", justify="center")
    table.add_column("Progress", justify="center")
    table.add_column("Created", justify="right")
    table.add_column("Updated", justify="right")

    for store in stores:
        try:
            cp = store.get_checkpoint()
        except Exception as e:
            console.print(f"  [red]Error reading {store.dir}: {e}[/red]")
            continue

        status_str = {
            "completed": "[green]completed[/green]",
            "in_progress": "[yellow]in_progress[/yellow]",
        }.get(cp.status, cp.status)

        # Show relative path when possible
        try:
            display_dir = str(store.dir.relative_to(Path.cwd()))
        except ValueError:
            display_dir = str(store.dir)

        # Shorten ISO timestamps to human-friendly form
        created = cp.created_at[:19].replace("T", " ")
        updated = cp.updated_at[:19].replace("T", " ")

        table.add_row(
            display_dir,
            cp.run_id,
            cp.scenario_name,
            status_str,
            cp.progress,
            created,
            updated,
        )

    console.print(table)


@checkpoint.command("clean")
@click.argument("search_dir", type=click.Path(exists=True), default=".")
@click.option("--all", "clean_all", is_flag=True,
              help="Remove ALL checkpoints, not just completed ones")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option("--no-recursive", is_flag=True,
              help="Only search the given directory, not subdirectories")
def checkpoint_clean(search_dir: str, clean_all: bool, force: bool, no_recursive: bool) -> None:
    """Remove checkpoint files to free disk space.

    By default only removes checkpoints with status "completed".
    Use --all to also remove in-progress checkpoints.

    Examples:

        compass checkpoint clean results/

        compass checkpoint clean results/ --all --force
    """
    from compass.core.checkpoint import find_checkpoints

    stores = find_checkpoints(Path(search_dir), recursive=not no_recursive)

    if not stores:
        console.print("[yellow]No checkpoints found[/yellow]")
        return

    # Filter targets
    targets: list[Any] = []
    for store in stores:
        try:
            cp = store.get_checkpoint()
        except Exception:
            continue
        if clean_all or cp.status == "completed":
            targets.append((store, cp))

    if not targets:
        console.print("[yellow]No checkpoints to clean (use --all to include in-progress)[/yellow]")
        return

    # Show what will be removed
    console.print(f"\nFound {len(targets)} checkpoint(s) to remove:\n")
    for store, cp in targets:
        try:
            display_dir = str(store.dir.relative_to(Path.cwd()))
        except ValueError:
            display_dir = str(store.dir)

        status_str = (
            "[green]completed[/green]" if cp.status == "completed"
            else "[yellow]in_progress[/yellow]"
        )
        console.print(
            f"  {display_dir}  run={cp.run_id}  "
            f"scenario={cp.scenario_name}  status={status_str}  progress={cp.progress}"
        )

    # Confirm
    if not force:
        click.confirm("\nRemove these checkpoints?", abort=True)

    # Clean
    total_deleted = 0
    for store, _cp in targets:
        try:
            deleted = store.clean()
            total_deleted += deleted
        except Exception as e:
            console.print(f"  [red]Error cleaning {store.dir}: {e}[/red]")

    console.print(
        f"\n[green]Cleaned {len(targets)} checkpoint(s) "
        f"({total_deleted} files removed)[/green]"
    )
