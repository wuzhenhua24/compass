"""``compass list`` — the registered graders and adapters."""

from compass.adapters import list_adapters
from compass.cli.app import cli, console


@cli.command("list")
def list_registered() -> None:
    """List available graders and adapters.

    Named ``list_registered`` rather than ``list``: a module-level function
    called ``list`` shadows the builtin for every other command in this file,
    which silently turned ``list(...)`` calls into invocations of this command.
    """
    from compass.graders import GraderType, list_graders
    from compass.graders.domains import DOMAINS, domain_of

    console.print("\n[bold]Available Graders:[/bold]")

    # The split that matters is not Code/Model/Human — it is whose job the
    # grader is. Process checks are the framework's and work on any agent;
    # correctness checks belong to a domain, and a user with a domain of their
    # own writes those. Showing them mixed hid that distinction at exactly the
    # place someone goes looking for a grader to use.
    # list_graders() is what loads the domain packages, so it comes first.
    everything = {gtype: list_graders(gtype)
                  for gtype in (GraderType.CODE, GraderType.MODEL, GraderType.HUMAN)}
    by_domain = {n: d for names in everything.values() for n in names
                 if (d := domain_of(n)) is not None}

    console.print("  [cyan]Framework[/cyan] [dim]— process & reliability, any agent[/dim]")
    for label, gtype in (("Code", GraderType.CODE), ("Model", GraderType.MODEL),
                         ("Human", GraderType.HUMAN)):
        names = [n for n in everything[gtype] if n not in by_domain]
        if names:
            console.print(f"    [dim]{label}:[/dim] " + ", ".join(sorted(names)))

    for domain in DOMAINS:
        names = sorted(n for n, d in by_domain.items() if d == domain)
        if names:
            console.print(f"  [cyan]Domain: {domain}[/cyan] [dim]— correctness[/dim]")
            console.print("    " + ", ".join(names))

    console.print("\n[bold]Available Adapters:[/bold]")
    console.print("  " + ", ".join(list_adapters()))
