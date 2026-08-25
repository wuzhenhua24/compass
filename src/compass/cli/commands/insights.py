"""``compass insights`` — model conclusions, each checked against the data."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.markup import escape
from rich.panel import Panel

from compass.cli.app import cli, console

if TYPE_CHECKING:
    from compass.report.insights import InsightsReport


@cli.command()
@click.argument("results", type=click.Path(exists=True))
@click.argument("results_b", type=click.Path(exists=True), required=False)
@click.option("--model", "-m", default=None, help="Judge model (default: the built-in one)")
@click.option("--provider", type=click.Choice(["openai", "anthropic"]), default="openai",
              show_default=True, help="Judge provider")
@click.option("--max-claims", default=6, show_default=True, help="Cap on surviving claims")
@click.option("--on", "on", metavar="GRADER",
              help="With two runs: pair on one grader's score, as `compass compare --on`")
@click.option("--show-dropped", is_flag=True,
              help="List the claims the data refused, with the reason for each")
@click.option("--output", "-o", type=click.Path(), help="Save the report as JSON")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON instead of prose")
def insights(
    results: str,
    results_b: str | None,
    model: str | None,
    provider: str,
    max_claims: int,
    on: str | None,
    show_dropped: bool,
    output: str | None,
    as_json: bool,
) -> None:
    """Ask a model what this run means — and keep only what the data supports.

    `compass compare` is precise and silent: it gives you the interval, the
    flips and the resolvable effect size, then stops, because the next sentence
    ("the failures cluster in the retrieval cases") is a reading rather than a
    measurement. This writes that sentence.

    A fluent wrong sentence about an eval is worse than no sentence, so the
    model is not asked to be careful. It writes what it likes, and every claim
    is then checked against the run: a claim citing a case that passed, or a
    case that crashed in the harness, or a case it never actually discusses, is
    thrown away whole. What survives is not "probably true" — it is "consistent
    with what this run measured".

    RESULTS accepts what `analyze` and `compare` accept. Give a second file to
    judge the change between two runs, which also unlocks the `regression` and
    `improvement` claim types.

    Needs a provider key (OPENAI_API_KEY / ANTHROPIC_API_KEY). Without one the
    command reports that the judge did not run — never an empty verdict, which
    would read as "nothing to report".

    It is a judge, so it is not deterministic. Grounding makes the output safe,
    not stable: two passes over the same file can surface different true
    claims. Read it as a reviewer's notes, never as a metric.
    """
    import json as json_mod

    from compass.report.compare import AmbiguousGraderError, compare_paths
    from compass.report.insights import DEFAULT_MODEL, InsightsJudge
    from compass.report.site import collect_run_payload

    def _payload(path: str) -> Any:
        target = Path(path)
        files = [target] if target.is_file() else sorted(target.glob("**/*.json"))
        merged: list[Any] = []
        for jf in files:
            try:
                with open(jf, encoding="utf-8") as f:
                    merged.append(json_mod.load(f))
            except (OSError, ValueError) as e:
                console.print(f"[yellow]Skipping {jf}: {e}[/yellow]")
        if not merged:
            console.print(f"[red]No readable results in {path}[/red]")
            sys.exit(1)
        return merged[0] if len(merged) == 1 else {"results": merged}

    # The judge reads the run it is asked about. With two files that is the
    # candidate, so a claim about a case is checked against the run that case
    # is claimed to have regressed *in*.
    subject = results_b or results
    doc = collect_run_payload(_payload(subject), name=Path(subject).stem)

    comparison = None
    if results_b:
        try:
            comparison = compare_paths(results, results_b, on=on)
        except AmbiguousGraderError as e:
            console.print(f"[red]Ambiguous --on {on!r}:[/red] {e}")
            sys.exit(2)

    judge = InsightsJudge(
        model=model or DEFAULT_MODEL, provider=provider, max_claims=max_claims
    )
    report = asyncio.run(judge.run(doc, comparison))

    if as_json:
        console.print_json(json_mod.dumps(report.to_dict(), ensure_ascii=False))
    else:
        _print_insights(report, show_dropped=show_dropped)

    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json_mod.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Insights saved to:[/green] {output_path}")

    if report.error:
        sys.exit(1)


_SEVERITY_STYLE = {"fail": "red", "warn": "yellow", "pass": "green"}


def _print_insights(report: "InsightsReport", *, show_dropped: bool) -> None:
    """Render the surviving claims, and account for the ones that did not."""
    console.print(Panel(
        f"[bold]Compass Insights[/bold]  |  judge: {report.model}  |  "
        f"{report.cases_considered} case(s) read"
        + (f", {report.cases_elided} elided" if report.cases_elided else "")
    ))

    if report.error:
        # A judge that could not run must never look like one that found
        # nothing — those mean opposite things to whoever is reading.
        console.print(f"[red]The judge did not run:[/red] {report.error}")
        return

    if not report.claims:
        console.print(
            "[dim]No claim survived grounding.[/dim]"
            if report.dropped
            else "[dim]The judge had nothing to report on this run.[/dim]"
        )
    for claim in report.claims:
        style = _SEVERITY_STYLE.get(claim.severity, "white")
        cited = f"  [dim]{', '.join(claim.case_ids)}[/dim]" if claim.case_ids else ""
        console.print(
            f"\n[{style}]●[/{style}] [bold]{escape(claim.title)}[/bold]"
            f"  [dim]({claim.claim_type})[/dim]{cited}"
        )
        console.print(f"  {escape(claim.message)}")

    if report.dropped:
        # The drop rate is the calibration signal: a judge whose claims mostly
        # fail grounding is being handed the wrong data, not writing badly.
        console.print(
            f"\n[dim]{len(report.dropped)} claim(s) the data refused "
            f"({report.grounded_rate:.0%} grounded)."
            + ("" if show_dropped else " Use --show-dropped to see why.")
            + "[/dim]"
        )
        if show_dropped:
            for item in report.dropped:
                label = item.title or item.claim_type or "(untitled)"
                console.print(f"  [dim]✗ {escape(label)} — {escape(item.reason)}[/dim]")
