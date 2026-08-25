"""``compass grade`` — score recorded transcripts, no agent re-run."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel
from rich.table import Table

from compass.cli.app import cli, console
from compass.core.scenario import Scenario

if TYPE_CHECKING:
    from compass.core.regrade import GradeReport


@cli.command()
@click.argument("traces", type=click.Path(exists=True))
@click.option(
    "--scenario", "-s", required=True, type=click.Path(exists=True),
    help="Scenario YAML supplying the graders to apply",
)
@click.option(
    "--name", "-n", "grade_set", default=None,
    help="Grade set name (default: the scenario file's stem)",
)
@click.option("--case", "-c", multiple=True, help="Only grade these case IDs")
@click.option("--regrade", is_flag=True, help="Discard existing grades in this set and redo them")
@click.option("--output", "-o", type=click.Path(),
              help="Save results as JSON (analyze/compare input)")
@click.option("--verbose", "-v", is_flag=True, help="Show per-grader breakdown")
def grade(
    traces: str,
    scenario: str,
    grade_set: str | None,
    case: tuple[str, ...],
    regrade: bool,
    output: str | None,
    verbose: bool,
) -> None:
    """Grade recorded transcripts without re-running the agent.

    Running and grading are separate verbs: `compass test --trace-dir` records
    immutable evidence, `compass grade` scores it. The same traces can be
    scored by several grader sets side by side, and re-scored after a rubric
    edit — no agent re-run, so before/after scores stay comparable instead of
    being confounded by the agent's non-determinism.

    Grades land in TRACES/grades/<name>/ next to a snapshot of the grader spec.
    Each grade records that spec's fingerprint, so an edited grader shows up as
    stale rather than silently standing in for the current one.

    \b
      compass grade ./traces -s scenario.yaml            # score what isn't scored
      compass grade ./traces -s judge.yaml -n judge      # a second grader set
      compass grade ./traces -s scenario.yaml --regrade  # after editing a rubric
    """
    import json

    from compass.core.regrade import grade_traces

    scn = Scenario.from_yaml(scenario)
    name = grade_set or Path(scenario).stem

    console.print(Panel(
        f"[bold]Compass Grade[/bold]\n"
        f"Traces:   {traces}\n"
        f"Scenario: {scenario}\n"
        f"Grade set: {name}" + ("  [yellow](--regrade)[/yellow]" if regrade else "")
    ))

    report = asyncio.run(
        grade_traces(
            scn,
            traces,
            grade_set=name,
            case_ids=list(case) if case else None,
            regrade=regrade,
        )
    )
    result = report.result

    if result.total_cases == 0:
        console.print("[red]No gradable traces matched this scenario[/red]")
        _print_grade_diagnostics(report)
        sys.exit(1)

    table = Table(title=f"Grades ({name})")
    table.add_column("Case")
    table.add_column("Status")
    table.add_column("Score", justify="right")
    table.add_column("Trials", justify="right")
    for r in result.case_results:
        status = {
            "passed": "[green]PASS[/green]",
            "failed": "[red]FAIL[/red]",
            "error": "[yellow]ERROR[/yellow]",
        }.get(r.status.value, r.status.value)
        table.add_row(
            r.case_id,
            status,
            f"{r.overall_score:.3f}",
            str(r.total_trials),
        )
    console.print(table)

    if verbose:
        for r in result.case_results:
            if not r.evaluator_results:
                continue
            breakdown = Table(title=f"{r.case_id} — graders")
            breakdown.add_column("Grader")
            breakdown.add_column("Scope")
            breakdown.add_column("Passed")
            breakdown.add_column("Score", justify="right")
            for er in r.evaluator_results:
                breakdown.add_row(
                    er.name,
                    er.grader_scope,
                    "[green]✓[/green]" if er.passed else "[red]✗[/red]",
                    f"{er.score:.3f}",
                )
            console.print(breakdown)

    console.print(
        f"\nGraded {report.graded} trace(s), reused {report.skipped} up-to-date grade(s)  |  "
        f"pass rate {result.pass_rate:.1%}, mean score {result.average_score:.3f}"
    )
    console.print(f"[green]Grades written to:[/green] {report.grade_dir}")

    _print_grade_diagnostics(report)

    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
        console.print(f"[green]Results saved to:[/green] {output_path}")

    sys.exit(0 if result.passed_cases == result.total_cases else 1)


def _print_grade_diagnostics(report: "GradeReport") -> None:
    """Surface everything that was skipped — silence would read as full coverage."""
    if report.has_stale:
        console.print(
            f"[yellow]⚠ {report.stale} existing grade(s) came from an older version "
            f"of this grader spec — use --regrade to discard and redo them[/yellow]"
        )
    if report.unmatched:
        console.print(
            f"[yellow]⚠ {len(report.unmatched)} trace task_id(s) have no matching case "
            f"in the scenario:[/yellow] {', '.join(report.unmatched[:10])}"
            + (" …" if len(report.unmatched) > 10 else "")
        )
    if report.missing_traces:
        console.print(
            f"[yellow]⚠ {len(report.missing_traces)} case(s) have no trace "
            f"(not graded):[/yellow] {', '.join(report.missing_traces[:10])}"
            + (" …" if len(report.missing_traces) > 10 else "")
        )
    for path, reason in report.unreadable:
        console.print(f"[yellow]⚠ skipped {path}: {reason}[/yellow]")
