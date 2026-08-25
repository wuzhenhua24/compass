"""``compass test`` — run a scenario, one model or several."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.markup import escape
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from compass.cli.app import cli, console
from compass.core.runner import Compass
from compass.core.scenario import Scenario
from compass.report.html import HTMLReporter

if TYPE_CHECKING:
    from compass.core.result import EvalResult
    from compass.report.leaderboard import Leaderboard


@cli.command()
@click.argument("scenario", type=click.Path(exists=True))
@click.option("--case", "-c", multiple=True, help="Specific case IDs to run")
@click.option("--stage", "-s", multiple=True, help="Filter by stage (smoke, integration, ...)")
@click.option("--category", multiple=True, help="Filter by category (backend, fullstack, gym, ...)")
@click.option("--parallel", "-p", is_flag=True, help="Run cases in parallel")
@click.option("--workers", "-w", default=4, help="Number of parallel workers")
@click.option("--report", "-r", type=click.Choice(["html", "json"]), help="Generate report")
@click.option("--output", "-o", type=click.Path(), help="Report output path")
@click.option("--trace-dir", type=click.Path(), help="Directory to save execution traces")
@click.option("--trace-format", type=click.Choice(["json", "jsonl"]), default="json",
              help="Trace file format")
@click.option("--resume", is_flag=True, help="Resume from last checkpoint (requires --trace-dir)")
@click.option(
    "--model", "-m", "models", multiple=True,
    help="Run the scenario once per model and rank the results (repeatable)",
)
@click.option(
    "--model-key", default="model", show_default=True,
    help="Which agent.config key --model overrides",
)
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def test(
    scenario: str,
    case: tuple[Any, ...],
    stage: tuple[Any, ...],
    category: tuple[Any, ...],
    parallel: bool,
    workers: int,
    report: str | None,
    output: str | None,
    trace_dir: str | None,
    trace_format: str,
    resume: bool,
    models: tuple[str, ...],
    model_key: str,
    verbose: bool,
) -> None:
    """Run test scenarios.

    SCENARIO can be a YAML file or directory containing scenario files.

    With one or more -m/--model the scenario runs once per model and the
    results are ranked. The table is a reading, not a measurement, so it
    reports mean ± stderr and tests whether the top two rows actually differ:

    \b
      compass test qa.yaml -m gpt-5 -m claude-sonnet-5
      compass test qa.yaml -m a -m b --trace-dir ./traces   # traces/<model>/
    """
    if resume and not trace_dir:
        console.print("[red]--resume requires --trace-dir[/red]")
        sys.exit(1)

    scenario_path = Path(scenario)

    # Collect scenarios
    if scenario_path.is_file():
        scenarios = [Scenario.from_yaml(scenario_path)]
    else:
        scenarios = [
            Scenario.from_yaml(f)
            for f in sorted(scenario_path.glob("**/*"))
            if f.suffix in (".yaml", ".yml") and not f.name.startswith("_")
        ]

    if not scenarios:
        console.print("[red]No scenarios found[/red]")
        sys.exit(1)

    from compass.report.leaderboard import build_leaderboard, slugify

    # Each (scenario, model) pair is one run. Without -m there is exactly one
    # variant per scenario and nothing changes.
    variants = _model_variants(scenarios, list(models), model_key)

    console.print(Panel(
        f"[bold]Compass Test Runner[/bold]\n{len(scenarios)} scenario(s) to run"
        + (f" × {len(models)} model(s)" if models else "")
    ))

    # Run tests
    compass = Compass()
    all_results = []
    ranked: list[tuple[str, EvalResult]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for label, scn in variants:
            # Escaped, not interpolated raw: a variant name carries the axis
            # value, and `--model-key append_system_prompt_file -m prompts/a.md`
            # puts a path in square brackets, which rich would read as markup.
            display_name = escape(scn.name)
            task = progress.add_task(f"Running: {display_name}", total=None)

            # Per-model trace subdirectory: without it the models would
            # overwrite each other's traces, and each one's agent config would
            # invalidate the previous one's checkpoint.
            variant_trace_dir = trace_dir
            if trace_dir and models:
                variant_trace_dir = str(Path(trace_dir) / slugify(label))

            try:
                result = asyncio.run(
                    compass.run(
                        scn,
                        case_ids=list(case) if case else None,
                        stages=list(stage) if stage else None,
                        categories=list(category) if category else None,
                        parallel=parallel,
                        max_workers=workers,
                        trace_dir=variant_trace_dir,
                        trace_format=trace_format,
                        resume=resume,
                    )
                )
                all_results.append(result)
                ranked.append((label, result))

                # Print summary. pass_rate excludes harness errors, so a run
                # with errors is not "all green" even at pass_rate == 1.0.
                if result.error_cases:
                    status = "[yellow]ERROR[/yellow]"
                elif result.pass_rate == 1.0:
                    status = "[green]PASS[/green]"
                else:
                    status = "[red]FAIL[/red]"
                progress.update(
                    task,
                    description=(
                        f"{display_name}: {status} "
                        f"({result.passed_cases}/{result.evaluated_cases})"
                    ),
                )

                if verbose:
                    _print_result_details(result)

            except Exception as e:
                console.print(f"[red]Error running {display_name}: {escape(str(e))}[/red]")
                progress.update(task, description=f"{display_name}: [red]ERROR[/red]")

    # Print summary table
    _print_summary_table(all_results)

    board = None
    if models and len(ranked) > 1:
        board = build_leaderboard(ranked)
        _print_leaderboard(board)

    # Generate report
    if report and all_results:
        import json

        output_path = output or f"compass_report.{report}"
        # `-o out/results.json` into a directory that does not exist yet used to
        # traceback *after* every case had run — losing the whole (expensive)
        # run to a missing mkdir.
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        if report == "html":
            reporter = HTMLReporter()
            reporter.generate(all_results, output_path)
            console.print(f"\n[green]Report generated:[/green] {output_path}")
        elif report == "json":
            report_data = {
                "results": [r.to_dict() for r in all_results],
                "summary": {
                    "total_scenarios": len(all_results),
                    "total_cases": sum(r.total_cases for r in all_results),
                    "passed_cases": sum(r.passed_cases for r in all_results),
                    "failed_cases": sum(r.failed_cases for r in all_results),
                    "error_cases": sum(r.error_cases for r in all_results),
                },
            }
            if board is not None:
                report_data["leaderboard"] = board.to_dict()
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]Report generated:[/green] {output_path}")

            # One file per model as well: `compass compare` pairs by case_id,
            # so it needs the runs separated to measure any two of them.
            #
            # Written whenever `-m` was given, including for a *single* model —
            # not only when there is a leaderboard to build. A cross-stack A/B
            # is two separate invocations of one model each (different adapters
            # cannot share a scenario), and both write the same
            # `results.json`: the second silently overwrote the first, leaving
            # nothing to compare and no sign that anything was lost.
            if models and ranked:
                base = Path(output_path)
                for label, result in ranked:
                    per_model = base.with_name(
                        f"{base.stem}.{slugify(label)}{base.suffix}"
                    )
                    with open(per_model, "w", encoding="utf-8") as f:
                        json.dump(result.to_dict(), f, ensure_ascii=False, indent=2)
                console.print(
                    f"[green]Per-model results:[/green] "
                    f"{base.stem}.<model>{base.suffix} — pair any two with "
                    f"[bold]compass compare[/bold]"
                )

    # Show trace directory info
    if trace_dir and all_results:
        console.print(f"[green]Traces saved to:[/green] {trace_dir}/ (format: {trace_format})")

    # Exit with appropriate code
    total_passed = sum(r.passed_cases for r in all_results)
    total_cases = sum(r.total_cases for r in all_results)
    sys.exit(0 if total_passed == total_cases else 1)


def _model_variants(
    scenarios: list[Any], models: list[str], model_key: str
) -> list[tuple[str, Scenario]]:
    """Expand scenarios into one labelled run per model.

    Without ``-m`` this is the identity: one variant per scenario, unchanged.
    With models, each variant is a deep copy whose ``agent.config[model_key]``
    is overridden — the agent config is the axis, so every other part of the
    scenario (cases, graders, thresholds) stays identical and the runs remain
    comparable.
    """
    if not models:
        return [(scn.name, scn) for scn in scenarios]

    variants: list[tuple[str, Scenario]] = []
    for scn in scenarios:
        for model in models:
            variant = scn.model_copy(deep=True)
            variant.agent.config[model_key] = model
            # The run really is a different configuration, so say so in the
            # name that reports and summaries display.
            variant.name = f"{scn.name} [{model}]"
            label = model if len(scenarios) == 1 else f"{scn.name} / {model}"
            variants.append((label, variant))
    return variants


def _print_leaderboard(board: "Leaderboard") -> None:
    """Render the ranking, then immediately qualify it."""
    show_errors = any(e.error_cases for e in board.entries)
    show_reliability = any(e.reliability is not None for e in board.entries)

    table = Table(title="Leaderboard")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model", style="bold")
    table.add_column("Score (mean ± stderr)", justify="right")
    table.add_column("Pass Rate", justify="right")
    if show_reliability:
        k = next(e.reliability_k for e in board.entries if e.reliability_k)
        table.add_column(f"pass^{k}", justify="right")
    table.add_column("Cases", justify="right")
    if show_errors:
        table.add_column("Errors", justify="right", style="yellow")

    for entry in board.entries:
        rate = entry.pass_rate
        color = "green" if rate >= 0.7 else "yellow" if rate >= 0.5 else "red"
        row = [
            str(entry.rank),
            escape(entry.label),
            entry.score_display,
            f"[{color}]{rate:.1%}[/{color}]",
        ]
        if show_reliability:
            row.append(
                "-" if entry.reliability is None else f"{entry.reliability:.3f}"
            )
        row.append(str(entry.evaluated_cases))
        if show_errors:
            row.append(str(entry.error_cases))
        table.add_row(*row)

    console.print()
    console.print(table)

    # A ranked table invites reading a winner out of noise; say plainly
    # whether the top gap survives a paired test.
    style = "green" if (board.top_gap and board.top_gap.significant) else "yellow"
    console.print(f"[{style}]{board.verdict}[/{style}]")

    if board.unpaired:
        console.print(
            f"[yellow]⚠ {len(board.unpaired)} case(s) were not run by both of the "
            f"top two, and are excluded from that test:[/yellow] "
            + ", ".join(board.unpaired[:10])
            + (" …" if len(board.unpaired) > 10 else "")
        )


def _print_result_details(result: "EvalResult") -> None:
    """Print detailed results for a scenario."""
    for case in result.case_results:
        status = "[green]PASS[/green]" if case.passed else "[red]FAIL[/red]"
        score_info = f"avg: {case.overall_score:.3f}"
        if case.has_trials:
            score_info += f", best: {case.best_score:.3f}"
            score_info += f", trials: {case.passed_trials}/{case.total_trials}"
        console.print(f"    {case.case_id}: {status} ({score_info})")

        if case.error:
            console.print(f"      [red]Error: {case.error}[/red]")

        for eval_result in case.evaluator_results:
            eval_status = "[green]✓[/green]" if eval_result.passed else "[red]✗[/red]"
            console.print(f"      {eval_status} {eval_result.name}: {eval_result.score:.3f}")


def _print_summary_table(results: list["EvalResult"]) -> None:
    """Print summary table of all results."""
    if not results:
        return

    # Check if any scenario has multi-trial data
    has_trials = any(
        cr.has_trials
        for r in results for cr in r.case_results
    )

    # Harness errors are excluded from pass rate, so show the column that
    # explains the denominator rather than letting the number look wrong.
    has_errors = any(r.error_cases for r in results)

    table = Table(title="Test Summary")
    table.add_column("Scenario", style="cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    if has_errors:
        table.add_column("Error", justify="right", style="yellow")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Avg Score", justify="right")
    if has_trials:
        table.add_column("Best-of-k", justify="right")
    table.add_column("Duration", justify="right")

    for r in results:
        pass_rate = f"{r.pass_rate * 100:.1f}%"
        avg_score = f"{r.average_score:.3f}"
        duration = f"{r.duration_ms:.0f}ms"

        row = [
            escape(r.scenario_name),
            str(r.total_cases),
            str(r.passed_cases),
            str(r.failed_cases),
        ]
        if has_errors:
            row.append(str(r.error_cases))
        row += [pass_rate, avg_score]
        if has_trials:
            row.append(f"{r.best_of_k_score:.3f}")
        row.append(duration)

        table.add_row(*row)

    console.print()
    console.print(table)

    if has_errors:
        total_errors = sum(r.error_cases for r in results)
        console.print(
            f"[yellow]⚠ {total_errors} case(s) failed in the harness "
            f"(not the agent) and are excluded from pass rate / avg score[/yellow]"
        )
