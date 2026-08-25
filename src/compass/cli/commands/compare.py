"""``compass compare`` — paired comparison of two runs."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import click
from rich.panel import Panel
from rich.table import Table

from compass.cli.app import cli, console

if TYPE_CHECKING:
    pass


def _render_metric_comparison(cmp: Any, output: str | None, as_json: bool) -> None:
    """Render a metric comparison — no pass rate, no flips, on purpose."""
    import json

    if as_json:
        console.print_json(json.dumps(cmp.to_dict(), ensure_ascii=False))
        return

    console.print(Panel(
        f"[bold]Compass Metric Comparison[/bold]\n"
        f"A (baseline):  {cmp.label_a}\n"
        f"B (candidate): {cmp.label_b}\n"
        f"metric:        {cmp.metric}\n"
        f"{cmp.n_paired} paired case(s)",
    ))

    if cmp.n_paired == 0:
        console.print(
            f"[red]No case measured {cmp.metric!r} in both runs.[/red] "
            "Metrics come from EvaluatorResult.metrics — check the name, and "
            "that a grader emitting it ran on these cases."
        )
        sys.exit(1)

    table = Table(title="Summary (paired cases only)")
    for col in ("Metric", "A", "B", "Diff (B−A)", "95% CI"):
        table.add_column(col, justify="right" if col != "Metric" else "left")
    s = cmp.stats
    table.add_row(
        cmp.metric,
        f"{cmp.mean_a:.4g}",
        f"{cmp.mean_b:.4g}",
        f"{s.mean_diff:+.4g}",
        f"[{s.ci_low:+.4g}, {s.ci_high:+.4g}]",
    )
    console.print(table)
    console.print(
        "[dim]No pass rate or flips: a metric has no pass/fail. Mind the "
        "direction — on turns, cost and tool calls, lower is better.[/dim]"
    )

    if cmp.multi_trial_cases:
        console.print(
            f"[yellow]⚠ {cmp.multi_trial_cases} of {cmp.n_paired} paired case(s) ran "
            f"multiple trials, but a case carries one trial's grader results — this "
            f"compares one attempt per case, not the average over them.[/yellow]"
        )

    for side, missing in (("A", cmp.unmeasured_a), ("B", cmp.unmeasured_b)):
        if missing:
            console.print(
                f"[yellow]⚠ {len(missing)} case(s) in {side} never measured "
                f"{cmp.metric!r} (dropped):[/yellow] {', '.join(missing[:10])}"
                + (" …" if len(missing) > 10 else "")
            )
    for side, extra in (("A", cmp.only_in_a), ("B", cmp.only_in_b)):
        if extra:
            console.print(
                f"[yellow]⚠ {len(extra)} case(s) only in {side} (dropped):[/yellow] "
                f"{', '.join(extra[:10])}" + (" …" if len(extra) > 10 else "")
            )

    style = "green" if s.significant else "yellow"
    console.print(Panel(f"[{style}]{cmp.verdict}[/{style}]", title="Verdict"))

    if output:
        Path(output).write_text(
            json.dumps(cmp.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        console.print(f"\n[green]Comparison saved to:[/green] {output}")



@cli.command()
@click.argument("results_a", type=click.Path(exists=True))
@click.argument("results_b", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save comparison report as JSON")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON instead of tables")
@click.option(
    "--on",
    "on",
    metavar="GRADER",
    help="Compare on one grader's score (label, else name) instead of overall_score",
)
@click.option(
    "--metric",
    "metric",
    metavar="NAME",
    help="Compare a numeric metric (turns, cost_usd, tool_calls, calls_<tool>, …)",
)
@click.option(
    "--missing-as-zero",
    is_flag=True,
    help="With --metric: read an absent metric as 0 rather than unmeasured",
)
def compare(
    results_a: str,
    results_b: str,
    output: str | None,
    as_json: bool,
    on: str | None,
    metric: str | None,
    missing_as_zero: bool,
) -> None:
    """Paired comparison of two evaluation runs (A = baseline, B = candidate).

    Pairs cases by id, lists pass/fail flips in both directions, and reports
    the paired difference with a 95% confidence interval — so a moved average
    that sits inside the noise band is not mistaken for a real change.

    RESULTS_A / RESULTS_B accept the same inputs as `compass analyze`:
    a results JSON file or a directory of result files.

    --on scopes the whole comparison to a single grader. Reach for it when
    correctness is decided by gate graders: gate scores are excluded from
    `overall_score` by design, so the default comparison is measuring process
    cost and the correctness signal never reaches it. Cases where that grader
    produced no measurement are reported and dropped, not scored as zero.

    --metric compares one number instead of a score: turns, cost_usd,
    tool_calls, calls_<tool>, loop_issues. There is no pass/fail on a metric,
    so no pass rate and no flips are reported — just the paired difference and
    its interval. Remember which direction is good: on most process metrics a
    negative difference is the better run.

    Metric names are lowercased, so it is `calls_grep`, not `calls_Grep`.
    `calls_<tool>` only exists where the tool was used at all, so comparing it
    drops every case that never touched that tool — which is the wrong reading
    when "never used it" is the finding. `--missing-as-zero` switches those
    cases to a measured 0. Do not reach for it on `cost_usd`: an absent cost
    means the grader did not run, and that is not a run that cost nothing.
    """
    import json

    from compass.report.compare import (
        AmbiguousGraderError,
        compare_metric,
        compare_paths,
    )

    if metric and on:
        console.print("[red]--on and --metric compare different things; pick one[/red]")
        sys.exit(2)

    if metric:
        try:
            _render_metric_comparison(
                compare_metric(results_a, results_b, metric, missing_as_zero),
                output, as_json,
            )
        except AmbiguousGraderError as e:
            console.print(f"[red]Ambiguous --metric {metric!r}:[/red] {e}")
            sys.exit(2)
        return

    try:
        report = compare_paths(results_a, results_b, on=on)
    except AmbiguousGraderError as e:
        console.print(f"[red]Ambiguous --on {on!r}:[/red] {e}")
        sys.exit(2)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    scoped = f"\nscored on:     {report.on}" if report.on else ""
    console.print(Panel(
        f"[bold]Compass Paired Comparison[/bold]\n"
        f"A (baseline):  {report.label_a}\n"
        f"B (candidate): {report.label_b}{scoped}\n"
        f"{report.n_paired} paired case(s)",
    ))

    if report.n_paired == 0:
        console.print("[red]No paired cases between the two runs[/red]")
        if report.on and (report.unmeasured_a or report.unmeasured_b):
            # The cases are there; the selected grader is not. Saying "no
            # paired cases" without this reads as a mismatched pair of runs.
            console.print(
                f"[yellow]No case measured a grader named {report.on!r} — "
                f"check the label/name against the results file[/yellow]"
            )
        if report.only_in_a:
            console.print(f"[yellow]Only in A:[/yellow] {', '.join(report.only_in_a)}")
        if report.only_in_b:
            console.print(f"[yellow]Only in B:[/yellow] {', '.join(report.only_in_b)}")
        sys.exit(1)

    # Summary table: point estimates + paired diff with uncertainty
    summary = Table(title="Summary (paired cases only)")
    summary.add_column("Metric")
    summary.add_column("A", justify="right")
    summary.add_column("B", justify="right")
    summary.add_column("Diff (B−A)", justify="right")
    summary.add_column("95% CI", justify="right")

    ps, ss = report.pass_stats, report.score_stats
    # n_paired > 0 (guarded above) is exactly the condition under which
    # compare_paths computes both — same invariant ComparisonReport.verdict asserts.
    assert ps is not None and ss is not None
    # Row labels name what was actually measured: with --on these are one
    # grader's numbers, not the case's.
    qualifier = f" ({report.on})" if report.on else ""
    summary.add_row(
        f"Pass rate{qualifier}",
        f"{report.pass_rate_a:.1%}",
        f"{report.pass_rate_b:.1%}",
        f"{ps.mean_diff:+.1%}",
        f"[{ps.ci_low:+.1%}, {ps.ci_high:+.1%}]",
    )
    summary.add_row(
        f"Mean score{qualifier}",
        f"{report.mean_score_a:.3f}",
        f"{report.mean_score_b:.3f}",
        f"{ss.mean_diff:+.3f}",
        f"[{ss.ci_low:+.3f}, {ss.ci_high:+.3f}]",
    )
    console.print(summary)

    # Flips: the per-case evidence behind (or against) the aggregate
    if report.improved or report.regressed:
        flips = Table(title="Case flips")
        flips.add_column("Case")
        flips.add_column("A → B")
        flips.add_column("Score A", justify="right")
        flips.add_column("Score B", justify="right")
        for f in report.regressed:
            flips.add_row(
                f.case_id, "[red]pass → fail[/red]",
                f"{f.score_a:.3f}", f"{f.score_b:.3f}",
            )
        for f in report.improved:
            flips.add_row(
                f.case_id, "[green]fail → pass[/green]",
                f"{f.score_a:.3f}", f"{f.score_b:.3f}",
            )
        console.print(flips)

    console.print(
        f"Unchanged: {report.both_pass} both-pass, {report.both_fail} both-fail  |  "
        f"Flips: [green]{len(report.improved)} improved[/green], "
        f"[red]{len(report.regressed)} regressed[/red]"
    )

    # A case scored under two different grading contracts is not a clean A/B
    if report.regraded:
        console.print(
            f"[yellow]⚠ {len(report.regraded)} case(s) were graded by a different "
            f"grader spec in B than in A — their diff is not attributable to the "
            f"agent:[/yellow] {', '.join(report.regraded[:10])}"
            + (" …" if len(report.regraded) > 10 else "")
        )

    # Cases the selected grader never measured. Not scored as zero — that
    # would report an agent failure where the truth is a missing sample — so
    # they have to be visible, or the comparison silently narrows.
    for side, missing in (("A", report.unmeasured_a), ("B", report.unmeasured_b)):
        if missing:
            console.print(
                f"[yellow]⚠ {len(missing)} case(s) in {side} have no "
                f"{report.on!r} measurement (dropped from stats):[/yellow] "
                f"{', '.join(missing[:10])}" + (" …" if len(missing) > 10 else "")
            )

    # Coverage changes are reported, never silently dropped
    if report.only_in_a:
        console.print(
            f"[yellow]⚠ {len(report.only_in_a)} case(s) only in A "
            f"(dropped from stats):[/yellow] {', '.join(report.only_in_a[:10])}"
            + (" …" if len(report.only_in_a) > 10 else "")
        )
    if report.only_in_b:
        console.print(
            f"[yellow]⚠ {len(report.only_in_b)} case(s) only in B "
            f"(dropped from stats):[/yellow] {', '.join(report.only_in_b[:10])}"
            + (" …" if len(report.only_in_b) > 10 else "")
        )

    style = (
        "green" if ps.significant and ps.mean_diff > 0
        else "red" if (ps.significant and ps.mean_diff < 0)
        or (ss.significant and ss.mean_diff < 0)
        else "yellow"
    )
    console.print(Panel(f"[{style}]{report.verdict}[/{style}]", title="Verdict"))

    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Comparison saved to:[/green] {output_path}")
