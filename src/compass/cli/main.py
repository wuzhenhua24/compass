"""CLI entry point for Compass."""

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from compass import __version__
from compass.core.runner import Compass
from compass.core.scenario import Scenario
from compass.adapters import list_adapters
from compass.report.html import HTMLReporter
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult, iter_case_dicts
from compass.report.console import ConsoleReporter
from compass.report.site import DEFAULT_HISTORY

if TYPE_CHECKING:
    from compass.core.regrade import GradeReport
    from compass.report.leaderboard import Leaderboard


console = Console()


@click.group()
@click.version_option(version=__version__, prog_name="compass")
def cli():
    """Compass - Agent QA Framework.

    A universal testing and evaluation framework for AI Agents.
    """
    pass


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
@click.option("--trace-format", type=click.Choice(["json", "jsonl"]), default="json", help="Trace file format")
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
    case: tuple,
    stage: tuple,
    category: tuple,
    parallel: bool,
    workers: int,
    report: Optional[str],
    output: Optional[str],
    trace_dir: Optional[str],
    trace_format: str,
    resume: bool,
    models: tuple[str, ...],
    model_key: str,
    verbose: bool,
):
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
    ranked: list[tuple[str, object]] = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for label, scn in variants:
            task = progress.add_task(f"Running: {scn.name}", total=None)

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
                progress.update(task, description=f"{scn.name}: {status} ({result.passed_cases}/{result.evaluated_cases})")

                if verbose:
                    _print_result_details(result)

            except Exception as e:
                console.print(f"[red]Error running {scn.name}: {e}[/red]")
                progress.update(task, description=f"{scn.name}: [red]ERROR[/red]")

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
            if board is not None:
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
    scenarios: list, models: list[str], model_key: str
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
            entry.label,
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


@cli.command()
@click.argument("image", type=click.Path(exists=True))
@click.option("--prompt", "-p", required=True, help="Prompt to evaluate against")
@click.option("--graders", "-g", default="semantic_match", help="Comma-separated graders")
@click.option("--verbose", "-v", is_flag=True, help="Verbose output")
def eval(image: str, prompt: str, graders: str, verbose: bool):
    """Evaluate a single image using graders.

    IMAGE is the path to the image file to evaluate.

    Examples:
        compass eval image.png -p "a cat sitting on a chair"
        compass eval image.png -p "landscape" -g semantic_match,aesthetic_score
    """
    from PIL import Image as PILImage
    from compass.graders import get_grader, GradeContext, GradeResult
    from compass.core.transcript import Outcome

    console.print(Panel(f"[bold]Compass Image Grader[/bold]\nImage: {image}"))

    # Load image
    try:
        img = PILImage.open(image)
    except Exception as e:
        console.print(f"[red]Failed to load image: {e}[/red]")
        sys.exit(1)

    # Create outcome with the image
    outcome = Outcome(image=img)

    # Create grade context
    context = GradeContext(prompt=prompt, outcome=outcome)

    # Run graders
    grader_names = [g.strip() for g in graders.split(",")]
    results: list[GradeResult] = []

    for name in grader_names:
        try:
            grader_cls = get_grader(name)
            grader = grader_cls({})
            result = asyncio.run(grader.grade(context))
            results.append(result)

            status = "[green]PASS[/green]" if result.passed else "[red]FAIL[/red]"
            console.print(f"  {name}: {status} (score: {result.score:.3f})")

            if verbose and result.details:
                for key, value in result.details.items():
                    console.print(f"    {key}: {value}")

        except KeyError:
            console.print(f"  [yellow]{name}: Unknown grader[/yellow]")
        except Exception as e:
            console.print(f"  [red]{name}: Error - {e}[/red]")

    # Overall result
    if results:
        avg_score = sum(r.score for r in results) / len(results)
        all_passed = all(r.passed for r in results)
        console.print(f"\n[bold]Overall:[/bold] {'[green]PASS[/green]' if all_passed else '[red]FAIL[/red]'} (avg score: {avg_score:.3f})")


@cli.command()
@click.argument("topic", required=False)
@click.option("--all", "show_all", is_flag=True, help="Print every topic, concatenated")
def docs(topic: Optional[str], show_all: bool) -> None:
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


@cli.command("list")
def list_registered():
    """List available graders and adapters.

    Named ``list_registered`` rather than ``list``: a module-level function
    called ``list`` shadows the builtin for every other command in this file,
    which silently turned ``list(...)`` calls into invocations of this command.
    """
    from compass.graders import list_graders, GraderType

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


@cli.command()
@click.argument("output", type=click.Path(), default="scenario.yaml")
def init(output: str):
    """Create a new scenario template."""
    template = """name: "My Test Scenario"
description: "Description of what this scenario tests"

agent:
  adapter: image
  endpoint: "http://localhost:8000"

# Default graders applied to all cases
default_graders:
  - type: model
    name: semantic_match
    weight: 0.4
    config:
      threshold: 0.25
  - type: model
    name: safety_check
    weight: 0.0
    required: true
    config:
      checks: [nsfw]

# Default aggregation settings
default_aggregation:
  method: weighted_sum
  pass_threshold: 0.7
  required_graders: [safety_check]

cases:
  - id: "example_case_1"
    input:
      prompt: "A beautiful landscape with mountains and a lake"
      negative_prompt: "blurry, low quality, distorted"
      params:
        width: 1024
        height: 1024
        steps: 30

    graders:
      - type: model
        name: aesthetic_score
        weight: 0.3
        config:
          min_score: 5.0

      - type: model
        name: vlm_judge
        weight: 0.3
        config:
          model: "gpt-4o"
          criteria:
            - "The image contains mountains"
            - "The image contains a lake"
            - "The overall scene is a landscape"

    aggregation:
      method: weighted_sum
      pass_threshold: 0.7

    tags:
      - landscape
      - nature
    stage: smoke

  - id: "example_case_2"
    input:
      prompt: "A portrait of a robot in cyberpunk style"
      negative_prompt: "realistic human, photograph"
      params:
        width: 768
        height: 1024

    tags:
      - portrait
      - cyberpunk
    stage: integration
"""

    output_path = Path(output)
    with open(output_path, "w") as f:
        f.write(template)

    console.print(f"[green]Created scenario template:[/green] {output_path}")


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
@click.option("--output", "-o", type=click.Path(), help="Save results as JSON (analyze/compare input)")
@click.option("--verbose", "-v", is_flag=True, help="Show per-grader breakdown")
def grade(
    traces: str,
    scenario: str,
    grade_set: Optional[str],
    case: tuple[str, ...],
    regrade: bool,
    output: Optional[str],
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


@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save report as JSON")
def analyze(results_path: str, output: Optional[str]):
    """Analyze evaluation results with Transcript/Outcome breakdown.

    RESULTS_PATH is a JSON file or directory containing evaluation result files.
    Each result file should contain a list of task evaluation results.
    """
    import json

    results_file = Path(results_path)

    # Load results
    if results_file.is_file():
        json_files = [results_file]
    else:
        json_files = list(results_file.glob("**/*.json"))

    if not json_files:
        console.print("[red]No result files found[/red]")
        sys.exit(1)

    # Parse into TaskEvalResult objects
    from compass.graders.base import GradeResult, GraderScope, GraderType

    task_results: list[TaskEvalResult] = []

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)

            # Accept every results shape Compass writes: a bare case list,
            # an EvalResult (compass grade -o), or a test --report json bundle
            items = iter_case_dicts(data)

            for item in items:
                grade_results = []
                # Support both grade_results and evaluator_results field names
                gr_list = item.get("grade_results") or item.get("evaluator_results", [])
                for gr_data in gr_list:
                    grade_results.append(GradeResult(
                        name=gr_data.get("name", ""),
                        grader_type=GraderType(gr_data.get("grader_type", "code")),
                        grader_scope=GraderScope(gr_data.get("grader_scope", "outcome")),
                        passed=gr_data.get("passed", False),
                        score=gr_data.get("score", 0.0),
                        weight=gr_data.get("weight", 1.0),
                        details=gr_data.get("details") or gr_data.get("metadata", {}),
                        tags=gr_data.get("tags", []),
                        metrics=gr_data.get("metrics", {}),
                        failure_tags=gr_data.get("failure_tags", []),
                        reasoning=gr_data.get("reasoning", ""),
                        error=gr_data.get("error"),
                    ))

                # Extract best_score from trial_metrics if available
                trial_m = item.get("trial_metrics") or {}
                best_score = (
                    trial_m.get("best_of_k")
                    or trial_m.get("score_max")
                    or item.get("best_score")
                )

                task_results.append(TaskEvalResult(
                    task_id=item.get("task_id", jf.stem),
                    passed=item.get("passed", False),
                    grade_results=grade_results,
                    duration_ms=item.get("duration_ms", 0.0),
                    metadata=item.get("metadata", {}),
                    tags=item.get("tags", []),
                    category=item.get("category", ""),
                    best_score=best_score,
                ))

        except Exception as e:
            console.print(f"[yellow]Warning: Failed to load {jf}: {e}[/yellow]")

    if not task_results:
        console.print("[red]No valid results parsed[/red]")
        sys.exit(1)

    console.print(Panel(
        f"[bold]Compass Result Analyzer[/bold]\n"
        f"{len(task_results)} task result(s) from {len(json_files)} file(s)",
    ))

    # Run analysis
    analyzer = EvalResultAnalyzer(task_results)
    report = analyzer.analyze()

    # Render to console
    reporter = ConsoleReporter(console)
    reporter.render(report)

    # Optionally save as JSON
    if output:
        output_path = Path(output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Report saved to:[/green] {output_path}")


@cli.command()
@click.argument("results_a", type=click.Path(exists=True))
@click.argument("results_b", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save comparison report as JSON")
@click.option("--json", "as_json", is_flag=True, help="Print raw JSON instead of tables")
def compare(results_a: str, results_b: str, output: Optional[str], as_json: bool):
    """Paired comparison of two evaluation runs (A = baseline, B = candidate).

    Pairs cases by id, lists pass/fail flips in both directions, and reports
    the paired difference with a 95% confidence interval — so a moved average
    that sits inside the noise band is not mistaken for a real change.

    RESULTS_A / RESULTS_B accept the same inputs as `compass analyze`:
    a results JSON file or a directory of result files.
    """
    import json

    from compass.report.compare import compare_paths

    report = compare_paths(results_a, results_b)

    if as_json:
        console.print_json(json.dumps(report.to_dict(), ensure_ascii=False))
        return

    console.print(Panel(
        f"[bold]Compass Paired Comparison[/bold]\n"
        f"A (baseline):  {report.label_a}\n"
        f"B (candidate): {report.label_b}\n"
        f"{report.n_paired} paired case(s)",
    ))

    if report.n_paired == 0:
        console.print("[red]No paired cases between the two runs[/red]")
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
    summary.add_row(
        "Pass rate",
        f"{report.pass_rate_a:.1%}",
        f"{report.pass_rate_b:.1%}",
        f"{ps.mean_diff:+.1%}",
        f"[{ps.ci_low:+.1%}, {ps.ci_high:+.1%}]",
    )
    summary.add_row(
        "Mean score",
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
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        console.print(f"\n[green]Comparison saved to:[/green] {output_path}")


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--steps", "-s", is_flag=True, help="Show tool call input/output details")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def trace(trace_file: str, steps: bool, as_json: bool):
    """View an execution transcript.

    TRACE_FILE is a JSON or JSONL trace file saved by `compass test --trace-dir`.

    Examples:
        compass trace results/case.json
        compass trace results/case.jsonl --steps
        compass trace results/case.json --json
    """
    import json as json_mod

    from compass.core.transcript import Transcript

    trace_path = Path(trace_file)

    # Load transcript based on file extension
    try:
        if trace_path.suffix == ".jsonl":
            transcript = Transcript.load_jsonl(trace_path)
        else:
            transcript = Transcript.load(trace_path)
    except Exception as e:
        console.print(f"[red]Failed to load transcript: {e}[/red]")
        sys.exit(1)

    # Raw JSON output
    if as_json:
        console.print_json(json_mod.dumps(transcript.to_dict(), ensure_ascii=False))
        return

    # --- Header ---
    status_str = "[green]PASS[/green]" if transcript.final_passed else "[red]FAIL[/red]"
    console.print(Panel(
        f"[bold]Task:[/bold] {transcript.task_id}  |  "
        f"[bold]Trial:[/bold] {transcript.trial_id}  |  "
        f"[bold]Result:[/bold] {status_str}  |  "
        f"[bold]Score:[/bold] {transcript.final_score:.3f}",
        title="Compass Trace Viewer",
    ))

    # --- Input ---
    console.print("\n[bold cyan]Input[/bold cyan]")
    console.print(f"  Prompt: {transcript.input_prompt or '(empty)'}")
    if transcript.input_params:
        for k, v in transcript.input_params.items():
            console.print(f"  {k}: {v}")

    # --- Timing ---
    console.print("\n[bold cyan]Timing[/bold cyan]")
    console.print(f"  Start: {transcript.start_time.isoformat()}")
    if transcript.end_time:
        console.print(f"  End:   {transcript.end_time.isoformat()}")
    console.print(f"  Duration: {transcript.total_duration_ms:.0f}ms")

    # --- Tool Calls ---
    if transcript.tool_calls:
        total_cost = transcript.sum_cost()
        total_tokens = transcript.sum_tokens()

        n_calls = len(transcript.tool_calls)
        console.print(f"\n[bold cyan]Tool Calls[/bold cyan] ({n_calls})")

        tool_table = Table(show_header=True, header_style="bold")
        tool_table.add_column("#", justify="right", width=4)
        tool_table.add_column("Tool", style="cyan")
        tool_table.add_column("Status", justify="center")
        tool_table.add_column("Duration", justify="right")
        tool_table.add_column("Tokens", justify="right")
        tool_table.add_column("Cost", justify="right")

        for i, tc in enumerate(transcript.tool_calls, 1):
            st = {
                "ok": "[green]ok[/green]",
                "error": "[red]error[/red]",
                "blocked": "[yellow]blocked[/yellow]",
            }
            status_display = st.get(tc.status, tc.status)
            tokens_str = str(tc.tokens.total_tokens) if tc.tokens else "-"
            cost_str = f"${tc.cost.total_usd:.4f}" if tc.cost else "-"

            tool_table.add_row(
                str(i),
                tc.tool_name,
                status_display,
                f"{tc.duration_ms:.0f}ms",
                tokens_str,
                cost_str,
            )

        console.print(tool_table)

        # Summary line
        summary_parts = [f"  Total: {transcript.sum_duration():.0f}ms"]
        if total_tokens.total_tokens > 0:
            tt = total_tokens
            summary_parts.append(
                f"{tt.total_tokens} tokens"
                f" ({tt.input_tokens} in / {tt.output_tokens} out)"
            )
        if total_cost.total_usd > 0:
            summary_parts.append(f"${total_cost.total_usd:.4f}")
        console.print("  ".join(summary_parts))

        # Detailed step view
        if steps:
            console.print("\n[bold cyan]Step Details[/bold cyan]")
            for i, tc in enumerate(transcript.tool_calls, 1):
                console.print(f"\n  [bold]Step {i}: {tc.tool_name}[/bold] [{tc.status}]")
                if tc.input:
                    console.print("  [dim]Input:[/dim]")
                    input_json = json_mod.dumps(
                        tc.input, ensure_ascii=False, default=str,
                    )
                    console.print_json(input_json, indent=4)
                if tc.output is not None:
                    console.print("  [dim]Output:[/dim]")
                    if isinstance(tc.output, str):
                        output_str = json_mod.dumps(tc.output)
                    else:
                        output_str = json_mod.dumps(
                            tc.output, ensure_ascii=False, default=str,
                        )
                    console.print_json(output_str, indent=4)
                if tc.error:
                    console.print(f"  [red]Error:[/red] {tc.error}")

    # --- Reasoning Steps ---
    if transcript.reasoning_steps:
        n_steps = len(transcript.reasoning_steps)
        console.print(
            f"\n[bold cyan]Reasoning Steps[/bold cyan] ({n_steps})"
        )
        for i, step in enumerate(transcript.reasoning_steps, 1):
            console.print(f"  {i}. {step}")

    # --- Outcome ---
    console.print("\n[bold cyan]Outcome[/bold cyan]")
    outcome = transcript.outcome
    if outcome.blocked:
        console.print(f"  [yellow]BLOCKED[/yellow]: {outcome.blocked_reason}")
    if outcome.image_path:
        console.print(f"  Image: {outcome.image_path}")
    if outcome.artifacts:
        console.print(f"  Artifacts: {len(outcome.artifacts)}")
        for a in outcome.artifacts:
            console.print(f"    - {a.artifact_type} (has_output={a.has_output})")
    if outcome.output_data:
        console.print(f"  Output data keys: {', '.join(outcome.output_data.keys())}")
    has_any = (
        outcome.blocked or outcome.image_path
        or outcome.artifacts or outcome.output_data
    )
    if not has_any:
        console.print("  (empty)")

    # --- Grading ---
    if transcript.grader_results:
        console.print("\n[bold cyan]Grading[/bold cyan]")

        grade_table = Table(show_header=True, header_style="bold")
        grade_table.add_column("Grader")
        grade_table.add_column("Result", justify="center")
        grade_table.add_column("Score", justify="right")
        grade_table.add_column("Weight", justify="right")

        for gr in transcript.grader_results:
            passed = gr.get("passed", False)
            result_display = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
            score = gr.get("score", 0.0)
            weight = gr.get("weight", 1.0)

            grade_table.add_row(
                gr.get("name", "?"),
                result_display,
                f"{score:.3f}",
                f"{weight:.1f}",
            )

        console.print(grade_table)

    console.print()


@cli.command(name="import")
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["auto", "pi", "otlp", "claude"]), default="auto",
    help="Trace format (default: auto-detect)",
)
@click.option(
    "--output", "-o", type=click.Path(),
    help="Save reconstructed transcript(s). Use a directory for multi-trace sources.",
)
@click.option(
    "--output-format", type=click.Choice(["json", "jsonl"]), default="json",
    help="Saved transcript format",
)
@click.option("--json", "as_json", is_flag=True, help="Print reconstructed transcript(s) as JSON")
def import_(source: str, fmt: str, output: str | None, output_format: str, as_json: bool):
    """Import an external agent trace into Compass transcript(s).

    SOURCE is an offline trace file (or a directory of pi sessions):

    \b
      pi      a pi (@earendil-works/pi-*) JSONL session file, or a directory of them
      otlp    an OTLP / OpenInference trace JSON (LangChain / LlamaIndex / CrewAI …
              exported via Arize Phoenix); a single file may hold many traces
      claude  a Claude Code `--output-format stream-json` file

    Format is auto-detected by default. The OpenAI Agents SDK integration is
    live/streaming (used programmatically via ``compass.integrations``); the
    Claude Agent SDK can be imported from its stream-json output.

    Examples:
        compass import session.jsonl                  # auto-detect + summarize
        compass import phoenix_export.json -o out/    # one saved file per trace
        compass import run.stream.jsonl -f claude -o t.json
    """
    import json as json_mod

    from compass.integrations import (
        import_claude_stream_json,
        import_otlp_file,
        import_pi_session,
        import_pi_sessions,
    )

    src = Path(source)
    detected = fmt if fmt != "auto" else _detect_trace_format(src)
    if detected is None:
        console.print(
            "[red]Could not auto-detect trace format.[/red] "
            "Pass [bold]--format pi|otlp|claude[/bold]."
        )
        sys.exit(1)

    try:
        if detected == "pi":
            transcripts = (
                import_pi_sessions(src) if src.is_dir() else [import_pi_session(src)]
            )
        elif detected == "claude":
            transcripts = [import_claude_stream_json(src)]
        else:  # otlp
            transcripts = import_otlp_file(src)
    except Exception as e:
        console.print(f"[red]Failed to import trace: {e}[/red]")
        sys.exit(1)

    if not transcripts:
        console.print("[yellow]No transcripts reconstructed from source.[/yellow]")
        return

    console.print(Panel(
        f"[bold]Compass Import[/bold]  |  format: {detected}  |  "
        f"{len(transcripts)} transcript(s)"
    ))

    if as_json:
        payload = [t.to_dict() for t in transcripts]
        console.print_json(
            json_mod.dumps(payload if len(payload) > 1 else payload[0], ensure_ascii=False)
        )

    _print_import_summary(transcripts)

    if output:
        saved = _save_transcripts(transcripts, Path(output), output_format)
        console.print(f"\n[green]Saved {len(saved)} transcript(s):[/green]")
        for p in saved:
            console.print(f"  {p}")
        console.print(f"\n[dim]View with:[/dim] compass trace {saved[0]}")


_CLAUDE_WIRE_TYPES = frozenset(
    {"assistant", "user", "result", "system", "stream_event", "rate_limit_event"}
)


def _detect_trace_format(path: Path) -> str | None:
    """Sniff the trace format from a file (or directory) without full parsing.

    Discriminate by each format's first-line invariant:
      - pi:     a ``{"type": "session", ...}`` session header
      - claude: a stream-json line whose ``type`` is a Claude Code message type
      - otlp:   anything else that is JSON (OTLP / OpenInference)
    """
    import json as json_mod

    if path.is_dir():
        return "pi"  # import_pi_sessions globs the directory for sessions

    first_line = ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    first_line = line.strip()
                    break
    except OSError:
        return None

    try:
        obj = json_mod.loads(first_line)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        if obj.get("type") == "session":
            return "pi"
        if obj.get("type") in _CLAUDE_WIRE_TYPES:
            return "claude"
    if first_line[:1] in ("{", "["):
        return "otlp"
    return None


def _print_import_summary(transcripts) -> None:
    """Print a compact per-transcript summary table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("#", justify="right", width=3)
    table.add_column("Trial / Task", style="cyan")
    table.add_column("Calls", justify="right")
    table.add_column("Tokens", justify="right")
    table.add_column("Cost", justify="right")
    table.add_column("Final output", overflow="fold")

    for i, t in enumerate(transcripts, 1):
        tokens = t.sum_tokens().total_tokens
        cost = t.sum_cost().total_usd
        final = ""
        if t.outcome and t.outcome.output_data:
            fo = t.outcome.output_data.get("final_output")
            if isinstance(fo, str):
                final = fo[:60] + ("…" if len(fo) > 60 else "")
        table.add_row(
            str(i),
            f"{t.trial_id or '-'}\n[dim]{t.task_id}[/dim]",
            str(len(t.tool_calls)),
            str(tokens) if tokens else "-",
            f"${cost:.4f}" if cost else "-",
            final or "[dim](none)[/dim]",
        )
    console.print(table)


def _save_transcripts(transcripts, output: Path, output_format: str) -> list:
    """Save transcript(s) to a file or directory; return the written paths."""
    import re

    def _write(t, path: Path) -> None:
        if output_format == "jsonl":
            t.save_jsonl(path)
        else:
            t.save(path)

    # Single transcript to an explicit file target.
    if len(transcripts) == 1 and output.suffix in (".json", ".jsonl"):
        _write(transcripts[0], output)
        return [output]

    # Otherwise treat the target as a directory, one file per transcript.
    ext = ".jsonl" if output_format == "jsonl" else ".json"
    output.mkdir(parents=True, exist_ok=True)
    saved = []
    for t in transcripts:
        stem = re.sub(r"[^A-Za-z0-9._-]", "_", t.trial_id or t.task_id or "transcript")[:100]
        path = output / f"{stem or 'transcript'}{ext}"
        _write(t, path)
        saved.append(path)
    return saved


def _print_result_details(result):
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


def _print_summary_table(results):
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
            r.scenario_name,
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


@cli.group()
def baseline():
    """Manage regression testing baselines."""
    pass


@baseline.command("set")
@click.argument("trace_dir", type=click.Path(exists=True))
@click.argument("case_id")
def baseline_set(trace_dir: str, case_id: str):
    """Set a case's current artifacts as baseline.

    Copies the artifacts from the latest trial of CASE_ID into
    the baselines directory under TRACE_DIR.
    """
    from compass.core.artifact_store import ArtifactStore

    store = ArtifactStore(Path(trace_dir))
    try:
        baseline_dir = store.set_baseline(case_id)
        console.print(f"[green]Baseline set:[/green] {baseline_dir}")
    except FileNotFoundError as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@baseline.command("compare")
@click.argument("trace_dir", type=click.Path(exists=True))
@click.argument("case_id")
def baseline_compare(trace_dir: str, case_id: str):
    """Compare current artifacts with baseline for CASE_ID."""
    from compass.core.artifact_store import ArtifactStore

    store = ArtifactStore(Path(trace_dir))
    result = store.compare_with_baseline(case_id)

    if result.identical:
        console.print(f"[green]IDENTICAL[/green] — hash match for '{case_id}'")
    else:
        console.print(f"[yellow]DIFFERENT[/yellow] — artifacts differ for '{case_id}'")

    if result.hash_match is not None:
        console.print(f"  Hash match: {result.hash_match}")
    if result.details:
        for key, val in result.details.items():
            console.print(f"  {key}: {val}")


@baseline.command("list")
@click.argument("trace_dir", type=click.Path(exists=True))
def baseline_list(trace_dir: str):
    """List all baselines in TRACE_DIR."""
    baselines_dir = Path(trace_dir) / "baselines"
    if not baselines_dir.is_dir():
        console.print("[yellow]No baselines found[/yellow]")
        return

    found = False
    for case_dir in sorted(baselines_dir.iterdir()):
        if not case_dir.is_dir():
            continue
        manifest_path = case_dir / "manifest.json"
        if manifest_path.exists():
            import json as json_mod

            manifest = json_mod.loads(manifest_path.read_text(encoding="utf-8"))
            case_id = manifest.get("case_id", case_dir.name)
            trial_id = manifest.get("trial_id", "?")
            timestamp = manifest.get("timestamp", "?")
            n_artifacts = len(manifest.get("artifacts", []))
            console.print(
                f"  [cyan]{case_id}[/cyan]  trial={trial_id}  "
                f"artifacts={n_artifacts}  ts={timestamp}"
            )
            found = True

    if not found:
        console.print("[yellow]No baselines found[/yellow]")


@cli.group()
def checkpoint():
    """Manage evaluation checkpoints."""
    pass


@checkpoint.command("list")
@click.argument("search_dir", type=click.Path(exists=True), default=".")
@click.option("--no-recursive", is_flag=True, help="Only search the given directory, not subdirectories")
def checkpoint_list(search_dir: str, no_recursive: bool):
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
@click.option("--all", "clean_all", is_flag=True, help="Remove ALL checkpoints, not just completed ones")
@click.option("--force", "-f", is_flag=True, help="Skip confirmation prompt")
@click.option("--no-recursive", is_flag=True, help="Only search the given directory, not subdirectories")
def checkpoint_clean(search_dir: str, clean_all: bool, force: bool, no_recursive: bool):
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
    targets: list = []
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

        status_str = "[green]completed[/green]" if cp.status == "completed" else "[yellow]in_progress[/yellow]"
        console.print(
            f"  {display_dir}  run={cp.run_id}  "
            f"scenario={cp.scenario_name}  status={status_str}  progress={cp.progress}"
        )

    # Confirm
    if not force:
        click.confirm("\nRemove these checkpoints?", abort=True)

    # Clean
    total_deleted = 0
    for store, cp in targets:
        try:
            deleted = store.clean()
            total_deleted += deleted
        except Exception as e:
            console.print(f"  [red]Error cleaning {store.dir}: {e}[/red]")

    console.print(f"\n[green]Cleaned {len(targets)} checkpoint(s) ({total_deleted} files removed)[/green]")


@cli.group()
def site():
    """Publish results as a shareable static site."""
    pass


@site.command("build")
@click.argument("results", type=click.Path(exists=True), nargs=-1, required=True)
@click.option("--output", "-o", type=click.Path(), default="site", show_default=True,
              help="Site directory; merged into if it already exists")
@click.option("--slug", help="Name to index this run under (default: derived from the results)")
@click.option("--name", help="Display name for the run (default: the slug)")
@click.option("--trace-dir", type=click.Path(exists=True),
              help="Publish these transcripts alongside the run (they carry prompts and outputs)")
@click.option("--include-details", is_flag=True,
              help="Publish grader details/metadata too (redacted by default)")
@click.option("--history", default=DEFAULT_HISTORY, show_default=True,
              help="Previous builds of this slug to remember for the trend line")
def site_build(
    results: tuple[str, ...],
    output: str,
    slug: str | None,
    name: str | None,
    trace_dir: str | None,
    include_details: bool,
    history: int,
):
    """Build (or refresh) a run in a static site.

    RESULTS is a results JSON written by `compass test --report json` or
    `compass grade -o` — the same shapes `analyze` and `compare` read.

    Only the named run is written; every other run already in the site is left
    alone. That is what lets separate repositories build into one shared
    directory (a gh-pages branch, a bucket prefix) and have the index
    accumulate — no server, no database.

    \b
      compass site build results.json -o site/
      compass site build results.json -o site/ --slug image-evals --trace-dir traces/
      python -m http.server -d site/          # browsers will not fetch from file://

    Grader details are left out unless --include-details: they are free-form
    and routinely hold model output. Transcripts are opt-in the same way —
    they are published only when --trace-dir names them.
    """
    import json

    from compass.report.site import build_site, collect_run_payload, slugify

    if slug and len(results) > 1:
        console.print("[red]--slug takes a single results file[/red]")
        sys.exit(1)

    built = []
    for path_str in results:
        path = Path(path_str)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            console.print(f"[red]Could not read {path}: {e}[/red]")
            sys.exit(1)

        run_slug = slugify(slug or path.stem)
        doc = collect_run_payload(payload, name=name or slug or path.stem)
        if not doc["run"]["total_cases"]:
            console.print(f"[yellow]No cases found in {path} — skipped[/yellow]")
            continue

        result = build_site(
            doc,
            output,
            slug=run_slug,
            trace_dir=trace_dir,
            include_details=include_details,
            history=history,
        )
        built.append(result)

    if not built:
        console.print("[red]Nothing to publish[/red]")
        sys.exit(1)

    last = built[-1]
    table = Table(title=f"Published to {last.site_dir}")
    table.add_column("Run", style="bold cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Pass Rate", justify="right")
    table.add_column("Traces", justify="right")
    table.add_column("History", justify="right")
    for result in built:
        entry = result.entry
        table.add_row(
            result.slug,
            f"{entry['passed_cases']}/{entry['evaluated_cases']}",
            f"{entry['pass_rate'] * 100:.1f}%",
            _format_bytes(result.trace_bytes) if result.trace_files else "—",
            str(result.history) if result.history else "—",
        )
    console.print(table)

    console.print(f"[green]{last.runs} run(s) in this site[/green]")
    if not include_details:
        console.print(
            "[dim]Grader details redacted — rebuild with --include-details to keep them.[/dim]"
        )
    if last.trace_files:
        console.print(
            f"[yellow]Published {last.trace_files} trace file(s): "
            f"transcripts carry prompts and model output.[/yellow]"
        )
    console.print(f"[dim]Preview with:[/dim] python -m http.server -d {last.site_dir}")


def _format_bytes(size: int) -> str:
    """Human-readable byte count — the site's weight has to be visible."""
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f}{unit}" if unit == "B" else f"{value:.1f}{unit}"
        value /= 1024
    return f"{value:.1f}GB"


if __name__ == "__main__":
    cli()
