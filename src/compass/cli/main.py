"""CLI entry point for Compass."""

import asyncio
import sys
from pathlib import Path
from typing import Optional

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
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult
from compass.report.console import ConsoleReporter


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
    verbose: bool,
):
    """Run test scenarios.

    SCENARIO can be a YAML file or directory containing scenario files.
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

    console.print(Panel(f"[bold]Compass Test Runner[/bold]\n{len(scenarios)} scenario(s) to run"))

    # Run tests
    compass = Compass()
    all_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for scn in scenarios:
            task = progress.add_task(f"Running: {scn.name}", total=None)

            try:
                result = asyncio.run(
                    compass.run(
                        scn,
                        case_ids=list(case) if case else None,
                        stages=list(stage) if stage else None,
                        categories=list(category) if category else None,
                        parallel=parallel,
                        max_workers=workers,
                        trace_dir=trace_dir,
                        trace_format=trace_format,
                        resume=resume,
                    )
                )
                all_results.append(result)

                # Print summary
                status = "[green]PASS[/green]" if result.pass_rate == 1.0 else "[red]FAIL[/red]"
                progress.update(task, description=f"{scn.name}: {status} ({result.passed_cases}/{result.total_cases})")

                if verbose:
                    _print_result_details(result)

            except Exception as e:
                console.print(f"[red]Error running {scn.name}: {e}[/red]")
                progress.update(task, description=f"{scn.name}: [red]ERROR[/red]")

    # Print summary table
    _print_summary_table(all_results)

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
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(report_data, f, ensure_ascii=False, indent=2)
            console.print(f"\n[green]Report generated:[/green] {output_path}")

    # Show trace directory info
    if trace_dir and all_results:
        console.print(f"[green]Traces saved to:[/green] {trace_dir}/ (format: {trace_format})")

    # Exit with appropriate code
    total_passed = sum(r.passed_cases for r in all_results)
    total_cases = sum(r.total_cases for r in all_results)
    sys.exit(0 if total_passed == total_cases else 1)


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
def list():
    """List available graders and adapters."""
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
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save report as JSON")
def analyze(results_path: str, output: Optional[str]):
    """Analyze evaluation results with Transcript/Outcome breakdown.

    RESULTS_PATH is a JSON file or directory containing evaluation result files.
    Each result file should contain a list of task evaluation results.
    """
    import builtins
    import json

    results_file = Path(results_path)

    # Load results
    # Note: builtins.list needed because the `list` CLI command shadows the builtin
    if results_file.is_file():
        json_files = [results_file]
    else:
        json_files = builtins.list(results_file.glob("**/*.json"))

    if not json_files:
        console.print("[red]No result files found[/red]")
        sys.exit(1)

    # Parse into TaskEvalResult objects
    from compass.graders.base import GradeResult, GraderScope, GraderType

    task_results: builtins.list[TaskEvalResult] = []

    for jf in json_files:
        try:
            with open(jf) as f:
                data = json.load(f)

            # Support both single result and list of results
            items = data if isinstance(data, builtins.list) else [data]

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

    table = Table(title="Test Summary")
    table.add_column("Scenario", style="cyan")
    table.add_column("Cases", justify="right")
    table.add_column("Passed", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
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
            pass_rate,
            avg_score,
        ]
        if has_trials:
            row.append(f"{r.best_of_k_score:.3f}")
        row.append(duration)

        table.add_row(*row)

    console.print()
    console.print(table)


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


if __name__ == "__main__":
    cli()
