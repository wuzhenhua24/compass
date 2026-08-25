"""``compass analyze`` — the Transcript/Outcome breakdown of a run."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel

from compass.cli.app import cli, console
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult, iter_case_dicts
from compass.report.console import ConsoleReporter

if TYPE_CHECKING:
    pass


@cli.command()
@click.argument("results_path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Save report as JSON")
def analyze(results_path: str, output: str | None) -> None:
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
