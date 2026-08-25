"""``compass trace`` — read one execution transcript."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import click
from rich.panel import Panel
from rich.table import Table

from compass.cli.app import cli, console

if TYPE_CHECKING:
    pass


@cli.command()
@click.argument("trace_file", type=click.Path(exists=True))
@click.option("--steps", "-s", is_flag=True, help="Show tool call input/output details")
@click.option("--json", "as_json", is_flag=True, help="Output raw JSON")
def trace(trace_file: str, steps: bool, as_json: bool) -> None:
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
