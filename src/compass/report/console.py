"""Rich console visualization for analysis reports.

Uses the `rich` library (already a project dependency) to render
analysis reports as formatted terminal output with tables, panels,
and color-coded indicators.
"""

from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from compass.report.analyzer import AnalysisReport


class ConsoleReporter:
    """Renders AnalysisReport to the terminal using Rich."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def render(self, report: AnalysisReport) -> None:
        """Render the full analysis report to the console."""
        self.console.print()
        self._render_summary(report.summary)
        self._render_category_analysis(report.category_analysis)
        self._render_tag_analysis(report.tag_analysis)
        self._render_scope_comparison(report.scope_comparison)
        self._render_dual_axis_chart(report.dual_axis_data)
        self._render_grader_analysis(
            "Transcript Graders", report.transcript_analysis,
        )
        self._render_grader_analysis(
            "Outcome Graders", report.outcome_analysis,
        )
        self._render_observed_tags(report.observed_tag_analysis)
        self._render_failure_patterns(report.failure_patterns)
        self._render_failure_tags(report.failure_tag_analysis)
        self._render_recommendations(report.recommendations)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _render_summary(self, summary: dict) -> None:
        """Render summary statistics panel."""
        total = summary.get("total_tasks", 0)
        passed = summary.get("passed_tasks", 0)
        pass_rate = summary.get("pass_rate", 0.0)
        avg_t = summary.get("avg_transcript_score", 0.0)
        avg_o = summary.get("avg_outcome_score", 0.0)
        avg_dur = summary.get("avg_duration_ms", 0.0)

        rate_color = "green" if pass_rate >= 0.7 else "yellow" if pass_rate >= 0.5 else "red"

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("metric", style="bold")
        table.add_column("value", justify="right")

        table.add_row("Total Tasks", str(total))
        table.add_row("Passed", f"[green]{passed}[/green]")
        table.add_row("Failed", f"[red]{total - passed}[/red]")
        table.add_row("Pass Rate", f"[{rate_color}]{pass_rate:.1%}[/{rate_color}]")
        table.add_row("Avg Transcript Score", f"{avg_t:.3f}")
        table.add_row("Avg Outcome Score", f"{avg_o:.3f}")

        # Best-of-k metrics (only shown when multi-trial data exists)
        avg_best = summary.get("avg_best_score")
        if avg_best is not None:
            avg_score = summary.get("avg_score", 0.0)
            gap = summary.get("best_vs_avg_gap", 0.0)
            table.add_row("Avg Score (mean)", f"{avg_score:.3f}")
            table.add_row(
                "Avg Score (best-of-k)",
                f"[bold]{avg_best:.3f}[/bold]",
            )
            gap_color = (
                "green" if gap < 0.05
                else "yellow" if gap < 0.15
                else "red"
            )
            table.add_row(
                "Best vs Avg Gap",
                f"[{gap_color}]+{gap:.3f}[/{gap_color}]",
            )

        table.add_row("Avg Duration", f"{avg_dur:.0f}ms")

        self.console.print(Panel(table, title="Summary", border_style="blue"))

    # ------------------------------------------------------------------
    # Category analysis
    # ------------------------------------------------------------------

    def _render_category_analysis(
        self, category_analysis: dict[str, dict],
    ) -> None:
        """Render per-category breakdown table."""
        if not category_analysis:
            return

        table = Table(title="Category Breakdown", border_style="blue")
        table.add_column("Category", style="bold")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Outcome", justify="right")
        table.add_column("Transcript", justify="right")

        for cat, stats in category_analysis.items():
            rate = stats.get("pass_rate", 0.0)
            rate_color = (
                "green" if rate >= 0.7 else "yellow" if rate >= 0.5 else "red"
            )
            table.add_row(
                cat,
                str(stats.get("total", 0)),
                f"[green]{stats.get('passed', 0)}[/green]",
                f"[red]{stats.get('failed', 0)}[/red]",
                f"[{rate_color}]{rate:.1%}[/{rate_color}]",
                f"{stats.get('avg_score', 0.0):.3f}",
                f"{stats.get('avg_outcome_score', 0.0):.3f}",
                f"{stats.get('avg_transcript_score', 0.0):.3f}",
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Tag analysis
    # ------------------------------------------------------------------

    def _render_tag_analysis(
        self, tag_analysis: dict[str, dict],
    ) -> None:
        """Render per-tag breakdown table."""
        if not tag_analysis:
            return

        table = Table(title="Tag Breakdown", border_style="cyan")
        table.add_column("Tag", style="bold")
        table.add_column("Total", justify="right")
        table.add_column("Passed", justify="right")
        table.add_column("Failed", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Score", justify="right")

        for tag, stats in tag_analysis.items():
            rate = stats.get("pass_rate", 0.0)
            rate_color = (
                "green" if rate >= 0.7 else "yellow" if rate >= 0.5 else "red"
            )
            table.add_row(
                tag,
                str(stats.get("total", 0)),
                f"[green]{stats.get('passed', 0)}[/green]",
                f"[red]{stats.get('failed', 0)}[/red]",
                f"[{rate_color}]{rate:.1%}[/{rate_color}]",
                f"{stats.get('avg_score', 0.0):.3f}",
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Scope comparison
    # ------------------------------------------------------------------

    def _render_scope_comparison(self, comparison: dict) -> None:
        """Render Transcript vs Outcome comparison panel."""
        avg_t = comparison.get("avg_transcript_score", 0.0)
        avg_o = comparison.get("avg_outcome_score", 0.0)
        gap = comparison.get("gap", 0.0)
        description = comparison.get("description", "")

        bar_width = 30

        t_filled = int(avg_t * bar_width)
        o_filled = int(avg_o * bar_width)

        t_bar = self._score_bar(avg_t, bar_width)
        o_bar = self._score_bar(avg_o, bar_width)

        gap_color = "green" if abs(gap) < 0.1 else "yellow" if abs(gap) < 0.2 else "red"

        lines = Text()
        lines.append("Transcript  ", style="bold")
        lines.append(t_bar)
        lines.append(f"  {avg_t:.3f}\n")
        lines.append("Outcome     ", style="bold")
        lines.append(o_bar)
        lines.append(f"  {avg_o:.3f}\n")
        lines.append("Gap         ", style="bold")
        lines.append(f"{gap:+.3f}", style=gap_color)
        lines.append(f"\n\n{description}")

        self.console.print(Panel(
            lines,
            title="Transcript vs Outcome",
            border_style="cyan",
        ))

    # ------------------------------------------------------------------
    # Per-grader analysis
    # ------------------------------------------------------------------

    def _render_grader_analysis(
        self, title: str, analysis: dict[str, dict],
    ) -> None:
        """Render per-grader analysis table."""
        if not analysis:
            return

        table = Table(title=title, border_style="dim")
        table.add_column("Grader", style="bold")
        table.add_column("Type")
        table.add_column("Pass Rate", justify="right")
        table.add_column("Avg Score", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("P/F", justify="right")
        table.add_column("Status", justify="center")

        for name, stats in analysis.items():
            rate = stats.get("pass_rate", 0.0)
            avg = stats.get("avg_score", 0.0)
            std = stats.get("score_std", 0.0)
            passed = stats.get("passed", 0)
            failed = stats.get("failed", 0)
            is_bottleneck = stats.get("is_bottleneck", False)
            is_gate = stats.get("is_gate", False)

            rate_color = "green" if rate >= 0.7 else "yellow" if rate >= 0.5 else "red"
            if is_gate:
                status = "[magenta]GATE[/magenta]"
            elif is_bottleneck:
                status = "[red]BOTTLENECK[/red]"
            else:
                status = "[green]OK[/green]"

            display_name = f"[magenta]{name}[/magenta]" if is_gate else name

            table.add_row(
                display_name,
                stats.get("grader_type", ""),
                f"[{rate_color}]{rate:.1%}[/{rate_color}]",
                f"{avg:.3f}",
                f"{std:.3f}",
                f"[green]{passed}[/green]/[red]{failed}[/red]",
                status,
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Failure patterns
    # ------------------------------------------------------------------

    def _render_failure_patterns(self, patterns: list[dict]) -> None:
        """Render failure pattern table."""
        if not patterns:
            self.console.print(
                Panel("[green]No failure patterns detected[/green]",
                      title="Failure Patterns", border_style="green"),
            )
            return

        table = Table(title="Top Failure Patterns", border_style="red")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Failed Graders")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("Example Tasks")

        for i, pattern in enumerate(patterns, 1):
            graders = ", ".join(pattern.get("failed_graders", []))
            count = pattern.get("count", 0)
            pct = pattern.get("percentage", 0.0)
            task_ids = pattern.get("task_ids", [])
            examples = ", ".join(task_ids[:3])
            if len(task_ids) > 3:
                examples += "..."

            table.add_row(
                str(i),
                graders,
                str(count),
                f"{pct:.1f}%",
                examples,
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Failure tags
    # ------------------------------------------------------------------

    def _render_observed_tags(self, tag_analysis: dict[str, dict[str, Any]]) -> None:
        """Render the behavioural profile: what graders observed, as shares.

        Counts every task, passing or not — a tag on a successful run carries
        as much information as one on a failure. The per-tag pass rate is the
        actionable column: "the runs tagged X fail more often".
        """
        if not tag_analysis:
            return

        table = Table(title="Observed Tags (presence-only)", border_style="cyan")
        table.add_column("Tag", style="bold")
        table.add_column("Runs", justify="right")
        table.add_column("Share", justify="right")
        table.add_column("Pass Rate", justify="right")
        table.add_column("From Graders")

        for i, (tag, data) in enumerate(tag_analysis.items(), 1):
            if i > 15:
                break
            pass_rate = data.get("pass_rate", 0.0)
            style = "red" if pass_rate < 0.5 else "green" if pass_rate >= 0.9 else ""
            table.add_row(
                tag,
                str(data.get("count", 0)),
                f"{data.get('share', 0.0):.0%}",
                f"[{style}]{pass_rate:.0%}[/{style}]" if style else f"{pass_rate:.0%}",
                ", ".join(data.get("graders", [])),
            )

        self.console.print(table)
        self.console.print(
            "[dim]presence-only：标签未出现表示「未观察到」，不代表「否」[/dim]"
        )

    def _render_failure_tags(self, tag_analysis: dict[str, dict]) -> None:
        """Render failure tag distribution table."""
        if not tag_analysis:
            return

        table = Table(title="Top Failure Tags", border_style="yellow")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Tag", style="bold")
        table.add_column("Count", justify="right")
        table.add_column("Percentage", justify="right")
        table.add_column("From Graders")

        for i, (tag, data) in enumerate(tag_analysis.items(), 1):
            if i > 10:
                break
            graders = ", ".join(data.get("graders", []))
            count = data.get("count", 0)
            pct = data.get("percentage", 0.0)

            table.add_row(
                str(i),
                tag,
                str(count),
                f"{pct:.1f}%",
                graders,
            )

        self.console.print(table)
        self.console.print()

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    def _render_recommendations(self, recommendations: list[str]) -> None:
        """Render recommendations panel."""
        lines = Text()
        for i, rec in enumerate(recommendations, 1):
            lines.append(f"  {i}. ", style="bold yellow")
            lines.append(f"{rec}\n")

        self.console.print(Panel(
            lines,
            title="Recommendations",
            border_style="yellow",
        ))

    # ------------------------------------------------------------------
    # Dual axis scatter plot (ASCII)
    # ------------------------------------------------------------------

    def _render_dual_axis_chart(self, dual_axis_data: list[dict]) -> None:
        """Render an ASCII scatter plot of Outcome (X) vs Transcript (Y) scores."""
        if not dual_axis_data:
            return

        # Check that at least some data has both axes with nonzero values
        has_outcome = any(d["outcome_score"] > 0 for d in dual_axis_data)
        has_transcript = any(d["transcript_score"] > 0 for d in dual_axis_data)
        if not has_outcome or not has_transcript:
            return

        grid_w = 20
        grid_h = 20

        # Build empty grid
        grid: list[list[str | None]] = [
            [None for _ in range(grid_w)] for _ in range(grid_h)
        ]
        # Style grid (None means empty)
        style_grid: list[list[str | None]] = [
            [None for _ in range(grid_w)] for _ in range(grid_h)
        ]

        for d in dual_axis_data:
            ox = d["outcome_score"]
            ty = d["transcript_score"]
            col = min(int(ox * grid_w), grid_w - 1)
            row = min(int(ty * grid_h), grid_h - 1)
            # Invert row so top = high Y
            row = grid_h - 1 - row
            style = "green" if d["passed"] else "red"
            grid[row][col] = "\u25cf"
            style_grid[row][col] = style

        lines = Text()
        # Title line
        lines.append("1.0", style="dim")
        lines.append("\n")

        threshold_row = grid_h - 1 - min(int(0.7 * grid_h), grid_h - 1)
        threshold_col = min(int(0.7 * grid_w), grid_w - 1)

        for r in range(grid_h):
            # Y-axis label at a few rows
            if r == 0:
                lines.append("T  ", style="dim")
            elif r == grid_h - 1:
                lines.append("0  ", style="dim")
            else:
                lines.append("   ")

            lines.append("\u2502" if r != threshold_row else "\u2504", style="dim")

            for c in range(grid_w):
                if grid[r][c] is not None:
                    lines.append(grid[r][c], style=style_grid[r][c])
                elif r == threshold_row and c == threshold_col:
                    lines.append("+", style="dim")
                elif r == threshold_row:
                    lines.append("\u2504", style="dim")
                elif c == threshold_col:
                    lines.append("\u2506", style="dim")
                else:
                    lines.append(" ")
            lines.append("\n")

        # X axis
        lines.append("   \u2514", style="dim")
        lines.append("\u2500" * grid_w, style="dim")
        lines.append("\n")
        lines.append("   0", style="dim")
        lines.append(" " * (grid_w - 5))
        lines.append("1.0  O", style="dim")
        lines.append("\n\n")

        # Legend
        lines.append("  Legend: ", style="bold")
        lines.append("\u25cf", style="green")
        lines.append(" passed  ")
        lines.append("\u25cf", style="red")
        lines.append(" failed")

        self.console.print(Panel(
            lines,
            title="Outcome vs Transcript Scatter",
            border_style="magenta",
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _score_bar(score: float, width: int = 30) -> Text:
        """Create a colored score bar."""
        filled = int(score * width)
        empty = width - filled

        if score >= 0.7:
            color = "green"
        elif score >= 0.5:
            color = "yellow"
        else:
            color = "red"

        bar = Text()
        bar.append("\u2588" * filled, style=color)
        bar.append("\u2591" * empty, style="dim")
        return bar
