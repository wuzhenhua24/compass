"""HTML report generator."""

from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

from compass.core.result import CaseResult, EvalResult


class HTMLReporter:
    """Generates HTML test reports."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize reporter.

        Args:
            config: Reporter configuration.
        """
        self.config = config or {}
        self.title = self.config.get("title", "Compass Test Report")

    def generate(self, results: list[EvalResult], output_path: str | Path) -> Path:
        """Generate HTML report.

        Args:
            results: List of evaluation results.
            output_path: Path to save the report.

        Returns:
            Path to generated report.
        """
        output_path = Path(output_path)
        html_content = self._render(results)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        return output_path

    def _render(self, results: list[EvalResult]) -> str:
        """Render results to HTML string."""
        # Calculate summary statistics
        total_scenarios = len(results)
        total_cases = sum(r.total_cases for r in results)
        total_passed = sum(r.passed_cases for r in results)
        total_failed = sum(r.failed_cases for r in results)
        total_errors = sum(r.error_cases for r in results)
        overall_pass_rate = (total_passed / total_cases * 100) if total_cases > 0 else 0

        # Compute per-case scope scores for scatter chart
        all_cases: list[CaseResult] = []
        for r in results:
            all_cases.extend(r.case_results)
        scatter_data = [self._compute_case_scope_scores(c) for c in all_cases]

        # Compute category breakdown
        category_html = self._render_category_breakdown(all_cases)

        # Only render scatter chart when both axes have data
        has_outcome_graders = (
            any(d["outcome_score"] > 0 for d in scatter_data) if scatter_data else False
        )
        has_transcript_graders = (
            any(d["transcript_score"] > 0 for d in scatter_data) if scatter_data else False
        )
        scatter_chart_html = ""
        if has_outcome_graders and has_transcript_graders:
            scatter_chart_html = self._render_dual_axis_chart(scatter_data)

        # Generate scenario cards
        scenario_cards = "\n".join(
            self._render_scenario_card(r) for r in results
        )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            margin-bottom: 30px;
        }}
        header h1 {{
            font-size: 2rem;
            margin-bottom: 10px;
        }}
        header .timestamp {{
            opacity: 0.8;
            font-size: 0.9rem;
        }}
        .summary {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }}
        .summary-card .value {{
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .summary-card .label {{
            color: #666;
            font-size: 0.9rem;
        }}
        .summary-card.passed .value {{ color: #10b981; }}
        .summary-card.failed .value {{ color: #ef4444; }}
        .summary-card.error .value {{ color: #f59e0b; }}
        .scenario-card {{
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
            overflow: hidden;
        }}
        .scenario-header {{
            padding: 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .scenario-header h2 {{
            font-size: 1.2rem;
        }}
        .badge {{
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.85rem;
            font-weight: 500;
        }}
        .badge.passed {{ background: #d1fae5; color: #065f46; }}
        .badge.failed {{ background: #fee2e2; color: #991b1b; }}
        .case-list {{
            padding: 0;
        }}
        .case-item {{
            padding: 15px 20px;
            border-bottom: 1px solid #eee;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .case-item:last-child {{
            border-bottom: none;
        }}
        .case-info {{
            flex: 1;
        }}
        .case-id {{
            font-weight: 500;
            margin-bottom: 5px;
        }}
        .case-meta {{
            font-size: 0.85rem;
            color: #666;
        }}
        .score-bar {{
            width: 100px;
            height: 8px;
            background: #eee;
            border-radius: 4px;
            overflow: hidden;
            margin-right: 15px;
        }}
        .score-fill {{
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }}
        .score-fill.good {{ background: #10b981; }}
        .score-fill.medium {{ background: #f59e0b; }}
        .score-fill.poor {{ background: #ef4444; }}
        .status-icon {{
            width: 24px;
            height: 24px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }}
        .status-icon.passed {{ background: #d1fae5; color: #065f46; }}
        .status-icon.failed {{ background: #fee2e2; color: #991b1b; }}
        .status-icon.error {{ background: #fef3c7; color: #92400e; }}
    </style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{self.title}</h1>
            <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
    </header>

    <div class="container">
        <div class="summary">
            <div class="summary-card">
                <div class="value">{total_scenarios}</div>
                <div class="label">Scenarios</div>
            </div>
            <div class="summary-card passed">
                <div class="value">{total_passed}</div>
                <div class="label">Passed</div>
            </div>
            <div class="summary-card failed">
                <div class="value">{total_failed}</div>
                <div class="label">Failed</div>
            </div>
            <div class="summary-card error">
                <div class="value">{total_errors}</div>
                <div class="label">Errors</div>
            </div>
            <div class="summary-card">
                <div class="value">{overall_pass_rate:.1f}%</div>
                <div class="label">Pass Rate</div>
            </div>
        </div>

        {category_html}

        {scatter_chart_html}

        {scenario_cards}
    </div>
</body>
</html>"""

    @staticmethod
    def _compute_case_scope_scores(case: CaseResult) -> dict[str, Any]:
        """Compute per-case outcome_score and transcript_score from evaluator results.

        Groups evaluator_results by grader_scope and averages each axis.
        """
        outcome_scores: list[float] = []
        transcript_scores: list[float] = []

        for er in case.evaluator_results:
            scope = er.grader_scope  # string: "outcome", "transcript", or "both"
            if scope in ("outcome", "both"):
                outcome_scores.append(er.score)
            if scope in ("transcript", "both"):
                transcript_scores.append(er.score)

        return {
            "case_id": case.case_id,
            "outcome_score": (
                sum(outcome_scores) / len(outcome_scores) if outcome_scores else 0.0
            ),
            "transcript_score": (
                sum(transcript_scores) / len(transcript_scores) if transcript_scores else 0.0
            ),
            "passed": case.passed,
        }

    @staticmethod
    def _render_category_breakdown(cases: list[CaseResult]) -> str:
        """Render an HTML table of per-category statistics."""
        from collections import defaultdict

        buckets: dict[str, list[CaseResult]] = defaultdict(list)
        for c in cases:
            if c.category:
                buckets[c.category].append(c)

        if not buckets:
            return ""

        rows = ""
        for cat in sorted(buckets):
            cat_cases = buckets[cat]
            total = len(cat_cases)
            passed = sum(1 for c in cat_cases if c.passed)
            failed = total - passed
            rate = passed / total * 100 if total > 0 else 0
            avg = sum(c.overall_score for c in cat_cases) / total if total else 0
            rate_color = "#10b981" if rate >= 70 else "#f59e0b" if rate >= 50 else "#ef4444"
            rows += f"""
                <tr>
                    <td style="font-weight:500">{escape(cat)}</td>
                    <td style="text-align:center">{total}</td>
                    <td style="text-align:center;color:#10b981">{passed}</td>
                    <td style="text-align:center;color:#ef4444">{failed}</td>
                    <td style="text-align:center;color:{rate_color};font-weight:600">
                        {rate:.1f}%</td>
                    <td style="text-align:center">{avg:.2f}</td>
                </tr>"""

        return f"""
        <div class="scenario-card" style="padding:20px;margin-bottom:20px">
            <h2 style="margin-bottom:15px;font-size:1.2rem">Category Breakdown</h2>
            <table style="width:100%;border-collapse:collapse">
                <thead>
                    <tr style="border-bottom:2px solid #eee">
                        <th style="text-align:left;padding:8px">Category</th>
                        <th style="text-align:center;padding:8px">Total</th>
                        <th style="text-align:center;padding:8px">Passed</th>
                        <th style="text-align:center;padding:8px">Failed</th>
                        <th style="text-align:center;padding:8px">Pass Rate</th>
                        <th style="text-align:center;padding:8px">Avg Score</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>
        </div>"""

    def _render_dual_axis_chart(self, scatter_data: list[dict[str, Any]]) -> str:
        """Generate an inline SVG scatter plot of Outcome (X) vs Transcript (Y)."""
        # SVG dimensions
        w, h = 500, 500
        pad_left, pad_bottom, pad_top, pad_right = 60, 60, 30, 30
        plot_w = w - pad_left - pad_right
        plot_h = h - pad_top - pad_bottom
        threshold = 0.7

        def sx(v: float) -> float:
            return pad_left + v * plot_w

        def sy(v: float) -> float:
            return pad_top + (1.0 - v) * plot_h

        # Quadrant background rects
        tx = sx(threshold)
        ty = sy(threshold)
        quadrants = (
            # top-right: both good (green)
            f'<rect x="{tx}" y="{pad_top}" '
            f'width="{sx(1.0) - tx}" height="{ty - pad_top}" '
            f'fill="#d1fae5" opacity="0.4"/>'
            # top-left: process good result bad (yellow)
            f'<rect x="{pad_left}" y="{pad_top}" '
            f'width="{tx - pad_left}" height="{ty - pad_top}" '
            f'fill="#fef3c7" opacity="0.4"/>'
            # bottom-right: result good process bad (yellow)
            f'<rect x="{tx}" y="{ty}" '
            f'width="{sx(1.0) - tx}" height="{sy(0.0) - ty}" '
            f'fill="#fef3c7" opacity="0.4"/>'
            # bottom-left: both bad (red)
            f'<rect x="{pad_left}" y="{ty}" '
            f'width="{tx - pad_left}" height="{sy(0.0) - ty}" '
            f'fill="#fee2e2" opacity="0.4"/>'
        )

        # Axes
        axes = (
            f'<line x1="{pad_left}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(0)}" '
            f'stroke="#333" stroke-width="1.5"/>'
            f'<line x1="{pad_left}" y1="{sy(0)}" x2="{pad_left}" y2="{sy(1)}" '
            f'stroke="#333" stroke-width="1.5"/>'
        )

        # Threshold dashed lines
        dashes = (
            f'<line x1="{pad_left}" y1="{ty}" x2="{sx(1)}" y2="{ty}" '
            f'stroke="#999" stroke-width="1" stroke-dasharray="5,5"/>'
            f'<line x1="{tx}" y1="{sy(0)}" x2="{tx}" y2="{sy(1)}" '
            f'stroke="#999" stroke-width="1" stroke-dasharray="5,5"/>'
        )

        # Diagonal reference line (y=x)
        diag = (
            f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" '
            f'stroke="#ccc" stroke-width="1" stroke-dasharray="3,3"/>'
        )

        # Tick marks and labels
        ticks = ""
        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # X-axis ticks
            ticks += (
                f'<line x1="{sx(v)}" y1="{sy(0)}" x2="{sx(v)}" y2="{sy(0) + 5}" '
                f'stroke="#333" stroke-width="1"/>'
                f'<text x="{sx(v)}" y="{sy(0) + 20}" text-anchor="middle" '
                f'font-size="11" fill="#666">{v:.1f}</text>'
            )
            # Y-axis ticks
            ticks += (
                f'<line x1="{pad_left - 5}" y1="{sy(v)}" x2="{pad_left}" y2="{sy(v)}" '
                f'stroke="#333" stroke-width="1"/>'
                f'<text x="{pad_left - 10}" y="{sy(v) + 4}" text-anchor="end" '
                f'font-size="11" fill="#666">{v:.1f}</text>'
            )

        # Axis labels
        labels = (
            f'<text x="{pad_left + plot_w / 2}" y="{h - 10}" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="#333">Outcome Score</text>'
            f'<text x="15" y="{pad_top + plot_h / 2}" text-anchor="middle" '
            f'font-size="13" font-weight="bold" fill="#333" '
            f'transform="rotate(-90, 15, {pad_top + plot_h / 2})">Transcript Score</text>'
        )

        # Data points
        points = ""
        for d in scatter_data:
            cx = sx(d["outcome_score"])
            cy = sy(d["transcript_score"])
            color = "#10b981" if d["passed"] else "#ef4444"
            title = escape(
                f'{d["case_id"]}: outcome={d["outcome_score"]:.2f}, '
                f'transcript={d["transcript_score"]:.2f}'
            )
            points += (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" '
                f'fill="{color}" opacity="0.8" stroke="white" stroke-width="1.5">'
                f'<title>{title}</title></circle>'
            )

        # Legend
        legend_y = h - 10
        legend = (
            f'<circle cx="{sx(1) - 130}" cy="{legend_y - 3}" r="5" fill="#10b981"/>'
            f'<text x="{sx(1) - 120}" y="{legend_y}" font-size="11" fill="#333">Passed</text>'
            f'<circle cx="{sx(1) - 60}" cy="{legend_y - 3}" r="5" fill="#ef4444"/>'
            f'<text x="{sx(1) - 50}" y="{legend_y}" font-size="11" fill="#333">Failed</text>'
        )

        svg = (
            f'<svg width="{w}" height="{h}" xmlns="http://www.w3.org/2000/svg" '
            f'style="font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
            f'{quadrants}{axes}{dashes}{diag}{ticks}{labels}{points}{legend}'
            f'</svg>'
        )

        return f"""
        <div class="scenario-card" style="padding: 20px; margin-bottom: 20px;">
            <h2 style="margin-bottom: 15px; font-size: 1.2rem;">
                Outcome vs Transcript Scatter Plot</h2>
            <div style="text-align: center;">{svg}</div>
        </div>"""

    def _render_scenario_card(self, result: EvalResult) -> str:
        """Render a single scenario card."""
        pass_rate = result.pass_rate * 100
        status_class = "passed" if pass_rate >= 70 else "failed"

        case_items = "\n".join(
            self._render_case_item(case) for case in result.case_results
        )

        return f"""
        <div class="scenario-card">
            <div class="scenario-header">
                <h2>{result.scenario_name}</h2>
                <span class="badge {status_class}">{result.passed_cases}/{result.total_cases} passed</span>
            </div>
            <div class="case-list">
                {case_items}
            </div>
        </div>"""

    def _render_case_item(self, case) -> str:
        """Render a single case item."""
        status_class = case.status.value
        status_icon = (
            "✓" if case.passed
            else "✗" if status_class == "failed" else "!"
        )

        # Build score meta text
        meta_parts = [f"Avg: {case.overall_score:.2f}"]
        if case.has_trials:
            meta_parts.append(f"Best: {case.best_score:.2f}")
            meta_parts.append(
                f"Trials: {case.passed_trials}/{case.total_trials}"
            )
        meta_parts.append(f"Duration: {case.duration_ms:.0f}ms")
        meta_text = " | ".join(meta_parts)

        # Use best_score for the bar when multi-trial
        bar_score = case.best_score if case.has_trials else case.overall_score
        bar_percent = bar_score * 100
        bar_class = (
            "good" if bar_percent >= 70
            else "medium" if bar_percent >= 40 else "poor"
        )

        return f"""
            <div class="case-item">
                <div class="case-info">
                    <div class="case-id">{case.case_id}</div>
                    <div class="case-meta">{meta_text}</div>
                </div>
                <div class="score-bar">
                    <div class="score-fill {bar_class}"
                         style="width: {bar_percent}%"></div>
                </div>
                <div class="status-icon {status_class}">{status_icon}</div>
            </div>"""
