"""HTML report generator.

A pure renderer: every number it shows comes from a ``compass.report.site``
document, so a report and a published site can never disagree.

They also look the same. The palette, spacing scale and chart styling below are
the ones ``app.html`` ships — same tokens, same values, same dark mode. A local
report and a published one are the same run seen twice, and two visual
languages for that is a lie about how related they are.
"""

from datetime import datetime, timedelta
from html import escape
from pathlib import Path
from typing import Any

from compass.core.result import EvalResult
from compass.report.site import collect_run, iter_cases

#: The design tokens and rules, kept out of the f-string template — CSS braces
#: inside one have to be doubled, which is how a stylesheet becomes unreadable.
_CSS = """
  :root {
    color-scheme: light;
    --bg: oklch(98.5% 0.003 274);
    --surface: oklch(100% 0 0);
    --surface-2: oklch(97.2% 0.005 274);
    --surface-3: oklch(94.6% 0.008 274);
    --border: oklch(91% 0.007 274);
    --border-strong: oklch(84% 0.012 274);
    --ink: oklch(25% 0.02 274);
    --ink-2: oklch(45% 0.017 274);
    --ink-3: oklch(52% 0.015 274);
    --accent: oklch(48% 0.18 274);

    --pass: oklch(58% 0.14 158);
    --pass-bg: oklch(95% 0.05 158);
    --pass-ink: oklch(41% 0.11 158);
    --fail: oklch(57% 0.21 27);
    --fail-bg: oklch(95.5% 0.035 27);
    --fail-ink: oklch(45% 0.18 27);
    --warn: oklch(72% 0.16 72);
    --warn-bg: oklch(96% 0.05 82);
    --warn-ink: oklch(46% 0.11 66);
    --quiet-bg: oklch(96% 0.006 274);
    --quiet-ink: oklch(42% 0.02 274);
    --zone-good: oklch(97.4% 0.024 158);
    --zone-mixed: oklch(97.8% 0.024 85);
    --zone-poor: oklch(97.4% 0.02 27);

    --s1: 4px; --s2: 8px; --s3: 12px; --s4: 16px; --s5: 24px; --s6: 32px;
    --r1: 6px; --r2: 10px;
    --t-xs: .75rem; --t-sm: .8125rem; --t-md: .875rem; --t-lg: 1rem; --t-2xl: 1.5rem;
    --shadow: 0 1px 2px oklch(25% 0.02 274 / .05), 0 1px 3px oklch(25% 0.02 274 / .06);
    --mono: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
  }

  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --bg: oklch(17.5% 0.012 274);
      --surface: oklch(21.5% 0.014 274);
      --surface-2: oklch(25% 0.014 274);
      --surface-3: oklch(29% 0.016 274);
      --border: oklch(31% 0.016 274);
      --border-strong: oklch(39% 0.018 274);
      --ink: oklch(95% 0.006 274);
      --ink-2: oklch(80% 0.012 274);
      --ink-3: oklch(69% 0.014 274);
      --accent: oklch(78% 0.12 274);

      --pass: oklch(74% 0.15 158);
      --pass-bg: oklch(30% 0.07 158);
      --pass-ink: oklch(87% 0.13 158);
      --fail: oklch(70% 0.17 27);
      --fail-bg: oklch(31% 0.09 27);
      --fail-ink: oklch(88% 0.09 27);
      --warn: oklch(80% 0.14 78);
      --warn-bg: oklch(32% 0.07 72);
      --warn-ink: oklch(90% 0.09 82);
      --quiet-bg: oklch(28% 0.012 274);
      --quiet-ink: oklch(84% 0.012 274);
      --zone-good: oklch(24.5% 0.019 158);
      --zone-mixed: oklch(24.5% 0.017 85);
      --zone-poor: oklch(24.5% 0.018 27);
      --shadow: 0 1px 2px oklch(0% 0 0 / .3);
    }
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
      Oxygen, Ubuntu, sans-serif;
    background: var(--bg); color: var(--ink); line-height: 1.55;
    font-size: var(--t-md);
    -webkit-font-smoothing: antialiased;
  }
  .container { max-width: 1200px; margin: 0 auto; padding: var(--s5) var(--s4); }

  header { background: var(--surface); border-bottom: 1px solid var(--border); }
  header .container { padding-top: var(--s5); padding-bottom: var(--s5); }
  header h1 {
    font-size: var(--t-2xl); font-weight: 600; letter-spacing: -.02em;
    line-height: 1.2; text-wrap: balance;
  }
  .timestamp {
    color: var(--ink-3); font-size: var(--t-sm); margin-top: var(--s1);
    display: flex; flex-wrap: wrap; align-items: center; gap: var(--s1) var(--s2);
  }
  .chip {
    font-family: var(--mono); font-size: var(--t-xs); white-space: nowrap;
    background: var(--quiet-bg); color: var(--quiet-ink);
    padding: 2px 8px; border-radius: 20px;
  }

  /* One divided strip, not five floating tiles: these numbers are read across,
     and hairlines carry that better than shadows and gaps. */
  .summary {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1px; background: var(--border);
    border: 1px solid var(--border); border-radius: var(--r2);
    overflow: hidden; margin-bottom: var(--s5);
  }
  .summary-card { background: var(--surface); padding: var(--s3) var(--s4); }
  .summary-card .value {
    font-size: 1.6rem; font-weight: 600; letter-spacing: -.02em; line-height: 1.15;
    font-variant-numeric: tabular-nums;
  }
  .summary-card .label { color: var(--ink-3); font-size: var(--t-xs); margin-top: 2px; }
  .summary-card.passed .value { color: var(--pass-ink); }
  .summary-card.failed .value { color: var(--fail-ink); }
  .summary-card.error .value { color: var(--warn-ink); }

  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r2); box-shadow: var(--shadow);
    margin-bottom: var(--s5); overflow: hidden;
  }
  .card > h2 {
    font-size: var(--t-lg); font-weight: 600; letter-spacing: -.01em;
    padding: var(--s3) var(--s4); border-bottom: 1px solid var(--border);
  }
  .card .body { padding: var(--s4); text-align: center; }

  .scenario-header {
    padding: var(--s3) var(--s4); border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; gap: var(--s3);
  }
  .scenario-header h2 {
    font-size: var(--t-lg); font-weight: 600; letter-spacing: -.01em; min-width: 0;
  }
  .badge {
    padding: 3px 10px; border-radius: 20px; white-space: nowrap;
    font-size: var(--t-xs); font-weight: 500;
  }
  .badge.passed { background: var(--pass-bg); color: var(--pass-ink); }
  .badge.failed { background: var(--fail-bg); color: var(--fail-ink); }

  .scroll { overflow-x: auto; }
  table { width: 100%; border-collapse: collapse; }
  th {
    text-align: left; padding: var(--s2) var(--s3); white-space: nowrap;
    border-bottom: 1px solid var(--border); background: var(--surface-2);
    font-size: var(--t-xs); font-weight: 600; color: var(--ink-3);
  }
  td { padding: var(--s2) var(--s3); border-bottom: 1px solid var(--border); }
  tbody tr:last-child td { border-bottom: none; }
  tbody td:first-child { white-space: nowrap; font-weight: 500; }
  .num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
  th.num { text-align: right; }
  /* A count of zero is not a verdict, so it does not get a verdict's colour. */
  .pos { color: var(--pass-ink); }
  .neg { color: var(--fail-ink); }

  .case-item {
    padding: var(--s3) var(--s4); border-bottom: 1px solid var(--border);
    display: flex; align-items: center; gap: var(--s3);
  }
  .case-item:last-child { border-bottom: none; }
  .case-info { flex: 1; min-width: 0; }
  .case-id {
    font-family: var(--mono); font-size: var(--t-sm); font-weight: 600;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .case-meta {
    font-size: var(--t-xs); color: var(--ink-3); margin-top: 2px;
    font-variant-numeric: tabular-nums;
  }
  .score-bar {
    width: 72px; height: 6px; background: var(--surface-3);
    border-radius: 3px; overflow: hidden; flex: none;
  }
  .score-fill { display: block; height: 100%; border-radius: 3px; }
  td .score-bar { display: inline-block; vertical-align: middle; margin-right: var(--s2); }
  .score-fill.good { background: var(--pass); }
  .score-fill.medium { background: var(--warn); }
  .score-fill.poor { background: var(--fail); }
  .status-icon {
    width: 22px; height: 22px; border-radius: 50%; flex: none;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 600;
  }
  .status-icon.passed { background: var(--pass-bg); color: var(--pass-ink); }
  .status-icon.failed { background: var(--fail-bg); color: var(--fail-ink); }
  .status-icon.error { background: var(--warn-bg); color: var(--warn-ink); }

  /* Charts take their ink from the same tokens as the text around them: a
     hard-coded hex is how a chart survives a theme switch looking like a
     screenshot of the other one. */
  .chart { max-width: 100%; height: auto; }
  .chart .axis { stroke: var(--border-strong); stroke-width: 1.5; }
  .chart .guide { stroke: var(--border-strong); stroke-dasharray: 5 5; }
  .chart .diag { stroke: var(--border); stroke-dasharray: 3 3; }
  .chart .tick { fill: var(--ink-3); font-size: 11px; }
  .chart .tick-line { stroke: var(--border-strong); }
  .chart .axis-label { fill: var(--ink-2); font-size: 12px; font-weight: 600; }
  .chart .zone-good { fill: var(--zone-good); }
  .chart .zone-mixed { fill: var(--zone-mixed); }
  .chart .zone-poor { fill: var(--zone-poor); }
  .chart .dot { stroke: var(--surface); stroke-width: 1.5; }
  .chart .dot.good { fill: var(--pass); }
  .chart .dot.poor { fill: var(--fail); }

  @media (max-width: 720px) {
    .container { padding: var(--s4) var(--s3); }
    .summary { grid-template-columns: repeat(auto-fit, minmax(118px, 1fr)); }
    .summary-card .value { font-size: 1.3rem; }
  }
"""


def _duration(ms: float) -> str:
    """Milliseconds at the scale a person compares them at.

    Same thresholds as the viewer's formatter: a run that took 46 seconds and
    one that took 4.9 are hard to tell apart as ``46231ms`` / ``4900ms``.
    """
    seconds = float(ms or 0.0) / 1000
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.1f}s" if seconds >= 1 else f"{ms:.0f}ms"


def _stamp(iso: str) -> str:
    """A timestamp a person reads, with the exact one on hover.

    The document stamps UTC ISO, which is right to store and noisy to read.
    Unlike the viewer this page is written once, on a machine that is not the
    reader's, so the zone is named rather than converted.
    """
    text = escape(str(iso or ""))
    try:
        moment = datetime.fromisoformat(str(iso))
    except (TypeError, ValueError):
        return text
    return (
        f'<time datetime="{text}" title="{text}">'
        f'{moment.strftime("%Y-%m-%d %H:%M")}'
        f'{" UTC" if moment.utcoffset() in (None, timedelta(0)) else ""}</time>'
    )


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
        return self.render_document(collect_run(results, name=self.title))

    def render_document(self, doc: dict[str, Any]) -> str:
        """Render a ``compass.report.site`` document to HTML.

        Args:
            doc: Document from ``collect_run``.

        Returns:
            A self-contained HTML page.
        """
        run = doc["run"]
        cases = list(iter_cases(doc))
        generated = _stamp(doc["generated"])

        # Pass rate is over evaluated cases — harness errors are not the
        # agent's failures. Name the denominator whenever it differs from the
        # case count, so the percentage cannot be read against the wrong base.
        pass_rate_label = "Pass Rate"
        if run["error_cases"]:
            pass_rate_label = f"Pass Rate ({run['evaluated_cases']} evaluated)"

        category_html = self._render_category_breakdown(doc["categories"])

        # What was under test, when the adapter installed something. The digest
        # is the identity — a source directory gets reused between versions and
        # its content does not — so it belongs beside the run's name.
        skills = run.get("skills") or []
        skills_line = ""
        if skills:
            named = ", ".join(
                f"{s['name']}@{str(s.get('digest') or '')[:8]}" for s in skills
            )
            skills_line = f'Skill: <span class="chip">{escape(named)}</span>'

        # The scatter needs both axes to mean something. Presence, not
        # magnitude: a run where every outcome grader scored 0.0 still has
        # an outcome axis, and that chart is worth looking at.
        scatter_chart_html = ""
        if doc["scopes"]["outcome"] and doc["scopes"]["transcript"]:
            scatter_chart_html = self._render_dual_axis_chart(cases)

        scenario_cards = "\n".join(
            self._render_scenario_card(s) for s in doc["scenarios"]
        )

        # Colour marks a verdict, so a zero count stays neutral: green on
        # "0 passed" and amber on "0 errors" are the same claim in opposite
        # directions, and neither is true.
        def tile(value: str, label: str, tone: str = "") -> str:
            klass = f"summary-card {tone}".strip() if tone and value != "0" else "summary-card"
            return (
                f'<div class="{klass}"><div class="value">{value}</div>'
                f'<div class="label">{label}</div></div>'
            )

        tiles = "\n            ".join([
            tile(str(run["scenarios"]), "Scenarios"),
            tile(str(run["passed_cases"]), "Passed", "passed"),
            tile(str(run["failed_cases"]), "Failed", "failed"),
            tile(str(run["error_cases"]), "Errors", "error"),
            tile(f"{run['pass_rate'] * 100:.1f}%", pass_rate_label),
            tile(f"{run['average_score']:.2f}", "Avg Score"),
        ])

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>{_CSS}</style>
</head>
<body>
    <header>
        <div class="container">
            <h1>{escape(self.title)}</h1>
            <div class="timestamp">
                <span>Generated: {generated}</span>
                {skills_line}
            </div>
        </div>
    </header>

    <div class="container">
        <div class="summary">
            {tiles}
        </div>

        {category_html}

        {scatter_chart_html}

        {scenario_cards}
    </div>
</body>
</html>"""

    @staticmethod
    def _render_category_breakdown(categories: list[dict[str, Any]]) -> str:
        """Render an HTML table from the document's category rows."""
        if not categories:
            return ""

        rows = ""
        for row in categories:
            rate = row["pass_rate"] * 100
            band = "good" if rate >= 70 else "medium" if rate >= 40 else "poor"
            rows += f"""
                <tr>
                    <td>{escape(row["category"])}</td>
                    <td class="num">{row["total"]}</td>
                    <td class="num{" pos" if row["passed"] else ""}">{row["passed"]}</td>
                    <td class="num{" neg" if row["failed"] else ""}">{row["failed"]}</td>
                    <td class="num">
                        <span class="score-bar"><span class="score-fill {band}"
                            style="width:{rate:.1f}%"></span></span>{rate:.1f}%</td>
                    <td class="num">{row["average_score"]:.2f}</td>
                </tr>"""

        return f"""
        <div class="card">
            <h2>Category Breakdown</h2>
            <div class="scroll">
            <table>
                <thead>
                    <tr>
                        <th>Category</th>
                        <th class="num">Total</th>
                        <th class="num">Passed</th>
                        <th class="num">Failed</th>
                        <th class="num">Pass Rate</th>
                        <th class="num">Avg Score</th>
                    </tr>
                </thead>
                <tbody>{rows}
                </tbody>
            </table>
            </div>
        </div>"""

    def _render_dual_axis_chart(self, cases: list[dict[str, Any]]) -> str:
        """Generate an inline SVG scatter plot of Outcome (X) vs Transcript (Y).

        Args:
            cases: Document case rows — each carries ``outcome_score``,
                ``transcript_score``, ``case_id`` and ``passed``.
        """
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
            # top-right: both good
            f'<rect x="{tx}" y="{pad_top}" '
            f'width="{sx(1.0) - tx}" height="{ty - pad_top}" class="zone-good"/>'
            # top-left: process good, result bad
            f'<rect x="{pad_left}" y="{pad_top}" '
            f'width="{tx - pad_left}" height="{ty - pad_top}" class="zone-mixed"/>'
            # bottom-right: result good, process bad
            f'<rect x="{tx}" y="{ty}" '
            f'width="{sx(1.0) - tx}" height="{sy(0.0) - ty}" class="zone-mixed"/>'
            # bottom-left: both bad
            f'<rect x="{pad_left}" y="{ty}" '
            f'width="{tx - pad_left}" height="{sy(0.0) - ty}" class="zone-poor"/>'
        )

        # Axes
        axes = (
            f'<line x1="{pad_left}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(0)}" class="axis"/>'
            f'<line x1="{pad_left}" y1="{sy(0)}" x2="{pad_left}" y2="{sy(1)}" class="axis"/>'
        )

        # Threshold dashed lines
        dashes = (
            f'<line x1="{pad_left}" y1="{ty}" x2="{sx(1)}" y2="{ty}" class="guide"/>'
            f'<line x1="{tx}" y1="{sy(0)}" x2="{tx}" y2="{sy(1)}" class="guide"/>'
        )

        # Diagonal reference line (y=x)
        diag = (
            f'<line x1="{sx(0)}" y1="{sy(0)}" x2="{sx(1)}" y2="{sy(1)}" class="diag"/>'
        )

        # Tick marks and labels
        ticks = ""
        for v in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
            # X-axis ticks
            ticks += (
                f'<line x1="{sx(v)}" y1="{sy(0)}" x2="{sx(v)}" y2="{sy(0) + 5}" '
                f'class="tick-line"/>'
                f'<text x="{sx(v)}" y="{sy(0) + 20}" text-anchor="middle" '
                f'class="tick">{v:.1f}</text>'
            )
            # Y-axis ticks
            ticks += (
                f'<line x1="{pad_left - 5}" y1="{sy(v)}" x2="{pad_left}" y2="{sy(v)}" '
                f'class="tick-line"/>'
                f'<text x="{pad_left - 10}" y="{sy(v) + 4}" text-anchor="end" '
                f'class="tick">{v:.1f}</text>'
            )

        # Axis labels
        labels = (
            f'<text x="{pad_left + plot_w / 2}" y="{h - 10}" text-anchor="middle" '
            f'class="axis-label">Outcome Score</text>'
            f'<text x="15" y="{pad_top + plot_h / 2}" text-anchor="middle" '
            f'class="axis-label" '
            f'transform="rotate(-90, 15, {pad_top + plot_h / 2})">Transcript Score</text>'
        )

        # Data points
        points = ""
        for d in cases:
            cx = sx(d["outcome_score"])
            cy = sy(d["transcript_score"])
            title = escape(
                f'{d["case_id"]}: outcome={d["outcome_score"]:.2f}, '
                f'transcript={d["transcript_score"]:.2f}'
            )
            points += (
                f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" opacity="0.85" '
                f'class="dot {"good" if d["passed"] else "poor"}">'
                f'<title>{title}</title></circle>'
            )

        # Legend
        legend_y = h - 10
        legend = (
            f'<circle cx="{sx(1) - 130}" cy="{legend_y - 3}" r="5" class="dot good"/>'
            f'<text x="{sx(1) - 120}" y="{legend_y}" class="tick">Passed</text>'
            f'<circle cx="{sx(1) - 60}" cy="{legend_y - 3}" r="5" class="dot poor"/>'
            f'<text x="{sx(1) - 50}" y="{legend_y}" class="tick">Failed</text>'
        )

        svg = (
            f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" class="chart" role="img" '
            f'aria-label="Outcome versus transcript score per case" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'{quadrants}{axes}{dashes}{diag}{ticks}{labels}{points}{legend}'
            f'</svg>'
        )

        return f"""
        <div class="card">
            <h2>Outcome vs Transcript Scatter Plot</h2>
            <div class="body">{svg}</div>
        </div>"""

    def _render_scenario_card(self, scenario: dict[str, Any]) -> str:
        """Render a single scenario card."""
        pass_rate = scenario["pass_rate"] * 100
        status_class = "passed" if pass_rate >= 70 else "failed"

        case_items = "\n".join(
            self._render_case_item(case) for case in scenario["cases"]
        )
        # Kept on one line in the template: whitespace inside the badge renders.
        badge = f'{scenario["passed_cases"]}/{scenario["total_cases"]} passed'

        return f"""
        <div class="card">
            <div class="scenario-header">
                <h2>{escape(scenario["name"])}</h2>
                <span class="badge {status_class}">{badge}</span>
            </div>
            <div class="case-list">
                {case_items}
            </div>
        </div>"""

    def _render_case_item(self, case: dict[str, Any]) -> str:
        """Render a single case item."""
        status_class = case.get("status", "failed")
        status_icon = (
            "✓" if case.get("passed")
            else "✗" if status_class == "failed" else "!"
        )

        overall_score = float(case.get("overall_score") or 0.0)
        best_score = float(case.get("best_score") or 0.0)
        has_trials = (case.get("total_trials") or 1) > 1

        # Build score meta text
        meta_parts = [f"Avg: {overall_score:.2f}"]
        if has_trials:
            meta_parts.append(f"Best: {best_score:.2f}")
            meta_parts.append(
                f"Trials: {case.get('passed_trials', 0)}/{case['total_trials']}"
            )
        meta_parts.append(f"Duration: {_duration(case.get('duration_ms', 0.0))}")
        meta_text = " | ".join(meta_parts)

        # Use best_score for the bar when multi-trial
        bar_percent = (best_score if has_trials else overall_score) * 100
        bar_class = (
            "good" if bar_percent >= 70
            else "medium" if bar_percent >= 40 else "poor"
        )

        return f"""
            <div class="case-item">
                <div class="case-info">
                    <div class="case-id">{escape(str(case.get("case_id", "")))}</div>
                    <div class="case-meta">{meta_text}</div>
                </div>
                <div class="score-bar">
                    <div class="score-fill {bar_class}"
                         style="width: {bar_percent}%"></div>
                </div>
                <div class="status-icon {status_class}">{status_icon}</div>
            </div>"""
