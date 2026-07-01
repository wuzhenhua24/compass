"""Tests for the HTML report generator's dual-axis scatter chart."""

from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.report.html import HTMLReporter

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_evaluator(
    name: str,
    scope: str,
    score: float,
    passed: bool = True,
) -> EvaluatorResult:
    return EvaluatorResult(
        name=name,
        score=score,
        passed=passed,
        grader_scope=scope,
    )


def _make_case(
    case_id: str,
    passed: bool,
    evaluators: list[EvaluatorResult],
    overall_score: float = 0.5,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=TestStatus.PASSED if passed else TestStatus.FAILED,
        passed=passed,
        overall_score=overall_score,
        evaluator_results=evaluators,
    )


def _make_eval_result(
    name: str,
    cases: list[CaseResult],
) -> EvalResult:
    passed = sum(1 for c in cases if c.passed)
    failed = sum(1 for c in cases if not c.passed)
    return EvalResult(
        scenario_name=name,
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=failed,
        error_cases=0,
        case_results=cases,
    )


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestHTMLScatterChart:
    """Tests for the scatter chart in HTML reports."""

    def test_html_report_contains_scatter_chart(self):
        """HTML output includes SVG scatter chart when both scopes present."""
        cases = [
            _make_case("c1", True, [
                _make_evaluator("sem", "outcome", 0.9),
                _make_evaluator("cost", "transcript", 0.8),
            ]),
            _make_case("c2", False, [
                _make_evaluator("sem", "outcome", 0.3, passed=False),
                _make_evaluator("cost", "transcript", 0.7),
            ]),
        ]
        result = _make_eval_result("scenario1", cases)
        reporter = HTMLReporter()
        html = reporter._render([result])

        assert "<svg" in html
        assert "Outcome vs Transcript Scatter Plot" in html
        assert "Outcome Score" in html
        assert "Transcript Score" in html
        # Should contain circles for each case
        assert "<circle" in html

    def test_html_report_no_chart_without_dual_scope(self):
        """No scatter chart when only OUTCOME graders are present."""
        cases = [
            _make_case("c1", True, [
                _make_evaluator("sem", "outcome", 0.9),
            ]),
            _make_case("c2", True, [
                _make_evaluator("sem", "outcome", 0.8),
            ]),
        ]
        result = _make_eval_result("scenario1", cases)
        reporter = HTMLReporter()
        html = reporter._render([result])

        assert "Outcome vs Transcript Scatter Plot" not in html

    def test_case_scope_score_computation(self):
        """_compute_case_scope_scores correctly averages by scope."""
        case = _make_case("c1", True, [
            _make_evaluator("sem", "outcome", 0.8),
            _make_evaluator("vlm", "outcome", 0.6),
            _make_evaluator("cost", "transcript", 1.0),
            _make_evaluator("eff", "both", 0.5),
        ])
        scores = HTMLReporter._compute_case_scope_scores(case)

        assert scores["case_id"] == "c1"
        # outcome: avg(0.8, 0.6, 0.5) = 0.6333...
        assert abs(scores["outcome_score"] - (0.8 + 0.6 + 0.5) / 3) < 1e-9
        # transcript: avg(1.0, 0.5) = 0.75
        assert abs(scores["transcript_score"] - 0.75) < 1e-9
        assert scores["passed"] is True

    def test_case_scope_score_no_evaluators(self):
        """Empty evaluators yield 0.0 for both axes."""
        case = _make_case("c1", True, [])
        scores = HTMLReporter._compute_case_scope_scores(case)

        assert scores["outcome_score"] == 0.0
        assert scores["transcript_score"] == 0.0

    def test_scatter_chart_hover_titles(self):
        """SVG circles include <title> elements for hover tooltips."""
        cases = [
            _make_case("my_case", True, [
                _make_evaluator("sem", "outcome", 0.85),
                _make_evaluator("cost", "transcript", 0.75),
            ]),
        ]
        result = _make_eval_result("s1", cases)
        reporter = HTMLReporter()
        html = reporter._render([result])

        assert "my_case" in html
        assert "<title>" in html
