"""Tests for the HTML report generator.

The numbers themselves are pinned in ``test_report_site.py`` — what these
assert is that the page renders what the document says.
"""

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
    errors = sum(1 for c in cases if c.status is TestStatus.ERROR)
    return EvalResult(
        scenario_name=name,
        total_cases=len(cases),
        passed_cases=passed,
        failed_cases=len(cases) - passed - errors,
        error_cases=errors,
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

    def test_chart_renders_when_every_score_is_zero(self):
        """Both axes were measured — a run that scored 0.0 across the board is
        exactly the run worth plotting."""
        cases = [
            _make_case("c1", False, [
                _make_evaluator("sem", "outcome", 0.0, passed=False),
                _make_evaluator("cost", "transcript", 0.0, passed=False),
            ], overall_score=0.0),
        ]
        html = HTMLReporter()._render([_make_eval_result("s1", cases)])

        assert "Outcome vs Transcript Scatter Plot" in html

    def test_unscored_grader_does_not_break_the_report(self):
        """A grader that crashed produced no number; it must not be averaged
        in (nor summed against ``None``)."""
        cases = [
            _make_case("c1", True, [
                _make_evaluator("sem", "outcome", 0.8),
                EvaluatorResult(
                    name="broken", score=None, passed=False, grader_scope="outcome",
                ),
                _make_evaluator("cost", "transcript", 0.5),
            ]),
        ]
        html = HTMLReporter()._render([_make_eval_result("s1", cases)])

        assert "<circle" in html

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


class TestHTMLSummary:
    """The summary cards read straight off the document."""

    def test_pass_rate_excludes_harness_errors(self):
        """1 of 2 evaluated cases passed — the crashed third case is not the
        agent's failure, and must not dilute the rate to 33%."""
        cases = [
            _make_case("ok", True, [], overall_score=1.0),
            _make_case("bad", False, [], overall_score=0.0),
            CaseResult(
                case_id="boom",
                status=TestStatus.ERROR,
                passed=False,
                overall_score=0.0,
            ),
        ]
        html = HTMLReporter()._render([_make_eval_result("s1", cases)])

        assert "50.0%" in html
        assert "Pass Rate (2 evaluated)" in html

    def test_pass_rate_label_stays_plain_without_errors(self):
        cases = [_make_case("ok", True, [], overall_score=1.0)]
        html = HTMLReporter()._render([_make_eval_result("s1", cases)])

        assert ">Pass Rate<" in html
        assert "100.0%" in html

    def test_case_ids_are_escaped(self):
        """Case ids come from user YAML and end up on a shareable page."""
        cases = [_make_case("<script>x</script>", True, [])]
        html = HTMLReporter()._render([_make_eval_result("s1", cases)])

        assert "<script>x</script>" not in html
        assert "&lt;script&gt;" in html
