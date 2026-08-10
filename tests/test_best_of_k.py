"""Tests for best-of-k scoring across trials."""

from __future__ import annotations

import pytest

from compass.core.metrics import TrialMetrics, calculate_metrics
from compass.core.result import CaseResult, EvalResult, TestStatus
from compass.graders.base import GradeResult, GraderScope, GraderType
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult

# ===================================================================
# TrialMetrics.best_of_k
# ===================================================================


class TestTrialMetricsBestOfK:
    def test_best_of_3(self):
        m = TrialMetrics(
            total_trials=3, passed_trials=2,
            scores=[0.4, 0.9, 0.6],
        )
        assert m.best_of_k(3) == 0.9

    def test_best_of_1(self):
        m = TrialMetrics(
            total_trials=3, passed_trials=2,
            scores=[0.4, 0.9, 0.6],
        )
        assert m.best_of_k(1) == 0.4

    def test_best_of_k_exceeds_trials(self):
        m = TrialMetrics(
            total_trials=2, passed_trials=1,
            scores=[0.3, 0.7],
        )
        assert m.best_of_k(5) == 0.7

    def test_best_of_k_zero(self):
        m = TrialMetrics(total_trials=0, passed_trials=0, scores=[])
        assert m.best_of_k(3) == 0.0

    def test_best_of_k_negative(self):
        m = TrialMetrics(total_trials=1, passed_trials=1, scores=[0.5])
        assert m.best_of_k(-1) == 0.0

    def test_best_of_k_single_trial(self):
        m = TrialMetrics(total_trials=1, passed_trials=1, scores=[0.85])
        assert m.best_of_k(1) == 0.85

    def test_to_dict_includes_best_of_k(self):
        m = TrialMetrics(
            total_trials=3, passed_trials=2,
            scores=[0.3, 0.8, 0.5],
        )
        d = m.to_dict()
        assert "best_of_k" in d
        assert d["best_of_k"] == 0.8

    def test_calculate_metrics_helper(self):
        m = calculate_metrics(
            passed_list=[True, False, True],
            scores=[0.9, 0.2, 0.7],
        )
        assert m.best_of_k(3) == 0.9
        assert m.score_mean == pytest.approx(0.6, abs=0.01)


# ===================================================================
# CaseResult.best_score
# ===================================================================


class TestCaseResultBestScore:
    def test_single_trial_returns_overall(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.75,
        )
        assert cr.best_score == 0.75
        assert not cr.has_trials

    def test_multi_trial_with_best_of_k(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.6,
            total_trials=3, passed_trials=2,
            trial_metrics={"best_of_k": 0.9, "score_max": 0.9},
        )
        assert cr.best_score == 0.9
        assert cr.has_trials

    def test_multi_trial_fallback_to_score_max(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.5,
            total_trials=3, passed_trials=1,
            trial_metrics={"score_max": 0.85},
        )
        assert cr.best_score == 0.85

    def test_multi_trial_no_metrics(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.5,
            total_trials=3, passed_trials=1,
        )
        assert cr.best_score == 0.5


# ===================================================================
# EvalResult.best_of_k_score
# ===================================================================


class TestEvalResultBestOfK:
    def test_best_of_k_score(self):
        cases = [
            CaseResult(
                case_id="c1", status=TestStatus.PASSED,
                passed=True, overall_score=0.5,
                total_trials=3, passed_trials=2,
                trial_metrics={"best_of_k": 0.9},
            ),
            CaseResult(
                case_id="c2", status=TestStatus.FAILED,
                passed=False, overall_score=0.3,
                total_trials=3, passed_trials=0,
                trial_metrics={"best_of_k": 0.5},
            ),
        ]
        result = EvalResult(
            scenario_name="test", total_cases=2,
            passed_cases=1, failed_cases=1, error_cases=0,
            case_results=cases,
        )
        # avg = (0.5 + 0.3) / 2 = 0.4
        assert result.average_score == pytest.approx(0.4)
        # best = (0.9 + 0.5) / 2 = 0.7
        assert result.best_of_k_score == pytest.approx(0.7)

    def test_serialization_includes_best(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.6,
            total_trials=3, passed_trials=2,
            trial_metrics={"best_of_k": 0.95},
        )
        result = EvalResult(
            scenario_name="test", total_cases=1,
            passed_cases=1, failed_cases=0, error_cases=0,
            case_results=[cr],
        )
        d = result.to_dict()
        assert d["best_of_k_score"] == pytest.approx(0.95)
        assert d["case_results"][0]["best_score"] == pytest.approx(0.95)

    def test_single_trial_best_equals_avg(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.8,
        )
        result = EvalResult(
            scenario_name="test", total_cases=1,
            passed_cases=1, failed_cases=0, error_cases=0,
            case_results=[cr],
        )
        assert result.best_of_k_score == result.average_score


# ===================================================================
# Analyzer best-of-k in summary
# ===================================================================


def _task(tid: str, passed: bool, score: float,
          best: float | None = None) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=tid, passed=passed,
        grade_results=[
            GradeResult(
                name="g", grader_type=GraderType.CODE,
                grader_scope=GraderScope.OUTCOME,
                passed=passed, score=score,
            ),
        ],
        best_score=best,
    )


class TestAnalyzerBestOfK:
    def test_summary_includes_best_when_present(self):
        results = [
            _task("t1", True, 0.6, best=0.9),
            _task("t2", False, 0.3, best=0.5),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert "avg_best_score" in report.summary
        assert report.summary["avg_best_score"] == pytest.approx(0.7)
        assert report.summary["avg_score"] == pytest.approx(0.45)
        assert "best_vs_avg_gap" in report.summary

    def test_summary_no_best_when_absent(self):
        results = [_task("t1", True, 0.8)]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert "avg_best_score" not in report.summary

    def test_mixed_with_and_without_best(self):
        results = [
            _task("t1", True, 0.6, best=0.9),
            _task("t2", True, 0.8),  # no best_score
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        # At least one has best_score → summary includes it
        assert "avg_best_score" in report.summary


# ===================================================================
# Console reporter with best-of-k
# ===================================================================


class TestConsoleReporterBestOfK:
    def test_renders_with_best_of_k(self):
        from rich.console import Console

        from compass.report.analyzer import AnalysisReport
        from compass.report.console import ConsoleReporter

        console = Console(file=open("/dev/null", "w"))
        reporter = ConsoleReporter(console=console)
        report = AnalysisReport(
            summary={
                "total_tasks": 2, "passed_tasks": 1, "pass_rate": 0.5,
                "avg_transcript_score": 0.5, "avg_outcome_score": 0.5,
                "avg_duration_ms": 100,
                "avg_best_score": 0.8, "avg_score": 0.5,
                "best_vs_avg_gap": 0.3,
            },
            transcript_analysis={},
            outcome_analysis={},
            scope_comparison={
                "avg_transcript_score": 0.5, "avg_outcome_score": 0.5,
                "gap": 0, "description": "",
            },
            failure_patterns=[],
            failure_tag_analysis={},
            recommendations=[],
            category_analysis={},
            tag_analysis={},
        )
        reporter.render(report)  # Should not raise


# ===================================================================
# HTML reporter with best-of-k
# ===================================================================


class TestHTMLReporterBestOfK:
    def test_renders_multi_trial_case(self):
        from compass.report.html import HTMLReporter

        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.5,
            total_trials=3, passed_trials=2,
            trial_metrics={"best_of_k": 0.9},
        )
        result = EvalResult(
            scenario_name="test", total_cases=1,
            passed_cases=1, failed_cases=0, error_cases=0,
            case_results=[cr],
        )
        reporter = HTMLReporter()
        html = reporter._render([result])
        assert "Best: 0.90" in html
        assert "Trials: 2/3" in html

    def test_single_trial_no_best(self):
        from compass.report.html import HTMLReporter

        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED,
            passed=True, overall_score=0.8,
        )
        result = EvalResult(
            scenario_name="test", total_cases=1,
            passed_cases=1, failed_cases=0, error_cases=0,
            case_results=[cr],
        )
        reporter = HTMLReporter()
        html = reporter._render([result])
        assert "Best:" not in html
        assert "Trials:" not in html
