"""Tests for category and tags aggregation features."""

from __future__ import annotations

from compass.core.result import CaseResult, EvalResult, TestStatus
from compass.core.scenario import (
    InputConfig,
    Scenario,
    TestCase,
)
from compass.graders.base import GradeResult, GraderScope, GraderType
from compass.report.analyzer import AnalysisReport, EvalResultAnalyzer, TaskEvalResult
from compass.report.console import ConsoleReporter
from compass.report.html import HTMLReporter

# ===================================================================
# Scenario model tests
# ===================================================================


class TestCategoryField:
    def test_testcase_category_default_empty(self):
        tc = TestCase(id="t1", input=InputConfig(prompt="test"))
        assert tc.category == ""

    def test_testcase_category_set(self):
        tc = TestCase(id="t1", input=InputConfig(prompt="test"), category="backend")
        assert tc.category == "backend"

    def test_scenario_category_default_empty(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
        )
        assert scn.category == ""

    def test_scenario_category_set(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
            category="fullstack",
        )
        assert scn.category == "fullstack"

    def test_get_category_for_case_inherits_scenario(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
            category="backend",
            cases=[TestCase(id="c1", input=InputConfig(prompt="p"))],
        )
        assert scn.get_category_for_case(scn.cases[0]) == "backend"

    def test_get_category_for_case_case_overrides(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
            category="backend",
            cases=[
                TestCase(id="c1", input=InputConfig(prompt="p"), category="gym"),
            ],
        )
        assert scn.get_category_for_case(scn.cases[0]) == "gym"

    def test_filter_by_category(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
            category="backend",
            cases=[
                TestCase(id="c1", input=InputConfig(prompt="p")),
                TestCase(
                    id="c2", input=InputConfig(prompt="p"), category="fullstack",
                ),
                TestCase(id="c3", input=InputConfig(prompt="p"), category="gym"),
            ],
        )
        # c1 inherits "backend", c2 is "fullstack", c3 is "gym"
        backend = scn.filter_by_category(["backend"])
        assert [c.id for c in backend] == ["c1"]

        fullstack = scn.filter_by_category(["fullstack"])
        assert [c.id for c in fullstack] == ["c2"]

        multi = scn.filter_by_category(["backend", "gym"])
        assert [c.id for c in multi] == ["c1", "c3"]

    def test_filter_by_category_no_match(self):
        scn = Scenario(
            name="test",
            agent={"adapter": "image", "endpoint": "http://test"},
            cases=[TestCase(id="c1", input=InputConfig(prompt="p"))],
        )
        assert scn.filter_by_category(["nonexistent"]) == []

    def test_yaml_load_with_category(self, tmp_path):
        """Test that category/tags can be loaded from YAML."""
        yaml_content = """
name: test
agent:
  adapter: image
  endpoint: http://test
category: backend
cases:
  - id: c1
    input:
      prompt: "p"
    category: gym
    tags:
      - api
      - stripe
"""
        yaml_path = tmp_path / "test.yaml"
        yaml_path.write_text(yaml_content)
        loaded = Scenario.from_yaml(yaml_path)
        assert loaded.category == "backend"
        assert loaded.cases[0].category == "gym"
        assert loaded.cases[0].tags == ["api", "stripe"]


# ===================================================================
# CaseResult tests
# ===================================================================


class TestCaseResultCategoryTags:
    def test_case_result_has_category_tags(self):
        cr = CaseResult(
            case_id="c1",
            status=TestStatus.PASSED,
            passed=True,
            overall_score=1.0,
            tags=["api"],
            category="backend",
        )
        assert cr.category == "backend"
        assert cr.tags == ["api"]

    def test_case_result_defaults(self):
        cr = CaseResult(
            case_id="c1",
            status=TestStatus.PASSED,
            passed=True,
            overall_score=1.0,
        )
        assert cr.category == ""
        assert cr.tags == []

    def test_eval_result_serializes_category_tags(self):
        cr = CaseResult(
            case_id="c1",
            status=TestStatus.PASSED,
            passed=True,
            overall_score=1.0,
            tags=["api"],
            category="backend",
        )
        result = EvalResult(
            scenario_name="test",
            total_cases=1,
            passed_cases=1,
            failed_cases=0,
            error_cases=0,
            case_results=[cr],
        )
        d = result.to_dict()
        assert d["case_results"][0]["tags"] == ["api"]
        assert d["case_results"][0]["category"] == "backend"


# ===================================================================
# Analyzer tests
# ===================================================================


def _make_task(
    task_id: str,
    passed: bool,
    score: float = 1.0,
    category: str = "",
    tags: list[str] | None = None,
) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=task_id,
        passed=passed,
        grade_results=[
            GradeResult(
                name="test_grader",
                grader_type=GraderType.CODE,
                grader_scope=GraderScope.OUTCOME,
                passed=passed,
                score=score,
            ),
        ],
        category=category,
        tags=tags or [],
    )


class TestAnalyzerCategoryAnalysis:
    def test_empty_categories(self):
        results = [_make_task("t1", True)]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert report.category_analysis == {}

    def test_single_category(self):
        results = [
            _make_task("t1", True, 1.0, category="backend"),
            _make_task("t2", False, 0.3, category="backend"),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert "backend" in report.category_analysis
        stats = report.category_analysis["backend"]
        assert stats["total"] == 2
        assert stats["passed"] == 1
        assert stats["failed"] == 1
        assert stats["pass_rate"] == 0.5

    def test_multiple_categories(self):
        results = [
            _make_task("t1", True, 1.0, category="backend"),
            _make_task("t2", True, 0.8, category="backend"),
            _make_task("t3", False, 0.2, category="fullstack"),
            _make_task("t4", True, 0.9, category="gym"),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert len(report.category_analysis) == 3
        assert report.category_analysis["backend"]["pass_rate"] == 1.0
        assert report.category_analysis["fullstack"]["pass_rate"] == 0.0
        assert report.category_analysis["gym"]["pass_rate"] == 1.0

    def test_mixed_categorized_and_uncategorized(self):
        results = [
            _make_task("t1", True, category="backend"),
            _make_task("t2", True),  # no category
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        # Only categorized tasks appear
        assert len(report.category_analysis) == 1
        assert "backend" in report.category_analysis

    def test_category_avg_score(self):
        results = [
            _make_task("t1", True, 0.8, category="api"),
            _make_task("t2", False, 0.4, category="api"),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        stats = report.category_analysis["api"]
        assert abs(stats["avg_score"] - 0.6) < 0.01


class TestAnalyzerTagAnalysis:
    def test_empty_tags(self):
        results = [_make_task("t1", True)]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert report.tag_analysis == {}

    def test_single_tag(self):
        results = [
            _make_task("t1", True, tags=["api"]),
            _make_task("t2", False, 0.3, tags=["api"]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert "api" in report.tag_analysis
        assert report.tag_analysis["api"]["total"] == 2
        assert report.tag_analysis["api"]["passed"] == 1

    def test_multiple_tags_per_task(self):
        results = [
            _make_task("t1", True, tags=["api", "stripe"]),
            _make_task("t2", False, 0.3, tags=["api"]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        assert report.tag_analysis["api"]["total"] == 2
        assert report.tag_analysis["stripe"]["total"] == 1

    def test_tags_sorted_by_count(self):
        results = [
            _make_task("t1", True, tags=["rare"]),
            _make_task("t2", True, tags=["common"]),
            _make_task("t3", True, tags=["common"]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        keys = list(report.tag_analysis.keys())
        assert keys[0] == "common"  # most frequent first


class TestAnalysisReportSerialization:
    def test_to_dict_includes_category_tag_analysis(self):
        results = [
            _make_task("t1", True, category="backend", tags=["api"]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()
        d = report.to_dict()
        assert "category_analysis" in d
        assert "tag_analysis" in d


# ===================================================================
# Console reporter tests
# ===================================================================


class TestConsoleReporterCategory:
    def test_renders_category_table(self, capsys):
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        reporter = ConsoleReporter(console=console)
        report = AnalysisReport(
            summary={"total_tasks": 2, "passed_tasks": 1, "pass_rate": 0.5,
                     "avg_transcript_score": 0.5, "avg_outcome_score": 0.5,
                     "avg_duration_ms": 100},
            transcript_analysis={},
            outcome_analysis={},
            scope_comparison={"avg_transcript_score": 0.5,
                              "avg_outcome_score": 0.5, "gap": 0,
                              "description": "balanced"},
            failure_patterns=[],
            failure_tag_analysis={},
            recommendations=[],
            category_analysis={
                "backend": {
                    "total": 2, "passed": 1, "failed": 1,
                    "pass_rate": 0.5, "avg_score": 0.6,
                    "avg_outcome_score": 0.6, "avg_transcript_score": 0.5,
                },
            },
            tag_analysis={},
        )
        # Should not raise
        reporter.render(report)

    def test_skips_empty_category(self, capsys):
        from rich.console import Console

        console = Console(file=open("/dev/null", "w"))
        reporter = ConsoleReporter(console=console)
        report = AnalysisReport(
            summary={"total_tasks": 0, "passed_tasks": 0, "pass_rate": 0.0,
                     "avg_transcript_score": 0.0, "avg_outcome_score": 0.0,
                     "avg_duration_ms": 0},
            transcript_analysis={},
            outcome_analysis={},
            scope_comparison={"avg_transcript_score": 0, "avg_outcome_score": 0,
                              "gap": 0, "description": ""},
            failure_patterns=[],
            failure_tag_analysis={},
            recommendations=[],
            category_analysis={},
            tag_analysis={},
        )
        # Should not raise
        reporter.render(report)


# ===================================================================
# HTML reporter tests
# ===================================================================


class TestHTMLReporterCategory:
    def test_renders_category_breakdown(self):
        cr1 = CaseResult(
            case_id="c1", status=TestStatus.PASSED, passed=True,
            overall_score=0.9, category="backend",
        )
        cr2 = CaseResult(
            case_id="c2", status=TestStatus.FAILED, passed=False,
            overall_score=0.3, category="fullstack",
        )
        result = EvalResult(
            scenario_name="test", total_cases=2, passed_cases=1,
            failed_cases=1, error_cases=0, case_results=[cr1, cr2],
        )
        reporter = HTMLReporter()
        html = reporter._render([result])
        assert "Category Breakdown" in html
        assert "backend" in html
        assert "fullstack" in html

    def test_no_category_no_table(self):
        cr = CaseResult(
            case_id="c1", status=TestStatus.PASSED, passed=True,
            overall_score=0.9,
        )
        result = EvalResult(
            scenario_name="test", total_cases=1, passed_cases=1,
            failed_cases=0, error_cases=0, case_results=[cr],
        )
        reporter = HTMLReporter()
        html = reporter._render([result])
        assert "Category Breakdown" not in html
