"""Tests for the evaluation result analyzer."""

from compass.graders.base import GradeResult, GraderScope, GraderType
from compass.report.analyzer import (
    AnalysisReport,
    EvalResultAnalyzer,
    TaskEvalResult,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_grade(
    name: str,
    scope: GraderScope,
    passed: bool,
    score: float,
    failure_tags: list[str] | None = None,
) -> GradeResult:
    return GradeResult(
        name=name,
        grader_type=GraderType.CODE,
        grader_scope=scope,
        passed=passed,
        score=score,
        failure_tags=failure_tags or [],
    )


def _make_task(
    task_id: str,
    passed: bool,
    grade_results: list[GradeResult],
    duration_ms: float = 100.0,
) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=task_id,
        passed=passed,
        grade_results=grade_results,
        duration_ms=duration_ms,
    )


# ------------------------------------------------------------------
# Failure tag analysis
# ------------------------------------------------------------------

class TestFailureTagAnalysis:
    """Tests for _analyze_failure_tags and its integration into the report."""

    def test_failure_tag_analysis_basic(self):
        """Tags from failed graders on failed tasks are clustered correctly."""
        results = [
            _make_task("t1", False, [
                _make_grade("exit_code_check", GraderScope.OUTCOME, False, 0.0,
                            ["exit_code_mismatch"]),
                _make_grade("style_check", GraderScope.OUTCOME, False, 0.3,
                            ["missing_section", "forbidden_phrase"]),
            ]),
            _make_task("t2", False, [
                _make_grade("exit_code_check", GraderScope.OUTCOME, False, 0.0,
                            ["exit_code_mismatch"]),
                _make_grade("semantic", GraderScope.OUTCOME, True, 0.9),
            ]),
            _make_task("t3", True, [
                _make_grade("exit_code_check", GraderScope.OUTCOME, True, 1.0),
                _make_grade("semantic", GraderScope.OUTCOME, True, 0.8),
            ]),
        ]

        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        tags = report.failure_tag_analysis
        assert "exit_code_mismatch" in tags
        assert tags["exit_code_mismatch"]["count"] == 2
        assert set(tags["exit_code_mismatch"]["graders"]) == {"exit_code_check"}
        assert set(tags["exit_code_mismatch"]["task_ids"]) == {"t1", "t2"}

        assert "missing_section" in tags
        assert tags["missing_section"]["count"] == 1
        assert tags["missing_section"]["graders"] == ["style_check"]

        assert "forbidden_phrase" in tags
        assert tags["forbidden_phrase"]["count"] == 1

    def test_failure_tag_analysis_empty_when_all_pass(self):
        """When every task passes, failure_tag_analysis is empty."""
        results = [
            _make_task("t1", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
            ]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        assert report.failure_tag_analysis == {}

    def test_failure_tag_analysis_empty_when_no_results(self):
        """Analyzer handles zero results gracefully."""
        analyzer = EvalResultAnalyzer([])
        report = analyzer.analyze()

        assert report.failure_tag_analysis == {}

    def test_failure_tag_percentage_calculation(self):
        """Percentage is relative to number of failed tasks."""
        results = [
            _make_task("t1", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["tag_a"]),
            ]),
            _make_task("t2", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["tag_a", "tag_b"]),
            ]),
            _make_task("t3", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
            ]),
            _make_task("t4", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
            ]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        # 2 failed tasks total
        assert report.failure_tag_analysis["tag_a"]["count"] == 2
        assert report.failure_tag_analysis["tag_a"]["percentage"] == 100.0
        assert report.failure_tag_analysis["tag_b"]["count"] == 1
        assert report.failure_tag_analysis["tag_b"]["percentage"] == 50.0

    def test_failure_tag_sorted_by_count_desc(self):
        """Tags should be sorted by count in descending order."""
        results = [
            _make_task("t1", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["rare_tag"]),
            ]),
            _make_task("t2", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["common_tag"]),
            ]),
            _make_task("t3", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["common_tag"]),
            ]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        tag_names = list(report.failure_tag_analysis.keys())
        assert tag_names[0] == "common_tag"
        assert tag_names[1] == "rare_tag"

    def test_failure_tag_multiple_graders_same_tag(self):
        """Same tag from different graders lists all source graders."""
        results = [
            _make_task("t1", False, [
                _make_grade("grader_a", GraderScope.OUTCOME, False, 0.0,
                            ["shared_tag"]),
                _make_grade("grader_b", GraderScope.TRANSCRIPT, False, 0.2,
                            ["shared_tag"]),
            ]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        assert report.failure_tag_analysis["shared_tag"]["count"] == 2
        assert set(report.failure_tag_analysis["shared_tag"]["graders"]) == {
            "grader_a", "grader_b",
        }

    def test_passed_grader_tags_ignored_on_failed_task(self):
        """On a failed task, tags from passing graders are not collected."""
        results = [
            _make_task("t1", False, [
                _make_grade("g_fail", GraderScope.OUTCOME, False, 0.0,
                            ["real_failure"]),
                # This grader passed — its tags (if any) should be ignored
                _make_grade("g_pass", GraderScope.OUTCOME, True, 1.0,
                            ["not_a_failure"]),
            ]),
        ]
        analyzer = EvalResultAnalyzer(results)
        report = analyzer.analyze()

        assert "real_failure" in report.failure_tag_analysis
        assert "not_a_failure" not in report.failure_tag_analysis


# ------------------------------------------------------------------
# Report structure
# ------------------------------------------------------------------

class TestAnalysisReport:
    """Tests for the AnalysisReport dataclass."""

    def test_report_to_dict_includes_failure_tag_analysis(self):
        """to_dict() includes the failure_tag_analysis field."""
        report = AnalysisReport(
            summary={},
            transcript_analysis={},
            outcome_analysis={},
            scope_comparison={},
            failure_patterns=[],
            failure_tag_analysis={"tag_x": {"count": 3, "percentage": 50.0}},
            recommendations=[],
        )
        d = report.to_dict()
        assert "failure_tag_analysis" in d
        assert d["failure_tag_analysis"]["tag_x"]["count"] == 3

    def test_full_analyze_returns_all_fields(self):
        """analyze() populates every field of AnalysisReport."""
        results = [
            _make_task("t1", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        assert isinstance(report.summary, dict)
        assert isinstance(report.transcript_analysis, dict)
        assert isinstance(report.outcome_analysis, dict)
        assert isinstance(report.scope_comparison, dict)
        assert isinstance(report.failure_patterns, list)
        assert isinstance(report.failure_tag_analysis, dict)
        assert isinstance(report.recommendations, list)


# ------------------------------------------------------------------
# Recommendations integration with failure tags
# ------------------------------------------------------------------

class TestRecommendationsWithTags:
    """Verify _generate_recommendations uses failure tag data."""

    def test_recommendation_includes_top_failure_tags(self):
        """Top failure tags with count >= 2 appear in recommendations."""
        results = [
            _make_task("t1", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["recurring_issue"]),
            ]),
            _make_task("t2", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["recurring_issue"]),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        # At least one recommendation should mention the recurring tag
        tag_recs = [r for r in report.recommendations if "recurring_issue" in r]
        assert len(tag_recs) >= 1

    def test_no_tag_recommendation_for_single_occurrence(self):
        """Tags appearing only once do not generate tag-specific recommendations."""
        results = [
            _make_task("t1", False, [
                _make_grade("g1", GraderScope.OUTCOME, False, 0.0,
                            ["one_time_tag"]),
            ]),
            _make_task("t2", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        tag_recs = [r for r in report.recommendations if "one_time_tag" in r]
        assert len(tag_recs) == 0


# ------------------------------------------------------------------
# Dual axis data (Outcome vs Transcript scatter)
# ------------------------------------------------------------------

class TestDualAxisData:
    """Tests for dual_axis_data computation and quadrant statistics."""

    def test_dual_axis_data_populated(self):
        """analyze() produces dual_axis_data with correct format."""
        results = [
            _make_task("t1", True, [
                _make_grade("sem", GraderScope.OUTCOME, True, 0.9),
                _make_grade("cost", GraderScope.TRANSCRIPT, True, 0.8),
            ]),
            _make_task("t2", False, [
                _make_grade("sem", GraderScope.OUTCOME, False, 0.3),
                _make_grade("cost", GraderScope.TRANSCRIPT, True, 0.7),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        assert len(report.dual_axis_data) == 2
        d1 = report.dual_axis_data[0]
        assert d1["task_id"] == "t1"
        assert d1["outcome_score"] == 0.9
        assert d1["transcript_score"] == 0.8
        assert d1["passed"] is True

        d2 = report.dual_axis_data[1]
        assert d2["task_id"] == "t2"
        assert d2["outcome_score"] == 0.3
        assert d2["transcript_score"] == 0.7
        assert d2["passed"] is False

    def test_dual_axis_data_empty_results(self):
        """Empty input produces empty dual_axis_data without crashing."""
        report = EvalResultAnalyzer([]).analyze()
        assert report.dual_axis_data == []

    def test_dual_axis_quadrant_stats(self):
        """Quadrant stats in scope_comparison are computed correctly."""
        results = [
            # both good: outcome=0.9, transcript=0.8 (both >= 0.7)
            _make_task("t1", True, [
                _make_grade("sem", GraderScope.OUTCOME, True, 0.9),
                _make_grade("cost", GraderScope.TRANSCRIPT, True, 0.8),
            ]),
            # result_good_process_bad: outcome=0.8, transcript=0.3
            _make_task("t2", False, [
                _make_grade("sem", GraderScope.OUTCOME, True, 0.8),
                _make_grade("cost", GraderScope.TRANSCRIPT, False, 0.3),
            ]),
            # process_good_result_bad: outcome=0.2, transcript=0.9
            _make_task("t3", False, [
                _make_grade("sem", GraderScope.OUTCOME, False, 0.2),
                _make_grade("cost", GraderScope.TRANSCRIPT, True, 0.9),
            ]),
            # both bad: outcome=0.1, transcript=0.1
            _make_task("t4", False, [
                _make_grade("sem", GraderScope.OUTCOME, False, 0.1),
                _make_grade("cost", GraderScope.TRANSCRIPT, False, 0.1),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        qs = report.scope_comparison["quadrant_stats"]
        assert qs["both_good"] == 1
        assert qs["result_good_process_bad"] == 1
        assert qs["process_good_result_bad"] == 1
        assert qs["both_bad"] == 1

    def test_dual_axis_mixed_scopes(self):
        """Tasks with both OUTCOME and TRANSCRIPT graders compute both axes."""
        results = [
            _make_task("t1", True, [
                _make_grade("sem", GraderScope.OUTCOME, True, 0.8),
                _make_grade("vlm", GraderScope.OUTCOME, True, 0.6),
                _make_grade("cost", GraderScope.TRANSCRIPT, True, 1.0),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        d = report.dual_axis_data[0]
        # outcome_score = avg(0.8, 0.6) = 0.7
        assert abs(d["outcome_score"] - 0.7) < 1e-9
        assert d["transcript_score"] == 1.0

    def test_dual_axis_only_outcome_graders(self):
        """When only OUTCOME graders exist, transcript_score is 0."""
        results = [
            _make_task("t1", True, [
                _make_grade("sem", GraderScope.OUTCOME, True, 0.9),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()

        d = report.dual_axis_data[0]
        assert d["outcome_score"] == 0.9
        assert d["transcript_score"] == 0.0

    def test_dual_axis_data_in_to_dict(self):
        """dual_axis_data is included in to_dict() output."""
        results = [
            _make_task("t1", True, [
                _make_grade("g1", GraderScope.OUTCOME, True, 1.0),
                _make_grade("g2", GraderScope.TRANSCRIPT, True, 0.9),
            ]),
        ]
        report = EvalResultAnalyzer(results).analyze()
        d = report.to_dict()

        assert "dual_axis_data" in d
        assert len(d["dual_axis_data"]) == 1
        assert d["dual_axis_data"][0]["task_id"] == "t1"
