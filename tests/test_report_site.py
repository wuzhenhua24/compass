"""Tests for the report data layer (``compass.report.site``).

This document is what every viewer reads — the HTML report today, a published
site later — so these tests pin the numbers and the shape, not the rendering.
"""

import json

from compass.core.result import CaseResult, EvalResult, EvaluatorResult, TestStatus
from compass.report.analyzer import iter_case_dicts
from compass.report.site import (
    SCHEMA,
    category_rows,
    collect_run,
    iter_cases,
    scope_scores,
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _grader(
    name: str,
    scope: str,
    score: float | None,
    passed: bool = True,
    skipped: bool = False,
) -> EvaluatorResult:
    return EvaluatorResult(
        name=name,
        score=score,
        passed=passed,
        grader_scope=scope,
        skipped=skipped,
    )


def _case(
    case_id: str,
    status: TestStatus = TestStatus.PASSED,
    overall_score: float = 0.5,
    graders: list[EvaluatorResult] | None = None,
    **kwargs,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=status,
        passed=status is TestStatus.PASSED,
        overall_score=overall_score,
        evaluator_results=graders or [],
        **kwargs,
    )


def _scenario(name: str, cases: list[CaseResult], **kwargs) -> EvalResult:
    return EvalResult(
        scenario_name=name,
        total_cases=len(cases),
        passed_cases=sum(1 for c in cases if c.passed),
        failed_cases=sum(
            1 for c in cases if not c.passed and c.status is not TestStatus.ERROR
        ),
        error_cases=sum(1 for c in cases if c.status is TestStatus.ERROR),
        case_results=cases,
        **kwargs,
    )


# ------------------------------------------------------------------
# scope_scores
# ------------------------------------------------------------------

class TestScopeScores:
    def test_averages_each_axis(self):
        """A ``both``-scope grader lands on both axes."""
        scores = scope_scores([
            _grader("sem", "outcome", 0.8).to_dict(),
            _grader("vlm", "outcome", 0.6).to_dict(),
            _grader("cost", "transcript", 1.0).to_dict(),
            _grader("eff", "both", 0.5).to_dict(),
        ])

        assert scores["outcome_score"] == (0.8 + 0.6 + 0.5) / 3
        assert scores["transcript_score"] == 0.75
        assert scores["has_outcome"] is True
        assert scores["has_transcript"] is True

    def test_no_graders_is_zero_and_absent(self):
        scores = scope_scores([])

        assert scores["outcome_score"] == 0.0
        assert scores["transcript_score"] == 0.0
        assert scores["has_outcome"] is False
        assert scores["has_transcript"] is False

    def test_unscored_grader_is_excluded_not_zero(self):
        """A grader that crashed produced no measurement — averaging it as 0.0
        would understate the axis (and summing ``None`` would crash outright)."""
        scores = scope_scores([
            _grader("sem", "outcome", 0.8).to_dict(),
            _grader("broken", "outcome", None, passed=False).to_dict(),
        ])

        assert scores["outcome_score"] == 0.8
        assert scores["has_outcome"] is True

    def test_only_unscored_graders_leaves_axis_absent(self):
        scores = scope_scores([
            _grader("broken", "transcript", None, passed=False).to_dict(),
        ])

        assert scores["transcript_score"] == 0.0
        assert scores["has_transcript"] is False

    def test_skipped_grader_is_excluded(self):
        """Short-circuited graders never ran; they hold nothing back."""
        scores = scope_scores([
            _grader("cost", "transcript", 1.0).to_dict(),
            _grader("late", "transcript", 0.0, skipped=True).to_dict(),
        ])

        assert scores["transcript_score"] == 1.0

    def test_axis_present_even_when_every_score_is_zero(self):
        """Presence is about graders, not magnitude — a run that scored 0.0
        across the board still has that axis."""
        scores = scope_scores([
            _grader("sem", "outcome", 0.0, passed=False).to_dict(),
        ])

        assert scores["outcome_score"] == 0.0
        assert scores["has_outcome"] is True


# ------------------------------------------------------------------
# Document shape
# ------------------------------------------------------------------

class TestCollectRun:
    def test_document_shape_and_schema(self):
        doc = collect_run(
            [_scenario("s1", [_case("c1")])],
            name="nightly",
            generated="2026-08-10T00:00:00+00:00",
        )

        assert doc["schema"] == SCHEMA
        assert doc["generated"] == "2026-08-10T00:00:00+00:00"
        assert doc["run"]["name"] == "nightly"
        assert doc["run"]["scenarios"] == 1
        assert [s["name"] for s in doc["scenarios"]] == ["s1"]

    def test_is_json_serializable(self):
        """The whole point: this document gets written to disk and fetched."""
        doc = collect_run([
            _scenario(
                "s1",
                [_case("c1", graders=[_grader("sem", "outcome", 0.9)])],
                run_id="run-1",
                config_hash="abc123",
            )
        ])

        reloaded = json.loads(json.dumps(doc))
        assert reloaded["scenarios"][0]["run_id"] == "run-1"
        assert reloaded["scenarios"][0]["config_hash"] == "abc123"

    def test_iter_cases_walks_scenarios_in_order(self):
        doc = collect_run([
            _scenario("s1", [_case("c1"), _case("c2")]),
            _scenario("s2", [_case("c3")]),
        ])

        rows = list(iter_cases(doc))
        assert [r["case_id"] for r in rows] == ["c1", "c2", "c3"]
        assert [r["scenario"] for r in rows] == ["s1", "s1", "s2"]

    def test_cases_have_one_home(self):
        """Rows live under their scenario only — a published document should
        not carry every case twice."""
        doc = collect_run([_scenario("s1", [_case("c1")])])

        assert "cases" not in doc
        assert len(doc["scenarios"][0]["cases"]) == 1

    def test_case_rows_stay_readable_by_the_analyzer(self):
        """``grade_results`` is dropped as a duplicate, so the keys the
        analysis pipeline actually reads must still be there."""
        doc = collect_run([
            _scenario("s1", [_case("c1", graders=[_grader("sem", "outcome", 0.9)])])
        ])
        row = next(iter_cases(doc))

        assert "grade_results" not in row
        assert row["task_id"] == "c1"
        assert [g["name"] for g in row["evaluator_results"]] == ["sem"]
        assert iter_case_dicts({"case_results": [row]}) == [row]

    def test_case_rows_carry_scope_axes(self):
        doc = collect_run([
            _scenario("s1", [
                _case("c1", graders=[
                    _grader("sem", "outcome", 0.9),
                    _grader("cost", "transcript", 0.5),
                ])
            ])
        ])
        row = next(iter_cases(doc))

        assert row["outcome_score"] == 0.9
        assert row["transcript_score"] == 0.5


# ------------------------------------------------------------------
# Aggregation semantics
# ------------------------------------------------------------------

class TestRunSummary:
    def test_harness_errors_leave_the_denominator(self):
        """One pass, one fail, one crashed adapter: the agent went 1-for-2."""
        doc = collect_run([
            _scenario("s1", [
                _case("ok", overall_score=1.0),
                _case("bad", status=TestStatus.FAILED, overall_score=0.0),
                _case("boom", status=TestStatus.ERROR, overall_score=0.0),
            ])
        ])
        run = doc["run"]

        assert run["total_cases"] == 3
        assert run["error_cases"] == 1
        assert run["evaluated_cases"] == 2
        assert run["pass_rate"] == 0.5
        assert run["average_score"] == 0.5

    def test_means_are_over_cases_not_over_scenarios(self):
        """A 1-case scenario must not outweigh a 3-case one."""
        doc = collect_run([
            _scenario("big", [
                _case("a", overall_score=0.0),
                _case("b", overall_score=0.0),
                _case("c", overall_score=0.0),
            ]),
            _scenario("small", [_case("d", overall_score=1.0)]),
        ])

        assert doc["run"]["average_score"] == 0.25

    def test_best_of_k_uses_the_best_trial(self):
        doc = collect_run([
            _scenario("s1", [
                _case(
                    "c1",
                    overall_score=0.4,
                    total_trials=3,
                    passed_trials=1,
                    trial_metrics={"best_of_k": 0.9},
                )
            ])
        ])

        assert doc["run"]["average_score"] == 0.4
        assert doc["run"]["best_of_k_score"] == 0.9

    def test_empty_run_does_not_divide_by_zero(self):
        doc = collect_run([])
        run = doc["run"]

        assert run["total_cases"] == 0
        assert run["pass_rate"] == 0.0
        assert run["average_score"] == 0.0
        assert doc["scopes"] == {"outcome": False, "transcript": False}

    def test_scopes_flag_what_the_run_measured(self):
        doc = collect_run([
            _scenario("s1", [
                _case("c1", graders=[_grader("sem", "outcome", 0.0, passed=False)]),
            ])
        ])

        assert doc["scopes"] == {"outcome": True, "transcript": False}


class TestCategoryRows:
    def test_buckets_sorted_with_rates_as_fractions(self):
        doc = collect_run([
            _scenario("s1", [
                _case("c1", overall_score=1.0, category="backend"),
                _case("c2", status=TestStatus.FAILED, overall_score=0.0,
                      category="backend"),
                _case("c3", overall_score=0.8, category="agent"),
            ])
        ])

        assert [row["category"] for row in doc["categories"]] == ["agent", "backend"]
        backend = doc["categories"][1]
        assert backend["total"] == 2
        assert backend["passed"] == 1
        assert backend["failed"] == 1
        assert backend["pass_rate"] == 0.5
        assert backend["average_score"] == 0.5

    def test_uncategorized_cases_produce_no_rows(self):
        doc = collect_run([_scenario("s1", [_case("c1")])])

        assert doc["categories"] == []

    def test_errors_are_reported_apart_from_failures(self):
        rows = category_rows([
            {"category": "backend", "status": "passed", "passed": True,
             "overall_score": 1.0},
            {"category": "backend", "status": "error", "passed": False,
             "overall_score": 0.0},
        ])

        assert rows[0]["total"] == 2
        assert rows[0]["errors"] == 1
        assert rows[0]["evaluated"] == 1
        assert rows[0]["failed"] == 0
        assert rows[0]["pass_rate"] == 1.0
