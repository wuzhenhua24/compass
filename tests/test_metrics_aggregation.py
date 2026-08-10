"""Tests for grader-emitted metrics.

Tags and metrics are the two halves of "what did we observe": tags are
categorical (*what was seen*), metrics quantitative (*how much*). Numbers
aggregate as mean ± stderr; booleans as rates, because averaging a flag into
"0.83" reads as a score rather than "true 83% of the time".
"""

from __future__ import annotations

import stat

import pytest

from compass.core.result import EvaluatorResult
from compass.graders.base import (
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
    normalize_metrics,
)
from compass.report.analyzer import EvalResultAnalyzer, TaskEvalResult
from compass.report.console import ConsoleReporter


def _task(task_id: str, passed: bool, metrics: dict) -> TaskEvalResult:
    return TaskEvalResult(
        task_id=task_id,
        passed=passed,
        grade_results=[
            GradeResult(
                name="g", grader_type=GraderType.CODE,
                grader_scope=GraderScope.OUTCOME,
                passed=passed, score=1.0 if passed else 0.0, metrics=metrics,
            )
        ],
    )


def _analyze(tasks: list[TaskEvalResult]) -> dict:
    return EvalResultAnalyzer(tasks).analyze().metrics_analysis


# ===================================================================
# Normalization at the boundary
# ===================================================================


class TestNormalization:
    def test_numbers_become_floats(self):
        assert normalize_metrics({"count": 3}) == {"count": 3.0}

    def test_booleans_stay_booleans(self):
        """bool is a subclass of int — coercing it would lose the rate meaning."""
        result = normalize_metrics({"ok": True})
        assert result == {"ok": True}
        assert isinstance(result["ok"], bool)

    def test_unaggregatable_values_are_dropped(self):
        """A string cannot be averaged; it belongs in details, not metrics."""
        assert normalize_metrics(
            {"n": 1, "note": "text", "nested": {"a": 1}, "items": [1, 2]}
        ) == {"n": 1.0}

    def test_names_are_normalized_like_tags(self):
        assert normalize_metrics({"Element Count": 3}) == {"element_count": 3.0}

    def test_grade_result_normalizes_on_construction(self):
        result = GradeResult(name="g", metrics={"Precision": 0.9, "junk": "x"})
        assert result.metrics == {"precision": 0.9}

    def test_metrics_survive_serialization(self):
        original = EvaluatorResult(
            name="g", score=1.0, passed=True, metrics={"p": 0.9, "ok": True}
        )
        assert EvaluatorResult.from_dict(original.to_dict()).metrics == {
            "p": 0.9, "ok": True,
        }


# ===================================================================
# Aggregation
# ===================================================================


class TestAggregation:
    def test_numbers_report_mean_and_stderr(self):
        analysis = _analyze(
            [
                _task("a", True, {"precision": 0.9}),
                _task("b", True, {"precision": 0.5}),
                _task("c", True, {"precision": 0.7}),
            ]
        )
        entry = analysis["precision"]
        assert entry["kind"] == "number"
        assert entry["mean"] == pytest.approx(0.7)
        assert entry["stderr"] > 0
        assert (entry["min"], entry["max"]) == (0.5, 0.9)
        assert entry["count"] == 3

    def test_booleans_report_a_rate(self):
        analysis = _analyze(
            [
                _task("a", True, {"status_correct": True}),
                _task("b", False, {"status_correct": False}),
                _task("c", True, {"status_correct": True}),
            ]
        )
        entry = analysis["status_correct"]
        assert entry["kind"] == "rate"
        assert entry["rate"] == pytest.approx(2 / 3)
        assert entry["true_count"] == 2
        assert entry["count"] == 3

    def test_a_single_observation_has_no_stderr(self):
        entry = _analyze([_task("a", True, {"n": 5})])["n"]
        assert entry["mean"] == 5.0
        assert entry["stderr"] == 0.0

    def test_count_reflects_who_actually_reported_it(self):
        """A mean over 2 of 4 cases is a different claim from a mean over 4."""
        analysis = _analyze(
            [
                _task("a", True, {"sometimes": 1.0}),
                _task("b", True, {}),
                _task("c", True, {"sometimes": 3.0}),
                _task("d", True, {}),
            ]
        )
        assert analysis["sometimes"]["count"] == 2
        assert analysis["sometimes"]["mean"] == pytest.approx(2.0)

    def test_mixed_types_fall_back_to_numeric(self):
        """Only an all-boolean metric is a rate; anything else is a number."""
        analysis = _analyze(
            [_task("a", True, {"m": True}), _task("b", True, {"m": 0.5})]
        )
        assert analysis["m"]["kind"] == "number"
        assert analysis["m"]["mean"] == pytest.approx(0.75)

    def test_metrics_from_several_graders_merge(self):
        task = TaskEvalResult(
            task_id="t", passed=True,
            grade_results=[
                GradeResult(name="a", grader_type=GraderType.CODE,
                            grader_scope=GraderScope.OUTCOME, passed=True,
                            score=1.0, metrics={"x": 1.0}),
                GradeResult(name="b", grader_type=GraderType.CODE,
                            grader_scope=GraderScope.TRANSCRIPT, passed=True,
                            score=1.0, metrics={"y": 2.0}),
            ],
        )
        analysis = _analyze([task])
        assert set(analysis) == {"x", "y"}

    def test_failing_tasks_contribute_too(self):
        """Metrics describe what was observed, not only what went wrong."""
        analysis = _analyze(
            [_task("a", True, {"n": 1.0}), _task("b", False, {"n": 3.0})]
        )
        assert analysis["n"]["count"] == 2

    def test_no_metrics_yields_an_empty_section(self):
        report = EvalResultAnalyzer([_task("a", True, {})]).analyze()
        assert report.metrics_analysis == {}
        assert report.to_dict()["metrics_analysis"] == {}

    def test_empty_result_set(self):
        assert _analyze([]) == {}

    def test_sorted_by_name_for_stable_output(self):
        analysis = _analyze([_task("a", True, {"zulu": 1, "alpha": 2})])
        assert list(analysis) == ["alpha", "zulu"]


# ===================================================================
# Rendering
# ===================================================================


class TestRendering:
    def test_rates_and_numbers_render_differently(self, capsys):
        report = EvalResultAnalyzer(
            [
                _task("a", True, {"precision": 0.9, "ok": True}),
                _task("b", False, {"precision": 0.5, "ok": False}),
            ]
        ).analyze()
        ConsoleReporter().render(report)

        out = capsys.readouterr().out
        assert "Grader Metrics" in out
        assert "50%" in out  # the boolean rate
        assert "0.700" in out  # the numeric mean

    def test_nothing_is_rendered_without_metrics(self, capsys):
        report = EvalResultAnalyzer([_task("a", True, {})]).analyze()
        ConsoleReporter().render(report)
        assert "Grader Metrics" not in capsys.readouterr().out


# ===================================================================
# The subprocess checker contract
# ===================================================================


class TestExternalCheckerMetrics:
    async def test_metrics_land_in_the_first_class_field(self, tmp_path):
        """Not in details: the aggregate reads them, so they are core-owned."""
        from compass.core.transcript import Outcome, Transcript
        from compass.graders.code.common.external import ExternalCheckerGrader

        checker = tmp_path / "c"
        checker.write_text(
            "#!/usr/bin/env python3\n"
            "import json\n"
            "print(json.dumps({'metrics': {'Word Count': 12, 'cited': True,"
            " 'note': 'dropped'}}))\n"
        )
        checker.chmod(checker.stat().st_mode | stat.S_IXUSR)

        ws = tmp_path / "ws"
        ws.mkdir()
        context = GradeContext(
            transcript=Transcript(task_id="t", trial_id="t1"),
            outcome=Outcome(),
            workspace=ws,
        )
        result = await ExternalCheckerGrader({"command": str(checker)}).grade(context)

        assert result.metrics == {"word_count": 12.0, "cited": True}
        assert "metrics" not in result.details
