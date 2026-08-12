"""Tests for paired run comparison (compass compare)."""

from __future__ import annotations

import json
import math

import pytest
from click.testing import CliRunner

from compass.cli.main import cli
from compass.report.compare import (
    AmbiguousGraderError,
    CaseRecord,
    compare_metric,
    compare_paths,
    compare_results,
    load_case_records,
    paired_stats,
    select_grader_score,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def rec(case_id: str, passed: bool, score: float | None = None, **kw) -> CaseRecord:
    if score is None:
        score = 1.0 if passed else 0.0
    return CaseRecord(
        case_id=case_id,
        passed=passed,
        score=score,
        pass_fraction=kw.get("pass_fraction", 1.0 if passed else 0.0),
    )


def case_dict(case_id: str, passed: bool, score: float | None = None, **kw) -> dict:
    d = {
        "case_id": case_id,
        "passed": passed,
        "overall_score": score if score is not None else (1.0 if passed else 0.0),
    }
    d.update(kw)
    return d


# ---------------------------------------------------------------------------
# paired_stats
# ---------------------------------------------------------------------------


class TestPairedStats:
    def test_empty(self):
        s = paired_stats([])
        assert s.n == 0
        assert s.significant is False
        assert math.isinf(s.mde)

    def test_known_values(self):
        # diffs = [1, 0, 1, 0]: mean 0.5, sd = sqrt(1/3), se = sd/2
        s = paired_stats([1.0, 0.0, 1.0, 0.0])
        assert s.mean_diff == 0.5
        se = math.sqrt((1 / 3) / 4)
        assert s.ci_low == pytest.approx(0.5 - 1.96 * se)
        assert s.ci_high == pytest.approx(0.5 + 1.96 * se)
        assert s.mde == pytest.approx(2.80 * se)

    def test_zero_diff_not_significant(self):
        s = paired_stats([0.0] * 10)
        assert s.mean_diff == 0.0
        assert s.significant is False

    def test_consistent_diff_significant(self):
        # All cases moved the same direction: zero variance, CI collapses
        s = paired_stats([0.1] * 10)
        assert s.significant is True

    def test_noisy_diff_not_significant(self):
        # Mean +0.1 but huge spread: CI includes 0
        s = paired_stats([1.0, -0.8, 0.9, -0.9, 0.3])
        assert s.significant is False
        assert s.ci_low < 0 < s.ci_high

    def test_single_observation(self):
        s = paired_stats([0.5])
        assert s.n == 1
        assert s.ci_low == s.ci_high == 0.5


# ---------------------------------------------------------------------------
# compare_results
# ---------------------------------------------------------------------------


class TestCompareResults:
    def test_flip_detection(self):
        a = {r.case_id: r for r in [rec("t1", True), rec("t2", False), rec("t3", True)]}
        b = {r.case_id: r for r in [rec("t1", False), rec("t2", True), rec("t3", True)]}
        report = compare_results(a, b)

        assert report.n_paired == 3
        assert [f.case_id for f in report.regressed] == ["t1"]
        assert [f.case_id for f in report.improved] == ["t2"]
        assert report.both_pass == 1
        assert report.both_fail == 0

    def test_flips_visible_even_when_average_flat(self):
        """One regression + one improvement: average unchanged, flips reported."""
        a = {r.case_id: r for r in [rec("t1", True), rec("t2", False)]}
        b = {r.case_id: r for r in [rec("t1", False), rec("t2", True)]}
        report = compare_results(a, b)

        assert report.pass_stats.mean_diff == 0.0
        assert len(report.regressed) == 1
        assert len(report.improved) == 1

    def test_only_in_one_side_excluded_from_stats(self):
        a = {r.case_id: r for r in [rec("t1", True), rec("only_a", True)]}
        b = {r.case_id: r for r in [rec("t1", True), rec("only_b", False)]}
        report = compare_results(a, b)

        assert report.n_paired == 1
        assert report.only_in_a == ["only_a"]
        assert report.only_in_b == ["only_b"]
        assert report.pass_rate_a == 1.0  # only_a's pass not counted

    def test_no_overlap(self):
        report = compare_results(
            {"t1": rec("t1", True)}, {"t2": rec("t2", True)}
        )
        assert report.n_paired == 0
        assert "nothing to compare" in report.verdict

    def test_verdict_significant_improvement(self):
        a = {f"t{i}": rec(f"t{i}", False) for i in range(10)}
        b = {f"t{i}": rec(f"t{i}", True) for i in range(10)}
        report = compare_results(a, b)
        assert report.pass_stats.significant is True
        assert "improvement" in report.verdict

    def test_verdict_significant_regression(self):
        a = {f"t{i}": rec(f"t{i}", True) for i in range(10)}
        b = {f"t{i}": rec(f"t{i}", i < 5) for i in range(10)}
        report = compare_results(a, b)
        assert "regression" in report.verdict

    def test_verdict_noise_band_mentions_mde(self):
        # 1 flip out of 20: not significant
        a = {f"t{i}": rec(f"t{i}", True) for i in range(20)}
        b = {f"t{i}": rec(f"t{i}", i != 0) for i in range(20)}
        report = compare_results(a, b)
        assert report.pass_stats.significant is False
        assert "noise band" in report.verdict
        assert "detectable" in report.verdict

    def test_score_verdict_when_pass_flat(self):
        """No flips but scores consistently better: score CI catches it."""
        a = {f"t{i}": rec(f"t{i}", True, score=0.7) for i in range(10)}
        b = {f"t{i}": rec(f"t{i}", True, score=0.8) for i in range(10)}
        report = compare_results(a, b)
        assert report.pass_stats.significant is False
        assert report.score_stats.significant is True
        assert "score improvement" in report.verdict

    def test_trial_fraction_used_for_pass_rate(self):
        a = {"t1": CaseRecord("t1", True, 1.0, pass_fraction=0.6)}
        b = {"t1": CaseRecord("t1", True, 1.0, pass_fraction=1.0)}
        report = compare_results(a, b)
        assert report.pass_rate_a == 0.6
        assert report.pass_stats.mean_diff == pytest.approx(0.4)

    def test_to_dict_roundtrips_to_json(self):
        report = compare_results(
            {"t1": rec("t1", True)}, {"t1": rec("t1", False)}
        )
        data = json.loads(json.dumps(report.to_dict()))
        assert data["n_paired"] == 1
        assert data["regressed"][0]["case_id"] == "t1"
        assert data["verdict"]


# ---------------------------------------------------------------------------
# load_case_records
# ---------------------------------------------------------------------------


class TestLoadCaseRecords:
    def test_eval_result_shape(self, tmp_path):
        f = tmp_path / "run.json"
        f.write_text(json.dumps({
            "scenario_name": "s",
            "case_results": [case_dict("t1", True), case_dict("t2", False)],
        }))
        records = load_case_records(f)
        assert set(records) == {"t1", "t2"}
        assert records["t1"].passed is True

    def test_list_shape(self, tmp_path):
        f = tmp_path / "cases.json"
        f.write_text(json.dumps([case_dict("t1", True, 0.9)]))
        records = load_case_records(f)
        assert records["t1"].score == 0.9

    def test_single_case_shape_with_task_id(self, tmp_path):
        f = tmp_path / "case.json"
        f.write_text(json.dumps({"task_id": "t1", "passed": True}))
        records = load_case_records(f)
        assert records["t1"].score == 1.0  # falls back to passed

    def test_directory_skips_unrecognized_files(self, tmp_path):
        (tmp_path / "a.json").write_text(json.dumps([case_dict("t1", True)]))
        (tmp_path / "junk.json").write_text(json.dumps({"not": "a result"}))
        (tmp_path / "broken.json").write_text("{oops")
        records = load_case_records(tmp_path)
        assert set(records) == {"t1"}

    def test_trial_fields_populate_pass_fraction(self, tmp_path):
        f = tmp_path / "run.json"
        f.write_text(json.dumps([
            case_dict("t1", True, 0.8, total_trials=5, passed_trials=3),
        ]))
        records = load_case_records(f)
        assert records["t1"].pass_fraction == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Comparing on one grader (--on)
# ---------------------------------------------------------------------------


def grader(name: str, score: float | None, passed: bool, label: str = "", **kw) -> dict:
    d = {"name": name, "label": label, "score": score, "passed": passed}
    d.update(kw)
    return d


def gated_case(case_id: str, correctness: float, cost: float = 0.9) -> dict:
    """The shape this feature exists for: correctness decided by a gate, so
    `overall_score` carries only the non-gate process graders."""
    return case_dict(
        case_id,
        passed=correctness == 1.0,
        score=cost,  # overall_score excludes the gate — process cost only
        evaluator_results=[
            grader("integration_test", correctness, correctness == 1.0,
                   label="correctness", gate=True),
            grader("integration_test", 1.0, True, label="regression", gate=True),
            grader("cost_budget", cost, True),
        ],
    )


class TestSelectGraderScore:
    def test_label_takes_precedence_over_name(self):
        case = gated_case("t1", 0.5)
        assert select_grader_score(case, "correctness") == (0.5, False)
        assert select_grader_score(case, "regression") == (1.0, True)

    def test_an_unlabelled_grader_is_selected_by_name(self):
        case = gated_case("t1", 1.0)
        assert select_grader_score(case, "cost_budget") == (0.9, True)

    def test_a_labelled_grader_is_still_findable_by_name(self):
        """A scenario that labels one instance should not lose the ability to
        select it by the registry name when that name is unambiguous."""
        case = case_dict("t1", True, evaluator_results=[
            grader("rubric", 0.75, True, label="explains_itself"),
        ])
        assert select_grader_score(case, "rubric") == (0.75, True)
        assert select_grader_score(case, "explains_itself") == (0.75, True)

    def test_an_absent_grader_is_unmeasured_not_zero(self):
        """The distinction the whole design turns on: 'never measured' must not
        read as 'the agent scored 0'."""
        assert select_grader_score(gated_case("t1", 1.0), "nope") is None

    def test_a_skipped_or_crashed_grader_is_unmeasured(self):
        skipped = case_dict("t1", True, evaluator_results=[
            grader("rubric", None, False, skipped=True),
        ])
        crashed = case_dict("t2", True, evaluator_results=[
            grader("rubric", None, False, error="boom"),
        ])
        assert select_grader_score(skipped, "rubric") is None
        assert select_grader_score(crashed, "rubric") is None

    def test_two_graders_of_the_same_name_is_an_error_not_a_guess(self):
        """Picking one would answer a question nobody asked, and look fine."""
        case = case_dict("t1", True, evaluator_results=[
            grader("integration_test", 0.5, False),
            grader("integration_test", 1.0, True),
        ])
        with pytest.raises(AmbiguousGraderError, match="2 graders match"):
            select_grader_score(case, "integration_test")


class TestCompareOnGrader:
    @staticmethod
    def _write(tmp_path, name, cases):
        p = tmp_path / name
        p.write_text(json.dumps({"case_results": cases}), encoding="utf-8")
        return p

    def test_the_gated_score_is_invisible_by_default_and_visible_with_on(
        self, tmp_path
    ):
        """The motivating failure: A is wrong on two cases and B is right, but
        `overall_score` (process cost) says A is ahead."""
        a = self._write(tmp_path, "a.json", [
            gated_case("t1", 0.5, cost=0.99),
            gated_case("t2", 0.5, cost=0.99),
            gated_case("t3", 1.0, cost=0.99),
        ])
        b = self._write(tmp_path, "b.json", [
            gated_case("t1", 1.0, cost=0.60),
            gated_case("t2", 1.0, cost=0.60),
            gated_case("t3", 1.0, cost=0.60),
        ])

        default = compare_paths(a, b)
        assert default.mean_score_a > default.mean_score_b  # cost, not correctness

        scoped = compare_paths(a, b, on="correctness")
        assert scoped.on == "correctness"
        assert scoped.mean_score_a == pytest.approx(2 / 3)
        assert scoped.mean_score_b == 1.0
        assert [f.case_id for f in scoped.improved] == ["t1", "t2"]
        assert scoped.regressed == []

    def test_partial_credit_survives_into_the_statistics(self, tmp_path):
        """8-of-8 vs 4-of-8 is the resolution the case set was built for; a
        binary comparison would score both sides 0 and see nothing."""
        a = self._write(tmp_path, "a.json", [gated_case(f"t{i}", 0.5) for i in range(4)])
        b = self._write(tmp_path, "b.json", [gated_case(f"t{i}", 0.875) for i in range(4)])

        scoped = compare_paths(a, b, on="correctness")
        assert scoped.score_stats.mean_diff == pytest.approx(0.375)
        # Both sides fail every case, so pass/fail alone reports nothing.
        assert scoped.pass_stats.mean_diff == 0.0
        assert scoped.improved == [] and scoped.regressed == []

    def test_cases_without_the_grader_are_reported_not_scored_zero(self, tmp_path):
        a = self._write(tmp_path, "a.json", [
            gated_case("t1", 1.0),
            case_dict("t2", True, evaluator_results=[grader("cost_budget", 0.9, True)]),
        ])
        b = self._write(tmp_path, "b.json", [
            gated_case("t1", 1.0),
            case_dict("t2", True, evaluator_results=[grader("cost_budget", 0.9, True)]),
        ])

        scoped = compare_paths(a, b, on="correctness")
        assert scoped.n_paired == 1
        assert scoped.unmeasured_a == ["t2"] and scoped.unmeasured_b == ["t2"]
        # Scoring t2 as 0.0 would have dragged both means to 0.5 and invented a
        # sample out of a case the grader never looked at.
        assert scoped.mean_score_a == 1.0

    def test_zero_spread_does_not_claim_infinite_resolution(self, tmp_path):
        """Every case agreeing collapses the CI and MDE to 0. Reporting
        'detectable at n=4: ~0.0%' from four identical numbers is the exact
        misreading the MDE was added to prevent."""
        cases = [gated_case(f"t{i}", 1.0) for i in range(4)]
        a = self._write(tmp_path, "a.json", cases)
        b = self._write(tmp_path, "b.json", cases)

        scoped = compare_paths(a, b, on="correctness")
        assert scoped.pass_stats.variance_observed is False
        assert "cannot be estimated" in scoped.verdict
        assert "~0.0%" not in scoped.verdict

    def test_a_real_spread_still_reports_an_mde(self, tmp_path):
        a = self._write(tmp_path, "a.json", [gated_case(f"t{i}", 0.5) for i in range(4)])
        b = self._write(tmp_path, "b.json", [
            gated_case("t0", 1.0), gated_case("t1", 0.5),
            gated_case("t2", 1.0), gated_case("t3", 0.5),
        ])
        scoped = compare_paths(a, b, on="correctness")
        assert scoped.pass_stats.variance_observed is True
        assert scoped.pass_stats.mde > 0


# ---------------------------------------------------------------------------
# Comparing a metric (--metric)
# ---------------------------------------------------------------------------


class TestCompareMetric:
    @staticmethod
    def _write(tmp_path, name, cases):
        p = tmp_path / name
        p.write_text(json.dumps({"case_results": cases}), encoding="utf-8")
        return p

    @staticmethod
    def _case(cid, **metrics):
        return case_dict(cid, True, evaluator_results=[
            grader("turn_count", 1.0, True, metrics=metrics),
        ])

    def test_a_metric_comparison_is_a_paired_difference(self, tmp_path):
        a = self._write(tmp_path, "a.json", [self._case(f"t{i}", turns=20 + i)
                                             for i in range(4)])
        b = self._write(tmp_path, "b.json", [self._case(f"t{i}", turns=12 + i)
                                             for i in range(4)])
        cmp = compare_metric(a, b, "turns")
        assert cmp.n_paired == 4
        assert cmp.mean_a == 21.5 and cmp.mean_b == 13.5
        assert cmp.stats.mean_diff == pytest.approx(-8.0)
        # No spread in the differences: every case moved by exactly -8.
        assert cmp.stats.variance_observed is False
        assert "cannot be estimated" in cmp.verdict

    def test_a_metric_the_run_never_emitted_is_not_zero(self, tmp_path):
        """The distinction the whole module turns on, in its most dangerous
        form: an absent `cost_usd` means the cost grader did not run, and
        reading that as a run that cost nothing invents evidence."""
        a = self._write(tmp_path, "a.json", [self._case("t1", turns=10)])
        b = self._write(tmp_path, "b.json", [self._case("t1", turns=10)])
        cmp = compare_metric(a, b, "cost_usd")
        assert cmp.n_paired == 0
        assert cmp.unmeasured_a == ["t1"] and cmp.unmeasured_b == ["t1"]

    def test_missing_as_zero_is_opt_in_because_both_readings_are_right(
        self, tmp_path
    ):
        """`calls_grep` is absent when the tool was never used — a measured
        zero. `cost_usd` is absent when nothing measured it. Same shape,
        opposite meaning, so the caller says which."""
        a = self._write(tmp_path, "a.json", [
            self._case("t1", calls_grep=2.0), self._case("t2", turns=9),
        ])
        b = self._write(tmp_path, "b.json", [
            self._case("t1", calls_grep=5.0), self._case("t2", calls_grep=4.0),
        ])
        assert compare_metric(a, b, "calls_grep").n_paired == 1

        filled = compare_metric(a, b, "calls_grep", missing_as_zero=True)
        assert filled.n_paired == 2
        assert filled.mean_a == 1.0  # (2 + 0) / 2

    def test_missing_as_zero_still_skips_a_case_nothing_measured(self, tmp_path):
        """Otherwise a run that never graded reads as a run of zeros."""
        a = self._write(tmp_path, "a.json", [
            self._case("t1", calls_grep=2.0),
            case_dict("t2", True, evaluator_results=[]),
        ])
        b = self._write(tmp_path, "b.json", [
            self._case("t1", calls_grep=5.0),
            case_dict("t2", True, evaluator_results=[]),
        ])
        cmp = compare_metric(a, b, "calls_grep", missing_as_zero=True)
        assert cmp.n_paired == 1
        assert cmp.unmeasured_a == ["t2"]

    def test_a_boolean_metric_compares_as_a_rate(self, tmp_path):
        a = self._write(tmp_path, "a.json",
                        [self._case(f"t{i}", has_critical_loop=i < 3)
                         for i in range(4)])
        b = self._write(tmp_path, "b.json",
                        [self._case(f"t{i}", has_critical_loop=False)
                         for i in range(4)])
        cmp = compare_metric(a, b, "has_critical_loop")
        assert cmp.mean_a == 0.75 and cmp.mean_b == 0.0

    def test_two_graders_disagreeing_on_one_metric_is_an_error(self, tmp_path):
        case = case_dict("t1", True, evaluator_results=[
            grader("tool_usage", 1.0, True, metrics={"tool_calls": 12.0}),
            grader("loop_detection", 1.0, True, metrics={"tool_calls": 30.0}),
        ])
        a = self._write(tmp_path, "a.json", [case])
        with pytest.raises(AmbiguousGraderError, match="different values"):
            compare_metric(a, a, "tool_calls")

    def test_the_same_value_from_two_graders_is_not_ambiguous(self, tmp_path):
        """Only a disagreement is unresolvable. Two graders that counted the
        same thing and agree are not a reason to refuse an answer."""
        case = case_dict("t1", True, evaluator_results=[
            grader("tool_usage", 1.0, True, metrics={"tool_calls": 12.0}),
            grader("loop_detection", 1.0, True, metrics={"tool_calls": 12.0}),
        ])
        a = self._write(tmp_path, "a.json", [case])
        assert compare_metric(a, a, "tool_calls").mean_a == 12.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.fixture
def runner():
    return CliRunner(env={"COLUMNS": "200"})


@pytest.fixture
def two_runs(tmp_path):
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a.write_text(json.dumps({"case_results": [
        case_dict("t1", True, 0.9),
        case_dict("t2", False, 0.2),
        case_dict("t3", True, 0.8),
    ]}))
    b.write_text(json.dumps({"case_results": [
        case_dict("t1", True, 0.9),
        case_dict("t2", True, 0.7),
        case_dict("t3", False, 0.3),
    ]}))
    return a, b


class TestCompareCli:
    def test_basic_output(self, runner, two_runs):
        a, b = two_runs
        result = runner.invoke(cli, ["compare", str(a), str(b)])
        assert result.exit_code == 0
        assert "3 paired case(s)" in result.output
        assert "pass → fail" in result.output
        assert "fail → pass" in result.output
        assert "Verdict" in result.output

    def test_json_output(self, runner, two_runs):
        a, b = two_runs
        result = runner.invoke(cli, ["compare", str(a), str(b), "--json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["n_paired"] == 3

    def test_save_report(self, runner, two_runs, tmp_path):
        a, b = two_runs
        out = tmp_path / "cmp.json"
        result = runner.invoke(cli, ["compare", str(a), str(b), "-o", str(out)])
        assert result.exit_code == 0
        assert json.loads(out.read_text())["n_paired"] == 3

    def test_no_overlap_exits_nonzero(self, runner, tmp_path):
        a = tmp_path / "a.json"
        b = tmp_path / "b.json"
        a.write_text(json.dumps([case_dict("t1", True)]))
        b.write_text(json.dumps([case_dict("t2", True)]))
        result = runner.invoke(cli, ["compare", str(a), str(b)])
        assert result.exit_code == 1
        assert "No paired cases" in result.output

    def test_on_names_the_axis_in_the_output(self, runner, tmp_path):
        cases_a = [gated_case("t1", 0.5), gated_case("t2", 1.0)]
        cases_b = [gated_case("t1", 1.0), gated_case("t2", 1.0)]
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps({"case_results": cases_a}))
        b.write_text(json.dumps({"case_results": cases_b}))

        result = runner.invoke(cli, ["compare", str(a), str(b), "--on", "correctness"])
        assert result.exit_code == 0
        # The row labels have to say which numbers these are, or the table
        # claims to be about the case when it is about one grader.
        assert "scored on:     correctness" in result.output
        assert "Mean score (correctness)" in result.output

    def test_an_ambiguous_selector_fails_loudly(self, runner, tmp_path):
        case = case_dict("t1", True, evaluator_results=[
            grader("integration_test", 0.5, False),
            grader("integration_test", 1.0, True),
        ])
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        a.write_text(json.dumps({"case_results": [case]}))
        b.write_text(json.dumps({"case_results": [case]}))

        result = runner.invoke(
            cli, ["compare", str(a), str(b), "--on", "integration_test"]
        )
        assert result.exit_code == 2
        assert "Ambiguous" in result.output
        assert "label" in result.output

    def test_a_selector_matching_nothing_says_so(self, runner, tmp_path):
        """Not 'no paired cases between the two runs' — the cases are there."""
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        for p in (a, b):
            p.write_text(json.dumps({"case_results": [gated_case("t1", 1.0)]}))

        result = runner.invoke(cli, ["compare", str(a), str(b), "--on", "vlm_judge"])
        assert result.exit_code == 1
        assert "No case measured a grader named 'vlm_judge'" in result.output

    def test_unmeasured_cases_are_warned_about(self, runner, tmp_path):
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        for p in (a, b):
            p.write_text(json.dumps({"case_results": [
                gated_case("t1", 1.0),
                case_dict("t2", True,
                          evaluator_results=[grader("cost_budget", 0.9, True)]),
            ]}))

        result = runner.invoke(cli, ["compare", str(a), str(b), "--on", "correctness"])
        assert result.exit_code == 0
        assert "1 paired case(s)" in result.output
        assert "no 'correctness' measurement" in result.output
        assert "t2" in result.output
