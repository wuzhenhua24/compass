"""Tests for paired run comparison (compass compare)."""

from __future__ import annotations

import json
import math

import pytest
from click.testing import CliRunner

from compass.cli.main import cli
from compass.report.compare import (
    CaseRecord,
    compare_results,
    load_case_records,
    paired_stats,
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
