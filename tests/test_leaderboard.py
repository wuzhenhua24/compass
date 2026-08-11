"""Tests for the multi-model leaderboard.

A ranked table invites reading a winner out of noise. These tests pin the two
things that stop it: every row carries its standard error, and the gap between
the top two rows is stated as a paired measurement with a confidence interval.
"""

from __future__ import annotations

import pytest
from rich.errors import MarkupError

from compass.core.result import CaseResult, EvalResult, TestStatus
from compass.report.leaderboard import (
    build_leaderboard,
    case_records,
    slugify,
)

# ===================================================================
# Helpers
# ===================================================================


def _result(
    scores: dict[str, float],
    *,
    errors: list[str] | None = None,
    threshold: float = 0.5,
    trials: int = 1,
    trial_metrics: dict | None = None,
) -> EvalResult:
    cases: list[CaseResult] = []
    for case_id, score in scores.items():
        passed = score >= threshold
        cases.append(
            CaseResult(
                case_id=case_id,
                status=TestStatus.PASSED if passed else TestStatus.FAILED,
                passed=passed,
                overall_score=score,
                total_trials=trials,
                passed_trials=trials if passed else 0,
                trial_metrics=trial_metrics,
            )
        )
    for case_id in errors or []:
        cases.append(
            CaseResult(
                case_id=case_id, status=TestStatus.ERROR, passed=False,
                overall_score=0.0, error="boom", error_trials=trials,
                total_trials=trials,
            )
        )
    return EvalResult(
        scenario_name="s",
        total_cases=len(cases),
        passed_cases=sum(1 for c in cases if c.passed),
        failed_cases=sum(
            1 for c in cases if c.status == TestStatus.FAILED
        ),
        error_cases=len(errors or []),
        case_results=cases,
    )


# ===================================================================
# Ranking
# ===================================================================


class TestRanking:
    def test_highest_mean_score_leads(self):
        board = build_leaderboard(
            [
                ("weak", _result({"a": 0.2, "b": 0.3})),
                ("strong", _result({"a": 0.9, "b": 1.0})),
                ("mid", _result({"a": 0.6, "b": 0.7})),
            ]
        )
        assert [e.label for e in board.entries] == ["strong", "mid", "weak"]
        assert [e.rank for e in board.entries] == [1, 2, 3]

    def test_ties_share_a_rank(self):
        """Printing 1. and 2. for two identical scores contradicts the display."""
        board = build_leaderboard(
            [
                ("a", _result({"c1": 0.8, "c2": 0.8})),
                ("b", _result({"c1": 0.8, "c2": 0.8})),
                ("c", _result({"c1": 0.1, "c2": 0.1})),
            ]
        )
        assert [e.rank for e in board.entries] == [1, 1, 3]

    def test_a_single_run_is_not_a_ranking(self):
        board = build_leaderboard([("only", _result({"a": 1.0}))])
        assert board.entries[0].rank == 1
        assert board.top_gap is None
        assert "nothing to rank against" in board.verdict

    def test_empty_input(self):
        board = build_leaderboard([])
        assert board.entries == []


# ===================================================================
# Uncertainty travels with the score
# ===================================================================


class TestUncertainty:
    def test_stderr_is_reported_alongside_the_mean(self):
        board = build_leaderboard(
            [("m", _result({"a": 1.0, "b": 0.0, "c": 0.5, "d": 0.5}))]
        )
        entry = board.entries[0]
        assert entry.mean_score == pytest.approx(0.5)
        assert entry.stderr > 0
        assert "±" in entry.score_display

    def test_a_single_case_has_no_stderr_to_show(self):
        board = build_leaderboard([("m", _result({"a": 1.0}))])
        assert board.entries[0].stderr == 0.0
        assert "±" not in board.entries[0].score_display

    def test_a_close_race_is_reported_as_noise(self):
        """The whole point: adjacent rows are usually not distinguishable."""
        board = build_leaderboard(
            [
                ("a", _result({"c1": 1.0, "c2": 0.0, "c3": 1.0, "c4": 0.0})),
                ("b", _result({"c1": 0.0, "c2": 1.0, "c3": 0.0, "c4": 0.9})),
            ]
        )
        assert board.top_gap is not None
        assert not board.top_gap.significant
        assert "within noise" in board.verdict

    def test_a_real_gap_is_reported_as_significant(self):
        board = build_leaderboard(
            [
                ("strong", _result({f"c{i}": 0.9 for i in range(6)})),
                ("weak", _result({f"c{i}": 0.2 for i in range(6)})),
            ]
        )
        assert board.top_gap.significant
        assert "beats" in board.verdict

    def test_a_tie_says_tie_rather_than_testing_a_gap(self):
        board = build_leaderboard(
            [("a", _result({"c1": 0.5})), ("b", _result({"c1": 0.5}))]
        )
        assert "tied" in board.verdict


# ===================================================================
# The P1 hygiene rules carry over
# ===================================================================


class TestHygiene:
    def test_harness_errors_are_out_of_the_denominator(self):
        board = build_leaderboard(
            [("m", _result({"a": 1.0, "b": 1.0}, errors=["c"]))]
        )
        entry = board.entries[0]
        assert entry.evaluated_cases == 2
        assert entry.error_cases == 1
        assert entry.pass_rate == 1.0
        assert entry.mean_score == pytest.approx(1.0)  # the error contributes no 0.0

    def test_errored_cases_are_not_paired(self):
        """A missing sample cannot take part in a paired comparison."""
        records = case_records(_result({"a": 1.0}, errors=["b"]))
        assert set(records) == {"a"}

    def test_reliability_reports_pass_hat_k_when_available(self):
        metrics = {"pass_rate": 0.6, "pass_all_3": 0.4, "pass_at_3": 0.9}
        board = build_leaderboard(
            [("m", _result({"a": 0.8, "b": 0.8}, trials=3, trial_metrics=metrics))]
        )
        assert board.entries[0].reliability == pytest.approx(0.4)
        assert board.entries[0].reliability_k == 3

    def test_reliability_is_omitted_for_single_trial_runs(self):
        board = build_leaderboard([("m", _result({"a": 0.8}))])
        assert board.entries[0].reliability is None

    def test_reliability_is_omitted_when_k_differs_across_cases(self):
        """Averaging pass^3 with pass^5 would be a meaningless number."""
        result = _result({"a": 0.8, "b": 0.8}, trials=3)
        result.case_results[0].trial_metrics = {"pass_all_3": 0.4}
        result.case_results[1].trial_metrics = {"pass_all_5": 0.2}
        board = build_leaderboard([("m", result)])
        assert board.entries[0].reliability is None


# ===================================================================
# Serialization and labelling
# ===================================================================


class TestSerializationAndLabels:
    def test_to_dict_carries_the_verdict(self):
        board = build_leaderboard(
            [("a", _result({"c1": 0.9})), ("b", _result({"c1": 0.1}))]
        )
        d = board.to_dict()
        assert [e["label"] for e in d["entries"]] == ["a", "b"]
        assert d["verdict"] == board.verdict
        assert "top_gap" in d

    def test_unpaired_cases_are_reported(self):
        """Coverage differences must be visible, not silently dropped."""
        board = build_leaderboard(
            [
                ("a", _result({"c1": 0.9, "c2": 0.9})),
                ("b", _result({"c1": 0.1, "c3": 0.1})),
            ]
        )
        assert board.unpaired == ["c2", "c3"]

    def test_slugify_makes_model_names_path_safe(self):
        assert slugify("anthropic/claude-sonnet-5") == "anthropic-claude-sonnet-5"
        assert slugify("gpt-4.1-mini") == "gpt-4.1-mini"
        assert slugify("///") == "run"


# ===================================================================
# CLI wiring
# ===================================================================


class TestModelVariants:
    def _scenario(self, name="s"):
        from compass.core.scenario import (
            AgentConfig,
            InputConfig,
            Scenario,
            TestCase,
        )

        return Scenario(
            name=name,
            agent=AgentConfig(adapter="image", config={"model": "default"}),
            cases=[TestCase(id="c1", input=InputConfig(prompt="x"))],
        )

    def test_no_models_is_the_identity(self):
        from compass.cli.main import _model_variants

        scn = self._scenario()
        variants = _model_variants([scn], [], "model")
        assert variants == [("s", scn)]

    def test_each_model_gets_its_own_variant(self):
        from compass.cli.main import _model_variants

        variants = _model_variants([self._scenario()], ["a", "b"], "model")

        assert [label for label, _ in variants] == ["a", "b"]
        assert [v.agent.config["model"] for _, v in variants] == ["a", "b"]
        assert [v.name for _, v in variants] == ["s [a]", "s [b]"]

    def test_the_original_scenario_is_not_mutated(self):
        from compass.cli.main import _model_variants

        scn = self._scenario()
        _model_variants([scn], ["a", "b"], "model")
        assert scn.agent.config["model"] == "default"
        assert scn.name == "s"

    def test_a_custom_key_can_be_overridden(self):
        from compass.cli.main import _model_variants

        variants = _model_variants([self._scenario()], ["v2"], "endpoint")
        assert variants[0][1].agent.config["endpoint"] == "v2"
        assert variants[0][1].agent.config["model"] == "default"

    def test_labels_disambiguate_across_scenarios(self):
        from compass.cli.main import _model_variants

        variants = _model_variants(
            [self._scenario("one"), self._scenario("two")], ["a"], "model"
        )
        assert [label for label, _ in variants] == ["one / a", "two / a"]


class TestVariantNamesAreNotRichMarkup:
    """Regression: a variant name is user data, and the display parsed it.

    ``_model_variants`` writes the axis value into the scenario name as
    ``name [value]``. Sweeping the *prompt* axis — the whole point of
    ``--model-key`` — puts a file path there, and rich read ``[/path/...]`` as
    a closing tag and raised MarkupError before a single case had run.
    """

    def _run_variant_scenario(self, value: str):
        from click.testing import CliRunner

        from compass.cli.main import cli

        runner = CliRunner()
        with runner.isolated_filesystem():
            from pathlib import Path

            Path("s.yaml").write_text(
                "name: S\n"
                "agent:\n"
                "  adapter: image\n"
                "  config: {model: default}\n"
                "cases:\n"
                "  - id: c1\n"
                "    input: {prompt: x}\n",
                encoding="utf-8",
            )
            return runner.invoke(
                cli, ["test", "s.yaml", "--model-key", "prompt_file", "-m", value]
            )

    def test_a_path_valued_variant_does_not_crash_the_display(self):
        result = self._run_variant_scenario("/abs/prompts/terse.md")
        assert not isinstance(result.exception, MarkupError)
        assert "MarkupError" not in (result.output or "")

    def test_square_brackets_in_a_variant_are_shown_literally(self):
        result = self._run_variant_scenario("[weird]")
        assert not isinstance(result.exception, MarkupError)
        # Displayed verbatim rather than consumed as a style tag.
        assert "S [[weird]]" in (result.output or "")


class TestListCommandDoesNotShadowBuiltins:
    """Regression: a CLI command named `list` shadowed the builtin.

    Every `list(...)` call in the module then invoked the command instead —
    which silently broke `compass test --case/--stage/--category`.
    """

    def test_the_command_is_still_called_list(self):
        from compass.cli.main import cli

        assert "list" in cli.commands

    def test_the_builtin_is_reachable_from_the_module(self):
        import compass.cli.main as main

        assert getattr(main, "list", list) is list

    def test_case_filtering_actually_filters(self):
        from click.testing import CliRunner

        from compass.cli.main import cli

        runner = CliRunner()
        with runner.isolated_filesystem():
            from pathlib import Path

            Path("s.yaml").write_text(
                "name: s\n"
                "agent: {adapter: image}\n"
                "cases:\n"
                "  - id: c1\n    input: {prompt: x}\n"
                "  - id: c2\n    input: {prompt: y}\n"
            )
            result = runner.invoke(cli, ["test", "s.yaml", "-c", "c1"])

        # Before the fix this printed the grader list and exited 2
        assert "Available Graders" not in result.output
        assert "Compass Test Runner" in result.output
