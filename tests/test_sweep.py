"""Tests for parameter sweep functionality."""


import pytest
import yaml

from compass.core.result import CaseResult, TestStatus
from compass.core.scenario import (
    GraderConfig,
    InputConfig,
    Scenario,
    TestCase,
)
from compass.core.sweep import (
    SweepConfig,
    SweepSummary,
    expand_sweeps,
    group_sweep_results,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_case(
    case_id: str = "test_case",
    prompt: str = "A cat",
    params: dict | None = None,
    sweep: SweepConfig | None = None,
    graders: list[GraderConfig] | None = None,
    tags: list[str] | None = None,
    expect: str = "pass",
    metadata: dict | None = None,
) -> TestCase:
    return TestCase(
        id=case_id,
        input=InputConfig(prompt=prompt, params=params or {}),
        sweep=sweep,
        graders=graders or [],
        tags=tags or [],
        expect=expect,
        metadata=metadata or {},
    )


def _make_case_result(
    case_id: str,
    passed: bool = True,
    score: float = 1.0,
    input_data: dict | None = None,
) -> CaseResult:
    return CaseResult(
        case_id=case_id,
        status=TestStatus.PASSED if passed else TestStatus.FAILED,
        passed=passed,
        overall_score=score,
        input_data=input_data or {},
    )


# ===========================================================================
# TestSweepConfig
# ===========================================================================


class TestSweepConfig:
    """Tests for SweepConfig model."""

    def test_empty_params(self):
        cfg = SweepConfig()
        assert cfg.params == {}
        assert cfg.total_combinations == 0
        assert cfg.combinations() == []

    def test_single_param(self):
        cfg = SweepConfig(params={"steps": [20, 30, 50]})
        assert cfg.total_combinations == 3
        combos = cfg.combinations()
        assert combos == [{"steps": 20}, {"steps": 30}, {"steps": 50}]

    def test_multi_param_cartesian(self):
        cfg = SweepConfig(params={"steps": [20, 30], "cfg_scale": [5.0, 7.5]})
        assert cfg.total_combinations == 4
        combos = cfg.combinations()
        # Keys sorted alphabetically: cfg_scale before steps
        assert combos == [
            {"cfg_scale": 5.0, "steps": 20},
            {"cfg_scale": 5.0, "steps": 30},
            {"cfg_scale": 7.5, "steps": 20},
            {"cfg_scale": 7.5, "steps": 30},
        ]

    def test_total_combinations_three_params(self):
        cfg = SweepConfig(params={"a": [1, 2], "b": [3, 4, 5], "c": [6]})
        assert cfg.total_combinations == 6  # 2 * 3 * 1

    def test_empty_value_list_raises(self):
        with pytest.raises(ValueError, match="empty value list"):
            SweepConfig(params={"steps": []})

    def test_from_dict(self):
        data = {"params": {"quality": ["low", "high"]}}
        cfg = SweepConfig.model_validate(data)
        assert cfg.total_combinations == 2
        assert cfg.combinations() == [
            {"quality": "low"},
            {"quality": "high"},
        ]


# ===========================================================================
# TestExpandSweeps
# ===========================================================================


class TestExpandSweeps:
    """Tests for expand_sweeps function."""

    def test_no_sweep_passthrough(self):
        cases = [_make_case("c1"), _make_case("c2")]
        result = expand_sweeps(cases)
        assert len(result) == 2
        assert result[0].id == "c1"
        assert result[1].id == "c2"

    def test_single_param_expansion(self):
        sweep = SweepConfig(params={"steps": [20, 30]})
        cases = [_make_case("cat", sweep=sweep)]
        result = expand_sweeps(cases)
        assert len(result) == 2
        assert result[0].id == "cat__sweep__steps=20"
        assert result[1].id == "cat__sweep__steps=30"

    def test_multi_param_expansion(self):
        sweep = SweepConfig(params={"steps": [20, 30], "cfg": [5.0, 7.5]})
        cases = [_make_case("cat", sweep=sweep)]
        result = expand_sweeps(cases)
        assert len(result) == 4
        # Keys sorted: cfg before steps
        assert result[0].id == "cat__sweep__cfg=5.0__steps=20"
        assert result[1].id == "cat__sweep__cfg=5.0__steps=30"
        assert result[2].id == "cat__sweep__cfg=7.5__steps=20"
        assert result[3].id == "cat__sweep__cfg=7.5__steps=30"

    def test_id_format(self):
        sweep = SweepConfig(params={"z_param": [1], "a_param": [2]})
        cases = [_make_case("base", sweep=sweep)]
        result = expand_sweeps(cases)
        assert len(result) == 1
        # a_param before z_param (sorted)
        assert result[0].id == "base__sweep__a_param=2__z_param=1"

    def test_params_merged(self):
        """Sweep params merge with (and override) base params."""
        sweep = SweepConfig(params={"steps": [50]})
        case = _make_case("c", params={"width": 1024, "steps": 20}, sweep=sweep)
        result = expand_sweeps([case])
        assert len(result) == 1
        assert result[0].input.params == {"width": 1024, "steps": 50}

    def test_metadata_correct(self):
        sweep = SweepConfig(params={"steps": [20]})
        case = _make_case("c", metadata={"existing": "value"}, sweep=sweep)
        result = expand_sweeps([case])
        meta = result[0].metadata
        assert meta["existing"] == "value"
        assert meta["sweep"]["base_case_id"] == "c"
        assert meta["sweep"]["sweep_index"] == 0
        assert meta["sweep"]["sweep_params"] == {"steps": 20}

    def test_sweep_cleared(self):
        """Expanded cases should have sweep=None to prevent re-expansion."""
        sweep = SweepConfig(params={"steps": [20, 30]})
        cases = [_make_case("c", sweep=sweep)]
        result = expand_sweeps(cases)
        for r in result:
            assert r.sweep is None

    def test_graders_preserved(self):
        grader = GraderConfig(name="semantic_match", weight=0.5)
        sweep = SweepConfig(params={"steps": [20]})
        case = _make_case("c", graders=[grader], sweep=sweep)
        result = expand_sweeps([case])
        assert len(result[0].graders) == 1
        assert result[0].graders[0].name == "semantic_match"

    def test_expect_preserved(self):
        sweep = SweepConfig(params={"steps": [20]})
        case = _make_case("c", expect="fail", sweep=sweep)
        result = expand_sweeps([case])
        assert result[0].expect == "fail"

    def test_tags_preserved(self):
        sweep = SweepConfig(params={"steps": [20]})
        case = _make_case("c", tags=["smoke", "fast"], sweep=sweep)
        result = expand_sweeps([case])
        assert result[0].tags == ["smoke", "fast"]

    def test_case_level_sweep(self):
        """Case-level sweep is used even when default_sweep is provided."""
        case_sweep = SweepConfig(params={"steps": [10]})
        default_sweep = SweepConfig(params={"steps": [20, 30]})
        cases = [_make_case("c", sweep=case_sweep)]
        result = expand_sweeps(cases, default_sweep=default_sweep)
        assert len(result) == 1  # Case sweep has 1 combo, not 2
        assert result[0].input.params["steps"] == 10

    def test_scenario_level_default_sweep(self):
        """Default sweep applies to cases without their own sweep."""
        default_sweep = SweepConfig(params={"steps": [20, 30]})
        cases = [_make_case("c1"), _make_case("c2")]
        result = expand_sweeps(cases, default_sweep=default_sweep)
        assert len(result) == 4  # 2 cases x 2 combos

    def test_case_overrides_scenario(self):
        """Case sweep overrides scenario default sweep."""
        case_sweep = SweepConfig(params={"quality": ["low", "high"]})
        default_sweep = SweepConfig(params={"steps": [20, 30, 50]})
        cases = [
            _make_case("c1", sweep=case_sweep),  # Uses case sweep (2 combos)
            _make_case("c2"),  # Uses default sweep (3 combos)
        ]
        result = expand_sweeps(cases, default_sweep=default_sweep)
        assert len(result) == 5  # 2 + 3

    def test_mixed_sweep_and_no_sweep(self):
        """Cases with and without sweep coexist."""
        sweep = SweepConfig(params={"steps": [20, 30]})
        cases = [
            _make_case("sweep_case", sweep=sweep),
            _make_case("normal_case"),
        ]
        result = expand_sweeps(cases)
        assert len(result) == 3  # 2 expanded + 1 normal
        ids = [r.id for r in result]
        assert "sweep_case__sweep__steps=20" in ids
        assert "sweep_case__sweep__steps=30" in ids
        assert "normal_case" in ids


# ===========================================================================
# TestSweepSummary
# ===========================================================================


class TestSweepSummary:
    """Tests for SweepSummary."""

    def test_pass_rate(self):
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=4,
            passed_combinations=3,
            results=[],
        )
        assert summary.pass_rate == 0.75

    def test_pass_rate_zero(self):
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=0,
            passed_combinations=0,
        )
        assert summary.pass_rate == 0.0

    def test_best_combination(self):
        results = [
            _make_case_result(
                "c__sweep__steps=20", score=0.5,
                input_data={"sweep": {"sweep_params": {"steps": 20}}},
            ),
            _make_case_result(
                "c__sweep__steps=30", score=0.9,
                input_data={"sweep": {"sweep_params": {"steps": 30}}},
            ),
        ]
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=2,
            passed_combinations=2,
            results=results,
        )
        assert summary.best_combination == {"steps": 30}

    def test_worst_combination(self):
        results = [
            _make_case_result(
                "c__sweep__steps=20", score=0.5,
                input_data={"sweep": {"sweep_params": {"steps": 20}}},
            ),
            _make_case_result(
                "c__sweep__steps=30", score=0.9,
                input_data={"sweep": {"sweep_params": {"steps": 30}}},
            ),
        ]
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=2,
            passed_combinations=2,
            results=results,
        )
        assert summary.worst_combination == {"steps": 20}

    def test_best_worst_empty(self):
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=0,
            passed_combinations=0,
        )
        assert summary.best_combination is None
        assert summary.worst_combination is None

    def test_score_by_param(self):
        results = [
            _make_case_result(
                "c__sweep__quality=low__steps=20", score=0.4,
                input_data={"sweep": {"sweep_params": {"quality": "low", "steps": 20}}},
            ),
            _make_case_result(
                "c__sweep__quality=low__steps=30", score=0.6,
                input_data={"sweep": {"sweep_params": {"quality": "low", "steps": 30}}},
            ),
            _make_case_result(
                "c__sweep__quality=high__steps=20", score=0.8,
                input_data={"sweep": {"sweep_params": {"quality": "high", "steps": 20}}},
            ),
            _make_case_result(
                "c__sweep__quality=high__steps=30", score=1.0,
                input_data={"sweep": {"sweep_params": {"quality": "high", "steps": 30}}},
            ),
        ]
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=4,
            passed_combinations=4,
            results=results,
        )
        by_quality = summary.score_by_param("quality")
        assert by_quality["low"] == pytest.approx(0.5)  # (0.4+0.6)/2
        assert by_quality["high"] == pytest.approx(0.9)  # (0.8+1.0)/2

        by_steps = summary.score_by_param("steps")
        assert by_steps[20] == pytest.approx(0.6)  # (0.4+0.8)/2
        assert by_steps[30] == pytest.approx(0.8)  # (0.6+1.0)/2

    def test_to_dict(self):
        summary = SweepSummary(
            base_case_id="c",
            total_combinations=2,
            passed_combinations=1,
        )
        d = summary.to_dict()
        assert d["base_case_id"] == "c"
        assert d["total_combinations"] == 2
        assert d["passed_combinations"] == 1
        assert d["pass_rate"] == 0.5


# ===========================================================================
# TestGroupSweepResults
# ===========================================================================


class TestGroupSweepResults:
    """Tests for group_sweep_results function."""

    def test_group_by_base_case_id(self):
        results = [
            _make_case_result("cat__sweep__steps=20", score=0.5),
            _make_case_result("cat__sweep__steps=30", score=0.8),
        ]
        summaries, non_sweep = group_sweep_results(results)
        assert len(summaries) == 1
        assert len(non_sweep) == 0
        assert summaries[0].base_case_id == "cat"
        assert summaries[0].total_combinations == 2

    def test_non_sweep_separated(self):
        results = [
            _make_case_result("cat__sweep__steps=20"),
            _make_case_result("normal_case"),
        ]
        summaries, non_sweep = group_sweep_results(results)
        assert len(summaries) == 1
        assert len(non_sweep) == 1
        assert non_sweep[0].case_id == "normal_case"

    def test_multiple_base_cases(self):
        results = [
            _make_case_result("cat__sweep__steps=20"),
            _make_case_result("cat__sweep__steps=30"),
            _make_case_result("dog__sweep__steps=20"),
            _make_case_result("normal"),
        ]
        summaries, non_sweep = group_sweep_results(results)
        assert len(summaries) == 2
        assert len(non_sweep) == 1
        base_ids = {s.base_case_id for s in summaries}
        assert base_ids == {"cat", "dog"}


# ===========================================================================
# TestSweepYAML
# ===========================================================================


class TestSweepYAML:
    """Tests for YAML parsing with sweep configs."""

    def test_case_level_sweep_yaml(self, tmp_path):
        yaml_content = {
            "name": "Sweep Test",
            "agent": {"adapter": "image"},
            "cases": [
                {
                    "id": "cat",
                    "input": {"prompt": "A cat"},
                    "sweep": {"params": {"steps": [20, 30]}},
                    "graders": [{"name": "semantic_match"}],
                }
            ],
        }
        path = tmp_path / "sweep.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        scenario = Scenario.from_yaml(path)
        assert scenario.cases[0].sweep is not None
        assert scenario.cases[0].sweep.total_combinations == 2

    def test_scenario_level_sweep_yaml(self, tmp_path):
        yaml_content = {
            "name": "Sweep Test",
            "agent": {"adapter": "image"},
            "sweep": {"params": {"steps": [20, 30, 50]}},
            "cases": [
                {"id": "cat", "input": {"prompt": "A cat"}},
                {"id": "dog", "input": {"prompt": "A dog"}},
            ],
        }
        path = tmp_path / "sweep.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        scenario = Scenario.from_yaml(path)
        assert scenario.sweep is not None
        assert scenario.sweep.total_combinations == 3
        # Cases don't have their own sweep
        for case in scenario.cases:
            assert case.sweep is None

    def test_no_sweep_backward_compat(self, tmp_path):
        yaml_content = {
            "name": "No Sweep",
            "agent": {"adapter": "image"},
            "cases": [
                {"id": "cat", "input": {"prompt": "A cat"}},
            ],
        }
        path = tmp_path / "no_sweep.yaml"
        with open(path, "w") as f:
            yaml.dump(yaml_content, f)

        scenario = Scenario.from_yaml(path)
        assert scenario.sweep is None
        assert scenario.cases[0].sweep is None
