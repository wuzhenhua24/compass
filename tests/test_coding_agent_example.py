"""Tests for the examples/coding_agent template.

An example that silently rots is worse than no example — someone clones it,
runs it, and gets a green wall that proves nothing. These pin the parts that
drift: the fixtures' Edit anchors against the project files, the hidden
acceptance tests against the implementations the fixtures write, and the
end-to-end verdict for each behaviour.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "coding_agent"


def _load(filename: str, alias: str) -> ModuleType:
    """Import an example module under a unique name.

    Not ``sys.path.insert`` + a bare import: several examples have an
    ``eval.py`` and a ``fixtures.py``, so the plain form makes whichever test
    module imports first win and the other silently test the wrong code.
    """
    path = _EXAMPLE / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


coding_eval = _load("eval.py", "coding_agent_eval")
fixtures = _load("fixtures.py", "coding_agent_fixtures")


# ---------------------------------------------------------------------------
# Fixtures vs. the project files they edit
# ---------------------------------------------------------------------------


class TestFixtureAnchors:
    """Every Edit anchor must exist in the file it targets.

    This is the failure the replay CLI shouts about at runtime; catching it in
    CI means the example never ships broken.
    """

    @pytest.mark.parametrize("requirement", fixtures.REQUIREMENTS)
    @pytest.mark.parametrize("behaviour", fixtures.BEHAVIOURS)
    def test_anchors_resolve_against_a_fresh_checkout(
        self, requirement, behaviour, tmp_path
    ):
        import shutil

        project = tmp_path / "project"
        shutil.copytree(_EXAMPLE / "project", project)

        prompt = {"add_discount": "apply_discount", "fix_rounding": "line_total"}[
            requirement
        ]
        for event in fixtures.run_for(prompt, behaviour):
            if event.get("type") != "assistant":
                continue
            for block in event["message"].get("content", []):
                if block.get("type") != "tool_use" or block["name"] != "Edit":
                    continue
                path = project / block["input"]["file_path"]
                content = path.read_text(encoding="utf-8")
                old = block["input"]["old_string"]
                assert old in content, (
                    f"{requirement}/{behaviour}: anchor missing from "
                    f"{block['input']['file_path']}"
                )
                path.write_text(
                    content.replace(old, block["input"]["new_string"], 1),
                    encoding="utf-8",
                )


class TestRunSelection:
    def test_prompt_maps_to_a_requirement(self):
        assert fixtures.requirement_of("add apply_discount(total, tier)") == (
            "add_discount"
        )
        assert fixtures.requirement_of("line_total leaves dust") == "fix_rounding"

    def test_an_unmatched_prompt_is_a_loud_error(self):
        with pytest.raises(KeyError, match="no canned run"):
            fixtures.requirement_of("refactor the billing module")

    def test_an_unknown_behaviour_is_a_loud_error(self):
        with pytest.raises(KeyError, match="no canned run"):
            fixtures.run_for("apply_discount", "sandbagging")

    def test_every_pair_has_a_run(self):
        for requirement in fixtures.REQUIREMENTS:
            prompt = {"add_discount": "apply_discount", "fix_rounding": "line_total"}[
                requirement
            ]
            for behaviour in fixtures.BEHAVIOURS:
                assert fixtures.run_for(prompt, behaviour)


# ---------------------------------------------------------------------------
# The scenario
# ---------------------------------------------------------------------------


class TestScenario:
    def _scenario(self, tmp_path):
        return coding_eval.build_scenario(tmp_path / "repo", "honest")

    def test_placeholders_are_all_filled(self, tmp_path):
        scenario = self._scenario(tmp_path)
        rendered = yaml.dump(scenario.model_dump(), allow_unicode=True)
        assert "{{" not in rendered

    def test_hidden_tests_live_outside_the_project(self):
        """The scoring signal must be somewhere the agent cannot read or edit."""
        graders_dir = _EXAMPLE / "grader_tests"
        assert graders_dir.is_dir()
        assert not (_EXAMPLE / "project" / "grader_tests").exists()
        assert list(graders_dir.glob("test_*.py"))

    def test_integration_test_targets_the_trial_workspace(self, tmp_path):
        scenario = self._scenario(tmp_path)
        for case in scenario.cases:
            grader = next(
                g for g in case.graders if g.name == "integration_test"
            )
            assert grader.config["workdir"] == "{workspace}"
            assert grader.gate is True

    def test_process_guards_apply_to_every_case(self, tmp_path):
        names = {g.name for g in self._scenario(tmp_path).default_graders}
        assert names == {
            "state_delta",
            "cost_budget",
            "turn_count",
            "loop_detection",
            "tool_usage",
        }

    def test_the_replay_cli_is_the_only_substitution(self, tmp_path):
        config = self._scenario(tmp_path).agent.config
        assert config["cli_path"].endswith("replay_cli.py")
        assert self._scenario(tmp_path).agent.adapter == "claude_code"


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


class TestEndToEnd:
    """The whole demo, through the real adapter. Slow-ish but it is the point."""

    @staticmethod
    async def _run(behaviour: str, tmp_path: Path):
        from compass.core.runner import Compass

        repo = coding_eval.bootstrap_repo(tmp_path)
        scenario = coding_eval.build_scenario(repo, behaviour)
        return await Compass().run(scenario)

    async def test_honest_passes_every_case(self, tmp_path):
        result = await self._run("honest", tmp_path)
        assert result.pass_rate == 1.0
        assert result.error_cases == 0

    async def test_cheats_is_caught_by_both_the_tests_and_the_delta(self, tmp_path):
        result = await self._run("cheats", tmp_path)
        assert result.pass_rate == 0.0

        case = result.case_results[0]
        by_name = {g.name: g for g in case.evaluator_results}

        # Wrong logic: the hidden tests fail, with partial credit.
        assert by_name["integration_test"].passed is False
        assert 0.0 < by_name["integration_test"].score < 1.0

        # ... and the test edit is caught independently.
        delta = by_name["state_delta"]
        assert delta.passed is False
        assert any(
            "tests/" in str(v) for v in delta.metadata.get("violations", [])
        )

    async def test_overreach_is_correct_and_still_fails(self, tmp_path):
        """The lesson: a right answer is not a shippable agent."""
        result = await self._run("overreach", tmp_path)
        assert result.pass_rate == 0.0

        by_name = {
            g.name: g for g in result.case_results[0].evaluator_results
        }
        assert by_name["integration_test"].passed is True   # the answer is right
        assert by_name["cost_budget"].passed is False
        assert by_name["turn_count"].passed is False
        assert by_name["loop_detection"].passed is False
        assert by_name["state_delta"].passed is False

    async def test_the_agent_never_touches_the_source_repo(self, tmp_path):
        import subprocess

        repo = coding_eval.bootstrap_repo(tmp_path)
        scenario = coding_eval.build_scenario(repo, "overreach")
        from compass.core.runner import Compass

        await Compass().run(scenario)

        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        assert status.stdout == ""

    async def test_the_demo_verdict_is_what_the_readme_claims(self, tmp_path):
        """honest passes, the other two do not — the README's whole table."""
        outcomes = {}
        for behaviour in fixtures.BEHAVIOURS:
            result = await self._run(behaviour, tmp_path / behaviour)
            outcomes[behaviour] = result.pass_rate == 1.0
        assert outcomes == {"honest": True, "cheats": False, "overreach": False}
