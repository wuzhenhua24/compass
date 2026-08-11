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


# ---------------------------------------------------------------------------
# suite.yaml — the case set users copy for their own project
# ---------------------------------------------------------------------------

check = _load("check.py", "coding_agent_check")


def _suite() -> dict:
    return yaml.safe_load((_EXAMPLE / "suite.yaml").read_text(encoding="utf-8"))


def _live_cases() -> list[dict]:
    return [c for c in _suite()["cases"] if not c.get("metadata", {}).get("todo")]


class TestSuiteShape:
    def test_every_live_case_has_hidden_tests(self):
        for case in _live_cases():
            path = _EXAMPLE / "grader_tests" / f"test_{case['id']}.py"
            assert path.exists(), f"{case['id']} has no {path.name}"

    def test_every_live_case_has_a_reference_solution(self):
        """Without one, nobody has shown the task is solvable."""
        for case in _live_cases():
            solution = _EXAMPLE / "solutions" / case["id"]
            assert solution.is_dir(), f"{case['id']} has no solutions/{case['id']}/"

    def test_every_live_case_gates_on_both_correctness_and_regression(self):
        """Hidden tests say the work is right; the repo suite says nothing else
        broke. A case with only the first cannot catch the regression trap."""
        for case in _live_cases():
            scripts = [
                g["config"]["script"]
                for g in case["graders"]
                if g["name"] == "integration_test" and g.get("gate")
            ]
            assert any("{{GRADERS}}" in s for s in scripts), case["id"]
            assert any("pytest tests/" in s for s in scripts), case["id"]

    def test_no_unfilled_slot_is_a_runnable_case(self):
        """Regression: the slots shipped as real cases with prompt "TODO".

        check.py skipped them on a `metadata.todo` marker, but `compass test`
        knows no such convention — it ran all three, found no correctness
        graders, passed the process guards, and scored each 0.997 as a PASS.
        Both models collected three free points and $0.31 of wasted spend.
        An unfilled slot has to be *absent*, not marked.
        """
        assert check.find_placeholder_cases(_suite()["cases"]) == []

    def test_the_checker_refuses_to_run_with_an_unfilled_slot(self):
        """And it is an error, not a skip — skipping is what hid it before."""
        stub = {"id": "TODO_something", "input": {"prompt": "TODO"}}
        assert check.find_placeholder_cases([stub]) == ["TODO_something"]
        # Caught by shape too, not just by the id convention.
        assert check.find_placeholder_cases(
            [{"id": "looks_real", "input": {"prompt": "  "}}]
        ) == ["looks_real"]
        assert check.find_placeholder_cases(
            [{"id": "fine", "input": {"prompt": "Do a real thing."}}]
        ) == []

    def test_the_slots_survive_as_commented_templates(self):
        """Commented out, but still there to fill in — otherwise the guidance
        about case-type mix has nothing to attach to."""
        raw = (_EXAMPLE / "suite.yaml").read_text(encoding="utf-8")
        assert "待填的槽位" in raw
        assert "# - id: bug_locate_2" in raw

    def test_process_guards_are_shared_and_never_gate_on_cost(self):
        suite = _suite()
        names = {g["name"] for g in suite["default_graders"]}
        assert {"cost_budget", "turn_count", "loop_detection"} <= names
        for grader in suite["default_graders"]:
            if grader["name"] in ("cost_budget", "turn_count", "loop_detection"):
                assert not grader.get("gate"), (
                    f"{grader['name']} must not gate — being expensive is not "
                    "the same as being wrong"
                )


class TestCaseValidation:
    """check.py's own verdicts. These are the guarantee that the case set is
    worth running at all."""

    def test_every_case_is_red_green_and_stable(self, tmp_path):
        reports = [
            check.check_case(case, tmp_path / case["id"])
            for case in _live_cases()
        ]
        for report in reports:
            assert report.red, f"{report.case_id}: already solved on the base project"
            assert report.green, f"{report.case_id}: reference solution fails — {report.notes}"
            assert report.no_regress, f"{report.case_id}: reference breaks the repo suite"
            assert report.stable, f"{report.case_id}: flaky"

    def test_the_checker_catches_an_already_solved_case(self, tmp_path, monkeypatch):
        """The RED check is the one that silently inflates scores when skipped."""
        case = {"id": "add_discount"}
        # Pretend the base project already ships the solution.
        monkeypatch.setattr(
            check, "_PROJECT", _EXAMPLE / "solutions" / "add_discount"
        )
        original = check._workspace

        def _workspace(dest, solution):
            work = original(dest, solution)
            # solutions/add_discount only holds pricing.py; bring the rest in.
            for src in (_EXAMPLE / "project").rglob("*"):
                target = work / src.relative_to(_EXAMPLE / "project")
                if src.is_file() and not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(src.read_bytes())
            return work

        monkeypatch.setattr(check, "_workspace", _workspace)
        report = check.check_case(case, tmp_path)

        assert report.red is False
        assert any("already" in n for n in report.notes)


class TestRegressionTrap:
    """The trap case only discriminates if the tempting fix passes the hidden
    tests and fails the repo suite. If that ever stops being true the case is
    just another easy one, and nobody would notice."""

    @staticmethod
    def _naive_fix(orders: str) -> str:
        """Reassign subtotal to the discounted figure — the obvious edit."""
        return orders.replace(
            "    # Shipping is judged on the subtotal, before the promo comes off.\n"
            "    shipping = shipping_fee(subtotal)",
            "    subtotal = round(subtotal - discount, 2)\n"
            "    shipping = shipping_fee(subtotal)",
        ).replace(
            '        "total": round(subtotal - discount + shipping, 2),',
            '        "total": round(subtotal + shipping, 2),',
        )

    def test_the_naive_fix_passes_hidden_tests_and_fails_the_repo_suite(self, tmp_path):
        import shutil

        work = tmp_path / "work"
        shutil.copytree(_EXAMPLE / "project", work)
        orders = (work / "orders.py").read_text(encoding="utf-8")
        naive = self._naive_fix(orders)
        assert naive != orders, "the anchor moved; update _naive_fix"
        (work / "orders.py").write_text(naive, encoding="utf-8")

        hidden = _EXAMPLE / "grader_tests" / "test_trap_regression_free_shipping.py"
        hidden_ok, _ = check._run_pytest(work, str(hidden))
        repo_ok, _ = check._run_pytest(work, "tests/")

        assert hidden_ok, "the hidden tests over-specify — they caught the naive fix"
        assert not repo_ok, "the repo suite no longer pins what the trap relies on"

    def test_the_hidden_tests_never_assert_the_trapped_field(self):
        """`subtotal` belongs to the repo suite. Pinning it in both places is
        how the split silently stops working."""
        source = (
            _EXAMPLE / "grader_tests" / "test_trap_regression_free_shipping.py"
        ).read_text(encoding="utf-8")
        code = "\n".join(
            line for line in source.splitlines() if not line.lstrip().startswith("#")
        )
        assert 'breakdown["subtotal"]' not in code


class TestRunPyResolvesTheSuite:
    """suite.yaml is not runnable as shipped — its paths only exist on the
    machine running it. An unfilled placeholder fails deep inside a grader,
    after the agent has already been paid for."""

    run_py = _load("run.py", "coding_agent_run")

    def test_no_placeholder_survives_resolution(self, tmp_path):
        resolved = self.run_py.resolve_suite(tmp_path / "repo", tmp_path, trials=None)
        assert "{{" not in resolved.read_text(encoding="utf-8")

    def test_every_placeholder_becomes_an_absolute_path(self, tmp_path):
        resolved = self.run_py.resolve_suite(tmp_path / "repo", tmp_path, trials=None)
        suite = yaml.safe_load(resolved.read_text(encoding="utf-8"))
        config = suite["agent"]["config"]

        assert Path(config["repo"]).is_absolute()
        assert Path(config["save_stream_to"]).is_absolute()
        # The raw stream lives under --out with everything else from the run.
        assert Path(config["save_stream_to"]).parent == tmp_path.resolve()
        for case in suite["cases"]:
            for grader in case.get("graders", []):
                script = grader.get("config", {}).get("script", "")
                assert "{{" not in script

    def test_trials_override_lands(self, tmp_path):
        resolved = self.run_py.resolve_suite(tmp_path / "repo", tmp_path, trials=5)
        assert yaml.safe_load(resolved.read_text())["defaults"]["trials"] == 5

    def test_the_bundled_project_becomes_a_real_git_repo(self, tmp_path):
        import subprocess

        repo = self.run_py.bootstrap_repo(tmp_path)
        assert (repo / ".git").is_dir()
        assert (repo / "orders.py").exists()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo, capture_output=True, text=True, check=True,
        )
        assert status.stdout == ""

    def test_bootstrap_is_idempotent(self, tmp_path):
        """Re-running must not re-baseline: traces from earlier runs are only
        comparable against the same starting commit."""
        import subprocess

        first = self.run_py.bootstrap_repo(tmp_path)
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=first,
            capture_output=True, text=True, check=True,
        ).stdout
        again = self.run_py.bootstrap_repo(tmp_path)
        assert again == first
        assert subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=again,
            capture_output=True, text=True, check=True,
        ).stdout == head
