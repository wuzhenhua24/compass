"""Tests for the examples/coding_agent template.

An example that silently rots is worse than no example — someone clones it,
runs it, and gets a green wall that proves nothing. These pin the parts that
drift: the fixtures' Edit anchors against the project files, the hidden
acceptance tests against the implementations the fixtures write, and the
end-to-end verdict for each behaviour.
"""

from __future__ import annotations

import importlib.util
import subprocess
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

    def test_every_live_case_has_something_to_validate_against(self):
        """A fix case needs a reference solution; without one nobody has shown
        the task is solvable. A no-change case has nothing to write, so what it
        needs instead is ``traps/<id>/`` — the wrong fix it must detect."""
        for case in _live_cases():
            if check.case_kind(case) == "no_change":
                overlay = _EXAMPLE / "traps" / case["id"]
                assert overlay.is_dir(), f"{case['id']} has no traps/{case['id']}/"
            else:
                overlay = _EXAMPLE / "solutions" / case["id"]
                assert overlay.is_dir(), f"{case['id']} has no solutions/{case['id']}/"

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

    def test_no_case_has_two_graders_under_the_same_key(self):
        """`label or name` has to be unique within a case, or the grader is
        unreachable: `--on <key>` refuses an ambiguous match, and `breakdown`
        falls back to a positional `#2` suffix that moves whenever the grader
        order changes. Caught this for real on `false_bug_max_uses`, whose
        case-level `state_delta` collided with the one in `default_graders` —
        and `state_delta` is that case's entire scoring signal."""
        suite = _suite()
        defaults = suite.get("default_graders") or []
        for case in _live_cases():
            keys = [
                g.get("label") or g["name"]
                for g in list(case.get("graders") or []) + defaults
            ]
            dupes = sorted({k for k in keys if keys.count(k) > 1})
            assert not dupes, f"{case['id']}: duplicate grader key(s) {dupes}"

    def test_a_grader_measuring_different_things_is_labelled_per_case(self):
        """Uniqueness within a case is not enough — the selector falls back to
        the registry name, so an unlabelled grader pools across cases.

        Pooling is often right: `diff_size` in seven cases is one measurement
        with seven thresholds, and `--on diff_size` legitimately compares them
        all. It is wrong when the config decides *what* is measured rather than
        how much: two `rubric` graders with different criteria are two
        different questions ('did it explain why the ticket is wrong' vs 'did
        it surface its assumption'), and averaging them is meaningless.
        """
        # Per grader name, the config key that decides what is being measured.
        DEFINES_THE_MEASUREMENT = {"rubric": "criteria", "integration_test": "script"}

        by_name: dict[str, list[tuple[str, str, str]]] = {}
        for case in _live_cases():
            for g in case.get("graders") or []:
                key = DEFINES_THE_MEASUREMENT.get(g["name"])
                if key is None:
                    continue
                by_name.setdefault(g["name"], []).append(
                    (case["id"], g.get("label") or "", repr(g["config"].get(key)))
                )

        assert set(by_name) == set(DEFINES_THE_MEASUREMENT), (
            "a grader in the table above is no longer used; drop it or fix the key"
        )
        for name, uses in by_name.items():
            distinct = {what for _, _, what in uses}
            if len(distinct) < 2:
                continue  # every use measures the same thing; pooling is fine
            unlabelled = sorted({c for c, label, _ in uses if not label})
            assert not unlabelled, (
                f"{name!r} measures {len(distinct)} different things across the "
                f"suite but is unlabelled in {unlabelled} — `--on {name}` would "
                f"pool them into one meaningless axis"
            )

    def test_a_no_change_case_gates_on_the_diff_not_on_its_hidden_tests(self):
        """The point of a no-change case is that doing nothing is correct — so
        its hidden tests pass for an agent that never ran. Without a case-level
        ``state_delta`` gate forbidding source edits there is no signal left,
        and the case scores every model a free point."""
        for case in _live_cases():
            if check.case_kind(case) != "no_change":
                continue
            gates = [
                g for g in case["graders"]
                if g["name"] == "state_delta" and g.get("gate")
            ]
            assert gates, f"{case['id']} has no state_delta gate"
            forbidden = [m for g in gates for m in g["config"].get("forbid", [])]
            assert any(
                m.get("target", "").endswith(".py") for m in forbidden
            ), f"{case['id']} does not forbid source edits: {forbidden}"

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

    def test_every_slot_the_template_promised_is_filled(self):
        """The three placeholder slots are now real cases. Their ids are what
        the README's case-mix table and the commit history refer to, so a
        rename has to be deliberate rather than incidental."""
        ids = {case["id"] for case in _live_cases()}
        assert {"bug_locate_2", "feature_incremental_2"} <= ids
        raw = (_EXAMPLE / "suite.yaml").read_text(encoding="utf-8")
        assert "待填的槽位" not in raw, "a slot came back; fill it or delete it"

    def test_the_guidance_for_adding_a_case_survives_the_slots(self):
        """The slots are gone but the two lessons they carried are not: an
        unfilled slot must be un-runnable rather than marked (three `TODO`
        stubs once scored 0.997 apiece and inflated both models' pass rates),
        and check.py's four checks prove your answer passes, not that the case
        discriminates. Losing either is how the next case gets added badly."""
        raw = (_EXAMPLE / "suite.yaml").read_text(encoding="utf-8")
        assert "0.997" in raw, "the free-points incident is no longer recorded"
        for verifier in (
            "TestIncrementalCaseScoresConventions",
            "TestLocalizationCaseDecoys",
            "TestAmbiguousCaseAcceptsEveryReading",
        ):
            assert verifier in raw, f"{verifier} is not pointed at from suite.yaml"
            assert verifier in globals(), f"{verifier} no longer exists"

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
            # `red`/`green` carry the mirror-image checks for a no-change case:
            # NO-BUG (hidden tests pass untouched) and TRAP (the wrong fix is
            # caught). Same fields, and both still have to be true.
            first, second = (
                ("hidden tests fail on the untouched project", "the trap fix is caught")
                if report.kind == "no_change"
                else ("already solved on the base project", "reference solution fails")
            )
            assert report.red, f"{report.case_id}: not {first} — {report.notes}"
            assert report.green, f"{report.case_id}: not {second} — {report.notes}"
            assert report.no_regress, f"{report.case_id}: repo suite red — {report.notes}"
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


class TestNoChangeCase:
    """``false_bug_max_uses`` hands the agent a bug report that is wrong. The
    case only measures anything if doing nothing is genuinely correct *and* the
    fix the report asks for is genuinely detectable — one without the other and
    it is either an unfair case or a free point."""

    _ID = "false_bug_max_uses"

    def _work(self, tmp_path, overlay: Path | None):
        import shutil

        work = tmp_path / "work"
        shutil.copytree(_EXAMPLE / "project", work)
        for src in (overlay.rglob("*") if overlay else []):
            if src.is_file():
                (work / src.relative_to(overlay)).write_bytes(src.read_bytes())
        return work

    def test_doing_nothing_passes_both_suites(self, tmp_path):
        """Which is why ``state_delta`` is the scoring signal and the hidden
        tests are only diagnosis. If this ever fails, there is a real defect and
        the case has quietly become an ordinary fix case."""
        work = self._work(tmp_path, None)
        hidden = _EXAMPLE / "grader_tests" / f"test_{self._ID}.py"
        assert check._run_pytest(work, str(hidden))[0]
        assert check._run_pytest(work, "tests/")[0]

    def test_the_fix_the_ticket_asks_for_is_caught_twice(self, tmp_path):
        """Once by the hidden tests (the counter is now wrong) and once by the
        repository's own suite. Two independent catches, so the case survives
        either one being edited away."""
        work = self._work(tmp_path, _EXAMPLE / "traps" / self._ID)
        hidden = _EXAMPLE / "grader_tests" / f"test_{self._ID}.py"
        assert not check._run_pytest(work, str(hidden))[0]
        assert not check._run_pytest(work, "tests/")[0]

    def test_the_checker_reports_an_undetectable_trap(self, tmp_path, monkeypatch):
        """A trap that changes nothing observable makes the case worthless, and
        it would otherwise look like a clean pass."""
        monkeypatch.setattr(check, "_TRAPS", tmp_path / "traps")
        harmless = tmp_path / "traps" / self._ID
        harmless.mkdir(parents=True)
        source = (_EXAMPLE / "project" / "promo.py").read_text(encoding="utf-8")
        (harmless / "promo.py").write_text(
            source + "\n\n_UNUSED = 1  # changes nothing\n", encoding="utf-8"
        )

        report = check.check_no_change_case(
            {"id": self._ID, "metadata": {"expects_no_change": True}}, tmp_path / "w"
        )

        assert report.red is True  # there is still no bug
        assert report.green is False  # but the trap is undetectable
        assert any("cannot tell restraint" in n for n in report.notes)

    async def test_the_gate_forbids_edits_but_allows_a_scratch_script(self):
        """``op: update`` rather than the whole of ``*.py``, deliberately.

        Writing a throwaway script to reproduce what the ticket claims is the
        behaviour this case is trying to reward. Forbidding every ``.py`` would
        score that the same as editing the source it just disproved."""
        from compass.core.transcript import StateChange, ToolCall, Transcript
        from compass.graders.base import GradeContext
        from compass.graders.registry import get_grader

        config = next(
            g["config"]
            for c in _live_cases() if c["id"] == self._ID
            for g in c["graders"] if g["name"] == "state_delta"
        )
        grader = get_grader("state_delta")(config)

        def _verdict(op: str, target: str):
            transcript = Transcript(task_id=self._ID, trial_id="t")
            transcript.tool_calls.append(
                ToolCall(
                    tool_name="Write",
                    state_delta=[StateChange(kind="file", op=op, target=target)],
                )
            )
            return grader.grade(GradeContext(transcript=transcript))

        assert (await _verdict("create", "check_ticket_4412.py")).passed
        assert not (await _verdict("update", "promo.py")).passed
        assert not (await _verdict("update", "tests/test_orders.py")).passed

    def test_a_missing_trap_is_unverified_not_a_pass(self, tmp_path, monkeypatch):
        monkeypatch.setattr(check, "_TRAPS", tmp_path / "absent")
        report = check.check_no_change_case(
            {"id": self._ID, "metadata": {"expects_no_change": True}}, tmp_path / "w"
        )
        assert report.unverified
        assert not report.verified
        assert any("no traps/" in n for n in report.notes)


class TestInteractionCaseScoresPartially:
    """The composition case exists to produce a per-case score in (0, 1). Five
    plausible wrong compositions, each landing on a different number — that is
    what buys resolution that a binary pass/fail cannot."""

    _ID = "interaction_promo_then_tier_rounding"

    @staticmethod
    def _fraction(work: Path, target: str) -> float:
        import re

        _, output = check._run_pytest(work, target)
        passed = int((re.search(r"(\d+) passed", output) or [0, 0])[1])
        failed = int((re.search(r"(\d+) failed", output) or [0, 0])[1])
        errors = int((re.search(r"(\d+) error", output) or [0, 0])[1])
        total = passed + failed + errors
        assert total, f"hidden tests did not run:\n{output}"
        return passed / total

    def test_a_wrong_composition_scores_between_zero_and_one(self, tmp_path):
        import shutil

        work = tmp_path / "work"
        shutil.copytree(_EXAMPLE / "project", work)
        reference = (
            _EXAMPLE / "solutions" / self._ID / "orders.py"
        ).read_text(encoding="utf-8")

        # The most common wrong reading: the tier comes off the subtotal rather
        # than off what is left after the promo.
        naive = reference.replace(
            "_to_cents(after_promo * TIER_DISCOUNT.get(tier, 0.0))",
            "_to_cents(subtotal * TIER_DISCOUNT.get(tier, 0.0))",
        )
        assert naive != reference, "the anchor moved; update this variant"
        (work / "orders.py").write_text(naive, encoding="utf-8")

        hidden = str(_EXAMPLE / "grader_tests" / f"test_{self._ID}.py")
        score = self._fraction(work, hidden)
        assert 0.0 < score < 1.0, (
            f"the wrong composition scored {score} — an all-or-nothing hidden "
            "test file gives up the resolution this case was built for"
        )

        # And it is a *wrong* answer, not a stylistic difference: the repo's own
        # suite stays green, so the hidden tests are the only thing that knows.
        assert check._run_pytest(work, "tests/")[0]

    def test_an_untouched_project_scores_zero(self, tmp_path):
        """No free points: every assertion in the file needs the new work."""
        import shutil

        work = tmp_path / "work"
        shutil.copytree(_EXAMPLE / "project", work)
        hidden = str(_EXAMPLE / "grader_tests" / f"test_{self._ID}.py")
        assert self._fraction(work, hidden) == 0.0


class TestAmbiguousCaseAcceptsEveryReading:
    """``ambiguous_no_stacking`` withholds one decision on purpose: which
    discount wins when both apply. The hidden tests are only honest if every
    faithful reading passes them — otherwise the case silently measures "did
    the agent guess the same way the author did", which is luck, not skill.

    check.py's GREEN only proves the *reference* reading passes. This proves
    the other two do, and that the readings which actually violate the stated
    requirement do not."""

    _ID = "ambiguous_no_stacking"

    _CHOICE = """    if tier_candidate > promo_candidate:
        discount, tier_discount = 0.0, tier_candidate
    else:
        discount, tier_discount = promo_candidate, 0.0"""

    # Each is a complete replacement for the reference's tie-break block.
    _FAITHFUL = {
        "better for the customer": None,  # the reference itself
        "promo always wins": """    if promo_candidate > 0:
        discount, tier_discount = promo_candidate, 0.0
    else:
        discount, tier_discount = 0.0, tier_candidate""",
        "tier always wins": """    if tier_candidate > 0:
        discount, tier_discount = 0.0, tier_candidate
    else:
        discount, tier_discount = promo_candidate, 0.0""",
    }

    _UNFAITHFUL = {
        # "They do not stack" — these all stack, one way or another.
        "both applied": "    discount, tier_discount = promo_candidate, tier_candidate",
        "tier on the post-promo amount": """    discount = promo_candidate
    tier_discount = _to_cents((subtotal - discount) * TIER_DISCOUNT.get(tier, 0.0))""",
        # Resolves the conflict by dropping the tier whenever a code is passed,
        # so a code that does not apply silently cancels the tier too.
        "tier dropped when any code is given": """    if promo_code:
        discount, tier_discount = promo_candidate, 0.0
    else:
        discount, tier_discount = 0.0, tier_candidate""",
    }

    def _score(self, tmp_path, label, block):
        import re
        import shutil

        reference = (
            _EXAMPLE / "solutions" / self._ID / "orders.py"
        ).read_text(encoding="utf-8")
        source = reference if block is None else reference.replace(self._CHOICE, block)
        assert block is None or source != reference, f"{label}: anchor moved"

        work = tmp_path / label.replace(" ", "_")
        shutil.copytree(_EXAMPLE / "project", work)
        (work / "orders.py").write_text(source, encoding="utf-8")

        hidden = str(_EXAMPLE / "grader_tests" / f"test_{self._ID}.py")
        ok, output = check._run_pytest(work, hidden)
        passed = int((re.search(r"(\d+) passed", output) or [0, 0])[1])
        repo_ok, _ = check._run_pytest(work, "tests/")
        return ok, passed, repo_ok, output

    def test_every_faithful_reading_passes(self, tmp_path):
        for label, block in self._FAITHFUL.items():
            ok, passed, repo_ok, output = self._score(tmp_path, label, block)
            assert ok, (
                f"reading {label!r} satisfies the requirement as written but "
                f"fails the hidden tests — they over-specify:\n{check._tail(output)}"
            )
            assert repo_ok, f"reading {label!r} breaks the repo's own suite"
            assert passed == 8

    def test_a_reading_that_stacks_or_drops_a_discount_fails(self, tmp_path):
        for label, block in self._UNFAITHFUL.items():
            ok, passed, _, _ = self._score(tmp_path, label, block)
            assert not ok, (
                f"{label!r} violates the stated requirement and still passes — "
                "the hidden tests under-specify and the case scores nothing"
            )
            # Still partial credit, not a flat zero: the probes that do not
            # involve both discounts keep working.
            assert 0 < passed < 8

    def test_the_hidden_tests_never_pin_a_winner(self, tmp_path):
        """The two readings must actually disagree somewhere, or the 'ambiguity'
        is decorative and the case is just another feature case."""
        import shutil

        reference = (
            _EXAMPLE / "solutions" / self._ID / "orders.py"
        ).read_text(encoding="utf-8")
        totals = {}
        for label in ("better for the customer", "promo always wins"):
            block = self._FAITHFUL[label]
            source = reference if block is None else reference.replace(
                self._CHOICE, block
            )
            work = tmp_path / f"probe_{label.replace(' ', '_')}"
            shutil.copytree(_EXAMPLE / "project", work)
            (work / "orders.py").write_text(source, encoding="utf-8")
            (work / "probe.py").write_text(
                "from orders import place_order\n"
                "b = place_order([{'sku':'mug','unit_price':40.0,'quantity':5}],"
                " {'mug':10}, promo_code='WELCOME10', tier='gold')\n"
                "print(b['total'])\n",
                encoding="utf-8",
            )
            out = subprocess.run(
                [sys.executable, "probe.py"], cwd=work, capture_output=True, text=True
            )
            totals[label] = out.stdout.strip()

        assert totals["better for the customer"] == "170.0"
        assert totals["promo always wins"] == "180.0"


class TestLocalizationCaseDecoys:
    """``bug_locate_2`` puts the symptom on a receipt and the cause two hops
    away, with two wrong places to land in between. It measures localization
    only for as long as both decoys stay caught — and a decoy that quietly
    stops being caught turns the case into a free point without failing
    anything."""

    _ID = "bug_locate_2"
    _BREAK = '        if order["status"] == "cancelled":\n            break'

    def _score(self, tmp_path, name, edits):
        import re
        import shutil

        work = tmp_path / name
        shutil.copytree(_EXAMPLE / "project", work)
        for rel, (old, new) in edits.items():
            path = work / rel
            source = path.read_text(encoding="utf-8")
            assert old in source, f"{name}: anchor moved in {rel}"
            path.write_text(source.replace(old, new, 1), encoding="utf-8")

        hidden = str(_EXAMPLE / "grader_tests" / f"test_{self._ID}.py")
        ok, output = check._run_pytest(work, hidden)
        passed = int((re.search(r"(\d+) passed", output) or [0, 0])[1])
        repo_ok, _ = check._run_pytest(work, "tests/")
        return ok, passed, repo_ok

    def test_the_real_fix_is_one_word(self, tmp_path):
        ok, passed, repo_ok = self._score(
            tmp_path,
            "real",
            {"history.py": (self._BREAK, self._BREAK.replace("break", "continue"))},
        )
        assert ok and repo_ok and passed == 8

    def test_the_threshold_table_is_a_decoy_the_repo_suite_guards(self, tmp_path):
        """One hop from the symptom and the obvious suspect for 'the tier is
        wrong'. Lowering it to make the reported customer gold turns
        tests/test_loyalty.py red — the regression gate is what catches this,
        which is the whole reason every case runs two gates."""
        ok, _, repo_ok = self._score(
            tmp_path, "threshold",
            {"loyalty.py": ('    (1000.0, "gold"),', '    (400.0, "gold"),')},
        )
        assert not ok, "the hidden tests no longer notice a moved threshold"
        assert not repo_ok, "tests/test_loyalty.py no longer pins the thresholds"

    def test_the_legacy_module_is_dead_code(self, tmp_path):
        """It has real bugs in it and is imported by nothing, so 'fixing' it
        must change no outcome at all. If anything ever imports it, the decoy
        becomes a genuine second cause and the case stops having one answer."""
        for module in (_EXAMPLE / "project").rglob("*.py"):
            if "legacy" in module.parts:
                continue
            assert "legacy" not in module.read_text(encoding="utf-8"), (
                f"{module.name} references legacy/ — it is no longer dead code"
            )

        did_nothing = self._score(tmp_path, "nothing", {})
        edited_legacy = self._score(
            tmp_path, "legacy",
            {"legacy/reporting_v1.py": (
                '    if lifetime > TIER_CUTOFFS["gold"]:',
                '    if lifetime >= TIER_CUTOFFS["gold"]:',
            )},
        )
        assert edited_legacy == did_nothing
        assert not edited_legacy[0]

    def test_deleting_the_cancelled_record_does_not_pass(self, tmp_path):
        """The reason most of the hidden tests build their own histories: with
        that record gone the reported customer's receipt is right, and against
        the shipped data alone the edit is indistinguishable from a fix."""
        ok, _, repo_ok = self._score(
            tmp_path, "deleted",
            {"history.py": (
                '        {"id": "o_5031", "status": "cancelled", "total": 180.00},\n',
                "",
            )},
        )
        assert not ok, "hidden tests only look at the shipped data"
        assert repo_ok, "the repo suite is not what catches this one"

    def test_the_bug_is_invisible_to_the_repos_own_suite(self, tmp_path):
        """Which is why it shipped, and why the case is about locating rather
        than noticing. If tests/ ever goes red on the untouched project, RED
        stops meaning what check.py reports."""
        ok, passed, repo_ok = self._score(tmp_path, "base", {})
        assert repo_ok
        assert not ok and 0 < passed < 8


class TestIncrementalCaseScoresConventions:
    """``feature_incremental_2`` is `add_discount` one notch up: the same shape
    (one function, signature given) but the difficulty moved from writing it to
    honouring contracts the module already has. These pin that the hidden tests
    grade the contracts and not a house style."""

    _ID = "feature_incremental_2"

    def _score(self, tmp_path, name, body):
        import re
        import shutil

        reference = (
            _EXAMPLE / "solutions" / self._ID / "inventory.py"
        ).read_text(encoding="utf-8")
        marker = (
            '    if quantity <= 0:\n'
            '        raise ValueError(f"quantity must be positive, got {quantity}")\n'
            "    product(sku)"
        )
        assert marker in reference, "the reference body moved; update the marker"
        source = reference if body is None else reference.replace(
            reference[reference.index(marker):], body + "\n"
        )
        assert body is None or source != reference, f"{name}: replacement failed"

        work = tmp_path / name
        shutil.copytree(_EXAMPLE / "project", work)
        (work / "inventory.py").write_text(source, encoding="utf-8")

        hidden = str(_EXAMPLE / "grader_tests" / f"test_{self._ID}.py")
        ok, output = check._run_pytest(work, hidden)
        passed = int((re.search(r"(\d+) passed", output) or [0, 0])[1])
        repo_ok, _ = check._run_pytest(work, "tests/")
        return ok, passed, repo_ok

    _CHECKS = """    if quantity <= 0:
        raise ValueError("quantity must be positive")
    product(sku)
    available = stock_from.get(sku, 0)
    if quantity > available:
        raise OutOfStock(sku, quantity, available)
"""

    def test_a_different_but_correct_style_also_passes(self, tmp_path):
        """Copy-then-mutate is as valid as building new maps. A hidden test
        that only accepts the reference's idiom is grading style, not the
        contract the requirement stated."""
        ok, passed, repo_ok = self._score(
            tmp_path,
            "copy_then_mutate",
            self._CHECKS + """    new_from, new_to = dict(stock_from), dict(stock_to)
    new_from[sku] = available - quantity
    new_to[sku] = new_to.get(sku, 0) + quantity
    return new_from, new_to""",
        )
        assert ok and repo_ok and passed == 10

    def test_mutating_a_caller_s_map_is_caught(self, tmp_path):
        ok, passed, _ = self._score(
            tmp_path,
            "mutates",
            self._CHECKS + """    stock_from[sku] = available - quantity
    stock_to[sku] = stock_to.get(sku, 0) + quantity
    return stock_from, stock_to""",
        )
        assert not ok and 0 < passed < 10

    def test_a_half_applied_transfer_is_caught(self, tmp_path):
        """With two maps, a partial application invents or destroys stock
        rather than merely mislaying it — the reason this case exists rather
        than a second single-map one."""
        ok, passed, _ = self._score(
            tmp_path,
            "half_applied",
            """    available = stock_from.get(sku, 0)
    stock_from[sku] = available - quantity
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    product(sku)
    if quantity > available:
        raise OutOfStock(sku, quantity, available)
    return stock_from, {**stock_to, sku: stock_to.get(sku, 0) + quantity}""",
        )
        assert not ok and 0 < passed < 10

    def test_reinventing_the_error_vocabulary_is_caught(self, tmp_path):
        """`OutOfStock` carries sku/wanted/available for support. A bare raise
        satisfies "it refuses" and loses everything the caller needed."""
        ok, passed, _ = self._score(
            tmp_path,
            "bare_raise",
            """    if quantity <= 0:
        raise ValueError("quantity must be positive")
    product(sku)
    available = stock_from.get(sku, 0)
    if quantity > available:
        raise OutOfStock(sku, 0, 0)
    return (
        {**stock_from, sku: available - quantity},
        {**stock_to, sku: stock_to.get(sku, 0) + quantity},
    )""",
        )
        assert not ok and 0 < passed < 10

    def test_the_case_does_not_depend_on_the_defect_seeded_for_another(self):
        """The base project ships `bug_locate_2`'s broken `total_spent`. A case
        that reached lifetime spend would silently require fixing that too, and
        its RED/GREEN would stop meaning what it says."""
        hidden = (
            _EXAMPLE / "grader_tests" / f"test_{self._ID}.py"
        ).read_text(encoding="utf-8")
        solution = (
            _EXAMPLE / "solutions" / self._ID / "inventory.py"
        ).read_text(encoding="utf-8")
        for forbidden in ("total_spent", "lifetime_spend", "receipts", "history"):
            assert forbidden not in hidden, f"hidden tests reach {forbidden}"
            assert forbidden not in solution, f"the reference reaches {forbidden}"


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
