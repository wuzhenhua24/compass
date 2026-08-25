"""The skill-eval example, end to end and offline.

``examples/skill_eval`` is documentation that runs: it claims a skill rewrite
can be shown to be an improvement, and claims a specific shape of finding —
v2 triggers more often *and* steals one request it should not. Both claims are
pinned here, because a README that has quietly stopped being true is worse than
no README.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from compass.core.runner import Compass

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "skill_eval"


def _load(filename: str, alias: str) -> ModuleType:
    """Import an example module under a unique name.

    Not ``sys.path.insert`` + a bare import: several examples ship an
    ``eval.py``, so the plain form makes whichever test module imports first
    win — and the other silently tests the wrong example.
    """
    path = _EXAMPLE / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


skill_eval = _load("eval.py", "skill_eval_example")


@pytest.fixture(scope="module")
def repo() -> Path:
    """The fixture project, as a git repo, shared by every test here."""
    with tempfile.TemporaryDirectory(prefix="compass_skill_example_") as tmp:
        yield skill_eval.bootstrap_repo(Path(tmp))


def _run(suite: str, skill: str, repo: Path, trials: int = 1):
    scenario = skill_eval.build_scenario(suite, repo, skill, trials)
    return asyncio.run(Compass().run(scenario))


def _trigger_rates(result) -> dict[str, float]:
    return {
        case.case_id: skill_eval._case_metric(case, "skill_triggered")
        for case in result.case_results
    }


# ---------------------------------------------------------------------------
# The claim on the tin
# ---------------------------------------------------------------------------


def test_v2_triggers_where_v1_misses(repo: Path):
    """The headline: the rewrite is measurably better at being reached."""
    v1 = _trigger_rates(_run("ab", str(skill_eval._SKILLS / "report-writer-v1"), repo))
    v2 = _trigger_rates(_run("ab", str(skill_eval._SKILLS / "report-writer-v2"), repo))

    # Naming the skill works in both; describing the task only works in v2.
    assert v1["explicit_invocation"] == 1.0
    assert v1["contextual_invocation"] == 0.0
    assert all(rate == 1.0 for rate in v2.values())


def test_the_baseline_arm_is_a_real_baseline(repo: Path):
    """``-m ""`` installs nothing, and the run says so rather than erroring."""
    result = _run("ab", "", repo)

    assert all(rate == 0.0 for rate in _trigger_rates(result).values())
    # Not an error: the agent still did the work, just without the skill.
    assert all(case.status.value != "error" for case in result.case_results)


def test_the_negative_control_catches_v2_overtriggering(repo: Path):
    """The finding a positives-only eval cannot produce.

    v2's pushier description wins two requests it was missing and one it has no
    business handling. If this ever comes back clean, either the fixture or the
    grader stopped measuring — both make the example a lie.
    """
    rates = _trigger_rates(
        _run("trigger", str(skill_eval._SKILLS / "report-writer-v2"), repo)
    )

    assert rates["pos_implicit"] == 1.0
    assert rates["pos_contextual"] == 1.0
    assert rates["neg_chart"] == 1.0          # the false positive
    assert rates["neg_format_convert"] == 0.0  # ... and it is not indiscriminate


def test_the_bundled_script_separates_the_two_versions(repo: Path):
    """v2 ships a script; the transcript shows whether it was actually run."""
    def used(skill: str) -> list[float]:
        result = _run("ab", str(skill_eval._SKILLS / skill), repo)
        return [
            skill_eval._case_metric(case, "skill_resources_used")
            for case in result.case_results
        ]

    assert used("report-writer-v1") == [0.0, 0.0, 0.0]
    assert used("report-writer-v2") == [1.0, 1.0, 1.0]


# ---------------------------------------------------------------------------
# The install is genuinely under test
# ---------------------------------------------------------------------------


def test_the_replay_reads_the_skill_the_adapter_installed(repo: Path):
    """The example's whole claim to be honest rests on this.

    ``replay_cli.py`` opens ``.claude/skills/report-writer/SKILL.md`` in its
    working directory instead of switching on a flag. So a broken install shows
    up as moved numbers — where a lookup table would have gone on producing a
    beautiful comparison of nothing.
    """
    source = (_EXAMPLE / "replay_cli.py").read_text(encoding="utf-8")
    assert ".claude/skills/report-writer/SKILL.md" in source

    result = _run("ab", str(skill_eval._SKILLS / "report-writer-v2"), repo)
    skills = result.case_results[0].output_data["metadata"]["skills"]
    assert skills[0]["name"] == "report-writer"
    assert skills[0]["digest"]


def test_the_result_records_which_skill_produced_it(repo: Path):
    """The digest has to reach the *results file*, not only the trace.

    Traces are opt-in and separate; a results file is what gets archived,
    compared and published. Without this the two arms of an A/B differ only in
    their numbers, and "v2 scored higher" can no longer be tied to *this* v2 —
    the promise ``digest`` exists to keep.
    """
    result = _run("ab", str(skill_eval._SKILLS / "report-writer-v2"), repo)
    case = result.case_results[0]

    assert case.skills[0]["name"] == "report-writer"
    digest = case.skills[0]["digest"]
    assert digest

    # ... and through serialization into the published run document.
    from compass.report.site import collect_run

    doc = collect_run([result], name="v2")
    assert doc["run"]["skills"] == [
        {
            "name": "report-writer",
            "digest": digest,
            "files": case.skills[0]["files"],
            "cases": 3,
        }
    ]


def test_the_baseline_arm_records_no_skill(repo: Path):
    """Not an omission — the arm genuinely installed nothing."""
    result = _run("ab", "", repo)

    assert all(case.skills == [] for case in result.case_results)


def test_the_installed_skill_is_not_reported_as_the_agents_diff(repo: Path):
    result = _run("ab", str(skill_eval._SKILLS / "report-writer-v2"), repo)
    changed = result.case_results[0].output_data["metadata"]["changed_files"]
    assert changed == ["report.md"]


# ---------------------------------------------------------------------------
# Scenario hygiene
# ---------------------------------------------------------------------------


def test_every_arm_pins_settings_to_the_workspace():
    """Without this the operator's own ~/.claude/skills leaks into the run —
    silently, and worst of all into the baseline arm, which has no installed
    skill for Compass's own default to protect."""
    for suite in ("ab", "trigger"):
        config = yaml.safe_load(
            (_EXAMPLE / f"{suite}.yaml").read_text(encoding="utf-8")
        )["agent"]["config"]
        assert config["setting_sources"] == "project"


def test_the_ab_suite_gates_on_dangerous_operations():
    """A skill that teaches the agent to ``rm -rf`` scores well on every other
    grader here — triggering, script use and cost all improve when the skill
    does more of the work itself. Only a gate catches the price of that."""
    graders = yaml.safe_load(
        (_EXAMPLE / "ab.yaml").read_text(encoding="utf-8")
    )["default_graders"]
    danger = [g for g in graders if g["name"] == "dangerous_operations"]
    assert len(danger) == 1
    # A gate, not a scored grader: not deleting the repo is not an achievement
    # to be rewarded, but doing so has to turn the case red.
    assert danger[0]["gate"] is True


def test_the_ab_suite_reports_a_clean_run(repo: Path):
    """The end-to-end check that the gate is wired to a real transcript and not
    just present in the YAML — an always-absent grader would pass this suite
    exactly as convincingly as an always-clean one."""
    result = _run("ab", str(_EXAMPLE / "skills" / "report-writer-v2"), repo)
    for case in result.case_results:
        found = [
            e for e in case.evaluator_results if e.name == "dangerous_operations"
        ]
        assert found, f"{case.case_id} never ran the gate"
        assert found[0].passed
        assert found[0].metrics["clean_run"] is True


def test_the_trigger_suite_keeps_its_expectations_per_case():
    """``default_graders`` are additive, not overriding: a default
    ``should_trigger: true`` would attach to every negative control too, and
    each one would then fail by construction."""
    scenario = yaml.safe_load(
        (_EXAMPLE / "trigger.yaml").read_text(encoding="utf-8")
    )
    assert "default_graders" not in scenario

    expectations = {
        case["id"]: case["graders"][0]["config"]["should_trigger"]
        for case in scenario["cases"]
    }
    assert [cid for cid, want in expectations.items() if want] == [
        "pos_explicit",
        "pos_implicit",
        "pos_contextual",
    ]
    assert [cid for cid, want in expectations.items() if not want] == [
        "neg_edit_existing",
        "neg_format_convert",
        "neg_chart",
    ]


def test_the_two_versions_are_the_same_skill_under_different_content():
    """Sibling directories, one name. Installing under the directory name would
    change what the agent sees the skill *called*, which is a different
    experiment from the one the example claims to run."""
    names = []
    for version in ("report-writer-v1", "report-writer-v2"):
        text = (_EXAMPLE / "skills" / version / "SKILL.md").read_text(encoding="utf-8")
        front = yaml.safe_load(text.split("---")[1])
        names.append(front["name"])
    assert names == ["report-writer", "report-writer"]


def test_eval_script_reports_the_direction_it_claims(repo: Path, capsys):
    """The whole demo, as a user runs it."""
    code = asyncio.run(skill_eval.main(["--suite", "ab", "--trials", "1"]))
    out = capsys.readouterr().out

    assert code == 0
    assert "触发" in out and "skill_triggered" in out
    # The comparison is a paired one with an interval, not a pair of means.
    assert "95% CI" in out


def test_the_documented_cli_recipe_works(repo: Path, tmp_path: Path):
    """``compass test --model-key skill -m "" -m v1 -m v2`` — the recipe the
    docs hand people, driven through the real CLI.

    Two things it pins beyond "it runs": each arm gets its own results file
    (``compass compare`` needs them separated), and ``-o out/results.json``
    creates ``out/`` — which it did not, so a sweep that had already paid for
    every trial died on the write.
    """
    from click.testing import CliRunner

    from compass.cli.main import cli

    raw = (_EXAMPLE / "ab.yaml").read_text(encoding="utf-8")
    raw = (
        raw.replace("{{REPO}}", str(repo))
        .replace("{{CLI}}", str(_EXAMPLE / "replay_cli.py"))
        .replace("trials: 3", "trials: 1")
    )
    scenario = tmp_path / "ab.yaml"
    scenario.write_text(raw, encoding="utf-8")
    out = tmp_path / "never-created" / "results.json"

    result = CliRunner(env={"COLUMNS": "200"}).invoke(
        cli,
        [
            "test", str(scenario),
            "-c", "explicit_invocation",
            "--model-key", "skill",
            "-m", "",
            "-m", str(_EXAMPLE / "skills" / "report-writer-v1"),
            "-m", str(_EXAMPLE / "skills" / "report-writer-v2"),
            "--report", "json", "-o", str(out),
        ],
    )

    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert out.exists()
    # One file per arm, including the "" baseline (which slugifies to "run").
    per_arm = sorted(p.name for p in out.parent.glob("results.*.json"))
    assert len(per_arm) == 3
    assert "results.run.json" in per_arm
    assert any("report-writer-v2" in name for name in per_arm)


def test_results_are_written_in_a_shape_compare_can_pair(repo: Path):
    """`compass compare` pairs by case_id across two files; the example's
    per-arm files have to be loadable that way or the recipe in the README is
    decoration."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        written = asyncio.run(skill_eval.run_suite("ab", repo, out, trials=1))
        assert set(written) == {"baseline", "v1", "v2"}

        from compass.report.compare import compare_metric

        comparison = compare_metric(written["v1"], written["v2"], "skill_triggered")
        assert comparison.n_paired == 3
        assert comparison.mean_b > comparison.mean_a

        payload = json.loads(written["v2"].read_text(encoding="utf-8"))
        assert {case["case_id"] for case in payload["case_results"]} == {
            "explicit_invocation",
            "implicit_invocation",
            "contextual_invocation",
        }
