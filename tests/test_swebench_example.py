"""Tests for the examples/swebench loader and scoring protocol.

The grader tests build a real git repo, apply a real test patch and run real
pytest — the protocol's three sharp edges (tests hidden from the agent, test
files reset before patching, a test that did not run counts as failed) are only
worth anything if they actually hold, so they are checked against a filesystem
rather than a mock.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from compass.core.artifacts import CodeArtifact
from compass.core.transcript import Outcome
from compass.graders import GradeContext, get_grader

_EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "swebench"
_FIXTURES = _EXAMPLE / "fixtures"


def _load(filename: str, alias: str) -> ModuleType:
    """Import an example module under a unique name.

    Not ``sys.path.insert`` + a bare import: ops_qa and swebench both ship a
    ``graders.py``, so the plain form gives ``sys.modules["graders"]`` to
    whichever test module imports first and leaves the other testing the wrong
    file — or, as happened here, never registering its grader at all.
    """
    path = _EXAMPLE / filename
    spec = importlib.util.spec_from_file_location(alias, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[alias] = module
    spec.loader.exec_module(module)
    return module


swebench_graders = _load("graders.py", "swebench_graders")  # registers swebench_tests
loader = _load("loader.py", "swebench_loader")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _instance(**overrides):
    base = {
        "instance_id": "acme__widget-42",
        "repo": "acme/widget",
        "base_commit": "abc123",
        "problem_statement": "Widgets explode when frobbed twice.\nSteps: ...",
        "hints_text": "The fix is in frob(); guard the second call.",
        "patch": "diff --git a/widget.py ...",
        "test_patch": "diff --git a/tests/test_widget.py ...",
        "FAIL_TO_PASS": ["tests/test_widget.py::test_double_frob"],
        "PASS_TO_PASS": ["tests/test_widget.py::test_single_frob"],
        "version": "1.2",
    }
    base.update(overrides)
    return base


class TestLoadInstances:
    def test_reads_jsonl(self, tmp_path):
        path = tmp_path / "i.jsonl"
        path.write_text(
            "\n".join(json.dumps(_instance(instance_id=f"i-{n}")) for n in range(3)),
            encoding="utf-8",
        )
        assert len(loader.load_instances(path)) == 3

    def test_reads_a_json_array(self, tmp_path):
        path = tmp_path / "i.json"
        path.write_text(json.dumps([_instance(), _instance(instance_id="b")]))
        assert [i["instance_id"] for i in loader.load_instances(path)] == [
            "acme__widget-42",
            "b",
        ]

    def test_empty_file_is_not_an_error(self, tmp_path):
        path = tmp_path / "i.jsonl"
        path.write_text("  \n")
        assert loader.load_instances(path) == []

    def test_instance_ids_select_and_order(self, tmp_path):
        path = tmp_path / "i.jsonl"
        path.write_text(
            "\n".join(json.dumps(_instance(instance_id=x)) for x in ("a", "b", "c"))
        )
        picked = loader.load_instances(path, instance_ids=["c", "a"])
        assert [i["instance_id"] for i in picked] == ["c", "a"]

    def test_unknown_instance_id_is_loud(self, tmp_path):
        """Silently returning fewer instances would quietly shrink the eval."""
        path = tmp_path / "i.jsonl"
        path.write_text(json.dumps(_instance()))
        with pytest.raises(KeyError, match="nope"):
            loader.load_instances(path, instance_ids=["nope"])

    def test_limit(self, tmp_path):
        path = tmp_path / "i.jsonl"
        path.write_text(
            "\n".join(json.dumps(_instance(instance_id=f"i-{n}")) for n in range(9))
        )
        assert len(loader.load_instances(path, limit=4)) == 4

    def test_the_shipped_demo_fixture_parses(self):
        instances = loader.load_instances(_FIXTURES / "instances.jsonl")
        assert len(instances) == 2
        assert all(i["FAIL_TO_PASS"] and i["PASS_TO_PASS"] for i in instances)


class TestInstanceToCase:
    def test_repo_and_commit_are_per_case(self):
        """Each instance sits on its own commit, so this cannot be scenario-level."""
        case = loader.instance_to_case(_instance(), repo_root="/repos")
        assert case.input.params["repo"] == "/repos/acme__widget"
        assert case.input.params["base_ref"] == "abc123"

    def test_the_grader_gets_what_it_needs(self):
        case = loader.instance_to_case(_instance(), repo_root="/repos")
        assert case.metadata["test_patch"].startswith("diff --git")
        assert case.metadata["fail_to_pass"] == [
            "tests/test_widget.py::test_double_frob"
        ]
        assert case.metadata["pass_to_pass"] == [
            "tests/test_widget.py::test_single_frob"
        ]
        assert case.metadata["base_commit"] == "abc123"

    def test_the_prompt_never_leaks_the_answer(self):
        """The gold patch, the tests and the hints all stay out of the prompt."""
        case = loader.instance_to_case(_instance(), repo_root="/repos")
        assert "Widgets explode when frobbed twice." in case.input.prompt
        assert "diff --git" not in case.input.prompt
        assert "guard the second call" not in case.input.prompt

    def test_hints_are_opt_in(self):
        case = loader.instance_to_case(
            _instance(), repo_root="/repos", include_hints=True
        )
        assert "guard the second call" in case.input.prompt
        # Still never the patches.
        assert "diff --git" not in case.input.prompt

    def test_the_correctness_grader_is_the_gate(self):
        case = loader.instance_to_case(_instance(), repo_root="/repos")
        (grader,) = case.graders
        assert grader.name == "swebench_tests"
        assert grader.gate is True

    def test_docker_image_template_is_expanded(self):
        case = loader.instance_to_case(
            _instance(),
            repo_root="/repos",
            docker_image="swebench/sweb.eval.x86_64.{repo_underscored}-{instance_id}",
        )
        assert case.graders[0].config["docker_image"] == (
            "swebench/sweb.eval.x86_64.acme_1776_widget-acme__widget-42"
        )


class TestBuildScenario:
    def test_uses_the_claude_code_adapter_and_keeps_the_workspace(self):
        scenario = loader.build_scenario([_instance()], repo_root="/repos")
        assert scenario.agent.adapter == "claude_code"
        # The grader runs after the adapter returns and needs the tree.
        assert scenario.agent.config["keep_workspace"] is True

    def test_agent_config_is_the_comparison_axis(self):
        scenario = loader.build_scenario(
            [_instance()], repo_root="/repos",
            agent_config={"model": "claude-opus-4-6", "max_turns": 80},
        )
        assert scenario.agent.config["model"] == "claude-opus-4-6"
        assert scenario.agent.config["max_turns"] == 80          # overrides default
        assert scenario.agent.config["isolation"] == "worktree"  # default kept

    def test_process_graders_are_attached_but_never_gate(self):
        scenario = loader.build_scenario([_instance()], repo_root="/repos")
        names = [g.name for g in scenario.default_graders]
        assert "cost_budget" in names and "turn_count" in names
        assert not any(g.gate for g in scenario.default_graders)

    def test_process_graders_can_be_switched_off(self):
        scenario = loader.build_scenario(
            [_instance()], repo_root="/repos", process_graders=[]
        )
        assert scenario.default_graders == []

    def test_clone_commands_are_per_repo_not_per_instance(self):
        instances = [
            _instance(instance_id="a"),
            _instance(instance_id="b"),
            _instance(instance_id="c", repo="acme/gadget"),
        ]
        commands = loader.clone_commands(instances, "/repos")
        assert len(commands) == 2
        assert any("acme/widget /repos/acme__widget" in c for c in commands)


# ---------------------------------------------------------------------------
# Grader helpers
# ---------------------------------------------------------------------------


class TestGraderHelpers:
    def test_as_list_accepts_both_export_shapes(self):
        """Some SWE-bench exports ship these as JSON strings, others as lists."""
        assert swebench_graders._as_list(["a", "b"]) == ["a", "b"]
        assert swebench_graders._as_list('["a", "b"]') == ["a", "b"]
        assert swebench_graders._as_list("") == []
        assert swebench_graders._as_list(None) == []

    def test_paths_in_patch(self):
        patch = (
            "diff --git a/tests/test_a.py b/tests/test_a.py\n"
            "--- a/tests/test_a.py\n+++ b/tests/test_a.py\n"
            "diff --git a/tests/test_b.py b/tests/test_b.py\n"
            "--- /dev/null\n+++ b/tests/test_b.py\n"
        )
        assert swebench_graders._paths_in_patch(patch) == [
            "tests/test_a.py",
            "tests/test_b.py",
        ]

    def test_parse_pytest_report(self):
        stdout = (
            "=== short test summary info ===\n"
            "PASSED tests/test_a.py::test_one\n"
            "FAILED tests/test_a.py::test_two - AssertionError\n"
            "ERROR tests/test_b.py::test_three\n"
        )
        assert swebench_graders._parse_pytest_report(stdout) == {
            "tests/test_a.py::test_one": "PASSED",
            "tests/test_a.py::test_two": "FAILED",
            "tests/test_b.py::test_three": "ERROR",
        }

    def test_unparseable_output_yields_nothing(self):
        """Which the grader treats as 'everything failed', never as 'fine'."""
        assert swebench_graders._parse_pytest_report("Ran 4 tests in 0.1s\nOK\n") == {}


# ---------------------------------------------------------------------------
# The protocol, against a real repo
# ---------------------------------------------------------------------------

# Right for negatives (the PASS_TO_PASS test), wrong for positives (the
# FAIL_TO_PASS one) — the shape a real instance has before the fix.
_LIB_BUGGY = "def add(a, b):\n    return -abs(a) - abs(b)\n"
_LIB_FIXED = "def add(a, b):\n    return a + b\n"
# Fixes the new test and breaks the old one.
_LIB_REGRESSED = "def add(a, b):\n    return abs(a) + abs(b)\n"

_TESTS_BASE = "from lib import add\n\n\ndef test_negatives():\n    assert add(-2, -3) == -5\n"
_TEST_PATCH = """\
diff --git a/tests/test_lib.py b/tests/test_lib.py
--- a/tests/test_lib.py
+++ b/tests/test_lib.py
@@ -3,3 +3,7 @@ from lib import add

 def test_negatives():
     assert add(-2, -3) == -5
+
+
+def test_positives():
+    assert add(2, 3) == 5
"""


@pytest.fixture
def workspace(tmp_path):
    """A git repo in the state an agent would have left it (unfixed)."""
    repo = tmp_path / "work"
    (repo / "tests").mkdir(parents=True)
    (repo / "lib.py").write_text(_LIB_BUGGY, encoding="utf-8")
    (repo / "tests" / "test_lib.py").write_text(_TESTS_BASE, encoding="utf-8")
    run = lambda *a: subprocess.run(  # noqa: E731
        ["git", *a], cwd=repo, check=True, capture_output=True, text=True
    )
    run("init", "-q")
    run("config", "user.email", "t@e.com")
    run("config", "user.name", "T")
    run("add", "-A")
    run("commit", "-qm", "base")
    return repo


def _ctx(workspace: Path):
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=workspace, capture_output=True, text=True
    ).stdout.strip()
    artifact = CodeArtifact(metadata={"workspace": str(workspace)})
    return GradeContext(
        outcome=Outcome(artifacts=[artifact]),
        metadata={
            "test_patch": _TEST_PATCH,
            "fail_to_pass": ["tests/test_lib.py::test_positives"],
            "pass_to_pass": ["tests/test_lib.py::test_negatives"],
            "base_commit": commit,
        },
    )


def _grader(**config):
    return get_grader("swebench_tests")(config)


class TestScoringProtocol:
    async def test_a_correct_fix_resolves(self, workspace):
        (workspace / "lib.py").write_text(_LIB_FIXED, encoding="utf-8")
        result = await _grader().grade(_ctx(workspace))

        assert result.passed is True
        assert result.score == 1.0
        assert result.details["resolved"] is True

    async def test_no_fix_does_not_resolve(self, workspace):
        result = await _grader().grade(_ctx(workspace))

        assert result.passed is False
        # Partial credit is visible even though the verdict is binary: the
        # PASS_TO_PASS test still passes, so this is 1 of 2, not 0 of 2.
        assert result.score == pytest.approx(0.5)
        assert "fail_to_pass_incomplete" in result.failure_tags

    async def test_breaking_an_existing_test_is_tagged_a_regression(self, workspace):
        (workspace / "lib.py").write_text(_LIB_REGRESSED, encoding="utf-8")
        result = await _grader().grade(_ctx(workspace))

        assert result.passed is False
        assert result.details["fail_to_pass"]["passed"] == 1   # new test green
        assert result.details["pass_to_pass"]["passed"] == 0   # old one broken
        assert "regression" in result.failure_tags

    async def test_an_agent_cannot_win_by_editing_the_tests(self, workspace):
        """The reset step is the whole reason this benchmark is not gameable."""
        (workspace / "tests" / "test_lib.py").write_text(
            "def test_positives():\n    assert True\n"
            "def test_negatives():\n    assert True\n",
            encoding="utf-8",
        )
        result = await _grader().grade(_ctx(workspace))

        assert result.passed is False
        # The gutted file was checked back out, so the real PASS_TO_PASS test
        # ran (and passed) while the unfixed FAIL_TO_PASS test did not.
        assert result.details["pass_to_pass"]["passed"] == 1
        assert result.details["fail_to_pass"]["passed"] == 0

    async def test_a_test_that_never_ran_counts_as_failed(self, workspace):
        """Not as absent — otherwise a broken environment reads as a pass.

        An unknown node id makes pytest fail collection and run *nothing*, so
        the sibling test goes down with it. That is the conservative reading and
        the right one: no evidence is not evidence of passing.
        """
        (workspace / "lib.py").write_text(_LIB_FIXED, encoding="utf-8")
        context = _ctx(workspace)
        context.metadata["fail_to_pass"] = [
            "tests/test_lib.py::test_positives",
            "tests/test_lib.py::test_does_not_exist",
        ]
        result = await _grader().grade(context)

        assert result.passed is False
        assert result.details["fail_to_pass"]["passed"] == 0
        assert "tests/test_lib.py::test_does_not_exist" in (
            result.details["fail_to_pass"]["failing"]
        )

    async def test_nothing_running_is_labelled_not_just_scored_zero(self, workspace):
        """0/24 from a collection error looks exactly like 0/24 from a lazy
        agent. The tag and the captured output tell them apart."""
        context = _ctx(workspace)
        context.metadata["fail_to_pass"] = ["tests/test_lib.py::test_does_not_exist"]
        result = await _grader().grade(context)

        assert result.score == 0.0
        assert "no_tests_ran" in result.failure_tags
        assert result.details["no_tests_ran"] is True
        assert "test_does_not_exist" in result.details["output_tail"]

    async def test_missing_workspace_is_unscored_not_zero(self):
        """score=None keeps a harness failure out of the score denominator."""
        result = await _grader().grade(
            GradeContext(
                outcome=Outcome(artifacts=[CodeArtifact()]),
                metadata={
                    "test_patch": _TEST_PATCH,
                    "fail_to_pass": ["tests/test_lib.py::test_positives"],
                    "pass_to_pass": [],
                    "base_commit": "abc",
                },
            )
        )
        assert result.passed is False
        assert result.score is None
        assert "workspace" in result.error
        assert "harness_error" in result.failure_tags

    async def test_an_unappliable_test_patch_is_reported(self, workspace):
        context = _ctx(workspace)
        context.metadata["test_patch"] = (
            "diff --git a/tests/nope.py b/tests/nope.py\n"
            "--- a/tests/nope.py\n+++ b/tests/nope.py\n"
            "@@ -1,1 +1,1 @@\n-nothing\n+something\n"
        )
        result = await _grader().grade(context)

        assert result.passed is False
        assert result.score is None
        assert "test_patch" in result.error

    async def test_missing_fail_to_pass_is_refused(self, workspace):
        """An instance with nothing to prove would otherwise 'resolve' trivially."""
        context = _ctx(workspace)
        context.metadata["fail_to_pass"] = []
        result = await _grader().grade(context)

        assert result.passed is False
        assert result.score is None
