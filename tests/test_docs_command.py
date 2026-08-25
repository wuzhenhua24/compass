"""Tests for `compass docs`.

The command exists so the terminal is enough — for a person or for an agent
driving the CLI. Two properties worth pinning: it reads the *current* files
rather than a stale install-time snapshot, and its topic list is curated so the
output does not depend on which local notes happen to be in the checkout.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from click.testing import CliRunner

from compass.cli.main import cli
from compass.docs_index import TOPICS, get_topic, read_topic, topic_names

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


# ===================================================================
# The topic index
# ===================================================================


class TestTopicIndex:
    def test_every_topic_resolves_to_a_real_file(self):
        for topic in TOPICS:
            assert read_topic(topic) is not None, f"{topic.name} is not readable"

    def test_every_topic_source_is_tracked_in_the_repo(self):
        for topic in TOPICS:
            assert (REPO_ROOT / topic.source).is_file()

    def test_local_working_notes_are_not_topics(self):
        """docs/ also holds gitignored notes; globbing would surface them."""
        sources = {t.source for t in TOPICS}
        for note in ("docs/interview.md", "docs/todos.md", "docs/idea.md"):
            assert note not in sources

    def test_lookup_is_forgiving_about_case_and_suffix(self):
        assert get_topic("GRADERS") is get_topic("graders")
        assert get_topic("graders.md") is get_topic("graders")

    def test_unknown_topic_is_none(self):
        assert get_topic("nope") is None

    def test_names_are_unique(self):
        names = topic_names()
        assert len(names) == len(set(names))


class TestPackaging:
    def test_every_topic_is_force_included_in_the_wheel(self):
        """A topic missing from the wheel would 404 for pip-installed users."""
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for topic in TOPICS:
            assert f'"{topic.source}" = "compass/docs/' in pyproject, (
                f"{topic.source} is not force-included in the wheel"
            )

    def test_gitignored_notes_are_not_shipped(self):
        pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        for note in ("docs/interview.md", "docs/todos.md", "docs/idea.md"):
            assert f'"{note}"' not in pyproject


# ===================================================================
# The command
# ===================================================================


class TestDocsCommand:
    def test_bare_invocation_lists_the_topics(self, runner):
        result = runner.invoke(cli, ["docs"])
        assert result.exit_code == 0
        for name in topic_names():
            assert name in result.output

    def test_a_topic_prints_raw_markdown(self, runner):
        """Raw so it pipes — that is the point of having it in the CLI."""
        result = runner.invoke(cli, ["docs", "graders"])
        assert result.exit_code == 0
        assert result.output.startswith("# Grader 体系")

    def test_the_readme_topic_reads_the_current_file(self, runner):
        """Package metadata carries a snapshot from install time; files do not.

        In an editable checkout that snapshot goes stale as soon as README.md
        is edited, and printing outdated docs is worse than printing none.
        """
        result = runner.invoke(cli, ["docs", "readme"])
        expected = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        assert result.output.strip() == expected.strip()

    def test_all_concatenates_every_topic(self, runner):
        result = runner.invoke(cli, ["docs", "--all"])
        assert result.exit_code == 0
        assert "# Grader 体系" in result.output
        assert "# 结果分析、对比与报告" in result.output
        assert result.output.count("\n---\n") >= len(TOPICS) - 1

    def test_an_unknown_topic_fails_and_lists_the_real_ones(self, runner):
        result = runner.invoke(cli, ["docs", "not-a-topic"])
        assert result.exit_code == 1
        assert "No such topic" in result.output
        assert "graders" in result.output

    def test_a_missing_file_is_reported_not_crashed(self, runner, monkeypatch):
        """A stripped installation should say so, not raise."""
        import compass.cli.main as main

        monkeypatch.setattr(
            "compass.docs_index.read_topic", lambda topic: None, raising=True
        )
        result = runner.invoke(main.cli, ["docs", "graders"])
        assert result.exit_code == 1
        assert "not available" in result.output

    def test_the_command_is_registered(self):
        assert "docs" in cli.commands


# ===================================================================
# The cheatsheet must stay true
# ===================================================================


class TestCheatsheetAccuracy:
    """A cheatsheet is only worth having if it is accurate.

    It is the page an agent is most likely to act on without reading further,
    so every name in it is checked against the code rather than trusted.
    """

    @pytest.fixture(scope="class")
    def text(self) -> str:
        return (REPO_ROOT / "docs" / "cheatsheet.md").read_text(encoding="utf-8")

    def test_it_is_a_registered_topic(self):
        assert get_topic("cheatsheet") is not None

    def test_it_stays_short_enough_to_read_in_one_pass(self, text):
        """Its whole reason to exist is being smaller than docs/graders.md."""
        graders_doc = (REPO_ROOT / "docs" / "graders.md").read_text(encoding="utf-8")
        assert len(text) < len(graders_doc) / 2

    @staticmethod
    def _table_rows(text: str, *, under: str | None = None) -> list[tuple[str, list[str]]]:
        """(label, grader names) from the grader tables.

        The cheatsheet has two of them and they group on different axes — one by
        scope, one by domain — so *under* selects the table whose header line
        contains that string. Without it, both are returned, which is what the
        coverage checks want.
        """
        rows: list[tuple[str, list[str]]] = []
        selected = under is None
        for line in text.splitlines():
            if line.startswith("|") and "---" not in line and "`" not in line:
                selected = under is None or under in line
                continue
            m = re.match(r"\|\s+\*\*(\w+)[^*]*\*\*[^|]*\|(.+)\|", line)
            if m and selected:
                rows.append((m.group(1), re.findall(r"`([a-z_]+)`", m.group(2))))
        return rows

    def test_the_grader_table_names_only_real_graders(self, text):
        import compass.graders  # noqa: F401  (registers everything)
        from compass.graders.registry import get_grader as lookup

        rows = self._table_rows(text)
        assert rows, "the grader table could not be parsed"
        for _scope, names in rows:
            for name in names:
                lookup(name)  # raises if the cheatsheet invented one

    def test_every_scope_in_the_table_is_correct(self, text):
        """A wrong scope tells the reader a grader can do something it cannot."""
        import compass.graders  # noqa: F401
        from compass.graders.registry import get_grader as lookup

        rows = self._table_rows(text, under="scope")
        assert rows, "the scope table could not be parsed"
        for scope, names in rows:
            if scope == "human":
                continue  # the human row groups by grader type, not scope
            for name in names:
                actual = getattr(lookup(name).grader_scope, "value", None)
                assert actual == scope, f"{name} is {actual}, listed under {scope}"

    def test_every_domain_in_the_table_is_correct(self, text):
        """Which half a grader is in is the cheatsheet's other claim about it."""
        from compass.graders.domains import domain_of

        rows = self._table_rows(text, under="领域")
        assert rows, "the domain table could not be parsed"
        for domain, names in rows:
            for name in names:
                actual = domain_of(name)
                assert actual == domain, f"{name} is in {actual}, listed under {domain}"

    def test_the_scope_table_holds_only_framework_graders(self, text):
        """The scope table is the framework's half; a domain grader there is a leak."""
        from compass.graders.domains import domain_of

        strays = {
            name: domain_of(name)
            for _scope, names in self._table_rows(text, under="scope")
            for name in names
            if domain_of(name) is not None
        }
        assert strays == {}, f"domain graders listed as framework: {strays}"

    @staticmethod
    def _builtin_graders() -> set[str]:
        """Graders that ship with Compass, by module — not by name.

        The registry is process-global: a full-suite run also holds the fakes
        other test modules register, plus the domain graders from
        examples/ops_qa. Filtering on where the class is defined is exact;
        filtering on a name prefix would only catch the ones that happen to
        follow the convention.
        """
        import compass.graders  # noqa: F401  (registers everything)
        from compass.graders.base import GraderType
        from compass.graders.registry import get_grader, list_graders

        return {
            name
            for t in GraderType
            for name in list_graders(t)
            if get_grader(name).__module__.startswith("compass.graders.")
        }

    def test_the_table_covers_every_registered_grader(self, text):
        """A grader missing from the table is one an agent will never reach for."""
        listed = {n for _s, names in self._table_rows(text) for n in names}
        missing = self._builtin_graders() - listed
        assert missing == set(), f"not in the cheatsheet table: {sorted(missing)}"

    def test_every_grader_named_in_a_yaml_example_is_real(self, text):
        import compass.graders  # noqa: F401
        from compass.graders.registry import get_grader as lookup

        for name in set(re.findall(r"name:\s*([a-z_]+)", text)):
            if name in {"accuracy", "my_check"}:  # rubric criteria / the example
                continue
            lookup(name)

    def test_the_claimed_grader_count_is_right(self, text):
        """A stale count is the cheapest kind of lie to leave in a cheatsheet."""
        total = len(self._builtin_graders())
        claimed = re.search(r"内置 grader 速查（(\d+) 个）", text)
        assert claimed, "the cheatsheet no longer states a grader count"
        assert int(claimed.group(1)) == total

    def test_every_yaml_field_it_documents_exists(self, text):
        from compass.core.scenario import (
            AggregationConfig,
            ExpectedConfig,
            GraderConfig,
            InputConfig,
            MetricsConfig,
            Scenario,
            TestCase,
        )

        for model, fields in (
            (Scenario, ["default_graders", "default_aggregation", "leak_markers"]),
            (TestCase, ["expect", "expect_reason", "stage", "trials", "expected"]),
            (InputConfig, ["prompt", "negative_prompt", "params", "reference_images"]),
            (GraderConfig, ["weight", "required", "gate", "creates", "config"]),
            (AggregationConfig, ["pass_threshold", "required_graders", "short_circuit"]),
            (MetricsConfig, ["pass_at_k", "consistency"]),
            (ExpectedConfig, ["contains", "equals", "equals_json", "similar_to"]),
        ):
            for field in fields:
                assert field in model.model_fields, f"{model.__name__}.{field} is gone"
                assert field in text, f"cheatsheet omits {model.__name__}.{field}"

    def test_every_cli_command_it_mentions_exists(self, text):
        for command in cli.commands:
            if f"compass {command}" in text:
                continue
            # `list` and `init` are referenced too; every mentioned one must be real
        mentioned = {
            line.split()[1]
            for line in text.splitlines()
            if line.startswith("compass ") and len(line.split()) > 1
        }
        unknown = mentioned - set(cli.commands)
        assert not unknown, f"cheatsheet mentions non-existent commands: {unknown}"

    def test_every_context_accessor_it_documents_exists(self, text):
        import re

        from compass.graders.base import GradeContext

        used = set(re.findall(r"context\.([a-z_]+)", text))
        for name in used:
            assert hasattr(GradeContext, name) or name in GradeContext.__annotations__, (
                f"cheatsheet documents context.{name}, which does not exist"
            )

    def test_the_short_circuit_modes_are_real(self, text):
        from compass.core.scenario import ShortCircuitMode

        for mode in ShortCircuitMode:
            assert mode.value in text

    def test_it_does_not_document_the_removed_assertions_field(self, text):
        """`expected.assertions` was deleted; the cheatsheet must not resurrect it."""
        from compass.core.scenario import ExpectedConfig

        assert "assertions" not in ExpectedConfig.model_fields
        assert "`assertions:`" not in text

    def test_it_documents_only_real_expected_fields(self, text):
        """Every `expected:` key the cheatsheet shows must still exist."""
        from compass.core.scenario import ExpectedConfig

        block = text.split("### `expected:` 简化写法", 1)[1].split("```", 2)[1]
        shown = {
            line.split(":")[0].strip()
            for line in block.splitlines()
            # Indented lines are the keys; `expected:` itself sits at column 0
            if line.startswith("  ") and ":" in line and not line.strip().startswith("#")
        }
        assert shown, "the expected: block could not be parsed"
        for key in shown:
            for name in key.split(" / "):  # "min_length / max_length"
                assert name in ExpectedConfig.model_fields, (
                    f"cheatsheet documents expected.{name}, which does not exist"
                )
