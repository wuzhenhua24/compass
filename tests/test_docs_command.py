"""Tests for `compass docs`.

The command exists so the terminal is enough — for a person or for an agent
driving the CLI. Two properties worth pinning: it reads the *current* files
rather than a stale install-time snapshot, and its topic list is curated so the
output does not depend on which local notes happen to be in the checkout.
"""

from __future__ import annotations

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
