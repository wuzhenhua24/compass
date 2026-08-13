"""Tests for ``skill_trigger`` — did the agent actually load the skill.

The traces here are the two shapes Claude Code really produces: a ``Skill``
tool call naming the skill, and a plain ``Read`` of a file inside the installed
skill directory. Both count as a load; neither is the only one.
"""

from __future__ import annotations

from pathlib import Path

from compass.core.skills import read_skill_name, resolve_skill_name
from compass.core.transcript import ToolCall, Transcript
from compass.graders.base import GradeContext
from compass.graders.registry import get_grader

SKILL_ROOT = "/tmp/wt_ab12/.claude/skills/report-writer"


def _transcript(calls: list[ToolCall], *, installed: str = "") -> Transcript:
    transcript = Transcript(task_id="q3-report", trial_id="t1")
    transcript.tool_calls.extend(calls)
    if installed:
        transcript.metadata["skills"] = [
            {"name": installed, "source": f"./skills/{installed}-v2", "digest": "abc123"}
        ]
    return transcript


def _context(calls: list[ToolCall], *, installed: str = "") -> GradeContext:
    transcript = _transcript(calls, installed=installed)
    return GradeContext(transcript=transcript, outcome=transcript.outcome)


def _grader(**config):
    return get_grader("skill_trigger")(config)


def _read(path: str, turn: int = 1) -> ToolCall:
    return ToolCall(tool_name="Read", input={"file_path": path}, turn_index=turn)


def _bash(command: str, turn: int = 2) -> ToolCall:
    return ToolCall(tool_name="Bash", input={"command": command}, turn_index=turn)


# ---------------------------------------------------------------------------
# The two ways a skill loads
# ---------------------------------------------------------------------------


async def test_the_skill_tool_naming_it_counts_as_a_load():
    context = _context(
        [ToolCall(tool_name="Skill", input={"skill": "report-writer"}, turn_index=1)]
    )

    result = await _grader(skill="report-writer").grade(context)

    assert result.passed is True
    assert result.score == 1.0
    assert result.metrics["skill_triggered"] is True
    assert result.metrics["skill_trigger_turn"] == 1.0
    assert result.details["signals"] == ["skill_tool"]
    assert "skill_via_skill_tool" in result.tags


async def test_reading_the_skill_md_counts_as_a_load():
    """Claude Code loads a skill by reading it as often as by calling the tool.

    Counting only the tool call would report a 0% trigger rate for a skill that
    is working perfectly.
    """
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md", turn=2)])

    result = await _grader(skill="report-writer").grade(context)

    assert result.passed is True
    assert result.details["signals"] == ["file"]
    assert result.metrics["skill_trigger_turn"] == 2.0
    assert result.details["evidence"][0]["tool"] == "Read"


async def test_a_reference_page_under_the_skill_counts_too():
    context = _context([_read(f"{SKILL_ROOT}/references/tone.md")])
    result = await _grader(skill="report-writer").grade(context)
    assert result.metrics["skill_triggered"] is True


async def test_a_run_that_never_touched_the_skill_fails():
    context = _context([_read("src/app.py"), _bash("pytest -q")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.passed is False
    assert result.score == 0.0
    assert result.metrics["skill_triggered"] is False
    assert result.failure_tags == ["not_triggered"]
    assert "never loaded" in result.reasoning


async def test_prose_that_merely_mentions_the_name_is_not_a_load():
    """A description-triggering metric that counts the agent *talking* about the
    skill would report improvements that are not there."""
    context = _context(
        [
            _bash("echo 'writing the report-writer output by hand'"),
            ToolCall(
                tool_name="Write",
                input={"file_path": "out.md", "content": "See the report-writer docs."},
            ),
        ]
    )

    result = await _grader(skill="report-writer").grade(context)

    assert result.metrics["skill_triggered"] is False


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------


async def test_a_negative_control_passes_when_the_skill_stays_out():
    context = _context([_read("src/app.py")])

    result = await _grader(skill="report-writer", should_trigger=False).grade(context)

    assert result.passed is True
    assert result.score == 1.0
    assert "correctly stayed out" in result.reasoning


async def test_a_description_that_fires_on_everything_is_caught():
    """The regression a positives-only eval cannot see: a 'pushier' description
    that starts winning requests it has no business handling."""
    context = _context([ToolCall(tool_name="Skill", input={"skill": "report-writer"})])

    result = await _grader(skill="report-writer", should_trigger=False).grade(context)

    assert result.passed is False
    assert result.failure_tags == ["unexpected_trigger"]


# ---------------------------------------------------------------------------
# Bundled resources
# ---------------------------------------------------------------------------


async def test_a_bundled_script_that_was_actually_run():
    context = _context(
        [
            _read(f"{SKILL_ROOT}/SKILL.md"),
            _bash(f"python {SKILL_ROOT}/scripts/render.py out.md"),
        ]
    )

    result = await _grader(
        skill="report-writer", required_resources=["scripts/render.py"]
    ).grade(context)

    assert result.passed is True
    assert result.details["resources"] == {"scripts/render.py": True}
    assert result.metrics["skill_resources_used"] == 1.0


async def test_a_bundled_script_the_agent_reinvented_instead():
    """The skill loaded and the answer may well be right — but the script it
    bundles has stopped being reachable, and only the transcript shows it."""
    context = _context(
        [
            _read(f"{SKILL_ROOT}/SKILL.md"),
            _bash("python my_own_render.py out.md"),
        ]
    )

    result = await _grader(
        skill="report-writer", required_resources=["scripts/render.py"]
    ).grade(context)

    assert result.passed is False
    assert result.score == 0.5  # loaded, but the resource check failed
    assert result.failure_tags == ["resource_unused"]
    assert "went unused" in result.reasoning


# ---------------------------------------------------------------------------
# Which skill are we asking about
# ---------------------------------------------------------------------------


async def test_the_skill_name_is_inherited_from_the_installed_one():
    """A scenario that already told the adapter which skill to install should
    not have to repeat it in every grader config."""
    context = _context(
        [_read(f"{SKILL_ROOT}/SKILL.md")], installed="report-writer"
    )

    result = await _grader().grade(context)

    assert result.details["skill"] == "report-writer"
    assert result.passed is True


async def test_a_directory_is_accepted_where_a_name_is_expected(tmp_path: Path):
    """``skill: ./skills/report-writer-v2`` names the same skill as
    ``skill: report-writer`` — accepting only one spelling produces a
    comparison that runs clean and measures nothing."""
    source = tmp_path / "report-writer-v2"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: report-writer\n---\n\n# rw\n", encoding="utf-8"
    )
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md")])

    result = await _grader(skill=str(source)).grade(context)

    assert result.details["skill"] == "report-writer"
    assert result.passed is True


async def test_with_no_skill_to_look_for_nothing_is_measured():
    """score=None keeps a configuration mistake out of the average, instead of
    scoring it as an agent failure."""
    context = _context([_read("src/app.py")])

    result = await _grader().grade(context)

    assert result.score is None
    assert result.passed is False
    assert result.error is not None and "skill:" in result.error


async def test_a_baseline_arm_reports_an_untriggered_skill_not_an_error():
    """The no-skill arm of a sweep has nothing installed, so the scenario names
    the skill explicitly — and the arm honestly scores 'never loaded'."""
    context = _context([_read("src/app.py"), _bash("cat > out.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.score == 0.0
    assert result.error is None
    assert result.metrics["skill_triggered"] is False


# ---------------------------------------------------------------------------
# Name resolution shared with the adapter
# ---------------------------------------------------------------------------


def test_frontmatter_name_wins_over_the_directory_name(tmp_path: Path):
    skill = tmp_path / "report-writer-v3"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: 'report-writer'\ndescription: x\n---\n", encoding="utf-8"
    )
    assert read_skill_name(skill) == "report-writer"
    assert resolve_skill_name(str(skill)) == "report-writer"


def test_a_skill_without_frontmatter_falls_back_to_its_directory(tmp_path: Path):
    skill = tmp_path / "legacy-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# legacy\n", encoding="utf-8")
    assert read_skill_name(skill) == ""
    assert resolve_skill_name(str(skill)) == "legacy-skill"


def test_a_bare_name_passes_through():
    assert resolve_skill_name("report-writer") == "report-writer"
    assert resolve_skill_name("  ") == ""
