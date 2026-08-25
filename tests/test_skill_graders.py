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


# ---------------------------------------------------------------------------
# Reading the skill vs rewriting it
#
# Both reach into the skill's directory, and only one is a load. An agent asked
# to *improve* a skill edits its SKILL.md without ever using it; counting that
# as a trigger would report an edit as a hit, and every trigger rate measured
# over such a run would be wrong in the flattering direction.
# ---------------------------------------------------------------------------


async def test_a_redirect_into_the_skill_is_not_a_load():
    context = _context([_bash(f"echo 'rewritten' > {SKILL_ROOT}/SKILL.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is False
    assert result.metrics["skill_write_calls"] == 1.0
    assert "skill_write_only" in result.tags
    assert "wrote into its directory" in result.reasoning


async def test_deleting_the_skill_is_not_a_load():
    context = _context([_bash(f"rm -rf {SKILL_ROOT}/references/")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is False
    assert result.details["writes"][0]["tool"] == "Bash"


async def test_the_write_tools_are_not_loads():
    context = _context(
        [ToolCall(tool_name="Write", input={"file_path": f"{SKILL_ROOT}/SKILL.md"})]
    )

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is False
    assert result.metrics["skill_write_calls"] == 1.0


async def test_reading_before_redirecting_elsewhere_is_still_a_load():
    """The skill is on the read side of the `>`; only `/tmp/out` is written."""
    context = _context([_bash(f"cat {SKILL_ROOT}/SKILL.md > /tmp/out.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is True
    assert result.metrics["skill_write_calls"] == 0.0


async def test_a_negative_control_is_not_tripped_by_an_edit():
    """The half of this that matters for scoring: no false `unexpected_trigger`."""
    context = _context([_bash(f"echo x >> {SKILL_ROOT}/SKILL.md")])

    result = await _grader(skill="report-writer", should_trigger=False).grade(context)

    assert result.passed is True
    assert result.failure_tags == []


async def test_a_run_that_both_edits_and_reads_counts_as_a_load():
    context = _context(
        [
            _bash(f"echo x > {SKILL_ROOT}/SKILL.md", turn=1),
            _read(f"{SKILL_ROOT}/SKILL.md", turn=2),
        ]
    )

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is True
    assert result.metrics["skill_write_calls"] == 1.0
    assert "skill_write_only" not in result.tags


# ---------------------------------------------------------------------------
# Variables in a shell command
# ---------------------------------------------------------------------------


async def test_a_read_through_a_shell_variable_counts():
    context = _context([_bash(f"D={SKILL_ROOT}; cat $D/SKILL.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is True


async def test_an_exported_variable_counts_too():
    context = _context([_bash(f"export D={SKILL_ROOT} && python $D/scripts/render.py")])

    result = await _grader(
        skill="report-writer", required_resources=["scripts/render.py"]
    ).grade(context)

    assert result.details["triggered"] is True
    assert result.details["resources"] == {"scripts/render.py": True}


async def test_assigning_a_path_without_using_it_is_not_a_load():
    context = _context([_bash(f"D={SKILL_ROOT}; echo done")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is False


async def test_an_unset_variable_is_left_alone_rather_than_emptied():
    """Expanding `$NOPE` to "" could splice a match out of unrelated pieces."""
    context = _context([_bash("cat $NOPE/SKILL.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is False


async def test_an_unparseable_command_keeps_the_old_reading():
    """A dangling quote falls back to the raw string — a stale false positive
    beats a new false negative, since this is the trigger signal itself."""
    context = _context([_bash(f"cat '{SKILL_ROOT}/SKILL.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is True


async def test_running_a_bundled_script_is_still_a_load():
    context = _context([_bash(f"python {SKILL_ROOT}/scripts/summarize.py --out x")])

    result = await _grader(skill="report-writer").grade(context)

    assert result.details["triggered"] is True
    assert result.metrics["skill_write_calls"] == 0.0


# ---------------------------------------------------------------------------
# acceptable_skills: when more than one skill is a right answer
#
# In a repo with several skills, "report-writer or doc-writer, either is fine"
# is a legitimate expectation — and scoring it as a miss would push whoever
# wrote the case toward deleting it, which is how a suite quietly loses its
# hardest routing questions.
# ---------------------------------------------------------------------------

DOC_ROOT = "/tmp/wt_ab12/.claude/skills/doc-writer"


async def test_an_accepted_alternate_counts_as_correct_routing():
    context = _context([_read(f"{DOC_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.passed is True
    assert result.details["triggered"] is True
    assert result.details["matched_skill"] == "doc-writer"


async def test_the_alternate_does_not_hide_that_the_skill_lost_the_routing():
    """The A/B question survives the widening: `skill_triggered` says the
    routing was acceptable, `skill_primary_triggered` says who won it."""
    context = _context([_read(f"{DOC_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.metrics["skill_triggered"] is True
    assert result.metrics["skill_primary_triggered"] is False
    assert "skill_alternate" in result.tags


async def test_the_primary_is_reported_even_when_an_alternate_went_first():
    context = _context(
        [_read(f"{DOC_ROOT}/SKILL.md", turn=1), _read(f"{SKILL_ROOT}/SKILL.md", turn=3)]
    )

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.details["matched_skill"] == "report-writer"
    assert result.metrics["skill_primary_triggered"] is True
    assert "skill_alternate" not in result.tags


async def test_an_unrelated_skill_is_still_a_miss():
    context = _context([_read("/tmp/wt_ab12/.claude/skills/chart-maker/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.passed is False
    assert result.details["matched_skill"] == ""
    assert "not_triggered" in result.failure_tags


async def test_the_skill_tool_can_name_an_alternate():
    context = _context(
        [ToolCall(tool_name="Skill", input={"skill": "doc-writer"}, turn_index=1)]
    )

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.details["matched_skill"] == "doc-writer"
    assert result.details["signals"] == ["skill_tool"]


async def test_the_metric_is_absent_when_no_alternates_are_configured():
    """`skill_triggered` alone is the whole answer then; a second metric that
    always duplicates it would just be noise in every compare."""
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md")])

    result = await _grader(skill="report-writer").grade(context)

    assert "skill_primary_triggered" not in result.metrics


async def test_the_primarys_resources_are_not_held_against_an_alternate():
    """`required_resources` name files bundled with the skill under test. When
    an accepted neighbour answered, there is nothing to check — not a
    checklist of failures."""
    context = _context([_read(f"{DOC_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer",
        acceptable_skills=["doc-writer"],
        required_resources=["scripts/render.py"],
    ).grade(context)

    assert result.passed is True
    assert result.details["resources"] == {}
    assert "resource_unused" not in result.failure_tags
    assert "skill_resources_used" not in result.metrics


async def test_the_resources_are_still_checked_when_the_skill_itself_answered():
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer",
        acceptable_skills=["doc-writer"],
        required_resources=["scripts/render.py"],
    ).grade(context)

    assert result.passed is False
    assert result.details["resources"] == {"scripts/render.py": False}
    assert "resource_unused" in result.failure_tags


async def test_a_negative_control_rejects_the_list_rather_than_ignoring_it():
    """It would silently do nothing here, and a config that measures nothing
    must not read as an agent that behaved."""
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"], should_trigger=False
    ).grade(context)

    assert result.score is None
    assert "should_trigger" in result.error


async def test_an_alternate_given_as_a_directory_resolves_to_its_real_name(tmp_path: Path):
    source = tmp_path / "doc-writer-v7"
    source.mkdir()
    (source / "SKILL.md").write_text(
        "---\nname: doc-writer\ndescription: x\n---\n", encoding="utf-8"
    )
    context = _context([_read(f"{DOC_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=[str(source)]
    ).grade(context)

    assert result.details["acceptable_skills"] == ["doc-writer"]
    assert result.details["matched_skill"] == "doc-writer"


async def test_listing_the_skill_itself_is_harmless():
    context = _context([_read(f"{SKILL_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["report-writer", "doc-writer"]
    ).grade(context)

    assert result.details["matched_skill"] == "report-writer"
    assert result.metrics["skill_primary_triggered"] is True


async def test_editing_an_alternate_is_no_more_a_load_than_editing_the_primary():
    context = _context([_bash(f"echo x > {DOC_ROOT}/SKILL.md")])

    result = await _grader(
        skill="report-writer", acceptable_skills=["doc-writer"]
    ).grade(context)

    assert result.details["triggered"] is False
    assert result.details["writes"][0]["match"].endswith("doc-writer/SKILL.md")
