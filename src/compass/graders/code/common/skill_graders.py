"""Did the agent actually load the skill?

Scope: TRANSCRIPT. This is the one signal a skill evaluation cannot do without,
and the one that no other grader can stand in for.

A skill reaches the model in two stages: its *description* competes for
attention against every other skill's, and only if it wins does the body get
loaded. So a skill can be rewritten beautifully and still make the product
worse, because the new description stopped triggering — and the failure is
invisible to correctness graders, which see a plausible answer produced the
hard way, exactly as they did before. Conversely a description that got
"pushier" can start winning on requests it has no business handling, and every
correctness number keeps looking fine while the agent scaffolds a project the
user did not ask for.

Both directions are measured here, with ``should_trigger``:

- ``should_trigger: true`` — the ordinary case. Did the skill load at all?
- ``should_trigger: false`` — the *negative control*. A prompt from the
  neighbourhood that must be handled without this skill. A trigger eval built
  only from positives cannot distinguish a good description from one that
  fires on everything.

Two signals count as a load, because Claude Code has two ways of doing it: the
``Skill`` tool naming it, or any tool touching a file inside the skill's own
directory (reading ``SKILL.md``, a ``references/`` page, or running a bundled
script). ``required_resources`` builds on the second: it answers "did it use
the script we bundled, or reinvent it", which is the check that catches a skill
whose scripts have quietly stopped being reachable.

The numbers land in ``metrics`` on purpose — ``skill_triggered`` aggregates
into a trigger *rate*, which is what ``compass compare --metric
skill_triggered`` diffs between two versions of a skill, confidence interval
and all.
"""

from __future__ import annotations

from typing import Any

from compass.core.skills import resolve_skill_name
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.code.common.toolcalls import (
    EVIDENCE_CHARS as _EVIDENCE_CHARS,
)
from compass.graders.code.common.toolcalls import (
    SHELL_TOOLS as _SHELL_TOOLS,
)
from compass.graders.code.common.toolcalls import (
    WRITE_TOOLS as _WRITE_TOOLS,
)
from compass.graders.code.common.toolcalls import (
    input_strings as _strings,
)
from compass.graders.code.common.toolcalls import (
    last_segment as _last_segment,
)
from compass.graders.code.common.toolcalls import (
    shell_targets as _shell_targets,
)
from compass.graders.registry import register_grader

# The last segment of a tool name, lowercased, that means "the agent asked for
# a skill by name" rather than "the agent read a file that happens to live in
# one". Namespaced tools (``mcp__x__Skill``) reduce to the same segment.
_SKILL_TOOL_NAMES = frozenset({"skill", "skills"})


@register_grader("skill_trigger")
class SkillTriggerGrader(CodeGrader):
    """Whether a named skill was loaded during the run.

    Config:
        skill:  str  — the skill's name, or the path to its directory (either
                spelling works; a path is read down to the name in its
                frontmatter). Defaults to the skill the adapter installed, from
                ``transcript.metadata["skills"]`` — but **name it explicitly in
                any sweep that includes a no-skill baseline arm**, since that
                arm has no installed skill to inherit the name from.
        should_trigger: bool — ``True`` (default) to require the load,
                ``False`` for a negative control that must *not* load it.
        acceptable_skills: list[str] — other skills whose load also counts as
                correct routing, for a prompt more than one skill may rightly
                answer. Only meaningful with ``should_trigger: true``; see
                below.
        required_resources: list[str] — bundled files that must have been used,
                relative to the skill root (``scripts/render.py``). Each is a
                check of its own.
        path:   str  — an extra path fragment that counts as the skill's, for a
                skill installed somewhere Compass did not put it.

    ``acceptable_skills`` splits one question into two, and both are reported.
    ``skill_triggered`` becomes *was the routing acceptable*, while
    ``skill_primary_triggered`` stays *did the skill under test win it* — which
    is the one an A/B between two versions of that skill is actually asking.
    Collapsing them would make a version that lost every routing decision to a
    neighbour look identical to one that won them all.
    """

    name = "skill_trigger"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.skill: str = str(self.config.get("skill", "") or "")
        self.should_trigger: bool = bool(self.config.get("should_trigger", True))
        self.acceptable_skills: list[str] = [
            name
            for raw in self.config.get("acceptable_skills", [])
            if (name := resolve_skill_name(str(raw)))
        ]
        self.required_resources: list[str] = [
            str(r).strip().lstrip("./")
            for r in self.config.get("required_resources", [])
            if str(r).strip()
        ]
        self.path: str = str(self.config.get("path", "") or "")

    async def grade(self, context: GradeContext) -> GradeResult:
        error = self.validate_context(context)
        if error:
            return self._unmeasured(error)

        skill = self._skill_under_test(context)
        if not skill:
            return self._unmeasured(
                "skill_trigger needs a skill to look for: set `skill:` in the "
                "grader config (a name or the directory), or run through an "
                "adapter that installs one."
            )

        if self.acceptable_skills and not self.should_trigger:
            return self._unmeasured(
                "acceptable_skills only means something with should_trigger: "
                "true — it names the skills that would *also* be correct "
                "routing. A negative control asks the opposite question, so "
                "the list would silently do nothing here. Drop it, or give "
                "the skill you mean to forbid its own `skill:`."
            )

        fragments = _fragments(skill, self.path)
        # Primary first: it is what `matched_skill` reports when more than one
        # accepted skill was loaded, and `path` belongs to it alone.
        candidates = [(skill, fragments)] + [
            (name, _fragments(name, "")) for name in self.acceptable_skills if name != skill
        ]
        touches, writes = _scan(context.tool_calls, candidates)

        triggered = bool(touches)
        primary = [t for t in touches if t.skill == skill]
        # Routing went to a neighbour we said we would accept. Correct, and not
        # the same thing as the skill under test winning.
        alternate_only = triggered and not primary

        checks: list[tuple[str, bool]] = [("triggered", triggered == self.should_trigger)]
        failure_tags: list[str] = []
        if triggered != self.should_trigger:
            failure_tags.append("not_triggered" if self.should_trigger else "unexpected_trigger")

        resources: dict[str, bool] = {}
        # `required_resources` name files bundled with the *primary* skill, so
        # when an accepted alternate answered instead there is nothing to check
        # — not a checklist of failures. Not measured is not measured as zero.
        for resource in self.required_resources if not alternate_only else []:
            used = any(
                _mentions_resource(text, skill, resource, fragments)
                for touch in primary
                for text in touch.texts
            )
            resources[resource] = used
            checks.append((f"resource:{resource}", used))
            if not used:
                failure_tags.append("resource_unused")

        passed_count = sum(1 for _, ok in checks if ok)
        score = passed_count / len(checks)

        signals = sorted({t.signal for t in touches})
        metrics: dict[str, float | bool] = {
            "skill_triggered": triggered,
            "skill_touch_calls": float(len(touches)),
            "skill_write_calls": float(len(writes)),
        }
        if triggered:
            metrics["skill_trigger_turn"] = float(touches[0].turn)
        if self.required_resources and not alternate_only:
            metrics["skill_resources_used"] = float(sum(resources.values()))
        if self.acceptable_skills:
            # `skill_triggered` has widened to "the routing was acceptable".
            # This is the one that still answers "did *this* skill win it",
            # which is what an A/B between two of its versions compares.
            metrics["skill_primary_triggered"] = bool(primary)

        tags = [f"skill_via_{s}" for s in signals]
        if writes and not triggered:
            # The one case where a reader would otherwise be misled: the agent
            # was all over the skill's directory and still never loaded it.
            tags.append("skill_write_only")
        if alternate_only:
            tags.append("skill_alternate")

        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=passed_count == len(checks),
            score=score,
            details={
                "skill": skill,
                "should_trigger": self.should_trigger,
                "triggered": triggered,
                # Which accepted skill answered. The primary whenever it loaded
                # at all — even if an alternate got there first — so a reader
                # never has to work out whether one stood in for it. Ordering
                # is still in `evidence` and `skill_trigger_turn`.
                "matched_skill": (
                    skill if primary else (touches[0].skill if touches else "")
                ),
                "acceptable_skills": self.acceptable_skills,
                "signals": signals,
                "resources": resources,
                # The grader's verdict is only as useful as what backs it up:
                # a "not triggered" with the paths it *did* touch is a lead,
                # a bare False is a shrug.
                "evidence": [
                    {"tool": t.tool, "turn": t.turn, "match": t.evidence}
                    for t in touches[:5]
                ],
                # Kept apart from `evidence` because it is evidence of the
                # opposite: these calls reached into the skill's directory to
                # change it, and none of them count as loading it.
                "writes": [
                    {"tool": t.tool, "turn": t.turn, "match": t.evidence}
                    for t in writes[:5]
                ],
            },
            tags=tags,
            metrics=metrics,
            failure_tags=failure_tags,
            reasoning=_reasoning(
                skill,
                triggered,
                self.should_trigger,
                resources,
                bool(writes),
                touches[0].skill if alternate_only else "",
            ),
        )

    # ------------------------------------------------------------------

    def _skill_under_test(self, context: GradeContext) -> str:
        """The skill's name: from config, else whatever the adapter installed."""
        if self.skill:
            return resolve_skill_name(self.skill)
        installed = (context.transcript.metadata.get("skills") or []) if context.transcript else []
        for entry in installed:
            name = str(entry.get("name") or "").strip() if isinstance(entry, dict) else ""
            if name:
                # The first is the one under test: `skill:` leads the adapter's
                # install list, and `skills:` are the companions behind it.
                return name
        return ""

    def _unmeasured(self, error: str) -> GradeResult:
        """A result that is honest about having measured nothing.

        ``score=None`` keeps it out of the weighted average instead of scoring
        a configuration mistake as an agent failure.
        """
        return GradeResult(
            name=self.name,
            grader_type=self.grader_type,
            grader_scope=self.grader_scope,
            passed=False,
            score=None,
            error=error,
        )


class _Touch:
    """One tool call that reached into a skill, and which one."""

    __slots__ = ("tool", "turn", "signal", "evidence", "texts", "skill")

    def __init__(
        self,
        tool: str,
        turn: int,
        signal: str,
        evidence: str,
        texts: list[str],
        skill: str,
    ) -> None:
        self.tool = tool
        self.turn = turn
        self.signal = signal
        self.evidence = evidence
        self.texts = texts
        self.skill = skill


def _scan(
    tool_calls: list[Any], candidates: list[tuple[str, list[str]]]
) -> tuple[list[_Touch], list[_Touch]]:
    """(loads, writes) — calls that opened a skill, and calls that changed it.

    *candidates* is ``(name, fragments)`` per accepted skill, primary first, so
    a call that names two of them is attributed to the one under test.

    Both lists are "the call mentioned a path inside a skill", and the
    difference between them is the whole point: a run that rewrote ``SKILL.md``
    and never read it did not load the skill, and reporting it as a trigger
    would make an edit look like a hit.
    """
    loads: list[_Touch] = []
    writes: list[_Touch] = []
    for index, call in enumerate(tool_calls):
        tool = str(getattr(call, "tool_name", "") or getattr(call, "tool", ""))
        texts = [t.replace("\\", "/") for t in _strings(getattr(call, "input", None)) if t]
        turn_index = getattr(call, "turn_index", None)
        turn = int(turn_index) if turn_index is not None else index

        if _is_skill_tool(tool):
            name, named = _first_named(texts, candidates)
            if name:
                loads.append(
                    _Touch(tool, turn, "skill_tool", named[:_EVIDENCE_CHARS], texts, name)
                )
                continue

        read_texts, write_texts = _by_intent(tool, texts)
        name, hit = _first_hit(read_texts, candidates)
        if name:
            loads.append(
                _Touch(tool, turn, "file", hit[:_EVIDENCE_CHARS], read_texts, name)
            )
            continue
        name, hit = _first_hit(write_texts, candidates)
        if name:
            writes.append(
                _Touch(tool, turn, "write", hit[:_EVIDENCE_CHARS], write_texts, name)
            )
    return loads, writes


def _first_hit(
    texts: list[str], candidates: list[tuple[str, list[str]]]
) -> tuple[str, str]:
    """(skill, text) for the first candidate any of *texts* points inside."""
    for skill, fragments in candidates:
        for text in texts:
            if any(fragment in text for fragment in fragments):
                return skill, text
    return "", ""


def _first_named(
    texts: list[str], candidates: list[tuple[str, list[str]]]
) -> tuple[str, str]:
    """(skill, text) for the first candidate any of *texts* names outright."""
    for skill, _ in candidates:
        named = next((t for t in texts if _names_skill(t, skill)), "")
        if named:
            return skill, named
    return "", ""


def _by_intent(tool: str, texts: list[str]) -> tuple[list[str], list[str]]:
    """Split a call's strings into what it read and what it changed.

    Three kinds of tool, three answers. A write tool changes everything it
    names. A shell tool has to be read as a command line, because the same path
    means opposite things on either side of a ``>``. Everything else is read as
    it always was — every string it carries counts as a reference.
    """
    if _last_segment(tool) in _WRITE_TOOLS:
        return [], texts
    if _last_segment(tool) not in _SHELL_TOOLS:
        return texts, []

    reads: list[str] = []
    writes: list[str] = []
    for text in texts:
        text_reads, text_writes = _shell_targets(text)
        reads.extend(text_reads)
        writes.extend(text_writes)
    return reads, writes





def _fragments(skill: str, extra_path: str) -> list[str]:
    """Path fragments that mean "inside this skill's directory".

    ``skills/<name>/`` covers wherever the CLI keeps them — ``.claude/skills``,
    ``.codex/skills``, a plugin's bundle — and ``<name>/SKILL.md`` catches a
    skill read from an unconventional location.
    """
    fragments = [f"skills/{skill}/", f"{skill}/SKILL.md"]
    if extra_path:
        fragment = extra_path.replace("\\", "/").rstrip("/")
        if fragment:
            fragments.append(f"{fragment}/")
    return fragments


def _is_skill_tool(tool_name: str) -> bool:
    """Whether this tool is "load a skill by name" (``Skill``, ``mcp__x__Skill``)."""
    return _last_segment(tool_name) in _SKILL_TOOL_NAMES


def _names_skill(text: str, skill: str) -> bool:
    """Whether *text* refers to the skill, as an argument to the Skill tool.

    Segment-wise rather than substring: ``plugin:report-writer`` and
    ``.claude/skills/report-writer`` both name it, while a prose sentence that
    happens to contain the words does not.
    """
    if text.strip() == skill:
        return True
    return skill in [segment.strip() for segment in text.replace(":", "/").split("/")]


def _mentions_resource(
    text: str, skill: str, resource: str, fragments: list[str]
) -> bool:
    """Whether *text* points at ``<skill>/<resource>``."""
    if f"{skill}/{resource}" in text:
        return True
    return any(f"{fragment}{resource}" in text for fragment in fragments)



def _reasoning(
    skill: str,
    triggered: bool,
    should_trigger: bool,
    resources: dict[str, bool],
    edited: bool = False,
    alternate: str = "",
) -> str:
    if triggered != should_trigger:
        if should_trigger:
            if edited:
                return (
                    f"'{skill}' was never loaded — the run wrote into its directory "
                    f"but never read from it."
                )
            return f"'{skill}' was never loaded — no Skill call and no file read under it."
        return f"'{skill}' loaded on a prompt that should have been handled without it."
    if alternate:
        return (
            f"'{skill}' did not load; accepted alternate '{alternate}' answered "
            f"instead. Routing is fine — the skill under test did not win it."
        )
    unused = [r for r, used in resources.items() if not used]
    if unused:
        return f"'{skill}' loaded, but bundled {', '.join(unused)} went unused."
    if triggered:
        return f"'{skill}' loaded."
    return f"'{skill}' correctly stayed out of this one."
