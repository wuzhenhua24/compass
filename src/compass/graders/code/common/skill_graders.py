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

import re
import shlex
from collections.abc import Iterator
from typing import Any

from compass.core.skills import resolve_skill_name
from compass.graders.base import (
    CodeGrader,
    GradeContext,
    GradeResult,
    GraderScope,
    GraderType,
)
from compass.graders.registry import register_grader

# How deep into a tool call's input to look for strings. Claude Code's are
# shallow ({"file_path": ...}); an MCP tool's can nest a level or two.
_MAX_INPUT_DEPTH = 4

# The last segment of a tool name, lowercased, that means "the agent asked for
# a skill by name" rather than "the agent read a file that happens to live in
# one". Namespaced tools (``mcp__x__Skill``) reduce to the same segment.
_SKILL_TOOL_NAMES = frozenset({"skill", "skills"})

# Longest matched string kept as evidence, per match.
_EVIDENCE_CHARS = 200

# Tools whose whole job is to change a file. Reaching into the skill's
# directory with one of these is *editing the skill*, not loading it — an agent
# asked to improve a skill rewrites its SKILL.md without ever using it, and
# counting that as a trigger turns "the skill got edited" into "the skill
# fired". Matched on the tool name's last segment, like ``Skill`` itself.
_WRITE_TOOLS = frozenset(
    {
        "write", "edit", "multiedit", "notebookedit", "apply_patch",
        "create_file", "write_file", "edit_file", "delete", "delete_file",
    }
)

# Tools that carry a shell command line, which has to be read as a command
# rather than as a bag of strings — see :func:`_shell_targets`.
_SHELL_TOOLS = frozenset(
    {
        "bash", "shell", "sh", "zsh", "exec", "run", "terminal", "local_shell",
        "command_execution", "run_command", "execute_command", "run_terminal_cmd",
    }
)

# Commands whose path arguments are things they destroy or move, not things
# they read. ``cp`` is deliberately absent: its source argument really is read.
_DESTRUCTIVE = frozenset({"rm", "rmdir", "unlink", "mv", "shred", "truncate", "tee"})

# Commands after which a bare ``NAME=value`` is still an assignment.
_EXPORTERS = frozenset({"export", "declare", "local", "typeset", "readonly", "set"})

# Control operators shlex hands back as their own tokens. Each one starts a new
# command, so the destructive/assignment state resets.
_SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "{", "}", "\n"})

_REDIRECT_RE = re.compile(r"^\d*(?:>>?|&>>?|>&)$")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
_VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


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
        required_resources: list[str] — bundled files that must have been used,
                relative to the skill root (``scripts/render.py``). Each is a
                check of its own.
        path:   str  — an extra path fragment that counts as the skill's, for a
                skill installed somewhere Compass did not put it.
    """

    name = "skill_trigger"
    grader_type = GraderType.CODE
    grader_scope = GraderScope.TRANSCRIPT

    def __init__(self, config: dict[str, Any] | None = None):
        super().__init__(config)
        self.skill: str = str(self.config.get("skill", "") or "")
        self.should_trigger: bool = bool(self.config.get("should_trigger", True))
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

        fragments = _fragments(skill, self.path)
        touches, writes = _scan(context.tool_calls, skill, fragments)

        triggered = bool(touches)
        checks: list[tuple[str, bool]] = [("triggered", triggered == self.should_trigger)]
        failure_tags: list[str] = []
        if triggered != self.should_trigger:
            failure_tags.append("not_triggered" if self.should_trigger else "unexpected_trigger")

        resources: dict[str, bool] = {}
        for resource in self.required_resources:
            used = any(
                _mentions_resource(text, skill, resource, fragments)
                for touch in touches
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
        if self.required_resources:
            metrics["skill_resources_used"] = float(sum(resources.values()))

        tags = [f"skill_via_{s}" for s in signals]
        if writes and not triggered:
            # The one case where a reader would otherwise be misled: the agent
            # was all over the skill's directory and still never loaded it.
            tags.append("skill_write_only")

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
                skill, triggered, self.should_trigger, resources, bool(writes)
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
    """One tool call that reached into the skill."""

    __slots__ = ("tool", "turn", "signal", "evidence", "texts")

    def __init__(
        self, tool: str, turn: int, signal: str, evidence: str, texts: list[str]
    ) -> None:
        self.tool = tool
        self.turn = turn
        self.signal = signal
        self.evidence = evidence
        self.texts = texts


def _scan(
    tool_calls: list[Any], skill: str, fragments: list[str]
) -> tuple[list[_Touch], list[_Touch]]:
    """(loads, writes) — calls that opened the skill, and calls that changed it.

    Both are "the call mentioned a path inside the skill", and the difference
    between them is the whole point: a run that rewrote ``SKILL.md`` and never
    read it did not load the skill, and reporting it as a trigger would make an
    edit look like a hit.
    """
    loads: list[_Touch] = []
    writes: list[_Touch] = []
    for index, call in enumerate(tool_calls):
        tool = str(getattr(call, "tool_name", "") or getattr(call, "tool", ""))
        texts = [t.replace("\\", "/") for t in _strings(getattr(call, "input", None)) if t]
        turn_index = getattr(call, "turn_index", None)
        turn = int(turn_index) if turn_index is not None else index

        if _is_skill_tool(tool):
            named = next((t for t in texts if _names_skill(t, skill)), "")
            if named:
                loads.append(_Touch(tool, turn, "skill_tool", named[:_EVIDENCE_CHARS], texts))
                continue

        read_texts, write_texts = _by_intent(tool, texts)
        read_hit = _first_hit(read_texts, fragments)
        if read_hit:
            loads.append(_Touch(tool, turn, "file", read_hit[:_EVIDENCE_CHARS], read_texts))
            continue
        write_hit = _first_hit(write_texts, fragments)
        if write_hit:
            writes.append(_Touch(tool, turn, "write", write_hit[:_EVIDENCE_CHARS], write_texts))
    return loads, writes


def _first_hit(texts: list[str], fragments: list[str]) -> str:
    """The first of *texts* that points inside the skill, or ``""``."""
    for text in texts:
        if any(fragment in text for fragment in fragments):
            return text
    return ""


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


def _shell_targets(command: str) -> tuple[list[str], list[str]]:
    """(read, written) paths in one shell command line, variables resolved.

    Two things this buys over searching the raw string. ``VAR=`` assignments are
    followed, so ``D=.claude/skills/rw; cat $D/SKILL.md`` is recognized as the
    read it is. And redirect targets and the arguments of ``rm``/``mv``/``tee``
    land on the written side, so rewriting or deleting the skill stops reading
    as loading it.

    Deliberately shallow. There is no substitution, no globbing, no following
    of a script that a command runs — a shell is not something to reimplement
    inside a grader. A command that will not tokenize is handed back whole, on
    the read side, which is exactly the behaviour this replaced: the fallback
    keeps the old false positive rather than inventing a new false negative.
    """
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return [command], []

    assignments: dict[str, str] = {}
    reads: list[str] = []
    writes: list[str] = []
    at_command_start = True
    taking_assignments = True
    destructive = False
    redirecting = False

    for token in tokens:
        if token in _SEPARATORS:
            at_command_start = taking_assignments = True
            destructive = redirecting = False
            continue
        if _REDIRECT_RE.match(token):
            redirecting = True
            continue

        assignment = _ASSIGN_RE.match(token) if taking_assignments else None
        if assignment and not redirecting:
            assignments[assignment.group(1)] = _expand(assignment.group(2), assignments)
            continue

        expanded = _expand(token, assignments)
        if redirecting:
            writes.append(expanded)
            redirecting = False
            continue
        if at_command_start:
            at_command_start = False
            command_word = expanded.rsplit("/", 1)[-1]
            taking_assignments = command_word in _EXPORTERS
            destructive = command_word in _DESTRUCTIVE
            reads.append(expanded)
            continue
        (writes if destructive else reads).append(expanded)

    return reads, writes


def _expand(text: str, assignments: dict[str, str]) -> str:
    """Substitute ``$VAR`` / ``${VAR}`` from *assignments*.

    A variable the command line never set is left as written. Expanding it to
    the empty string would splice unrelated path pieces together and could
    manufacture a match that never happened.
    """
    if "$" not in text:
        return text

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return assignments.get(name, match.group(0))

    return _VAR_RE.sub(replace, text)


def _last_segment(tool_name: str) -> str:
    """The bare tool name behind any namespacing (``mcp__x__Bash`` -> ``bash``)."""
    return tool_name.replace("__", ".").replace(":", ".").rsplit(".", 1)[-1].strip().lower()


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


def _strings(value: Any, depth: int = 0) -> Iterator[str]:
    """Every string inside a tool call's input, to a bounded depth."""
    if depth > _MAX_INPUT_DEPTH:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _strings(item, depth + 1)


def _reasoning(
    skill: str,
    triggered: bool,
    should_trigger: bool,
    resources: dict[str, bool],
    edited: bool = False,
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
    unused = [r for r, used in resources.items() if not used]
    if unused:
        return f"'{skill}' loaded, but bundled {', '.join(unused)} went unused."
    if triggered:
        return f"'{skill}' loaded."
    return f"'{skill}' correctly stayed out of this one."
