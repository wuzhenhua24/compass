"""Reading what a tool call actually did.

A tool call is the only first-hand record of an agent's behaviour, and three
different graders have to read the same call and disagree about it on purpose:
``skill_trigger`` asks *which files did this touch*, ``dangerous_operations``
asks *what command did this run*, and both have to agree on the boring parts —
what the tool is called once the namespacing is stripped, which strings the
call carried, and how to read a shell command line without reimplementing a
shell.

Those boring parts live here. They were written for ``skill_trigger`` and moved
out unchanged when the second reader arrived; the shell tokenizer in
particular has a lot of small decisions in it (``VAR=`` assignments are
followed, redirect targets are not arguments, a line that will not tokenize is
handed back whole) that are worth having in one place rather than two.

Two views of a shell line are offered, because the two questions want different
shapes:

- :func:`shell_targets` — *(read, written)* paths. "Did this touch the skill?"
- :func:`shell_commands` — the commands themselves, segmented, with their
  arguments and their pipe relationship. "Did this run ``rm -rf``?"

Both are deliberately shallow. There is no globbing, no command substitution,
and no following of a script a command runs: what happens inside a subprocess
is invisible to a transcript, and pretending otherwise would be worse than
saying so.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Iterator
from typing import Any

# How deep to walk a tool call's input for strings. A native tool's input is
# shallow ({"file_path": ...}); an MCP tool's can nest a level or two.
MAX_INPUT_DEPTH = 4

# Longest matched string kept as evidence, per match.
EVIDENCE_CHARS = 200

# Tools whose whole job is to change a file. Matched on the tool name's last
# segment, so ``mcp__x__Edit`` reduces to ``edit``.
WRITE_TOOLS = frozenset(
    {
        "write", "edit", "multiedit", "notebookedit", "apply_patch",
        "create_file", "write_file", "edit_file", "delete", "delete_file",
    }
)

# Tools that carry a shell command line, which has to be read as a command
# rather than as a bag of strings — see :func:`shell_targets`.
SHELL_TOOLS = frozenset(
    {
        "bash", "shell", "sh", "zsh", "exec", "run", "terminal", "local_shell",
        "command_execution", "run_command", "execute_command", "run_terminal_cmd",
    }
)

# Commands whose path arguments are things they destroy or move, not things
# they read. ``cp`` is deliberately absent: its source argument really is read.
DESTRUCTIVE = frozenset({"rm", "rmdir", "unlink", "mv", "shred", "truncate", "tee"})

# Commands after which a bare ``NAME=value`` is still an assignment.
EXPORTERS = frozenset({"export", "declare", "local", "typeset", "readonly", "set"})

# Control operators shlex hands back as their own tokens. Each one starts a new
# command, so the destructive/assignment state resets.
SEPARATORS = frozenset({";", "&&", "||", "|", "&", "(", ")", "{", "}", "\n"})

REDIRECT_RE = re.compile(r"^\d*(?:>>?|&>>?|>&)$")
ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.DOTALL)
VAR_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def last_segment(tool_name: str) -> str:
    """The bare tool name behind any namespacing (``mcp__x__Bash`` -> ``bash``)."""
    return tool_name.replace("__", ".").replace(":", ".").rsplit(".", 1)[-1].strip().lower()


def input_strings(value: Any, depth: int = 0) -> Iterator[str]:
    """Every string inside a tool call's input, to a bounded depth."""
    if depth > MAX_INPUT_DEPTH:
        return
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from input_strings(item, depth + 1)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from input_strings(item, depth + 1)


def expand(text: str, assignments: dict[str, str]) -> str:
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

    return VAR_RE.sub(replace, text)


def _tokenize(command: str) -> list[str] | None:
    """Shell tokens, or None for a line that will not tokenize."""
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        return list(lexer)
    except ValueError:
        return None


def shell_targets(command: str) -> tuple[list[str], list[str]]:
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
    tokens = _tokenize(command)
    if tokens is None:
        return [command], []

    assignments: dict[str, str] = {}
    reads: list[str] = []
    writes: list[str] = []
    at_command_start = True
    taking_assignments = True
    destructive = False
    redirecting = False

    for token in tokens:
        if token in SEPARATORS:
            at_command_start = taking_assignments = True
            destructive = redirecting = False
            continue
        if REDIRECT_RE.match(token):
            redirecting = True
            continue

        assignment = ASSIGN_RE.match(token) if taking_assignments else None
        if assignment and not redirecting:
            assignments[assignment.group(1)] = expand(assignment.group(2), assignments)
            continue

        expanded = expand(token, assignments)
        if redirecting:
            writes.append(expanded)
            redirecting = False
            continue
        if at_command_start:
            at_command_start = False
            command_word = expanded.rsplit("/", 1)[-1]
            taking_assignments = command_word in EXPORTERS
            destructive = command_word in DESTRUCTIVE
            reads.append(expanded)
            continue
        (writes if destructive else reads).append(expanded)

    return reads, writes


class ShellCommand:
    """One command inside a shell line: what was invoked, and with what.

    ``word`` is the command as written after variable expansion
    (``/usr/bin/rm``); ``name`` is its basename lowercased (``rm``), which is
    what a rule matches on. Keeping both matters — a rule asks about ``rm``,
    but the evidence a human reads should say which ``rm``.

    ``piped`` is whether this command's stdin came from the previous one. It is
    the whole of what makes ``curl … | sh`` recognisable as *running fetched
    code* rather than as a download next to an unrelated shell.
    """

    __slots__ = ("word", "name", "args", "redirects", "piped")

    def __init__(
        self,
        word: str,
        args: list[str],
        redirects: list[str],
        piped: bool,
    ) -> None:
        self.word = word
        self.name = word.rsplit("/", 1)[-1].lower()
        self.args = args
        self.redirects = redirects
        self.piped = piped

    @property
    def text(self) -> str:
        """The command re-joined, for evidence a human reads."""
        return " ".join([self.word, *self.args])

    def has_flag(self, *flags: str) -> bool:
        """Whether any of *flags* was passed, bundled short flags included.

        ``rm -rf`` passes ``has_flag("-r")`` — the bundle is expanded — while
        ``--recursive`` is matched whole. A bare ``-`` or ``--`` is never a
        flag, and a value that merely starts with a dash is not searched for
        bundled letters.
        """
        for arg in self.args:
            if not arg.startswith("-") or arg in ("-", "--"):
                continue
            for flag in flags:
                if arg == flag:
                    return True
                if (
                    len(flag) == 2
                    and flag.startswith("-")
                    and not arg.startswith("--")
                    and flag[1] in arg[1:]
                ):
                    return True
        return False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ShellCommand({self.text!r}, piped={self.piped})"


def shell_commands(command: str) -> list[ShellCommand]:
    """The commands in one shell line, segmented, variables resolved.

    The sibling of :func:`shell_targets` for the other question. That one asks
    which paths a line touched; this one asks which programs it ran, because
    "was ``rm -rf`` executed" cannot be answered by a list of paths.

    Reading the command *word* rather than the raw line is the point:
    ``echo "rm -rf /"`` runs ``echo``, and a scanner that greps the string
    reports a deletion that never happened. That false positive is most of what
    separates this from grepping a script's source.

    A line that will not tokenize yields nothing. Unlike
    :func:`shell_targets`, which falls back to the whole string, there is no
    safe fallback here: an untokenizable line has no identifiable command word,
    and guessing one would invent a finding.
    """
    tokens = _tokenize(command)
    if tokens is None:
        return []

    commands: list[ShellCommand] = []
    assignments: dict[str, str] = {}
    word = ""
    args: list[str] = []
    redirects: list[str] = []
    started = False
    piped = False
    next_piped = False
    taking_assignments = True
    redirecting = False

    def flush() -> None:
        nonlocal word, args, redirects, started
        if started:
            commands.append(ShellCommand(word, args, redirects, piped))
        word, args, redirects, started = "", [], [], False

    for token in tokens:
        if token in SEPARATORS:
            flush()
            piped = next_piped = token == "|"
            taking_assignments = True
            redirecting = False
            continue
        if REDIRECT_RE.match(token):
            redirecting = True
            continue

        assignment = ASSIGN_RE.match(token) if taking_assignments else None
        if assignment and not redirecting:
            assignments[assignment.group(1)] = expand(assignment.group(2), assignments)
            continue

        expanded = expand(token, assignments)
        if redirecting:
            redirects.append(expanded)
            redirecting = False
            continue
        if not started:
            started = True
            word = expanded
            piped = next_piped
            taking_assignments = expanded.rsplit("/", 1)[-1] in EXPORTERS
            continue
        args.append(expanded)

    flush()
    return commands
