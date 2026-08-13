"""Reading a skill directory's own identity.

A *skill* here is the convention Claude Code and Codex share: a directory with
a ``SKILL.md`` whose YAML frontmatter carries ``name`` and ``description``, plus
optional ``scripts/`` , ``references/`` and ``assets/`` beside it.

Two distant parts of Compass need the ``name`` out of that frontmatter, and they
have to agree:

- :class:`~compass.adapters.cli_agent.CliAgentAdapter` installs the directory
  under that name, so ``report-writer-v1`` and ``report-writer-v2`` both reach
  the agent as ``report-writer`` — otherwise a v1/v2 comparison silently
  renames the thing it is measuring.
- ``skill_trigger`` has to recognise the same name in a trace to answer "did
  the agent actually load it".

If those two ever disagreed, the grader would report an untriggered skill for
every run of a skill that worked fine. Hence one rule, in one place.
"""

from __future__ import annotations

from pathlib import Path


def read_skill_name(skill_dir: Path) -> str:
    """The ``name:`` from ``skill_dir/SKILL.md``'s frontmatter, or "".

    Deliberately hand-rolled rather than a ``yaml.safe_load`` of the file: the
    frontmatter of a skill written for another tool may carry keys PyYAML
    chokes on, and the only thing needed here is one scalar. A skill whose
    frontmatter cannot be read falls back to its directory name — which is the
    convention anyway — rather than failing the run.
    """
    try:
        text = (skill_dir / "SKILL.md").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if not text.startswith("---"):
        return ""
    _, _, rest = text.partition("\n")
    for line in rest.splitlines():
        if line.strip() == "---":
            break
        key, sep, value = line.partition(":")
        if sep and key.strip() == "name":
            return value.strip().strip("'\"")
    return ""


def resolve_skill_name(reference: str) -> str:
    """A configured skill reference — a name *or* a directory — as its name.

    Scenarios name the skill under test twice: once on the adapter (as a path,
    because something has to be installed) and once on the grader (as a name,
    because something has to be recognised). Accepting either spelling in
    either place removes a whole class of comparison that runs clean and
    measures nothing.
    """
    ref = reference.strip()
    if not ref:
        return ""
    path = Path(ref).expanduser()
    if path.is_dir():
        return read_skill_name(path) or path.name
    return ref
